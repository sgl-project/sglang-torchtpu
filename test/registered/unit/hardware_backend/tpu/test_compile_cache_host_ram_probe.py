"""Unit tests for the compile-cache host RAM probe (issue #1429)."""

import os
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

_REPO = Path(__file__).resolve().parents[5]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from benchmark.tpu import compile_cache_host_ram_probe as probe  # noqa: E402

# Pure logic over a fake /proc tree, no server or TPU needed.
register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _write_proc(root: str, pid: int, *, cmdline: str, comm: str, rss_pages: int):
    d = Path(root, str(pid))
    d.mkdir()
    (d / "cmdline").write_bytes(cmdline.encode() + b"\0--extra\0")
    (d / "comm").write_text(comm + "\n")
    # statm: size resident shared text lib data dt
    (d / "statm").write_text(f"{rss_pages * 2} {rss_pages} 10 1 0 5 0\n")


class TestRankDiscovery(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.root = tempfile.mkdtemp()
        self.addCleanup(lambda: os.system(f"rm -rf {self.root}"))

    def test_finds_scheduler_processes_by_title_only(self):
        _write_proc(
            self.root, 300, cmdline="sglang::scheduler_TP1", comm="x", rss_pages=4
        )
        _write_proc(
            self.root, 200, cmdline="sglang::scheduler_TP0", comm="x", rss_pages=3
        )
        _write_proc(
            self.root, 400, cmdline="python3 -m sglang", comm="python3", rss_pages=9
        )
        Path(self.root, "self").mkdir()  # non-numeric entries are skipped
        procs = probe.find_rank_processes(self.root)
        self.assertEqual(
            [(p.pid, p.title) for p in procs],
            [
                (200, "sglang::scheduler_TP0"),
                (300, "sglang::scheduler_TP1"),
            ],
        )

    def test_comm_prefix_identifies_a_scheduler_without_cmdline(self):
        # cmdline unreadable (missing), comm truncated to 15 bytes by the kernel.
        d = Path(self.root, "500")
        d.mkdir()
        (d / "comm").write_text("sglang::schedul\n")  # codespell:ignore schedul
        (d / "statm").write_text("20 10 1 1 0 1 0\n")
        procs = probe.find_rank_processes(self.root)
        self.assertEqual([p.pid for p in procs], [500])

    def test_rss_reads_statm_pages_and_tolerates_exited_processes(self):
        _write_proc(
            self.root, 200, cmdline="sglang::scheduler_TP0", comm="x", rss_pages=3
        )
        self.assertEqual(probe.read_rss_bytes(200, self.root), 3 * probe._PAGE_SIZE)
        self.assertEqual(probe.read_rss_bytes(999, self.root), 0)


class TestGrowthEstimate(CustomTestCase):
    def _samples(self, per_rank_steps, ranks=8, base=1000 * probe._MIB):
        samples = []
        for i, step in enumerate(per_rank_steps):
            rss = {pid: int(base + step) for pid in range(ranks)}
            samples.append(probe.Sample(prompt_len=1 + 4 * i, rss_bytes=rss))
        return samples

    def test_linear_growth_is_attributed_per_rank_and_per_host(self):
        # Every rank grows 0.5 MiB per new length: the issue's 8 x 0.5 = 4 MiB/host.
        half = 0.5 * probe._MIB
        growth = probe.estimate_growth(self._samples([0, half, 2 * half, 3 * half]))
        self.assertEqual(growth.ranks, 8)
        self.assertEqual(growth.lengths, 4)
        self.assertAlmostEqual(growth.per_rank_per_length_bytes / probe._MIB, 0.5)
        self.assertAlmostEqual(growth.host_per_length_bytes / probe._MIB, 4.0)
        self.assertAlmostEqual(growth.host_total_bytes / probe._MIB, 12.0)
        self.assertIn("0.50 MiB per rank per length", probe.format_growth(growth))
        self.assertIn("4.00 MiB per host per length", probe.format_growth(growth))

    def test_flat_rss_reports_zero_growth(self):
        growth = probe.estimate_growth(self._samples([0, 0, 0]))
        self.assertEqual(growth.per_rank_per_length_bytes, 0.0)
        self.assertEqual(growth.host_total_bytes, 0)

    def test_empty_run_does_not_divide_by_zero(self):
        growth = probe.estimate_growth([])
        self.assertEqual((growth.ranks, growth.lengths), (0, 0))
        self.assertEqual(growth.host_per_length_bytes, 0.0)

    def test_rows_show_host_delta_since_the_previous_length(self):
        first, second = self._samples([0, 2 * probe._MIB])
        row = probe.format_row(first, second)
        self.assertIn(f"{second.prompt_len:>8}", row)
        self.assertIn("+16.0", row)  # 8 ranks x 2 MiB
        self.assertEqual(len(probe.format_header().split()), 8)


class TestCli(CustomTestCase):
    def test_exits_when_no_scheduler_process_exists(self):
        with tempfile.TemporaryDirectory() as root:
            with _CaptureStderr() as err:
                code = probe.main(["--proc-root", root, "--count", "0"])
        self.assertEqual(code, 2)
        self.assertIn("sglang::scheduler", err.text)

    def test_explicit_pids_skip_discovery_and_zero_count_prints_header_only(self):
        with _CaptureStdout() as out:
            code = probe.main(["--pids", "1", "--count", "0"])
        self.assertEqual(code, 0)
        self.assertEqual(out.text.strip(), probe.format_header().strip())


class TestSlope(CustomTestCase):
    def test_fewer_than_two_points_has_no_slope(self):
        self.assertEqual(probe._slope([], []), 0.0)
        self.assertEqual(probe._slope([3.0], [9.0]), 0.0)

    def test_two_points_give_the_exact_slope(self):
        self.assertAlmostEqual(probe._slope([0.0, 2.0], [1.0, 5.0]), 2.0)

    def test_identical_xs_do_not_divide_by_zero(self):
        self.assertEqual(probe._slope([1.0, 1.0, 1.0], [0.0, 5.0, 9.0]), 0.0)

    def test_non_monotonic_ys_fit_least_squares(self):
        self.assertAlmostEqual(probe._slope([0, 1, 2, 3], [0, 10, 0, 10]), 2.0)

    def test_shrinking_ys_give_a_negative_slope(self):
        self.assertAlmostEqual(probe._slope([0, 1, 2], [6, 4, 2]), -2.0)


class TestGrowthEstimateEdgeCases(CustomTestCase):
    def test_single_sample_has_no_growth_but_keeps_the_rank_count(self):
        growth = probe.estimate_growth([probe.Sample(1, {10: 100, 11: 100})])
        self.assertEqual((growth.ranks, growth.lengths), (2, 1))
        self.assertEqual(growth.per_rank_per_length_bytes, 0.0)
        self.assertEqual(growth.host_total_bytes, 0)

    def test_zero_ranks_per_host_do_not_divide_by_zero(self):
        samples = [probe.Sample(1, {}), probe.Sample(5, {})]
        growth = probe.estimate_growth(samples)
        self.assertEqual(growth.ranks, 0)
        self.assertEqual(growth.per_rank_per_length_bytes, 0.0)
        self.assertEqual(samples[0].per_rank_bytes, 0.0)
        self.assertIn("x0 rank processes", probe.format_growth(growth))

    def test_a_rank_that_exits_mid_run_reads_as_zero(self):
        # The pid stays in every sample; its RSS drops to 0, so the host total
        # falls and the fit goes negative instead of raising.
        samples = [
            probe.Sample(1, {10: 100, 11: 100}),
            probe.Sample(5, {10: 100, 11: 0}),
        ]
        growth = probe.estimate_growth(samples)
        self.assertEqual(growth.ranks, 2)
        self.assertEqual(growth.host_total_bytes, -100)
        self.assertAlmostEqual(growth.host_per_length_bytes, -100.0)
        self.assertIn("-0 MiB host total", probe.format_growth(growth))

    def test_rank_count_comes_from_the_first_sample(self):
        samples = [probe.Sample(1, {10: 1}), probe.Sample(5, {10: 1, 11: 1})]
        self.assertEqual(probe.estimate_growth(samples).ranks, 1)

    def test_non_monotonic_rss_is_fitted_not_differenced(self):
        samples = [
            probe.Sample(1 + 4 * i, {10: int(v * probe._MIB)})
            for i, v in enumerate([100, 110, 100, 110])
        ]
        growth = probe.estimate_growth(samples)
        self.assertAlmostEqual(growth.host_per_length_bytes / probe._MIB, 2.0)
        self.assertEqual(growth.host_total_bytes, int(10 * probe._MIB))

    def test_first_row_has_no_previous_and_shows_a_zero_delta(self):
        row = probe.format_row(None, probe.Sample(7, {10: int(3 * probe._MIB)}))
        self.assertEqual(row.split(), ["7", "1", "3.0", "3.0", "+0.0"])

    def test_rows_stay_five_columns_with_many_ranks_and_large_values(self):
        many = {pid: int(40 * (1 << 30)) for pid in range(64)}
        row = probe.format_row(None, probe.Sample(123456, many))
        fields = row.split()
        self.assertEqual(len(fields), 5)
        self.assertEqual(fields[:2], ["123456", "64"])
        self.assertEqual(fields[2], "40960.0")
        self.assertEqual(fields[3], f"{64 * 40960:.1f}")
        # Rows align with the header's column widths.
        self.assertEqual(len(row), len(probe.format_header()))


class TestProcReading(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.root = tempfile.mkdtemp()
        self.addCleanup(lambda: os.system(f"rm -rf {self.root}"))

    def test_malformed_or_empty_statm_reads_as_zero(self):
        for content in ("", "garbage", "12\n", "a b c\n"):
            with self.subTest(content=content):
                d = Path(self.root, "77")
                d.mkdir(exist_ok=True)
                (d / "statm").write_text(content)
                self.assertEqual(probe.read_rss_bytes(77, self.root), 0)

    def test_pid_dir_without_cmdline_or_comm_is_skipped(self):
        Path(self.root, "88").mkdir()  # vanished between listdir and read
        _write_proc(self.root, 89, cmdline="", comm="bash", rss_pages=1)
        self.assertEqual(probe.find_rank_processes(self.root), [])

    def test_cmdline_title_wins_without_reading_comm(self):
        d = Path(self.root, "90")
        d.mkdir()
        (d / "cmdline").write_bytes(b"sglang::scheduler_TP3\0")
        procs = probe.find_rank_processes(self.root)
        self.assertEqual(
            [(p.pid, p.title) for p in procs], [(90, "sglang::scheduler_TP3")]
        )

    def test_undecodable_cmdline_does_not_raise(self):
        d = Path(self.root, "91")
        d.mkdir()
        (d / "cmdline").write_bytes(b"\xff\xfesglang::scheduler\0")
        (d / "comm").write_text("python3\n")
        self.assertEqual(probe.find_rank_processes(self.root), [])

    def test_missing_proc_root_raises_for_the_caller(self):
        with self.assertRaises(FileNotFoundError):
            probe.find_rank_processes(os.path.join(self.root, "nope"))


class TestCliEdgeCases(CustomTestCase):
    def test_defaults(self):
        args = probe._parse_args([])
        self.assertEqual(args.url, "http://127.0.0.1:30000")
        self.assertEqual((args.start, args.step, args.count), (1, 4, 64))
        self.assertEqual(args.max_new_tokens, 1)
        self.assertEqual(args.settle, 1.0)
        self.assertEqual(args.timeout, 600.0)
        self.assertEqual(args.proc_root, "/proc")
        self.assertIsNone(args.pids)

    def test_bad_arguments_exit_with_usage_error(self):
        for argv in (["--count", "abc"], ["--pids", "x"], ["--bogus"]):
            with self.subTest(argv=argv), _CaptureStderr():
                with self.assertRaises(SystemExit) as cm:
                    probe._parse_args(argv)
                self.assertEqual(cm.exception.code, 2)

    def test_empty_pids_falls_back_to_discovery(self):
        with tempfile.TemporaryDirectory() as root, _CaptureStderr() as err:
            code = probe.main(["--pids", "--proc-root", root, "--count", "1"])
        self.assertEqual(code, 2)
        self.assertIn("pass --pids", err.text)

    def test_negative_count_prints_header_only(self):
        with _CaptureStdout() as out, _patched_generate() as calls:
            code = probe.main(["--pids", "1", "--count", "-3"])
        self.assertEqual(code, 0)
        self.assertEqual(calls, [])
        self.assertEqual(out.text.strip(), probe.format_header().strip())

    def test_run_samples_every_length_and_reports_growth(self):
        root = tempfile.mkdtemp()
        self.addCleanup(lambda: os.system(f"rm -rf {root}"))
        _write_proc(root, 200, cmdline="sglang::scheduler_TP0", comm="x", rss_pages=100)
        _write_proc(root, 201, cmdline="sglang::scheduler_TP1", comm="x", rss_pages=100)
        pages = [100]

        def grow(url, prompt_len, max_new_tokens, timeout):
            pages[0] += 10  # every new length costs 10 pages per rank
            for pid in (200, 201):
                Path(root, str(pid), "statm").write_text(f"0 {pages[0]} 0 0 0 0 0\n")

        with (
            _CaptureStdout() as out,
            _CaptureStderr() as err,
            _patched_generate(grow) as calls,
            _patched_sleep() as sleeps,
        ):
            code = probe.main(
                [
                    "--proc-root",
                    root,
                    "--start",
                    "3",
                    "--step",
                    "5",
                    "--count",
                    "4",
                    "--settle",
                    "0.25",
                    "--url",
                    "http://h:1/",
                    "--max-new-tokens",
                    "2",
                    "--timeout",
                    "9",
                ]
            )
        self.assertEqual(code, 0)
        self.assertEqual(calls, [("http://h:1/", 3 + 5 * i, 2, 9.0) for i in range(4)])
        self.assertEqual(sleeps, [0.25] * 4)
        self.assertIn("rank process 200: sglang::scheduler_TP0", err.text)
        lines = out.text.rstrip("\n").splitlines()
        self.assertEqual(lines[0], probe.format_header())
        self.assertEqual(len(lines), 6)  # header, 4 rows, growth line
        rows = [line.split() for line in lines[1:5]]
        self.assertEqual([r[0] for r in rows], ["3", "8", "13", "18"])
        self.assertEqual({r[1] for r in rows}, {"2"})
        per_length = 2 * 10 * probe._PAGE_SIZE / probe._MIB
        self.assertEqual(rows[0][4], "+0.0")
        self.assertEqual(rows[1][4], f"{per_length:+.1f}")
        self.assertIn(
            f"4 new length(s) on 2 rank(s): {10 * probe._PAGE_SIZE / probe._MIB:.2f} "
            "MiB per rank per length",
            lines[5],
        )

    def test_a_rank_exiting_mid_run_is_reported_not_fatal(self):
        root = tempfile.mkdtemp()
        self.addCleanup(lambda: os.system(f"rm -rf {root}"))
        _write_proc(root, 200, cmdline="sglang::scheduler_TP0", comm="x", rss_pages=100)
        _write_proc(root, 201, cmdline="sglang::scheduler_TP1", comm="x", rss_pages=100)
        n = [0]

        def exit_rank(url, prompt_len, max_new_tokens, timeout):
            n[0] += 1
            if n[0] == 2:
                os.system(f"rm -rf {Path(root, '201')}")

        with (
            _CaptureStdout() as out,
            _CaptureStderr(),
            _patched_generate(exit_rank),
            _patched_sleep(),
        ):
            code = probe.main(["--proc-root", root, "--count", "3"])
        self.assertEqual(code, 0)
        rows = [line.split() for line in out.text.strip().splitlines()[1:4]]
        self.assertEqual([r[1] for r in rows], ["2", "2", "2"])
        self.assertTrue(rows[1][4].startswith("-"))

    def test_http_errors_propagate(self):
        import urllib.error

        for exc in (urllib.error.URLError("refused"), TimeoutError("slow")):
            with self.subTest(exc=exc):

                def fail(url, prompt_len, max_new_tokens, timeout, exc=exc):
                    raise exc

                with _CaptureStdout(), _patched_generate(fail), _patched_sleep():
                    with self.assertRaises(type(exc)):
                        probe.main(["--pids", "1", "--count", "2"])

    def test_generate_posts_plain_ids_at_the_requested_length(self):
        import io
        import json
        import urllib.request

        seen = {}

        class _Resp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def urlopen(req, timeout=None):
            seen["req"], seen["timeout"] = req, timeout
            return _Resp(b"{}")

        with unittest.mock.patch.object(urllib.request, "urlopen", urlopen):
            probe._generate("http://h:1/", 5, 3, 7.5)
        req = seen["req"]
        self.assertEqual(req.full_url, "http://h:1/generate")
        self.assertEqual(req.get_header("Content-type"), "application/json")
        self.assertEqual(seen["timeout"], 7.5)
        body = json.loads(req.data)
        self.assertEqual(len(body["input_ids"]), 5)
        self.assertTrue(all(1000 <= t < 3000 for t in body["input_ids"]))
        self.assertEqual(
            body["sampling_params"], {"max_new_tokens": 3, "temperature": 0}
        )

    def test_module_uses_only_the_standard_library(self):
        import ast

        tree = ast.parse(Path(probe.__file__).read_text())
        modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                modules.add(node.module.split(".")[0])
        self.assertTrue(modules)
        self.assertEqual(modules - set(sys.stdlib_module_names), set())


class _patched_generate:
    """Replace the HTTP call; records (url, prompt_len, max_new_tokens, timeout)."""

    def __init__(self, side_effect=None):
        self.calls = []
        self.side_effect = side_effect

    def __enter__(self):
        def fake(url, prompt_len, max_new_tokens, timeout):
            self.calls.append((url, prompt_len, max_new_tokens, timeout))
            if self.side_effect is not None:
                self.side_effect(url, prompt_len, max_new_tokens, timeout)

        self._patch = unittest.mock.patch.object(probe, "_generate", fake)
        self._patch.start()
        return self.calls

    def __exit__(self, *exc):
        self._patch.stop()
        return False


class _patched_sleep:
    def __enter__(self):
        self.sleeps = []
        self._patch = unittest.mock.patch.object(
            probe.time, "sleep", lambda s: self.sleeps.append(s)
        )
        self._patch.start()
        return self.sleeps

    def __exit__(self, *exc):
        self._patch.stop()
        return False


class _CaptureStderr:
    def __enter__(self):
        import io

        self._old = sys.stderr
        self._buf = io.StringIO()
        sys.stderr = self._buf
        return self

    def __exit__(self, *exc):
        sys.stderr = self._old
        self.text = self._buf.getvalue()
        return False


class _CaptureStdout(_CaptureStderr):
    def __enter__(self):
        import io

        self._old = sys.stdout
        self._buf = io.StringIO()
        sys.stdout = self._buf
        return self

    def __exit__(self, *exc):
        sys.stdout = self._old
        self.text = self._buf.getvalue()
        return False


if __name__ == "__main__":
    unittest.main()
