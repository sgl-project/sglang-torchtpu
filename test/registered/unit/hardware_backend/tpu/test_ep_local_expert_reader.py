"""TPU EP-local safetensors reads: each EP rank reads only its routed experts."""

import contextlib
import json
import os
import struct
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import safetensors.torch
import torch

from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.environ import envs
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.model_loader import weight_utils
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu import ep_local_expert_reader

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

_MOD = "sglang_tpu.ep_local_expert_reader"
_LOADER = "sglang.srt.model_loader.loader"
_ITERATORS = ("single", "buffered")


def _write_checkpoint(folder, num_experts=4, layers=(0, 1), extra=None):
    g = torch.Generator().manual_seed(0)
    tensors = {"model.embed_tokens.weight": torch.randn(8, 4, generator=g)}
    for layer in layers:
        prefix = f"model.layers.{layer}.mlp"
        tensors[f"{prefix}.shared_experts.down_proj.weight"] = torch.randn(
            4, 4, generator=g
        ).bfloat16()
        tensors[f"{prefix}.gate.weight"] = torch.randn(num_experts, 4, generator=g)
        for e in range(num_experts):
            tensors[f"{prefix}.experts.{e}.down_proj.weight"] = torch.randn(
                4, 4, generator=g
            ).bfloat16()
            tensors[f"{prefix}.experts.{e}.down_proj.weight_scale"] = torch.randint(
                0, 255, (4, 1), dtype=torch.uint8, generator=g
            )
    tensors.update(extra or {})
    names = sorted(tensors)
    half = len(names) // 2
    files, weight_map = [], {}
    for i, chunk in enumerate((names[:half], names[half:])):
        fname = f"model-0000{i + 1}-of-00002.safetensors"
        safetensors.torch.save_file(
            {n: tensors[n] for n in chunk}, os.path.join(folder, fname)
        )
        files.append(os.path.join(folder, fname))
        weight_map.update({n: fname for n in chunk})
    with open(os.path.join(folder, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": weight_map}, f)
    return files, tensors


class _CountingFile:
    """File wrapper counting the bytes readinto() fills."""

    def __init__(self, f, counter):
        self._f, self._counter = f, counter

    def readinto(self, buf):
        n = self._f.readinto(buf)
        self._counter.append(n or 0)
        return n

    def __getattr__(self, name):
        return getattr(self._f, name)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._f.close()


def _expert_id(name):
    return (
        int(name.split(".experts.")[1].split(".")[0]) if ".experts." in name else None
    )


def _read(files, iterator, expert_range, disable_mmap=True):
    if iterator == "buffered":
        it = weight_utils.buffered_multi_thread_safetensors_weights_iterator(
            files,
            max_workers=2,
            disable_mmap=disable_mmap,
            local_expert_range=expert_range,
        )
    else:
        it = weight_utils.safetensors_weights_iterator(
            files, disable_mmap=disable_mmap, local_expert_range=expert_range
        )
    return dict(it)


class TestEpLocalExpertReads(CustomTestCase):
    def test_stacked_experts_read_only_local_payload(self):
        """Muse stacks must be sliced on disk, without slicing shared norms."""
        prefix = "model.language_model.layers.0.mlp.experts"
        stacks = {
            f"{prefix}.gate_up_proj": torch.arange(4 * 16 * 8)
            .reshape(4, 16, 8)
            .bfloat16(),
            f"{prefix}.down_proj": torch.arange(4 * 8 * 8).reshape(4, 8, 8).bfloat16(),
            f"{prefix}.post_expert_norm.weight": torch.arange(8).bfloat16(),
        }
        for iterator in _ITERATORS:
            with self.subTest(iterator=iterator), tempfile.TemporaryDirectory() as d:
                files, tensors = _write_checkpoint(d, layers=(), extra=stacks)
                read_bytes = []
                real_open = open

                def counting_open(path, mode="r", *args, **kwargs):
                    return _CountingFile(
                        real_open(path, mode, *args, **kwargs), read_bytes
                    )

                with (
                    patch(f"{_MOD}._SPAN_MERGE_GAP_BYTES", 0),
                    patch(f"{_MOD}.open", counting_open, create=True),
                ):
                    loaded = _read(files, iterator, (2, 4))
                expected_bytes = 0
                for name, tensor in tensors.items():
                    expected = (
                        tensor[2:4]
                        if name.endswith((".gate_up_proj", ".down_proj"))
                        else tensor
                    )
                    torch.testing.assert_close(loaded[name], expected, rtol=0, atol=0)
                    expected_bytes += expected.numel() * expected.element_size()
                self.assertEqual(sum(read_bytes), expected_bytes)

    def test_stacked_experts_are_verified_and_range_checked(self):
        name = "model.language_model.layers.0.mlp.experts.gate_up_proj"
        with tempfile.TemporaryDirectory() as d:
            files, _ = _write_checkpoint(
                d, layers=(), extra={name: torch.ones(4, 8, 8)}
            )
            reader = ep_local_expert_reader.EpLocalExpertReader(files, (2, 4))
            with self.assertRaisesRegex(RuntimeError, "missed 1 local expert"):
                reader.verify()
            reader = ep_local_expert_reader.EpLocalExpertReader(files, (3, 5))
            with self.assertRaisesRegex(ValueError, "Invalid stacked expert range"):
                for path in files:
                    reader.read(path)

    def test_reads_only_local_experts_and_all_other_weights(self):
        for iterator in ("single", "buffered"):
            with self.subTest(iterator=iterator), tempfile.TemporaryDirectory() as d:
                files, tensors = _write_checkpoint(d)
                loaded = _read(files, iterator, (2, 4))
                expected = {n for n in tensors if _expert_id(n) in (None, 2, 3)}
                self.assertEqual(set(loaded), expected)
                for name in expected:
                    self.assertEqual(loaded[name].dtype, tensors[name].dtype)
                    self.assertTrue(torch.equal(loaded[name], tensors[name]), name)

    def test_span_reads_fetch_only_the_kept_bytes(self):
        """Without gap merging every run of kept tensors is its own span, as on
        real shards: values still match and the skipped experts' bytes are never
        read."""
        for iterator in _ITERATORS:
            with (
                self.subTest(iterator=iterator),
                tempfile.TemporaryDirectory() as d,
                patch(f"{_MOD}._SPAN_MERGE_GAP_BYTES", 0),
            ):
                files, tensors = _write_checkpoint(d)
                read_bytes = []
                real_open = open

                def counting_open(path, mode="r", *args, **kwargs):
                    return _CountingFile(
                        real_open(path, mode, *args, **kwargs), read_bytes
                    )

                with patch(f"{_MOD}.open", counting_open, create=True):
                    loaded = _read(files, iterator, (2, 4))
                expected = {n for n in tensors if _expert_id(n) in (None, 2, 3)}
                self.assertEqual(set(loaded), expected)
                for name in expected:
                    self.assertTrue(torch.equal(loaded[name], tensors[name]), name)
                # Span payloads go through readinto; headers through read().
                kept = sum(
                    tensors[n].numel() * tensors[n].element_size() for n in expected
                )
                self.assertEqual(sum(read_bytes), kept)

    def test_reads_everything_without_range_mmap_off_or_index(self):
        with (
            tempfile.TemporaryDirectory() as d,
            patch(f"{_MOD}.log_info_on_rank0") as log,
        ):
            files, tensors = _write_checkpoint(d)
            self.assertEqual(set(_read(files, "single", None)), set(tensors))
            log.assert_not_called()
            self.assertEqual(
                set(_read(files, "single", (2, 4), disable_mmap=False)), set(tensors)
            )
            self.assertIn("--weight-loader-disable-mmap", log.call_args.args[1])
            os.remove(os.path.join(d, "model.safetensors.index.json"))
            self.assertEqual(set(_read(files, "single", (2, 4))), set(tensors))
            self.assertIn("no model.safetensors.index.json", log.call_args.args[1])

    def test_files_outside_the_index_are_read_whole(self):
        # e.g. an MTP shard that maybe_add_mtp_safetensors appends beside the index.
        with tempfile.TemporaryDirectory() as d:
            files, tensors = _write_checkpoint(d)
            mtp = {
                f"model.mtp.layers.0.mlp.experts.{e}.down_proj.weight": torch.ones(2)
                for e in range(4)
            }
            mtp_file = os.path.join(d, "mtp.safetensors")
            safetensors.torch.save_file(mtp, mtp_file)
            loaded = _read(files + [mtp_file], "single", (2, 4))
            self.assertTrue(set(mtp) <= set(loaded))

    def test_missing_local_expert_raises(self):
        for iterator in _ITERATORS:
            with self.subTest(iterator=iterator), tempfile.TemporaryDirectory() as d:
                files, _ = _write_checkpoint(d)
                index = os.path.join(d, "model.safetensors.index.json")
                with open(index) as f:
                    meta = json.load(f)
                # The index promises a local expert tensor no shard holds.
                meta["weight_map"]["model.layers.0.mlp.experts.3.up_proj.weight"] = (
                    os.path.basename(files[0])
                )
                with open(index, "w") as f:
                    json.dump(meta, f)
                with self.assertRaisesRegex(
                    RuntimeError,
                    "missed 1 local expert.*SGLANG_TPU_DISABLE_EP_LOCAL_EXPERT_READS=1",
                ):
                    _read(files, iterator, (2, 4))

    def test_index_shards_outside_this_load_are_not_expected(self):
        for iterator in _ITERATORS:
            with self.subTest(iterator=iterator), tempfile.TemporaryDirectory() as d:
                files, tensors = _write_checkpoint(d)
                index = os.path.join(d, "model.safetensors.index.json")
                with open(index) as f:
                    meta = json.load(f)
                # A local expert in an indexed shard this load does not read.
                meta["weight_map"]["model.layers.0.mlp.experts.3.up_proj.weight"] = (
                    "model-00003-of-00003.safetensors"
                )
                with open(index, "w") as f:
                    json.dump(meta, f)
                loaded = _read(files, iterator, (2, 4))
                expected = {n for n in tensors if _expert_id(n) in (None, 2, 3)}
                self.assertEqual(set(loaded), expected)

    def test_tensor_unaligned_for_its_dtype_is_copied_out(self):
        # A hand-written shard: a 3-byte tensor, then a bf16 expert at offset 3.
        local = torch.arange(4, dtype=torch.bfloat16)
        names = {"model.norm.bias": torch.arange(3, dtype=torch.uint8)}
        names["model.layers.0.mlp.experts.2.down_proj.weight"] = local
        header = {
            "model.norm.bias": {"dtype": "U8", "shape": [3], "data_offsets": [0, 3]},
            "model.layers.0.mlp.experts.2.down_proj.weight": {
                "dtype": "BF16",
                "shape": [4],
                "data_offsets": [3, 11],
            },
        }
        payload = names["model.norm.bias"].numpy().tobytes()
        payload += local.view(torch.int16).numpy().tobytes()
        for iterator in _ITERATORS:
            with self.subTest(iterator=iterator), tempfile.TemporaryDirectory() as d:
                shard = os.path.join(d, "model-00001-of-00001.safetensors")
                header_bytes = json.dumps(header).encode()
                with open(shard, "wb") as f:
                    f.write(struct.pack("<Q", len(header_bytes)) + header_bytes)
                    f.write(payload)
                with open(os.path.join(d, "model.safetensors.index.json"), "w") as f:
                    json.dump(
                        {"weight_map": {n: os.path.basename(shard) for n in header}}, f
                    )
                loaded = _read([shard], iterator, (2, 4))
                self.assertEqual(set(loaded), set(names))
                for name, tensor in names.items():
                    self.assertTrue(torch.equal(loaded[name], tensor), name)
                    self.assertEqual(
                        loaded[name].data_ptr() % tensor.element_size(), 0, name
                    )

    def test_truncated_shard_fails_before_reading(self):
        for iterator in _ITERATORS:
            with self.subTest(iterator=iterator), tempfile.TemporaryDirectory() as d:
                files, _ = _write_checkpoint(d)
                size = os.path.getsize(files[0])
                with open(files[0], "r+b") as f:
                    f.truncate(size - 8)
                with self.assertRaisesRegex(IOError, "is truncated"):
                    _read(files, iterator, (2, 4))

    def test_zero_element_tensor(self):
        for iterator in _ITERATORS:
            with self.subTest(iterator=iterator), tempfile.TemporaryDirectory() as d:
                files, _ = _write_checkpoint(
                    d, extra={"model.empty": torch.empty(0, 4)}
                )
                loaded = _read(files, iterator, (2, 4))
                self.assertEqual(tuple(loaded["model.empty"].shape), (0, 4))

    def test_fnuz_and_complex_dtypes_are_read_directly(self):
        g = torch.Generator().manual_seed(1)
        extra = {
            "model.layers.0.mlp.experts.2.fnuz": torch.randn(4, 4, generator=g).to(
                torch.float8_e4m3fnuz
            ),
            "model.layers.0.mlp.experts.0.fnuz": torch.randn(4, 4, generator=g).to(
                torch.float8_e4m3fnuz
            ),
            "model.e5m2fnuz": torch.randn(4, 4, generator=g).to(torch.float8_e5m2fnuz),
            "model.c64": torch.randn(3, dtype=torch.complex64, generator=g),
        }
        for iterator in _ITERATORS:
            with self.subTest(iterator=iterator), tempfile.TemporaryDirectory() as d:
                files, _ = _write_checkpoint(d, extra=extra)
                with patch.object(
                    ep_local_expert_reader.safetensors,
                    "safe_open",
                    side_effect=AssertionError,
                ):
                    loaded = _read(files, iterator, (2, 4))
                self.assertNotIn("model.layers.0.mlp.experts.0.fnuz", loaded)
                for name in (
                    "model.layers.0.mlp.experts.2.fnuz",
                    "model.e5m2fnuz",
                    "model.c64",
                ):
                    self.assertEqual(loaded[name].dtype, extra[name].dtype)
                    self.assertTrue(
                        torch.equal(
                            loaded[name].view(torch.uint8),
                            extra[name].view(torch.uint8),
                        ),
                        name,
                    )

    def test_unviewable_dtype_is_read_by_safetensors(self):
        dtypes = {
            k: v
            for k, v in ep_local_expert_reader._SAFETENSORS_DTYPES.items()
            if k != "BF16"
        }
        for iterator in _ITERATORS:
            with self.subTest(iterator=iterator), tempfile.TemporaryDirectory() as d:
                files, tensors = _write_checkpoint(d)
                with patch.object(
                    ep_local_expert_reader, "_SAFETENSORS_DTYPES", dtypes
                ):
                    loaded = _read(files, iterator, (2, 4))
                expected = {n for n in tensors if _expert_id(n) in (None, 2, 3)}
                self.assertEqual(set(loaded), expected)
                for name in expected:
                    self.assertTrue(torch.equal(loaded[name], tensors[name]), name)

    def test_chunked_span_reads(self):
        with tempfile.TemporaryDirectory() as d:
            files, tensors = _write_checkpoint(d)
            with patch.object(ep_local_expert_reader, "_READ_CHUNK_BYTES", 7):
                loaded = _read(files, "single", (0, 2))
            for name, tensor in loaded.items():
                self.assertTrue(torch.equal(tensor, tensors[name]), name)

    def test_routed_expert_dtype_probe_regex_untouched(self):
        # The reader's expert-id pattern must not replace weight_utils'
        # `.weight`-anchored routed-expert pattern used for dtype probing.
        self.assertIsNone(
            weight_utils._ROUTED_EXPERT_KEY_RE.search(
                "model.layers.0.mlp.experts.3.w1.scale"
            )
        )
        self.assertIsNotNone(
            weight_utils._ROUTED_EXPERT_KEY_RE.search(
                "model.layers.0.mlp.experts.3.down_proj.weight"
            )
        )


def _fused_moe(
    ep_size,
    storage_rank,
    num_local_routed,
    num_global_routed,
    *,
    fused_shared=False,
    global_experts_param=False,
):
    layer = FusedMoE.__new__(FusedMoE)
    torch.nn.Module.__init__(layer)
    layer.moe_ep_size = ep_size
    layer._expert_storage_rank = storage_rank
    layer._num_local_routed = num_local_routed
    layer._num_global_routed = num_global_routed
    layer._has_fused_shared = fused_shared
    if global_experts_param:
        layer.w13_input_scale = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        layer.w13_input_scale._sglang_require_global_experts = True
    return layer


def _model(layers, opted_in=True):
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList(layers)
    model.supports_ep_local_checkpoint_reads = opted_in
    return model


class TestTpuLocalRoutedExpertRange(CustomTestCase):
    @staticmethod
    def _env(*, tpu=True, eplb=None, join="none"):
        stack = contextlib.ExitStack()
        stack.enter_context(patch(f"{_MOD}._is_tpu", tpu))
        stack.enter_context(
            patch(
                "sglang.srt.eplb.expert_location.get_global_expert_location_metadata",
                return_value=eplb,
            )
        )
        stack.enter_context(
            patch(
                "sglang.srt.runtime_context.get_exec",
                return_value=SimpleNamespace(moe=SimpleNamespace(ep_join_mode=join)),
            )
        )
        return stack

    def test_range_comes_from_the_fused_moe_layers(self):
        # 28 checkpoint experts + 4 redundant slots over EP 4: 8 slots per rank.
        layers = [_fused_moe(4, 2, 8, 32), _fused_moe(4, 2, 8, 32)]
        with self._env():
            rng = ep_local_expert_reader.tpu_local_routed_expert_range(_model(layers))
        self.assertEqual(rng, (16, 24))
        for eid in range(32):
            local = FusedMoE._map_global_expert_id_to_local_expert_id(layers[0], eid)
            self.assertEqual(local != -1, rng[0] <= eid < rng[1], eid)

    def test_no_range(self):
        cases = {
            "not_tpu": (dict(tpu=False), [_fused_moe(4, 1, 8, 32)], True),
            "not_opted_in": ({}, [_fused_moe(4, 1, 8, 32)], False),
            "eplb": (dict(eplb=object()), [_fused_moe(4, 1, 8, 32)], True),
            "elastic": (dict(join="scale"), [_fused_moe(4, 1, 8, 32)], True),
            "ep1": ({}, [_fused_moe(1, 0, 32, 32)], True),
            "no_moe": ({}, [torch.nn.Linear(2, 2)], True),
            "layers_disagree": (
                {},
                [_fused_moe(4, 1, 8, 32), _fused_moe(4, 1, 4, 16)],
                True,
            ),
            "fused_shared_experts": (
                {},
                [_fused_moe(4, 1, 8, 32, fused_shared=True)],
                True,
            ),
            "param_needs_every_expert": (
                {},
                [_fused_moe(4, 1, 8, 32, global_experts_param=True)],
                True,
            ),
        }
        # Reads that could never be EP-local stay silent; an opted-in model under
        # EP that falls back says why on rank 0.
        silent = {"not_tpu", "not_opted_in", "ep1", "no_moe"}
        for label, (env, layers, opted_in) in cases.items():
            with (
                self.subTest(label),
                self._env(**env),
                patch(f"{_MOD}.log_info_on_rank0") as log,
            ):
                self.assertIsNone(
                    ep_local_expert_reader.tpu_local_routed_expert_range(
                        _model(layers, opted_in)
                    )
                )
                self.assertEqual(log.call_count, 0 if label in silent else 1)

    def test_kill_switch(self):
        with (
            self._env(),
            envs.SGLANG_TPU_DISABLE_EP_LOCAL_EXPERT_READS.override(True),
            patch(f"{_MOD}.log_info_on_rank0") as log,
        ):
            self.assertIsNone(
                ep_local_expert_reader.tpu_local_routed_expert_range(
                    _model([_fused_moe(4, 1, 8, 32)])
                )
            )
        self.assertIn("SGLANG_TPU_DISABLE_EP_LOCAL_EXPERT_READS", log.call_args.args[1])


class TestLoaderPassesTheRange(CustomTestCase):
    def _iterator_kwargs(self, extra_config):
        loader = DefaultModelLoader(
            LoadConfig(
                load_format=LoadFormat.SAFETENSORS,
                model_loader_extra_config=extra_config,
            )
        )
        source = DefaultModelLoader.Source(
            model_or_path="/dummy",
            revision=None,
            fall_back_to_pt=False,
            local_expert_range=(2, 4),
        )
        server_args = SimpleNamespace(
            weight_loader_disable_mmap=True,
            weight_loader_prefetch_checkpoints=False,
            weight_loader_prefetch_num_threads=4,
            weight_loader_drop_cache_after_load=False,
        )
        with (
            patch.object(
                DefaultModelLoader,
                "_prepare_weights",
                return_value=("/dummy", ["f.safetensors"], True),
            ),
            patch(f"{_LOADER}.get_model", return_value=server_args),
            patch(
                f"{_LOADER}.buffered_multi_thread_safetensors_weights_iterator",
                return_value=iter([]),
            ) as buffered,
            patch(
                f"{_LOADER}.safetensors_weights_iterator", return_value=iter([])
            ) as single,
        ):
            list(loader._get_weights_iterator(source))
        called = buffered if buffered.called else single
        return called.call_args.kwargs

    def test_both_iterators_get_the_source_range(self):
        for extra_config in ({}, {"enable_multithread_load": False}):
            with self.subTest(extra_config=extra_config):
                kwargs = self._iterator_kwargs(extra_config)
                self.assertEqual(kwargs["local_expert_range"], (2, 4))

    def test_init_new_takes_the_range_from_the_model(self):
        model = torch.nn.Module()
        model_config = SimpleNamespace(model_path="/dummy", revision=None)
        for tpu in (True, False):
            with (
                self.subTest(tpu=tpu),
                patch(f"{_LOADER}._is_tpu", tpu),
                patch(
                    f"{_MOD}.tpu_local_routed_expert_range", return_value=(2, 4)
                ) as expert_range,
            ):
                source = DefaultModelLoader.Source.init_new(model_config, model)
                if tpu:
                    expert_range.assert_called_once_with(model)
                    self.assertEqual(source.local_expert_range, (2, 4))
                else:
                    expert_range.assert_not_called()
                    self.assertIsNone(source.local_expert_range)


if __name__ == "__main__":
    unittest.main()
