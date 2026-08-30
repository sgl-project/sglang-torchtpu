"""Fills sgl-eval's mmlu cache so the eval never downloads its dataset at run time.

sgl-eval's vendored NeMo-Skills loader downloads MMLU from a Berkeley webserver
that has repeated outages, with no mirror, retry or override. It only downloads
when its cache file is missing, so writing that file here means the eval never
contacts the host.

TODO: drop this once sgl-eval's own loader can fall back to a mirror
(sgl-project/sgl-eval#33).
"""

import shutil
import tempfile
import urllib.request
from pathlib import Path

# cais/mmlu has the same archive. The URL pins a revision, not a branch: a Hub
# revision is immutable, so a rebuild cannot change the dataset the scores come
# from.
MMLU_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"
TAR_URL = f"https://huggingface.co/datasets/cais/mmlu/resolve/{MMLU_REVISION}/data.tar"
SPLIT = "test"


def main() -> None:
    # _CACHE_ROOT rather than a literal path: this script and the loader have to
    # agree on where the cache lives, and an import error is louder than a cache
    # written where nothing reads it. prepare is private for the same reason it
    # is patched below, so both imports are load-bearing coupling to sgl-eval.
    from sgl_eval._vendored.nemo_skills.dataset.mmlu import prepare
    from sgl_eval.evals._loader import _CACHE_ROOT

    with tempfile.TemporaryDirectory() as tmp:
        tar = Path(tmp) / "data.tar"
        urllib.request.urlretrieve(TAR_URL, tar)
        # save_data always downloads. Pointing its URL at the local copy through
        # file:// leaves the code that turns the archive into the jsonl alone,
        # so the result matches what a working download would produce.
        prepare.URL = tar.as_uri()
        prepare.save_data(SPLIT)

    cache = _CACHE_ROOT / "mmlu" / f"{SPLIT}.jsonl"
    cache.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(
        str(Path(prepare.__file__).resolve().parent / f"{SPLIT}.jsonl"), str(cache)
    )
    with cache.open(encoding="utf-8") as fh:
        rows = sum(1 for _ in fh)
    print(f"warmed {cache} ({rows} rows)")


if __name__ == "__main__":
    main()
