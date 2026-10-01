"""Retain full evaluation predictions in CI logs before ephemeral pods exit."""

import base64
import io
import tarfile


def archive_results(directory):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in sorted(directory.iterdir()):
            if path.is_file():
                archive.add(path, arcname=path.name)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    print("FULL_EVAL_ARCHIVE_BEGIN", flush=True)
    for start in range(0, len(encoded), 4096):
        print("FULL_EVAL_ARCHIVE_CHUNK " + encoded[start : start + 4096], flush=True)
    print("FULL_EVAL_ARCHIVE_END", flush=True)
