"""Stream local IR into a single durable uncompressed tar; never delete source."""
from pathlib import Path
import tarfile


def archive(source, target):
    source=Path(source).resolve();target=Path(target)
    if not source.is_dir() or target.exists():
        raise ValueError('existing source and fresh target required')
    target.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive creation prevents a retry overwriting evidence. A failed partial
    # tar is retained explicitly; callers must mark archive failure, never pass.
    with target.open('xb') as stream:
        with tarfile.open(fileobj=stream,mode='w|') as output:
            for path in source.rglob('*'):
                if path.is_symlink():
                    raise ValueError('IR archive refuses symbolic links')
                if path.is_file():
                    output.add(path,arcname=str(path.relative_to(source)),recursive=False)
    return dict(path=str(target),size_bytes=target.stat().st_size,
                scope='transport archive; caller verifies completeness/readback')
