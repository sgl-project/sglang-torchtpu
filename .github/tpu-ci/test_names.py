"""Build GitHub display names and compact Kubernetes names from test paths."""

import hashlib
import os


def names_from_path(path: str) -> dict[str, str]:
    """Return one full display name and one collision-resistant compact name."""
    name = os.path.basename(path).removesuffix(".py")
    run_name = hashlib.sha256(name.encode()).hexdigest()[:12]
    return {"name": name, "run_name": run_name}
