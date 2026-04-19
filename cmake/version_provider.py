"""scikit-build-core metadata provider for DNDSR version.

Reads the base version from the VERSION file and appends git describe
info for non-release commits (PEP 440 format).

Usage in pyproject.toml:
    [project]
    dynamic = ["version"]
    [tool.scikit-build.metadata]
    version.provider = "provider"
    version.provider-path = "cmake"
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


def dynamic_metadata(
    field: str,
    settings: dict[str, Any] | None = None,
) -> str:
    if field != "version":
        msg = f"Only 'version' is supported, got {field!r}"
        raise ValueError(msg)

    root = Path(__file__).resolve().parent.parent
    version_file = root / "VERSION"
    base = version_file.read_text().strip()

    try:
        desc = subprocess.check_output(
            ["git", "describe", "--tags", "--long", "--match", "v*"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        # No git or no tags — return base with commit hash if possible
        try:
            short = subprocess.check_output(
                ["git", "rev-parse", "--short=7", "HEAD"],
                cwd=root,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            return f"{base}.dev0+g{short}"
        except (subprocess.CalledProcessError, FileNotFoundError):
            return base

    # Parse: v0.0.2-235-gbe407e3
    import re

    m = re.match(r"^v(\d+\.\d+\.\d+)-(\d+)-g([0-9a-f]+)$", desc)
    if not m:
        return base

    distance = int(m.group(2))
    commit = m.group(3)

    if distance == 0:
        return base
    return f"{base}.dev{distance}+g{commit}"


def get_requires_for_dynamic_metadata(
    settings: dict[str, Any] | None = None,
) -> list[str]:
    return []
