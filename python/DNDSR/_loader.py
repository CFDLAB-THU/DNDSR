"""Shared native library preloader for DNDSR submodules.

Each submodule (DNDS, Geom, CFV, EulerP) needs certain shared libraries
loaded via ctypes.CDLL *before* importing its pybind11 extension, because
the extension's DT_NEEDED entries may not resolve otherwise (the libraries
live in a non-standard directory next to the installed package).

Usage from a submodule __init__.py::

    from DNDSR._loader import preload
    preload("dnds")          # loads external deps + libdnds_shared.so
    preload("geom")          # loads libgeom_shared.so
"""

from __future__ import annotations

import os
import sys
from ctypes import CDLL
from pathlib import Path
from typing import Sequence

# Paths are resolved relative to *this* file's location.
# In an installed/editable layout the tree looks like:
#
#   python/DNDSR/_loader.py         <- this file
#   python/DNDSR/_lib/              <- shared C++ libraries
#   python/DNDSR/_lib/dndsr_external/  <- external deps (zlib, hdf5, ...)
#
# In the legacy src/ layout the tree looks like:
#
#   src/DNDSR/../lib/               <- shared C++ libraries
#   src/DNDSR/../lib/dndsr_external/

_THIS_DIR = Path(__file__).resolve().parent

_loaded: set[str] = set()


def _find_lib_dirs() -> tuple[Path, Path]:
    """Return (lib_dir, libext_dir) for the current installation."""
    # New layout: python/DNDSR/_lib/
    new_lib = _THIS_DIR / "_lib"
    new_ext = new_lib / "dndsr_external"
    if new_lib.is_dir():
        return new_lib, new_ext

    # Legacy layout: src/DNDSR/../lib/  (i.e. src/lib/)
    legacy_lib = _THIS_DIR.parent / "lib"
    legacy_ext = legacy_lib / "dndsr_external"
    if legacy_lib.is_dir():
        return legacy_lib, legacy_ext

    # Fallback: try relative to the *source tree* root (editable install
    # where python/ is a sibling of the CMake install prefix).
    # build/install/DNDSR/lib/
    for candidate in [
        _THIS_DIR.parent.parent / "build" / "install" / "DNDSR" / "lib",
    ]:
        if candidate.is_dir():
            return candidate, candidate / "dndsr_external"

    raise RuntimeError(
        "Cannot find DNDSR shared libraries.  "
        "Ensure the project is built and installed (pip install -e . or cmake --install)."
    )


def _load_so(path: Path) -> None:
    """Load a shared library with RTLD_GLOBAL so its symbols are available
    to subsequently loaded libraries.  Tolerates missing files."""
    import ctypes
    if path.exists():
        # RTLD_GLOBAL: make symbols available for subsequent DT_NEEDED resolution.
        # os.RTLD_DEEPBIND: prefer this library's own dependencies over
        # already-loaded ones (works around conda/anaconda providing an older
        # libstdc++ that was loaded at Python startup).
        #
        # RTLD_DEEPBIND can interfere with libraries that use dlopen and
        # expect to share symbols (e.g., some MPI implementations).  Set
        # the environment variable DNDSR_NO_DEEPBIND=1 to disable it for
        # debugging.  When disabled, LD_LIBRARY_PATH and RPATH control
        # symbol resolution as usual.
        use_deepbind = getattr(os, "RTLD_DEEPBIND", 0)
        if os.environ.get("DNDSR_NO_DEEPBIND", "").strip() in ("1", "true", "yes"):
            use_deepbind = 0
        mode = ctypes.RTLD_GLOBAL | use_deepbind
        CDLL(str(path), mode=mode)


# Library dependency graph (order matters: load dependencies first).
# libstdc++ must be loaded first if bundled, to override the (potentially
# older) version provided by conda/anaconda's Python runtime.
_EXTERNAL_LIBS: Sequence[str] = [
    "libstdc++.so",
    "libz.so",
    "libhdf5.so",
    "libcgns.so",
    "libmetis.so",
    "libparmetis.so",
]

_MODULE_LIBS: dict[str, Sequence[str]] = {
    "dnds":  ["libdnds_shared.so"],
    "geom":  ["libgeom_shared.so"],
    "cfv":   ["libcfv_shared.so"],
    "eulerP": ["libdnds_shared.so", "libgeom_shared.so",
               "libcfv_shared.so", "libeulerP_shared.so"],
}


def preload(module: str) -> None:
    """Preload shared libraries required by *module* (e.g. ``"dnds"``)."""
    if os.name != "posix":
        return

    if module in _loaded:
        return

    lib_dir, libext_dir = _find_lib_dirs()

    # External deps only need loading once (for the first module).
    if not _loaded:
        for name in _EXTERNAL_LIBS:
            _load_so(libext_dir / name)

    for name in _MODULE_LIBS.get(module, []):
        try:
            _load_so(lib_dir / name)
        except OSError as e:
            if "GLIBCXX" in str(e) or "CXXABI" in str(e):
                raise OSError(
                    f"{e}\n\n"
                    "This typically means Python loaded an older libstdc++ "
                    "(e.g. from conda/anaconda) before DNDSR's bundled version.\n"
                    "Fix: set LD_LIBRARY_PATH before running Python:\n"
                    f"  export LD_LIBRARY_PATH={libext_dir}:{lib_dir}:$LD_LIBRARY_PATH"
                ) from e
            raise

    _loaded.add(module)
