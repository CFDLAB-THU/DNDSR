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
        #
        # RTLD_DEEPBIND is OFF by default.  It makes each library prefer its
        # own dependencies over already-loaded ones, which was intended to
        # work around conda/anaconda shipping an older libstdc++ that gets
        # loaded at Python startup.  However, when the bundled and system
        # libstdc++ are the same (or close) version, RTLD_DEEPBIND creates
        # two separate allocator instances that cause double-free crashes
        # (e.g., "free(): double free detected in tcache 2") because MPI
        # and other system libraries use the system libstdc++ while DNDSR
        # uses the bundled copy.
        #
        # Set DNDSR_USE_DEEPBIND=1 to enable it (only needed when running
        # under conda/anaconda with an incompatible libstdc++).
        use_deepbind = 0
        if os.environ.get("DNDSR_USE_DEEPBIND", "").strip() in ("1", "true", "yes"):
            use_deepbind = getattr(os, "RTLD_DEEPBIND", 0)
        mode = ctypes.RTLD_GLOBAL | use_deepbind
        CDLL(str(path), mode=mode)


# Library dependency graph (order matters: load dependencies first).
#
# Only libraries built by cfd_externals are preloaded here.  System-provided
# libraries (libstdc++, libmpi) must NOT be bundled or preloaded:
#
# - libstdc++: bundling it causes dual-allocator double-free crashes when
#   system libraries (e.g. libmpi) use the system copy while DNDSR uses
#   the bundled copy.  If conda provides an older libstdc++, use
#   LD_LIBRARY_PATH or DNDSR_USE_DEEPBIND=1 instead.
#
# - libmpi: MPI libraries are tightly coupled to the system MPI runtime
#   (orted/prte, PMIx, fabric drivers).  A bundled copy from one machine
#   will not work on another and conflicts with the system mpirun.
_EXTERNAL_LIBS: Sequence[str] = [
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
