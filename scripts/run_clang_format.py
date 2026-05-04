#!/usr/bin/env python3
"""Run clang-format over DNDSR C++ sources.

The style lives in /.clang-format at the project root and is used
unchanged.  This script only decides which files to format and runs
clang-format in parallel.

Quick usage::

    scripts/run_clang_format.py                       # all default modules
    scripts/run_clang_format.py Geom CFV              # module subset
    scripts/run_clang_format.py src/Geom/Mesh         # directory subtree
    scripts/run_clang_format.py src/DNDS/Defines.cpp  # specific file(s)
    scripts/run_clang_format.py --changed             # files dirty vs HEAD
    scripts/run_clang_format.py --since origin/main   # files changed vs a ref
    scripts/run_clang_format.py --check               # CI mode: exit 1 if drift

See --help for all options.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parent
PROJECT_ROOT = SCRIPTS_DIR.parent

DEFAULT_MODULES = ("DNDS", "Solver", "Geom", "CFV", "Euler", "EulerP")
DEFAULT_ROOTS = ("src", "app", "test/cpp")

FORMAT_EXTS = (
    ".c", ".cc", ".cpp", ".cxx",
    ".h", ".hpp", ".hxx",
    ".cu", ".cuh",
)


def _which_clang_format(user: str | None) -> str:
    binary = user or os.environ.get("CLANG_FORMAT") or "clang-format"
    if shutil.which(binary) is None:
        sys.exit(f"error: clang-format not found (looked for '{binary}')")
    return binary


def _walk_dir(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in FORMAT_EXTS:
            yield path


def _changed_files(since_ref: str | None, include_workdir: bool) -> list[Path]:
    paths: list[str] = []
    if include_workdir:
        for cmd in (
            ["git", "diff", "--name-only", "--diff-filter=ACMR"],
            ["git", "diff", "--name-only", "--diff-filter=ACMR", "--cached"],
        ):
            out = subprocess.run(
                cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False
            )
            paths.extend(line for line in out.stdout.splitlines() if line)
    if since_ref:
        out = subprocess.run(
            ["git", "diff", "--name-only", "--diff-filter=ACMR",
             f"{since_ref}...HEAD"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, check=False,
        )
        if out.returncode != 0:
            sys.exit(f"error: git diff vs '{since_ref}' failed:\n{out.stderr}")
        paths.extend(line for line in out.stdout.splitlines() if line)
    seen: set[str] = set()
    out_paths: list[Path] = []
    for p in paths:
        if p in seen:
            continue
        seen.add(p)
        full = (PROJECT_ROOT / p).resolve()
        if full.exists() and full.suffix in FORMAT_EXTS:
            out_paths.append(full)
    return out_paths


def _is_under_roots(path: Path, roots: Sequence[str]) -> bool:
    try:
        rel = path.resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        return False
    parts = rel.parts
    for root in roots:
        root_parts = tuple(root.split("/"))
        if parts[: len(root_parts)] == root_parts:
            return True
    return False


def _select_files(scope_args: Sequence[str]) -> list[Path]:
    tokens: list[str] = []
    for a in scope_args:
        tokens.extend(t for t in a.split(",") if t)

    collected: list[Path] = []

    if not tokens:
        for module in DEFAULT_MODULES:
            root = PROJECT_ROOT / "src" / module
            if root.is_dir():
                collected.extend(_walk_dir(root))
        for extra in ("app", "test/cpp"):
            root = PROJECT_ROOT / extra
            if root.is_dir():
                collected.extend(_walk_dir(root))
    else:
        for t in tokens:
            module_root = PROJECT_ROOT / "src" / t
            if "/" not in t and module_root.is_dir():
                collected.extend(_walk_dir(module_root))
                continue
            p = Path(t)
            if not p.is_absolute():
                p = (PROJECT_ROOT / t).resolve()
            else:
                p = p.resolve()
            if not p.exists():
                sys.exit(f"error: path not found: {p}")
            if p.is_dir():
                collected.extend(_walk_dir(p))
            elif p.suffix in FORMAT_EXTS:
                collected.append(p)

    seen: set[str] = set()
    uniq: list[Path] = []
    for p in collected:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return uniq


def _format_one(args: tuple[str, Path, bool]) -> tuple[Path, int, str, bool]:
    binary, path, check_only = args
    if check_only:
        res = subprocess.run(
            [binary, "--dry-run", "--Werror", str(path)],
            capture_output=True, text=True, check=False,
        )
        changed = res.returncode != 0
        return path, res.returncode, (res.stderr or res.stdout), changed
    else:
        res = subprocess.run(
            [binary, "-i", str(path)],
            capture_output=True, text=True, check=False,
        )
        return path, res.returncode, (res.stderr or res.stdout), False


def _run(
    files: Sequence[Path],
    binary: str,
    *,
    jobs: int,
    check_only: bool,
    quiet: bool,
) -> tuple[int, int, int]:
    n = len(files)
    width = len(str(n))
    n_changed = 0
    n_errors = 0
    exit_code = 0

    with cf.ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(_format_one, (binary, f, check_only)): f for f in files
        }

        def _sig(_signo, _frame):
            print("\n^C: cancelling...", file=sys.stderr)
            pool.shutdown(cancel_futures=True)
            sys.exit(130)
        old = signal.signal(signal.SIGINT, _sig)
        try:
            done = 0
            for fut in cf.as_completed(futures):
                path, rc, msg, changed = fut.result()
                done += 1
                rel = str(path.relative_to(PROJECT_ROOT)
                          ) if path.is_absolute() else str(path)
                if check_only:
                    if changed:
                        n_changed += 1
                        if not quiet:
                            print(
                                f"[{done:>{width}}/{n}] would reformat: {rel}")
                else:
                    if rc != 0:
                        n_errors += 1
                        print(f"[{done:>{width}}/{n}] ERROR   {rel}: {msg.strip()}",
                              file=sys.stderr)
                    elif not quiet:
                        print(f"[{done:>{width}}/{n}] {rel}")
        finally:
            signal.signal(signal.SIGINT, old)

    if check_only and n_changed > 0:
        exit_code = 1
    if n_errors > 0:
        exit_code = max(exit_code, 1)
    return exit_code, n_changed, n_errors


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_clang_format.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "scope",
        nargs="*",
        help=(
            "Module names (DNDS, Geom, ...), directory paths, or files. "
            "Empty means all default modules under src/, plus app/ and test/cpp/."
        ),
    )
    p.add_argument(
        "-j", "--jobs",
        type=int,
        default=os.cpu_count() or 4,
        help="Parallel jobs (default: all cores).",
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="Report only; exit 1 if any file would be reformatted.",
    )
    p.add_argument(
        "--changed",
        action="store_true",
        help="Only format files dirty in the work tree (staged + unstaged).",
    )
    p.add_argument(
        "--since",
        metavar="REF",
        help="Only format files changed vs REF.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Print only errors and summary.",
    )
    p.add_argument(
        "--list-files",
        action="store_true",
        help="Print the selected files and exit (no formatting).",
    )
    p.add_argument(
        "--clang-format",
        default=None,
        help="clang-format binary (default: $CLANG_FORMAT or 'clang-format').",
    )
    p.add_argument(
        "-n", "--dry-run",
        action="store_true",
        help="Print what would be run and exit (no clang-format invocations).",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    binary = _which_clang_format(args.clang_format)

    if args.changed or args.since:
        files = _changed_files(args.since, include_workdir=args.changed)
        files = [f for f in files if _is_under_roots(f, DEFAULT_ROOTS)]
        if args.scope:
            scoped = set(str(p.resolve()) for p in _select_files(args.scope))
            files = [f for f in files if str(f.resolve()) in scoped]
    else:
        files = _select_files(args.scope)

    if args.list_files:
        for f in files:
            rel = str(f.relative_to(PROJECT_ROOT)
                      ) if f.is_absolute() else str(f)
            print(rel)
        return 0

    version = subprocess.run(
        [binary, "--version"], capture_output=True, text=True, check=False
    ).stdout.strip()
    print(version or "(clang-format version unknown)")
    print(f"clang-format : {binary}")
    print(f"jobs         : {args.jobs}")
    print(f"files        : {len(files)}")
    print(f"mode         : {'check' if args.check else 'in-place'}")

    if args.dry_run:
        print("\n>> files (first 20 of {}):".format(len(files)))
        for f in files[:20]:
            rel = str(f.relative_to(PROJECT_ROOT)
                      ) if f.is_absolute() else str(f)
            print("   " + rel)
        return 0

    if not files:
        print("no files matched the selected scope -- nothing to do.")
        return 0

    rc, n_changed, n_errors = _run(
        files, binary, jobs=args.jobs, check_only=args.check, quiet=args.quiet,
    )

    print()
    if args.check:
        if n_changed == 0:
            print(f"all {len(files)} files are already formatted.")
        else:
            print(f"{n_changed}/{len(files)} files need reformatting.")
    else:
        msg = f"formatted {len(files) - n_errors}/{len(files)} files"
        if n_errors:
            msg += f" ({n_errors} errors)"
        print(msg + ".")
    return rc


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
