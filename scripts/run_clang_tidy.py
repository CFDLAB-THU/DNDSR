#!/usr/bin/env python3
"""Drive clang-tidy over the DNDSR C++ sources.

The checks live in /.clang-tidy at the project root and are used
unchanged.  This script only decides which files to tidy, drives
clang-tidy in parallel, and prints a per-check summary.

Quick usage::

    scripts/run_clang_tidy.py                       # all modules under src/
    scripts/run_clang_tidy.py Geom CFV              # module subset
    scripts/run_clang_tidy.py Geom,CFV              # same (comma-separated)
    scripts/run_clang_tidy.py src/Geom/Mesh         # any path under src/
    scripts/run_clang_tidy.py src/DNDS/Defines.cpp  # one file
    scripts/run_clang_tidy.py --changed             # files dirty vs HEAD
    scripts/run_clang_tidy.py --since origin/main   # files changed vs a ref
    scripts/run_clang_tidy.py --fix src/DNDS        # apply .clang-tidy-fix
    scripts/run_clang_tidy.py --summary             # only the summary
    scripts/run_clang_tidy.py --top-checks 20       # show 20 most common checks

See --help for the full option list.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import datetime as _dt
import json
import os
import re
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

TU_EXTS = (".c", ".cc", ".cpp", ".cxx")
CUDA_EXTS = (".cu",)

CHECK_RE = re.compile(
    # Check names always have a '-' somewhere (category-name-...).
    # This excludes noise like [loc], [pos], [name], [nodiscard] that
    # can appear in clang-tidy notes/hints.
    r"\[([a-z][a-z0-9]*(?:[.-][a-z0-9]+)+)(?:,-warnings-as-errors)?\]"
)

SITE_RE = re.compile(
    # Match a clang-tidy warning/error/note line:
    #   path:line:col: severity: message [check]
    r"^(.+?):(\d+):(\d+):\s+(warning|error|note):\s+(.+?)(?:\s+\[(.+?)\])?\s*$"
)


def _resolve_build_dir(arg: str | None) -> Path:
    if arg:
        return Path(arg).resolve()
    env = os.environ.get("DNDS_BUILD_DIR")
    if env:
        return Path(env).resolve()
    return (PROJECT_ROOT / "build").resolve()


def _which_clang_tidy(user: str | None) -> str:
    binary = user or os.environ.get("CLANG_TIDY") or "clang-tidy"
    if shutil.which(binary) is None:
        sys.exit(f"error: clang-tidy not found (looked for '{binary}')")
    return binary


def _load_compile_db(build_dir: Path) -> list[dict]:
    path = build_dir / "compile_commands.json"
    if not path.exists():
        sys.exit(
            f"error: no compile_commands.json in {build_dir}\n"
            "hint: configure with -DCMAKE_EXPORT_COMPILE_COMMANDS=ON "
            "or -DDNDS_GENERATE_COMPILE_COMMANDS=ON"
        )
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        sys.exit(f"error: failed to parse {path}: {exc}")


def _tu_files_from_db(db: Iterable[dict]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for entry in db:
        path = entry.get("file")
        if not path:
            continue
        if path in seen:
            continue
        seen.add(path)
        out.append(path)
    return out


def _changed_files(since_ref: str | None, include_workdir: bool) -> list[str]:
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
    deduped: list[str] = []
    for p in paths:
        if p in seen:
            continue
        seen.add(p)
        deduped.append(p)
    return deduped


def _is_under(path: Path, roots: Sequence[str]) -> bool:
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


def _match_scope(
    all_tus: Sequence[str],
    scope_args: Sequence[str],
    *,
    include_cu: bool,
) -> list[str]:
    tokens: list[str] = []
    for a in scope_args:
        tokens.extend(t for t in a.split(",") if t)

    exts = tuple(TU_EXTS) + (tuple(CUDA_EXTS) if include_cu else ())

    def ext_ok(p: str) -> bool:
        return p.endswith(exts)

    if not tokens:
        modules = DEFAULT_MODULES
        allowed_prefixes = {
            str((PROJECT_ROOT / "src" / m).resolve()) + os.sep
            for m in modules
        }
        for extra_root in ("app", "test/cpp"):
            allowed_prefixes.add(
                str((PROJECT_ROOT / extra_root).resolve()) + os.sep
            )
        return [
            tu for tu in all_tus
            if ext_ok(tu) and any(tu.startswith(p) for p in allowed_prefixes)
        ]

    def looks_like_path(tok: str) -> bool:
        return (
            "/" in tok
            or tok.startswith(".")
            or os.path.isabs(tok)
            or (PROJECT_ROOT / tok).exists()
        )

    paths_toks = [t for t in tokens if looks_like_path(t)]
    module_toks = [t for t in tokens if not looks_like_path(t)]

    resolved_prefixes: set[str] = set()
    explicit_files: set[str] = set()

    for t in module_toks:
        p = (PROJECT_ROOT / "src" / t).resolve()
        if not p.is_dir():
            sys.exit(f"error: module '{t}' not found at {p}")
        resolved_prefixes.add(str(p) + os.sep)

    for t in paths_toks:
        p = Path(t)
        if not p.is_absolute():
            p = (PROJECT_ROOT / t).resolve()
        else:
            p = p.resolve()
        if not p.exists():
            sys.exit(f"error: path not found: {p}")
        if p.is_dir():
            resolved_prefixes.add(str(p) + os.sep)
        else:
            explicit_files.add(str(p))

    selected: list[str] = []
    for tu in all_tus:
        if tu in explicit_files:
            selected.append(tu)
            continue
        if not ext_ok(tu):
            continue
        if any(tu.startswith(pref) for pref in resolved_prefixes):
            selected.append(tu)
    return selected


def _base_args(
    clang_tidy: str,
    build_dir: Path,
    config_file: Path,
    header_filter: str | None,
    fix: bool,
) -> list[str]:
    args = [
        clang_tidy,
        "-p", str(build_dir),
        "--config-file", str(config_file),
    ]
    if header_filter:
        args += ["--header-filter", header_filter]
    if fix:
        args += ["--fix-errors"]
    return args


def _run_one(cmd: list[str]) -> tuple[str, int, str]:
    file = cmd[-1]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except KeyboardInterrupt:
        raise
    output = (res.stdout or "") + (res.stderr or "")
    return file, res.returncode, output


def _run_parallel(
    files: Sequence[str],
    base: Sequence[str],
    jobs: int,
    *,
    quiet: bool,
    log_path: Path,
) -> tuple[int, collections.Counter[str]]:
    exit_code = 0
    counts: collections.Counter[str] = collections.Counter()
    n = len(files)
    width = len(str(n))

    futures: dict[cf.Future, str] = {}

    def _cancel_all(pool: cf.Executor) -> None:
        pool.shutdown(cancel_futures=True)

    with log_path.open("w") as log, cf.ThreadPoolExecutor(max_workers=jobs) as pool:
        for f in files:
            futures[pool.submit(_run_one, list(base) + [f])] = f

        def _sig(_signo, _frame):
            print("\n^C: cancelling...", file=sys.stderr)
            _cancel_all(pool)
            sys.exit(130)

        old = signal.signal(signal.SIGINT, _sig)
        try:
            done_n = 0
            for fut in cf.as_completed(futures):
                done_n += 1
                file, rc, output = fut.result()
                if rc != 0 and exit_code == 0:
                    exit_code = rc
                rel = os.path.relpath(file, PROJECT_ROOT)
                header = f"[{done_n:>{width}}/{n}] {rel}  (rc={rc})"
                log.write(header + "\n")
                if output:
                    log.write(output)
                    if not output.endswith("\n"):
                        log.write("\n")
                if not quiet:
                    print(header)
                    if output.strip():
                        sys.stdout.write(output)
                        if not output.endswith("\n"):
                            sys.stdout.write("\n")
                counts.update(CHECK_RE.findall(output))
        finally:
            signal.signal(signal.SIGINT, old)
    return exit_code, counts


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_clang_tidy.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "scope",
        nargs="*",
        help=(
            "Module names (e.g. DNDS, Geom), comma-separated module lists "
            "(e.g. Geom,CFV), directory paths, or individual files. "
            "Empty means all default modules."
        ),
    )
    p.add_argument(
        "-j", "--jobs",
        type=int,
        default=os.cpu_count() or 4,
        help="Parallel jobs (default: all cores).",
    )
    p.add_argument(
        "--build",
        default=None,
        help=(
            "Build directory with compile_commands.json "
            "(default: $DNDS_BUILD_DIR or ./build)."
        ),
    )
    p.add_argument(
        "--config-file",
        default=None,
        help="Override .clang-tidy path (default: <root>/.clang-tidy).",
    )
    p.add_argument(
        "--fix",
        action="store_true",
        help="Use .clang-tidy-fix and apply --fix-errors.",
    )
    p.add_argument(
        "--include-cu",
        action="store_true",
        help=(
            "Also tidy .cu files. Off by default: nvcc-only flags in "
            "compile_commands.json break clang's CUDA frontend."
        ),
    )
    p.add_argument(
        "--no-header-filter",
        action="store_true",
        help="Skip project header filter on the command line (debug).",
    )
    p.add_argument(
        "--header-filter",
        default=None,
        help=(
            "Override the --header-filter regex. "
            "Default: '.*/DNDSR/(src|app|test/cpp)/.*'."
        ),
    )
    scope_grp = p.add_argument_group("scope shortcuts")
    scope_grp.add_argument(
        "--changed",
        action="store_true",
        help="Only files dirty in the work tree (staged + unstaged).",
    )
    scope_grp.add_argument(
        "--since",
        metavar="REF",
        help="Only files changed vs REF (git ref). Can combine with --changed.",
    )
    out_grp = p.add_argument_group("output")
    out_grp.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-file output; only print the summary.",
    )
    out_grp.add_argument(
        "--summary",
        action="store_true",
        help="Alias for --quiet.",
    )
    out_grp.add_argument(
        "--top-checks",
        type=int,
        default=0,
        metavar="N",
        help="Limit the summary to the top-N checks (0 = show all).",
    )
    out_grp.add_argument(
        "--log-dir",
        default=None,
        help="Directory for log files (default: <build>/clang-tidy-logs).",
    )
    out_grp.add_argument(
        "--list-files",
        action="store_true",
        help="Print the selected files and exit (no tidy run).",
    )
    out_grp.add_argument(
        "--list",
        action="store_true",
        help=(
            "After the run, extract and print every unique "
            "file:line:col: warning/error site (deduplicated "
            "by location + message). Implies --quiet."
        ),
    )
    p.add_argument(
        "--clang-tidy",
        default=None,
        help="clang-tidy binary (default: $CLANG_TIDY or 'clang-tidy').",
    )
    p.add_argument(
        "-n", "--dry-run",
        action="store_true",
        help="Print what would be run and exit.",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Propagate clang-tidy's non-zero exit status. By default the "
            "script exits 0 regardless of diagnostics (advisory mode)."
        ),
    )
    p.add_argument(
        "--unsafe-parallel-fix",
        action="store_true",
        help=(
            "Allow --fix to run with jobs>1. Unsafe: parallel fix passes "
            "on the same header race and can corrupt files. Use only on "
            "disjoint scopes (e.g. one TU) or with serialized fanout "
            "handled by the caller."
        ),
    )
    return p


def _print_site_list(log_path: Path) -> None:
    """Parse clang-tidy log and print unique warning/error sites."""
    seen: set[tuple[str, str, str, str]] = set()  # (file, line, col, message)
    grouped: dict[str, list[tuple[str, str, str, str]]
                  ] = collections.defaultdict(list)
    if not log_path.exists():
        return
    for line in log_path.read_text().splitlines():
        m = SITE_RE.match(line)
        if not m:
            continue
        file, lno, col, sev, msg, check = m.groups()
        # Skip note lines (they accompany a warning/error)
        if sev == "note":
            continue
        key = (file, lno, col, msg)
        if key in seen:
            continue
        seen.add(key)
        check_name = check or "?"
        grouped[check_name].append((file, lno, col, msg))
    if not grouped:
        return
    print()
    print("=== site listing ===")
    for check_name in sorted(grouped.keys()):
        sites = grouped[check_name]
        print(f"\n[{check_name}]  ({len(sites)} site(s))")
        for file, lno, col, msg in sites:
            # Only print the short filename (relative to project)
            try:
                rel = str(Path(file).resolve().relative_to(PROJECT_ROOT))
            except ValueError:
                rel = file
            msg_trunc = msg[:120] + "…" if len(msg) > 120 else msg
            print(f"  {rel}:{lno}:{col}: {msg_trunc}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    build_dir = _resolve_build_dir(args.build)
    if not build_dir.is_dir():
        sys.exit(f"error: build directory does not exist: {build_dir}")
    clang_tidy = _which_clang_tidy(args.clang_tidy)

    if args.config_file:
        config = Path(args.config_file).resolve()
    elif args.fix:
        config = PROJECT_ROOT / ".clang-tidy-fix"
    else:
        config = PROJECT_ROOT / ".clang-tidy"
    if not config.exists():
        sys.exit(f"error: config file not found: {config}")

    header_filter: str | None
    if args.no_header_filter:
        header_filter = None
    else:
        header_filter = args.header_filter or ".*/DNDSR/(src|app|test/cpp)/.*"

    log_dir = Path(args.log_dir) if args.log_dir else build_dir / \
        "clang-tidy-logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"run-{stamp}.log"

    db = _load_compile_db(build_dir)
    all_tus = _tu_files_from_db(db)

    if args.changed or args.since:
        changed = _changed_files(args.since, include_workdir=args.changed)
        accepted_exts = tuple(TU_EXTS) + (tuple(CUDA_EXTS)
                                          if args.include_cu else ())
        rel_roots = tuple(DEFAULT_ROOTS)
        abs_changed: set[str] = set()
        for rel in changed:
            if not rel.endswith(accepted_exts):
                continue
            abs_path = str((PROJECT_ROOT / rel).resolve())
            if _is_under(Path(abs_path), rel_roots):
                abs_changed.add(abs_path)
        selected = [tu for tu in all_tus if tu in abs_changed]
        if args.scope:
            scoped = set(_match_scope(all_tus, args.scope,
                         include_cu=args.include_cu))
            selected = [tu for tu in selected if tu in scoped]
    else:
        selected = _match_scope(all_tus, args.scope,
                                include_cu=args.include_cu)

    if args.list_files:
        for f in selected:
            print(os.path.relpath(f, PROJECT_ROOT))
        return 0

    if not selected:
        print("no files matched the selected scope -- nothing to do.")
        return 0

    base = _base_args(
        clang_tidy=clang_tidy,
        build_dir=build_dir,
        config_file=config,
        header_filter=header_filter,
        fix=args.fix,
    )

    version = subprocess.run(
        [clang_tidy, "--version"], capture_output=True, text=True, check=False
    ).stdout.splitlines()[:1]

    # --fix cannot run in parallel safely: multiple TUs editing the same
    # header concurrently will race and corrupt the file (interleaved
    # insertions). Force -j 1 unless the user explicitly opts in with
    # --unsafe-parallel-fix.
    effective_jobs = args.jobs
    if args.fix and not args.unsafe_parallel_fix and args.jobs > 1:
        print("note: --fix forces jobs=1 (parallel --fix is unsafe); "
              "pass --unsafe-parallel-fix to override.")
        effective_jobs = 1

    print(version[0] if version else "(clang-tidy version unknown)")
    print(f"clang-tidy   : {clang_tidy}")
    print(f"config       : {config}")
    print(f"compile db   : {build_dir / 'compile_commands.json'}")
    print(f"jobs         : {effective_jobs}")
    print(f"files        : {len(selected)}")
    if header_filter:
        print(f"header filter: {header_filter}")
    print(f"fix mode     : {args.fix}")
    print(f"log          : {log_path}")

    if args.dry_run:
        print("\n>> base command:")
        print("   " + " ".join(base))
        print("\n>> files (first 20 of {}):".format(len(selected)))
        for f in selected[:20]:
            print("   " + os.path.relpath(f, PROJECT_ROOT))
        return 0

    quiet = args.quiet or args.summary or args.list
    exit_code, counts = _run_parallel(
        selected, base, jobs=effective_jobs, quiet=quiet, log_path=log_path,
    )

    if args.list:
        _print_site_list(log_path)

    print()
    print(f"=== diagnostic summary ({log_path}) ===")
    if not counts:
        print("  (no diagnostics)")
    else:
        items = counts.most_common(args.top_checks or None)
        total = sum(counts.values())
        for name, n in items:
            print(f"  {n:>6}  {name}")
        print("  " + "-" * 6)
        print(f"  {total:>6}  TOTAL ({len(counts)} distinct checks)")

    return exit_code if args.strict else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
