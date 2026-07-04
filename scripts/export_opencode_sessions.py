#!/usr/bin/env python3
"""
Export or delete opencode sessions for a project.

Discovers sessions two ways:
  1. opencode session list --format json   (shows recent-version sessions)
  2. SQLite query of opencode's database   (finds older-version sessions too)

Merges both sources and deduplicates.

Usage:
    python scripts/export_opencode_sessions.py [--dir PROJECT_DIR] [--out OUTPUT_DIR]
                                               [--db PATH] [--no-db] [--skip-existing]
    python scripts/export_opencode_sessions.py --delete [--older-than DAYS]
                                               [--dir PROJECT_DIR] [--db PATH] [--dry-run]

Example:
    python scripts/export_opencode_sessions.py
    python scripts/export_opencode_sessions.py --delete --older-than 7 --dry-run
"""

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SECONDS_PER_DAY = 86400


def _project_dir_match(session_directory: str, project_dir: str) -> bool:
    return str(Path(session_directory).resolve()) == str(Path(project_dir))


def _find_db_path() -> Path | None:
    candidates = [
        Path.home() / ".local/share/opencode/opencode.db",
        Path.home() / ".opencode/opencode.db",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _load_existing_ids(out_dir: Path) -> set[str]:
    ids: set[str] = set()
    if not out_dir.exists():
        return ids
    for f in out_dir.iterdir():
        if f.suffix == ".json":
            ids.add(f.stem)
    return ids


def _run_json(cmd: list[str]) -> dict | list:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running: {' '.join(cmd)}", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(result.returncode)
    return json.loads(result.stdout)


def _gather_cli_sessions(project_dir: str) -> list[dict]:
    sessions = _run_json(["opencode", "session", "list", "--format", "json"])
    return [s for s in sessions if _project_dir_match(s["directory"], project_dir)]


def _gather_db_sessions(db_path: Path, project_dir: str) -> list[dict]:
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT id, title, directory, version, time_updated FROM session"
    ).fetchall()
    conn.close()
    return [
        {"id": r[0], "title": r[1], "directory": r[2],
            "version": r[3], "time_updated": r[4]}
        for r in rows
        if _project_dir_match(r[2], project_dir)
    ]


def _export_one(sid: str, title: str, out_dir: Path) -> bool:
    out_path = out_dir / f"{sid}.json"
    print(f"  Exporting {sid}  \"{title}\"", end=" ... ", flush=True)

    tmp_path = None
    try:
        now = int(time.time())
        prefix = f".{sid}.pid{os.getpid()}.t{now}."
        with tempfile.NamedTemporaryFile(
            mode="w", prefix=prefix, suffix=".json", dir=out_dir, delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)
            result = subprocess.run(
                ["opencode", "export", sid],
                stdout=tmp,
                stderr=subprocess.PIPE,
                text=True,
                timeout=300,
            )

        if result.returncode != 0:
            print(f"FAILED: {result.stderr.strip()}")
            return False

        with open(tmp_path) as f:
            json.load(f)

        tmp_path.rename(out_path)
        kb = out_path.stat().st_size // 1024
        print(f"OK ({kb}KB)")
        tmp_path = None
        return True

    except subprocess.TimeoutExpired:
        print("TIMEOUT")
        return False
    except json.JSONDecodeError as exc:
        print(f"INVALID JSON: {exc}")
        return False
    except Exception as exc:
        print(f"FAILED: {exc}")
        return False
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def _delete_one(sid: str, title: str) -> bool:
    print(f"  Deleting {sid}  \"{title}\"", end=" ... ", flush=True)
    try:
        result = subprocess.run(
            ["opencode", "session", "delete", sid],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            print(f"FAILED: {result.stderr.strip()}")
            return False
        print("OK")
        return True
    except subprocess.TimeoutExpired:
        print("TIMEOUT")
        return False
    except Exception as exc:
        print(f"FAILED: {exc}")
        return False


def _print_delete_preview(candidates: list[dict], age_days: int) -> None:
    print(
        f"\nSessions older than {age_days} day(s) (eligible for deletion):\n")
    for s in candidates:
        ts = s.get("time_updated")
        if ts:
            age = (time.time() - ts / 1000) / SECONDS_PER_DAY
            print(f"  {s['id']}  {age:.1f}d ago  \"{s.get('title', '')}\"")
        else:
            print(f"  {s['id']}  (age unknown)  \"{s.get('title', '')}\"")


def cmd_export(args) -> int:
    project_dir = str(Path(args.dir).resolve())
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Project directory: {project_dir}")
    print(f"Output directory:  {out_dir}")

    existing_ids = _load_existing_ids(out_dir) if args.skip_existing else set()

    cli_sessions = _gather_cli_sessions(project_dir)
    print(f"CLI sessions found: {len(cli_sessions)}")

    db_sessions: list[dict] = []
    if not args.no_db:
        db_path = Path(args.db) if args.db else _find_db_path()
        if db_path:
            print(f"DB path: {db_path}")
            db_sessions = _gather_db_sessions(db_path, project_dir)
            print(f"DB sessions found:  {len(db_sessions)}")
        else:
            print("No opencode database found; using CLI sessions only.")

    merged: dict[str, dict] = {}
    for s in db_sessions:
        merged[s["id"]] = s
    for s in cli_sessions:
        merged[s["id"]] = s

    if not merged:
        print("No sessions found for this project.")
        return 0

    total = len(merged)
    if existing_ids:
        todo = {k: v for k, v in merged.items() if k not in existing_ids}
        print(
            f"\nTotal sessions: {total}  ({len(todo)} new, {total - len(todo)} already exported)\n")
    else:
        todo = merged
        print(f"\nTotal sessions: {total}\n")

    exported = 0
    failed = 0
    for sid, s in sorted(todo.items(), key=lambda kv: kv[1].get("title", "")):
        if _export_one(sid, s.get("title", ""), out_dir):
            exported += 1
        else:
            failed += 1

    print(f"\nExported {exported}/{len(todo)} session(s) to {out_dir}")
    return 0 if failed == 0 else 1


def cmd_delete(args) -> int:
    project_dir = str(Path(args.dir).resolve())
    age_days = args.older_than
    cutoff_ms = int((time.time() - age_days * SECONDS_PER_DAY) * 1000)

    print(f"Project directory: {project_dir}")
    print(f"Delete sessions older than {age_days} day(s)")

    db_path = Path(args.db) if args.db else _find_db_path()
    if not db_path:
        print("Error: opencode database not found. Time info is required for delete.", file=sys.stderr)
        print("Specify --db PATH or ensure the database is in a standard location.", file=sys.stderr)
        return 1

    print(f"DB path: {db_path}")

    db_sessions = _gather_db_sessions(db_path, project_dir)

    now_ms = int(time.time() * 1000)
    candidates = [
        s for s in db_sessions
        if s.get("time_updated") is not None and s["time_updated"] < cutoff_ms
    ]

    no_time = [s for s in db_sessions if s.get("time_updated") is None]
    if no_time:
        print(
            f"Skipped {len(no_time)} session(s) with no time info (cannot determine age).")

    if not candidates:
        print(f"No sessions older than {age_days} day(s) found.")
        return 0

    _print_delete_preview(candidates, age_days)

    if args.dry_run:
        print(f"\n[dry-run] Would delete {len(candidates)} session(s).")
        return 0

    print(f"\n{len(candidates)} session(s) will be deleted.")
    try:
        answer = input("Confirm deletion? Type 'yes': ")
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.")
        return 1

    if answer.strip().lower() != "yes":
        print("Aborted.")
        return 1

    print()
    deleted = 0
    failed = 0
    for s in sorted(candidates, key=lambda kv: kv.get("time_updated", 0)):
        if _delete_one(s["id"], s.get("title", "")):
            deleted += 1
        else:
            failed += 1

    print(f"\nDeleted {deleted}/{len(candidates)} session(s)")
    return 0 if failed == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export or delete opencode sessions for a project."
    )
    parser.add_argument(
        "--dir",
        default=str(Path.cwd()),
        help="Project directory to filter sessions by (default: current directory)",
    )
    parser.add_argument(
        "--out",
        default="session_backup",
        help="Output directory for exported JSON files (default: ./session_backup)",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Path to opencode's SQLite database (auto-detected if omitted)",
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Skip database query; only use opencode session list",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip sessions whose JSON file already exists in the output directory",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete sessions instead of exporting them",
    )
    parser.add_argument(
        "--older-than",
        type=int,
        default=1,
        metavar="DAYS",
        help="When deleting, only delete sessions older than this many days (default: 1)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview which sessions would be deleted without actually deleting",
    )
    args = parser.parse_args()

    if args.delete:
        return cmd_delete(args)
    return cmd_export(args)


if __name__ == "__main__":
    sys.exit(main())
