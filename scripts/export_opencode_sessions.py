#!/usr/bin/env python3
"""
Export all opencode sessions for a project as individual JSON files.

Usage:
    python scripts/export_opencode_sessions.py [--dir PROJECT_DIR] [--out OUTPUT_DIR]

Example:
    python scripts/export_opencode_sessions.py
    python scripts/export_opencode_sessions.py --dir /home/user/myproject --out ./backups
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def run_json(cmd: list[str]) -> dict | list:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running: {' '.join(cmd)}", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(result.returncode)
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(
        description="Export all opencode sessions for a project as JSON files."
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
    args = parser.parse_args()

    project_dir = str(Path(args.dir).resolve())
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Project directory: {project_dir}")
    print(f"Output directory:  {out_dir}")

    sessions = run_json(["opencode", "session", "list", "--format", "json"])
    project_sessions = [s for s in sessions if Path(
        s["directory"]).resolve() == Path(project_dir)]

    if not project_sessions:
        print("No sessions found for this project.")
        return 0

    print(f"Found {len(project_sessions)} session(s).\n")

    exported = 0
    for s in project_sessions:
        sid = s["id"]
        title = s["title"]
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
                raise SystemExit

            with open(tmp_path) as f:
                json.load(f)

            tmp_path.rename(out_path)
            exported += 1
            kb = out_path.stat().st_size // 1024
            print(f"OK ({kb}KB)")
            tmp_path = None

        except SystemExit:
            pass
        except subprocess.TimeoutExpired:
            print("TIMEOUT")
        except json.JSONDecodeError as exc:
            print(f"INVALID JSON: {exc}")
        except Exception as exc:
            print(f"FAILED: {exc}")
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

    print(
        f"\nExported {exported}/{len(project_sessions)} session(s) to {out_dir}")
    return 0 if exported == len(project_sessions) else 1


if __name__ == "__main__":
    sys.exit(main())
