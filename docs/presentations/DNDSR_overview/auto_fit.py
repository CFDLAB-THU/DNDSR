#!/usr/bin/env python3
"""
auto_fit.py — iteratively tag Marp slides with density classes that best
match their content volume.

Two-phase pipeline:

  Phase 1 — UPGRADE overflowing slides until the deck fits.
    Upgrade order: (none) → dense → denser → tight.

  Phase 2 — DOWNGRADE slides that have excess vertical slack.
    A slide tagged `dense`/`denser`/`tight` with >= DOWNGRADE_SLACK_PX
    of unused vertical space gets relaxed one step. Upgrades take
    priority (no downgrade happens until all slides fit).

Per iteration:
  1. Run build.sh --html to render the deck.
  2. Run check_overflow.js to get slides + slackY (JSON).
  3. Apply at most one class change per slide this iteration.
  4. Rebuild + re-check.
  5. Stop when the deck is stable (no upgrades AND no downgrades).
"""

import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent
PARTS_DIR = ROOT / "parts"
BUILD_SH = ROOT / "build.sh"
CHECK_JS = ROOT / "check_overflow.js"
OUT_MD = ROOT.parent / "DNDSR_overview.md"
OUT_HTML = ROOT.parent / "DNDSR_overview.html"
OUT_JSON = ROOT.parent / "DNDSR_overview.overflow.json"

# Order matters: earlier → later is the upgrade direction.
DENSITY_ORDER = ["", "dense", "denser", "tight"]

MAX_ITER = 6
TOL_PX = 2

# A slide with a density class can be relaxed one step when it has at
# least this many pixels of unused vertical space. Large enough that
# we don't flip-flop across the overflow threshold on subsequent runs.
DOWNGRADE_SLACK_PX = 200


def run(cmd, **kw):
    result = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise SystemExit(f"command failed: {' '.join(map(str, cmd))}")
    return result


def build_html():
    run(["bash", str(BUILD_SH), "--html"])


def run_checker():
    env = {"NODE_PATH": "/tmp/marp-overflow-checker/node_modules"}
    import os
    full_env = {**os.environ, **env}
    result = subprocess.run(
        ["node", str(CHECK_JS), str(OUT_HTML)],
        capture_output=True, text=True, env=full_env,
    )
    # exit code 1 = overflow detected (expected); 0 = clean; >=2 = error
    if result.returncode >= 2:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise SystemExit("overflow checker failed")
    print(result.stdout)
    return json.loads(OUT_JSON.read_text())


def deck_slide_map():
    """Return list of (part_filename, local_slide_index_1based) for each deck slide."""
    part_files = sorted(PARTS_DIR.glob("*.md"))
    mapping = []
    for fp in part_files:
        text = fp.read_text()
        slides = re.split(r"(?m)^---\s*$", text)
        non_empty = [(j, s) for j, s in enumerate(slides) if s.strip()]
        for local_idx, _ in enumerate(non_empty, start=1):
            mapping.append((fp.name, local_idx))
    return mapping


def current_class(slide_text: str) -> str:
    m = re.search(r"<!--\s*_class:\s*([A-Za-z0-9_ ]+?)\s*-->", slide_text)
    if not m:
        return ""
    # Filter out chapter/lead meta; only density tokens matter here.
    tokens = [t for t in m.group(1).split() if t in (
        "dense", "denser", "tight")]
    return tokens[0] if tokens else ""


def upgrade_class(cls: str) -> str | None:
    try:
        idx = DENSITY_ORDER.index(cls)
    except ValueError:
        idx = 0
    if idx + 1 >= len(DENSITY_ORDER):
        return None
    return DENSITY_ORDER[idx + 1]


def downgrade_class(cls: str) -> str | None:
    try:
        idx = DENSITY_ORDER.index(cls)
    except ValueError:
        return None
    if idx <= 0:
        return None
    return DENSITY_ORDER[idx - 1]


def set_slide_class(slide_text: str, new_cls: str) -> str:
    """Replace existing density class or insert a new one."""
    # If there's an existing _class: directive with dense/denser/tight, replace the token.
    # We only replace the density token; keep other class words intact.
    def repl(m):
        tokens = m.group(1).split()
        tokens = [t for t in tokens if t not in ("dense", "denser", "tight")]
        tokens.append(new_cls)
        return f"<!-- _class: {' '.join(tokens)} -->"

    class_re = re.compile(r"<!--\s*_class:\s*([A-Za-z0-9_ ]+?)\s*-->")
    if class_re.search(slide_text):
        return class_re.sub(repl, slide_text, count=1)

    # Otherwise, insert the directive.
    footer_re = re.compile(r"(<!--\s*_footer:.*?-->)")
    m = footer_re.search(slide_text)
    if m:
        insert_at = m.end()
        return (slide_text[:insert_at]
                + f"\n<!-- _class: {new_cls} -->"
                + slide_text[insert_at:])

    stripped = slide_text.lstrip("\n")
    lead_ws = slide_text[:len(slide_text) - len(stripped)]
    return lead_ws + f"<!-- _class: {new_cls} -->\n\n" + stripped


def apply_changes(changes):
    """changes: list of (part_filename, local_idx_1based, new_class)."""
    per_file = {}
    for f, i, c in changes:
        per_file.setdefault(f, []).append((i, c))

    for fname, items in per_file.items():
        fp = PARTS_DIR / fname
        text = fp.read_text()
        slides = re.split(r"(?m)^---\s*$", text)
        non_empty = [j for j, s in enumerate(slides) if s.strip()]

        for local_idx, new_cls in items:
            j = non_empty[local_idx - 1]
            slides[j] = set_slide_class(slides[j], new_cls)

        fp.write_text("---".join(slides))


def _slide_source_class(fname: str, local: int) -> tuple[str, list[str], int]:
    """Return (current_density_class, all_slides_list, non_empty_j).

    Looks up the live class token in the source file for a given
    (filename, 1-based non-empty slide index). Used so we make changes
    against the authoritative source rather than cached HTML output.
    """
    fp = PARTS_DIR / fname
    slides_list = re.split(r"(?m)^---\s*$", fp.read_text())
    non_empty = [j for j, sl in enumerate(slides_list) if sl.strip()]
    j = non_empty[local - 1]
    return current_class(slides_list[j]), slides_list, j


def main():
    deck_map = deck_slide_map()
    print(f"Deck has {len(deck_map)} slides (per source split)")

    for it in range(1, MAX_ITER + 1):
        print(f"\n=== iteration {it} ===")
        build_html()
        report = run_checker()

        overflowing = [s for s in report["slides"] if s["overflows"]]
        changes = []

        # ---- Phase 1: upgrade overflowing slides (priority) ----
        if overflowing:
            unfixable = []
            for s in overflowing:
                idx = s["index"]
                fname, local = deck_map[idx - 1]
                cur, _, _ = _slide_source_class(fname, local)
                up = upgrade_class(cur)
                if up is None:
                    unfixable.append((idx, s["title"], cur, s["overflowY"]))
                    continue
                changes.append((fname, local, up))
                print(
                    f"  UP   deck#{idx:3d} ({fname}[{local}]) "
                    f"{cur or '(none)'} -> {up}  "
                    f"'{s['title'][:50]}' [-{s['overflowY']}px]"
                )
            if unfixable:
                print("\n  Unfixable (already at 'tight'):")
                for (idx, title, cur, oy) in unfixable:
                    print(f"    deck#{idx}: {cur}  [-{oy}px]  '{title[:60]}'")

        # ---- Phase 2: downgrade over-classed slides when the deck fits ----
        else:
            # Only content slides are candidates; chapter/lead cards don't
            # carry density classes.
            candidates = [
                s for s in report["slides"]
                if s["kind"] == "content" and s["slackY"] >= DOWNGRADE_SLACK_PX
            ]
            # Relax the slackiest first so one or two pass per iteration
            # dominates; the next iteration re-measures.
            candidates.sort(key=lambda s: -s["slackY"])
            for s in candidates:
                idx = s["index"]
                fname, local = deck_map[idx - 1]
                cur, _, _ = _slide_source_class(fname, local)
                dn = downgrade_class(cur)
                if dn is None:
                    continue
                changes.append((fname, local, dn))
                print(
                    f"  DOWN deck#{idx:3d} ({fname}[{local}]) "
                    f"{cur} -> {dn or '(none)'}  "
                    f"'{s['title'][:50]}' [+{s['slackY']}px slack]"
                )

        if not changes:
            if overflowing:
                print("  no more upgrades available — stop")
                return 1
            print("✓ deck stable (no more overflows and no relaxable slides)")
            return 0

        apply_changes(changes)

    print(
        f"\nReached MAX_ITER={MAX_ITER}; rebuilding once more for final state")
    build_html()
    report = run_checker()
    return 1 if [s for s in report["slides"] if s["overflows"]] else 0


if __name__ == "__main__":
    raise SystemExit(main())
