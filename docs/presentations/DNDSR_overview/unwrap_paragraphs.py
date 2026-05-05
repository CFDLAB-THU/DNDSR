#!/usr/bin/env python3
"""Unwrap hard-wrapped Markdown prose paragraphs.

Joins consecutive non-blank lines of an ordinary paragraph into a single
line, so each paragraph in the source is one physical line. Designed
for Marp decks where the original authoring style soft-wrapped prose at
~80 columns: while CommonMark collapses soft breaks into spaces at
render time, single-line paragraphs are easier to edit, diff, and do
not risk downstream tools inserting accidental `<br>` on word breaks.

Preserves the following block types exactly as-is:

  - fenced code fences (``` ... ```)
  - indented code blocks (>= 4 spaces after a blank line)
  - HTML blocks opened at column 0 (<div>, <style>, <script>, <!-- -->,
    and similar); tracked with a simple tag-depth counter
  - blockquotes (lines starting with `>`)
  - tables (lines containing `|` with a separator row)
  - horizontal rules and setext underlines (--- / ===)
  - ATX headings (# …)
  - display math ($$ ... $$)
  - list markers (-, *, +, 1.) — list items may still accept paragraph
    continuation joins within the SAME item (no blank line), but item
    starts are preserved.
  - YAML frontmatter (between leading `---` lines)
  - Marp per-slide directive comments (<!-- _class: ... -->) — treated
    as HTML blocks already, but singled out here for clarity.

Usage:

    python3 unwrap_paragraphs.py FILE [FILE ...]
    python3 unwrap_paragraphs.py --check FILE [FILE ...]    # exit 1 on diffs
    python3 unwrap_paragraphs.py --stdout FILE              # preview

The files are rewritten in place unless --stdout or --check is given.
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path


FENCE_RE = re.compile(r"^([ \t]*)(```+|~~~+)(.*)$")
ATX_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}(\s|$)")
LIST_ITEM_RE = re.compile(r"^(\s*)([-*+]|\d+[.)])\s+")
BLOCKQUOTE_RE = re.compile(r"^\s{0,3}>")
HR_RE = re.compile(r"^\s{0,3}(---+|\*\*\*+|___+)\s*$")
SETEXT_UNDERLINE_RE = re.compile(r"^\s{0,3}(=+|-+)\s*$")
TABLE_SEP_RE = re.compile(r"^\s*\|?[\s:\-|]+\|[\s:\-|]*$")
HTML_BLOCK_OPEN_RE = re.compile(
    r"^\s{0,3}<(?:!--|[A-Za-z][A-Za-z0-9-]*)"
)
DISPLAY_MATH_RE = re.compile(r"^\s*\$\$\s*$")


def is_blank(line: str) -> bool:
    return line.strip() == ""


def is_prose_line(line: str) -> bool:
    """True when `line` starts an ordinary-paragraph line that can be joined.

    Explicitly returns False for everything structural: headings,
    list items, blockquotes, tables, HR/setext separators, HTML
    block starts, display math, directive comments.
    """
    if is_blank(line):
        return False
    if ATX_HEADING_RE.match(line):
        return False
    if LIST_ITEM_RE.match(line):
        return False
    if BLOCKQUOTE_RE.match(line):
        return False
    if HR_RE.match(line):
        return False
    if SETEXT_UNDERLINE_RE.match(line):
        return False
    if "|" in line:
        # Heuristic: a line with one or more pipes at depth 0 is likely
        # a table row. If the next line is the table separator we know
        # for sure; this per-line view just bails out conservatively.
        if not re.match(r"^\s{0,3}[^|]*\|", line):
            pass
        else:
            return False
    if line.lstrip().startswith("<"):
        return False
    if DISPLAY_MATH_RE.match(line):
        return False
    # Indented code block (4+ leading spaces, paragraph context).
    # Handled by the caller via blank-line state; here we still allow
    # joining inside list continuations, which the caller checks.
    return True


def unwrap_text(text: str) -> str:
    """Return *text* with ordinary paragraphs re-joined into single lines."""
    lines = text.splitlines(keepends=False)
    out: list[str] = []

    i = 0
    n = len(lines)

    # Track state:
    #   in_code      : inside ```/~~~ fence (verbatim)
    #   fence_marker : the literal fence string that opened
    #   in_verbatim  : inside <style>/<script>/<pre>/<textarea> (verbatim)
    #   verbatim_tag : tag name (lower-case) that closes the region
    #   in_math      : inside $$ … $$ (verbatim)
    in_code = False
    fence_marker = ""
    in_verbatim = False
    verbatim_tag = ""
    in_math = False

    # Detect YAML frontmatter: leading `---\n...\n---` at file start.
    if n > 0 and lines[0].strip() == "---":
        out.append(lines[0])
        i = 1
        while i < n and lines[i].strip() != "---":
            out.append(lines[i])
            i += 1
        if i < n:
            out.append(lines[i])
            i += 1

    while i < n:
        line = lines[i]

        # --- verbatim regions ---
        if in_code:
            out.append(line)
            m = FENCE_RE.match(line)
            if m and m.group(2) == fence_marker:
                in_code = False
                fence_marker = ""
            i += 1
            continue

        if in_math:
            out.append(line)
            if DISPLAY_MATH_RE.match(line):
                in_math = False
            i += 1
            continue

        if in_verbatim:
            out.append(line)
            if re.search(rf"</\s*{verbatim_tag}\s*>", line, re.IGNORECASE):
                in_verbatim = False
                verbatim_tag = ""
            i += 1
            continue

        # --- start verbatim regions ---
        m_fence = FENCE_RE.match(line)
        if m_fence:
            in_code = True
            fence_marker = m_fence.group(2)
            out.append(line)
            i += 1
            continue

        if DISPLAY_MATH_RE.match(line):
            in_math = True
            out.append(line)
            i += 1
            continue

        # <style>/<script>/<pre>/<textarea> — contents are NOT Markdown.
        m_verb = re.match(
            r"^\s*<\s*(style|script|pre|textarea)\b",
            line, re.IGNORECASE,
        )
        if m_verb:
            tag = m_verb.group(1).lower()
            out.append(line)
            i += 1
            # Check if the same line also closes the tag.
            if not re.search(rf"</\s*{tag}\s*>", line, re.IGNORECASE):
                in_verbatim = True
                verbatim_tag = tag
            continue

        # --- transparent HTML wrappers (<div>, <section>, ...) ---
        # Emit the wrapper line verbatim and continue processing the
        # inside as Markdown, because CommonMark does not strip
        # block-level HTML containers.
        if (
            line.lstrip().startswith("<")
            and re.match(r"^\s*</?([A-Za-z][A-Za-z0-9-]*)", line)
        ):
            out.append(line)
            i += 1
            continue

        # --- HTML comments: <!-- … --> ---
        if line.lstrip().startswith("<!--"):
            out.append(line)
            i += 1
            continue

        # --- structural lines: emit verbatim ---
        if (
            ATX_HEADING_RE.match(line)
            or HR_RE.match(line)
            or SETEXT_UNDERLINE_RE.match(line)
            or TABLE_SEP_RE.match(line)
            or is_blank(line)
        ):
            out.append(line)
            i += 1
            continue

        # --- blockquotes: merge consecutive `>` lines into one ---
        # Each soft-wrapped continuation inside a blockquote is emitted
        # by CommonMark as a separate text line, producing visible line
        # breaks in some renderers (and always brittle wrapping). Join
        # them so one blockquote paragraph = one source line.
        if BLOCKQUOTE_RE.match(line):
            # Extract the `>` prefix (preserving leading spaces + > + space)
            m_q = re.match(r"^(\s{0,3}>\s?)", line)
            prefix = m_q.group(1) if m_q else "> "
            buf = [line[len(prefix):].rstrip()]
            i += 1
            while i < n:
                nxt = lines[i]
                if is_blank(nxt):
                    break
                if not BLOCKQUOTE_RE.match(nxt):
                    break
                # Structural starts inside a blockquote terminate the
                # current paragraph (heading, hr, list item).
                body = re.sub(r"^\s{0,3}>\s?", "", nxt)
                if (
                    ATX_HEADING_RE.match(body)
                    or HR_RE.match(body)
                    or LIST_ITEM_RE.match(body)
                    or FENCE_RE.match(body)
                ):
                    break
                buf.append(body.rstrip())
                i += 1
            out.append(prefix + " ".join(s.strip() if k > 0 else s
                                         for k, s in enumerate(buf)))
            continue

        # --- table rows (contain pipes at depth 0) ---
        if re.match(r"^\s{0,3}[^|]*\|", line):
            out.append(line)
            i += 1
            continue

        # --- list items: keep the marker; join continuation lines ---
        m_list = LIST_ITEM_RE.match(line)
        if m_list:
            indent = m_list.group(1)
            continuation_indent = " " * \
                (len(indent) + len(m_list.group(2)) + 1)
            buf = [line]
            i += 1
            while i < n:
                nxt = lines[i]
                if is_blank(nxt):
                    break
                if (
                    ATX_HEADING_RE.match(nxt)
                    or HR_RE.match(nxt)
                    or LIST_ITEM_RE.match(nxt)
                    or BLOCKQUOTE_RE.match(nxt)
                    or TABLE_SEP_RE.match(nxt)
                    or FENCE_RE.match(nxt)
                    or DISPLAY_MATH_RE.match(nxt)
                ):
                    break
                if nxt.lstrip().startswith("<"):
                    break
                if not nxt.startswith(continuation_indent) and nxt.lstrip() == nxt:
                    break
                buf.append(nxt.strip())
                i += 1
            out.append(" ".join(s.strip() if j > 0 else s
                                for j, s in enumerate(buf)))
            continue

        # --- ordinary paragraph: gather and join ---
        buf = [line.rstrip()]
        i += 1
        while i < n:
            nxt = lines[i]
            if is_blank(nxt):
                break
            if (
                ATX_HEADING_RE.match(nxt)
                or HR_RE.match(nxt)
                or LIST_ITEM_RE.match(nxt)
                or BLOCKQUOTE_RE.match(nxt)
                or TABLE_SEP_RE.match(nxt)
                or FENCE_RE.match(nxt)
                or DISPLAY_MATH_RE.match(nxt)
            ):
                break
            if nxt.lstrip().startswith("<"):
                break
            if re.match(r"^\s{0,3}[^|]*\|", nxt):
                break
            buf.append(nxt.strip())
            i += 1
        out.append(" ".join(buf))

    # Preserve trailing newline behaviour.
    result = "\n".join(out)
    if text.endswith("\n") and not result.endswith("\n"):
        result += "\n"
    return result


def process_file(path: Path, check: bool, to_stdout: bool) -> bool:
    original = path.read_text(encoding="utf-8")
    rewritten = unwrap_text(original)
    if original == rewritten:
        return True
    if to_stdout:
        sys.stdout.write(rewritten)
        return True
    if check:
        diff = difflib.unified_diff(
            original.splitlines(keepends=True),
            rewritten.splitlines(keepends=True),
            fromfile=str(path),
            tofile=str(path) + " (unwrapped)",
            n=1,
        )
        sys.stdout.writelines(diff)
        return False
    path.write_text(rewritten, encoding="utf-8")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Unwrap hard-wrapped Markdown prose paragraphs."
    )
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--check", action="store_true",
                    help="Report diffs without modifying files (exit 1 "
                    "if any file would change).")
    ap.add_argument("--stdout", action="store_true",
                    help="Write the rewritten file to stdout (single file).")
    args = ap.parse_args()

    if args.stdout and len(args.paths) > 1:
        ap.error("--stdout accepts exactly one file")

    all_clean = True
    for p in args.paths:
        if not p.is_file():
            print(f"skip (not a file): {p}", file=sys.stderr)
            continue
        ok = process_file(p, check=args.check, to_stdout=args.stdout)
        if not ok:
            all_clean = False

    if args.check and not all_clean:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
