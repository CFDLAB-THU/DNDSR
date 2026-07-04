"""JSON-with-comments patching via line-based replacement.

Usage:
    from jsonc_util import read_config, patch_config

    cfg = read_config("config.json")
    cfg["eulerSettings"]["farFieldStaticValue"]["state"][1] = -1616.6
    patch_config("config.json", ["eulerSettings", "farFieldStaticValue"],
                 cfg["eulerSettings"]["farFieldStaticValue"])
"""

from __future__ import annotations

import json
import re


def read_config(path):
    """Read a JSONC file, stripping ``//`` line comments, return parsed dict."""
    with open(path) as f:
        text = f.read()
    cleaned = re.sub(r'//[^\n]*', '', text)
    return json.loads(cleaned)


def patch_config(path, key, new_value):
    """Replace the value of top-level-adjacent key *key* in *path* with *new_value*.

    *key* — a single key name (string).  Finds any key at any nesting level,
    matches by name; if multiple keys share the name the first is used.

    *new_value* — a Python object, serialised with ``json.dumps(indent=4)``.

    The closing ``}`` or ``]`` of the old value is detected by looking for a
    line that contains only ``}``, ``]``, or ``},`` / ``],`` at the same
    indentation as the key itself.  All other content (including ``//``
    comments and whitespace) is preserved.
    """
    with open(path) as f:
        lines = f.readlines()

    # 1. Find the key line
    start = None
    indent = None
    for i, line in enumerate(lines):
        m = re.match(r'^(\s*)"([^"]+)"\s*:', line)
        if m and m.group(2) == key:
            indent = m.group(1)
            start = i
            break
    if start is None or indent is None:
        raise KeyError(f"Key {key!r} not found in {path}")

    # 2. Find the closing line — the first line at the same indent that is
    #    a lone } or ] (or },, ],).
    end = start
    for i in range(start + 1, len(lines)):
        stripped = lines[i].strip()
        if stripped in ("}", "]", "},"):
            ws = lines[i][: -len(lines[i].lstrip())
                          ] if lines[i].lstrip() else ""
            if ws == indent:
                end = i
                break

    # 3. Serialize and indent the new value
    ser = json.dumps(new_value, indent=4)
    val_lines = ser.splitlines(True)

    # If the key line ends in '{' or '[', strip that trailing brace from
    # the key line (the serialized value already includes it).
    key_prefix = lines[start].split(":", 1)[0].rstrip()
    key_tail = lines[start].split(":", 1)[1].strip()
    if key_tail in ("{", "[", "{,", "[,"):
        lines[start] = f"{key_prefix}: {val_lines[0].lstrip()}"
        val_lines = val_lines[1:]

    # Indent one level deeper than the key
    val_indent = indent + "    "
    indented = [val_indent + ln.lstrip() for ln in val_lines]

    # 4. Assemble the replacement block
    block = [lines[start]]   # keep the (modified) '"key": …' line
    block.extend(indented)

    # 5. Ensure trailing comma if the next sibling needs it
    next_line = end + 1
    while next_line < len(lines) and lines[next_line].strip() == "":
        next_line += 1
    if next_line < len(lines):
        m = re.match(r'^(\s*)"', lines[next_line])
        if m and m.group(1) == indent:
            if not block[-1].rstrip().endswith(","):
                block[-1] = block[-1].rstrip("\n") + ",\n"

    result = lines[:start] + block + lines[end + 1:]
    with open(path, "w") as f:
        f.writelines(result)
