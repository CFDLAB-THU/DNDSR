#!/usr/bin/env python3
"""Parse process-local Runtime AllocationSnapshot dumps; render offline drilldown.

Allocation identity is deduplicated within one process/snapshot, never across
processes or successive snapshots. Aliases do not add bytes. Provider-private
memory absent from the registry remains unknown and is never inferred from GPU
usage reported by another tool or another process.
"""
from __future__ import annotations
import argparse
import gzip
import uuid
import html
import json
from pathlib import Path
import re

HEADER = re.compile(r"Runtime allocation ownership:\s+live_bytes=(\d+)\s+peak_bytes=(\d+)\s+pinned_host_bytes=(\d+)(?:\s+pid=(\d+))?")
RECORD = re.compile(r"^\s*(\d+)\s+(cuda|host):(-?\d+)\s+bytes=(\d+)(.*)$")
FIELD = re.compile(r"\s+(provider|stream_ordered|pinned_host|pending_reads|pending_writes|name|site)=")


def parse_snapshots(text: str, label: str, source: str = "") -> list[dict]:
    """Return snapshots with exact unique bytes and retained aliases/provenance.

    Lines outside a snapshot are ordinary solver output. A second occurrence
    of one identity can add aliases, but must agree on bytes/device/provider.
    Creation stacks are intentionally not embedded: sites/functions are kept,
    and the source log remains the authority for optional stack text.
    """
    snapshots: list[dict] = []
    current = None
    records: dict[int, dict] = {}
    in_stack = False

    def finish():
        if current is None:
            return
        current["allocations"] = sorted(records.values(), key=lambda r: (-r["bytes"], r["identity"]))
        current["unique_bytes"] = sum(r["bytes"] for r in records.values())
        current["summary_difference_bytes"] = current["reported_live_bytes"] - current["unique_bytes"]
        current["unknown_provider_bytes"] = None
        current["creation_stacks_embedded"] = False
        current["id"] = f"{label}:{len(snapshots)}:pid{current['pid']}"
        current["label"] = label if not snapshots else f"{label} · snapshot {len(snapshots) + 1}"
        snapshots.append(current)

    for line in text.splitlines():
        header = HEADER.search(line)
        if header:
            finish()
            records = {}
            current = {"source": source, "pid": int(header[4]) if header[4] else None,
                       "reported_live_bytes": int(header[1]), "reported_peak_bytes": int(header[2]),
                       "reported_pinned_host_bytes": int(header[3])}
            in_stack = False
            continue
        if line.startswith("allocation_stack identity="):
            in_stack = True
            continue
        if in_stack:
            if line.strip() == "end_allocation_stack":
                in_stack = False
            continue
        match = RECORD.match(line) if current is not None else None
        if not match:
            continue
        tail = match[5]
        fields = list(FIELD.finditer(tail))
        parsed: dict[str, list[str]] = {}
        for i, field in enumerate(fields):
            end = fields[i + 1].start() if i + 1 < len(fields) else len(tail)
            parsed.setdefault(field[1], []).append(tail[field.end():end].strip())
        record = {"identity": int(match[1]), "device": f"{match[2]}:{match[3]}", "bytes": int(match[4]),
                  "provider": parsed.get("provider", ["unknown"])[0],
                  "names": list(dict.fromkeys(parsed.get("name", []))),
                  "site": parsed.get("site", [""])[0]}
        for key in ("stream_ordered", "pinned_host", "pending_reads", "pending_writes"):
            value = parsed.get(key, [None])[0]
            record[key] = int(value) if value is not None else None
        old = records.get(record["identity"])
        if old:
            for key in ("bytes", "device", "provider"):
                if old[key] != record[key]:
                    raise ValueError(f"{label}: conflicting {key} for allocation {record['identity']}")
            old["names"] = list(dict.fromkeys(old["names"] + record["names"]))
            if old["site"] and record["site"] and old["site"] != record["site"]:
                old.setdefault("additional_sites", []).append(record["site"])
            elif record["site"]:
                old["site"] = record["site"]
        else:
            records[record["identity"]] = record
    finish()
    if not snapshots:
        raise ValueError(f"{label}: no Runtime AllocationSnapshot header found")
    return snapshots


STYLE = Path(__file__).with_name("explorer.css").read_text()
STANDALONE_STYLE = r"""
:root{color-scheme:light dark;--font-size-base:14px;--foreground:light-dark(#202938,#e6eaf1);--muted:light-dark(#526175,#b7c1d1);--muted-foreground:var(--muted);--background:light-dark(#fff,#171a20);--border:light-dark(#c7cdd5,#434d5e);--popover:var(--background);--popover-foreground:var(--foreground);--destructive:light-dark(#b11c36,#ff99a8);--viz-series-1:light-dark(#168a99,#62d0dd);--viz-series-2:light-dark(#648f2e,#add675);--viz-series-3:light-dark(#bb8038,#e8bb83);--viz-series-4:light-dark(#ac5f82,#df9cbd);--viz-series-5:light-dark(#7b6eac,#bfb3ed);--viz-series-6:light-dark(#3c8b71,#7ed9b8)}
body{font-family:system-ui,sans-serif;background:var(--background);color:var(--foreground);max-width:1200px;margin:auto;padding:1em}
.btn,.form-select{font:inherit;color:var(--foreground);background:var(--background);border:1px solid var(--muted);border-radius:.35em;padding:.35em .6em}
.btn{cursor:pointer}
.btn-ghost{border-color:transparent;background:transparent}
.btn-ghost:hover{background:color-mix(in srgb,var(--foreground) 7%,transparent)}
.btn[aria-current]{background:color-mix(in srgb,var(--foreground) 7%,transparent)}
.btn:disabled{opacity:.4;cursor:default}
.form-select{width:100%}
h1{font-weight:500;font-size:1.4em;margin:0 0 1em}
button:focus-visible,summary:focus-visible{outline:2px solid var(--foreground);outline-offset:2px}
.offline-tooltip{position:fixed;inset:auto;margin:0;padding:.6em .8em;max-width:min(360px,calc(100vw - 16px));width:max-content;white-space:pre-line;overflow-wrap:anywhere;pointer-events:none;font:inherit;color:var(--popover-foreground);background:var(--popover);border:1px solid var(--border);border-radius:.45em;box-shadow:0 4px 16px color-mix(in srgb,var(--foreground) 12%,transparent)}
@media(pointer:coarse){.btn,.form-select{min-height:44px}}
"""
SCRIPT = Path(__file__).with_name("explorer.js").read_text()
STANDALONE_TOOLTIPS = Path(__file__).with_name("standalone-tooltips.js").read_text()


def render_html(data: dict, fragment: bool = False, max_embedded_bytes: int = 1_000_000) -> str:
    """Self-contained HTML or embeddable fragment, with no network requests."""
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    if len(payload.encode()) > max_embedded_bytes:
        raise ValueError("allocation payload exceeds embedding budget; select fewer snapshots, never silently drop bytes")
    root = "allocation-explorer-" + uuid.uuid4().hex[:12]
    # A source path/name can contain HTML, so JSON must not terminate its script.
    safe_payload = payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    title = html.escape(data.get("title", "Runtime allocation explorer"))
    body = f'''<style>{STYLE.replace('ROOT', root)}</style>
<section id="{root}" aria-label="{title}">
<div class="controls"><div class="control"><label class="form-label" for="{root}-setup">Setup</label><select id="{root}-setup" class="setup form-select" aria-label="Setup snapshot"></select></div><div class="control"><label class="form-label" for="{root}-device">Memory</label><select id="{root}-device" class="device form-select" aria-label="Host or CUDA device"></select></div></div>
<nav class="crumbs" aria-label="Allocation path"></nav>
<div class="summary tabular-nums"><span class="total"></span><span class="count text-small text-muted"></span></div>
<div class="inspect"><span class="inspect-path"></span><span class="inspect-values text-small tabular-nums"></span></div>
<div class="map" role="group" aria-label="Immediate allocation children"></div>
<p class="warning" role="status"></p><details class="details"><summary class="detail-title">Allocation details</summary><div class="records"></div></details>
<div class="provenance text-small">Shared aliases counted once. Provider, pool, driver and other-process memory excluded.
<details><summary>Snapshot provenance</summary><div class="registry"></div><div class="source source-path"></div></details></div>
<span class="announcement sr-announcement" aria-live="polite"></span>
<script type="application/json" class="payload">{safe_payload}</script></section><script>{SCRIPT.replace('ROOT', root)}</script>'''
    offline_script = STANDALONE_TOOLTIPS.replace('ROOT', root)
    return body if fragment else f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{title}</title><style>{STANDALONE_STYLE}</style></head><body><h1>{title}</h1>{body}<script>{offline_script}</script></body></html>'


def read_text(path: Path) -> str:
    """Read an archived .gz snapshot or a plain process-local text dump."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return stream.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", action="append", metavar="LABEL=LOG", help="Repeat for full/triangle or other actual snapshots")
    source.add_argument("--data", type=Path, help="Render an existing exact snapshot JSON or JSON.gz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json", type=Path, help="Save deduplicated exact identity records")
    parser.add_argument("--fragment", action="store_true", help="Emit scoped HTML fragment for inline display")
    parser.add_argument("--synthetic", action="store_true", help="Mark parser/UI fixture; never solver evidence")
    parser.add_argument("--title", default="Runtime allocation explorer")
    args = parser.parse_args()
    if args.data:
        opener = gzip.open if args.data.suffix == '.gz' else open
        with opener(args.data, 'rt') as stream:
            data = json.load(stream)
        if data.get('format') != 'dndsr-allocation-explorer-v1' or not data.get('snapshots'):
            parser.error('unsupported or empty allocation snapshot data')
        snapshots = data['snapshots']
    else:
        snapshots = []
        for item in args.input:
            label, separator, filename = item.partition("=")
            if not separator:
                parser.error("--input requires LABEL=LOG")
            path = Path(filename)
            snapshots.extend(parse_snapshots(read_text(path), label, str(path)))
        data = {"format": "dndsr-allocation-explorer-v1", "title": args.title, "synthetic": args.synthetic, "snapshots": snapshots}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_html(data, args.fragment), encoding="utf-8")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        opener = gzip.open if args.json.suffix == '.gz' else open
        with opener(args.json, 'wt', encoding="utf-8") as stream:
            json.dump(data, stream, indent=2)
            stream.write("\n")
    print(f"Wrote {args.output}: {len(snapshots)} snapshots; {sum(len(s['allocations']) for s in snapshots)} unique snapshot identities")

if __name__ == "__main__":
    main()
