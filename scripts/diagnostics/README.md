# Runtime allocation explorer

`allocation_explorer.py` converts process-local Runtime allocation snapshots
into a self-contained, offline HTML tree map. It uses Python's standard library;
the generated file embeds its data and the adjacent CSS/JavaScript assets.
There are no network requests or external browser dependencies.

```sh
python3 scripts/diagnostics/allocation_explorer.py \
  --input 'baseline=/path/to/archive/baseline.txt.gz' \
  --input 'compact=/path/to/archive/compact.txt.gz' \
  --json /path/to/archive/allocations.json.gz \
  --output /path/to/archive/explorer.html

python3 scripts/diagnostics/allocation_explorer.py \
  --data /path/to/archive/allocations.json.gz \
  --output /path/to/archive/explorer.html
```

The rectangular map shows only the immediate children of the selected group.
Click a rectangle or a child-list entry to descend; use Up, the path bar,
Left Arrow or Escape to return. Hover/focus previews the exact byte count and
path. Setup and placement controls preserve a matching navigation path when
possible. Allocation leaves expose aliases, identity, device/provider, creation
site and pending access counts. This is capacity accounting, excluding allocator
pools, driver storage, provider-private allocations and unrelated processes.

Keep actual traces, compressed archives, generated HTML, screenshots and browser
receipts in the plan or experiment evidence archive. Do not add them here.
The converter never reads GPU payloads and never modifies its inputs. Names
and source paths are treated as text, including when embedded in HTML.
IDs are deduplicated within one process/snapshot; an inconsistent duplicate is
rejected, and a registry-sum mismatch remains visible. No small allocations
are silently dropped to fit the HTML embedding budget.

The three UI assets (`explorer.js`, `explorer.css`, `standalone-tooltips.js`)
originate from the improved Plan 61-3 explorer. This directory is the maintained
implementation; historical plan evidence remains reproducible and unchanged.

Run the four parser/export tests with:

```sh
python3 -m unittest discover -s scripts/tests -p test_allocation_explorer.py -v
```

The optional browser check requires already installed Chrome and a Node version
with `WebSocket`. It compares at least two snapshots containing a common `CFV`
group on `cuda:0`, as in the actual full/compact solver evidence. It does not
download tools. All generated screenshots/receipts go to the explicit second
argument; neither the input HTML nor this source directory is modified:

```sh
node scripts/diagnostics/test_allocation_explorer.mjs \
  /path/to/archive/explorer.html /path/to/archive/browser-checks
```

It checks pointer/focus tooltips, one-level navigation, exact leaf metadata,
preserved paths, stale state echoes, navigation response time and the 320px
layout. See `docs/architecture/AllocationDiagnostics.md` for solver output
controls and the limits of the underlying allocation registry.
