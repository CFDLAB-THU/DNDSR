# DNDSR Overview Deck

Source for the comprehensive Marp presentation at
`docs/presentations/DNDSR_overview.md`.

## Layout

```
docs/presentations/
├── DNDSR_overview.md         <- GENERATED combined Markdown
├── DNDSR_overview.html       <- GENERATED (build.sh --html)
├── DNDSR_overview.pdf        <- GENERATED (build.sh --pdf)
├── res/                      <- GENERATED image cache (build.sh)
└── DNDSR_overview/           <- SOURCE TREE
    ├── 00_frontmatter.md      YAML + <style> (Light-GitHub theme)
    ├── parts/
    │   ├── 00_title.md            Title slide
    │   ├── 01_chapter_1.md ... 09_chapter_9.md
    ├── res_manifest.txt        Image SOURCE → TARGET mapping
    ├── build.sh                Assemble + render + image copy
    ├── check_overflow.js       Headless-Chrome overflow detector
    ├── auto_fit.py             Iterative tagger (dense → denser → tight)
    └── README.md               This file
```

Totals: **89 slides** across 9 chapters + title + 9 chapter cards.

## Build

```bash
# Assemble the combined Markdown + copy images into res/ (no rendering)
bash docs/presentations/DNDSR_overview/build.sh

# Assemble + render PDF
bash docs/presentations/DNDSR_overview/build.sh --pdf

# Assemble + render HTML
bash docs/presentations/DNDSR_overview/build.sh --html

# Both
bash docs/presentations/DNDSR_overview/build.sh --pdf --html

# Skip the image-copy step (rarely needed)
bash docs/presentations/DNDSR_overview/build.sh --no-res

# Verbose (echo every copy and part append)
bash docs/presentations/DNDSR_overview/build.sh --verbose
```

Outputs go to `docs/presentations/DNDSR_overview.{md,pdf,html}` and
`docs/presentations/res/*.png`.

Requires `marp-cli` on PATH for `--pdf` / `--html` (install via
`npm install -g @marp-team/marp-cli`).

## Images and `res_manifest.txt`

Images referenced by slides live under canonical locations in the repo
(`docs/elements/`, `docs/theory/`, etc.). To keep the rendered HTML deck
self-contained and portable, every referenced image is copied into
`docs/presentations/res/` at build time. Slide sources reference them as
`res/<filename>.png`.

The build script reads the mapping from `res_manifest.txt`:

```
# SOURCE_PATH (repo-relative)            TARGET_FILENAME (under res/)
docs/elements/Tri3_nodes.png             Tri3_nodes.png
docs/elements/Hex27_nodes.png            Hex27_nodes.png
docs/theory/PyramidShow.png              PyramidShow.png
```

Rules:

- Blank lines and `#` comments are ignored.
- Any whitespace run (spaces/tabs) separates the two fields.
- Existing target files that already match the source byte-for-byte are
  skipped — build is idempotent.
- Missing source files cause the build to **fail** (no silent gaps in the
  deck).

### Adding a new image

1. Place the authoritative image in an appropriate repo location
   (`docs/elements/`, `docs/theory/`, or similar).
2. Add a line to `res_manifest.txt` mapping it into `res/`.
3. Reference it from a slide as `res/<filename>`.
4. Run `bash build.sh` — the file gets copied automatically.

Do **not** commit anything to `docs/presentations/res/` directly — it is
a generated cache. Add it to `.gitignore` if the repo prefers that
(current choice: commit the cache for faster CI renders).

## Editing a chapter

Open the relevant `parts/NN_chapter_N.md` file and edit in place. The
chapter-card slide (`<!-- _class: chapter -->`) is the first `---`-separated
section in each file. All subsequent sections are content slides.

Slide separators are plain `---` lines. Each content slide starts with an
optional per-slide directive comment:

```markdown
---

<!-- _footer: "src/DNDS/Array.hpp:100-150" -->
<!-- _class: dense -->

## My slide title

Content...
```

After editing, run `build.sh` to regenerate the combined Markdown. Then
check overflow (see below) before committing.

## Overflow control

Marp sections are fixed at 1280×720 px (16:9). Content that exceeds the
`clientHeight` is silently clipped in the PDF — Marp does **not** page-break
a slide automatically, and it does **not** auto-scale the whole section.
(It does `data-auto-scaling="downscale-only"` for `<pre>` and math blocks,
which helps inside a code fence but not for long lists or full pages.)

### Density classes available

Four density classes are defined in `00_frontmatter.md`. Apply them with a
per-slide directive comment (`<!-- _class: NAME -->`):

| Class     | Base font | `h2` | `pre` | Use                                        |
|-----------|-----------|------|-------|--------------------------------------------|
| (default) | 21 px     | 28 px| 15 px | normal slide                               |
| `dense`   | 18 px     | 25 px| 13 px | tight tables / many bullets                |
| `denser`  | 16 px     | 22 px| 12 px | dense reference slides                     |
| `tight`   | 14 px     | 20 px| 11 px | maximum density — use sparingly            |

The classes also reduce padding, row padding in tables, `h3` size, and
two-column gap. See `00_frontmatter.md` for the full CSS.

### The overflow workflow

Every time you edit a slide (add bullets, grow a code block, add a table
row), follow this three-step workflow **before committing**:

```bash
# 1. Rebuild the combined Markdown + HTML
bash docs/presentations/DNDSR_overview/build.sh --html

# 2. Run the overflow detector
NODE_PATH=/tmp/marp-overflow-checker/node_modules \
    node docs/presentations/DNDSR_overview/check_overflow.js \
         docs/presentations/DNDSR_overview.html

# 3a. If clean (exit 0), you are done.
# 3b. If overflowing slides are reported, either:
#     - Run the iterative auto-fitter, OR
#     - Fix the slides manually (preferred for large overflows).
```

### One-time setup for the overflow detector

`check_overflow.js` uses **headless Chrome via `puppeteer-core`**, which
reuses the system `/usr/bin/google-chrome` binary (no Chromium download).
Install `puppeteer-core` into a shared location once per machine:

```bash
mkdir -p /tmp/marp-overflow-checker
cd /tmp/marp-overflow-checker
npm init -y
npm install puppeteer-core
```

The `NODE_PATH=/tmp/marp-overflow-checker/node_modules` env var in the
commands above points Node at this install. You can also install
`puppeteer-core` globally (`npm install -g puppeteer-core`) if you
prefer; in that case drop the `NODE_PATH=` prefix.

If `google-chrome` is not at `/usr/bin/google-chrome` on your system, edit
the `CHROME_PATH` constant at the top of `check_overflow.js`.

### Step-by-step decision tree

```
  build --html  +  check_overflow.js
        │
        ├── No overflow?  →  DONE.
        │
        └── Overflow report printed
                │
                ├── Small (< 30 px on 1-3 slides)  →  run auto_fit.py
                │
                ├── Medium (30-100 px on any slide)
                │       │
                │       ├── Is the slide content cohesive?
                │       │     YES  →  promote class by one step
                │       │              (none → dense → denser → tight)
                │       │              run auto_fit.py or edit manually
                │       │
                │       └── Can it be split cleanly?
                │             YES  →  split into two slides
                │                     (prefer splitting over "tight")
                │
                └── Large (> 100 px or already at "tight")
                        │
                        ├── Trim content (shorter bullets, drop a
                        │       code example, move detail to footer)
                        ├── Split into two or more slides
                        └── Reformat (two columns → one, tables → prose,
                                      or vice versa)
```

### Option A — Automated (recommended for small overflows)

`auto_fit.py` drives the whole loop: render → check → upgrade class →
repeat. It upgrades each overflowing slide by **one** step per iteration
(`(none)` → `dense` → `denser` → `tight`) and stops when the deck is clean
or no further upgrades are possible.

```bash
python3 docs/presentations/DNDSR_overview/auto_fit.py
```

Typical output:

```
=== iteration 1 ===
Slides inspected: 89
Overflowing:      6
  48  content  dense   29  CUDA path — DeviceTransferable CRTP
  33  content  dense   22  Variational Reconstruction — the functional
  ...
  upgrade deck#48 (05_chapter_5.md[6]) dense -> denser  [-29px]
  ...

=== iteration 2 ===
Slides inspected: 89
Overflowing:      0
✓ all slides fit
```

Converges in 1–3 iterations for typical content changes. Max 4 iterations
before giving up (configurable via `MAX_ITER` in the script).

**When auto-fit is the right tool:**

- Small, generalized overflow (< 30 px) spread across several slides after
  a reformat.
- The slides are already well-composed and should not be split.
- You are OK with tighter text; the reader is expected to view the PDF on
  a full-size screen.

**When auto-fit is NOT the right tool:**

- A single slide overflows by 100+ px — that means the content simply
  doesn't fit, and upgrading to `tight` hides the problem instead of
  solving it.
- The overflow is caused by one oversized element (a 40-line code block,
  a too-wide table) — shrinking the whole slide is the wrong fix.
- The slide is already `tight` and still overflows — you must edit.

### Option B — Manual fix (recommended for large or structural overflows)

For each flagged slide:

1. **Read what the detector reported.** Each entry shows `overflowY`
   (vertical overflow in px), current class, and title. Sort by
   `overflowY` descending — tackle the worst first.

2. **Diagnose.** Open the slide in `parts/NN_chapter_N.md` and look for:
   - **Long code blocks** (> 20 lines) — trim, or split the code across
     two slides.
   - **Many bullets** (> 7 in one column) — split the bullet list across
     two columns using `<div class="cols">`, or move half to a second
     slide.
   - **Wide tables** — shrink column count, drop a column, or switch
     `denser` (reduces both font and padding).
   - **Mermaid diagrams** — Mermaid renders at its natural size; if the
     diagram is tall, either simplify nodes or split into two diagrams.
   - **Stacked `h3` sub-sections** — the `h3` top-margin adds up; reduce
     count or promote one `h3` to `h2` on a new slide.

3. **Apply the fix.**
   - Prefer **splitting** over **shrinking** for slides with coherent
     subsections (e.g. "X — part 1 / 2").
   - Prefer **shrinking** (density class) when the content is a dense
     reference table that should stay whole.
   - Add a density class directive right after the `_footer` comment:

     ```markdown
     <!-- _footer: "src/DNDS/Array.hpp:100-150" -->
     <!-- _class: denser -->

     ## My slide title
     ```

4. **Rebuild and re-check.** `bash build.sh --html` + the detector again.

### Manual split example

Before (overflows by 200+ px — too dense at any font size):

```markdown
---

## 13 Riemann solvers

```cpp
enum RiemannSolverType {
    UnknownRS = 0,
    Roe = 1, HLLC = 2, HLLEP = 3, HLLEP_V1 = 21,
    Roe_M1 = 11, Roe_M2 = 12, ...
};
```

| Variant | Entropy-fix / eigenvalue scheme |
| ...13 rows... |

Shared helper: `ComputeRoePreamble<dim>()`.
```

After (split into two slides, each with density `dense`):

```markdown
---

<!-- _class: dense -->

## 13 Riemann solvers (1/2) — enum and naming

```cpp
enum RiemannSolverType { ... };
```

Four families: Roe, HLLC, HLLEP, Roe_M1..M9.

---

<!-- _class: dense -->

## 13 Riemann solvers (2/2) — entropy fixes

| Variant | Scheme |
| ...13 rows... |
```

### Pre-commit hook (optional but recommended)

Wire the overflow check into a pre-commit hook or CI job so no overflow
slips into main:

```bash
#!/usr/bin/env bash
# .git/hooks/pre-commit (excerpt)
set -e
bash docs/presentations/DNDSR_overview/build.sh --html > /dev/null
NODE_PATH=/tmp/marp-overflow-checker/node_modules \
    node docs/presentations/DNDSR_overview/check_overflow.js \
         docs/presentations/DNDSR_overview.html
# exit code 1 → overflow detected → commit blocked
```

### Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Detector reports new overflows on unchanged slides | Chrome font-metric variability between runs (< 5 px) | Re-run; add tolerance via `TOL_PX` in `check_overflow.js` |
| Slide with `tight` class still overflows | Content genuinely doesn't fit | Split or trim — do NOT escalate further |
| `fit` or `fit2` appears in an old commit | Removed in favor of `tight` | `sed -i 's/_class: fit[0-9]*/_class: tight/g' parts/*.md` |
| PDF page doesn't match HTML slide | `marp --pdf` vs `--html` rendering diff | Check with `--pdf` directly; for embedding, always pass `--allow-local-files` |
| Mermaid blocks render raw text | Outdated marp-cli | Upgrade: `npm install -g @marp-team/marp-cli` |

## Design choices

- **Theme:** custom Light-GitHub palette (GitHub Primer colors) embedded
  in `00_frontmatter.md` as a `<style>` block. No external CSS file.
- **Math:** MathJax (`math: mathjax` in the YAML front-matter).
- **Mermaid:** fenced ` ```mermaid ` blocks; marp-cli renders them
  automatically in recent versions. For older versions, pass
  `--mermaid` to the CLI.
- **Images:** 13 element PNGs from `../elements/` and
  `../theory/PyramidShow.png` are referenced by relative path. Build
  output must sit at `docs/presentations/DNDSR_overview.md` for these to
  resolve.
- **Footer citations:** per-slide `<!-- _footer: "file:line" -->`
  directives provide source references in the page footer.

## CI integration

The build script has no external dependencies beyond `bash` and `awk` for
the assembly step, so it is safe to call from CI. A simple pre-commit
check could be:

```bash
bash docs/presentations/DNDSR_overview/build.sh --html
NODE_PATH=/tmp/marp-overflow-checker/node_modules \
    node docs/presentations/DNDSR_overview/check_overflow.js \
         docs/presentations/DNDSR_overview.html
# exit code 1 on overflow — fail the build
```
