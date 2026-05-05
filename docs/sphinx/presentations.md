(presentations_page)=
# Presentations

Slide decks introducing DNDSR. Each deck is authored in Markdown and
rendered with [Marp](https://marp.app/); the CMake docs pipeline builds
the HTML and PDF outputs and stages them into this site under
`/presentations/`.

## DNDSR Overview

An 89-slide technical introduction for software and CS engineers,
covering architecture, the Geom pipeline, numerics, parallelism, I/O
and interop, solvers, engineering quality, and roadmap.

```{raw} html
<ul>
  <li><strong>Open slideshow:</strong>
      <a href="../presentations/DNDSR_overview.html" target="_blank"
         rel="noopener">DNDSR_overview.html</a></li>
  <li><strong>Download PDF:</strong>
      <a href="../presentations/DNDSR_overview.pdf" target="_blank"
         rel="noopener">DNDSR_overview.pdf</a></li>
  <li><strong>Source tree:</strong>
      <code>docs/presentations/DNDSR_overview/</code></li>
</ul>
```

Embedded preview (open in a new tab for fullscreen navigation):

```{raw} html
<iframe
    src="../presentations/DNDSR_overview.html"
    width="100%"
    height="600"
    style="border: 1px solid #d0d7de; border-radius: 6px;"
    title="DNDSR Overview Slide Deck"
    loading="lazy">
</iframe>
```

## Building the decks

Decks are rebuilt as part of the documentation target:

```sh
cmake --build build -t docs
```

To render a single deck manually (requires `marp-cli`):

```sh
bash docs/presentations/DNDSR_overview/build.sh --html --pdf
```

The build script assembles per-chapter Markdown fragments, copies image
assets listed in `res_manifest.txt` into `docs/presentations/res/`, and
(with `--html` / `--pdf`) invokes Marp to render the final slide decks.

See `docs/presentations/DNDSR_overview/README.md` for editing workflow,
overflow-detection tooling, and density-class conventions.
