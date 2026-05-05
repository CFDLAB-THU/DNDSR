---
marp: true
theme: default
paginate: true
math: mathjax
size: 16:9
header: "DNDSR — CFD Research Code"
footer: "v0.2.0 · CFD Lab, Tsinghua University"
---

<!--
  DNDSR comprehensive overview deck.

  This file is GENERATED. Source lives in:
      docs/presentations/DNDSR_overview/
          00_frontmatter.md
          parts/00_title.md
          parts/01_chapter_1.md
          ...
          parts/09_chapter_9.md

  To rebuild:
      bash docs/presentations/DNDSR_overview/build.sh

  To render directly from source (PDF):
      bash docs/presentations/DNDSR_overview/build.sh --pdf

  Best viewer:   "Marp for VS Code" extension (Mermaid + MathJax built-in).

  Paths to images are relative to the final DNDSR_overview.md location:
      ../elements/... and ../theory/...

  Overflow-control classes (append per-slide as Marp directives):
      _class: dense     -- 18px base  (tighter tables / lots of bullets)
      _class: denser    -- 16px base  (very dense reference slides)
      _class: tight     -- 14px base  (maximum density; use sparingly)
-->

<style>
  /* --- Light-GitHub palette overrides ------------------------------------ */
  :root {
    --gh-fg:        #1f2328;
    --gh-fg-muted:  #656d76;
    --gh-accent:    #660874;
    --gh-accent-2:  #8b1a9e;
    --gh-bg:        #ffffff;
    --gh-code-bg:   #f6f8fa;
    --gh-border:    #d0d7de;
    --gh-success:   #1a7f37;
    --gh-danger:    #cf222e;
    --gh-warning:   #9a6700;
    --gh-mono:      ui-monospace, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace;
    --gh-sans:      -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", Helvetica, Arial, sans-serif;
  }

  section {
    background: var(--gh-bg);
    color:      var(--gh-fg);
    font-family: var(--gh-sans);
    font-size:  21px;
    padding:    44px 56px 52px 56px;
    border-top: 3px solid var(--gh-accent);
  }

  section.lead {
    text-align: center;
    justify-content: center;
  }
  section.lead h1 { font-size: 72px; letter-spacing: -1px; margin-bottom: 8px; }
  section.lead h2 { font-weight: 400; color: var(--gh-fg-muted); border: none; margin-top: 0; }

  section.chapter {
    background: linear-gradient(135deg, #f6f8fa 0%, #ffffff 100%);
    justify-content: center;
    text-align: left;
    padding-left: 120px;
  }
  section.chapter h1 {
    font-size: 56px;
    color: var(--gh-accent);
    border: none;
  }
  section.chapter h2 {
    font-weight: 400;
    color: var(--gh-fg-muted);
    border: none;
    margin-top: 8px;
  }
  section.chapter .ch-num {
    font-family: var(--gh-mono);
    color: var(--gh-fg-muted);
    font-size: 22px;
    letter-spacing: 2px;
  }

  h1, h2, h3 { color: var(--gh-fg); font-weight: 600; }
  h1 { font-size: 36px; }
  h2 {
    font-size: 28px;
    border-bottom: 1px solid var(--gh-border);
    padding-bottom: 6px;
    margin-top: 0;
  }
  h3 {
    font-size: 20px;
    color: var(--gh-accent);
    margin-top: 14px;
    margin-bottom: 4px;
  }

  a { color: var(--gh-accent); text-decoration: none; }

  code {
    font-family: var(--gh-mono);
    background: var(--gh-code-bg);
    padding: 1px 6px;
    border-radius: 4px;
    font-size: 0.92em;
  }
  pre {
    background: var(--gh-code-bg);
    border: 1px solid var(--gh-border);
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 15px;
    line-height: 1.35;
    overflow: auto;
  }
  pre code { background: transparent; padding: 0; }

  blockquote {
    border-left: 4px solid var(--gh-accent);
    color: var(--gh-fg-muted);
    background: var(--gh-code-bg);
    margin: 6px 0;
    padding: 6px 12px;
    border-radius: 0 6px 6px 0;
    font-size: 0.95em;
  }

  table {
    border-collapse: collapse;
    margin: 6px 0;
    font-size: 17px;
  }
  th, td { border: 1px solid var(--gh-border); padding: 4px 10px; }
  th { background: var(--gh-code-bg); font-weight: 600; text-align: left; }

  header, footer { color: var(--gh-fg-muted); font-size: 13px; }
  section::after { color: var(--gh-fg-muted); font-size: 13px; }

  .cols       { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
  .cols-60-40 { display: grid; grid-template-columns: 3fr 2fr; gap: 24px; }
  .cols-40-60 { display: grid; grid-template-columns: 2fr 3fr; gap: 24px; }
  .cols-3     { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; }

  .elem-grid {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 8px;
    align-items: end;
  }
  .elem-grid figure { margin: 0; text-align: center; }
  .elem-grid img { max-width: 100%; }
  .elem-grid figcaption {
    font-family: var(--gh-mono);
    font-size: 13px;
    color: var(--gh-fg-muted);
  }

  .callout {
    border: 1px solid var(--gh-border);
    border-left: 4px solid var(--gh-accent);
    background: var(--gh-code-bg);
    border-radius: 0 6px 6px 0;
    padding: 8px 12px;
    margin: 6px 0;
    font-size: 0.94em;
  }
  .callout-warn  { border-left-color: var(--gh-warning); }
  .callout-ok    { border-left-color: var(--gh-success); }
  .callout-bug   { border-left-color: var(--gh-danger); }

  .kbd {
    font-family: var(--gh-mono);
    background: var(--gh-code-bg);
    border: 1px solid var(--gh-border);
    border-bottom-width: 2px;
    border-radius: 4px;
    padding: 1px 6px;
    font-size: 0.9em;
  }

  .small { font-size: 17px; }
  .tiny  { font-size: 14px; color: var(--gh-fg-muted); }

  /* --- Overflow control -------------------------------------------------- */
  /* Per-slide density classes: append one of these to any slide that
     overflows at the default 21px base.
        _class: dense     - 18px base, tighter tables / lots of bullets
        _class: denser    - 16px base, very dense reference slides
        _class: tight     - 14px base, maximum density; use sparingly
  */
  section.dense          { font-size: 18px; padding: 38px 50px 46px 50px; }
  section.dense h2       { font-size: 25px; }
  section.dense h3       { font-size: 18px; }
  section.dense pre      { font-size: 13px; padding: 6px 10px; }
  section.dense table    { font-size: 15px; }
  section.dense td, section.dense th { padding: 3px 8px; }

  section.denser         { font-size: 16px; padding: 34px 46px 42px 46px; }
  section.denser h2      { font-size: 22px; padding-bottom: 4px; }
  section.denser h3      { font-size: 16px; }
  section.denser pre     { font-size: 12px; padding: 5px 8px; line-height: 1.3; }
  section.denser table   { font-size: 14px; }
  section.denser td, section.denser th { padding: 2px 6px; }
  section.denser .cols, section.denser .cols-60-40,
  section.denser .cols-40-60, section.denser .cols-3 { gap: 16px; }

  section.tight          { font-size: 14px; padding: 28px 40px 36px 40px; }
  section.tight h2       { font-size: 20px; padding-bottom: 3px; }
  section.tight h3       { font-size: 14px; margin-top: 8px; }
  section.tight pre      { font-size: 11px; padding: 4px 8px; line-height: 1.25; }
  section.tight table    { font-size: 13px; }
  section.tight td, section.tight th { padding: 2px 6px; }
  section.tight .cols, section.tight .cols-60-40,
  section.tight .cols-40-60, section.tight .cols-3 { gap: 12px; }
  section.tight ul, section.tight ol { margin: 4px 0; padding-left: 20px; }
  section.tight li       { margin: 1px 0; }
  section.tight p        { margin: 4px 0; }

  /* -----------------------------------------------------------------
     Mermaid-rendered SVG diagrams (filename pattern: res/mermaid_*.svg).
     mmdc emits SVGs with inline width="100%" and a viewBox whose height
     reflects the natural diagram size. On a 720 px slide this overflows
     vertically for flowcharts with many nodes. Cap the rendered height
     and let width scale proportionally, so Mermaid diagrams fit the
     slide without shrinking the slide font-size class.
  */
  img[src*="mermaid_"] {
    max-width: 100%;
    max-height: 460px;
    height: auto;
    width: auto;
    object-fit: contain;
    display: block;
    margin: 0 auto;
  }
  section.denser img[src*="mermaid_"] { max-height: 420px; }
  section.tight  img[src*="mermaid_"] { max-height: 380px; }

  /* Cap external images (GitHub raw URLs) to prevent overflow.
     Marp cannot introspect remote image sizes, so the overflow
     checker may miss tall images without this constraint. */
  img[src*="raw.githubusercontent.com"] {
    max-width: 100%;
    max-height: 480px;
    width: auto;
    height: auto;
    object-fit: contain;
  }
</style>
