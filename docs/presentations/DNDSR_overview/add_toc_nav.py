#!/usr/bin/env python3
"""Inject a sidebar outline/TOC into a Marp-generated HTML slide deck.

Extracts the first heading from each <section> slide, builds a
<nav class="slide-toc"> sidebar, and appends CSS + JS so clicking
a link scrolls directly to that slide. Chapter (h1) headings are
bold; ordinary (h2) slides are indented under the current chapter.

Usage:
    python3 add_toc_nav.py DNDSR_overview.html
    python3 add_toc_nav.py DNDSR_overview.html --open  # open sidebar by default
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

NAV_ID = "slide-toc"

INJECT_CSS = """\
<style id="slide-toc-css">
body {
    margin-left: 0;
    transition: margin-left 0.2s;
}
body.toc-open {
    margin-left: 280px;
}

#slide-toc {
    position: fixed;
    top: 0; left: 0;
    width: 260px; height: 100vh;
    overflow-y: auto;
    background: #f6f8fa;
    border-right: 1px solid #d0d7de;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-size: 13px;
    line-height: 1.4;
    z-index: 1000;
    padding: 16px 12px;
    box-sizing: border-box;
    transform: translateX(-280px);
    transition: transform 0.2s;
    user-select: none;
}
body.toc-open #slide-toc {
    transform: translateX(0);
}

#slide-toc .toc-title {
    font-weight: 700;
    font-size: 14px;
    margin: 0 0 12px 0;
    color: #24292f;
}

#slide-toc a {
    display: block;
    color: #0969da;
    text-decoration: none;
    padding: 2px 0 2px 4px;
    border-radius: 3px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
#slide-toc a:hover {
    background: #ddf4ff;
    color: #0550ae;
}
#slide-toc a.toc-chapter {
    font-weight: 600;
    color: #1f2328;
    padding: 4px 0 2px 0;
    margin-top: 4px;
}
#slide-toc a.toc-slide {
    padding-left: 16px;
    font-size: 12px;
    color: #656d76;
}

#slide-toc-toggle {
    position: fixed;
    top: 12px; left: 12px;
    z-index: 1001;
    width: 32px; height: 32px;
    border: 1px solid #d0d7de;
    border-radius: 6px;
    background: #ffffff;
    cursor: pointer;
    font-size: 16px;
    line-height: 30px;
    text-align: center;
    color: #656d76;
    user-select: none;
    transition: left 0.2s;
}
body.toc-open #slide-toc-toggle {
    left: 292px;
}
</style>
"""

INJECT_HTML = """\
<button id="slide-toc-toggle" title="Toggle slide outline (O)"\
 onclick="document.body.classList.toggle('toc-open')">☰</button>
<nav id="slide-toc">
  <div class="toc-title">Slide Outline</div>
  __TOC_LINKS__
</nav>
"""

INJECT_JS = """\
<script id="slide-toc-js">
(function() {
    var toggle = document.getElementById('slide-toc-toggle');
    var body = document.body;

    document.addEventListener('keydown', function(e) {
        if (e.key === 'o' && !e.ctrlKey && !e.metaKey && !e.altKey &&
            document.activeElement === document.body) {
            e.preventDefault();
            body.classList.toggle('toc-open');
        }
    });

    document.getElementById('slide-toc').addEventListener('click', function(e) {
        var a = e.target.closest('a');
        if (!a) return;
        e.preventDefault();
        var id = a.getAttribute('href').slice(1);
        // Marp bespoke listens for hashchange and parses #N → slide(N-1).
        window.location.hash = id;
        if (window.innerWidth < 800) body.classList.remove('toc-open');
    });

    var links = document.querySelectorAll('#slide-toc a');
    var observer = new IntersectionObserver(function(entries) {
        entries.forEach(function(entry) {
            if (entry.isIntersecting) {
                links.forEach(function(l) { l.style.background = ''; });
                var a = document.querySelector(
                    '#slide-toc a[href="#' + entry.target.id + '"]');
                if (a) a.style.background = '#ddf4ff';
            }
        });
    }, { rootMargin: '-20% 0px -70% 0px' });

    for (var i = 1; i <= __SLIDE_COUNT__; i++) {
        var s = document.getElementById('' + i);
        if (s) observer.observe(s);
    }
})();
</script>
"""

HEADING_RE = re.compile(r"<(h[12])\b[^>]*>(.*?)</\1>", re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")


def extract_headings(html: str) -> list[tuple[str, str, str]]:
    """Return [(section_id, tag, text)] for the first heading in each <section>."""
    sections = re.split(r"(<section\b[^>]*>)", html)
    result: list[tuple[str, str, str]] = []
    i = 0
    sid = 0
    while i < len(sections):
        if sections[i].startswith("<section"):
            sid += 1
            body = sections[i + 1] if i + 1 < len(sections) else ""
            m = HEADING_RE.search(body)
            if m:
                tag = m.group(1)
                text = TAG_RE.sub("", m.group(2)).strip()
                text = text.replace("&lt;", "<").replace("&gt;", ">")
                text = text.replace("&amp;", "&")
                result.append((str(sid), tag, text))
            i += 2
        else:
            i += 1
    return result


def build_toc_links(headings: list[tuple[str, str, str]]) -> str:
    """Build <a> tags, nesting h2 under the previous h1."""
    parts: list[str] = []
    for sid, tag, text in headings:
        cls = "toc-chapter" if tag == "h1" else "toc-slide"
        parts.append(
            f'<a class="{cls}" href="#{xml_escape(sid)}">'
            f"{xml_escape(text)}</a>"
        )
    return "\n  ".join(parts)


def process(html: str, open_default: bool) -> str:
    headings = extract_headings(html)
    links = build_toc_links(headings)

    css = INJECT_CSS
    nav_html = INJECT_HTML.replace("__TOC_LINKS__", links)
    js = INJECT_JS.replace("__SLIDE_COUNT__", str(len(headings)))

    # Inject CSS just before </head>
    html = html.replace("</head>", css + "\n</head>", 1)

    # Inject nav HTML just after <body>
    html = html.replace("<body>", "<body>" + nav_html, 1)
    if open_default:
        html = html.replace("<body>", '<body class="toc-open">', 1)

    # Inject JS just before </body>
    html = html.replace("</body>", js + "\n</body>", 1)

    return html


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Inject a sidebar slide outline/TOC into Marp HTML."
    )
    ap.add_argument("html", type=Path,
                    help="Marp-generated HTML file to modify")
    ap.add_argument("--open", action="store_true",
                    help="Open the sidebar by default")
    ap.add_argument("--stdout", action="store_true",
                    help="Write to stdout instead of modifying in place")
    args = ap.parse_args()

    original = args.html.read_text(encoding="utf-8")
    rewritten = process(original, args.open)

    if args.stdout:
        sys.stdout.write(rewritten)
    else:
        args.html.write_text(rewritten, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
