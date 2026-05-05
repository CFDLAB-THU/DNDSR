#!/usr/bin/env node
/*
 * check_overflow.js — detect overflowing slides in a Marp-rendered HTML deck.
 *
 * Usage:
 *   node check_overflow.js <path-to-deck.html>
 *
 * Approach:
 *   1. Marp CLI renders the deck to HTML (done externally).
 *   2. Launch headless Chrome via puppeteer-core, using the system
 *      google-chrome binary at /usr/bin/google-chrome.
 *   3. For every <section> (= one slide), read:
 *        clientHeight (= the CSS "slide box" = 720 for 16:9 at 1280x720)
 *        scrollHeight (= the intrinsic content height, including overflow)
 *        clientWidth / scrollWidth for horizontal overflow
 *      We also list which `class` tokens (e.g. "dense", "denser") are set.
 *   4. Emit a JSON/text report of overflowing slides, sorted worst-first.
 *
 * A slide is considered "overflowing" when scrollHeight > clientHeight + tol
 * (default tol = 2 px) or scrollWidth > clientWidth + tol.
 */

const path = require('path');
const fs   = require('fs');
const puppeteer = require('puppeteer-core');

const CHROME_PATH = '/usr/bin/google-chrome';
const VIEWPORT = { width: 1280, height: 720, deviceScaleFactor: 1 };
const TOL_PX   = 2;

async function main() {
    const deckHtml = process.argv[2];
    if (!deckHtml) {
        console.error('usage: node check_overflow.js <deck.html>');
        process.exit(2);
    }
    const absPath = path.resolve(deckHtml);
    if (!fs.existsSync(absPath)) {
        console.error(`file not found: ${absPath}`);
        process.exit(2);
    }

    const browser = await puppeteer.launch({
        executablePath: CHROME_PATH,
        headless: 'new',
        args: [
            '--no-sandbox',
            '--disable-setuid-sandbox',
            '--allow-file-access-from-files',
            '--disable-web-security',
        ],
    });

    try {
        const page = await browser.newPage();
        await page.setViewport(VIEWPORT);
        const url = 'file://' + absPath;
        await page.goto(url, { waitUntil: 'networkidle0', timeout: 60000 });

        // Small wait so Mermaid / MathJax finish layout
        await new Promise(r => setTimeout(r, 1500));

        const results = await page.evaluate((tol) => {
            const sections = Array.from(document.querySelectorAll('section'));
            return sections.map((s, i) => {
                // Bounding box gives rendered size after transforms
                const rect  = s.getBoundingClientRect();
                const clientH = s.clientHeight;
                const clientW = s.clientWidth;
                const scrollH = s.scrollHeight;
                const scrollW = s.scrollWidth;
                const overflowY = Math.max(0, scrollH - clientH);
                const overflowX = Math.max(0, scrollW - clientW);

                // Measure actual content extent by scanning direct children,
                // ignoring Marp chrome (header, footer, pagination, etc.) which
                // is absolutely positioned near the slide edges and would
                // falsely consume all available vertical space.
                let contentTop = Infinity;
                let contentBottom = -Infinity;
                const sRect = s.getBoundingClientRect();
                const IGNORE_TAGS = new Set(['HEADER', 'FOOTER', 'SPAN']);
                for (const child of s.children) {
                    if (IGNORE_TAGS.has(child.tagName)) continue;
                    // Skip pagination / custom marpit UI nodes.
                    if (child.classList.contains('footnotes')) continue;
                    if (child.hasAttribute('data-marpit-pagination')) continue;
                    const cr = child.getBoundingClientRect();
                    if (cr.height === 0 && cr.width === 0) continue;
                    const top    = cr.top - sRect.top;
                    const bottom = cr.bottom - sRect.top;
                    if (top    < contentTop)    contentTop    = top;
                    if (bottom > contentBottom) contentBottom = bottom;
                }
                // When a slide has only UI chrome children (e.g. the title
                // slide uses an H1 inside a wrapper), fall back to scrollH.
                if (!isFinite(contentBottom)) contentBottom = scrollH;
                if (!isFinite(contentTop))    contentTop    = 0;
                const contentHeight = Math.max(0, contentBottom - contentTop);
                // Slack is how much vertical room is unused. Non-negative
                // when content fits; zero/negative once it overflows.
                const slackY = Math.max(0, clientH - Math.ceil(contentBottom));

                // Find a human-readable title (first h1/h2 text)
                const h = s.querySelector('h1, h2');
                const title = h ? h.textContent.trim().replace(/\s+/g, ' ') : '(no title)';

                // Collect class tokens (excluding 'lead'/'chapter' meta)
                const cls = s.className
                    .split(/\s+/)
                    .filter(c => c && c !== 'lead' && c !== 'chapter');

                const isLead    = s.classList.contains('lead');
                const isChapter = s.classList.contains('chapter');

                return {
                    index:   i + 1,
                    id:      s.id,
                    title:   title,
                    classes: cls,
                    kind:    isChapter ? 'chapter' : (isLead ? 'lead' : 'content'),
                    clientH, scrollH, overflowY, slackY, contentHeight,
                    clientW, scrollW, overflowX,
                    overflows: (overflowY > tol) || (overflowX > tol),
                };
            });
        }, TOL_PX);

        const overflowing = results.filter(r => r.overflows);
        overflowing.sort((a, b) => (b.overflowY + b.overflowX) - (a.overflowY + a.overflowX));

        console.log(`\nSlides inspected: ${results.length}`);
        console.log(`Overflowing:      ${overflowing.length}  (tolerance ${TOL_PX}px)`);
        console.log(``);

        if (overflowing.length > 0) {
            console.log(
                '  idx  kind    classes          overflow-y  overflow-x  title'
            );
            console.log(
                '  ---  ------  ---------------  ----------  ----------  -------------------------------------------------------'
            );
            for (const r of overflowing) {
                const cls = r.classes.join(',') || '(none)';
                console.log(
                    `  ${String(r.index).padStart(3)}  ${r.kind.padEnd(6)}  ${cls.padEnd(15)}  ${String(r.overflowY).padStart(10)}  ${String(r.overflowX).padStart(10)}  ${r.title.slice(0, 55)}`
                );
            }
        } else {
            console.log('  ✓ all slides fit');
        }

        // also dump JSON next to HTML for scripting
        const jsonPath = absPath.replace(/\.html$/, '.overflow.json');
        fs.writeFileSync(jsonPath, JSON.stringify({
            tolPx: TOL_PX,
            viewport: VIEWPORT,
            total: results.length,
            overflowing: overflowing.length,
            slides: results,
        }, null, 2));
        console.log(`\nJSON report: ${jsonPath}`);

        process.exitCode = overflowing.length > 0 ? 1 : 0;
    } finally {
        await browser.close();
    }
}

main().catch(err => {
    console.error(err);
    process.exit(3);
});
