(() => {
    'use strict';
    const root = document.getElementById('ROOT');
    const data = JSON.parse(root.querySelector('.payload').textContent);
    const q = selector => root.querySelector(selector);
    const formatter = new Intl.NumberFormat(undefined, {maximumFractionDigits: 2});
    const integers = new Intl.NumberFormat();
    const format = bytes => {
        const unit = bytes >= 2 ** 30 ? 'GiB' : bytes >= 2 ** 20 ? 'MiB' : bytes >= 1024 ? 'KiB' : 'B';
        return `${formatter.format(bytes / {GiB: 2 ** 30, MiB: 2 ** 20, KiB: 1024, B: 1}[unit])} ${unit}`;
    };
    const exact = bytes => `${integers.format(bytes)} bytes`;
    const create = (tag, value, cls) => {
        const element = document.createElement(tag);
        if (value !== undefined) element.textContent = value;
        if (cls) element.className = cls;
        return element;
    };
    let selected = 0, path = [], currentNode, currentChildren = [], marks = [];
    let renderFrame = 0, layoutFrame = 0, persistTimer = 0, keyboardFocus = false;
    let lastSaved = '', highlighted;
    const trees = new Map(), savedStates = new Set();
    const route = record => record.names.length ? record.names[0].split('/').filter(Boolean) :
        record.site ? ['Unnamed (site recorded)', record.site.split(' ')[0].split('/').pop()] : ['Unnamed (owner unknown)'];

    function tree() {
        const key = `${selected}:${q('.device').value}`;
        if (trees.has(key)) return trees.get(key);
        const top = {name: 'All allocations', bytes: 0, children: new Map(), records: []};
        for (const record of data.snapshots[selected].allocations) {
            if (q('.device').value !== 'all' && record.device !== q('.device').value) continue;
            let node = top;
            for (const part of [null, ...route(record), `Allocation ${record.identity}`]) {
                if (part !== null) {
                    if (!node.children.has(part)) node.children.set(part, {name: part, bytes: 0, children: new Map(), records: []});
                    node = node.children.get(part);
                }
                node.bytes += record.bytes;
                node.records.push(record);
            }
        }
        trees.set(key, top);
        return top;
    }
    const snapshotState = () => ({setup: data.snapshots[selected].label, device: q('.device').value, path: path.slice()});
    function persist() {
        const api = window.openai;
        if (!api || typeof api.setWidgetState !== 'function') return;
        clearTimeout(persistTimer);
        persistTimer = setTimeout(() => {
            const state = snapshotState(), signature = JSON.stringify(state);
            if (signature === lastSaved || new TextEncoder().encode(signature).byteLength > 12000) return;
            lastSaved = signature;
            savedStates.add(signature);
            if (savedStates.size > 24) savedStates.delete(savedStates.values().next().value);
            try { Promise.resolve(api.setWidgetState({modelContent: state, privateContent: {format: 'dndsr-allocation-explorer-v1'}})).catch(() => {}); }
            catch (_) { /* Optional host persistence does not block navigation. */ }
        }, 120);
    }
    function restore(state) {
        const saved = state?.modelContent ?? state;
        if (!saved || !Array.isArray(saved.path)) return;
        const signature = JSON.stringify({setup: saved.setup, device: saved.device, path: saved.path});
        if (signature === JSON.stringify(snapshotState()) || savedStates.has(signature)) return;
        const index = data.snapshots.findIndex(snapshot => snapshot.label === saved.setup);
        if (index < 0) return;
        selected = index;
        path = saved.path.filter(part => typeof part === 'string').slice(0, 64);
        q('.setup').value = String(selected);
        devices(saved.device);
        requestRender();
    }
    function devices(preferred = q('.device').value) {
        const select = q('.device');
        select.replaceChildren(new Option('Host + device', 'all'));
        for (const name of [...new Set(data.snapshots[selected].allocations.map(record => record.device))].sort())
            select.append(new Option(name.startsWith('cuda:') ? `CUDA device ${name.split(':')[1]}` : 'Host memory', name));
        select.value = [...select.options].some(option => option.value === preferred) ? preferred : 'all';
    }
    function inspect(node, nodePath = path) {
        if (!node) return;
        q('.inspect-path').textContent = nodePath.length ? nodePath.join(' / ') : 'All allocations';
        const share = currentNode?.bytes ? node.bytes / currentNode.bytes * 100 : 0;
        q('.inspect-values').textContent = `${format(node.bytes)} · ${exact(node.bytes)} · ${formatter.format(share)}% of this level · ${node.records.length} allocations`;
    }
    function activate(node, nodePath, element) {
        if (highlighted) highlighted.removeAttribute('data-active');
        highlighted = element;
        element?.setAttribute('data-active', '');
        inspect(node, nodePath);
    }
    function navigate(nextPath, keyboard = false) {
        path = nextPath.slice();
        keyboardFocus = keyboard;
        requestRender();
        persist();
    }
    function requestRender() {
        if (!renderFrame) renderFrame = requestAnimationFrame(() => { renderFrame = 0; render(); });
    }
    function rectangles(items, x, y, width, height, output = []) {
        if (!items.length) return output;
        if (items.length === 1) { output.push({item: items[0], x, y, width, height}); return output; }
        const total = items.reduce((sum, item) => sum + item.bytes, 0);
        let split = 1, first = items[0].bytes;
        while (split < items.length - 1 && Math.abs(first + items[split].bytes - total / 2) < Math.abs(first - total / 2)) first += items[split++].bytes;
        const ratio = total ? first / total : split / items.length;
        if (width >= height) {
            rectangles(items.slice(0, split), x, y, width * ratio, height, output);
            rectangles(items.slice(split), x + width * ratio, y, width * (1 - ratio), height, output);
        } else {
            rectangles(items.slice(0, split), x, y, width, height * ratio, output);
            rectangles(items.slice(split), x, y + height * ratio, width, height * (1 - ratio), output);
        }
        return output;
    }
    function layout() {
        layoutFrame = 0;
        const map = q('.map');
        const boxes = rectangles(currentChildren, 0, 0, map.clientWidth, map.clientHeight);
        boxes.forEach((box, index) => {
            const mark = marks[index];
            if (!mark) return;
            const gap = Math.min(1.25, box.width / 8, box.height / 8);
            Object.assign(mark.style, {left: `${box.x + gap}px`, top: `${box.y + gap}px`, width: `${Math.max(0, box.width - 2 * gap)}px`, height: `${Math.max(0, box.height - 2 * gap)}px`});
            mark.classList.toggle('wide', box.width > 170);
            mark.classList.toggle('tiny', box.width < 44 || box.height < 25);
            mark.querySelector('.tile-name').hidden = box.width < 44 || box.height < 25;
            mark.querySelector('.tile-value').hidden = box.width < 95 || box.height < 68;
            mark.querySelector('.tile-ratio').hidden = box.width < 130 || box.height < 100;
        });
    }
    function renderDetails() {
        const records = q('.records');
        records.replaceChildren();
        if (!q('.details').open) return;
        if (currentChildren.length) {
            const base = path.slice();
            for (const node of currentChildren) {
                const button = create('button', `${node.name} · ${format(node.bytes)}`, 'btn btn-ghost');
                button.type = 'button';
                button.dataset.tooltip = `${base.concat(node.name).join(' / ')}: ${exact(node.bytes)}`;
                button.onpointerenter = () => activate(node, base.concat(node.name), marks[currentChildren.indexOf(node)]);
                button.onfocus = button.onpointerenter;
                button.onclick = event => navigate(base.concat(node.name), event.detail === 0);
                records.append(button);
            }
            return;
        }
        for (const record of currentNode.records) {
            const panel = create('div', undefined, 'leaf-record');
            const terms = create('dl');
            for (const [term, value] of Object.entries({Identity: record.identity, Size: exact(record.bytes), Placement: record.device,
                Provider: record.provider, 'Pending reads / writes': `${record.pending_reads ?? '—'} / ${record.pending_writes ?? '—'}`,
                'Allocation site': record.site || 'Not recorded'})) {
                terms.append(create('dt', term), create('dd', String(value)));
            }
            panel.append(terms);
            const aliases = create('ul');
            for (const name of record.names) aliases.append(create('li', name));
            if (aliases.children.length) panel.append(create('span', 'Names and shared aliases'), aliases);
            records.append(panel);
        }
    }
    function render() {
        let node = tree();
        const valid = [];
        for (const part of path) { if (!node.children.has(part)) break; node = node.children.get(part); valid.push(part); }
        path = valid;
        currentNode = node;
        currentChildren = [...node.children.values()].filter(child => child.bytes > 0).sort((a, b) => b.bytes - a.bytes || a.name.localeCompare(b.name));
        highlighted = undefined;
        root.classList.toggle('is-leaf', !currentChildren.length);
        const crumbs = q('.crumbs');
        crumbs.replaceChildren();
        const up = create('button', 'Up', 'btn btn-ghost');
        up.type = 'button'; up.disabled = path.length === 0;
        up.onclick = event => navigate(path.slice(0, -1), event.detail === 0);
        crumbs.append(up);
        for (let index = -1; index < path.length; ++index) {
            if (index >= 0) crumbs.append(create('span', '/', 'crumb-separator'));
            const button = create('button', index < 0 ? 'All allocations' : path[index], 'btn btn-ghost');
            button.type = 'button';
            if (index === path.length - 1) button.setAttribute('aria-current', 'location');
            button.dataset.tooltip = index < 0 ? 'All registered allocations' : path.slice(0, index + 1).join(' / ');
            const next = path.slice(0, index + 1);
            button.onclick = event => navigate(next, event.detail === 0);
            crumbs.append(button);
        }
        q('.total').textContent = format(node.bytes);
        q('.count').textContent = `${node.records.length} unique allocations · ${exact(node.bytes)}`;
        inspect(node);
        const base = path.slice(), map = q('.map');
        marks = currentChildren.map(child => {
            const mark = create('button', undefined, 'tile cursor-interaction');
            mark.type = 'button';
            const nodePath = base.concat(child.name), category = nodePath[0];
            const series = category === 'Cauchy' ? 1 : category === 'CFV' ? 2 : category === 'Geom' ? 3 : category.startsWith('Unnamed') ? 4 : 5;
            mark.style.setProperty('--tile-color', `var(--viz-series-${series})`);
            mark.style.setProperty('--tile-fill', `color-mix(in srgb,var(--viz-series-${series}) 24%,var(--background))`);
            const description = `${nodePath.join(' / ')}\n${format(child.bytes)} · ${exact(child.bytes)}\n${formatter.format(child.bytes / (node.bytes || 1) * 100)}% of this level · ${child.records.length} allocations`;
            mark.dataset.tooltip = description;
            mark.setAttribute('aria-label', description + (child.children.size ? ' — open children' : ' — inspect allocation'));
            mark.append(create('span', child.name, 'tile-name'), create('span', format(child.bytes), 'tile-value tabular-nums'),
                create('span', `${formatter.format(child.bytes / (node.bytes || 1) * 100)}%`, 'tile-ratio tabular-nums'));
            mark.onpointerenter = () => activate(child, nodePath, mark);
            mark.onfocus = mark.onpointerenter;
            mark.onclick = event => navigate(nodePath, event.detail === 0);
            return mark;
        });
        map.replaceChildren(...marks);
        layout();
        q('.detail-title').textContent = currentChildren.length ? `Browse ${currentChildren.length} children by name` : 'Allocation details';
        if (!currentChildren.length) q('.details').open = true;
        renderDetails();
        const snapshot = data.snapshots[selected];
        q('.source').textContent = snapshot.source || 'Source not supplied';
        q('.registry').textContent = `PID ${snapshot.pid ?? '—'} · registered live ${format(snapshot.reported_live_bytes)} · peak ${format(snapshot.reported_peak_bytes)} · aliases counted once`;
        q('.warning').textContent = (data.synthetic ? 'Synthetic UI fixture — not a solver measurement. ' : '') +
            (snapshot.summary_difference_bytes ? `Registry total differs by ${exact(snapshot.summary_difference_bytes)}; inspect source data.` : '');
        q('.announcement').textContent = `${path.join(' / ') || 'All allocations'}: ${format(node.bytes)}, ${node.records.length} allocations`;
        if (keyboardFocus) { (marks[0] || crumbs.lastElementChild)?.focus({preventScroll: true}); keyboardFocus = false; }
    }
    q('.details').ontoggle = renderDetails;
    q('.map').onpointerleave = () => { if (highlighted) highlighted.removeAttribute('data-active'); highlighted = undefined; inspect(currentNode); };
    q('.setup').replaceChildren(...data.snapshots.map((snapshot, index) => new Option(snapshot.label, index)));
    q('.setup').onchange = event => { selected = Number(event.target.value); devices(); requestRender(); persist(); };
    q('.device').onchange = () => { requestRender(); persist(); };
    root.addEventListener('keydown', event => {
        if ((event.key === 'Escape' || event.key === 'ArrowLeft') && path.length && !/INPUT|SELECT|TEXTAREA/.test(event.target.tagName)) {
            event.preventDefault(); navigate(path.slice(0, -1), true);
        }
    });
    if (typeof ResizeObserver !== 'undefined') new ResizeObserver(() => {
        if (!layoutFrame) layoutFrame = requestAnimationFrame(layout);
    }).observe(q('.map'));
    else window.addEventListener('resize', () => { if (!layoutFrame) layoutFrame = requestAnimationFrame(layout); });
    devices(data.initialDevice); render(); restore(window.openai?.widgetState);
    window.addEventListener('openai:set_globals', event => restore(event.detail?.globals?.widgetState ?? window.openai?.widgetState));
})();
