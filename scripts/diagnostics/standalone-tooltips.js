// The conversation host supplies data-tooltip. Offline exports use this local
// fallback; it is deliberately absent from the inline fragment.
(() => {
    const root = document.getElementById('ROOT');
    const tooltip = document.createElement('div');
    tooltip.id = 'ROOT-tooltip';
    tooltip.className = 'offline-tooltip';
    tooltip.setAttribute('role', 'tooltip');
    tooltip.setAttribute('popover', 'manual');
    document.body.append(tooltip);
    let anchor, pointer, frame = 0, suppressPointer = false;
    function place() {
        frame = 0;
        if (!anchor?.isConnected) { hide(); return; }
        const box = anchor.getBoundingClientRect(), rect = tooltip.getBoundingClientRect();
        const x = pointer ? pointer.x + 14 : box.left;
        const y = pointer ? pointer.y + 16 : box.bottom + 8;
        tooltip.style.left = `${Math.max(8, Math.min(x, window.innerWidth - rect.width - 8))}px`;
        tooltip.style.top = `${Math.max(8, y + rect.height + 8 < window.innerHeight ? y : (pointer ? pointer.y - rect.height - 12 : box.top - rect.height - 8))}px`;
    }
    function hide() {
        if (anchor) anchor.removeAttribute('aria-describedby');
        anchor = undefined;
        tooltip.hidePopover();
    }
    function show(element, event) {
        if (!element) return;
        if (event?.pointerType === 'touch') return;
        if (anchor && anchor !== element) anchor.removeAttribute('aria-describedby');
        anchor = element;
        pointer = event?.clientX !== undefined ? {x: event.clientX, y: event.clientY} : null;
        tooltip.textContent = element.dataset.tooltip;
        element.setAttribute('aria-describedby', tooltip.id);
        tooltip.showPopover();
        place();
    }
    root.addEventListener('pointerover', event => {
        if (suppressPointer) return;
        const element = event.target.closest('[data-tooltip]');
        if (element !== anchor) show(element, event);
    });
    root.addEventListener('pointermove', event => {
        if (suppressPointer) {
            suppressPointer = false;
            show(event.target.closest('[data-tooltip]'), event);
        }
        if (!anchor || !pointer) return;
        pointer = {x: event.clientX, y: event.clientY};
        if (!frame) frame = requestAnimationFrame(place);
    });
    root.addEventListener('pointerout', event => {
        if (anchor && !anchor.contains(event.relatedTarget)) hide();
    });
    root.addEventListener('focusin', event => show(event.target.closest('[data-tooltip]')));
    root.addEventListener('focusout', hide);
    root.addEventListener('click', () => { suppressPointer = true; hide(); });
    root.addEventListener('keydown', event => { if (event.key === 'Escape') hide(); });
    window.addEventListener('scroll', hide, {passive: true});
    // Navigation can detach the hovered node before a pointerout is delivered.
    new MutationObserver(() => { if (anchor && !anchor.isConnected) hide(); }).observe(root, {childList: true, subtree: true});
})();
