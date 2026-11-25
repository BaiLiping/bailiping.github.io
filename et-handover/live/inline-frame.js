(() => {
  if (window.parent === window) return;

  function post(type, detail = {}) {
    window.parent.postMessage({ type, ...detail }, '*');
  }

  function setPaused(paused) {
    document.documentElement.dataset.bentoPaused = String(paused);
    window.dispatchEvent(new CustomEvent('bento-live-visibility', {
      detail: { paused }
    }));
  }

  function isControl(target) {
    return Boolean(target?.closest?.(
      'input, textarea, select, button, a[href], [contenteditable="true"], ' +
      '[role="slider"], [role="spinbutton"], [role="textbox"], [role="tab"], [role="menuitem"]'
    ));
  }

  window.addEventListener('message', event => {
    if (event.source !== window.parent) return;
    if (event.data?.type === 'bento-live-pause') setPaused(true);
    if (event.data?.type === 'bento-live-resume') setPaused(false);
  });

  window.addEventListener('keydown', event => {
    if (event.key === 'Escape') {
      event.preventDefault();
      post('bento-inline-focus');
      return;
    }

    if (event.key === 'PageUp' || event.key === 'PageDown') {
      event.preventDefault();
      post('bento-inline-nav', { direction: event.key === 'PageUp' ? -1 : 1 });
      return;
    }

    if (isControl(event.target)) return;
    const backward = ['ArrowLeft', 'ArrowUp'].includes(event.key);
    const forward = ['ArrowRight', 'ArrowDown', ' '].includes(event.key);
    if (!backward && !forward) return;
    event.preventDefault();
    post('bento-inline-nav', { direction: backward ? -1 : 1 });
  });

  document.addEventListener('visibilitychange', () => setPaused(document.hidden));
  window.addEventListener('pagehide', () => setPaused(true));
  window.addEventListener('pageshow', () => setPaused(document.hidden));

  function announceReady() {
    const embedded = new URLSearchParams(location.search).has('slide-embed');
    const target = embedded
      ? document.querySelector('[data-slide-embed-target="true"]')
      : document.body;
    // A failed embed stylesheet must never expose the full source page.
    if (!target || (embedded && getComputedStyle(target).position !== 'fixed')) return;

    const observer = new MutationObserver(finish);
    function finish() {
      const math = [...target.querySelectorAll('.math-tex')];
      if (math.some(node => !node.querySelector('mjx-container'))) return;
      observer.disconnect();
      document.fonts.ready.then(() => requestAnimationFrame(() => post('bento-inline-ready')));
    }
    observer.observe(target, { childList: true, subtree: true });
    finish();
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', announceReady, { once: true });
  } else {
    announceReady();
  }
})();
