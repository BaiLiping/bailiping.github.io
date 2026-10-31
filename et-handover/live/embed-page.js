(() => {
  const params = new URLSearchParams(location.search);
  const selector = params.get('slide-embed');
  if (!selector) return;
  // Hide the full source page before its body can paint in an embedded view.
  document.documentElement.dataset.slideEmbed = 'true';

  function mount() {
    let target;
    try { target = document.querySelector(selector); } catch (error) { console.error('Invalid slide-embed selector', selector, error); }
    if (!target) {
      delete document.documentElement.dataset.slideEmbed;
      document.body.innerHTML = `<main style="font:16px system-ui;padding:2rem"><h1>Demo not found</h1><p>The requested presentation target <code></code> is missing.</p></main>`;
      document.querySelector('code').textContent = selector;
      return;
    }

    document.body.dataset.slideEmbed = 'true';
    target.dataset.slideEmbedTarget = 'true';
    for (let node = target.parentElement; node && node !== document.body; node = node.parentElement) {
      node.dataset.slideEmbedAncestor = 'true';
    }

    target.scrollTop = 0;
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', mount, {once: true});
  else mount();

  window.addEventListener('keydown', event => {
    if (event.key === 'Escape' && window.parent !== window) {
      window.parent.postMessage({type: 'private-slides-escape'}, location.origin);
    }
  });
})();
