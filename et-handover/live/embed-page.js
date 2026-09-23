(() => {
  const params = new URLSearchParams(location.search);
  const selector = params.get('slide-embed');
  if (!selector) return;

  function mount() {
    let target;
    try { target = document.querySelector(selector); } catch (error) { console.error('Invalid slide-embed selector', selector, error); }
    if (!target) {
      document.body.innerHTML = `<main style="font:16px system-ui;padding:2rem"><h1>Demo not found</h1><p>The requested presentation target <code></code> is missing.</p></main>`;
      document.querySelector('code').textContent = selector;
      return;
    }

    document.documentElement.dataset.slideEmbed = 'true';
    document.body.dataset.slideEmbed = 'true';
    target.dataset.slideEmbedTarget = 'true';
    for (let node = target.parentElement; node && node !== document.body; node = node.parentElement) {
      node.dataset.slideEmbedAncestor = 'true';
    }

    const style = document.createElement('link');
    style.rel = 'stylesheet';
    style.href = './embed-page.css';
    document.head.appendChild(style);
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
