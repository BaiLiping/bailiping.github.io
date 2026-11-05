// Use Bento's native print page structure for the browser's Print command.
// Read-only decks do not expose the editor's Export PDF action.
(() => {
  let owned = null;
  const doc = JSON.parse(document.getElementById('bento-doc').textContent);
  addEventListener('beforeprint', () => {
    if (document.getElementById('bento-print')) return;
    const root = document.createElement('div');
    root.id = 'bento-print';
    for (const slide of doc.slides.filter(s => !s.stateOf)) {
      const native = document.querySelector(`.reveal .slides .bento-slide[data-slide-id="${CSS.escape(slide.id)}"]`);
      if (!native) continue;
      const page = document.createElement('div');
      page.className = 'bp-page';
      const copy = native.cloneNode(true);
      copy.querySelectorAll('.companion-demo-stage, .companion-demo-source, iframe').forEach(n => n.remove());
      copy.classList.remove('companion-demo-slide', 'companion-demo-region');
      copy.style.transformOrigin = '0 0';
      copy.style.transform = 'scale(1.25)';
      copy.style.position = 'relative';
      copy.style.left = '0';
      copy.style.top = '0';
      copy.removeAttribute('aria-hidden');
      page.append(copy);
      root.append(page);
    }
    owned = root;
    document.body.append(root);
  });
  addEventListener('afterprint', () => { owned?.remove(); owned = null; });
})();
