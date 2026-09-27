(() => {
  // Bento sanitizes rich-text anchors. Restore only this deck's authored links
  // in presentation, overview, and print copies without changing its runtime.
  const doc = JSON.parse(document.getElementById('bento-doc').textContent);
  const entries = doc.slides.flatMap(slide => slide.elements.filter(e => e.type === 'text' && e.html.includes('<a ')).map(element => {
    const template = document.createElement('template');
    template.innerHTML = element.html;
    const anchors = [...template.content.querySelectorAll('a')];
    for (const anchor of anchors) {
      const url = new URL(anchor.getAttribute('href'), location.href);
      if (!['http:', 'https:'].includes(url.protocol)) anchor.removeAttribute('href');
      anchor.target = '_self';
    }
    return { slide: slide.id, id: element.id, template, count: anchors.length };
  }));
  let pending = false;
  function sync() {
    pending = false;
    for (const entry of entries) {
      const selector = `.bento-slide[data-slide-id="${CSS.escape(entry.slide)}"] [data-el-id="${CSS.escape(entry.id)}"] .bento-text-inner`;
      for (const inner of document.querySelectorAll(selector)) {
        if (inner.dataset.rfsLinks === 'true' && inner.querySelectorAll('a').length === entry.count) continue;
        inner.replaceChildren(entry.template.content.cloneNode(true));
        inner.dataset.rfsLinks = 'true';
        for (const link of inner.querySelectorAll('a')) {
          link.addEventListener('click', e => e.stopPropagation());
          link.addEventListener('keydown', e => { if (e.key === 'Enter') e.stopPropagation(); });
        }
      }
    }
  }
  new MutationObserver(() => { if (!pending) { pending = true; queueMicrotask(sync); } }).observe(document.body, { childList: true, subtree: true });
  sync();
})();
