// Restore authored companion links in all Bento presentation/overview/print copies.
// Internal site navigation stays in the current tab so Back returns to the deck.
let references;
function syncLinks() {
  if (!references) {
    const doc = JSON.parse(document.getElementById('bento-doc')?.textContent || '{}');
    if (!doc.slides) return;
    references = doc.slides.flatMap(slide => slide.elements
      .filter(element => element.type === 'text' && element.html.includes('class="deck-extension-link"'))
      .map(element => {
        const template = document.createElement('template');
        template.innerHTML = element.html;
        const source = template.content.querySelector('a');
        return { slide: slide.id, element: element.id, href: source.getAttribute('href'), label: source.textContent };
      })).filter(reference => reference.href.startsWith('/') && !reference.href.startsWith('//'));
  }
  for (const reference of references) {
    const selector = `.bento-slide[data-slide-id="${CSS.escape(reference.slide)}"] [data-el-id="${CSS.escape(reference.element)}"] .bento-text-inner`;
    for (const inner of document.querySelectorAll(selector)) {
      if (inner.querySelector('a.deck-extension-link')) continue;
      const link = document.createElement('a');
      link.className = 'deck-extension-link';
      link.href = reference.href;
      link.textContent = reference.label;
      link.style.cssText = 'color:inherit;text-decoration:underline;text-underline-offset:.18em;cursor:pointer;';
      link.addEventListener('click', event => event.stopPropagation());
      link.addEventListener('keydown', event => { if (event.key === 'Enter') event.stopPropagation(); });
      inner.replaceChildren(link);
    }
  }
}
new MutationObserver(syncLinks).observe(document.body, { childList: true, subtree: true });
syncLinks();
