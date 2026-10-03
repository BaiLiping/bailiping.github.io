// Add stable source context to visual feedback in the ChatGPT desktop browser.
(() => {
  function mark() {
    for (const slide of document.querySelectorAll('.bento-slide[data-slide-id]')) {
      slide.setAttribute('oai-annotation-container', '');
      for (const element of slide.querySelectorAll('[data-el-id]')) {
        if (element.hasAttribute('oai-annotatable')) continue;
        const slideId = slide.dataset.slideId, elementId = element.dataset.elId;
        element.setAttribute('oai-annotatable', `${slideId} · ${elementId}`);
        element.setAttribute('oai-annotation-metadata', JSON.stringify({
          page: location.pathname.slice(0, 256), slide: slideId.slice(0, 256), object: elementId.slice(0, 256),
          savedEdits: (location.pathname.replace(/(?:index\.html)?$/, '') + 'authoring.json').slice(0, 256)
        }));
      }
    }
  }
  new MutationObserver(mark).observe(document.documentElement, {childList: true, subtree: true});
  mark();
})();
