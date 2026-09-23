(() => {
  'use strict';

  // Bento strips anchors from rich text. Restore the authored site-local
  // links in each presentation, overview, and print copy without changing it.
  const doc = JSON.parse(document.getElementById('bento-doc').textContent);
  const topics = doc.slides.flatMap(slide => slide.elements
    .filter(element => element.type === 'text' && /class="(?:scalability-topic|cover-reference)"/.test(element.html))
    .map(element => {
      const template = document.createElement('template');
      template.innerHTML = element.html;
      const source = template.content.querySelector('a.scalability-topic, a.cover-reference');
      return {
        slide: slide.id, element: element.id,
        href: source.getAttribute('href'), label: source.firstElementChild.textContent,
        className: source.className
      };
    })).filter(topic => new URL(topic.href, location.href).origin === location.origin);

  function syncLinks() {
    for (const topic of topics) {
      const selector = `.bento-slide[data-slide-id="${CSS.escape(topic.slide)}"] ` +
        `[data-el-id="${CSS.escape(topic.element)}"] .bento-text-inner`;
      for (const inner of document.querySelectorAll(selector)) {
        if (inner.querySelector('a[href]')) continue;
        const link = document.createElement('a');
        link.className = topic.className;
        link.href = topic.href;
        link.target = '_self';
        const label = document.createElement('span');
        label.textContent = topic.label;
        const arrow = document.createElement('span');
        arrow.className = topic.className === 'scalability-topic' ? 'topic-arrow' : 'reference-arrow';
        arrow.setAttribute('aria-hidden', 'true');
        arrow.textContent = '→';
        link.append(label, arrow);
        link.addEventListener('click', event => event.stopPropagation());
        link.addEventListener('keydown', event => {
          if (event.key === 'Enter') event.stopPropagation();
        });
        inner.replaceChildren(link);
      }
    }
  }

  new MutationObserver(syncLinks).observe(document.body, { childList: true, subtree: true });
  syncLinks();
})();
