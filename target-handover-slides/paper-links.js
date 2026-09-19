(() => {
  'use strict'

  // Bento's rich-text sanitizer removes anchors. Restore only the HTTPS
  // references authored in this deck, including its presentation copies.
  const doc = JSON.parse(document.getElementById('bento-doc').textContent)
  const references = doc.slides.flatMap(slide => slide.elements
    .filter(element => element.type === 'text' && element.html.includes('<a '))
    .map(element => ({ slide: slide.id, element: element.id, html: element.html })))

  function syncLinks() {
    for (const reference of references) {
      const selector = `.bento-slide[data-slide-id="${CSS.escape(reference.slide)}"] ` +
        `[data-el-id="${CSS.escape(reference.element)}"] .bento-text-inner`
      for (const inner of document.querySelectorAll(selector)) {
        if (inner.querySelector('a[href]')) continue
        const template = document.createElement('template')
        template.innerHTML = reference.html
        const content = document.createDocumentFragment()
        for (const node of template.content.childNodes) {
          const href = node.nodeType === Node.ELEMENT_NODE && node.tagName === 'A'
            ? node.getAttribute('href') : null
          if (href && new URL(href, location.href).protocol === 'https:') {
            const link = document.createElement('a')
            link.href = href
            link.target = '_blank'
            link.rel = 'noopener noreferrer'
            link.textContent = node.textContent
            link.addEventListener('click', event => event.stopPropagation())
            link.addEventListener('keydown', event => {
              if (event.key === 'Enter') event.stopPropagation()
            })
            content.append(link)
          } else {
            content.append(document.createTextNode(node.textContent))
          }
        }
        inner.replaceChildren(content)
      }
    }
  }

  new MutationObserver(syncLinks).observe(document.body, { childList: true, subtree: true })
  syncLinks()
})()
