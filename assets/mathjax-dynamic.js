(() => {
  const selector = '.math-tex:not([data-math-rendered]):not([data-math-pending])'
  let scheduled = false
  let queue = Promise.resolve()

  function restoreBentoMathClasses() {
    document.querySelectorAll('.bento-text-inner span:not(.math-tex)').forEach(node => {
      const source = node.textContent.trim()
      if (source.startsWith('\\(') && source.endsWith('\\)')) {
        node.classList.add('math-tex', 'math-inline')
      } else if (source.startsWith('\\[') && source.endsWith('\\]')) {
        node.classList.add('math-tex', 'math-display')
      }
    })
  }

  function typeset() {
    scheduled = false
    const mathJax = window.MathJax
    if (!mathJax || typeof mathJax.typesetPromise !== 'function') return

    restoreBentoMathClasses()
    const nodes = [...document.querySelectorAll(selector)]
    if (!nodes.length) return
    nodes.forEach(node => node.setAttribute('data-math-pending', ''))

    queue = queue
      .then(() => mathJax.typesetPromise(nodes))
      .then(() => {
        nodes.forEach(node => {
          node.removeAttribute('data-math-pending')
          node.setAttribute('data-math-rendered', '')
        })
      })
      .catch(error => {
        nodes.forEach(node => node.removeAttribute('data-math-pending'))
        console.error('MathJax typesetting failed', error)
      })
  }

  function schedule() {
    if (scheduled) return
    scheduled = true
    requestAnimationFrame(typeset)
  }

  const observer = new MutationObserver(mutations => {
    if (mutations.some(mutation => mutation.addedNodes.length)) schedule()
  })

  function start() {
    observer.observe(document.documentElement, { childList: true, subtree: true })
    schedule()
    window.MathJax?.startup?.promise?.then(schedule).catch(() => {})
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', start, { once: true })
  } else {
    start()
  }

  window.addEventListener('mathjax-ready', schedule)
  window.typesetDynamicMath = schedule
})()
