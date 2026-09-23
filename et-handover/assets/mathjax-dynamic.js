(() => {
  const selector = '.math-tex:not([data-math-rendered]):not([data-math-pending])'
  let timer = 0
  let running = false
  let rerun = false

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

  function markRenderedMath() {
    document.querySelectorAll('.math-tex').forEach(node => {
      if (!node.querySelector('mjx-container')) return
      node.removeAttribute('data-math-pending')
      node.setAttribute('data-math-rendered', '')
    })
  }

  function readMathSource(node) {
    const source = node.textContent.trim()
    if (source.startsWith('\\(') && source.endsWith('\\)')) {
      return { source: source.slice(2, -2), display: false }
    }
    if (source.startsWith('\\[') && source.endsWith('\\]')) {
      return { source: source.slice(2, -2), display: true }
    }
    return null
  }

  async function typeset() {
    const mathJax = window.MathJax
    if (!mathJax || typeof mathJax.tex2svgPromise !== 'function') return

    restoreBentoMathClasses()
    markRenderedMath()
    const nodes = [...document.querySelectorAll(selector)].filter(node => node.isConnected)
    if (!nodes.length) return

    running = true
    for (const node of nodes) {
      if (!node.isConnected || node.querySelector('mjx-container')) continue
      const math = readMathSource(node)
      if (!math) continue
      node.setAttribute('data-math-pending', '')
      try {
        const rendered = await mathJax.tex2svgPromise(math.source, { display: math.display })
        const current = node.isConnected ? readMathSource(node) : null
        if (current && current.source === math.source && current.display === math.display) {
          node.replaceChildren(rendered)
          node.setAttribute('data-math-rendered', '')
        }
      } catch (error) {
        if (node.isConnected) console.error('MathJax typesetting failed', error)
      } finally {
        if (node.isConnected) node.removeAttribute('data-math-pending')
      }
    }
    running = false
    markRenderedMath()
    if (rerun) {
      rerun = false
      schedule()
    }
  }

  function schedule() {
    clearTimeout(timer)
    timer = setTimeout(() => {
      timer = 0
      if (running) {
        rerun = true
        return
      }
      typeset()
    }, 80)
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
