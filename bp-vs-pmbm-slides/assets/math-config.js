window.MathJax = {
  tex: { inlineMath: [['\\(', '\\)']], displayMath: [['\\[', '\\]']], processEscapes: true,
    macros: { bm: ['\\boldsymbol{#1}', 1] } },
  svg: { fontCache: 'local' },
  options: { skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'] },
  startup: { typeset: false, ready() {
    MathJax.startup.defaultReady();
    MathJax.startup.promise.then(() => window.dispatchEvent(new Event('mathjax-ready')));
  } }
};
