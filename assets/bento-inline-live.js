(() => {
  const configNode = document.getElementById('bento-inline-live-map') ||
    document.getElementById('companion-demo-map');
  if (!configNode) return;

  let demos;
  try {
    demos = JSON.parse(configNode.textContent);
  } catch (error) {
    console.error('Invalid Bento inline-live map', error);
    return;
  }

  if (!Array.isArray(demos)) return;

  const mounted = new Map();
  let queued = false;

  function postToFrame(item, type) {
    const opaqueSandbox = item.entry.sandbox && !item.entry.sandbox.includes('allow-same-origin');
    item.frame.contentWindow?.postMessage({ type }, opaqueSandbox ? '*' : location.origin);
  }

  function selectorFor(slideId) {
    const escaped = window.CSS?.escape ? CSS.escape(slideId) : slideId.replace(/["\\]/g, '\\$&');
    return `.bento-slide[data-slide-id="${escaped}"]`;
  }

  function findSlide(slideId) {
    const selector = selectorFor(slideId);
    return document.querySelector(`.bento-present-overlay ${selector}`) ||
      document.querySelector(selector);
  }

  function install(entry, root) {
    root.classList.add('companion-demo-slide', `companion-demo-${entry.layout || 'region'}`);

    const stage = document.createElement('div');
    stage.className = 'companion-demo-stage';
    if (entry.bounds) {
      stage.style.left = entry.bounds.x + 'px';
      stage.style.top = entry.bounds.y + 'px';
      stage.style.width = entry.bounds.width + 'px';
      stage.style.height = entry.bounds.height + 'px';
    }

    const frame = document.createElement('iframe');
    frame.className = 'companion-demo-frame';
    frame.title = entry.title;
    // Reveal reserves data-src for its own lazy-loading pipeline. Keep the
    // inline demo URL private so scroll mode cannot consume it.
    frame.dataset.bentoSrc = entry.src;
    frame.allow = 'fullscreen';
    frame.referrerPolicy = 'same-origin';
    frame.tabIndex = 0;
    if (entry.sandbox) frame.setAttribute('sandbox', entry.sandbox);
    frame.addEventListener('load', () => {
      if (!entry.readyMessage) frame.dataset.ready = 'true';
      const item = mounted.get(entry.slide);
      if (item) postToFrame(item, 'bento-live-resume');
    });
    stage.append(frame);

    const source = document.createElement('a');
    source.className = 'companion-demo-source';
    source.href = entry.source;
    source.target = '_blank';
    source.rel = 'noopener';
    source.textContent = 'Source ↗';

    root.append(stage);
    if (entry.hideSource === false) root.append(source);
    mounted.set(entry.slide, { root, frame, entry });
  }

  function sync() {
    queued = false;

    for (const entry of demos) {
      let item = mounted.get(entry.slide);
      if (!item || !item.root.isConnected) {
        const root = findSlide(entry.slide);
        if (!root) continue;
        install(entry, root);
        item = mounted.get(entry.slide);
      }

      const active = item.root.closest('section')?.classList.contains('present');
      if (active && !item.frame.hasAttribute('src')) item.frame.src = item.frame.dataset.bentoSrc;
      if (active && item.frame.hasAttribute('src')) postToFrame(item, 'bento-live-resume');
      if (!active && item.frame.hasAttribute('src')) {
        postToFrame(item, 'bento-live-pause');
        if (entry.unloadWhenHidden !== false) {
          item.frame.removeAttribute('src');
          delete item.frame.dataset.ready;
        }
      }
    }
  }

  function schedule() {
    if (queued) return;
    queued = true;
    queueMicrotask(sync);
  }

  function navigateFromFrame(item, direction) {
    const scrollRoot = document.querySelector(
      '.bento-present-overlay .reveal.reveal-scroll, .reveal.reveal-scroll'
    );
    if (scrollRoot) {
      const pages = [...scrollRoot.querySelectorAll('.scroll-page')];
      const currentPage = item.root.closest('.scroll-page');
      const currentIndex = pages.indexOf(currentPage);
      const targetIndex = Math.min(
        pages.length - 1,
        Math.max(0, currentIndex + direction)
      );
      const targetPage = pages[targetIndex];
      if (currentIndex >= 0 && targetPage && targetPage !== currentPage) {
        const rootRect = scrollRoot.getBoundingClientRect();
        const pageRect = targetPage.getBoundingClientRect();
        const top = scrollRoot.scrollTop + pageRect.top - rootRect.top;
        const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        scrollRoot.scrollTo({ top, behavior: reducedMotion ? 'auto' : 'smooth' });
        schedule();
        return;
      }
    }

    // Bento does not expose its Reveal instance. These controls perform true
    // relative navigation even when a deep-link hash has become stale.
    const control = document.querySelector(
      direction < 0 ? '.navigate-left.enabled' : '.navigate-right.enabled'
    );
    if (control) {
      control.click();
      return;
    }

    const sections = [...document.querySelectorAll('.slides > section')];
    const currentIndex = sections.indexOf(item.root.closest('section'));
    const fallbackIndex = currentIndex >= 0 ? currentIndex : item.entry.slideIndex;
    const targetIndex = Math.min(
      Math.max(0, sections.length - 1),
      Math.max(0, fallbackIndex + direction)
    );
    location.hash = '#/' + targetIndex;
  }

  new MutationObserver(schedule).observe(document.documentElement, {
    childList: true,
    subtree: true,
    attributes: true,
    attributeFilter: ['class']
  });

  window.addEventListener('message', event => {
    const item = [...mounted.values()].find(candidate => candidate.frame.contentWindow === event.source);
    const opaqueSandbox = item?.entry?.sandbox && !item.entry.sandbox.includes('allow-same-origin');
    const trustedOrigin = event.origin === location.origin || (opaqueSandbox && event.origin === 'null');
    if (!item || !trustedOrigin) return;

    const type = event.data?.type;
    if (type === 'bento-inline-ready') {
      item.frame.dataset.ready = 'true';
      postToFrame(item, 'bento-live-resume');
      return;
    }

    if (type === 'bento-inline-nav' && item.entry.inline && Number.isInteger(item.entry.slideIndex)) {
      const direction = event.data.direction < 0 ? -1 : 1;
      navigateFromFrame(item, direction);
    } else if (type !== 'bento-inline-focus') {
      return;
    }

    const reveal = document.querySelector('.reveal');
    if (reveal) {
      reveal.tabIndex = -1;
      reveal.focus({ preventScroll: true });
    }
  });

  window.addEventListener('resize', schedule);
  window.addEventListener('pagehide', () => {
    mounted.forEach(item => postToFrame(item, 'bento-live-pause'));
  });
  schedule();
})();
