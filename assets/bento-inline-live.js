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
    frame.dataset.src = entry.src;
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
      if (active && !item.frame.hasAttribute('src')) item.frame.src = item.frame.dataset.src;
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
      const lastIndex = Math.max(0, document.querySelectorAll('.slides > section').length - 1);
      const targetIndex = Math.min(lastIndex, Math.max(0, item.entry.slideIndex + direction));
      location.hash = '#/' + targetIndex;
    } else if (type !== 'bento-inline-focus') {
      return;
    }

    const reveal = document.querySelector('.reveal');
    if (reveal) {
      reveal.tabIndex = -1;
      reveal.focus();
    }
  });

  window.addEventListener('resize', schedule);
  window.addEventListener('pagehide', () => {
    mounted.forEach(item => postToFrame(item, 'bento-live-pause'));
  });
  schedule();
})();
