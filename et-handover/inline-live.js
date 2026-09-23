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

  function install(entry, root) {
    root.classList.add('companion-demo-slide');
    if (entry.layout === 'full') root.classList.add('companion-demo-full');
    if (entry.layout === 'region') root.classList.add('companion-demo-region');

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
    source.target = '_self';
    source.textContent = 'Source →';

    const additions = [stage];
    if (entry.hideSource === false) additions.push(source);

    if (['full', 'region'].includes(entry.layout) && Number.isInteger(entry.parentIndex)) {
      const back = document.createElement('button');
      back.className = 'companion-demo-back';
      back.type = 'button';
      back.setAttribute('aria-label', 'Back to the static Bento slide');
      back.title = 'Back to Bento slide (Esc)';
      back.textContent = '‹';
      back.addEventListener('click', () => {
        location.hash = '#/' + entry.parentIndex;
        const reveal = document.querySelector('.reveal');
        if (reveal) {
          reveal.tabIndex = -1;
          reveal.focus();
        }
      });
      additions.push(back);
    }

    root.append(...additions);
    mounted.set(entry.slide, { root, frame, entry });
  }

  function sync() {
    queued = false;

    for (const entry of demos) {
      let item = mounted.get(entry.slide);
      if (!item || !item.root.isConnected) {
        const root = document.querySelector(selectorFor(entry.slide));
        if (!root) continue;
        install(entry, root);
        item = mounted.get(entry.slide);
      }

      const active = item.root.closest('section')?.classList.contains('present');
      if (active && !item.frame.hasAttribute('src')) item.frame.src = item.frame.dataset.src;
      if (active && item.frame.hasAttribute('src')) {
        postToFrame(item, 'bento-live-resume');
      }
      if (!active && item.frame.hasAttribute('src')) {
        postToFrame(item, 'bento-live-pause');
        item.frame.removeAttribute('src');
        delete item.frame.dataset.ready;
      }
    }
  }

  function schedule() {
    if (queued) return;
    queued = true;
    queueMicrotask(sync);
  }

  const observer = new MutationObserver(schedule);
  observer.observe(document.documentElement, {
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
    if (type === 'bento-inline-ready' || type === 'private-slides-ready') {
      item.frame.dataset.ready = 'true';
      postToFrame(item, 'bento-live-resume');
      return;
    }
    if ((type === 'bento-inline-nav' || type === 'private-slides-nav') && item.entry.inline && Number.isInteger(item.entry.slideIndex)) {
      const direction = event.data.direction < 0 ? -1 : 1;
      const reveal = document.querySelector('.reveal');
      if (reveal) {
        reveal.tabIndex = -1;
        reveal.focus();
      }
      // Use Bento's own handler: its active slide can differ from the URL hash.
      document.dispatchEvent(new KeyboardEvent('keydown', {
        key: direction < 0 ? 'PageUp' : 'PageDown',
        code: direction < 0 ? 'PageUp' : 'PageDown',
        keyCode: direction < 0 ? 33 : 34,
        bubbles: true,
        cancelable: true
      }));
      return;
    }
    if (type !== 'bento-inline-focus' && type !== 'private-slides-escape') return;
    if (!item.entry.inline && Number.isInteger(item?.entry?.parentIndex)) {
      location.hash = '#/' + item.entry.parentIndex;
    }
    const reveal = document.querySelector('.reveal');
    if (reveal) {
      reveal.tabIndex = -1;
      reveal.focus();
    }
  });

  schedule();
})();
