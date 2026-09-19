(function () {
  'use strict';

  const configNode = document.getElementById('bento-live-config');
  if (!configNode) return;

  let config;
  try {
    config = JSON.parse(configNode.textContent);
  } catch (error) {
    console.error('Bento live-demo config is invalid.', error);
    return;
  }

  const demos = Array.isArray(config.demos) ? config.demos : [];
  if (!demos.length) return;

  const demoByState = new Map(demos.map(function (demo) {
    return [demo.state, demo];
  }));
  const frameOrigin = location.origin;
  const canLoadFrames = location.protocol === 'http:' || location.protocol === 'https:';

  let active = null;
  let pendingFocus = null;
  let scrollRoot = null;
  let syncQueued = false;

  function slideById(id) {
    return Array.from(document.querySelectorAll(
      '.bento-present-overlay .bento-slide[data-slide-id]'
    )).find(function (slide) {
      return slide.dataset.slideId === id;
    }) || null;
  }

  function desktopActiveSlide() {
    return document.querySelector(
      '.bento-present-overlay .slides > section.present > .bento-slide[data-slide-id]'
    );
  }

  function visibleRatio(element, root) {
    if (!element || !root) return 0;
    const rect = element.getBoundingClientRect();
    const rootRect = root.getBoundingClientRect();
    const width = Math.max(0, Math.min(rect.right, rootRect.right) - Math.max(rect.left, rootRect.left));
    const height = Math.max(0, Math.min(rect.bottom, rootRect.bottom) - Math.max(rect.top, rootRect.top));
    const area = Math.max(1, rect.width * rect.height);
    return width * height / area;
  }

  function mostVisibleSlide(candidates, root) {
    let best = null;
    let bestRatio = 0;
    candidates.forEach(function (slide) {
      const ratio = visibleRatio(slide, root);
      if (ratio > bestRatio) {
        best = slide;
        bestRatio = ratio;
      }
    });
    return bestRatio >= 0.35 ? best : null;
  }

  function currentSlide() {
    if (scrollRoot) {
      return mostVisibleSlide(Array.from(scrollRoot.querySelectorAll(
        '.bento-slide[data-slide-id]'
      )), scrollRoot);
    }
    return desktopActiveSlide();
  }

  function currentDemoSlide() {
    if (!scrollRoot) return desktopActiveSlide();
    const stateSlides = demos.map(function (demo) {
      return slideById(demo.state);
    }).filter(Boolean);
    return mostVisibleSlide(stateSlides, scrollRoot);
  }

  function parentIdForState(state) {
    return slideById(state)
      ?.querySelector('[data-el-id="live-back-hit"][data-link]')
      ?.dataset.link || null;
  }

  function launchControlForState(state) {
    const parent = slideById(parentIdForState(state));
    if (!parent) return null;
    return Array.from(parent.querySelectorAll('[data-link]')).find(function (element) {
      return element.dataset.link === state && element.dataset.elId?.endsWith('-hit');
    }) || null;
  }

  function postToFrame(frame, message) {
    try {
      frame.contentWindow?.postMessage(message, frameOrigin);
    } catch (error) {
      console.warn('Could not message the live demo.', error);
    }
  }

  function maybeRestoreFocus() {
    if (!pendingFocus || !pendingFocus.isConnected) {
      pendingFocus = null;
      return;
    }
    const parentSlide = pendingFocus.closest('.bento-slide[data-slide-id]');
    const ready = scrollRoot
      ? visibleRatio(parentSlide, scrollRoot) >= 0.35
      : desktopActiveSlide() === parentSlide;
    if (!ready) return;
    const control = pendingFocus;
    pendingFocus = null;
    requestAnimationFrame(function () {
      control.focus({ preventScroll: true });
    });
  }

  function destroyFrame() {
    if (!active) {
      maybeRestoreFocus();
      return;
    }
    const frame = active.frame;
    window.clearTimeout(active.loadTimer);
    postToFrame(frame, { type: 'bento-live-pause' });
    frame.src = 'about:blank';
    frame.remove();
    active = null;
    maybeRestoreFocus();
  }

  function failFrame(frame) {
    if (!active || active.frame !== frame) return;
    window.clearTimeout(active.loadTimer);
    frame.src = 'about:blank';
    frame.remove();
    active = null;
  }

  function backToParent() {
    if (!active) return;
    pendingFocus = active.trigger || launchControlForState(active.demo.state);
    const back = active.slide.querySelector('[data-el-id="live-back-hit"]');
    if (back) back.click();
  }

  function mountFrame(slide, demo) {
    const mount = slide.querySelector('[data-el-id="live-demo-mount"]');
    if (!mount) return;

    const frame = document.createElement('iframe');
    frame.dataset.bentoLive = demo.id || demo.state;
    frame.title = demo.title || 'Interactive example';
    frame.loading = 'eager';
    frame.referrerPolicy = 'same-origin';
    frame.tabIndex = 0;
    frame.setAttribute('sandbox', 'allow-scripts allow-same-origin');
    frame.allow = 'fullscreen';
    frame.style.left = mount.style.left;
    frame.style.top = mount.style.top;
    frame.style.width = mount.style.width;
    frame.style.height = mount.style.height;

    frame.addEventListener('error', function () {
      failFrame(frame);
    });
    frame.addEventListener('load', function () {
      if (!active || active.frame !== frame || frame.src === 'about:blank') return;
      window.clearTimeout(active.loadTimer);
      try {
        if (frame.contentWindow.location.href === 'about:blank') return;
        if (!frame.contentDocument?.body?.hasAttribute('data-bento-live-app')) {
          failFrame(frame);
          return;
        }
      } catch (error) {
        failFrame(frame);
        return;
      }
      postToFrame(frame, { type: 'bento-live-resume' });
      try {
        frame.contentWindow.addEventListener('keydown', function (event) {
          if (event.key !== 'Escape') return;
          event.preventDefault();
          event.stopImmediatePropagation();
          backToParent();
        }, true);
      } catch (error) {
        console.warn('Could not install the live-demo Escape shortcut.', error);
      }
      requestAnimationFrame(function () {
        if (active?.frame === frame) frame.focus({ preventScroll: true });
      });
    });

    slide.appendChild(frame);
    active = {
      demo,
      frame,
      slide,
      trigger: launchControlForState(demo.state),
      loadTimer: window.setTimeout(function () {
        failFrame(frame);
      }, 10000)
    };
    frame.src = demo.src;
  }

  function updateControlTabStops() {
    const visible = currentSlide();
    document.querySelectorAll('[data-bento-live-control]').forEach(function (control) {
      control.tabIndex = control.closest('.bento-slide[data-slide-id]') === visible ? 0 : -1;
    });
  }

  function sync() {
    syncQueued = false;
    updateControlTabStops();

    const slide = currentDemoSlide();
    const demo = demoByState.get(slide?.dataset.slideId);
    if (!demo) {
      destroyFrame();
      return;
    }

    if (active && active.demo.state === demo.state && active.frame.isConnected) return;
    destroyFrame();
    if (!canLoadFrames) return;
    mountFrame(slide, demo);
  }

  function queueSync() {
    if (syncQueued) return;
    syncQueued = true;
    requestAnimationFrame(sync);
  }

  function scrollToSlide(id) {
    if (!scrollRoot) return false;
    const slide = slideById(id);
    if (!slide) return false;
    const page = slide.closest('.scroll-page') || slide;
    const rootRect = scrollRoot.getBoundingClientRect();
    const pageRect = page.getBoundingClientRect();
    const top = scrollRoot.scrollTop + pageRect.top - rootRect.top;
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    scrollRoot.scrollTo({ top, behavior: reducedMotion ? 'auto' : 'smooth' });
    queueSync();
    return true;
  }

  function enhanceControls() {
    demos.forEach(function (demo) {
      const stateSlide = slideById(demo.state);
      const parentId = parentIdForState(demo.state);
      const parentSlide = slideById(parentId);
      if (!stateSlide || !parentSlide) return;

      const launchLinks = Array.from(parentSlide.querySelectorAll('[data-link]')).filter(function (element) {
        return element.dataset.link === demo.state;
      });
      const launch = launchLinks.find(function (element) {
        return element.dataset.elId?.endsWith('-hit');
      });
      if (launch) {
        launch.dataset.bentoLiveControl = 'launch';
        launch.dataset.bentoLiveState = demo.state;
        launch.setAttribute('role', 'button');
        launch.setAttribute('aria-label', 'Open ' + (demo.title || 'interactive example'));
      }
      launchLinks.filter(function (element) { return element !== launch; }).forEach(function (label) {
        label.setAttribute('aria-hidden', 'true');
        label.style.pointerEvents = 'none';
      });

      const backLinks = Array.from(stateSlide.querySelectorAll('[data-link]')).filter(function (element) {
        return element.dataset.link === parentId;
      });
      const back = backLinks.find(function (element) {
        return element.dataset.elId === 'live-back-hit';
      });
      if (back) {
        back.dataset.bentoLiveControl = 'back';
        back.dataset.bentoLiveState = demo.state;
        back.setAttribute('role', 'button');
        back.setAttribute('aria-label', 'Back to presentation');
      }
      backLinks.filter(function (element) { return element !== back; }).forEach(function (label) {
        label.setAttribute('aria-hidden', 'true');
        label.style.pointerEvents = 'none';
      });
    });
  }

  function connectScrollRoot() {
    const next = document.querySelector(
      '.bento-present-overlay .reveal.reveal-scroll'
    );
    if (next === scrollRoot) return;
    if (scrollRoot) scrollRoot.removeEventListener('scroll', queueSync);
    scrollRoot = next;
    if (scrollRoot) scrollRoot.addEventListener('scroll', queueSync, { passive: true });
  }

  function refresh() {
    connectScrollRoot();
    enhanceControls();
    queueSync();
  }

  document.addEventListener('click', function (event) {
    const control = event.target.closest?.('[data-bento-live-control]');
    if (!control) return;
    const state = control.dataset.bentoLiveState;

    if (control.dataset.bentoLiveControl === 'launch') {
      const demo = demoByState.get(state);
      if (!demo) return;
      if (active?.demo.state === state) active.trigger = control;
      if (scrollRoot) {
        event.preventDefault();
        event.stopImmediatePropagation();
        scrollToSlide(state);
      }
      return;
    }

    if (control.dataset.bentoLiveControl === 'back') {
      pendingFocus = launchControlForState(state);
      if (scrollRoot) {
        event.preventDefault();
        event.stopImmediatePropagation();
        scrollToSlide(parentIdForState(state));
      }
    }
  }, true);

  document.addEventListener('keydown', function (event) {
    const control = event.target.closest?.('[data-bento-live-control]');
    if (control && (event.key === 'Enter' || event.key === ' ')) {
      event.preventDefault();
      event.stopImmediatePropagation();
      control.click();
      return;
    }

    if (event.key === 'Escape' && active) {
      event.preventDefault();
      event.stopImmediatePropagation();
      backToParent();
    } else if (event.key === 'ArrowLeft' && active && !scrollRoot) {
      pendingFocus = active.trigger || launchControlForState(active.demo.state);
    }
  }, true);

  window.addEventListener('message', function (event) {
    if (!active || event.source !== active.frame.contentWindow || event.origin !== frameOrigin) return;
    if (event.data && event.data.type === 'bento-live-back') backToParent();
  });

  document.addEventListener('visibilitychange', function () {
    if (!active) return;
    postToFrame(active.frame, {
      type: document.hidden ? 'bento-live-pause' : 'bento-live-resume'
    });
  });

  new MutationObserver(refresh).observe(document.body, {
    childList: true,
    subtree: true,
    attributes: true,
    attributeFilter: ['class', 'hidden']
  });

  window.addEventListener('resize', refresh);
  window.addEventListener('pagehide', destroyFrame);
  refresh();
})();
