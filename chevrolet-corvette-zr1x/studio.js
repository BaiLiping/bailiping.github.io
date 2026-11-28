const room = document.getElementById('photo-room');
const photoFrame = room.querySelector('.photo-frame');
const photo = document.getElementById('photo');
const error = document.getElementById('photo-error');
const angles = document.getElementById('photo-angle');
const creditEl = document.getElementById('photo-credit');
const more = document.getElementById('more');
const nav = document.querySelector('.mode-switch');
const matchButton = document.getElementById('match-view');
const compareButton = document.getElementById('compare-view');
const comparison = document.getElementById('comparison-controls');
const split = document.getElementById('comparison-split');

const DEFAULT_CONFIGURATOR = 'https://www.chevrolet.com/performance/corvette/zr1x';
const GROUP_ORDER = ['Exterior', 'Powertrain', 'Interior'];

let sourcePromise;
export let frames = [];
const inspectionFrames = {
  '#powertrain': 'mech-hybrid-plan',
  '#rear-powertrain': 'mech-dct-rear34',
  '#front-electric-drive': 'mech-edu-front-render',
  '#hybrid-battery': 'mech-batt-cutaway-hi',
};

function requestedIndex() {
  const query = new URLSearchParams(location.search);
  let value = inspectionFrames[location.hash] || query.get('frame') || query.get('view');
  if (!value) return -1;
  if (/^ext-\d{3}$/.test(value)) value = 'view-' + String(Number(value.slice(4)) - 1).padStart(2, '0');
  return [...angles.options].findIndex((option, index) => option.dataset.id === value || String(index) === value);
}

function writeFrameUrl(hash, push = false) {
  const frame = selectedFrame();
  if (!frame) return;
  const url = new URL(location.href);
  url.hash = hash;
  url.searchParams.set('frame', frame.id);
  url.searchParams.delete('view');
  if (hash === '#chevrolet-views' && !comparison.hidden) url.searchParams.set('compare', '1');
  else url.searchParams.delete('compare');
  history[push ? 'pushState' : 'replaceState'](null, '', url);
}

function normalizeFrames(raw) {
  if (Array.isArray(raw)) {
    return raw.map((item, idx) => ({
      id: item.id || `view-${String(idx).padStart(2, '0')}`,
      label: item.label,
      image: item.image,
      originalUrl: item.originalUrl || item.image,
      group: item.group || 'Exterior',
      credit: item.credit || 'Images © General Motors / Chevrolet',
      source: item.source || DEFAULT_CONFIGURATOR,
      width: Number(item.width) || 1500,
      height: Number(item.height) || 750,
      pose: item.pose || null,
    }));
  }
  return Object.entries(raw).map(([label, image], idx) => ({
    id: `view-${String(idx).padStart(2, '0')}`,
    label,
    image,
    group: 'Exterior',
    credit: 'Images © General Motors / Chevrolet',
    source: DEFAULT_CONFIGURATOR,
    width: 1500,
    height: 750,
    pose: null,
  }));
}

function selectedFrame() {
  const option = angles.selectedOptions[0];
  if (!option) return null;
  const idx = Number(option.dataset.index);
  return Number.isInteger(idx) ? frames[idx] : null;
}

function updateFooter(frame) {
  if (!creditEl || !frame) return;
  const link = document.createElement('a');
  link.href = frame.source || DEFAULT_CONFIGURATOR;
  link.textContent = frame.source && frame.source !== DEFAULT_CONFIGURATOR ? 'Reference source' : 'Open configurator';
  const credit = frame.credit || 'Images © General Motors / Chevrolet';
  const original = document.createElement('a');
  original.href = frame.originalUrl || frame.image;
  original.textContent = 'Full-resolution image';
  const report = document.createElement('a');
  report.href = './review.html#' + encodeURIComponent(frame.id);
  report.textContent = 'Benchmark';
  creditEl.replaceChildren(
    `${frame.width} × ${frame.height} · ${credit} · `,
    original, ' · ', link, ' · ', report
  );
}

function applyFrameBox(frame) {
  if (!frame) return;
  const w = frame.width || 1500;
  const h = frame.height || 750;
  photo.width = w;
  photo.height = h;
  photoFrame.style.setProperty('--frame-ar', String(w / h));
}

function matchAngle() {
  const frame = selectedFrame();
  if (!frame) return;
  applyFrameBox(frame);
  document.documentElement.dataset.matchView = frame.label;
  window.dispatchEvent(new CustomEvent('match-chevrolet-view', { detail: frame.label }));
}

function setComparison(enabled) {
  compareButton.setAttribute('aria-pressed', String(enabled));
  comparison.hidden = !enabled;
  document.documentElement.classList.toggle('compare-photos', enabled);
  if (enabled) matchAngle();
  else {
    delete document.documentElement.dataset.matchView;
    window.dispatchEvent(new CustomEvent('end-chevrolet-compare'));
  }
}

compareButton.addEventListener('click', () => {
  if (angles.selectedIndex >= 0 && !compareButton.disabled) {
    setComparison(comparison.hidden);
    writeFrameUrl('#chevrolet-views');
  }
});

split.addEventListener('input', () => {
  room.style.setProperty('--photo-share', split.value + '%');
  split.setAttribute('aria-valuetext', split.value + ' percent photo');
});

matchButton.addEventListener('click', () => {
  const frame = selectedFrame();
  if (!frame || matchButton.disabled) return;
  writeFrameUrl('#model', true);
  updateMode();
  matchAngle();
});

const preloaded = new Set();
function preloadAngle(index) {
  if (!frames.length) return;
  const frame = frames[(index + frames.length) % frames.length];
  if (!frame || !frame.image || preloaded.has(frame.image)) return;
  preloaded.add(frame.image);
  const img = new Image();
  img.src = frame.image;
}

function updateCompareAvailability(frame) {
  const override = new URLSearchParams(location.search).has('model');
  const available = !override && Boolean(frame && (frame.pose || window.__zr1xHasPose?.(frame.label)));
  const modelReady = Boolean(window.__zr1xModelLoaded);
  compareButton.disabled = !(modelReady && available);
  matchButton.disabled = !(modelReady && available);
  if (!available && !comparison.hidden) setComparison(false);
}

function showAngle() {
  const frame = selectedFrame();
  if (!frame) return;
  error.hidden = true;
  applyFrameBox(frame);
  updateFooter(frame);
  updateCompareAvailability(frame);
  if (!frame.image) {
    photo.hidden = true;
    error.hidden = false;
    return;
  }
  photo.hidden = false;
  photo.alt = frame.label + ' — Chevrolet Corvette ZR1X reference';
  photo.src = frame.image;
  preloaded.add(frame.image);
  if (!comparison.hidden) matchAngle();
  const idx = angles.selectedIndex;
  if (frames.length > 1) {
    preloadAngle(idx + 1);
    preloadAngle(idx - 1);
  }
}

function moveAngle(delta) {
  if (!angles.options.length) return;
  angles.selectedIndex = (angles.selectedIndex + delta + angles.options.length) % angles.options.length;
  showAngle();
  writeFrameUrl('#chevrolet-views');
}

photo.addEventListener('error', () => {
  photo.hidden = true;
  error.hidden = false;
});
photo.draggable = false;
angles.addEventListener('change', () => { showAngle(); writeFrameUrl('#chevrolet-views'); });
document.getElementById('previous-view').addEventListener('click', () => moveAngle(-1));
document.getElementById('next-view').addEventListener('click', () => moveAngle(1));

window.addEventListener('zr1x-model-loaded', () => {
  updateCompareAvailability(selectedFrame());
});

async function loadAngles() {
  if (angles.options.length) return;
  try {
    sourcePromise ??= fetch('./chevrolet-views.json?v=80bf7f756587').then(response => {
      if (!response.ok) throw new Error('Camera list unavailable');
      return response.json();
    });
    const raw = await sourcePromise;
    if (angles.options.length) return;
    frames = normalizeFrames(raw);
    window.dispatchEvent(new CustomEvent('chevrolet-views-loaded', { detail: frames }));

    const groups = new Map();
    for (const g of GROUP_ORDER) groups.set(g, []);
    frames.forEach((frame, index) => {
      const g = frame.group || 'Exterior';
      if (!groups.has(g)) groups.set(g, []);
      groups.get(g).push({ frame, index });
    });
    const nonEmptyGroups = [...groups.entries()].filter(([, list]) => list.length > 0);
    const useOptgroups = nonEmptyGroups.length > 1 || (nonEmptyGroups.length === 1 && nonEmptyGroups[0][0] !== 'Exterior');

    for (const [groupName, list] of nonEmptyGroups) {
      const container = useOptgroups ? document.createElement('optgroup') : angles;
      if (useOptgroups) {
        container.label = groupName;
        angles.append(container);
      }
      for (const { frame, index } of list) {
        const opt = new Option(frame.label, frame.image || '');
        opt.dataset.index = String(index);
        opt.dataset.id = frame.id;
        container.append(opt);
      }
    }

    const query = new URLSearchParams(location.search);
    const index = requestedIndex();
    if (index >= 0) angles.selectedIndex = index;
    showAngle();
    if (index >= 0 && !query.has('model')) {
      if (location.hash === '#chevrolet-views' && query.has('compare')) setComparison(true);
      else if (['', '#model'].includes(location.hash) || inspectionFrames[location.hash]) matchAngle();
    }
  } catch {
    sourcePromise = undefined;
    photo.hidden = true;
    error.hidden = false;
  }
}

// Each tab is a URL hash. Every hash without its own tab, such as #rear or #top, shows the 3D model.
function updateMode() {
  const gallery = location.hash === '#chevrolet-views';
  if (!gallery && !comparison.hidden) setComparison(false);
  const tab = gallery ? '#chevrolet-views' : '#model';
  document.documentElement.classList.toggle('show-photos', gallery);
  document.documentElement.classList.remove('hide-hud');
  room.hidden = !gallery;
  more.open = false;
  for (const link of nav.querySelectorAll('a')) {
    if (link.hash === tab) link.setAttribute('aria-current', 'true');
    else link.removeAttribute('aria-current');
  }
  if (gallery || inspectionFrames[location.hash]) {
    if (!angles.options.length) loadAngles();
    else {
      const index = requestedIndex();
      if (index >= 0) angles.selectedIndex = index;
      showAngle();
      if (!gallery) matchAngle();
      else if (location.search.includes('compare=') && selectedFrame()?.pose) setComparison(true);
    }
  }
  window.dispatchEvent(new Event('studio-mode-change'));
}

window.addEventListener('hashchange', updateMode);
updateMode();
// Preload chevrolet-views.json poses silently on model tab if deep-linked via ?frame= or data-match-view
if (new URLSearchParams(location.search).has('frame') || new URLSearchParams(location.search).has('view')) loadAngles();

document.addEventListener('pointerdown', event => {
  if (more.open && !more.contains(event.target)) more.open = false;
});
more.addEventListener('click', event => {
  if (event.target.closest('[data-panel]')) {
    more.open = false;
    more.querySelector('summary').focus({ preventScroll: true });
  }
});
window.addEventListener('keydown', event => {
  if (event.altKey || event.ctrlKey || event.metaKey) return;
  if (event.key === 'Escape' && more.open) {
    more.open = false;
    more.querySelector('summary').focus();
  }
  if (room.hidden || event.target === angles || event.target === split) return;
  if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
    event.preventDefault();
    moveAngle(event.key === 'ArrowLeft' ? -1 : 1);
  }
});

let swipe;
room.querySelector('figure').addEventListener('pointerdown', event => {
  if (!event.isPrimary || event.button !== 0 || event.target.closest('a, button, #comparison-controls')) return;
  swipe = { x: event.clientX, y: event.clientY };
  event.currentTarget.setPointerCapture(event.pointerId);
});
room.addEventListener('pointerup', event => {
  if (!swipe) return;
  const dx = event.clientX - swipe.x;
  if (Math.abs(dx) > 45 && Math.abs(dx) > Math.abs(event.clientY - swipe.y)) moveAngle(dx < 0 ? 1 : -1);
  swipe = null;
});
room.addEventListener('pointercancel', () => { swipe = null; });
