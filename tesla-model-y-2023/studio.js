import { VEHICLE, loadViews } from './model-y-config.js?v=my23-dbfa5aa7-2';

const room = document.getElementById('photo-room');
const photoFrame = room.querySelector('.photo-frame');
const photo = document.getElementById('photo');
const error = document.getElementById('photo-error');
const angles = document.getElementById('photo-angle');
const prevButton = document.getElementById('previous-view');
const nextButton = document.getElementById('next-view');
const caption = document.getElementById('photo-caption');
const more = document.getElementById('more');
const nav = document.querySelector('.mode-switch');
const matchButton = document.getElementById('match-view');
const compareButton = document.getElementById('compare-view');
const comparison = document.getElementById('comparison-controls');
const split = document.getElementById('comparison-split');


let modelReady = false;
let overrideModel = new URLSearchParams(location.search).has('model');
let availablePoses = new Set();

function updateActionAvailability() {
  const option = angles.selectedOptions[0];
  const hasViews = angles.options.length > 0;
  prevButton.disabled = !hasViews;
  nextButton.disabled = !hasViews;
  angles.disabled = !hasViews;
  const canMatch = !overrideModel && modelReady && Boolean(option) && availablePoses.has(option.dataset.id);
  matchButton.disabled = !canMatch;
  compareButton.disabled = !canMatch;
  if (!canMatch && !comparison.hidden) setComparison(false);
}

window.addEventListener('model-ready', event => {
  modelReady = true;
  overrideModel = Boolean(event.detail?.overrideModel);
  availablePoses = event.detail?.poseNames instanceof Set ? event.detail.poseNames : new Set();
  updateActionAvailability();
});

function matchAngle() {
  const option = angles.selectedOptions[0];
  if (!option) return;
  document.documentElement.dataset.matchView = option.dataset.id;
  window.dispatchEvent(new CustomEvent('match-tesla-view', { detail: option.dataset.id }));
}

function setComparison(enabled) {
  compareButton.setAttribute('aria-pressed', String(enabled));
  comparison.hidden = !enabled;
  document.documentElement.classList.toggle('compare-photos', enabled);
  if (enabled) matchAngle();
  else delete document.documentElement.dataset.matchView;
}

compareButton.addEventListener('click', () => {
  if (angles.selectedIndex >= 0 && !compareButton.disabled) setComparison(comparison.hidden);
});

split.addEventListener('input', () => {
  room.style.setProperty('--photo-share', split.value + '%');
  split.setAttribute('aria-valuetext', split.value + ' percent photo');
});

matchButton.addEventListener('click', () => {
  const option = angles.selectedOptions[0];
  if (!option || matchButton.disabled) return;
  document.documentElement.dataset.matchView = option.dataset.id;
  location.hash = 'model';
  window.dispatchEvent(new CustomEvent('match-tesla-view', { detail: option.dataset.id }));
});

const preloaded = new Set();
function preloadAngle(index) {
  const option = angles.options[(index + angles.options.length) % angles.options.length];
  if (!option || !option.value || preloaded.has(option.value)) return;
  preloaded.add(option.value);
  const img = new Image();
  img.src = option.value;
}

function updateFrameDimensions(option) {
  const width = Number(option.dataset.width) || VEHICLE.defaultPhotoWidth;
  const height = Number(option.dataset.height) || VEHICLE.defaultPhotoHeight;
  photo.width = width;
  photo.height = height;
  photoFrame.style.setProperty('--frame-aspect', `${width} / ${height}`);
}

function updateFooterCaption(option) {
  const source = document.createElement('a');
  source.href = option.dataset.source;
  source.textContent = 'Source';
  const original = document.createElement('a');
  original.href = option.value;
  original.textContent = option.dataset.width + ' × ' + option.dataset.height;
  caption.replaceChildren(`${angles.selectedIndex + 1} / ${angles.options.length} · ${option.dataset.credit} · `,
    original, ' · ', source);
}

function showAngle() {
  const option = angles.selectedOptions[0];
  if (!option) {
    photo.hidden = true;
    error.hidden = false;
    updateActionAvailability();
    return;
  }
  error.hidden = true;
  photo.hidden = false;
  updateFrameDimensions(option);
  photo.alt = `${option.dataset.group} · Model Y · ${option.label}`;
  error.querySelector('a').href = option.dataset.source;
  photo.src = option.value;
  preloaded.add(option.value);
  updateFooterCaption(option);
  updateActionAvailability();
  if (!comparison.hidden && !compareButton.disabled) matchAngle();
  if (angles.options.length > 1) {
    preloadAngle(angles.selectedIndex + 1);
    preloadAngle(angles.selectedIndex - 1);
  }
}

function moveAngle(delta) {
  if (!angles.options.length) return;
  angles.selectedIndex = (angles.selectedIndex + delta + angles.options.length) % angles.options.length;
  showAngle();
}

photo.addEventListener('error', () => {
  photo.hidden = true;
  error.hidden = false;
});
photo.draggable = false;
angles.addEventListener('change', showAngle);
prevButton.addEventListener('click', () => moveAngle(-1));
nextButton.addEventListener('click', () => moveAngle(1));

function populateAngles(views) {
  const groups = new Map();
  for (const view of views) {
    if (!groups.has(view.group)) {
      const group = document.createElement('optgroup');
      group.label = view.group;
      groups.set(view.group, group);
      angles.append(group);
    }
    const option = new Option(view.label, view.image);
    for (const key of ['id', 'width', 'height', 'group', 'source', 'credit']) {
      option.dataset[key] = view[key];
    }
    groups.get(view.group).append(option);
  }
}

async function loadAngles() {
  if (angles.options.length) return;
  try {
    const sources = await loadViews();
    if (angles.options.length) return;
    populateAngles(sources);
    if (!angles.options.length) {
      photo.hidden = true;
      error.hidden = false;
      updateActionAvailability();
      return;
    }
    const query = new URLSearchParams(location.search);
    const frame = query.get('frame');
    const validFrame = frame !== null && /^\d+$/.test(frame) && Number(frame) < angles.options.length;
    if (validFrame) angles.selectedIndex = Number(frame);
    showAngle();
    if (validFrame && query.has('compare') && !query.has('model') && location.hash === '#tesla-views') {
      if (modelReady) {
        if (!compareButton.disabled) setComparison(true);
      } else {
        window.addEventListener('model-ready', () => {
          if (location.hash === '#tesla-views' && !compareButton.disabled) setComparison(true);
        }, { once: true });
      }
    }
  } catch {
    photo.hidden = true;
    error.hidden = false;
    updateActionAvailability();
  }
}

// Each tab is a URL hash. Every hash without its own tab, such as #rear or #top, shows the 3D model.
function updateMode() {
  const gallery = location.hash === '#tesla-views';
  if (!gallery && !comparison.hidden) setComparison(false);
  const tab = gallery ? '#tesla-views' : '#model';
  document.documentElement.classList.toggle('show-photos', gallery);
  document.documentElement.classList.remove('hide-hud');
  room.hidden = !gallery;
  more.open = false;
  for (const link of nav.querySelectorAll('a')) {
    if (link.hash === tab) link.setAttribute('aria-current', 'true');
    else link.removeAttribute('aria-current');
  }
  if (gallery) loadAngles();
  window.dispatchEvent(new Event('studio-mode-change'));
}

window.addEventListener('hashchange', updateMode);
updateMode();

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
