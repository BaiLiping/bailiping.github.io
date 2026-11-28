const root = document.documentElement;
const room = document.getElementById('photo-room');
const photo = document.getElementById('photo');
const error = document.getElementById('photo-error');
const angles = document.getElementById('photo-angle');
const more = document.getElementById('more');
const nav = document.querySelector('.mode-switch');
const matchButton = document.getElementById('match-view');
const compareButton = document.getElementById('compare-view');
const comparison = document.getElementById('comparison-controls');
const split = document.getElementById('comparison-split');
const frame = room.querySelector('.photo-frame');
const params = new URLSearchParams(location.search);
const galleryOpen = () => ['#ford-views', '#raptor-views'].includes(location.hash);
let sources;
const loadedViews = [];
let initialComparison = false;

function selectedView() {
  return loadedViews[angles.selectedIndex];
}

function matchAngle() {
  const view = selectedView();
  if (!view || matchButton.disabled) return;
  root.dataset.matchView = view.photoView;
  window.dispatchEvent(new CustomEvent('match-raptor-view', { detail: view.photoView }));
}

function setComparison(enabled) {
  const active = enabled && !compareButton.disabled && galleryOpen();
  comparison.hidden = !active;
  compareButton.setAttribute('aria-pressed', String(active));
  root.classList.toggle('compare-photos', active);
  if (active) matchAngle();
}

function updateButtons() {
  const view = selectedView();
  const ready = view && window.raptorHasPhotoPose?.(view.photoView) && root.dataset.modelReady === 'stock';
  matchButton.disabled = !ready;
  compareButton.disabled = !ready || !photo.complete || !photo.naturalWidth || error.hidden === false;
  if (initialComparison && !compareButton.disabled && galleryOpen()) {
    initialComparison = false;
    setComparison(true);
  }
}
window.addEventListener('model-ready', updateButtons);
compareButton.addEventListener('click', () => setComparison(comparison.hidden));
function setSplit(value) {
  split.value = value;
  room.style.setProperty('--photo-share', split.value + '%');
  split.setAttribute('aria-valuetext', split.value + ' percent reference');
}
split.addEventListener('input', () => setSplit(split.value));
const grip = frame.querySelector('.split-grip');
function dragSplit(event) {
  const rect = frame.getBoundingClientRect();
  setSplit(Math.round(100 * (event.clientX - rect.left) / rect.width));
}
grip.addEventListener('pointerdown', event => {
  if (!event.isPrimary || event.button !== 0) return;
  grip.setPointerCapture(event.pointerId);
  dragSplit(event);
});
grip.addEventListener('pointermove', event => {
  if (grip.hasPointerCapture(event.pointerId)) dragSplit(event);
});

matchButton.addEventListener('click', () => {
  if (matchButton.disabled) return;
  location.hash = 'model';
  matchAngle();
});

const preloaded = new Set();
function frameURL(view) {
  // Only the full-size link points away from the studio.
  return view.image?.startsWith('./') ? view.image : view.preview;
}
function preloadNeighbor(index) {
  const url = frameURL(loadedViews[(index + loadedViews.length) % loadedViews.length]);
  if (!url || preloaded.has(url)) return;
  preloaded.add(url);
  const img = new Image();
  img.src = url;
}

function showAngle() {
  const view = selectedView();
  if (!view) return;
  error.hidden = true;
  photo.hidden = false;
  photo.alt = view.alt || view.label + ' on a ' + (view.finish || 'Shelter Green') + ' 2024 Ford F-150 Raptor R';
  if (view.width && view.height) {
    photo.width = view.width;
    photo.height = view.height;
  }
  frame.style.setProperty('--photo-aspect', view.width && view.height ? view.width / view.height : view.aspect);
  document.getElementById('reference-kind').textContent = view.sourceType === 'render' ? 'Ford render' : 'Photo';
  const source = view.source || 'https://bringatrailer.com/listing/2024-ford-f-150-raptor-r-18/';
  document.getElementById('photo-count').textContent = (angles.selectedIndex + 1) + ' / ' + loadedViews.length;
  const finish = view.finish ?? (view.credit ? '' : 'Shelter Green');
  document.getElementById('photo-finish').textContent = finish ? finish + ' · ' : '';
  document.getElementById('photo-credit').textContent = view.credit || '710HP via Bring a Trailer (Lot #181,611)';
  document.getElementById('photo-source').href = error.querySelector('a').href = source;
  const fullPhoto = document.getElementById('full-photo');
  fullPhoto.href = view.original || view.image;
  fullPhoto.hidden = false;
  const truckPhotoLink = document.getElementById('truck-photo-link');
  truckPhotoLink.hidden = !view.truckPhoto;
  if (view.truckPhoto) truckPhotoLink.querySelector('a').href = view.truckPhoto;
  photo.src = frameURL(view);
  updateButtons();
  if (!comparison.hidden) {
    if (matchButton.disabled) setComparison(false);
    else matchAngle();
  }
  preloadNeighbor(angles.selectedIndex - 1);
  preloadNeighbor(angles.selectedIndex + 1);
}

function moveAngle(delta) {
  if (!angles.length) return;
  angles.selectedIndex = (angles.selectedIndex + delta + angles.length) % angles.length;
  showAngle();
}
photo.addEventListener('load', () => {
  photo.width = photo.naturalWidth;
  photo.height = photo.naturalHeight;
  frame.style.setProperty('--photo-aspect', photo.naturalWidth / photo.naturalHeight);
  updateButtons();
});
photo.addEventListener('error', () => {
  photo.hidden = true;
  error.hidden = false;
  setComparison(false);
  updateButtons();
});
photo.draggable = false;
angles.addEventListener('change', showAngle);
document.getElementById('previous-view').addEventListener('click', () => moveAngle(-1));
document.getElementById('next-view').addEventListener('click', () => moveAngle(1));

function selectRequestedPhoto() {
  const preset = root.dataset.photoPreset;
  if (!preset) return;
  const index = loadedViews.findIndex(view => view.photoView === preset || view.preset === preset || view.id === preset || view.legacyId === preset || (view.legacyId && 'photo-' + view.legacyId === preset) || view.aliases?.includes(preset));
  if (index >= 0) angles.selectedIndex = index;
  delete root.dataset.photoPreset;
}

async function loadAngles() {
  try {
    if (!angles.length) {
      sources ??= fetch('./raptor-views.json?v=photo-r10').then(response => {
        if (!response.ok) throw new Error('Camera list unavailable');
        return response.json();
      });
      const views = await sources;
      if (angles.length) return;
      for (const groupName of new Set(views.map(view => view.group))) {
        const group = document.createElement('optgroup');
        group.label = groupName;
        for (const view of views.filter(item => item.group === groupName)) {
          const option = new Option(view.label, view.id);
          option.dataset.photoView = view.photoView;
          group.append(option);
          loadedViews.push(view);
        }
        angles.append(group);
      }
      const requested = params.get('frame');
      if (requested !== null && /^\d+$/.test(requested) && Number(requested) < angles.length) {
        angles.selectedIndex = Number(requested);
        initialComparison = params.get('compare') === '1' && !params.has('model');
      }
    }
    selectRequestedPhoto();
    showAngle();
  } catch {
    sources = undefined;
    photo.hidden = true;
    error.hidden = false;
    setComparison(false);
  }
}

function updateMode() {
  const gallery = galleryOpen() && params.get('capture') !== '1';
  if (!gallery) {
    initialComparison = false;
    setComparison(false);
  }
  root.classList.toggle('show-photos', gallery);
  if (params.get('capture') !== '1') root.classList.remove('hide-hud');
  room.hidden = !gallery;
  more.open = false;
  for (const link of nav.querySelectorAll('a')) {
    if (link.hash === (gallery ? '#ford-views' : '#model')) link.setAttribute('aria-current', 'true');
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
  if (!event.isPrimary || event.button !== 0 || event.target.closest('a, button, .split-grip, #comparison-controls')) return;
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
