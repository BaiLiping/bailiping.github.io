import { loadCivicViews } from './civic-views.js?v=37fa5358e1';

const room = document.getElementById('photo-room');
const photo = document.getElementById('photo');
const error = document.getElementById('photo-error');
const angles = document.getElementById('photo-angle');
const more = document.getElementById('more');
const nav = document.querySelector('.mode-switch');
const creditText = document.getElementById('photo-credit');
const matchButton = document.getElementById('match-view');

let loadedViews;

matchButton.addEventListener('click', () => {
  const option = angles.selectedOptions[0];
  if (!option?.dataset.photoView) return;
  document.documentElement.dataset.matchView = option.dataset.photoView;
  location.hash = 'model';
  window.dispatchEvent(new CustomEvent('match-civic-view', { detail: option.dataset.photoView }));
});

function updateMatch() {
  const option = angles.selectedOptions[0];
  matchButton.disabled = !option?.dataset.photoView || document.documentElement.dataset.modelReady !== 'stock';
  matchButton.textContent = 'Match in 3D';
}
window.addEventListener('model-ready', updateMatch);

function showAngle() {
  const option = angles.selectedOptions[0];
  if (!option) return;
  error.hidden = true;
  photo.hidden = false;
  photo.alt = option.label + ' · ' + option.dataset.credit;
  const fullPhoto = document.getElementById('full-photo');
  const fullUrl = option.dataset.original || option.value;
  if (fullUrl) {
    fullPhoto.href = fullUrl;
    fullPhoto.hidden = false;
  } else {
    fullPhoto.hidden = true;
  }
  if (creditText && option.dataset.credit) {
    creditText.textContent = option.dataset.credit;
  }
  for (const id of ['photo-source', 'photo-source-fallback']) document.getElementById(id).href = option.dataset.source;
  updateMatch();
  document.getElementById('compare-frames').href = './frame-by-frame/' + (option.dataset.photoView ? '#' + option.dataset.photoView : '');
  photo.src = option.value;
}

function moveAngle(delta) {
  if (!angles.length) return;
  angles.selectedIndex = (angles.selectedIndex + delta + angles.length) % angles.length;
  showAngle();
}

photo.addEventListener('error', () => {
  photo.hidden = true;
  error.hidden = false;
});
photo.draggable = false;
angles.addEventListener('change', showAngle);
document.getElementById('previous-view').addEventListener('click', () => moveAngle(-1));
document.getElementById('next-view').addEventListener('click', () => moveAngle(1));

function selectRequestedPhoto() {
  const preset = document.documentElement.dataset.photoPreset;
  if (!preset) return;
  const option = [...angles.options].find(
    item => item.dataset.photoView === preset || item.dataset.preset === preset || item.dataset.id === preset
  );
  if (option) angles.selectedIndex = option.index;
  delete document.documentElement.dataset.photoPreset;
}

async function loadAngles() {
  if (angles.length) {
    selectRequestedPhoto();
    showAngle();
    return;
  }
  try {
    loadedViews ??= loadCivicViews();
    const { views } = await loadedViews;
    if (angles.length) return;
    if (!views.length) {
      photo.hidden = true;
      error.hidden = false;
      return;
    }
    const orderedGroups = ['Exterior', 'Interior', 'Powertrain'];
    const extraGroups = [...new Set(views.map(v => v.group).filter(g => g && !orderedGroups.includes(g)))];
    for (const groupName of [...orderedGroups, ...extraGroups]) {
      const items = views.filter(item => (item.group || 'Exterior') === groupName);
      if (!items.length) continue;
      const group = document.createElement('optgroup');
      group.label = groupName;
      for (const view of items) {
        const option = new Option(view.label, view.preview || view.image || '');
        option.dataset.id = view.id || '';
        option.dataset.original = view.image || '';
        option.dataset.photoView = view.camera ? view.photoView : '';
        option.dataset.source = view.source;
        option.dataset.preset = view.preset || '';
        option.dataset.credit = view.credit || '';
        group.append(option);
      }
      angles.append(group);
    }
    selectRequestedPhoto();
    showAngle();
  } catch {
    loadedViews = undefined;
    photo.hidden = true;
    error.hidden = false;
  }
}

// Each tab is a URL hash. Every hash without its own tab, such as #rear or #photo-..., shows the 3D model.
function updateMode() {
  const gallery = location.hash === '#civic-views';
  const tab = gallery ? '#civic-views' : '#model';
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
  if (event.target.closest('.panel-buttons button')) {
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
  if (room.hidden || event.target === angles) return;
  if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
    event.preventDefault();
    moveAngle(event.key === 'ArrowLeft' ? -1 : 1);
  }
});

let swipe;
room.querySelector('figure').addEventListener('pointerdown', event => {
  if (!event.isPrimary || event.button !== 0 || event.target.closest('a, button')) return;
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
