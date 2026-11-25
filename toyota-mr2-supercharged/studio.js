const room = document.getElementById('photo-room');
const photo = document.getElementById('photo');
const error = document.getElementById('photo-error');
const angles = document.getElementById('photo-angle');
const more = document.getElementById('more');
const nav = document.querySelector('.mode-switch');
let sourcePromise;
const matchButton = document.getElementById('match-view');
matchButton.addEventListener('click', () => {
  const option = angles.selectedOptions[0];
  if (!option?.dataset.photoView) return;
  document.documentElement.dataset.matchView = option.dataset.photoView;
  location.hash = 'model';
  window.dispatchEvent(new CustomEvent('match-mr2-view', { detail: option.dataset.photoView }));
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
  photo.alt = option.dataset.alt || option.label + ' on the reference 1988 Toyota MR2 Supercharged';
  document.getElementById('photo-credit').textContent = option.dataset.credit || 'JWeston listing photograph via Bring a Trailer';
  const listing = document.getElementById('photo-listing');
  listing.hidden = Boolean(option.dataset.credit);
  const fallback = error.querySelector('a');
  fallback.href = option.dataset.credit ? option.dataset.original : listing.querySelector('a').href;
  fallback.textContent = option.dataset.credit ? 'Open the supplied image' : 'View the Bring a Trailer listing';
  const fullPhoto = document.getElementById('full-photo');
  fullPhoto.href = option.dataset.original || option.value;
  fullPhoto.hidden = false;
  updateMatch();
  document.getElementById('compare-frames').href = './frame-by-frame/#' + option.dataset.photoView;
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
  const option = [...angles.options].find(item => item.dataset.photoView === preset || item.dataset.preset === preset);
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
    sourcePromise ??= fetch('./mr2-views.json?v=0b4d32aa8d').then(response => {
      if (!response.ok) throw new Error('Camera list unavailable');
      return response.json();
    });
    const sources = await sourcePromise;
    if (angles.length) return;
    for (const groupName of ['Interior', 'Powertrain', 'Exterior']) {
      const group = document.createElement('optgroup');
      group.label = groupName;
      for (const view of sources.filter(item => item.group === groupName)) {
        const option = new Option(view.label, view.preview || view.image);
        option.dataset.original = view.image;
        if (view.credit) option.dataset.credit = view.credit;
        if (view.alt) option.dataset.alt = view.alt;
        option.dataset.photoView = view.photoView;
        option.dataset.preset = view.preset || '';
        group.append(option);
      }
      angles.append(group);
    }
    selectRequestedPhoto();
    showAngle();
  } catch {
    sourcePromise = undefined;
    photo.hidden = true;
    error.hidden = false;
  }
}

// Each tab is a URL hash. Every hash without its own tab, such as #rear, shows the 3D model.
function updateMode() {
  const gallery = location.hash === '#mr2-views';
  const tab = gallery ? '#mr2-views' : '#model';
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
