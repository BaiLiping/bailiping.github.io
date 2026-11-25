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
  window.dispatchEvent(new CustomEvent('match-crv-view', { detail: option.dataset.photoView }));
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
  const subject = option.dataset.subject || 'Second-generation Honda CR-V';
  photo.alt = subject + ' — ' + option.label;
  document.getElementById('photo-subject').textContent = subject;
  const fullPhoto = document.getElementById('full-photo');
  const originalUrl = option.dataset.original || option.value;
  if (originalUrl) {
    fullPhoto.href = originalUrl;
    fullPhoto.hidden = false;
  } else {
    fullPhoto.hidden = true;
  }
  const sourceLink = document.getElementById('photo-source-link');
  if (sourceLink && option.dataset.source) {
    sourceLink.href = option.dataset.source;
    sourceLink.hidden = false;
  } else if (sourceLink) {
    sourceLink.hidden = true;
  }
  const creditEl = document.getElementById('photo-credit');
  if (creditEl && option.dataset.credit) {
    creditEl.textContent = option.dataset.credit;
  }
  updateMatch();
  document.getElementById('compare-frames').href = './frame-by-frame/#' + option.dataset.photoView;
  if (option.value) {
    photo.src = option.value;
  } else {
    photo.hidden = true;
    error.hidden = false;
  }
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
    sourcePromise ??= fetch('./crv-views.json?v=aeb94960c6').then(response => {
      if (!response.ok) throw new Error('Camera list unavailable');
      return response.json();
    });
    const sources = await sourcePromise;
    if (angles.length) return;
    const groupOrder = ['Exterior', 'Interior', 'Powertrain'];
    for (const view of sources) {
      if (view.group && !groupOrder.includes(view.group)) groupOrder.push(view.group);
    }
    for (const groupName of groupOrder) {
      const items = sources.filter(item => item.group === groupName);
      if (!items.length) continue;
      const group = document.createElement('optgroup');
      group.label = groupName;
      for (const view of items) {
        const option = new Option(view.label, view.preview || view.image || '');
        option.dataset.subject = view.subject || '';
        option.dataset.original = view.image || '';
        option.dataset.photoView = view.photoView;
        option.dataset.preset = view.preset || '';
        if (view.source) option.dataset.source = view.source;
        if (view.credit) option.dataset.credit = view.credit;
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
  const gallery = location.hash === '#crv-views';
  const tab = gallery ? '#crv-views' : '#model';
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
