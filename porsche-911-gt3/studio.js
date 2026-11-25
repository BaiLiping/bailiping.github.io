const room = document.getElementById('photo-room');
const photo = document.getElementById('photo');
const error = document.getElementById('photo-error');
const angles = document.getElementById('photo-angle');
const more = document.getElementById('more');
const nav = document.querySelector('.mode-switch');
const labRoom = document.getElementById('lab-room');
const labFrames = [...labRoom.querySelectorAll('iframe[data-lab]')];
let sourcePromise;
const matchButton = document.getElementById('match-view');
const compareButton = document.getElementById('compare-view');
const comparison = document.getElementById('comparison-controls');
const split = document.getElementById('comparison-split');
function matchAngle() {
  const option = angles.selectedOptions[0];
  if (option) window.dispatchEvent(new CustomEvent('match-porsche-view', {detail: option.label}));
}
function setComparison(enabled) {
  compareButton.setAttribute('aria-pressed', String(enabled));
  comparison.hidden = !enabled;
  document.documentElement.classList.toggle('compare-photos', enabled);
  if (enabled) matchAngle();
}
compareButton.addEventListener('click', () => {
  if (angles.selectedIndex >= 0) setComparison(comparison.hidden);
});
split.addEventListener('input', () => {
  room.style.setProperty('--photo-share', split.value + '%');
  split.setAttribute('aria-valuetext', split.value + ' percent photo');
});
matchButton.addEventListener('click', () => {
  const option = angles.selectedOptions[0];
  if (!option) return;
  document.documentElement.dataset.matchView = option.label;
  location.hash = 'model';
  window.dispatchEvent(new CustomEvent('match-porsche-view', {detail: option.label}));
});

const preloaded = new Set();
function preloadAngle(index) {
  const option = angles.options[(index + angles.length) % angles.length];
  if (!option || preloaded.has(option.value)) return;
  preloaded.add(option.value);
  const img = new Image();
  img.src = option.value;
}

function showAngle() {
  const option = angles.selectedOptions[0];
  if (!option) return;
  error.hidden = true;
  photo.hidden = false;
  photo.alt = 'Lugano Blue Porsche 911 GT3, ' + option.label.toLowerCase() + ' view';
  photo.src = option.value;
  preloaded.add(option.value);
  if (!comparison.hidden) matchAngle();
  if (angles.length > 1) {
    preloadAngle(angles.selectedIndex + 1);
    preloadAngle(angles.selectedIndex - 1);
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

async function loadAngles() {
  if (angles.length) return;
  try {
    sourcePromise ??= fetch('./porsche-views.json?v=surface-studio-r4').then(response => {
      if (!response.ok) throw new Error('Camera list unavailable');
      return response.json();
    });
    const sources = await sourcePromise;
    if (angles.length) return;
    for (const [name, src] of Object.entries(sources)) angles.add(new Option(name, src));
    showAngle();
  } catch {
    sourcePromise = undefined;
    photo.hidden = true;
    error.hidden = false;
  }
}

// Each tab is a URL hash. Every hash without its own tab, such as #rear, shows the 3D model.
function updateMode() {
  const gallery = location.hash === '#porsche-views';
  if (!gallery && !comparison.hidden) setComparison(false);
  const lab = labFrames.find(frame => location.hash === '#' + frame.dataset.lab);
  const tab = gallery || lab ? location.hash : '#model';
  document.documentElement.classList.toggle('show-photos', gallery);
  document.documentElement.classList.toggle('show-lab', !!lab);
  document.documentElement.classList.remove('hide-hud');
  room.hidden = !gallery;
  labRoom.hidden = !lab;
  for (const frame of labFrames) {
    // A lab loads on its first visit and keeps its state while other tabs are shown.
    if (frame === lab && !frame.getAttribute('src')) frame.src = frame.dataset.src;
    frame.hidden = frame !== lab;
  }
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
    more.querySelector('summary').focus({preventScroll: true});
  }
});
window.addEventListener('keydown', event => {
  if (event.altKey || event.ctrlKey || event.metaKey) return;
  if (event.key === 'Escape' && more.open && !document.getElementById('transmission').open) {
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
  swipe = {x: event.clientX, y: event.clientY};
  event.currentTarget.setPointerCapture(event.pointerId);
});
room.addEventListener('pointerup', event => {
  if (!swipe) return;
  const dx = event.clientX - swipe.x;
  if (Math.abs(dx) > 45 && Math.abs(dx) > Math.abs(event.clientY - swipe.y)) moveAngle(dx < 0 ? 1 : -1);
  swipe = null;
});
room.addEventListener('pointercancel', () => { swipe = null; });
