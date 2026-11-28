const room = document.getElementById('photo-room');
const photoFrame = room.querySelector('.photo-frame');
const photo = document.getElementById('photo');
const error = document.getElementById('photo-error');
const angles = document.getElementById('photo-angle');
const footerCredit = document.getElementById('photo-credit');
const more = document.getElementById('more');
const nav = document.querySelector('.mode-switch');
const inspection = document.getElementById('inspection-choice');
const inspectionRoom = document.getElementById('powertrain-room');
const inspectionHashes = new Set(Array.from(inspection.options, option => option.value));
inspection.addEventListener('change', () => { location.hash = inspection.value; });
let sourcePromise;
const viewMeta = new Map();
const defaultFooterHTML = footerCredit ? footerCredit.innerHTML : '';

const matchButton = document.getElementById('match-view');
const compareButton = document.getElementById('compare-view');
const comparison = document.getElementById('comparison-controls');
const split = document.getElementById('comparison-split');

const previousFrames = {"ext-low-front-left":0,"ext-low-front":35,"ext-low-side-left":4,"ext-low-side-right":25,"ext-low-rear-left":6,"ext-low-rear-right":19,"ext-low-rear":12,"ext-high-rear-left":50,"ext-low-front-left-wheel":53,"ascari-top-down":43,"dashboard":59,"front-seats":60,"rear-seats":61,"provence-front":0,"provence-rear":6,"ice-grey-profile":44,"provence-overhead":51,"factory-360-00":0,"factory-360-01":1,"factory-360-02":2,"factory-360-03":3,"factory-360-04":4,"factory-360-05":5,"factory-360-06":6,"factory-360-07":7,"factory-360-08":8,"factory-360-09":9,"factory-360-10":10,"factory-360-11":11,"factory-360-12":12,"factory-360-13":13,"factory-360-14":14,"factory-360-15":15,"factory-360-16":16,"factory-360-17":17,"factory-360-18":18,"factory-360-19":19,"factory-360-20":20,"factory-360-21":21,"factory-360-22":22,"factory-360-23":23,"factory-360-24":24,"factory-360-25":25,"factory-360-27":26,"factory-360-28":27,"factory-360-29":28,"factory-360-30":29,"factory-360-31":30,"factory-360-32":31,"factory-360-33":32,"factory-360-34":33,"factory-360-35":34,"factory-360-36":35,"factory-360-37":36,"factory-360-38":37,"factory-360-39":38,"factory-360-40":39,"factory-360-41":40,"factory-360-42":41,"factory-360-43":42,"factory-360-56":43,"factory-360-55":44,"factory-360-54":45,"factory-360-51":46,"factory-360-46":47,"factory-360-52":48,"factory-360-45":49,"factory-360-53":50,"factory-360-44":51,"factory-360-26":52,"factory-360-50":53,"factory-360-47":59,"factory-360-48":60,"factory-360-49":61,"factory-360-57":43};

function selectedOption() {
  return angles.selectedOptions[0] || angles.options[angles.selectedIndex] || null;
}

function applyFrameGeometry(meta) {
  const w = meta?.width || 3840;
  const h = meta?.height || 2160;
  photo.width = w;
  photo.height = h;
  photoFrame.style.setProperty('--frame-w', String(w));
  photoFrame.style.setProperty('--frame-h', String(h));
}

function updateFooter(meta) {
  if (!footerCredit) return;
  if (meta && (meta.credit || meta.source)) {
    const parts = [];
    if (meta.group && meta.group !== 'Exterior' && !meta.label.startsWith(meta.group)) parts.push(meta.group);
    parts.push(meta.label);
    if (meta.credit) parts.push(meta.credit);
    footerCredit.replaceChildren();
    footerCredit.append(parts.join(' · '));
    if (meta.source) {
      footerCredit.append(' · ');
      const a = document.createElement('a');
      a.href = meta.source;
      a.textContent = meta.source.includes('configurator.porsche.com') ? 'Open configurator' : 'Source reference';
      a.rel = 'noopener';
      footerCredit.append(a);
    }
  } else {
    footerCredit.innerHTML = defaultFooterHTML;
  }
}

function matchAngle() {
  const option = selectedOption();
  if (!option) return;
  const label = option.dataset.label || option.textContent;
  document.documentElement.dataset.matchView = label;
  window.dispatchEvent(new CustomEvent('match-porsche-view', { detail: label }));
}

function setComparison(enabled) {
  compareButton.setAttribute('aria-pressed', String(enabled));
  comparison.hidden = !enabled;
  document.documentElement.classList.toggle('compare-photos', enabled);
  if (enabled) matchAngle();
  else delete document.documentElement.dataset.matchView;
}

compareButton.addEventListener('click', () => {
  if (angles.selectedIndex >= 0) setComparison(comparison.hidden);
});

split.addEventListener('input', () => {
  room.style.setProperty('--photo-share', split.value + '%');
  split.setAttribute('aria-valuetext', split.value + ' percent photo');
});

matchButton.addEventListener('click', () => {
  const option = selectedOption();
  if (!option) return;
  const label = option.dataset.label || option.textContent;
  document.documentElement.dataset.matchView = label;
  location.hash = 'model';
  window.dispatchEvent(new CustomEvent('match-porsche-view', { detail: label }));
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
  const option = selectedOption();
  if (!option) return;
  const label = option.dataset.label || option.textContent;
  const meta = viewMeta.get(label);
  applyFrameGeometry(meta);
  updateFooter(meta);
  error.hidden = true;
  photo.hidden = false;
  photo.alt = 'Gentian Blue Metallic Porsche 911 Turbo S Cabriolet, ' + label.toLowerCase() + ' view';
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

function populateSelect(sources) {
  viewMeta.clear();
  if (Array.isArray(sources)) {
    const groupOrder = ['Exterior', 'Powertrain', 'Interior'];
    const presentGroups = [...new Set(sources.map(item => item.group || 'Exterior'))];
    const useGroups = presentGroups.length > 1;
    if (useGroups) {
      const orderedGroups = [
        ...groupOrder.filter(g => presentGroups.includes(g)),
        ...presentGroups.filter(g => !groupOrder.includes(g))
      ];
      for (const groupName of orderedGroups) {
        const og = document.createElement('optgroup');
        og.label = groupName;
        for (const item of sources.filter(s => (s.group || 'Exterior') === groupName)) {
          const opt = new Option(item.label, item.image);
          opt.dataset.label = item.label;
          viewMeta.set(item.label, item);
          og.append(opt);
        }
        angles.append(og);
      }
    } else {
      for (const item of sources) {
        const opt = new Option(item.label, item.image);
        opt.dataset.label = item.label;
        viewMeta.set(item.label, item);
        angles.add(opt);
      }
    }
  } else {
    for (const [name, src] of Object.entries(sources)) {
      const opt = new Option(name, src);
      opt.dataset.label = name;
      viewMeta.set(name, { label: name, image: src, width: 3840, height: 2160 });
      angles.add(opt);
    }
  }
}

async function loadAngles() {
  if (angles.length) return;
  try {
    const version = document.documentElement.dataset.viewsVersion || 'turbo-r1';
    sourcePromise ??= fetch(`./porsche-views.json?v=${encodeURIComponent(version)}`).then(response => {
      if (!response.ok) throw new Error('Camera list unavailable');
      return response.json();
    });
    const sources = await sourcePromise;
    if (angles.length) return;
    populateSelect(sources);
    const query = new URLSearchParams(location.search);
    const frame = query.get('frame');
    const frameIndex = previousFrames[frame] ?? (/^\d+$/.test(frame ?? '') ? Number(frame) :
      sources.findIndex(item => item.id === frame));
    const validFrame = Number.isInteger(frameIndex) && frameIndex >= 0 && frameIndex < angles.length;
    if (validFrame) angles.selectedIndex = frameIndex;
    showAngle();
    if (validFrame && location.hash === '#model') matchAngle();
    if (validFrame && query.has('compare') && !query.has('model') && location.hash === '#porsche-views') {
      setComparison(true);
    }
  } catch {
    sourcePromise = undefined;
    photo.hidden = true;
    error.hidden = false;
  }
}

// Each tab is a URL hash. Every hash without its own tab, such as #rear or #top, shows the 3D model.
function updateMode() {
  const gallery = location.hash === '#porsche-views';
  if (!gallery && !comparison.hidden) setComparison(false);
  const inspecting = inspectionHashes.has(location.hash);
  const tab = gallery ? '#porsche-views' : inspecting ? '#powertrain' : '#model';
  inspectionRoom.hidden = !inspecting;
  document.documentElement.classList.toggle('show-powertrain', inspecting);
  if (inspecting) inspection.value = location.hash;
  document.documentElement.classList.toggle('show-photos', gallery);
  document.documentElement.classList.remove('hide-hud');
  room.hidden = !gallery;
  more.open = false;
  for (const link of nav.querySelectorAll('a')) {
    if (link.hash === tab) link.setAttribute('aria-current', 'true');
    else link.removeAttribute('aria-current');
  }
  if (gallery || (location.hash === '#model' && new URLSearchParams(location.search).has('frame'))) loadAngles();
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
