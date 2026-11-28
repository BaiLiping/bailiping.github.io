// The photo and the WebGL viewport share the original image's aspect ratio.
const el = id => document.getElementById(id);
const gallery = el('honda-views');
const selector = el('photo-choice');
const picture = el('reference-photo');
const room = el('room');
const modelPage = el('model-main');
let catalog;
let frames = [];
let comparing = false;
let native = false;
let center = {x:.5,y:.5};
let crop;
const box = el('photo-frame');
function updateCrop() {
  const frame=frames.find(f=>f.id===selector.value);
  if (!frame || gallery.hidden || !box.clientWidth) return;
  const zoom=native ? Math.max(1,frame.size[0]/(box.clientWidth*devicePixelRatio)) : 1;
  const radius=.5/zoom;
  for (const axis of ['x','y']) center[axis]=Math.max(radius,Math.min(1-radius,center[axis]));
  const left=center.x-radius, top=center.y-radius;
  crop={zoom,left,top};
  box.classList.toggle('native',native);
  box.tabIndex=native ? 0 : -1;
  box.setAttribute('aria-label',native ? 'Original pixels. Drag or use arrow keys to pan.' : 'Reference photograph');
  picture.style.transform=zoom===1 ? '' : `translate(${-left*zoom*100}%,${-top*zoom*100}%) scale(${zoom})`;
  window.carStudio?.setPhotoCrop(zoom===1 ? null : crop);
  el('native-photo').setAttribute('aria-pressed',String(native));
}
el('native-photo').onclick=()=>{native=!native;updateCrop();if(native)box.focus();};
new ResizeObserver(updateCrop).observe(box);
let drag;
box.onpointerdown=e=>{
  if(!native || !e.isPrimary) return;
  box.setPointerCapture(e.pointerId);
  drag={x:e.clientX,y:e.clientY,fromX:center.x,fromY:center.y};
};
box.onpointermove=e=>{
  if(!drag || !native) return;
  center={x:drag.fromX-(e.clientX-drag.x)/(box.clientWidth*crop.zoom),
          y:drag.fromY-(e.clientY-drag.y)/(box.clientHeight*crop.zoom)};
  updateCrop();
};
box.onlostpointercapture=()=>{drag=null;};
box.onkeydown=e=>{
  const axis={ArrowLeft:[-1,0],ArrowRight:[1,0],ArrowUp:[0,-1],ArrowDown:[0,1]}[e.key];
  if(!native || !axis)return;
  e.preventDefault();
  center.x+=axis[0]*.2/crop.zoom;center.y+=axis[1]*.2/crop.zoom;
  updateCrop();
};
let generation = 0;

function updateLink() {
  const url = new URL(location.href);
  url.searchParams.set('frame', selector.value);
  url.hash = 'model';
  url.searchParams.delete('compare');
  el('match-photo').href = url;
  url.hash = 'honda-views';
  if (comparing) url.searchParams.set('compare','1');
  history.replaceState(null,'',url);
}

async function showFrame() {
  const version = ++generation;
  const frame = frames.find(f => f.id === selector.value);
  if (!frame || gallery.hidden) return;
  updateLink();
  el('photo-frame').style.setProperty('--aspect',frame.size[0] / frame.size[1]);
  el('credit-link').href = frame.source;
  el('credit-link').textContent = frame.credit;
  el('original-link').href = frame.photo;
  el('original-link').textContent = 'Original ' + frame.size.join(' × ');
  el('photo-observation').textContent = frame.note || frame.label;
  el('photo-config').textContent = frame.configuration;
  el('photo-error').hidden = true;
  picture.alt = frame.label;
  if (picture.getAttribute('src')!==frame.photo) {
    center={x:.5,y:.5};
    picture.src=frame.photo;
  }
  updateCrop();
  el('compare-photo').setAttribute('aria-pressed',String(comparing));
  el('photo-frame').classList.toggle('comparing',comparing);
  el('split-control').hidden = !comparing;
  el('split-line').hidden = !comparing;
  try {
    await window.carStudio.ready;
    if (version === generation && !gallery.hidden) {
      window.carStudio.match(frame.id);
      updateCrop();
      el('review').setAttribute('aria-current','page');
      el('model-link').removeAttribute('aria-current');
    }
  } catch {
    if (version !== generation) return;
    el('compare-photo').disabled = true;
    el('photo-frame').classList.remove('comparing');
    el('split-control').hidden = el('split-line').hidden = true;
  }
}

async function enter() {
  const version = ++generation;
  const active = location.hash === '#honda-views';
  gallery.hidden = !active;
  if (!active) {
    window.carStudio?.setPhotoCrop(null);
    native=false;
    el('reference-library').open=false;
    if (!modelPage.contains(room)) modelPage.prepend(room);
    modelPage.hidden = false;
    return;
  }
  el('photo-model').append(room);
  modelPage.hidden = true;
  for (const link of document.querySelectorAll('header nav a')) {
    if (link.id === 'review') link.setAttribute('aria-current','page');
    else link.removeAttribute('aria-current');
  }
  try {
    catalog ??= fetch('honda-views.json?v=d85c6708').then(r => {
      if (!r.ok) throw Error('catalog');
      return r.json();
    });
    const data = await catalog;
    if (version !== generation || gallery.hidden) return;
    if (!frames.length) {
      frames = data.frames.filter(f => !f.localOnly);
      const groups = new Map();
      for (const frame of frames) {
        if (!groups.has(frame.group)) {
          const optgroup = document.createElement('optgroup');
          optgroup.label = frame.group;
          groups.set(frame.group,optgroup);
          selector.append(optgroup);
        }
        groups.get(frame.group).append(new Option(frame.label,frame.id));
      }
    }
    const query = new URLSearchParams(location.search);
    selector.value = frames.some(f => f.id === query.get('frame')) ? query.get('frame') : frames[0].id;
    comparing = query.has('compare');
    showFrame();
  } catch {
    catalog = undefined;
    el('photo-error').textContent = 'Could not load the photograph index.';
    el('photo-error').hidden = false;
  }
}

picture.onerror = () => {
  el('photo-error').textContent = 'The source could not load this photograph. Open the original below or select another view.';
  el('photo-error').hidden = false;
};
picture.onload = () => { el('photo-error').hidden = true; };
selector.onchange = showFrame;
function move(step) {
  selector.selectedIndex = (selector.selectedIndex + step + selector.length) % selector.length;
  showFrame();
}
el('previous-photo').onclick = () => move(-1);
el('next-photo').onclick = () => move(1);
el('compare-photo').onclick = () => { comparing = !comparing; showFrame(); };
el('split').oninput = e => {
  el('photo-frame').style.setProperty('--photo-share',e.target.value + '%');
  e.target.setAttribute('aria-valuetext',e.target.value + ' percent photo');
};
el('match-photo').onclick = e => {
  e.preventDefault();
  history.pushState(null,'',el('match-photo').href);
  enter();
  window.carStudio.ready.then(() => window.carStudio.match(selector.value)).catch(() => {});
  el('review').removeAttribute('aria-current');
  el('model-link').setAttribute('aria-current','page');
};
window.addEventListener('hashchange',enter);
enter();

const library = el('reference-library');
const libraryChoice = el('library-choice');
let references;
library.ontoggle = async () => {
  if (!library.open) return;
  try {
    references ??= fetch('reference-library.json?v=d85c6708').then(r => {
      if (!r.ok) throw Error('Reference library unavailable');
      return r.json();
    });
    const rows=(await references).references;
    if(!libraryChoice.length) {
      for(const name of ['Exterior','Cabin','Mechanical references','Workshop diagrams','Identification and specifications']) {
        const group=document.createElement('optgroup');
        group.label=name;
        for(const row of rows.filter(r=>r.group===name)) group.append(new Option(row.label,row.id));
        libraryChoice.append(group);
      }
    }
    const updateSource=()=>{
      const item=rows.find(r=>r.id===libraryChoice.value);
      el('library-original').href=item.original;
      el('library-original').textContent='Original '+item.size.join(' × ');
      el('library-source').href=item.source;
      el('library-source').textContent=item.credit.split(' (')[0];
    };
    libraryChoice.value=selector.value;
    libraryChoice.onchange=updateSource;
    updateSource();
  } catch {
    references=undefined;
    el('library-source').href='reference-library.json';
    el('library-source').textContent='Open source index';
  }
};
