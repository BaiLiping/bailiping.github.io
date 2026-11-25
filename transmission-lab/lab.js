import { drivingStep, driveLength, gearNames, phaseTime } from './sequences.js?v=bac112378e26';
import { engagementAt, mountLinkage, travelAt } from './linkage.js?v=c918544ca383';
import { mountSelector, selectorPose } from './selector.js?v=756fd58ab53a';
import { mountSolenoid } from './solenoid.js?v=a9f96f58230d';

const lab = document.querySelector('#transmission-lab');
const find = selector => lab.querySelector(selector);
const all = selector => [...lab.querySelectorAll(selector)];
const slots = Object.fromEntries(['manual', 'pdk'].map(side => [side, find('[data-side="' + side + '"] .cutaway')]));
const readouts = Object.fromEntries(all('[data-read]').map(node => [node.dataset.read, node]));
const linkage = mountLinkage(lab);
const selector = mountSelector(lab, pose => {
  readouts.stick.textContent = pose.gear ? gearNames[pose.gear] + ' gear' : 'Neutral';
  models?.setSelector(pose);
});
const circuits = find('#valve-circuits'), showValve = mountSolenoid(find('#inside-solenoid'), circuits);
const angle = find('#view-angle');
const reduced = matchMedia('(prefers-reduced-motion: reduce)');
let index = 0, timer = null, models = null;
let frozen = null;
let queued = !reduced.matches, started = false, staticModels = false;
let assembly = reduced.matches || ['#clutch-linkage', '#gear-stick-linkage', '#inside-solenoid'].includes(location.hash) ? 'assembled' : 'opening';

function packCaption(gear, grip, overlap) {
  const condition = grip === 0 ? 'ready' : grip === 1 ? 'coupled' : overlap ? 'handover' : 'slipping';
  return gearNames[gear] + ' · ' + condition;
}

function showPlayback() {
  const playing = timer !== null;
  lab.toggleAttribute('data-playing', playing);
  lab.toggleAttribute('data-queued', queued);
  find('#play .play-label').textContent = playing || queued ? 'Pause drive' : started ? 'Resume drive' : 'Play drive';
  find('#assembly-hint').textContent = queued ? 'The gears will shift when assembly finishes. ' : 'Assembling both transmissions. ';
}

function refresh(animate = false) {
  const step = drivingStep(index);
  const progress = frozen?.progress ?? (step.manualShifts ? [0, 0, .60, 1][step.phase] : 0);
  const pose = selectorPose(step.manualFrom, step.manualTo, progress);
  const travel = frozen?.travel ?? travelAt(step.manualClutch);
  const grip = engagementAt(travel);
  const overlap = step.oddClutch > 0 && step.evenClutch > 0;
  find('#shift-title').textContent = step.title;
  find('#phase-name').textContent = step.label;
  readouts.manualClutch.textContent = grip === 0 ? 'Open · pedal down' : grip === 1 ? 'Clamped' : 'Slipping · bite point';
  readouts.odd.textContent = packCaption(step.oddGear, step.oddClutch, overlap);
  readouts.even.textContent = packCaption(step.evenGear, step.evenClutch, overlap);
  readouts.command.textContent = step.command;
  readouts.sensors.textContent = step.sensors;
  showValve(step, animate && !reduced.matches);
  linkage.move(travel, pose.gear, animate && !reduced.matches);
  selector.move(step.manualFrom, step.manualTo, progress, animate && !reduced.matches ? (step.phase === 3 ? 605 : 1100) : 0);
  models?.setState({ manualGear: pose.gear, selector: selector.pose, manualClutch: grip, manualPedal: travel,
    oddGear: step.oddGear, evenGear: step.evenGear, oddClutch: step.oddClutch, evenClutch: step.evenClutch,
    outputSpeed: step.outputSpeed, moving: true, running: timer !== null && !document.hidden });
  showPlayback();
}

function startQueued() {
  if (!queued || document.hidden || assembly === 'opening' || timer !== null) return;
  frozen = null;
  queued = false;
  started = true;
  timer = window.setInterval(() => {
    index = (index + 1) % driveLength;
    refresh(true);
  }, phaseTime);
  refresh(true);
}

function pause(resumeWhenVisible = false) {
  queued = resumeWhenVisible;
  if (timer !== null) {
    frozen = { progress: selector.progress, travel: linkage.travel };
    window.clearInterval(timer);
  }
  timer = null;
  refresh();
}

function showAssembly(mode) {
  assembly = mode;
  lab.dataset.assembly = mode;
  find('#assembly-hint').hidden = mode !== 'opening';
  find('#skip-assembly').hidden = mode !== 'opening';
  find('#assembly-action').setAttribute('aria-pressed', String(mode === 'exploded'));
  for (const source of all('.cutaway source')) source.media = mode === 'assembled' ? 'not all' : 'all';
  showPlayback();
  if (mode !== 'opening') startQueued();
}

function setAssembly(mode) {
  if (models) models.setAssembly(mode);
  else showAssembly(mode);
}

function toggleDrive() {
  if (timer !== null || queued) { pause(); return; }
  queued = true;
  if (assembly === 'opening') setAssembly('assembled');
  startQueued();
}

function useStills() {
  if (staticModels) return;
  staticModels = true;
  lab.dataset.models = 'static';
  find('#camera-tools').hidden = true;
  find('#angle-control').hidden = true;
  find('#view-instructions').textContent = 'The cutaway images show first gear. The readouts and diagrams below follow the drive.';
  const previous = models;
  models = null;
  previous?.dispose();
  for (const [side, slot] of Object.entries(slots)) {
    slot.tabIndex = -1;
    slot.setAttribute('aria-label', (side === 'manual' ? 'Manual' : 'PDK') + ' transmission cutaway image.');
  }
  showAssembly(assembly === 'exploded' ? 'exploded' : 'assembled');
}

async function prepareModels() {
  try {
    const { mountMechanisms } = await import('./mechanisms.js?v=bff57ac1cebd');
    models = mountMechanisms({ ...slots, initialAssembly: assembly, onOrbit: () => { angle.value = 'custom'; },
      onAssemblyChange: showAssembly, onFirstDraw: () => { if (!staticModels) lab.dataset.models = 'ready'; } });
    for (const [side, slot] of Object.entries(slots)) {
      slot.tabIndex = 0;
      slot.setAttribute('aria-label', (side === 'manual' ? 'Manual' : 'PDK') + ' transmission cutaway. Use the arrow keys to rotate both views.');
    }
    find('#camera-tools').hidden = false;
    find('#angle-control').hidden = false;
    refresh();
    models.resize();
  } catch { useStills(); }
}

find('#play').addEventListener('click', toggleDrive);
circuits.addEventListener('change', () => showValve(drivingStep(index), false));
function zoom(factor) {
  if (!models) return;
  const value = models.zoomBy(factor);
  find('#zoom-out').disabled = value <= .65;
  find('#zoom-in').disabled = value >= 2.6;
}
find('#zoom-out').addEventListener('click', () => zoom(1 / 1.2));
find('#zoom-in').addEventListener('click', () => zoom(1.2));
find('#assembly-action').addEventListener('click', () => setAssembly(assembly === 'exploded' ? 'assembled' : 'exploded'));
find('#skip-assembly').addEventListener('click', () => setAssembly('assembled'));
angle.addEventListener('change', () => models?.setView(angle.value));
find('#reset-view').addEventListener('click', () => {
  angle.value = 'selector';
  models?.reset(); zoom(1);
});
function interruptOpening(event) {
  if (assembly !== 'opening') return;
  if (event.target instanceof Element && event.target.closest('#assembly-action, #skip-assembly, [data-play]') &&
      (event.type !== 'keydown' || ['Enter', ' '].includes(event.key))) return;
  setAssembly('assembled');
}
for (const name of ['pointerdown', 'click', 'input']) lab.addEventListener(name, interruptOpening, { capture: true });
for (const slot of Object.values(slots)) {
  slot.addEventListener('wheel', interruptOpening, { passive: true });
  slot.addEventListener('webglcontextlost', useStills, true);
}
window.addEventListener('keydown', interruptOpening, { capture: true });
reduced.addEventListener('change', () => {
  if (!reduced.matches) return;
  pause();
  if (assembly === 'opening') setAssembly('assembled');
});
document.addEventListener('visibilitychange', () => {
  if (document.hidden && timer !== null) pause(true);
  else if (!document.hidden) startQueued();
});
window.addEventListener('pageshow', startQueued);
window.addEventListener('pagehide', event => {
  if (event.persisted) { if (timer !== null) pause(true); }
  else { pause(); models?.dispose(); }
});
lab.dataset.enhanced = '';
for (const element of all('.interactive[inert]')) element.inert = false;
showAssembly(assembly);
refresh();
prepareModels();
