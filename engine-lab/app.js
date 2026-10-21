import { createJetScene } from './jet.js';
import { createTurboScene } from './turbo.js';

const $ = (selector) => document.querySelector(selector);
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
const defaults = { jetDemand: .45, carDemand: .45, jet: .45, car: .45, turbo: .45, cycle: 90, time: 0 };
const state = { ...defaults, view: 'jet', playing: !reducedMotion.matches, slow: false, bypass: false, powerView: false };
const selected = { jet: 'compressor', car: 'cylinder' };

const parts = {
  jet: [
    { id: 'intake', name: 'The intake', text: 'The intake guides outside air into the engine.', detail: 'Air supplies the oxygen for combustion. In turbofan mode, the front fan divides the flow into core air and bypass air.' },
    { id: 'compressor', name: 'The compressor', text: 'Rotating blades squeeze incoming air. The turbine supplies the power through a shared shaft.', detail: 'Higher-pressure air enters the combustor, where fuel is added and burned.' },
    { id: 'combustor', name: 'The combustor', text: 'Fuel burns continuously in the compressed air, adding heat to the flowing gas.', detail: 'All engine stages operate at the same time. The combustor mainly raises temperature; pressure drops slightly across it.' },
    { id: 'turbine', name: 'The turbine', text: 'Hot gas expands through the turbine, turning its blades and powering the compressor.', detail: 'The shaft carries mechanical power. A turbofan also has a turbine driving its fan; gas does not travel through either shaft.' },
    { id: 'nozzle', name: 'The nozzle', text: 'The remaining gas expands and accelerates out the back. The engine experiences thrust in the opposite direction.', detail: 'With airflow moving right in this drawing, forward thrust points left. In a turbofan, the bypass stream also contributes thrust.' },
  ],
  car: [
    { id: 'compressor', name: 'The compressor', text: 'The turbo compressor packs more fresh air into the engine. More oxygen allows more fuel to be burned.', detail: 'It is driven by the exhaust turbine through the turbo shaft. Intake air and exhaust use separate passages.' },
    { id: 'intercooler', name: 'The intercooler', text: 'Compressing air heats it up. The intercooler removes some of that heat before the air reaches the cylinder.', detail: 'Cooling increases air density at a given pressure. The piston then compresses the trapped charge further.' },
    { id: 'cylinder', name: 'The cylinder', text: 'Fuel burns in the cylinder. Expanding gas pushes the piston and turns the crankshaft.', detail: 'Follow intake, compression, power, and exhaust over two crankshaft revolutions. Select a stroke to pause and inspect it.' },
    { id: 'turbine', name: 'The exhaust turbine', text: 'Exhaust leaving the cylinder spins the turbine. Its shaft turns the turbo compressor.', detail: 'Increase power demand to watch boost respond gradually. The turbo recovers exhaust energy, and also creates exhaust backpressure.' },
    { id: 'crankshaft', name: 'The crankshaft', text: 'A connecting rod turns the piston’s up-and-down motion into crankshaft rotation.', detail: 'The drivetrain carries that power to the wheels. The turbo shaft is separate; it has no mechanical connection to the crankshaft in this conventional arrangement.' },
  ],
};

const strokes = [
  { name: 'Intake', text: 'The piston moves down. The intake valve opens and fresh charge enters.', valves: 'Intake open · exhaust closed' },
  { name: 'Compression', text: 'The piston moves up, compressing the trapped charge further.', valves: 'Both valves closed' },
  { name: 'Power', text: 'After ignition near the top, hot gas expands and pushes the piston down.', valves: 'Both valves closed' },
  { name: 'Exhaust', text: 'The piston moves up and pushes burned gas toward the turbo turbine.', valves: 'Intake closed · exhaust open' },
];

function announce(message) { $('#status').textContent = message; }
const jetScene = createJetScene($('#jet-scene'), { onSelect: (id) => selectPart('jet', id) });
const carScene = createTurboScene($('#car-scene'), { onSelect: (id) => selectPart('car', id) });

for (const target of ['jet', 'car']) {
  const bars = $(`#${target}-response`);
  for (let i = 0; i < 8; i++) {
    const bar = document.createElement('i');
    bar.style.setProperty('--bar-height', `${15 + i * 1.8}px`);
    bars.append(bar);
  }
}

function renderMeter(target, fraction) {
  const meter = $(`#${target}-response`);
  const percent = Math.round(fraction * 100);
  meter.setAttribute('aria-valuenow', String(percent));
  meter.setAttribute('aria-valuetext', `${percent} percent of illustrative range; not measured performance`);
  Array.from(meter.children).forEach((bar, i) => bar.classList.toggle('on', i < Math.round(fraction * 8)));
}

let lastStroke = -1;
function renderStroke() {
  const index = Math.min(3, Math.floor(state.cycle / 180));
  if (lastStroke !== index) {
    const stroke = strokes[index];
    $('#stroke-title').textContent = stroke.name;
    $('#stroke-description').textContent = stroke.text;
    $('#valve-state').textContent = stroke.valves;
    document.querySelectorAll('button[data-stroke]').forEach((button) => button.setAttribute('aria-pressed', String(Number(button.dataset.stroke) === index)));
    lastStroke = index;
  }
  $('#cycle-label').textContent = `${Math.floor(state.cycle)}° / 720°`;
}

function render() {
  if (state.view !== 'car') jetScene.render({ time: state.time, jet: state.jet, demand: state.jetDemand, bypass: state.bypass, powerView: state.powerView });
  if (state.view !== 'jet') carScene.render({ time: state.time, cycle: state.cycle, turbo: state.turbo, demand: state.carDemand, powerView: state.powerView });
  renderMeter('jet', state.jet);
  renderMeter('car', state.turbo);
  renderStroke();
  $('#engine-panels').dataset.playing = String(state.playing);
}

function syncPlayback() {
  const button = $('#play-pause');
  button.querySelector('span').textContent = state.playing ? 'Pause' : 'Play';
  button.querySelector('svg').innerHTML = state.playing ? '<path d="M6 4v12M14 4v12"/>' : '<path d="m6 3 10 7-10 7Z" fill="currentColor" stroke="none"/>';
  button.setAttribute('aria-label', state.playing ? 'Pause both engine animations' : 'Play both engine animations');
  $('#engine-panels').dataset.playing = String(state.playing);
}

function setPlaying(playing, quiet = false) {
  state.playing = playing;
  syncPlayback();
  if (!quiet) announce(playing ? 'Animations playing.' : 'Animations paused. Select a component or step through a stroke.');
}

function showInspector(engine) {
  const list = parts[engine];
  const current = list.find((part) => part.id === selected[engine]) || list[0];
  const container = $('#component-buttons');
  container.replaceChildren();
  list.forEach((part, index) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = String(index + 1);
    button.setAttribute('aria-label', `Explore ${part.name.toLowerCase()}`);
    button.setAttribute('aria-pressed', String(part.id === current.id));
    button.title = part.name;
    button.dataset.component = part.id;
    button.addEventListener('click', () => selectPart(engine, part.id, true));
    container.append(button);
  });
  $('#component-title').textContent = current.name;
  $('#component-description').textContent = current.text;
  $('#component-detail').textContent = current.detail;
}

function selectPart(engine, id, preserveFocus = false) {
  if (!parts[engine].some((part) => part.id === id)) return;
  selected[engine] = id;
  (engine === 'jet' ? jetScene : carScene).select(id);
  if (state.view === 'compare') setView(engine);
  else showInspector(engine);
  if (preserveFocus) $(`[data-component="${id}"]`).focus({ preventScroll: true });
  requestAnimationFrame(() => centerDiagram(engine, id));
  announce(`${parts[engine].find((part) => part.id === id).name}: ${$('#component-description').textContent}`);
}

function setView(view, updateHash = true) {
  if (!['jet', 'car', 'compare'].includes(view)) view = 'jet';
  state.view = view;
  document.body.classList.toggle('is-compare', view === 'compare');
  document.querySelectorAll('[data-view]').forEach((button) => {
    const active = button.dataset.view === view;
    button.setAttribute('aria-selected', String(active));
    button.tabIndex = active ? 0 : -1;
  });
  $('#jet-panel').hidden = view === 'car';
  $('#car-panel').hidden = view === 'jet';
  $('#inspector').hidden = view === 'compare';
  $('#comparison').hidden = view !== 'compare';
  $('#engine-panels').setAttribute('aria-labelledby', `tab-${view}`);
  $('#compare-link').hidden = view === 'compare';
  if (view !== 'compare') showInspector(view);
  jetScene.select(selected.jet);
  carScene.select(selected.car);
  if (updateHash && location.hash !== `#${view}`) history.replaceState(null, '', `#${view}`);
  render();
}

document.querySelectorAll('[data-view]').forEach((button, index, buttons) => {
  button.addEventListener('click', () => setView(button.dataset.view));
  button.addEventListener('keydown', (event) => {
    let next = index;
    if (event.key === 'ArrowRight') next = (index + 1) % buttons.length;
    else if (event.key === 'ArrowLeft') next = (index + buttons.length - 1) % buttons.length;
    else if (event.key === 'Home') next = 0;
    else if (event.key === 'End') next = buttons.length - 1;
    else return;
    event.preventDefault();
    buttons[next].focus();
    setView(buttons[next].dataset.view);
  });
});

$('#play-pause').addEventListener('click', () => setPlaying(!state.playing));
$('#slow-motion').addEventListener('change', (event) => {
  state.slow = event.target.checked;
  announce(state.slow ? 'Slow motion on. Playback is one quarter speed; power demand is unchanged.' : 'Normal playback speed.');
});
$('#power-view').addEventListener('change', (event) => {
  state.powerView = event.target.checked;
  render();
  announce(state.powerView ? 'Shaft power highlighted. Green mechanical connections transfer power; gas follows separate passages.' : 'Airflow view restored.');
});
$('#bypass').addEventListener('change', (event) => {
  state.bypass = event.target.checked;
  $('#bypass-note').hidden = !state.bypass;
  $('#jet-title').textContent = state.bypass ? 'Turbofan engine' : 'Jet engine';
  render();
  announce(state.bypass ? 'Turbofan bypass shown. The fan sends some air around the hot core.' : 'Basic turbojet shown.');
});

for (const engine of ['jet', 'car']) {
  $(`#${engine}-demand`).addEventListener('input', (event) => {
    const value = Number(event.target.value);
    state[`${engine}Demand`] = value / 100;
    $(`#${engine}-demand-value`).textContent = `${value}%`;
    event.target.setAttribute('aria-valuetext', `${value} percent power demand`);
    if (!state.playing) announce('Power demand changed. Press Play to watch the response.');
  });
}

function stepToStroke(index) {
  if (!Number.isFinite(index)) return;
  setPlaying(false, true);
  state.cycle = ((index % 4) + 4) % 4 * 180 + 90;
  if (state.view === 'jet') setView('car');
  render();
  const stroke = strokes[Math.floor(state.cycle / 180)];
  centerDiagram('car', 'cylinder');
  announce(`${stroke.name}. ${stroke.text} ${stroke.valves}. Animation paused.`);
}
document.querySelectorAll('button[data-stroke]').forEach((button) => button.addEventListener('click', () => stepToStroke(Number(button.dataset.stroke))));
$('#next-stroke').addEventListener('click', () => stepToStroke(Math.floor(state.cycle / 180) + 1));

$('#reset').addEventListener('click', () => {
  Object.assign(state, defaults, { slow: false, bypass: false, powerView: false });
  $('#slow-motion').checked = false;
  $('#bypass').checked = false;
  $('#power-view').checked = false;
  $('#bypass-note').hidden = true;
  $('#jet-title').textContent = 'Jet engine';
  for (const engine of ['jet', 'car']) {
    $(`#${engine}-demand`).value = '45';
    $(`#${engine}-demand-value`).textContent = '45%';
    $(`#${engine}-demand`).setAttribute('aria-valuetext', '45 percent power demand');
  }
  selected.jet = 'compressor'; selected.car = 'cylinder';
  setPlaying(!reducedMotion.matches, true);
  setView(state.view);
  announce('Engine lab reset to its starting state.');
});
$('#compare-link').addEventListener('click', () => { setView('compare'); $('#tab-compare').focus({ preventScroll: true }); });
$('#sources-link').addEventListener('click', () => {
  const source = $('#sources');
  source.open = !source.open;
  $('#sources-link').setAttribute('aria-expanded', String(source.open));
  if (source.open) source.scrollIntoView({ behavior: reducedMotion.matches ? 'instant' : 'smooth', block: 'start' });
});
$('#sources').addEventListener('toggle', (event) => $('#sources-link').setAttribute('aria-expanded', String(event.target.open)));

function centerDiagram(engine, part) {
  const scroller = $(`#${engine}-scroll`);
  if (!scroller.classList.contains('is-zoomed')) return;
  const positions = engine === 'jet' ? { intake: 194, compressor: 371, combustor: 602, turbine: 761, nozzle: 920 } : { compressor: 195, intercooler: 228, cylinder: 740, turbine: 365, crankshaft: 740 };
  const x = positions[part] || 550;
  scroller.scrollTo({ left: x / 1100 * scroller.scrollWidth - scroller.clientWidth / 2, behavior: reducedMotion.matches ? 'instant' : 'smooth' });
}
document.querySelectorAll('[data-zoom-engine]').forEach((button) => {
  button.addEventListener('click', () => {
    const engine = button.dataset.zoomEngine;
    const scroller = $(`#${engine}-scroll`);
    const zoomed = scroller.classList.toggle('is-zoomed');
    button.setAttribute('aria-pressed', String(zoomed));
    button.querySelector('span').textContent = zoomed ? 'Fit whole diagram' : 'Enlarge diagram';
    if (zoomed) requestAnimationFrame(() => centerDiagram(engine, selected[engine]));
    else scroller.scrollLeft = 0;
    announce(zoomed ? 'Diagram enlarged. Swipe horizontally to explore; choosing a component brings it into view.' : 'Whole engine diagram shown.');
  });
});
window.addEventListener('hashchange', () => setView(location.hash.slice(1), false));
reducedMotion.addEventListener('change', (event) => { if (event.matches) setPlaying(false); });

// Qualitative response model. These time constants and normalized levels are
// teaching choices, not an engine simulation or measured vehicle specifications.
let previous = performance.now();
function frame(now) {
  const dt = Math.min(.05, Math.max(0, (now - previous) / 1000));
  previous = now;
  if (state.playing && !document.hidden) {
    const step = dt * (state.slow ? .25 : 1);
    state.time += step;
    state.jet += (state.jetDemand - state.jet) * (1 - Math.exp(-step / 1.35));
    state.car += (state.carDemand - state.car) * (1 - Math.exp(-step / .45));
    state.turbo += (state.car - state.turbo) * (1 - Math.exp(-step / 1.9));
    state.cycle = (state.cycle + step * (50 + 100 * state.car)) % 720;
    render();
  }
  requestAnimationFrame(frame);
}

setView(location.hash.slice(1) || 'jet', false);
syncPlayback();
render();
requestAnimationFrame(frame);
