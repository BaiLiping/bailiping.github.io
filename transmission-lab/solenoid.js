import { gearNames } from './sequences.js?v=bac112378e26';

// A normally-low regulator can feed or drain the clutch during a handover.
const readings = {
  apply: ['Rising', 'Rising'],
  hold: ['Steady', 'Held'],
  release: ['Falling', 'Falling'],
  vented: ['Off', 'Low'],
};

export function mountSolenoid(figure, circuits) {
  const drawing = figure.querySelector('#solenoid-drawing');
  const current = drawing.querySelector('#valve-current'), pressure = drawing.querySelector('#valve-pressure');
  const pistonLeader = drawing.querySelector('.piston-leader'), context = figure.querySelector('#valve-context');

  return function show(step, animate) {
    const circuit = circuits.querySelector('input:checked').value;
    const odd = circuit === 'K1';
    const grip = odd ? step.oddClutch : step.evenClutch;
    const incoming = step.pdkTo % 2 ? 'K1' : 'K2';
    const mode = grip === 0 ? 'vented' : grip === 1 ? 'hold' : circuit === incoming ? 'apply' : 'release';
    drawing.toggleAttribute('data-animate', animate);
    drawing.dataset.state = mode;
    const [coilCurrent, clutchPressure] = readings[mode];
    current.textContent = coilCurrent;
    pressure.textContent = clutchPressure;
    pistonLeader.setAttribute('d', mode === 'vented' ? 'M365 288h4l3-20' : 'M365 288h13l23-20');
    context.textContent = circuit + ' · ' + gearNames[odd ? step.oddGear : step.evenGear] + ' gear';
  };
}
