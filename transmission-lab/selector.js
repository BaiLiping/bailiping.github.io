import { gearNames } from './sequences.js?v=bac112378e26';

const laneOf = gear => Math.floor((gear - 1) / 2);
const endOf = gear => gear % 2 ? -1 : 1;
function ramp(value, start, end) {
  const t = Math.max(0, Math.min(1, (value - start) / (end - start)));
  return t * t * (3 - 2 * t);
}

// A forward gear has two coordinates: a rail and a signed throw along that rail.
// The lever can change rails only while every sleeve is centered.
export function selectorPose(from, to, progress) {
  const oldRail = laneOf(from), newRail = laneOf(to);
  const cross = ramp(progress, .31, .57);
  const lane = oldRail + (newRail - oldRail) * cross;
  const rail = cross < .5 ? oldRail : newRail;
  let stroke = endOf(from), stage = 'engaged';
  if (from !== to) {
    if (progress < .28) {
      stroke *= 1 - ramp(progress, .02, .28);
      if (progress > .02) stage = 'release';
    } else if (progress <= .62) {
      stroke = 0;
      stage = oldRail !== newRail && cross > 0 && cross < 1 ? 'cross' : 'neutral';
    } else {
      // At the cone the sleeve pauses while friction equalizes the speeds.
      stroke = endOf(to) * (.52 * ramp(progress, .62, .73) + .48 * ramp(progress, .83, .98));
      stage = progress < .83 ? 'sync' : progress < .98 ? 'lock' : 'engaged';
    }
  }
  const gear = Math.abs(stroke) >= .88 ? rail * 2 + (stroke < 0 ? 1 : 2) : 0;
  return { lane, rail, stroke, stage, gear };
}

export function mountSelector(lab, onPose) {
  const drawing = lab.querySelector('#stick-drawing');
  const find = id => lab.querySelector('#' + id);
  const knob = find('stick-knob'), selectPins = find('select-pins'), shiftPins = find('shift-pins');
  const selectCore = drawing.querySelector('.select-cable .cable-core'), shiftCore = drawing.querySelector('.shift-cable .cable-core');
  const selectArm = find('select-arm'), shiftArm = find('shift-arm'), finger = find('selector-finger');
  const rails = [...drawing.querySelectorAll('.rail-pair')].map(group => ({ group, bar: group.querySelector('use') }));
  const pair = find('stick-pair'), leftGear = find('stick-left-gear'), rightGear = find('stick-right-gear');
  const fork = find('fork-and-sleeve'), forkLeader = find('fork-leader'), sleeveLeader = find('sleeve-leader');
  let frame = 0, position = 0, currentPose, previousFrom = 1, previousTo = 2, previousRail = -1;
  const translate = (part, x, y = 0) => part.setAttribute('transform', 'translate(' + x + ' ' + y + ')');

  function draw(from, to, progress) {
    position = progress;
    const pose = selectorPose(from, to, progress);
    currentPose = pose;
    const { lane, rail, stroke, gear, stage } = pose;
    const selectTravel = (lane - 1) * 9, shiftTravel = stroke * 9;
    translate(knob, 35 + 32 * lane, 82 + 24 * stroke);
    onPose?.(pose);
    translate(selectPins, selectTravel);
    translate(shiftPins, shiftTravel);
    selectCore.setAttribute('d', 'M118 68L' + (138 + selectTravel) + ' 60H' + (284 + selectTravel));
    shiftCore.setAttribute('d', 'M118 90L' + (138 + shiftTravel) + ' 98H' + (284 + shiftTravel));
    const x = 367 + stroke * 12, y = 48 + lane * 31;
    selectArm.setAttribute('d', 'M' + (284 + selectTravel) + ' 60L307 ' + y + 'H' + (x - 24));
    shiftArm.setAttribute('d', 'M' + (284 + shiftTravel) + ' 98L' + (307 + shiftTravel) + ' 79L' + (x - 24) + ' ' + y);
    finger.setAttribute('d', 'M' + (x - 24) + ' ' + y + 'l9 -8h15v8');
    rails.forEach(({ group, bar }, i) => {
      group.toggleAttribute('data-active', lane === i);
      translate(bar, rail === i ? x : 367, 51 + 31 * i);
    });
    const slide = 29 * stroke;
    translate(fork, slide);
    forkLeader.setAttribute('d', 'M272 164h-15L' + (242 + slide) + ' 184');
    sleeveLeader.setAttribute('d', 'M382 276h-27L' + (260 + slide) + ' 267');
    drawing.dataset.coupled = gear ? (gear % 2 ? 'left' : 'right') : 'none';
    drawing.dataset.toward = to % 2 ? 'left' : 'right';
    drawing.dataset.stage = stage;
    if (previousRail !== rail) {
      previousRail = rail;
      const left = gearNames[2 * rail + 1], right = gearNames[2 * rail + 2];
      pair.textContent = left + ' and ' + right + ' on one shaft';
      leftGear.textContent = left + ' gear'; rightGear.textContent = right + ' gear';
    }
  }

  function move(from, to, progress, duration) {
    cancelAnimationFrame(frame);
    const start = position;
    const sameShift = from === previousFrom && to === previousTo;
    previousFrom = from; previousTo = to;
    if (!duration || !sameShift || start === progress) { draw(from, to, progress); return; }
    const time = performance.now();
    function advance(now) {
      const t = Math.min(1, (now - time) / duration);
      draw(from, to, start + (progress - start) * t);
      if (t < 1) frame = requestAnimationFrame(advance);
    }
    frame = requestAnimationFrame(advance);
  }
  return { move, get progress() { return position; }, get pose() { return currentPose; } };
}
