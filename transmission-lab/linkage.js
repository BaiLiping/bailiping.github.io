// Schematic pedal travel. The bite zone is enlarged to make slip visible.
export function engagementAt(travel) {
  return Math.min(1, Math.max(0, (.75 - travel) / .60));
}

export function clearanceAt(travel) {
  return Math.min(1, Math.max(0, (travel - .75) / .25));
}

export function travelAt(engagement) {
  if (engagement >= 1) return 0;
  if (engagement <= 0) return 1;
  return .75 - .60 * engagement;
}

function pressureLabel(grip, travel) {
  if (grip === 0) return 'Bearing loaded';
  if (grip < 1) return 'Partial pressure';
  return travel === 0 ? 'Pressure released' : 'Taking up travel';
}

export function mountLinkage(lab) {
  const svg = lab.querySelector('#linkage-drawing');
  const node = name => svg.querySelector('[data-link-' + name + ']');
  const pedal = node('pedal'), master = node('master'), rod = node('rod');
  const masterFluid = node('master-fluid'), releaseFluid = node('release-fluid');
  const disc = node('disc'), plate = node('plate'), spring = node('spring');
  const bearing = node('bearing'), pressure = node('pressure');
  const plateLeader = node('plate-leader'), discLeader = node('disc-leader'), bearingLeader = node('bearing-leader');
  let position = 0, gearSelected = 1, frame = 0;
  const slide = (part, distance) => part.setAttribute('transform', 'translate(' + distance.toFixed(2) + ' 0)');

  function draw(travel) {
    position = travel;
    const grip = engagementAt(travel);
    // Much of the stroke changes clamping force while the linings still touch.
    const clearance = 16 * clearanceAt(travel);
    const angle = -39 * travel;
    const radians = angle * Math.PI / 180;
    const pinX = 61 - 17 * Math.cos(radians) - 24 * Math.sin(radians);
    const pinY = 43 - 17 * Math.sin(radians) + 24 * Math.cos(radians);
    pedal.setAttribute('transform', 'rotate(' + angle + ' 61 43)');
    rod.setAttribute('d', 'M' + pinX + ' ' + pinY + ' L' + (108 + 17 * travel) + ' 56 H' + (125 + 17 * travel));
    slide(master, 17 * travel);
    slide(bearing, -24 * travel);
    const masterFace = 132 + 17 * travel, releaseFace = 335 - 24 * travel;
    masterFluid.setAttribute('d', 'M' + masterFace + ' 48H182v15H' + masterFace + 'z');
    releaseFluid.setAttribute('d', 'M' + releaseFace + ' 155H369v11H' + releaseFace + 'zM' + releaseFace + ' 192H369v11H' + releaseFace + 'z');
    slide(disc, clearance * .38);
    slide(plate, clearance);
    const edge = 159 + clearance, fingers = 257 - 24 * travel;
    spring.setAttribute('d', 'M' + edge + ' 139 194 145 ' + fingers + ' 162M' + edge + ' 219 194 213 ' + fingers + ' 196');
    plateLeader.setAttribute('d', 'M183 110h-17L' + (156 + clearance) + ' 127');
    discLeader.setAttribute('d', 'M111 252h17L' + (141 + clearance * .38) + ' 229');
    bearingLeader.setAttribute('d', 'M212 252h39L' + (264 - 24 * travel) + ' 205');
    svg.style.setProperty('--pressure', (100 * travel).toFixed(1) + '%');
    svg.dataset.grip = grip === 0 ? 'open' : grip === 1 ? 'clamped' : 'slipping';
    pressure.textContent = pressureLabel(grip, travel);
  }

  function move(travel, gear, animate) {
    cancelAnimationFrame(frame);
    const start = position, shift = gear !== gearSelected;
    gearSelected = gear;
    if (!animate || travel === start) { draw(travel); return; }
    const pressFirst = shift && travel > start, selectFirst = shift && travel < start;
    const smooth = n => { const t = Math.min(1, Math.max(0, n)); return t * t * (3 - 2 * t); };
    const time = performance.now();
    function advance(now) {
      const t = Math.min(1, (now - time) / 1100);
      const foot = smooth(pressFirst ? t / .49 : selectFirst ? (t - .55) / .45 : t);
      draw(start + (travel - start) * foot);
      if (t < 1) frame = requestAnimationFrame(advance);
    }
    frame = requestAnimationFrame(advance);
  }
  return { move, get travel() { return position; } };
}
