import * as THREE from './vendor/three-r170.js?v=31c84364';

export const VIEWS_URL = './civic-views.json?v=1d0efd7fff';

// Default 1:1 metric studio cameras (glTF Y-up: +X forward, +Y up, -Z driver left, +Z passenger right).
export const DEFAULT_CAMERAS = {
  exterior: {
    theta: -0.78,
    phi: 1.33,
    distance: 7.65,
    fov: 25.0,
    center: [0.17, 0.66, 0.0],
    roll: 0,
    shiftX: 0,
    shiftY: 0,
  },
  rear: {
    theta: 2.36,
    phi: 1.40,
    distance: 7.35,
    fov: 25.0,
    center: [-0.02, 0.65, 0.0],
    roll: 0,
    shiftX: 0,
    shiftY: 0,
  },
  cabin: {
    // Driver-eye at (-0.08, 1.13, -0.34) looking ahead of the red bucket headrest
    // across the Type R steering wheel, digital cluster, center touchscreen,
    // aluminum shift knob, and live planar rearview mirror.
    theta: -2.9745,
    phi: 1.3488,
    radius: 0.863,
    center: [0.75, 0.94, -0.20],
  },
};

export const orbitRadius = pose => pose.distance * Math.tan(pose.fov * Math.PI / 360) / Math.tan(Math.PI / 12);

// Normalize either spherical {theta, phi, distance, center} or explicit {position, target} camera records.
export function normalizeCameraPose(raw) {
  if (!raw) return null;
  const fov = Number(raw.fov ?? 26.0);
  const roll = Number(raw.roll ?? 0.0);
  const shiftX = Number(raw.shiftX ?? 0.0);
  const shiftY = Number(raw.shiftY ?? 0.0);

  let center = Array.isArray(raw.center)
    ? raw.center.map(Number)
    : Array.isArray(raw.target)
      ? raw.target.map(Number)
      : [0, 0.66, 0];

  let theta = raw.theta;
  let phi = raw.phi;
  let distance = raw.distance;

  if (Array.isArray(raw.position)) {
    const dx = Number(raw.position[0]) - center[0];
    const dy = Number(raw.position[1]) - center[1];
    const dz = Number(raw.position[2]) - center[2];
    distance = Math.hypot(dx, dy, dz);
    phi = distance > 1e-6 ? Math.acos(THREE.MathUtils.clamp(dy / distance, -1, 1)) : Math.PI / 2;
    theta = Math.atan2(dz, dx);
  }

  if (!Number.isFinite(theta) || !Number.isFinite(phi) || !Number.isFinite(distance)) {
    return null;
  }

  return {
    theta: Number(theta),
    phi: Number(phi),
    distance: Number(distance),
    fov,
    center: new THREE.Vector3(center[0], center[1], center[2]),
    roll,
    shiftX,
    shiftY,
    radius: orbitRadius({ distance: Number(distance), fov }),
  };
}

function compilePattern(pattern) {
  if (!pattern) return null;
  if (pattern instanceof RegExp) return pattern;
  try {
    return new RegExp(pattern);
  } catch {
    return null;
  }
}

export function buildReferenceViews(views) {
  const cabinParts = /^Civic_(?:Seat|DoorCard|Pedal|SteeringWheel|GaugeCluster|InstrumentCluster|Dash|Infotainment|Vent|Climate|Shifter|Handbrake|CenterConsole|Int_|Carpet|Headliner|Mirror_Cabin)/;

  const map = new Map([
    ['cabin', { label: 'Driver’s view', cabin: true }],
    ['engine', {
      label: 'Engine bay',
      theta: -0.28,
      phi: 0.72,
      distance: 2.25,
      fov: 30,
      radius: orbitRadius({ distance: 2.25, fov: 30 }),
      center: new THREE.Vector3(1.38, 0.78, 0.0),
      panels: { hood: 1.15 },
      group: 'Powertrain',
    }],
    ['drivetrain', {
      label: 'K20C1 & 6-speed transaxle',
      theta: -0.74,
      phi: 1.15,
      distance: 2.78,
      fov: 28,
      radius: orbitRadius({ distance: 2.78, fov: 28 }),
      center: new THREE.Vector3(1.46, 0.62, -0.06),
      isolate: /^(?:Pivot_(?:Drivetrain_6MT|Trans_|Engine_(?:Block|OilPan|Head|ValveCover|Turbo|TurboShield|IntakeManifold)|Spin_(?!RadFan)|Piston_|Conrod_)|Civic_(?:Transaxle_6MT|Eng_(?:Block|ValveCover|TimingCover|Turbo|Head|OilPan|Piston|Conrod|Crank|Cam|Pulley|Belt|Manifold)))/,
      isolateByPivot: true,
      powertrainCutaway: true,
      hideFloor: false,
      group: 'Powertrain',
    }],
    ['seats', {
      label: 'Type R bucket seats',
      theta: -1.06,
      phi: 0.96,
      distance: 2.55,
      fov: 32,
      radius: orbitRadius({ distance: 2.55, fov: 32 }),
      center: new THREE.Vector3(-0.08, 0.78, -0.05),
      panels: { doorL: 1 },
      isolate: cabinParts,
      group: 'Interior',
    }],
    ['hatch', {
      label: 'Rear hatch & wing',
      theta: 2.86,
      phi: 1.06,
      distance: 3.15,
      fov: 30,
      radius: orbitRadius({ distance: 3.15, fov: 30 }),
      center: new THREE.Vector3(-1.42, 0.88, 0.0),
      panels: { hatch: 1 },
      group: 'Exterior',
    }],
  ]);

  for (const view of views) {
    const camSource = view.camera || view;
    const cam = normalizeCameraPose(camSource);
    const photoView = view.photoView || (view.id ? `photo-${view.id}` : null);
    if (!photoView || !cam) continue;

    const pose = {
      ...cam,
      id: view.id || photoView.replace(/^photo-/, ''),
      photoView,
      label: view.label || photoView,
      group: view.group || 'Exterior',
      photo: true,
      image: view.preview || view.image || '',
      originalImage: view.image || '',
      aspect: Number(view.aspect || (view.width && view.height ? view.width / view.height : 1.5)),
      width: Number(view.width || 1280),
      height: Number(view.height || 853),
      panels: view.panels || null,
      steer: Number(view.steer ?? 0),
      hideFloor: Boolean(view.hideFloor),
      hide: compilePattern(view.hide),
      isolate: compilePattern(view.isolate),
      background: typeof view.background === 'number'
        ? view.background
        : typeof view.background === 'string'
          ? parseInt(view.background.replace(/^#|^0x/, ''), 16)
          : undefined,
      cabin: Boolean(view.cabin),
      note: view.note || '',
    };
    map.set(photoView, pose);
  }

  return map;
}

let cachedPromise = null;

export function loadCivicViews() {
  cachedPromise ??= fetch(VIEWS_URL).then(async response => {
    if (!response.ok) throw new Error(`Failed to load ${VIEWS_URL}: ${response.status}`);
    const data = await response.json();
    const views = Array.isArray(data) ? data : (Array.isArray(data.views) ? data.views : []);
    const meta = Array.isArray(data) ? {} : (data.vehicle || {});
    return { views, meta };
  });
  return cachedPromise;
}
