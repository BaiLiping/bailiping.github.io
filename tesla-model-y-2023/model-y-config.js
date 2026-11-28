// Single-source configuration for the 2023 Tesla Model Y viewer.
// Units: 1:18-scale millimetres (1 model unit = 18 real mm; +X forward, +Y up, +Z right).
// Update this file (and the <head> preload token in index.html) when locking final
// paint, GLB revision, calibrated cameras, rig pivot names, or explode tuning.

export const ASSET_VERSION = 'my23-dbfa5aa7-2';
export const VIEWS_URL = `./tesla-views.json?v=${ASSET_VERSION}`;

export const VEHICLE = {
  year: 2023,
  make: 'Tesla',
  model: 'Model Y',
  trim: 'Long Range AWD',
  shortTitle: '2023 Tesla Model Y',
  footerConfig: '2023 Model Y Long Range AWD',
  paintName: 'Deep Blue Metallic',
  paintCode: 'PPSB',
  paintHex: '#1f4064',
  configuratorUrl: 'https://www.tesla.com/modely/design',
  footerCredit: 'Images © Tesla, Inc.',
  defaultPhotoWidth: 2048,
  defaultPhotoHeight: 1152,
  unitMm: 18,
};

export const SCENE_TUNING = {
  bgHex: 0xf1f2f3,
  studioFogDensity: 0.000135,
  exteriorEnvironmentGain: 1.65,
  exteriorRimGain: 0.006,
  exteriorKeyGain: 0.015,
  keyLightPos: [185, 340, -85],
  rimLightPos: [-110, 250, 235],
  shadowExtent: 235,
  floorFadeNear: 165.0,
  floorFadeFar: 1560.0,
  // Floor UV space: 2048 px = 2400 model units (0.8533 px/unit).
  // Model Y WB = 2890 mm -> half-WB = 80.28 u -> 68.5 px; Track = 1636 mm -> half-track = 45.44 u -> 38.8 px.
  contactShadows: {
    axles: [-69, 69],
    sides: [-39, 39],
    tireRx: 16,
    tireRy: 8.5,
    tireOpacity: 0.92,
    chassis: [
      [1020, 1029, 158, 71, 0.58],
      [1024, 1024, 132, 53, 0.78],
      [1024, 1024, 98, 43, 0.64],
    ],
  },
  zoomExterior: [130, 860],
  zoomCabin: [20, 48],
  pullParkDist: 40,
  pullParkSmallDist: 3.2,
};

// Exterior, rear, top and driver's-eye camera presets (1:18 model units).
// Exterior view will be locked to calibrated view-00 once studio cameras are fitted.
export const CAMERAS = {
  exteriorView: { theta: -0.75, phi: 1.44, radius: 325, fov: 22.0 },
  focusExterior: [0.0, 44.0, 0.0],
  topCenter: [0.0, 45.0, 0.0],
  topRadius: 330,
  rearCenter: [-28.0, 44.0, 0.0],
  rearTheta: -2.69,
  rearPhi: 1.51,
  rearRadius: 255,
  rearFov: 15.0,
  // Driver eye near (-7.0, 68.0, -21.1) looking at dash/15" screen center.
  focusInterior: [36.0, 61.0, -15.0],
  cabinRadius: 44,
  cabinEntryTheta: -3.00,
  cabinEntryPhi: 1.41,
};

// Articulated closures (glTF Y-up: +X forward, +Y up, -Z driver left, +Z passenger right).
export const PANEL_SPECS = {
  doorFL:   { node: 'Pivot_Door_FL',     axis: 'y', angle: -1.05,  label: 'front-left door' },
  doorFR:   { node: 'Pivot_Door_FR',     axis: 'y', angle:  1.05,  label: 'front-right door' },
  doorRL:   { node: 'Pivot_Door_RL',     axis: 'y', angle: -1.05,  label: 'rear-left door' },
  doorRR:   { node: 'Pivot_Door_RR',     axis: 'y', angle:  1.05,  label: 'rear-right door' },
  hood:     { node: 'Pivot_Front_Lid',   axis: 'z', angle:  1.152, label: 'frunk lid' },
  liftgate: { node: 'Pivot_Liftgate',    axis: 'z', angle: -1.100, label: 'liftgate' },
  charge:   { node: 'Pivot_Charge_Port', axis: 'x', angle:  1.484, label: 'charge port' },
};

// Compact exploded offsets per Pivot_* in the pivot's parent frame (1:18 units).
// L = 263.94 u, H = 90.22 u. Keeps chassis anchored (0) and total envelope <= 2.0 L x 1.0 L x 1.3 L.
export const EXPLODE_PIVOT_OFFSETS = {
  Pivot_Drivetrain_Chassis: [0, 0, 0],
  Pivot_Battery_Pack:       [0, -3, 0],
  Pivot_Body_Assembly:      [0, 130, 0],
  Pivot_Front_Lid:          [34, 38, 0],
  Pivot_Liftgate:           [-56, 34, 0],
  Pivot_Charge_Port:        [-14, 10, -18],
  Pivot_Door_FL:            [12, -16, -86],
  Pivot_Door_RL:            [-28, -26, -86],
  Pivot_Door_FR:            [22, 10, 90],
  Pivot_Door_RR:            [-24, 4, 90],
  Pivot_Front_Drive:        [68, 12, 0],
  Pivot_Rear_Drive:         [-78, 14, 0],
  Pivot_Interior_Cabin:     [132, 62, -62],
  Pivot_Corner_FL:          [6, 0, -42],
  Pivot_Corner_FR:          [6, 0, 44],
  Pivot_Corner_RL:          [-8, 0, -42],
  Pivot_Corner_RR:          [-8, 0, 44],
};

// Sub-assembly offsets per mesh inside its parent Pivot_* (1:18 units).
export function computePartOffset(meshName, parentName) {
  if (parentName.startsWith('Pivot_WheelSpin')) {
    const side = parentName.endsWith('L') ? -1 : 1;
    if (/Gemini_?Cover|Aero_?Cover|WheelCover/.test(meshName)) return [0, 0, side * 60];
    if (/CenterCap|Lug/.test(meshName)) return [0, 0, side * 48];
    if (/Tire/.test(meshName)) return [0, 0, side * 38];
    return null;
  }
  if (parentName.startsWith('Pivot_Corner_')) {
    const side = parentName.endsWith('L') ? -1 : 1;
    const front = parentName.includes('_F') ? 1 : -1;
    if (/Caliper/.test(meshName)) return [front * 6, 8, side * 12];
    return null;
  }
  if (parentName === 'Pivot_Battery_Pack') {
    if (/Ancillary_?Cover|Penthouse_?Cover|Service_?Cover/.test(meshName)) return [-8, 42, 0];
    if (/Battery_?Lid|Pack_?Lid/.test(meshName)) return [0, 32, 0];
    return null;
  }
  if (parentName === 'Pivot_Front_Drive') {
    if (/Case|Housing_?Top|Inverter_?Cover/.test(meshName)) return [12, 20, 0];
    return null;
  }
  if (parentName === 'Pivot_Rear_Drive') {
    if (/Case|Housing_?Top|Inverter_?Cover/.test(meshName)) return [-14, 22, 0];
    return null;
  }
  if (parentName === 'Pivot_Body_Assembly') {
    if (/Headlamp|Headlight|Front_?Fascia|Front_?Bumper|Front_?Grille|Fog/.test(meshName)) return [24, 0, 0];
    if (/Taillamp|Taillight|Rear_?Fascia|Rear_?Bumper|Diffuser/.test(meshName)) return [-24, 0, 0];
    if (/Glass_Windshield|Windshield|Glass_Roof|Roof_?Glass|Wiper/.test(meshName)) return [0, 22, 0];
    return null;
  }
  if (parentName.startsWith('Pivot_Door_') && /Glass_Door|Door_Glass/.test(meshName)) {
    return [0, 16, 0];
  }
  if (parentName === 'Pivot_Interior_Cabin') {
    if (/SteeringWheel|Steering_/.test(meshName)) return [4, 12, -16];
    if (/Dash_|Display|Touchscreen|Wood_?Trim/.test(meshName)) return [16, 8, 0];
    if (/Seat_.*Cushion/.test(meshName)) return [0, 10, 0];
    if (/Frunk_?Tub|Frunk_?Liner/.test(meshName)) return [32, 12, 0];
    return null;
  }
  return null;
}

// Mesh & material classification patterns for x-ray, shadows, glazing, and enclosed EV internals.
export const NAME_PATTERNS = {
  rigRoot: 'ModelY_Interactive_Rig',
  exteriorRe: /Body_Shell|Door_Shell|Front_Lid|Hood|Liftgate|Tailgate|Charge_?Port|Front_?Fascia|Rear_?Fascia|Fender|Rocker|Side_Skirt|Diffuser|Spoiler|Mirror|Door_Handle|Wiper|Glass|Windshield|Roof_?Glass|Headlamp|Headlight|Taillamp|Taillight|Fog_?Lamp|Reflector|Badge|Emblem|Camera_?Repeater|License_?Plate|Battery_?Lid|Ancillary_?Cover|Drive_?Case|Inverter_?Cover/,
  shadowCasterRe: /Body_Shell|Door_Shell|Front_Lid|Hood|Liftgate|Tailgate|Front_?Fascia|Rear_?Fascia|Fender|Tire|Gemini_?Cover|Aero_?Cover|Mirror|Underbody|Battery_?Tray|Subframe|Drive/,
  shadowReceiverRe: /Floor|Carpet|Dash|Seat|Console|Frunk|Cargo|Grille|Intake|Rotor|Caliper|Spoke|Alloy|Battery|Module|Subframe/,
  glassRe: /Glass|Windshield|Roof_?Glass|Lens|Headlamp_Clear|Headlight_Clear/,
  doorGlassRe: /Glass_Door|Door_Glass/,
  headlightLensMat: 'Mat_Headlight_Clear_Lens',
  paintMatRe: /^Mat_(Deep_Blue_Metallic|Body_Paint|ModelY_Paint)/,
  glassMatRe: /^Mat_(Automotive_Glass|Windshield_Glass|Roof_Glass|Rear_Privacy_Glass|Door_Glass|Quarter_Glass)$/,
  roofGlassMatRe: /^Mat_Roof_Glass$/,
  rearGlassMatRe: /^Mat_(Rear_Privacy_Glass|Quarter_Glass)$/,
  rearMirrorMatRe: /^Mat_(Cabin_Rearview|Mirror_Interior)$/,
  enclosedDrivetrainRe: /Battery_?Module|Cell_?Pack|Busbar|Contactor|Reduction_?Gear|Pinion|Diff_?Gear|Stator|Rotor_?Core/,
  chargeHitRe: /Charge_?Port|Charge_?Door|Charge_?Inlet|Charge_?Flap/,
};

let viewsRequest;
export function loadViews() {
  viewsRequest ??= fetch(VIEWS_URL).then(response => {
    if (!response.ok) throw new Error('Reference photos unavailable');
    return response.json();
  }).catch(error => {
    viewsRequest = undefined;
    throw error;
  });
  return viewsRequest;
}

// Hover callout part catalogue — factual 2023 Tesla Model Y descriptions.
export const PART_CATALOG = [
  [/Charge_?Port|Charge_?Flap|Charge_?Door/, 'Charge port door', 'integrated into left taillight · click to open or close'],
  [/Charge_?Inlet/, 'North American charge inlet', 'AC & DC fast-charging port behind left taillight'],
  [/Front_Lid|Hood/, 'Front luggage lid (frunk)', 'click to open or close · rear-hinged aluminum hood'],
  [/Frunk/, 'Front trunk compartment', 'sealed under-hood storage tub and cowl apron'],
  [/Liftgate|Tailgate/, 'Power rear liftgate', 'click to open or close · roof-hinged hatchback with rear glass'],
  [/Door_Shell_FL|Door_FL/, 'Left front frameless door', 'click to open or close · flush handle and side mirror'],
  [/Door_Shell_FR|Door_FR/, 'Right front frameless door', 'click to open or close · flush handle and side mirror'],
  [/Door_Shell_RL|Door_RL/, 'Left rear frameless door', 'click to open or close · tinted privacy glazing'],
  [/Door_Shell_RR|Door_RR/, 'Right rear frameless door', 'click to open or close · tinted privacy glazing'],
  [/Glass_Door|Door_Glass/, 'Frameless door glass', 'swings open with the door'],
  [/Glass_Roof|Roof_?Glass/, 'Panoramic all-glass roof', 'tinted laminated overhead roof panel'],
  [/Glass_Windshield|Windshield/, 'Acoustic laminated windshield', 'houses forward-facing Autopilot camera enclosure'],
  [/Glass_Quarter|Quarter_Glass/, 'Fixed rear quarter glass', 'tinted C-pillar daylight opening'],
  [/Mirror_Cabin|Rearview/, 'Interior rearview mirror', 'windshield-mounted cabin mirror'],
  [/Mirror/, 'Side wing mirror', 'door-mounted aero housing with black stalk'],
  [/Door_Handle/, 'Flush mechanical door handle', 'satin black push-to-present exterior latch'],
  [/Camera_?Repeater|Fender_?Camera/, 'Side repeater camera', 'fender-mounted rearward camera and turn indicator'],
  [/Wiper/, 'Windshield wiper', 'recessed cowl aero wiper blades'],
  [/Headlamp|Headlight/, 'LED matrix headlamp', 'swept housing with upper daytime running light signature'],
  [/Fog_?Lamp/, 'LED front fog lamp', 'recessed lower bumper corner light'],
  [/Taillamp|Taillight/, 'Split LED taillight', 'divided across rear quarter panel and liftgate'],
  [/Front_?Fascia|Front_?Bumper/, 'Front bumper fascia', 'grille-free EV nose with lower radiator intake'],
  [/Rear_?Fascia|Rear_?Bumper|Diffuser/, 'Rear bumper fascia', 'painted upper cover with matte black lower valance'],
  [/Badge|Emblem|Wordmark/, 'Tesla emblem', 'front hood T-emblem and liftgate dual-motor badging'],
  [/Body_Shell|Fender|Rocker/, '2023 Model Y body shell', 'Deep Blue Metallic (PPSB)'],
  [/Gemini_?Cover|Aero_?Cover/, '19-inch Gemini aero wheel cover', 'seven-vane aerodynamic cover over alloy wheel'],
  [/Alloy|Rim|Spoke/, '19-inch Gemini alloy wheel', 'seven-split-spoke aluminum wheel beneath the aero cover'],
  [/Tire/, 'All-season tire', '255/45 R19 square wheel and tire fitment'],
  [/Caliper/, 'Disc brake caliper', 'ventilated disc brake assembly'],
  [/Rotor|Disc/, 'Ventilated brake rotor', 'cast-iron disc on hub carrier'],
  [/Front_?Subframe|Front_?Wishbone|Front_?Suspension|Knuckle_F|Damper_F|Strut/, 'Front double-wishbone suspension', 'upper wishbone, lower links and coil-over damper'],
  [/Rear_?Subframe|Rear_?Mega|Rear_?Control|Rear_?Suspension|Knuckle_R|Spring_R|Damper_R/, 'Rear multi-link suspension', 'cast rear subframe with separate coil spring and damper'],
  [/Battery_?Lid|Pack_?Lid/, 'Battery pack top cover', 'sealed enclosure lid over the underfloor modules'],
  [/Ancillary|Penthouse/, 'Rear electronics bay (penthouse)', 'contactors, pyro-fuse and DC-DC / PCS housing under rear seat'],
  [/Battery_?Module|Cell_?Pack/, 'High-voltage battery module', 'one of four longitudinal 2170 modules inside the pack tray'],
  [/Battery_?Tray|Battery_?Pack/, 'Underfloor high-voltage battery pack', 'floor-mounted structural tray between front and rear axles'],
  [/High_?Voltage|HV_?Harness|HV_?Cable/, 'High-voltage orange harness', 'routes DC power from battery bay to front and rear inverters'],
  [/Front_?Drive|FrontDrive|Induction/, 'Front induction drive unit', 'AC induction motor, inverter, fixed reduction gear and halfshafts'],
  [/Rear_?Drive|RearDrive|PM_?Motor/, 'Rear permanent-magnet drive unit', 'PM synchronous motor, inverter, reduction gearbox and halfshafts'],
  [/Reduction_?Gear|Diff_?Gear|Pinion/, 'Single-speed reduction gearbox', 'helical reduction gearset and open differential'],
  [/Underbody|Aero_?Shield/, 'Flat aerodynamic underbody', 'front and rear undertray shields'],
  [/Steering/, 'Three-spoke steering wheel', 'twin thumb scroll wheels with column stalks'],
  [/Display|Touchscreen/, '15-inch central touchscreen', 'landscape display for instruments, navigation and controls'],
  [/Wood_?Trim|Dash/, 'Minimalist horizontal dashboard', 'open-pore wood accent with concealed full-width air vent'],
  [/Console/, 'Center console', 'dual wireless charging pad and sliding storage compartment'],
  [/Seat_Front|Driver_?Seat|Passenger_?Seat/, 'Front bolstered seat', 'five-seat vegan leather cabin upholstery'],
  [/Seat_Rear|Rear_?Bench/, 'Second-row 60/40 bench seat', 'three-passenger rear bench with folding backrest'],
  [/Cargo/, 'Rear luggage compartment', 'carpeted load floor with under-floor storage well'],
  [/Pedal/, 'Accelerator and brake pedals', 'driver footwell controls'],
];
