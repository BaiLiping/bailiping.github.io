// Calibrated rev3 poses for 2003 Honda CR-V EX 4WD interior photographs.
// Coordinates use the exported 1:18 glTF Y-up frame (+X forward, +Y up, -Z left, +Z right).
const cabin = /^CRV_(?:Seat|DoorCard|Pedal|SteeringWheel|InstrumentCluster|Dashboard|ClimateControl|AudioUnit|CenterTray|Shifter|ParkingBrake|PicnicTable|CargoFloor|Int_FloorCarpet|Mirror_Cabin)/;
const dashAndWheel = /^CRV_(?:Dashboard|InstrumentCluster|ClimateControl|AudioUnit|Shifter|ParkingBrake|SteeringWheel|Pedal|DoorCard_FL|Int_FloorCarpet)/;
const dash = /^CRV_(?:Dashboard|InstrumentCluster|ClimateControl|AudioUnit|Shifter|ParkingBrake|Int_FloorCarpet)/;
const clusterOnly = /^CRV_(?:Dashboard|InstrumentCluster)/;
const doorCardFL = /^CRV_DoorCard_FL/;

export const interiorPhotoViews = {
  "Driver cockpit and steering wheel": { theta: 3.072238, phi: 0.978201, distance: 49.207644, fov: 30.0, center: [23.732567, 54.399692, -18.982299], roll: -0.034494, isolate: dashAndWheel, background: 0x181b20 },
  "Full-width dashboard": { theta: 3.073442, phi: 0.901384, distance: 70.211044, fov: 36.0, center: [22.917335, 47.589556, -7.870304], roll: -0.073935, isolate: dashAndWheel, background: 0x181b20 },
  "Center stack, 4AT shifter and dash handbrake": { theta: 2.559218, phi: 1.470044, distance: 96.73747, fov: 24.0, center: [27.655743, 46.707059, -3.811944], roll: 0.014243, isolate: dash, background: 0x181b20 },
  "4-speed automatic dash shifter": { theta: -2.98, phi: 1.48, distance: 30.5, fov: 34.0, center: [29.5, 58.2, -9.2], roll: 0.0, isolate: dash, background: 0x181b20 },
  "Vertical dash-integrated parking brake": { theta: -2.8, phi: 1.48, distance: 28.5, fov: 34.0, center: [29.8, 41.2, -6.2], roll: 0.0, isolate: dash, background: 0x181b20 },
  "Walk-through folding center tray": { theta: -1.6, phi: 0.52, distance: 23.5, fov: 42.0, center: [5.2, 34.0, 0.2], roll: 0.18, isolate: cabin, background: 0x181b20 },
  "Instrument cluster gauges": { theta: -3.1416, phi: 1.56, distance: 18.2, fov: 26.0, center: [32.8, 56.8, -21.2], roll: 0.0, isolate: clusterOnly, background: 0x181b20 },
  "Driver door card (Saddle)": { theta: 1.57, phi: 1.42, distance: 27.5, fov: 34.0, center: [16.5, 45.8, -40.5], roll: -0.28, isolate: doorCardFL, background: 0x181b20 },
  "Cargo bay and stowed picnic floor": { theta: -2.76529, phi: 1.415158, distance: 162.554549, fov: 28.0, center: [-81.093932, 49.380831, 14.42459], roll: 0.01623, panels: {"tailgate": 1} },
};
