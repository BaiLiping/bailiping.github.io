// Calibrated to the native BaTMR2 photographs. Distances use the exported 1:18 model.
const cabin = /^MR2_(?:Seat|DoorCard|Pedal|SteeringWheel|InstrumentCluster|SwitchPod|Dash|ClimateControl|CassetteStereo|Shifter|Handbrake|CenterConsole|Int_)/;
const seatAndConsole = /^MR2_(?:Seat|Pedal|SteeringWheel|InstrumentCluster|SwitchPod|Dash|ClimateControl|CassetteStereo|Shifter|Handbrake|CenterConsole|Int_)/;
const dashboard = /^MR2_(?:Dash|ClimateControl|CassetteStereo|InstrumentCluster|SwitchPod|CenterConsole|Shifter|Handbrake|Int_)/;
const radio = /^MR2_(?:Dash|ClimateControl|CassetteStereo|InstrumentCluster|SwitchPod|CenterConsole|Int_)/;
const leftDoor = /^MR2_(?:DoorCard_L|Door_Shell_L|Door_Weatherstrip_L|Door_Window_L|Window_Frame_L|Mirror_L)/;
const footwell = /^MR2_(?:Pedal|ClutchControl_PedalLink|Int_(?:Cabin_Floor|FloorMat)|CenterConsole)/;

export const interiorPhotoViews = {
  "Cockpit": {
    theta: -3.1472, phi: 1.00796, distance: 12.13721, fov: 81.67279,
    center: [14.18088, 43.23456, -19.22], roll: -0.00716, isolate: cabin,
  },
  "Driver’s seat and stitched bolsters": {
    theta: -1.12504, phi: 0.61457, distance: 49.21272, fov: 42.17227,
    center: [-13.32869, 31.17591, -20.4696], roll: 0.43655, isolate: seatAndConsole, background: 0x181b20,
  },
  "Perforated leather and cushion pleats": {
    theta: -1.86063, phi: 0.57991, distance: 16.97281, fov: 53.81936,
    center: [-0.71736, 21.19829, -13.6978], roll: -0.261, isolate: seatAndConsole, background: 0x181b20,
  },
  "Door trim and recessed handle": {
    theta: 2.21475, phi: 1.56203, distance: 181.57718, fov: 12.0,
    center: [2.7427, 33.07488, -37.54012], roll: -0.01291, isolate: leftDoor,
  },
  "Pedal box and driver’s footrest": {
    theta: -2.95697, phi: 1.29, distance: 33.27463, fov: 30.45018,
    center: [29.2637, 18.77998, -24.84936], roll: -0.09106, isolate: footwell, background: 0x181b20,
  },
  "Instrument cluster and supercharger indicator": {
    theta: -3.24367, phi: 1.44, distance: 28.14589, fov: 26.4145,
    center: [20.35026, 46.01534, -20.0474], roll: -0.00922, isolate: dashboard, background: 0x181b20,
  },
  "Digital clock and center vent": {
    theta: 2.24794, phi: 1.3229, distance: 53.15668, fov: 14.99902,
    center: [21.08906, 40.93203, -1.60126], roll: -0.0387, isolate: dashboard, background: 0x181b20,
  },
  "Heater controls and temperature slider": {
    theta: -2.82362, phi: 0.85965, distance: 23.78928, fov: 22.29133,
    center: [14.92169, 36.94352, -0.36007], roll: 0.13479, isolate: dashboard, background: 0x181b20,
  },
  "Toyota radio and cassette deck": {
    theta: -2.66672, phi: 1.1616, distance: 39.7731, fov: 15.58291,
    center: [14.21547, 31.26803, 0.32845], roll: 0.05261, isolate: radio, background: 0x181b20,
  },
  "Five-speed shifter and parking brake": {
    theta: 2.22403, phi: 0.82916, distance: 29.48862, fov: 31.86846,
    center: [2.90502, 27.35851, -0.8869], roll: 0.01153, isolate: cabin, background: 0x181b20,
  },
};
