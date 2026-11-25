// Calibrated to the native BaTMR2 photographs. Distances use the exported 1:18 model.
const cabin = /^MR2_(?:Seat|DoorCard|Pedal|SteeringWheel|InstrumentCluster|SwitchPod|Dash|ClimateControl|CassetteStereo|Shifter|Handbrake|CenterConsole|Int_)/;
const seatAndConsole = /^MR2_(?:Seat|Pedal|SteeringWheel|InstrumentCluster|SwitchPod|Dash|ClimateControl|CassetteStereo|Shifter|Handbrake|CenterConsole|Int_)/;
const dashboard = /^MR2_(?:Dash|ClimateControl|CassetteStereo|InstrumentCluster|SwitchPod|CenterConsole|Shifter|Handbrake|Int_)/;
const radio = /^MR2_(?:Dash|ClimateControl|CassetteStereo|InstrumentCluster|SwitchPod|CenterConsole|Int_)/;
const leftDoor = /^MR2_(?:DoorCard_L|Door_Shell_L|Door_Weatherstrip_L|Door_Window_L|Window_Frame_L|Mirror_L)/;
const footwell = /^MR2_(?:Pedal|ClutchControl_PedalLink|Int_(?:Cabin_Floor|FloorMat)|CenterConsole)/;

export const interiorPhotoViews = {
  "Cockpit": {
    theta: -3.1313, phi: 1.06708, distance: 11.78013, fov: 84.61107,
    center: [14.54891, 43.13703, -19.29538], roll: 0.00539, isolate: cabin,
  },
  "Driver’s seat and stitched bolsters": {
    theta: -1.12504, phi: 0.61457, distance: 49.21272, fov: 42.17227,
    center: [-13.32869, 31.17591, -20.4696], roll: 0.43655, isolate: seatAndConsole, background: 0x181b20,
  },
  "Perforated leather and cushion pleats": {
    theta: -1.93381, phi: 0.74424, distance: 14.05454, fov: 64.24579,
    center: [-1.22352, 19.85416, -13.95254], roll: -0.22176, isolate: seatAndConsole, background: 0x181b20,
  },
  "Door trim and recessed handle": {
    theta: 2.28017, phi: 1.60377, distance: 203.50935, fov: 10.18718,
    center: [0.61826, 32.26169, -34.05268], roll: -0.04082, isolate: leftDoor,
  },
  "Pedal box and driver’s footrest": {
    theta: -2.95697, phi: 1.29, distance: 33.27463, fov: 30.45018,
    center: [29.2637, 18.77998, -24.84936], roll: -0.09106, isolate: footwell, background: 0x181b20,
  },
  "Instrument cluster and supercharger indicator": {
    theta: -3.23744, phi: 1.59009, distance: 26.287, fov: 28.56834,
    center: [20.76193, 45.8604, -19.51955], roll: -0.00295, isolate: dashboard, background: 0x181b20,
  },
  "Digital clock and center vent": {
    theta: 2.21992, phi: 1.3538, distance: 51.69227, fov: 14.9926,
    center: [21.12332, 41.18167, -1.74564], roll: -0.04999, isolate: dashboard, background: 0x181b20,
  },
  "Heater controls and temperature slider": {
    theta: -2.84988, phi: 0.91424, distance: 23.12637, fov: 23.52801,
    center: [15.09136, 36.89297, -0.27871], roll: 0.10457, isolate: dashboard, background: 0x181b20,
  },
  "Toyota radio and cassette deck": {
    theta: -2.65187, phi: 1.17819, distance: 40.35781, fov: 15.25273,
    center: [14.18798, 31.31797, 0.29464], roll: 0.0656, isolate: radio, background: 0x181b20,
  },
  "Five-speed shifter and parking brake": {
    theta: 2.26688, phi: 0.76912, distance: 26.37811, fov: 35.02382,
    center: [3.11513, 26.86488, -1.06003], roll: 0.01636, isolate: cabin, background: 0x181b20,
  },
};
