export const gearNames = ['Neutral', '1st', '2nd', '3rd', '4th', '5th', '6th', '7th'];

// One pass up and back down. The gearbox stays in gear across the loop boundary.
const gears = [1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2];
export const driveLength = gears.length * 4;
export const phaseTime = 1850;

export function drivingStep(index) {
  const shift = Math.floor(index / 4), phase = index % 4;
  const from = gears[shift], to = gears[(shift + 1) % gears.length];
  const rising = to > from;
  const manualFrom = Math.min(6, from), manualTo = Math.min(6, to);
  const manualShifts = manualFrom !== manualTo;
  const manualGear = manualShifts ? [manualFrom, manualFrom, 0, manualTo][phase] : 6;
  const manualClutch = manualShifts ? [1, 0, 0, .45][phase] : 1;
  const fromOdd = from % 2 === 1;
  const loaded = fromOdd ? 'K1' : 'K2', incoming = fromOdd ? 'K2' : 'K1';
  const incomingGrip = [0, 0, .5, 1][phase];
  const name = gearNames[to];
  const commands = ['Preselect ' + name + ' on ' + incoming, 'Hold ' + name + ' ready',
    'Apply ' + incoming + ' · release ' + loaded, incoming + ' drives ' + name];
  const stages = manualShifts ? ['In ' + gearNames[from] + ' gear', 'Pedal down', 'Stick through neutral', 'Pedal rising'] :
    ['Manual holds 6th', 'PDK prepares ' + name, 'PDK clutch handover', 'PDK in ' + name];
  return {
    title: (rising ? 'Upshift' : 'Downshift') + ' · ' + gearNames[from] + ' to ' + name,
    label: stages[phase], phase, pdkFrom: from, pdkTo: to, manualGear, manualClutch, manualFrom, manualTo, manualShifts,
    oddGear: fromOdd ? from : to, evenGear: fromOdd ? to : from,
    oddClutch: fromOdd ? 1 - incomingGrip : incomingGrip,
    evenClutch: fromOdd ? incomingGrip : 1 - incomingGrip,
    command: commands[phase], sensors: rising ? 'Accelerating · road speed rising' : 'Braking · road speed falling',
    // An illustrative output-shaft speed kept continuous between successive shifts.
    outputSpeed: .54 + .22 * (from - 1 + (to - from) * phase / 4),
  };
}
