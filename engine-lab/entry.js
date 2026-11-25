const comparison = document.getElementById('engine-comparison');
const turboNote = comparison.querySelector('[data-note="turbo"]');
const scNote = comparison.querySelector('[data-note="sc"]');
const notes = {
  assembled: [turboNote.textContent, scNote.textContent],
  intake: [
    'Fresh air enters the compressor. Follow the amber stream up the charge pipe to the intercooler. The cooled air, in blue, then travels to the cylinders.',
    'Follow the shorter route from the air cleaner through the airflow meter and throttle. Six runners carry the air from the plenum to the cylinders.',
  ],
  exhaust: [
    'Follow the headers to the turbine. Exhaust spins the turbine and drives the intake compressor on the same shaft. The wastegate can divert some exhaust around the turbine.',
    'Follow the headers under each cylinder bank. They gather the exhaust and route it through the silencer to the tailpipe.',
  ],
  gas: [
    'The red exhaust stream spins the turbine. On the other side of its shaft, the compressor sends hot air (amber) to the intercooler. Cooled air (blue) flows to all six cylinders.',
    'The pistons draw fresh air (blue) through six runners. Exhaust (red) leaves the cylinders and passes through the silencers to the single tailpipe.',
  ],
  cutaway: [
    'Each Turbo cylinder has a 97 mm bore. Run the engines to watch its piston travel 74.4 mm between the ends of its stroke.',
    'Each SC cylinder has a 95 mm bore and a shorter, 70.4 mm piston stroke. Both dimensions contribute to its smaller displacement.',
  ],
  exploded: [
    'Find the intercooler above the engine, then the compressor, turbine and wastegate around the exhaust. These parts form the boost system.',
    'Find the airbox and runners above the two banks of cylinders. Between the banks is the crankshaft shared by all six pistons.',
  ],
};
comparison.addEventListener('comparemodechange', ({ detail }) => {
  [turboNote.textContent, scNote.textContent] = notes[detail.mode];
});
try {
  const { mountComparison } = await import('./compare.js?v=e816bcea50');
  await mountComparison(comparison);
  for (const controls of comparison.querySelectorAll('[inert]')) controls.inert = false;
} catch (error) {
  console.error('Engine comparison could not start', error);
  for (const source of comparison.querySelectorAll('.model-frame source')) source.remove();
  comparison.dataset.failed = 'true';
  comparison.querySelector('[data-compare-status]').textContent =
    'Showing still views. To rotate and inspect the engines, enable WebGL 2 and reload the page.';
}
