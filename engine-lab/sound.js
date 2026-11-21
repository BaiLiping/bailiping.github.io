// sound.js: a synthesised flat-six and turbo. Nothing is recorded: the engine note is a set of oscillators tied to
// the firing frequency (three firings per crank turn), the turbo is a pair of sines that climb with rotor speed,
// and the bypass valve is a burst of filtered noise when the throttle shuts under boost.

/**
 * Build the sound graph. Must be called from a user gesture.
 * Takes nothing. Returns { update, setOn, context }.
 */
export function createSound() {
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const master = ctx.createGain();
  master.gain.value = 0;
  master.connect(ctx.destination);

  // engine: firing fundamental, its half order for the uneven flat-six beat, and one overtone, through a low-pass
  const tone = ctx.createBiquadFilter();
  tone.type = 'lowpass';
  tone.Q.value = 0.8;
  const engineGain = ctx.createGain();
  tone.connect(engineGain).connect(master);
  const voices = [['sawtooth', 1, 0.5], ['square', 0.5, 0.34], ['sawtooth', 1.5, 0.16], ['triangle', 0.25, 0.3]].map(([type, ratio, level]) => {
    const osc = ctx.createOscillator(), g = ctx.createGain();
    osc.type = type;
    g.gain.value = level;
    osc.connect(g).connect(tone);
    osc.start();
    return { osc, ratio };
  });

  // one noise source shared by the intake roar and the bypass valve
  const buffer = ctx.createBuffer(1, ctx.sampleRate * 2, ctx.sampleRate);
  const data = buffer.getChannelData(0);
  for (let i = 0; i < data.length; i++) data[i] = Math.random() * 2 - 1;
  const noise = ctx.createBufferSource();
  noise.buffer = buffer;
  noise.loop = true;
  noise.start();
  const roarFilter = ctx.createBiquadFilter();
  roarFilter.type = 'bandpass';
  roarFilter.Q.value = 0.7;
  const roarGain = ctx.createGain();
  roarGain.gain.value = 0;
  noise.connect(roarFilter).connect(roarGain).connect(master);
  const hissFilter = ctx.createBiquadFilter();
  hissFilter.type = 'highpass';
  hissFilter.frequency.value = 2600;
  const hissGain = ctx.createGain();
  hissGain.gain.value = 0;
  noise.connect(hissFilter).connect(hissGain).connect(master);

  // turbo whistle
  const whistleGain = ctx.createGain();
  whistleGain.gain.value = 0;
  whistleGain.connect(master);
  const whistles = [1, 1.012].map((detune) => {
    const osc = ctx.createOscillator();
    osc.type = 'sine';
    osc.connect(whistleGain);
    osc.start();
    return { osc, detune };
  });

  /**
   * Follow the engine.
   * Takes the engine state from sim.js. Returns nothing.
   */
  function update(sim) {
    const t = ctx.currentTime, k = 0.04;
    const firing = Math.max(8, sim.rpm / 60 * 3);          // six firings in two turns
    for (const v of voices) v.osc.frequency.setTargetAtTime(firing * v.ratio, t, k);
    tone.frequency.setTargetAtTime(260 + firing * 2.2 + 2600 * sim.throttle * (0.4 + sim.boost), t, k);
    const alive = Math.min(1, sim.rpm / 500);
    engineGain.gain.setTargetAtTime(alive * (0.1 + 0.16 * sim.throttle + 0.1 * sim.boost), t, k);
    roarFilter.frequency.setTargetAtTime(500 + sim.rpm * 0.28, t, k);
    roarGain.gain.setTargetAtTime(alive * 0.05 * (0.2 + sim.throttle) * (sim.rpm / sim.spec.redline + sim.boost), t, k);
    for (const w of whistles) w.osc.frequency.setTargetAtTime((1500 + 7600 * Math.min(1.2, sim.turboSpeed)) * w.detune, t, 0.06);
    whistleGain.gain.setTargetAtTime(0.02 * Math.min(1, sim.turboSpeed) ** 2, t, 0.06);
    hissGain.gain.setTargetAtTime(0.13 * sim.bypass + 0.03 * sim.wastegate, t, 0.03);
  }

  /**
   * Turn the sound on or off, with a short fade.
   * Takes a boolean. Returns nothing.
   */
  function setOn(on) {
    if (on) ctx.resume();
    master.gain.setTargetAtTime(on ? 0.55 : 0, ctx.currentTime, 0.08);
  }

  return { update, setOn, context: ctx };
}
