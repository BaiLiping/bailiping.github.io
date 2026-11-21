// sim.js: a small running model of the engine. It is not a thermodynamic simulation: it turns the throttle into
// revs, the exhaust into turbo speed, turbo speed into boost, and opens the wastegate at the factory boost
// limit, with time constants chosen so that boost arrives late, the way the car is known for.

const SLOW = 1 / 40;            // the crank is shown at one fortieth of its real speed

/**
 * Create the engine state.
 * Takes the spec block of engine930.json. Returns the state object that step() advances.
 */
export function createSim(spec) {
  return {
    spec, rpm: spec.idle, throttle: 0, crank: 0, turbo: 0, turboSpeed: 0, boost: 0, wastegate: 0, bypass: 0, airflow: 0,
    running: true, slow: SLOW,
  };
}

/**
 * Advance the engine by one frame.
 * Takes the state, the time step in seconds and the throttle demand (0 to 1). Returns nothing; the state holds
 * the new revs, boost, crank angle and the positions of the wastegate, bypass valve and airflow plate.
 */
export function step(s, dt, demand) {
  const { idle, redline, boost_bar: limit, boost_rpm: boostRpm } = s.spec;
  const lifted = demand < s.throttle - 0.02;
  s.throttle += (demand - s.throttle) * Math.min(1, dt * 14);
  if (!s.running) {
    s.rpm += (0 - s.rpm) * Math.min(1, dt * 3);
  } else {
    // revs: the pull grows with boost, friction pulls back toward idle, a limiter holds the redline
    const pull = s.throttle * (0.34 + 0.66 * Math.min(1, s.boost / limit)) * 5200;
    const drag = (s.rpm - idle) * (0.55 + 1.9 * (1 - s.throttle));
    s.rpm += (pull - drag * 0.62) * dt;
    if (s.rpm > redline) s.rpm = redline - 60 * Math.random();
    if (s.rpm < idle * 0.98) s.rpm += (idle - s.rpm) * Math.min(1, dt * 6);
  }
  // exhaust energy spins the turbine; the rotor is slow to gather speed and slower to lose it
  const energy = (s.rpm / boostRpm) * (0.16 + 0.84 * s.throttle);
  const want = Math.min(1.6, energy);
  const tau = want > s.turboSpeed ? 0.85 : 1.7;
  s.turboSpeed += (want - s.turboSpeed) * Math.min(1, dt / tau);
  const raw = limit * s.turboSpeed * s.turboSpeed;       // what the compressor would give unchecked
  // the wastegate opens on the factory limit and lets the surplus exhaust bypass the turbine
  const over = Math.max(0, raw - limit);
  s.wastegate += (Math.min(1, over / 0.35) - s.wastegate) * Math.min(1, dt * 9);
  s.boost = Math.min(raw, limit + 0.02 * s.wastegate);
  if (s.wastegate > 0.01) s.turboSpeed = Math.min(s.turboSpeed, 1.02 + 0.12 * s.wastegate);
  // shut the throttle under boost and the bypass valve returns the charge to the compressor inlet
  const dump = lifted && s.boost > 0.15 ? 1 : 0;
  s.bypass += (dump - s.bypass) * Math.min(1, dt * (dump ? 18 : 2.5));
  if (s.throttle < 0.05) s.boost *= Math.max(0, 1 - dt * 3.5);
  s.airflow += (Math.min(1, (s.rpm / redline) * (0.25 + 0.75 * s.throttle) * (1 + s.boost)) - s.airflow) * Math.min(1, dt * 8);
  s.crank += s.rpm * 6 * dt * s.slow;                     // rpm * 360 / 60 degrees per second, slowed
  s.turbo += (0.6 + 22 * Math.min(1.2, s.turboSpeed)) * dt;
}

/**
 * Turn the crank by hand while the engine is stopped.
 * Takes the state and a step in degrees. Returns nothing.
 */
export function nudge(s, degrees) {
  s.crank += degrees;
}

/**
 * Which stroke a cylinder is on.
 * Takes a cylinder row from engine930.json and the crank angle in degrees. Returns 'power', 'exhaust', 'intake'
 * or 'compression'.
 */
export function stroke(cyl, crank) {
  const a = (((crank - cyl.fire) % 720) + 720) % 720;
  return a < 180 ? 'power' : a < 360 ? 'exhaust' : a < 540 ? 'intake' : 'compression';
}

/**
 * How far a valve is open, from the events the model was built with.
 * Takes the [open, close] crank angles of the valve and the crank angle. Returns 0 to 1, a raised cosine.
 */
export function valveOpen(window, crank) {
  const centre = (window[0] + window[1]) / 2, half = (window[1] - window[0]) / 2;
  const d = ((((crank - centre) % 720) + 1080) % 720) - 360;
  return Math.abs(d) >= half ? 0 : 0.5 * (1 + Math.cos(Math.PI * d / half));
}
