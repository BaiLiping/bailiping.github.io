let sceneCount = 0;

const NS = 'http://www.w3.org/2000/svg';
const STROKES = [
  ['intake', '1 · Intake', 'Piston moves down', 'Intake open · exhaust closed'],
  ['compression', '2 · Compression', 'Piston moves up', 'Both valves closed'],
  ['power', '3 · Power', 'Hot gas pushes the piston down', 'Both valves closed'],
  ['exhaust', '4 · Exhaust', 'Piston pushes burned gas out', 'Intake closed · exhaust open'],
];

/** A representative petrol cylinder and its mechanically independent turbocharger. */
export function createTurboScene(container, { onSelect = () => {}, compact = false } = {}) {
  const id = `turbo-scene-${++sceneCount}`;
  const gid = (name) => `${id}-${name}`;
  const ref = (name) => `url(#${gid(name)})`;
  const p = (part, label, content) => `<g class="turbo-part" data-part="${part}" role="button" tabindex="0" aria-label="Explore ${label}" aria-pressed="false">${content}</g>`;

  container.innerHTML = `<svg class="engine-svg turbo-svg" viewBox="0 0 1100 360" role="group" aria-labelledby="${gid('title')} ${gid('desc')}" xmlns="${NS}" data-compact="${compact}">
    <title id="${gid('title')}">Turbocharged petrol engine: airflow, turbine shaft and four-stroke piston</title>
    <desc id="${gid('desc')}">Fresh air enters the compressor, passes through an intercooler, and enters the cylinder when the intake valve opens. The piston drives a crankshaft. Exhaust gas passes through the turbine. A separate shaft connects only the turbine and compressor. Four strokes take two crankshaft revolutions.</desc>
    <defs>
      <linearGradient id="${gid('metal')}" x1="0" y1="0" x2="1" y2="0"><stop stop-color="#e8edf0"/><stop offset=".22" stop-color="#f7f9fa"/><stop offset=".52" stop-color="#b4c1c7"/><stop offset=".74" stop-color="#e7edef"/><stop offset="1" stop-color="#9aabb3"/></linearGradient>
      <linearGradient id="${gid('piston-metal')}" x1="0" y1="0" x2="0" y2="1"><stop stop-color="#edf3f5"/><stop offset=".28" stop-color="#bac8d0"/><stop offset=".6" stop-color="#e5ecef"/><stop offset="1" stop-color="#9eafb8"/></linearGradient>
      <linearGradient id="${gid('cold-wheel')}" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#c7e4f3"/><stop offset=".5" stop-color="#729fb7"/><stop offset="1" stop-color="#3c6e86"/></linearGradient>
      <linearGradient id="${gid('hot-wheel')}" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#f4c6a3"/><stop offset=".5" stop-color="#cc7952"/><stop offset="1" stop-color="#8f4930"/></linearGradient>
      <linearGradient id="${gid('shaft')}" x1="0" y1="0" x2="0" y2="1"><stop stop-color="#76b98f"/><stop offset=".5" stop-color="#2f7d5b"/><stop offset="1" stop-color="#205940"/></linearGradient>
      <linearGradient id="${gid('charge')}" x1="0" y1="0" x2="0" y2="1"><stop stop-color="#cce9f7"/><stop offset="1" stop-color="#eef8fc"/></linearGradient>
      <linearGradient id="${gid('fire')}" x1="0" y1="0" x2="0" y2="1"><stop stop-color="#f8b94f"/><stop offset=".5" stop-color="#f6cd7a"/><stop offset="1" stop-color="#fff0ce"/></linearGradient>
      <linearGradient id="${gid('exhaust-fill')}" x1="0" y1="0" x2="0" y2="1"><stop stop-color="#f3c6aa"/><stop offset="1" stop-color="#fff1e7"/></linearGradient>
      <marker id="${gid('cold-arrow')}" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M1 1 9 5 1 9z" fill="#349bdd"/></marker>
      <marker id="${gid('hot-arrow')}" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M1 1 9 5 1 9z" fill="#e98438"/></marker>
      <marker id="${gid('power-arrow')}" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M1 1 9 5 1 9z" fill="#2f7d5b"/></marker>
      <clipPath id="${gid('chamber-clip')}"><rect x="686" y="126" width="108" height="116"/></clipPath>
    </defs>
    <style>
      #${id} { font-family: Inter, ui-sans-serif, system-ui, -apple-system, sans-serif; }
      .turbo-svg text { font-family: Inter, ui-sans-serif, system-ui, -apple-system, sans-serif; fill:#1d2520; }
      .turbo-svg .turbo-label { font-size:15px; font-weight:600; }
      .turbo-svg .turbo-small { font-size:12px; fill:#647067; }
      .turbo-svg .turbo-leader { stroke:#8a9890; stroke-width:1.2; fill:none; }
      .turbo-svg .turbo-part { cursor:pointer; outline:none; }
      .turbo-svg .turbo-selection { fill:#eaf4ee; fill-opacity:0; stroke:#2f7d5b; stroke-opacity:0; stroke-width:2; }
      .turbo-svg .turbo-part[data-selected="true"] .turbo-selection,
      .turbo-svg .turbo-part:focus-visible .turbo-selection { fill-opacity:.25; stroke-opacity:.9; }
      .turbo-svg .turbo-part:hover .turbo-selection { stroke-opacity:.65; }
      .turbo-svg .turbo-flow { pointer-events:none; }
      .turbo-svg .power-emphasis { opacity:.82; }
      .turbo-svg[data-power-view="true"] .power-emphasis { opacity:1; }
      .turbo-svg[data-power-view="true"] .turbo-pipes { opacity:.4; }
      .turbo-svg[data-compact="true"] .turbo-detail { display:none; }
    </style>

    <g class="turbo-pipes" fill="none" stroke-linecap="round" stroke-linejoin="round">
      <path d="M30 150H143" stroke="#cee7f3" stroke-width="18"/>
      <path d="M195 199V268M307 287H530Q549 287 549 268V99Q549 80 568 80H692Q711 80 711 99V124" stroke="#b9c8d0" stroke-width="20"/>
      <path d="M195 199V268M307 287H530Q549 287 549 268V99Q549 80 568 80H692Q711 80 711 99V124" stroke="#e2f2f9" stroke-width="15"/>
      <path d="M769 124V64Q769 45 750 45H384Q365 45 365 64V100M414 150H441Q460 150 460 169V201Q460 220 479 220H511" stroke="#d0b8a8" stroke-width="20"/>
      <path d="M769 124V64Q769 45 750 45H384Q365 45 365 64V100M414 150H441Q460 150 460 169V201Q460 220 479 220H511" stroke="#fff0e4" stroke-width="15"/>
      <path d="M41 150H107" stroke="#349bdd" stroke-width="2" marker-end="${ref('cold-arrow')}"/>
      <path d="M379 287H434M549 196V157M588 80H645" stroke="#349bdd" stroke-width="2" marker-end="${ref('cold-arrow')}"/>
      <path d="M636 45H568M469 220H512" stroke="#e98438" stroke-width="2" marker-end="${ref('hot-arrow')}"/>
    </g>
    <text class="turbo-small" x="30" y="124">Fresh air</text>
    <text class="turbo-small" x="441" y="245">Exhaust out</text>
    <text class="turbo-small turbo-detail" x="485" y="29" text-anchor="middle">Exhaust drives the turbine</text>
    <text class="turbo-small turbo-detail" x="410" y="315" text-anchor="middle">Cooled, compressed air</text>

    <g class="power-emphasis">
      <rect x="195" y="143" width="170" height="14" rx="5" fill="${ref('shaft')}" stroke="#235b43" stroke-width="1.2"/>
      <rect x="248" y="133" width="63" height="34" rx="6" fill="${ref('metal')}" stroke="#61747c" stroke-width="1.2"/>
      <rect x="250" y="144" width="59" height="12" rx="3" fill="${ref('shaft')}"/>
      <path d="M321 182H246" fill="none" stroke="#2f7d5b" stroke-width="2" marker-end="${ref('power-arrow')}"/>
      <text class="turbo-small" x="280" y="219" text-anchor="middle">Shared turbo shaft</text>
    </g>

    ${p('compressor', 'the compressor', `
      <rect class="turbo-selection" x="128" y="85" width="133" height="145" rx="16"/>
      <path d="M143 150C143 116 157 97 190 97H211C238 97 249 116 249 146S234 198 207 204L185 203V191C157 188 143 174 143 150Z" fill="${ref('metal')}" stroke="#566b75" stroke-width="1.6"/>
      <path d="M153 150C153 124 171 107 194 107S235 123 235 150 217 191 194 191 153 176 153 150Z" fill="#e4f2f8" stroke="#96adb9" stroke-width="1.2"/>
      <ellipse cx="195" cy="150" rx="29" ry="38" fill="#8aaebf" stroke="#476a7d" stroke-width="1.2"/>
      <g data-rotor="compressor"></g>
      <ellipse cx="195" cy="150" rx="9" ry="11" fill="#d0e0e7" stroke="#476a7d" stroke-width="1.2"/>
      <circle cx="195" cy="150" r="3" fill="#526d7a"/>
      <path class="turbo-leader" d="M162 198 153 218H132"/>
      <text class="turbo-label" x="170" y="237" text-anchor="end">Compressor</text>
    `)}

    ${p('turbine', 'the exhaust turbine', `
      <rect class="turbo-selection" x="303" y="85" width="126" height="126" rx="16"/>
      <path d="M313 150C313 121 326 103 352 98H375C405 99 418 119 418 150S398 201 367 201 313 179 313 150Z" fill="${ref('metal')}" stroke="#655c55" stroke-width="1.6"/>
      <ellipse cx="365" cy="150" rx="39" ry="42" fill="#fae9dd" stroke="#b89e8e" stroke-width="1.2"/>
      <ellipse cx="365" cy="150" rx="28" ry="38" fill="#bd805d" stroke="#855139" stroke-width="1.2"/>
      <g data-rotor="turbine"></g>
      <ellipse cx="365" cy="150" rx="9" ry="11" fill="#e2b292" stroke="#855139" stroke-width="1.2"/>
      <circle cx="365" cy="150" r="3" fill="#815540"/>
      <path class="turbo-leader" d="M379 198V216"/>
      <text class="turbo-label" x="379" y="237" text-anchor="middle">Turbine</text>
    `)}

    ${p('intercooler', 'the intercooler', `
      <rect class="turbo-selection" x="141" y="250" width="177" height="79" rx="11"/>
      <rect x="151" y="262" width="155" height="49" rx="5" fill="${ref('metal')}" stroke="#6c848f" stroke-width="1.5"/>
      <rect x="165" y="267" width="128" height="39" rx="2" fill="#e1f1f7" stroke="#9bb5c2" stroke-width="1"/>
      <g stroke="#9dbac8" stroke-width="1.5">${Array.from({length:17},(_,i)=>`<path d="M${170+i*7.3} 268V305"/>`).join('')}</g>
      <path d="M165 276H293M165 285H293M165 295H293" stroke="#c4dce6" stroke-width="2"/>
      <text class="turbo-label" x="228" y="345" text-anchor="middle">Intercooler</text>
    `)}

    ${p('cylinder', 'the cylinder and valves', `
      <rect class="turbo-selection" x="663" y="92" width="155" height="158" rx="13"/>
      <path d="M674 127V243H685V128H795V243H807V127Z" fill="${ref('metal')}" stroke="#617781" stroke-width="1.5"/>
      <path d="M672 126V115Q672 108 679 108H801Q808 108 808 115V126Z" fill="${ref('metal')}" stroke="#617781" stroke-width="1.5"/>
      <path d="M703 108V127H719V108" fill="#e2f2f9" stroke="#9bb5c2" stroke-width="1.2"/>
      <path d="M761 108V127H777V108" fill="#fff0e4" stroke="#b89e8e" stroke-width="1.2"/>
      <rect data-chamber x="686" y="127" width="108" height="20" fill="${ref('charge')}"/>
      <path d="M672 139H683M672 155H683M672 171H683M672 187H683M672 203H683M672 219H683M797 139H809M797 155H809M797 171H809M797 187H809M797 203H809M797 219H809" stroke="#8ba0aa" stroke-width="3"/>
      <g data-valve="intake" fill="#5f8597" stroke="#3f6475" stroke-width="1.5"><path d="M709 99H713V132H709Z"/><path d="M700 132H722L719 136H703Z"/></g>
      <g data-valve="exhaust" fill="#b78365" stroke="#876048" stroke-width="1.5"><path d="M767 99H771V132H767Z"/><path d="M758 132H780L777 136H761Z"/></g>
      <g stroke="#617781" stroke-width="1.2"><path d="M736 98H744V120H736Z" fill="#eef1e9"/><path d="M735 104H745M735 108H745M735 112H745"/><path d="M739 120V126H742" fill="none"/></g>
      <g clip-path="${ref('chamber-clip')}"><g data-cylinder-particles></g><path data-spark d="M739 128 733 138 741 136 737 147 748 133 741 135 744 128Z" fill="#fff9ce" stroke="#ee9c28" stroke-width="1.2" opacity="0"/></g>
      <path class="turbo-leader" d="M808 153H839"/>
      <text class="turbo-label" x="850" y="158">Cylinder</text>
      <text class="turbo-small" x="850" y="178">One petrol cylinder shown</text>
    `)}

    ${p('crankshaft', 'the crankshaft', `
      <rect class="turbo-selection" x="680" y="248" width="138" height="94" rx="15"/>
      <g class="power-emphasis"><circle cx="740" cy="281" r="41" fill="#edf3ee" stroke="#c7d7cd" stroke-width="1.5"/>
      <circle cx="740" cy="281" r="33" fill="none" stroke="#afc9ba" stroke-width="9"/>
      <g data-crank><path d="M730 281V248Q740 236 750 248V281Z" fill="${ref('shaft')}" stroke="#2a6349" stroke-width="1.5"/><circle cx="740" cy="248" r="7" fill="#d2dfe2" stroke="#54776a" stroke-width="1.5"/></g>
      <circle cx="740" cy="281" r="11" fill="${ref('metal')}" stroke="#2f7d5b" stroke-width="2"/><circle cx="740" cy="281" r="4" fill="#2f7d5b"/>
      <path d="M782 281H1018" stroke="#2f7d5b" stroke-width="4" marker-end="${ref('power-arrow')}"/>
      <text class="turbo-label" x="894" y="267" text-anchor="middle">Power to the wheels</text>
      <text class="turbo-small" x="894" y="303" text-anchor="middle">Through the drivetrain</text></g>
      <text class="turbo-label" x="740" y="345" text-anchor="middle">Crankshaft</text>
    `)}

    <g pointer-events="none">
      <path data-rod stroke="#526b77" stroke-width="13" fill="none" stroke-linecap="round"/>
      <path data-rod-shine stroke="#b6c7cf" stroke-width="8" fill="none" stroke-linecap="round"/>
      <g data-piston><rect x="686" y="0" width="108" height="28" rx="3" fill="${ref('piston-metal')}" stroke="#5f7682" stroke-width="1.5"/><path d="M687 6H793M687 10H793M687 20H793" stroke="#8299a5" stroke-width="1.2"/><circle cx="740" cy="16" r="6" fill="#d8e4e9" stroke="#6a818d" stroke-width="1.5"/></g>
    </g>

    <g class="turbo-flow" data-pipe-particles></g>
    <g class="turbo-detail"><text data-stroke-label x="850" y="89" style="font-size:19px;font-weight:650;fill:#215c43">1 · Intake</text><text data-stroke-motion class="turbo-small" x="850" y="111">Piston moves down</text><text data-valve-status class="turbo-small" x="850" y="211">Intake open · exhaust closed</text></g>
    <path id="${gid('fresh-path')}" d="M28 150H144" fill="none"/>
    <path id="${gid('charge-path-a')}" d="M195 204V261" fill="none"/>
    <path id="${gid('charge-path-b')}" d="M308 287H530Q549 287 549 268V99Q549 80 568 80H692Q711 80 711 99V122" fill="none"/>
    <path id="${gid('hot-path')}" d="M769 122V64Q769 45 750 45H384Q365 45 365 64V100" fill="none"/>
    <path id="${gid('outlet-path')}" d="M419 150H441Q460 150 460 169V201Q460 220 479 220H511" fill="none"/>
  </svg>`;

  const svg = container.querySelector('svg');
  svg.id = id;
  const q = (selector) => svg.querySelector(selector);
  const wheelGroups = [q('[data-rotor="compressor"]'), q('[data-rotor="turbine"]')];
  const wheelBlades = wheelGroups.map((group, index) => Array.from({length:10}, () => {
    const blade = document.createElementNS(NS, 'path');
    blade.setAttribute('fill', ref(index ? 'hot-wheel' : 'cold-wheel'));
    blade.setAttribute('stroke', index ? '#885338' : '#456e83');
    blade.setAttribute('stroke-width', '.8');
    group.append(blade);
    return blade;
  }));
  const pipeParticleGroup = q('[data-pipe-particles]');
  const pipePaths = [
    { name:'fresh-path', count:7, color:'#349bdd', flow:1 },
    { name:'charge-path-a', count:4, color:'#349bdd', flow:1 },
    { name:'charge-path-b', count:20, color:'#349bdd', flow:1 },
    { name:'hot-path', count:19, color:'#e98438', flow:-1 },
    { name:'outlet-path', count:9, color:'#e98438', flow:-1 },
  ].map((spec) => {
    const path = q(`#${gid(spec.name)}`);
    return { ...spec, path, length:path.getTotalLength(), dots:Array.from({length:spec.count}, () => {
      const circle = document.createElementNS(NS, 'circle');
      circle.setAttribute('r', '2.8'); circle.setAttribute('fill', spec.color);
      pipeParticleGroup.append(circle); return circle;
    }) };
  });
  const cylinderDots = Array.from({length:12}, () => {
    const circle = document.createElementNS(NS, 'circle');
    circle.setAttribute('r', '2.6'); q('[data-cylinder-particles]').append(circle); return circle;
  });
  const partGroups = [...svg.querySelectorAll('[data-part]')];
  partGroups.forEach((group) => {
    const choose = () => onSelect(group.dataset.part);
    group.addEventListener('click', choose);
    group.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); choose(); }
    });
  });
  let lastTime = null;
  let rotorPhase = 0;
  let flowPhase = 0;
  let currentStroke = -1;
  const piston = q('[data-piston]');
  const rod = q('[data-rod]');
  const rodShine = q('[data-rod-shine]');
  const chamber = q('[data-chamber]');
  const spark = q('[data-spark]');
  const intakeValve = q('[data-valve="intake"]');
  const exhaustValve = q('[data-valve="exhaust"]');
  const crank = q('[data-crank]');

  function select(partId) {
    partGroups.forEach((group) => {
      const selected = group.dataset.part === partId;
      group.dataset.selected = String(selected);
      group.setAttribute('aria-pressed', String(selected));
    });
  }

  function render(state) {
    const time = Number.isFinite(state.time) ? state.time : 0;
    const turbo = Math.max(0, Math.min(1, Number(state.turbo) || 0));
    const cycle = ((Number(state.cycle) || 0) % 720 + 720) % 720;
    if (lastTime !== null && time < lastTime) { rotorPhase = 0; flowPhase = 0; }
    const dt = lastTime === null ? 0 : Math.max(0, Math.min(.2, time-lastTime));
    lastTime = time;
    rotorPhase = (rotorPhase + dt * (3 + 12 * turbo)) % (2*Math.PI);
    flowPhase += dt * (34 + 70 * turbo);
    const strokeIndex = Math.floor(cycle / 180);
    const stroke = STROKES[strokeIndex];
    container.dataset.stroke = stroke[0];
    svg.dataset.powerView = String(Boolean(state.powerView));
    if (strokeIndex !== currentStroke) {
      q('[data-stroke-label]').textContent = stroke[1];
      q('[data-stroke-motion]').textContent = stroke[2];
      q('[data-valve-status]').textContent = stroke[3];
      currentStroke = strokeIndex;
    }
    const theta = cycle*Math.PI/180;
    const pinX = 740 + 33*Math.sin(theta);
    const pinY = 281 - 33*Math.cos(theta);
    const pistonPinY = pinY-Math.sqrt(87*87-(pinX-740)**2);
    const pistonTop = pistonPinY-16;
    piston.setAttribute('transform', `translate(0 ${pistonTop.toFixed(3)})`);
    const rodPath = `M740 ${pistonPinY.toFixed(3)}L${pinX.toFixed(3)} ${pinY.toFixed(3)}`;
    rod.setAttribute('d', rodPath); rodShine.setAttribute('d', rodPath);
    crank.setAttribute('transform', `rotate(${cycle} 740 281)`);
    chamber.setAttribute('height', Math.max(4,pistonTop-127).toFixed(3));
    chamber.setAttribute('fill', ref(strokeIndex === 2 ? 'fire' : strokeIndex === 3 ? 'exhaust-fill' : 'charge'));
    chamber.setAttribute('opacity', strokeIndex === 1 ? String(.5+.5*((cycle-180)/180)) : '1');
    const intakeOpen = strokeIndex === 0;
    const exhaustOpen = strokeIndex === 3;
    intakeValve.setAttribute('transform', `translate(0 ${intakeOpen ? 7 : 0})`);
    exhaustValve.setAttribute('transform', `translate(0 ${exhaustOpen ? 7 : 0})`);
    spark.setAttribute('opacity', cycle >= 360 && cycle < 383 ? '1' : '0');

    wheelBlades.forEach((blades, wheel) => {
      const cx = wheel ? 365 : 195;
      blades.forEach((blade,index) => {
        const a = rotorPhase+index*2*Math.PI/10;
        const at = a+.19;
        const ax = cx+27*Math.cos(a), ay = 150+36*Math.sin(a);
        const bx = cx+27*Math.cos(at), by = 150+36*Math.sin(at);
        const hx = cx+7*Math.cos(a+.55), hy = 150+9*Math.sin(a+.55);
        blade.setAttribute('d', `M${cx} 150L${ax.toFixed(2)} ${ay.toFixed(2)}Q${bx.toFixed(2)} ${by.toFixed(2)} ${hx.toFixed(2)} ${hy.toFixed(2)}Z`);
      });
    });
    pipePaths.forEach((entry) => {
      entry.dots.forEach((dot, index) => {
        const fraction = ((flowPhase/entry.length + index/entry.count) % 1 + 1) % 1;
        const point = entry.path.getPointAtLength(fraction*entry.length);
        dot.setAttribute('cx', point.x.toFixed(2)); dot.setAttribute('cy', point.y.toFixed(2));
        // Every pipe ends before a valve seat: no tracer crosses a closed valve.
        const edgeFade = Math.min(1,fraction*12,(1-fraction)*12);
        dot.setAttribute('opacity', (edgeFade*(state.powerView ? .3 : .8)).toFixed(2));
      });
    });
    cylinderDots.forEach((dot,index) => {
      // Closed strokes move only the trapped charge; they create no inlet/outlet flow.
      let x, y;
      const h = Math.max(4,pistonTop-137);
      const f = ((flowPhase/63+index/12)%1+1)%1;
      if (intakeOpen) {
        x = 711 + Math.sin(f*Math.PI)*((index%3-1)*17+17);
        y = 129 + f*Math.max(8,h-3);
      } else if (exhaustOpen) {
        x = 740 + 29*f + (1-f)*((index%3)-1)*17;
        y = 135+(1-f)*Math.max(3,h-3);
      } else {
        x = 698+(index%4)*27;
        y = 135+(Math.floor(index/4)+.5)*h/3;
      }
      dot.setAttribute('cx', x.toFixed(2)); dot.setAttribute('cy', y.toFixed(2));
      dot.setAttribute('fill', strokeIndex>=2 ? '#e98438' : '#349bdd');
      dot.setAttribute('opacity', state.powerView ? '.25' : '.75');
    });
  }

  render({ time:0, cycle:70, turbo:.3, powerView:false });
  return { render, select };
}
