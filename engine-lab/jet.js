const NS = 'http://www.w3.org/2000/svg';
let sceneCount = 0;
const clamp = (value, low = 0, high = 1) => Math.min(high, Math.max(low, Number(value) || 0));

/** A teaching cutaway: tracers show gas paths, green shafts show mechanical power. */
export function createJetScene(container, { onSelect = () => {} } = {}) {
  const id = `jet-scene-${++sceneCount}`;
  const gid = (name) => `${id}-${name}`;
  const ref = (name) => `url(#${gid(name)})`;
  const part = (name, label, x, width, content) => `<g class="jet-part" data-part="${name}" role="button" tabindex="0" aria-label="Explore the ${label}" aria-pressed="false"><rect class="part-outline jet-selection" x="${x}" y="74" width="${width}" height="176" rx="12"/>${content}</g>`;
  const label = (x, text, top = 239) => `<path class="jet-leader" d="M${x} ${top}V286"/><circle cx="${x}" cy="${top}" r="2.3" fill="#667b83"/><text class="jet-label" x="${x}" y="309" text-anchor="middle">${text}</text>`;

  container.innerHTML = `<svg id="${id}" class="engine-svg jet-svg" viewBox="0 0 1100 340" role="group" aria-labelledby="${gid('title')} ${gid('description')}" xmlns="${NS}" data-time="0">
    <title id="${gid('title')}">Jet engine: continuous airflow and turbine-driven compression</title>
    <desc id="${gid('description')}">Air moves from left to right through the compressor, continuously burning combustor, turbine and nozzle. The turbine drives the compressor through a green shaft. The engine's thrust points left, opposite the exhaust. Optional turbofan mode adds a cool bypass stream and a separate fan turbine and shaft.</desc>
    <defs>
      <linearGradient id="${gid('metal')}" x1="0" y1="0" x2="0" y2="1"><stop stop-color="#b8c5cc"/><stop offset=".17" stop-color="#eff4f6"/><stop offset=".45" stop-color="#a5b7c0"/><stop offset=".68" stop-color="#e2eaee"/><stop offset="1" stop-color="#899fa9"/></linearGradient>
      <linearGradient id="${gid('hub')}" x1="0" y1="0" x2="0" y2="1"><stop stop-color="#78929f"/><stop offset=".28" stop-color="#e2eaed"/><stop offset=".52" stop-color="#f5f8f8"/><stop offset="1" stop-color="#879fa9"/></linearGradient>
      <linearGradient id="${gid('cold-blade')}" x1="0" y1="0" x2="1" y2="0"><stop stop-color="#4d7e98"/><stop offset=".42" stop-color="#b8d5e4"/><stop offset=".7" stop-color="#759fb7"/><stop offset="1" stop-color="#3f6f8a"/></linearGradient>
      <linearGradient id="${gid('hot-blade')}" x1="0" y1="0" x2="1" y2="0"><stop stop-color="#a95835"/><stop offset=".4" stop-color="#f5bd8e"/><stop offset=".66" stop-color="#db8e5b"/><stop offset="1" stop-color="#a55c3a"/></linearGradient>
      <linearGradient id="${gid('core-air')}" x1="0" y1="0" x2="1" y2="0"><stop stop-color="#e7f5fc"/><stop offset=".5" stop-color="#c9e5f3"/><stop offset="1" stop-color="#e1f0f7"/></linearGradient>
      <linearGradient id="${gid('hot-gas')}" x1="0" y1="0" x2="1" y2="0"><stop stop-color="#fff4d9"/><stop offset=".43" stop-color="#ffe0a4"/><stop offset="1" stop-color="#f5ba8a"/></linearGradient>
      <linearGradient id="${gid('flame')}" x1="0" y1="0" x2="1" y2="0"><stop stop-color="#f5a23a"/><stop offset=".3" stop-color="#ffd070"/><stop offset="1" stop-color="#fff0b7" stop-opacity=".28"/></linearGradient>
      <linearGradient id="${gid('plume')}" x1="0" y1="0" x2="1" y2="0"><stop stop-color="#ef9b59" stop-opacity=".38"/><stop offset="1" stop-color="#f4b77d" stop-opacity="0"/></linearGradient>
      <linearGradient id="${gid('inlet-air')}" x1="0" y1="0" x2="1" y2="0"><stop stop-color="#73c5f2" stop-opacity="0"/><stop offset="1" stop-color="#7dc8ef" stop-opacity=".25"/></linearGradient>
      <linearGradient id="${gid('shaft')}" x1="0" y1="0" x2="0" y2="1"><stop stop-color="#7dbb92"/><stop offset=".46" stop-color="#3c946b"/><stop offset="1" stop-color="#235c43"/></linearGradient>
      <linearGradient id="${gid('bypass-air')}" x1="0" y1="0" x2="1" y2="0"><stop stop-color="#e5f4fc"/><stop offset=".55" stop-color="#d9eef8"/><stop offset="1" stop-color="#eaf6fb" stop-opacity=".45"/></linearGradient>
      <marker id="${gid('cold-arrow')}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M1 1 9 5 1 9Z" fill="#349bdd"/></marker>
      <marker id="${gid('hot-arrow')}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M1 1 9 5 1 9Z" fill="#e98438"/></marker>
      <marker id="${gid('power-arrow')}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M1 1 9 5 1 9Z" fill="#2f7d5b"/></marker>
      <clipPath id="${gid('upper-burner')}"><path d="M511 100Q593 76 672 102L704 114V143H511Z"/></clipPath>
      <clipPath id="${gid('lower-burner')}"><path d="M511 220Q593 244 672 218L704 206V177H511Z"/></clipPath>
    </defs>
    <style>
      #${id} text{font-family:Inter,ui-sans-serif,system-ui,-apple-system,sans-serif;fill:#1d2520}
      #${id} .jet-label{font-size:15px;font-weight:560}
      #${id} .jet-small{font-size:11.5px;fill:#63766d}
      #${id} .jet-leader{fill:none;stroke:#879a9f;stroke-width:1.1}
      #${id} .jet-part{cursor:pointer;outline:none}
      #${id} .jet-selection{fill:#e9f3eb;fill-opacity:0;stroke:#2f7d5b;stroke-opacity:0;stroke-width:1.5}
      #${id} .jet-part[data-selected="true"] .jet-selection{fill-opacity:.28;stroke-opacity:.68}
      #${id} .jet-part:hover .jet-selection{stroke-opacity:.78}
      #${id} .jet-part:focus-visible .jet-selection{stroke-opacity:1;stroke-width:3}
      #${id} .jet-part[data-selected="true"] .jet-label{fill:#215e43;font-weight:650}
      #${id} .jet-bypass{display:none}
      #${id}[data-bypass="true"] .jet-bypass{display:inline}
      #${id}[data-bypass="true"] .jet-only{display:none}
      #${id} .jet-power-overlay{display:none}
      #${id}[data-power-view="true"] .jet-power-overlay{display:inline}
      #${id}[data-power-view="true"] .jet-gas{opacity:.35}
      #${id}[data-power-view="true"] .jet-shafts{filter:drop-shadow(0 0 3px #87c49b)}
      #${id} .jet-tracers,#${id} .jet-shafts,#${id} .jet-power-overlay{pointer-events:none}
    </style>

    <g class="jet-gas" pointer-events="none">
      <path d="M0 91 184 105V215L0 229Z" fill="${ref('inlet-air')}"/>
      <path data-plume d="M955 123Q1020 116 1100 108V212Q1020 204 955 197Z" fill="${ref('plume')}"/>
      <path d="M36 112H120M36 208H120" fill="none" stroke="#349bdd" stroke-width="2.6" marker-end="${ref('cold-arrow')}"/>
      <path d="M985 136H1065M985 184H1065" fill="none" stroke="#e98438" stroke-width="2.6" marker-end="${ref('hot-arrow')}"/>
    </g>
    <g pointer-events="none">
      <path d="M135 35H42" stroke="#2f7d5b" stroke-width="2.6" fill="none" marker-end="${ref('power-arrow')}"/>
      <text x="88" y="21" text-anchor="middle" class="jet-small" style="fill:#2f7d5b;font-weight:600">Forward thrust</text>
    </g>

    <g class="jet-bypass" pointer-events="none">
      <path d="M174 49Q420 22 686 48Q847 60 975 89L975 102Q827 77 674 70Q414 45 181 77Z" fill="${ref('bypass-air')}"/>
      <path d="M174 271Q420 298 686 272Q847 260 975 231L975 218Q827 243 674 250Q414 275 181 243Z" fill="${ref('bypass-air')}"/>
      <path d="M174 45Q420 18 686 44Q847 56 975 85L975 91Q827 62 684 51Q422 25 177 53Z" fill="${ref('metal')}" stroke="#8199a5" stroke-width="1.2"/>
      <path d="M174 275Q420 302 686 276Q847 264 975 235L975 229Q827 258 684 269Q422 295 177 267Z" fill="${ref('metal')}" stroke="#8199a5" stroke-width="1.2"/>
      <text class="jet-small" x="546" y="23" text-anchor="middle" style="fill:#3387b9">Cool bypass air · around the core</text>
      <path d="M407 48H472M846 73 899 80M407 272H472M846 247 899 240" fill="none" stroke="#349bdd" stroke-width="1.8" marker-end="${ref('cold-arrow')}"/>
      <g data-fan-rotor></g>
      <path d="M212 44V32H246" class="jet-leader"/>
      <text class="jet-small" x="252" y="36">Fan</text>
    </g>

    <g pointer-events="none">
      <path d="M176 82Q227 74 252 79L482 96 514 91Q594 64 674 90L715 101 820 104Q886 108 965 116L965 124Q877 119 820 113L713 111 671 100Q594 77 516 102L483 106 252 89Q219 85 183 94Z" fill="${ref('metal')}" stroke="#657c88" stroke-width="1.4"/>
      <path d="M176 238Q227 246 252 241L482 224 514 229Q594 256 674 230L715 219 820 216Q886 212 965 204L965 196Q877 201 820 207L713 209 671 220Q594 243 516 218L483 214 252 231Q219 235 183 226Z" fill="${ref('metal')}" stroke="#657c88" stroke-width="1.4"/>
      <path d="M183 94Q220 85 252 89L483 106 511 102V218L483 214 252 231Q220 235 183 226Z" fill="${ref('core-air')}"/>
      <path d="M511 102Q593 77 671 100L713 111 820 113Q883 120 965 124V196Q883 200 820 207L713 209 671 220Q593 243 511 218Z" fill="${ref('hot-gas')}"/>
      <path d="M249 79V90M249 230V241M484 97V107M484 213V223M711 101V111M711 209V219M818 104V114M818 206V216" stroke="#6d8490" stroke-width="6"/>
      <path d="M249 78V90M249 230V242M484 96V107M484 213V224M711 100V111M711 209V220M818 103V114M818 206V217" stroke="#d7e2e6" stroke-width="2"/>
      <path d="M215 135Q316 125 470 136L511 141H795Q839 141 899 160Q839 179 795 179H511L470 184Q316 195 215 185L166 160Z" fill="${ref('hub')}" stroke="#718b98" stroke-width="1.4"/>
      <path d="M220 139Q328 132 470 142M220 181Q328 188 470 178" stroke="#f5f9fa" stroke-width="1.2" fill="none"/>
    </g>

    ${part('intake', 'intake', 149, 93, `
      <path d="M178 83Q159 118 160 160T178 237L185 225Q172 192 173 160T185 95Z" fill="${ref('metal')}" stroke="#647e8c" stroke-width="1.3"/>
      <path d="M166 160Q186 135 220 135V185Q185 185 166 160Z" fill="${ref('hub')}" stroke="#6c8895" stroke-width="1.2"/>
      ${label(194, 'Intake', 238)}
    `)}
    ${part('compressor', 'compressor', 247, 244, `
      <g data-compressor-rotors></g>
      ${label(371, 'Compressor', 232)}
    `)}
    ${part('combustor', 'combustor', 504, 194, `
      <path d="M512 108Q587 88 665 108L698 120M512 212Q587 232 665 212L698 200" fill="none" stroke="#c39d73" stroke-width="2"/>
      <path d="M519 103V121M519 217V199" stroke="#7a8583" stroke-width="4"/>
      <path d="M520 115H534M520 205H534" stroke="#cc873e" stroke-width="3"/>
      <g class="jet-gas" data-flames>
        <g clip-path="${ref('upper-burner')}"><path data-flame-upper fill="${ref('flame')}" d="M531 115Q555 94 573 112Q598 94 613 113Q641 98 676 120Q646 138 616 128Q589 144 567 127Q545 135 531 115Z"/><path d="M533 115Q556 106 575 117Q597 110 626 122Q598 128 570 122Q549 128 533 115Z" fill="#fff2c2" opacity=".8"/></g>
        <g clip-path="${ref('lower-burner')}"><path data-flame-lower fill="${ref('flame')}" d="M531 205Q555 226 573 208Q598 226 613 207Q641 222 676 200Q646 182 616 192Q589 176 567 193Q545 185 531 205Z"/><path d="M533 205Q556 214 575 203Q597 210 626 198Q598 192 570 198Q549 192 533 205Z" fill="#fff2c2" opacity=".8"/></g>
      </g>
      <g fill="#b9a288">${[543, 568, 593, 618, 643, 668].map((x) => `<circle cx="${x}" cy="${103 - Math.sin((x - 535) / 145 * Math.PI) * 7}" r="1.8"/><circle cx="${x}" cy="${217 + Math.sin((x - 535) / 145 * Math.PI) * 7}" r="1.8"/>`).join('')}</g>
      ${label(602, 'Combustor', 246)}
    `)}
    ${part('turbine', 'turbine', 704, 119, `
      <g data-turbine-rotors></g>
      <g class="jet-bypass" data-fan-turbine></g>
      ${label(761, 'Turbine', 220)}
    `)}
    ${part('nozzle', 'nozzle', 834, 141, `
      <path d="M844 112Q904 123 965 124V132Q906 132 844 121Z" fill="${ref('metal')}" stroke="#718894" stroke-width="1.2"/>
      <path d="M844 208Q904 197 965 196V188Q906 188 844 199Z" fill="${ref('metal')}" stroke="#718894" stroke-width="1.2"/>
      <path d="M965 124V132M965 188V196" stroke="#627b87" stroke-width="3"/>
      ${label(920, 'Nozzle', 203)}
    `)}

    <g class="jet-shafts">
      <rect class="jet-only" x="269" y="154" width="522" height="12" rx="4" fill="${ref('shaft')}" stroke="#286648" stroke-width="1"/>
      <g class="jet-bypass">
        <rect x="269" y="150" width="522" height="20" rx="5" fill="#c5e0cf" fill-opacity=".72" stroke="#2f7d5b" stroke-width="2"/>
        <path d="M279 154H780M279 166H780" stroke="#519772" stroke-width="2"/>
        <rect x="209" y="158" width="620" height="4" rx="2" fill="#236b48" stroke="#b9e2a7" stroke-width=".7"/>
        <circle cx="212" cy="160" r="6" fill="#4d9365" stroke="#204d39" stroke-width="1.2"/>
        <circle cx="826" cy="160" r="5" fill="#4d9365" stroke="#204d39" stroke-width="1.2"/>
      </g>
      <path d="M270 148V172M790 148V172" stroke="#718c82" stroke-width="5"/>
      <path d="M270 148V172M790 148V172" stroke="#c9dfd3" stroke-width="2"/>
    </g>

    <g class="jet-power-overlay">
      <g class="jet-only"><path d="M741 160H326" fill="none" stroke="#bce5a7" stroke-width="2.4" marker-end="${ref('power-arrow')}"/><rect x="486" y="176" width="205" height="23" rx="5" fill="#f7faf4" fill-opacity=".93"/><text x="588" y="192" text-anchor="middle" class="jet-small" style="fill:#256a48;font-weight:600">Turbine powers compressor</text></g>
      <g class="jet-bypass"><rect x="519" y="127" width="103" height="20" rx="4" fill="#f7faf4" fill-opacity=".95"/><text x="570" y="141" text-anchor="middle" class="jet-small" style="fill:#256a48;font-weight:600">Outer core shaft</text><path d="M570 147V151" stroke="#2f7d5b"/><rect x="519" y="177" width="103" height="20" rx="4" fill="#f7faf4" fill-opacity=".95"/><text x="570" y="191" text-anchor="middle" class="jet-small" style="fill:#256a48;font-weight:600">Inner fan shaft</text><path d="M570 176V160" stroke="#2f7d5b" stroke-width="1.2"/><path d="M755 154H658M755 160H658" stroke="#235c43" fill="none" stroke-width="1.5" marker-end="${ref('power-arrow')}"/></g>
    </g>

    <g class="jet-tracers jet-gas" data-core-particles></g>
    <g class="jet-tracers jet-gas jet-bypass" data-bypass-particles></g>
    <g class="jet-bypass" pointer-events="none"><path d="M827 207 841 246H872" class="jet-leader"/><text class="jet-small" x="879" y="250">Fan turbine</text></g>
    <g fill="none" stroke="none" pointer-events="none">
      <path id="${gid('upper-path')}" d="M24 115Q123 111 183 113L255 111Q367 115 488 125Q518 119 542 116Q605 107 670 121L744 126Q828 125 900 141Q981 138 1080 129"/>
      <path id="${gid('lower-path')}" d="M24 205Q123 209 183 207L255 209Q367 205 488 195Q518 201 542 204Q605 213 670 199L744 194Q828 195 900 179Q981 182 1080 191"/>
      <path id="${gid('bypass-upper-path')}" d="M54 66Q137 56 212 63Q440 36 674 60Q839 70 970 96L1080 83"/>
      <path id="${gid('bypass-lower-path')}" d="M54 254Q137 264 212 257Q440 284 674 260Q839 250 970 224L1080 237"/>
    </g>
  </svg>`;

  const svg = container.querySelector('svg');
  const q = (selector) => svg.querySelector(selector);
  const rotors = [];

  // Radial blades projected onto an axial cutaway. Every stage turns together;
  // the phase offset only makes the repeated geometry easier to read.
  function addRotor(group, x, radius, hot = false, fan = false, phase = 0) {
    const make = (name, attributes) => {
      const el = document.createElementNS(NS, name);
      Object.entries(attributes).forEach(([key, value]) => el.setAttribute(key, value));
      group.appendChild(el);
      return el;
    };
    const inner = fan ? 14 : 23;
    make('ellipse', { cx: x, cy: 160, rx: fan ? 10 : 9, ry: radius, fill: hot ? '#df9c73' : '#8fb6cb', 'fill-opacity': '.25', stroke: hot ? '#b87b57' : '#6d98ad', 'stroke-width': '.8' });
    const blades = Array.from({ length: fan ? 14 : 11 }, () => make('path', { fill: ref(hot ? 'hot-blade' : 'cold-blade'), stroke: hot ? '#9d633f' : '#4e798f', 'stroke-width': '.7', 'stroke-linejoin': 'round' }));
    make('ellipse', { cx: x, cy: 160, rx: 7, ry: inner, fill: ref('hub'), stroke: '#6d8792', 'stroke-width': '1' });
    rotors.push({ x, radius, inner, hot, fan, phase, blades });
  }

  const compressor = q('[data-compressor-rotors]');
  [267, 300, 333, 366, 399, 432, 465].forEach((x, i) => addRotor(compressor, x, 70 - i * 3.8, false, false, i * .31));
  const turbine = q('[data-turbine-rotors]');
  addRotor(turbine, 733, 51, true, false, .2);
  addRotor(turbine, 775, 49, true, false, .8);
  const fanGroup = q('[data-fan-rotor]');
  // Keep the fan visible across both the core inlet and the outer bypass duct.
  // The central cone and shafts still provide a clear separation from airflow.
  fanGroup.setAttribute('class', 'jet-bypass');
  fanGroup.setAttribute('pointer-events', 'none');
  svg.insertBefore(fanGroup, q('.jet-shafts'));
  addRotor(fanGroup, 212, 108, false, true, .1);
  addRotor(q('[data-fan-turbine]'), 826, 44, true, true, .7);

  // Sample each unambiguous gas passage once. Tracers stay outside the central
  // hub and shaft; these dots are qualitative tracers, not a density scale.
  function samplePath(name) {
    const path = q(`#${gid(name)}`);
    const length = path.getTotalLength();
    const points = Array.from({ length: 241 }, (_, i) => {
      const p = path.getPointAtLength(length * i / 240);
      return [p.x, p.y];
    });
    return { points, length };
  }
  const corePaths = ['upper-path', 'lower-path'].map(samplePath);
  const bypassPaths = ['bypass-upper-path', 'bypass-lower-path'].map(samplePath);
  const particles = [];
  function addParticles(group, paths, count, bypass = false) {
    paths.forEach((path, side) => {
      const lanes = bypass ? [-3, 3] : [-8, 0, 8];
      lanes.forEach((offset, lane) => {
        for (let i = 0; i < count; i++) {
          const circle = document.createElementNS(NS, 'circle');
          circle.setAttribute('r', bypass ? 2 : 2.1 + ((i + lane) % 3) * .45);
          circle.setAttribute('fill', '#349bdd');
          circle.setAttribute('opacity', `${.42 + ((i * 7 + lane) % 5) * .1}`);
          group.appendChild(circle);
          particles.push({ circle, path, offset, bypass, phase: (i + lane * .37 + side * .23) / count });
        }
      });
    });
  }
  addParticles(q('[data-core-particles]'), corePaths, 17);
  addParticles(q('[data-bypass-particles]'), bypassPaths, 12, true);

  const partNodes = [...svg.querySelectorAll('[data-part]')];
  const knownParts = new Set(partNodes.map((el) => el.dataset.part));
  let selected = 'compressor';
  function select(partId) {
    if (!knownParts.has(partId)) return;
    selected = partId;
    partNodes.forEach((el) => {
      const active = el.dataset.part === selected;
      el.dataset.selected = String(active);
      el.setAttribute('aria-pressed', String(active));
    });
    svg.dataset.selected = selected;
  }
  partNodes.forEach((el) => {
    const activate = () => { select(el.dataset.part); onSelect(el.dataset.part); };
    el.addEventListener('click', activate);
    el.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        activate();
      }
    });
  });

  let lastTime = null;
  let angle = 0;
  let fanAngle = 0;
  let distance = 0;
  let bypassDistance = 0;
  const upperFlame = q('[data-flame-upper]');
  const lowerFlame = q('[data-flame-lower]');
  const flames = q('[data-flames]');
  const plume = q('[data-plume]');

  function render(state = {}) {
    const time = Math.max(0, Number(state.time) || 0);
    const spool = clamp(state.jet);
    const bypass = Boolean(state.bypass);
    const powerView = Boolean(state.powerView);
    if (lastTime !== null && time < lastTime) {
      angle = 0;
      fanAngle = 0;
      distance = 0;
      bypassDistance = 0;
    }
    const dt = lastTime === null ? 0 : Math.max(0, time - lastTime);
    lastTime = time;
    angle = (angle + dt * (1.15 + spool * 5.8)) % (Math.PI * 2);
    fanAngle = (fanAngle + dt * (.78 + spool * 3.6)) % (Math.PI * 2);
    distance += dt * (45 + spool * 100);
    bypassDistance += dt * (35 + spool * 68);
    svg.dataset.time = time.toFixed(4);
    svg.dataset.bypass = String(bypass);
    svg.dataset.powerView = String(powerView);
    container.dataset.bypass = String(bypass);

    for (const rotor of rotors) {
      if (rotor.fan && !bypass) continue;
      const a = (rotor.fan ? fanAngle : angle) + rotor.phase;
      const depth = rotor.fan ? 10 : 8;
      const point = (radius, theta, twist = 0) => [rotor.x + Math.sin(theta) * depth + twist, 160 + Math.cos(theta) * radius];
      rotor.blades.forEach((blade, index) => {
        const theta = a + index * Math.PI * 2 / rotor.blades.length;
        const sweep = rotor.hot ? -.18 : .18;
        const p1 = point(rotor.inner, theta - .11, -3);
        const p2 = point(rotor.radius, theta + sweep - .09, 2);
        const p3 = point(rotor.radius, theta + sweep + .09, 7);
        const p4 = point(rotor.inner, theta + .11, 3);
        blade.setAttribute('d', `M${p1[0].toFixed(2)} ${p1[1].toFixed(2)}L${p2[0].toFixed(2)} ${p2[1].toFixed(2)} ${p3[0].toFixed(2)} ${p3[1].toFixed(2)} ${p4[0].toFixed(2)} ${p4[1].toFixed(2)}Z`);
        blade.setAttribute('opacity', (Math.sin(theta) > 0 ? .95 : .65).toString());
      });
    }

    for (const particle of particles) {
      if (particle.bypass && !bypass) continue;
      const travelled = particle.bypass ? bypassDistance : distance;
      const u = ((particle.phase + travelled / particle.path.length) % 1) * 240;
      const index = Math.floor(u);
      const fraction = u - index;
      const p = particle.path.points[index];
      const n = particle.path.points[Math.min(index + 1, 240)];
      const x = p[0] + (n[0] - p[0]) * fraction;
      const y = p[1] + (n[1] - p[1]) * fraction + particle.offset;
      particle.circle.setAttribute('cx', x.toFixed(2));
      particle.circle.setAttribute('cy', y.toFixed(2));
      particle.circle.setAttribute('fill', particle.bypass || x < 532 ? '#349bdd' : '#e98438');
    }

    // A steady flame with gentle shape changes: combustion never becomes a
    // repeating on/off event as it does in the piston engine.
    const tip = 671 + spool * 12 + Math.sin(time * 4.1) * 4;
    upperFlame.setAttribute('d', `M531 115Q555 94 573 112Q598 94 613 113Q641 98 ${tip.toFixed(2)} 120Q646 138 616 128Q589 144 567 127Q545 135 531 115Z`);
    lowerFlame.setAttribute('d', `M531 205Q555 226 573 208Q598 226 613 207Q641 222 ${tip.toFixed(2)} 200Q646 182 616 192Q589 176 567 193Q545 185 531 205Z`);
    flames.setAttribute('opacity', (.68 + spool * .24).toFixed(2));
    plume.setAttribute('opacity', (.5 + spool * .5).toFixed(2));
  }

  select(selected);
  render({ time: 0, jet: .45, demand: .45, bypass: false, powerView: false });
  return { render, select };
}
