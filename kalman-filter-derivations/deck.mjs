import { routes } from './routes.mjs';

const WIDTH = 1280;
const HEIGHT = 720;
const SERIF = "Georgia, 'Times New Roman', serif";
const SANS = "Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
const MONO = "'SFMono-Regular', Consolas, 'Liberation Mono', monospace";

const C = {
  paper: '#F7F4EC',
  panel: '#FFFEFA',
  ink: '#172631',
  muted: '#627078',
  faint: '#889298',
  rule: '#D8D1C4',
  accent: '#C9684E',
  accentSoft: '#F3E0D9',
  deep: '#335D55',
  deepSoft: '#E0EAE6',
  blue: '#4F728A',
  blueSoft: '#E3EBF0',
  violet: '#74638E',
  violetSoft: '#ECE8F2',
  gold: '#8A6A31',
  goldSoft: '#F1E9D8',
  rose: '#8B5369',
  roseSoft: '#F1E3E9'
};

const familyPalette = {
  Probability: [C.deep, C.deepSoft],
  'Graphical models': [C.blue, C.blueSoft],
  Optimization: [C.accent, C.accentSoft],
  'Numerical linear algebra': [C.violet, C.violetSoft],
  Estimation: [C.gold, C.goldSoft],
  'Control duality': [C.rose, C.roseSoft],
  'Information theory': [C.deep, C.deepSoft]
};

const outcomes = {
  1: 'Canonical Gaussian parameters (J⁺, h⁺) are ready for the shared update.',
  2: 'Information adds first; covariance inversion can wait until the end.',
  3: 'A Schur complement produces the conditional Gaussian block.',
  4: 'The forward message stays Gaussian through prediction and correction.',
  5: 'Message passing sums canonical parameters at the state variable.',
  6: 'One Gaussian-process conditioning step recovers the temporal update.',
  7: 'The normal equations locate the posterior mode—and, here, its mean.',
  8: 'Woodbury turns a batch least-squares solve into a rank-m recursion.',
  9: 'QR reaches the same solution without forming squared covariance products.',
  10: 'The optimal residual is orthogonal to the innovation space.',
  11: 'The minimizing gain is fixed by one matrix normal equation.',
  12: 'Unbiasedness fixes the form; minimum covariance fixes the gain.',
  13: 'The innovation contributes exactly the new orthogonal component.',
  14: 'Covariance propagation closes as a Riccati map; LQR supplies its transpose-dual.',
  15: 'Gaussian closure compresses the general Bayes recursion to moments.',
  16: 'KL projection changes the prior only as much as the evidence requires.'
};

function text(id, x, y, w, h, html, options = {}) {
  return {
    id,
    type: 'text',
    x, y, w, h,
    rotation: 0,
    opacity: 1,
    html,
    fontSize: options.fontSize ?? 20,
    fontFamily: options.fontFamily ?? SANS,
    fontWeight: options.fontWeight ?? 400,
    color: options.color ?? C.ink,
    align: options.align ?? 'left',
    valign: options.valign ?? 'top',
    lineHeight: options.lineHeight ?? 1.25,
    ...(options.letterSpacing !== undefined ? { letterSpacing: options.letterSpacing } : {}),
    ...(options.link ? { link: options.link } : {}),
    ...(options.fx ? { fx: options.fx } : {})
  };
}

function shape(id, shapeName, x, y, w, h, options = {}) {
  return {
    id,
    type: 'shape',
    shape: shapeName,
    x, y, w, h,
    fill: options.fill ?? 'none',
    stroke: options.stroke ?? 'none',
    strokeWidth: options.strokeWidth ?? 0,
    radius: options.radius ?? 0,
    rotation: options.rotation ?? 0,
    opacity: options.opacity ?? 1,
    ...(options.lineStart ? { lineStart: options.lineStart } : {}),
    ...(options.lineEnd ? { lineEnd: options.lineEnd } : {}),
    ...(options.fx ? { fx: options.fx } : {})
  };
}

function card(id, x, y, w, h, fill = C.panel, stroke = C.rule, radius = 18) {
  return shape(id, 'rect', x, y, w, h, { fill, stroke, strokeWidth: 1, radius });
}

function chrome(section, page, accent = C.accent) {
  return [
    shape('chrome-rule', 'rect', 72, 674, 1136, 1, { fill: C.rule }),
    text('chrome-site', 72, 686, 300, 18, 'BAI LIPING · KALMAN DERIVATION ATLAS', {
      fontSize: 10.5, fontFamily: MONO, fontWeight: 700, color: C.faint, letterSpacing: 0.7
    }),
    text('chrome-section', 456, 684, 368, 20, section.toUpperCase(), {
      fontSize: 10.5, fontFamily: MONO, fontWeight: 800, color: accent, align: 'center', letterSpacing: 1.1
    }),
    text('chrome-count', 1080, 684, 128, 20, String(page).padStart(2, '0'), {
      fontSize: 11, fontFamily: MONO, fontWeight: 800, color: C.muted, align: 'right'
    })
  ];
}

function heading(eyebrow, title, page, accent = C.accent, subtitle = '') {
  const elements = [
    text('slide-eyebrow', 72, 46, 880, 24, eyebrow.toUpperCase(), {
      fontSize: 12, fontFamily: MONO, fontWeight: 800, color: accent, letterSpacing: 1.7
    }),
    text('slide-title', 72, 76, 1032, 68, title, {
      fontSize: 43, fontFamily: SERIF, fontWeight: 700, color: C.ink, lineHeight: 1.04
    }),
    card('slide-number-card', 1124, 45, 84, 84, C.panel, accent, 17),
    text('slide-number-label', 1124, 58, 84, 18, 'PAGE', {
      fontSize: 9, fontFamily: MONO, fontWeight: 800, color: accent, align: 'center', letterSpacing: 1.2
    }),
    text('slide-number-value', 1124, 79, 84, 36, String(page).padStart(2, '0'), {
      fontSize: 25, fontFamily: SERIF, fontWeight: 700, color: accent, align: 'center'
    })
  ];
  if (subtitle) {
    elements.push(text('slide-subtitle', 72, 136, 1040, 45, subtitle, {
      fontSize: 17, color: C.muted, lineHeight: 1.35
    }));
  }
  return elements;
}

function coverSlide() {
  const elements = [
    shape('cover-accent', 'rect', 0, 0, 18, HEIGHT, { fill: C.accent }),
    text('cover-eyebrow', 88, 86, 720, 28, 'LINEAR–GAUSSIAN ESTIMATION · A DERIVATION ATLAS', {
      fontSize: 13, fontFamily: MONO, fontWeight: 800, color: C.deep, letterSpacing: 1.8,
      fx: { enter: 'fade-up', order: 0 }
    }),
    text('cover-title', 86, 132, 700, 205, "One filter.<br><span style='color:#C9684E'>Sixteen derivations.</span>", {
      fontSize: 72, fontFamily: SERIF, fontWeight: 700, lineHeight: 0.96,
      fx: { enter: 'fade-up', order: 1 }
    }),
    text('cover-subtitle', 90, 365, 650, 110, 'Probability, optimization, estimation, numerical linear algebra, graphical models, control, and information theory all describe one update.', {
      fontSize: 22, color: C.muted, lineHeight: 1.38,
      fx: { enter: 'fade-up', order: 2 }
    }),
    shape('cover-rule', 'line', 90, 512, 620, 2, {
      fill: C.accent, lineEnd: 'arrow', fx: { loop: { type: 'dash-march' } }
    }),
    text('cover-cue', 90, 535, 620, 42, 'Follow the roads separately. Meet the posterior once.', {
      fontSize: 16, fontFamily: MONO, fontWeight: 700, color: C.deep
    }),
    shape('cover-orbit-outer', 'ellipse', 810, 112, 350, 350, { fill: C.deepSoft, stroke: C.deep, strokeWidth: 1 }),
    shape('cover-orbit-mid', 'ellipse', 864, 166, 242, 242, { fill: C.panel, stroke: C.accent, strokeWidth: 2 }),
    shape('cover-orbit-core', 'ellipse', 916, 218, 138, 138, { fill: C.accentSoft, stroke: C.accent, strokeWidth: 2 }),
    text('cover-core-label', 916, 244, 138, 85, 'p(x | z)', {
      fontSize: 33, fontFamily: SERIF, fontWeight: 700, color: C.ink, align: 'center', valign: 'middle'
    })
  ];
  const labels = [
    ['Bayes', 776, 83], ['GP', 1084, 91], ['WLS', 1120, 230], ['BLUE', 1081, 421],
    ['QR', 907, 472], ['LMMSE', 748, 401], ['graphs', 724, 237], ['entropy', 917, 70]
  ];
  labels.forEach(([label, x, y], i) => {
    elements.push(card(`cover-pill-${i}`, x, y, 86, 34, C.panel, C.rule, 17));
    elements.push(text(`cover-pill-label-${i}`, x, y + 1, 86, 32, label, {
      fontSize: 11, fontFamily: MONO, fontWeight: 800, color: C.muted, align: 'center', valign: 'middle'
    }));
  });
  elements.push(...chrome('Start', 1, C.accent));
  return {
    id: 'cover', background: C.paper, transition: 'none',
    notes: 'Open with the central claim: the sixteen derivations are not sixteen filters. They are sixteen coordinate systems for one linear–Gaussian correction. The deck deliberately postpones the common posterior formula until the routes have converged.',
    elements
  };
}

function modelSlide() {
  const elements = [
    ...heading('The common problem', 'One update, stated once.', 2, C.deep, 'Fix the model first. Every derivation starts with the same prior and the same noisy linear observation.'),
    card('model-prior-card', 72, 214, 326, 180, C.deepSoft, C.deep, 18),
    text('model-prior-label', 96, 238, 278, 22, 'PRIOR', { fontSize: 11, fontFamily: MONO, fontWeight: 800, color: C.deep, letterSpacing: 1.3 }),
    text('model-prior-eq', 96, 278, 278, 52, 'x ∼ 𝒩(m⁻, P⁻)', { fontSize: 31, fontFamily: SERIF, fontWeight: 700, align: 'center' }),
    text('model-prior-copy', 96, 346, 278, 32, 'What we believe before z arrives.', { fontSize: 14, color: C.muted, align: 'center' }),
    shape('model-arrow-a', 'line', 414, 300, 84, 2, { fill: C.accent, lineEnd: 'arrow' }),
    card('model-likelihood-card', 514, 214, 326, 180, C.accentSoft, C.accent, 18),
    text('model-like-label', 538, 238, 278, 22, 'MEASUREMENT MODEL', { fontSize: 11, fontFamily: MONO, fontWeight: 800, color: C.accent, letterSpacing: 1.3 }),
    text('model-like-eq', 538, 278, 278, 52, 'z = Hx + v', { fontSize: 31, fontFamily: SERIF, fontWeight: 700, align: 'center' }),
    text('model-like-copy', 538, 342, 278, 38, 'v ∼ 𝒩(0, R), independent of x.', { fontSize: 14, color: C.muted, align: 'center' }),
    shape('model-arrow-b', 'line', 856, 300, 84, 2, { fill: C.accent, lineEnd: 'arrow' }),
    card('model-question-card', 956, 214, 252, 180, C.panel, C.rule, 18),
    text('model-question-label', 980, 238, 204, 22, 'QUESTION', { fontSize: 11, fontFamily: MONO, fontWeight: 800, color: C.blue, letterSpacing: 1.3 }),
    text('model-question-eq', 980, 278, 204, 52, 'p(x | z) = ?', { fontSize: 31, fontFamily: SERIF, fontWeight: 700, align: 'center' }),
    text('model-question-copy', 980, 342, 204, 38, 'Correct the belief with one observation.', { fontSize: 14, color: C.muted, align: 'center' }),
    card('model-notation-card', 72, 430, 1136, 190, C.panel, C.rule, 18),
    text('model-notation-title', 96, 452, 240, 24, 'THE SHARED OBJECTS', { fontSize: 11, fontFamily: MONO, fontWeight: 800, color: C.deep, letterSpacing: 1.3 }),
    text('model-notation-innovation', 96, 495, 300, 55, 'ν = z − Hm⁻', { fontSize: 28, fontFamily: SERIF, fontWeight: 700 }),
    text('model-notation-innovation-copy', 96, 552, 300, 35, 'innovation · the unexplained part of z', { fontSize: 13.5, color: C.muted }),
    text('model-notation-cross', 452, 495, 310, 55, 'Cov(x, ν) = P⁻Hᵀ', { fontSize: 26, fontFamily: SERIF, fontWeight: 700 }),
    text('model-notation-cross-copy', 452, 552, 310, 35, 'how the measurement direction reaches the state', { fontSize: 13.5, color: C.muted }),
    text('model-notation-variance', 816, 495, 340, 55, 'Var(ν) = HP⁻Hᵀ + R', { fontSize: 26, fontFamily: SERIF, fontWeight: 700 }),
    text('model-notation-variance-copy', 816, 552, 340, 35, 'predicted uncertainty plus measurement noise', { fontSize: 13.5, color: C.muted }),
    ...chrome('Setup', 2, C.deep)
  ];
  return {
    id: 'model', background: C.paper, transition: 'morph',
    notes: 'Define the prior, measurement model, and innovation. Stress that every route receives the same ingredients. Do not derive the final posterior here; the deck will show it once after all sixteen roads.',
    elements
  };
}

function atlasSlide() {
  const elements = [
    ...heading('Atlas', 'Sixteen roads, grouped by lens.', 3, C.blue, 'Choose the route that best exposes the structure you care about. The three experiments begin only after the roads converge.'),
  ];
  const xs = [72, 360, 648, 936];
  const ys = [190, 288, 386, 484];
  routes.forEach((route, index) => {
    const col = index % 4;
    const row = Math.floor(index / 4);
    const [accent, soft] = familyPalette[route.family];
    const x = xs[col], y = ys[row];
    elements.push(card(`atlas-card-${route.n}`, x, y, 272, 82, soft, accent, 14));
    elements.push(text(`atlas-number-${route.n}`, x + 14, y + 12, 42, 20, String(route.n).padStart(2, '0'), {
      fontSize: 11, fontFamily: MONO, fontWeight: 900, color: accent, link: `route-${String(route.n).padStart(2, '0')}`
    }));
    elements.push(text(`atlas-family-${route.n}`, x + 58, y + 11, 196, 19, route.family.toUpperCase(), {
      fontSize: 8.5, fontFamily: MONO, fontWeight: 800, color: accent, letterSpacing: 0.65, link: `route-${String(route.n).padStart(2, '0')}`
    }));
    elements.push(text(`atlas-title-${route.n}`, x + 14, y + 36, 244, 36, route.title, {
      fontSize: 14, fontFamily: SERIF, fontWeight: 700, color: C.ink, lineHeight: 1.12, link: `route-${String(route.n).padStart(2, '0')}`
    }));
  });
  const labLinks = [
    ['SHARED POSTERIOR', 'Scalar fusion', 'shared-posterior', C.accent],
    ['GEOMETRY', 'Rotate the measurement', 'covariance-geometry', C.deep],
    ['NUMERICS', 'Stress finite precision', 'finite-precision', C.violet]
  ];
  labLinks.forEach(([label, title, link, accent], index) => {
    const x = 72 + index * 380;
    elements.push(shape(`atlas-lab-line-${index}`, 'rect', x, 596, 348, 3, { fill: accent }));
    elements.push(text(`atlas-lab-label-${index}`, x, 608, 130, 20, label, { fontSize: 9, fontFamily: MONO, fontWeight: 800, color: accent, letterSpacing: 0.9, link }));
    elements.push(text(`atlas-lab-title-${index}`, x + 132, 603, 216, 30, title, { fontSize: 15, fontFamily: SERIF, fontWeight: 700, align: 'right', valign: 'middle', link }));
  });
  elements.push(...chrome('Atlas', 3, C.blue));
  return {
    id: 'atlas', background: C.paper, transition: 'morph',
    notes: 'Use the atlas as a navigation slide. The cards are links to ordinary route slides. The three experiment links point to their introductory slides—not directly to the live slides—so the audience always gets context first.',
    elements
  };
}

function routeSlide(route, page) {
  const [accent, soft] = familyPalette[route.family];
  const longestStep = Math.max(...route.steps.map(step => step.replace(/<[^>]+>/g, '').length));
  const stepSize = longestStep > 155 ? 14.5 : longestStep > 118 ? 15.5 : 16.5;
  const elements = [
    shape('route-accent-rail', 'rect', 0, 0, 14, HEIGHT, { fill: accent }),
    text('route-eyebrow', 72, 45, 850, 25, `ROAD ${String(route.n).padStart(2, '0')} · ${route.family.toUpperCase()}`, {
      fontSize: 12, fontFamily: MONO, fontWeight: 800, color: accent, letterSpacing: 1.5
    }),
    text('route-title', 72, 78, 1024, 70, route.title, {
      fontSize: route.title.length > 42 ? 36 : route.title.length > 38 ? 39 : 44, fontFamily: SERIF, fontWeight: 700, lineHeight: 1.04
    }),
    card('route-number-card', 1124, 45, 84, 84, soft, accent, 17),
    text('route-number-label', 1124, 58, 84, 18, 'ROAD', { fontSize: 9, fontFamily: MONO, fontWeight: 800, color: accent, align: 'center', letterSpacing: 1.2 }),
    text('route-number-value', 1124, 79, 84, 36, String(route.n).padStart(2, '0'), { fontSize: 25, fontFamily: SERIF, fontWeight: 700, color: accent, align: 'center' }),
    card('route-lens-card', 72, 176, 344, 360, soft, accent, 18),
    text('route-lens-label', 96, 200, 296, 22, 'VIEWPOINT', { fontSize: 10, fontFamily: MONO, fontWeight: 800, color: accent, letterSpacing: 1.2 }),
    text('route-lens-copy', 96, 242, 296, 196, route.idea, { fontSize: 20, fontFamily: SERIF, fontWeight: 700, lineHeight: 1.34 }),
    shape('route-lens-rule', 'rect', 96, 463, 296, 2, { fill: accent }),
    text('route-lens-question', 96, 482, 296, 40, 'What does this language make obvious?', { fontSize: 13.5, fontFamily: MONO, fontWeight: 700, color: C.muted, lineHeight: 1.35 }),
    card('route-spine-card', 448, 176, 760, 360, C.panel, C.rule, 18),
    text('route-spine-label', 474, 198, 704, 22, 'DERIVATION SPINE', { fontSize: 10, fontFamily: MONO, fontWeight: 800, color: accent, letterSpacing: 1.2 })
  ];
  route.steps.forEach((step, index) => {
    const y = 234 + index * 94;
    elements.push(card(`route-step-card-${index}`, 474, y, 704, 78, '#FBFAF6', C.rule, 12));
    elements.push(shape(`route-step-dot-${index}`, 'ellipse', 490, y + 22, 34, 34, { fill: soft, stroke: accent, strokeWidth: 1 }));
    elements.push(text(`route-step-number-${index}`, 490, y + 22, 34, 34, String(index + 1), { fontSize: 13, fontFamily: MONO, fontWeight: 900, color: accent, align: 'center', valign: 'middle' }));
    elements.push(text(`route-step-copy-${index}`, 540, y + 10, 616, 58, step, { fontSize: stepSize, color: C.ink, valign: 'middle', lineHeight: 1.28 }));
  });
  elements.push(
    card('route-handoff-card', 72, 558, 1136, 76, C.panel, accent, 14),
    text('route-handoff-label', 94, 580, 126, 22, 'HANDOFF', { fontSize: 10, fontFamily: MONO, fontWeight: 900, color: accent, letterSpacing: 1.1 }),
    text('route-handoff-copy', 226, 570, 956, 46, outcomes[route.n], { fontSize: 17, fontFamily: SERIF, fontWeight: 700, color: C.ink, valign: 'middle' }),
    text('route-reference', 72, 640, 1136, 28, `<b>REFERENCE ·</b> ${route.ref}`, { fontSize: route.ref.length > 150 ? 9.3 : 10.5, color: C.faint, lineHeight: 1.2 }),
    ...chrome(route.family, page, accent)
  );
  return {
    id: `route-${String(route.n).padStart(2, '0')}`,
    background: C.paper,
    transition: 'morph',
    notes: `Road ${route.n}: ${route.title}. Start from the ${route.family.toLowerCase()} viewpoint, walk the three displayed steps, and stop at the handoff. Do not repeat the common posterior formula; it appears once after all routes. Representative reference: ${route.ref}`,
    elements
  };
}

function sharedPosteriorSlide(page) {
  const elements = [
    ...heading('Convergence · experiment intro', 'Sixteen roads. One posterior.', page, C.accent, 'The routes now collapse onto one correction. This is the deck’s single shared destination formula.'),
    card('posterior-flow-card', 72, 198, 320, 408, C.deepSoft, C.deep, 20),
    text('posterior-flow-label', 98, 224, 268, 22, 'MENTAL MODEL', { fontSize: 10, fontFamily: MONO, fontWeight: 900, color: C.deep, letterSpacing: 1.2 }),
    card('posterior-prior-chip', 104, 274, 252, 72, C.panel, C.rule, 14),
    text('posterior-prior-title', 124, 286, 212, 24, 'prior belief', { fontSize: 14, fontFamily: MONO, fontWeight: 800, color: C.muted, align: 'center' }),
    text('posterior-prior-eq', 124, 311, 212, 26, '𝒩(m⁻, P⁻)', { fontSize: 20, fontFamily: SERIF, fontWeight: 700, align: 'center' }),
    text('posterior-flow-arrow-a', 188, 348, 84, 48, '↓', { fontSize: 30, fontFamily: SERIF, fontWeight: 700, color: C.accent, align: 'center', valign: 'middle' }),
    card('posterior-evidence-chip', 104, 404, 252, 72, C.panel, C.rule, 14),
    text('posterior-evidence-title', 124, 416, 212, 24, 'innovation weighted by trust', { fontSize: 13, fontFamily: MONO, fontWeight: 800, color: C.muted, align: 'center' }),
    text('posterior-evidence-eq', 124, 441, 212, 26, 'Kν', { fontSize: 22, fontFamily: SERIF, fontWeight: 700, color: C.accent, align: 'center' }),
    text('posterior-flow-arrow-b', 188, 478, 84, 48, '↓', { fontSize: 30, fontFamily: SERIF, fontWeight: 700, color: C.accent, align: 'center', valign: 'middle' }),
    card('posterior-result-chip', 104, 534, 252, 52, C.accentSoft, C.accent, 14),
    text('posterior-result-label', 124, 544, 212, 34, 'corrected belief', { fontSize: 16, fontFamily: SERIF, fontWeight: 700, color: C.accent, align: 'center', valign: 'middle' }),
    card('posterior-equations-card', 424, 198, 784, 330, C.panel, C.rule, 20),
    text('posterior-equations-label', 454, 222, 724, 22, 'THE SHARED POSTERIOR · SHOWN ONCE', { fontSize: 10, fontFamily: MONO, fontWeight: 900, color: C.accent, letterSpacing: 1.25 }),
    text('posterior-equation-s', 454, 268, 724, 44, 'S = HP⁻Hᵀ + R', { fontSize: 25, fontFamily: SERIF, fontWeight: 700 }),
    text('posterior-equation-k', 454, 319, 724, 44, 'K = P⁻HᵀS⁻¹', { fontSize: 25, fontFamily: SERIF, fontWeight: 700 }),
    text('posterior-equation-m', 454, 370, 724, 44, 'm⁺ = m⁻ + K(z − Hm⁻)', { fontSize: 25, fontFamily: SERIF, fontWeight: 700, color: C.accent }),
    text('posterior-equation-p', 454, 421, 724, 44, 'P⁺ = P⁻ − KSKᵀ', { fontSize: 25, fontFamily: SERIF, fontWeight: 700 }),
    text('posterior-equations-note', 454, 477, 724, 30, 'For implementation, the Joseph or square-root form may preserve this covariance more reliably.', { fontSize: 13.5, color: C.muted }),
    card('posterior-watch-card', 424, 548, 784, 58, C.accentSoft, C.accent, 14),
    text('posterior-watch-label', 446, 560, 130, 34, 'WATCH NEXT', { fontSize: 10, fontFamily: MONO, fontWeight: 900, color: C.accent, valign: 'middle', letterSpacing: 1.1 }),
    text('posterior-watch-copy', 584, 555, 596, 42, 'Increase one source’s certainty: the common posterior should move toward it and narrow.', { fontSize: 16, fontFamily: SERIF, fontWeight: 700, valign: 'middle' }),
    ...chrome('Shared posterior', page, C.accent)
  ];
  return {
    id: 'shared-posterior', background: C.paper, transition: 'morph',
    notes: 'This is the only slide that presents the complete shared posterior formula. Connect the gain to a trust-weighted innovation. Tell the audience to watch how greater certainty pulls and narrows the posterior on the next slide.',
    elements
  };
}

function scalarFallback(bounds) {
  const { x, y, width, height } = bounds;
  const elements = [
    card('scalar-fallback-region', x, y, width, height, '#FBFAF6', C.rule, 15),
    card('scalar-fallback-controls', x + 16, y + 16, 280, height - 32, C.deepSoft, C.rule, 12),
    text('scalar-fallback-controls-title', x + 36, y + 34, 240, 26, 'DEFAULT EXPERIMENT', { fontSize: 10, fontFamily: MONO, fontWeight: 900, color: C.deep, letterSpacing: 1.1 }),
    text('scalar-fallback-controls-copy', x + 36, y + 72, 240, 74, 'prior  m⁻ = −1.2,  σₚ = 1.35<br>measure  z = 2.1,  σᵣ = 0.75', { fontSize: 16, fontFamily: SERIF, fontWeight: 700, lineHeight: 1.5 }),
    text('scalar-fallback-controls-hint', x + 36, y + 378, 240, 64, 'Live controls replace this panel in presentation mode.', { fontSize: 13, color: C.muted, lineHeight: 1.4 }),
    card('scalar-fallback-plot', x + 314, y + 16, width - 330, height - 32, C.panel, C.rule, 12),
    text('scalar-fallback-plot-title', x + 338, y + 34, 500, 24, 'PRIOR × LIKELIHOOD → ONE POSTERIOR', { fontSize: 10, fontFamily: MONO, fontWeight: 900, color: C.accent, letterSpacing: 1.1 }),
    shape('scalar-axis', 'line', x + 356, y + 330, 610, 2, { fill: C.muted, lineEnd: 'arrow' }),
    shape('scalar-prior-band', 'ellipse', x + 440, y + 192, 260, 138, { fill: C.blueSoft, stroke: C.blue, strokeWidth: 3 }),
    shape('scalar-like-band', 'ellipse', x + 724, y + 164, 164, 166, { fill: C.goldSoft, stroke: C.gold, strokeWidth: 3 }),
    shape('scalar-post-band', 'ellipse', x + 667, y + 108, 124, 222, { fill: C.deepSoft, stroke: C.deep, strokeWidth: 5 }),
    text('scalar-prior-label', x + 450, y + 346, 230, 26, 'prior · m⁻ = −1.2', { fontSize: 13, fontFamily: MONO, fontWeight: 800, color: C.blue, align: 'center' }),
    text('scalar-like-label', x + 718, y + 346, 180, 26, 'likelihood · z = 2.1', { fontSize: 13, fontFamily: MONO, fontWeight: 800, color: C.gold, align: 'center' }),
    card('scalar-posterior-metric', x + 922, y + 82, 164, 124, C.accentSoft, C.accent, 12),
    text('scalar-posterior-metric-label', x + 940, y + 98, 128, 20, 'POSTERIOR', { fontSize: 9, fontFamily: MONO, fontWeight: 900, color: C.accent, align: 'center', letterSpacing: 1 }),
    text('scalar-posterior-metric-value', x + 940, y + 126, 128, 62, 'm⁺ = 1.322<br>σ⁺ = 0.656', { fontSize: 17, fontFamily: SERIF, fontWeight: 700, align: 'center', lineHeight: 1.45 }),
    text('scalar-fallback-agreement', x + 930, y + 246, 148, 54, 'agreement<br><b>Δ = 0</b>', { fontSize: 14, color: C.deep, align: 'center', lineHeight: 1.45 })
  ];
  return elements;
}

function geometryFallback(bounds) {
  const { x, y, width, height } = bounds;
  return [
    card('geometry-fallback-region', x, y, width, height, '#FBFAF6', C.rule, 15),
    card('geometry-fallback-controls', x + 16, y + 16, 280, height - 32, C.deepSoft, C.rule, 12),
    text('geometry-fallback-controls-title', x + 36, y + 34, 240, 26, 'DEFAULT COVARIANCE', { fontSize: 10, fontFamily: MONO, fontWeight: 900, color: C.deep, letterSpacing: 1.1 }),
    text('geometry-fallback-controls-copy', x + 36, y + 75, 240, 138, 'σₓ = 1.8<br>σᵧ = 1.0<br>ρ = 0.65<br>measurement angle = 32°', { fontSize: 16, fontFamily: SERIF, fontWeight: 700, lineHeight: 1.55 }),
    text('geometry-fallback-controls-hint', x + 36, y + 378, 240, 64, 'Rotate H live and watch uncertainty contract normal to the measured slice.', { fontSize: 13, color: C.muted, lineHeight: 1.4 }),
    card('geometry-fallback-plot', x + 314, y + 16, width - 330, height - 32, C.panel, C.rule, 12),
    text('geometry-fallback-plot-title', x + 338, y + 34, 500, 24, '2D PRIOR → MEASURED SLICE → POSTERIOR', { fontSize: 10, fontFamily: MONO, fontWeight: 900, color: C.deep, letterSpacing: 1.1 }),
    shape('geometry-prior-ellipse', 'ellipse', x + 450, y + 126, 390, 252, { fill: C.blueSoft, stroke: C.blue, strokeWidth: 4, rotation: 24 }),
    shape('geometry-posterior-ellipse', 'ellipse', x + 548, y + 176, 218, 144, { fill: C.deepSoft, stroke: C.deep, strokeWidth: 5, rotation: 24 }),
    shape('geometry-measurement-line', 'rect', x + 438, y + 274, 450, 4, { fill: C.accent, rotation: -30 }),
    text('geometry-prior-label', x + 438, y + 392, 210, 28, 'prior covariance P⁻', { fontSize: 13, fontFamily: MONO, fontWeight: 800, color: C.blue }),
    text('geometry-post-label', x + 710, y + 332, 210, 28, 'posterior covariance P⁺', { fontSize: 13, fontFamily: MONO, fontWeight: 800, color: C.deep }),
    card('geometry-metric', x + 920, y + 94, 170, 142, C.accentSoft, C.accent, 12),
    text('geometry-metric-label', x + 940, y + 110, 130, 20, 'OBSERVE', { fontSize: 9, fontFamily: MONO, fontWeight: 900, color: C.accent, align: 'center', letterSpacing: 1 }),
    text('geometry-metric-copy', x + 940, y + 144, 130, 72, 'Only the measured direction contracts strongly.', { fontSize: 16, fontFamily: SERIF, fontWeight: 700, align: 'center', lineHeight: 1.35 })
  ];
}

function equivalenceFallback(bounds) {
  const { x, y, width, height } = bounds;
  const methods = [
    ['Covariance', 'P − KSKᵀ', C.accent, C.accentSoft],
    ['Information', '(P⁻¹ + HᵀR⁻¹H)⁻¹', C.blue, C.blueSoft],
    ['Joseph', '(I−KH)P(I−KH)ᵀ + KRKᵀ', C.deep, C.deepSoft],
    ['QR / square root', 'whiten → QR → solve', C.violet, C.violetSoft]
  ];
  const elements = [
    card('equivalence-fallback-region', x, y, width, height, '#FBFAF6', C.rule, 15),
    card('equivalence-fallback-controls', x + 16, y + 16, 280, height - 32, C.violetSoft, C.rule, 12),
    text('equivalence-fallback-controls-title', x + 36, y + 34, 240, 26, 'DEFAULT STRESS TEST', { fontSize: 10, fontFamily: MONO, fontWeight: 900, color: C.violet, letterSpacing: 1.1 }),
    text('equivalence-fallback-controls-copy', x + 36, y + 75, 240, 116, 'state dimension  n = 3<br>measurement dimension  m = 1<br>cond(P⁻) = 10³<br>precision = native double', { fontSize: 15, fontFamily: SERIF, fontWeight: 700, lineHeight: 1.5 }),
    text('equivalence-fallback-controls-hint', x + 36, y + 378, 240, 64, 'Lower the simulated precision to make algebraically equal forms separate numerically.', { fontSize: 13, color: C.muted, lineHeight: 1.4 }),
    card('equivalence-fallback-grid', x + 314, y + 16, width - 330, height - 32, C.panel, C.rule, 12),
    text('equivalence-fallback-grid-title', x + 338, y + 34, 650, 24, 'ONE TARGET MATRIX · FOUR ARITHMETIC PATHS', { fontSize: 10, fontFamily: MONO, fontWeight: 900, color: C.violet, letterSpacing: 1.1 })
  ];
  methods.forEach(([name, formula, accent, soft], index) => {
    const col = index % 2, row = Math.floor(index / 2);
    const mx = x + 338 + col * 375, my = y + 82 + row * 162;
    elements.push(card(`equivalence-method-card-${index}`, mx, my, 348, 138, soft, accent, 12));
    elements.push(text(`equivalence-method-name-${index}`, mx + 18, my + 16, 310, 24, name, { fontSize: 17, fontFamily: SERIF, fontWeight: 700, color: accent }));
    elements.push(text(`equivalence-method-formula-${index}`, mx + 18, my + 52, 310, 38, formula, { fontSize: 13, fontFamily: MONO, fontWeight: 700, color: C.ink, valign: 'middle' }));
    elements.push(text(`equivalence-method-status-${index}`, mx + 18, my + 102, 310, 20, 'max Δ ≈ 0 · positive definite', { fontSize: 10.5, fontFamily: MONO, fontWeight: 800, color: C.deep }));
  });
  return elements;
}

function liveSlide({ id, page, eyebrow, title, prompt, accent, demo, fallback }) {
  const bounds = { x: 74, y: 153, width: 1132, height: 494 };
  const elements = [
    text('live-eyebrow', 74, 40, 850, 24, eyebrow.toUpperCase(), { fontSize: 11, fontFamily: MONO, fontWeight: 900, color: accent, letterSpacing: 1.5 }),
    text('live-title', 74, 70, 1030, 52, title, { fontSize: title.length > 44 ? 32 : 38, fontFamily: SERIF, fontWeight: 700, lineHeight: 1.04 }),
    text('live-prompt', 74, 122, 1030, 28, prompt, { fontSize: 15.5, color: C.muted }),
    ...fallback(bounds),
    shape('live-demo-mount', 'rect', bounds.x, bounds.y, bounds.width, bounds.height, { fill: 'rgba(255,255,255,0)', stroke: 'rgba(255,255,255,0)', strokeWidth: 0, opacity: 0 }),
    ...chrome('Experiment', page, accent)
  ];
  return {
    id, background: C.paper, transition: 'morph',
    notes: `Run the ${demo} experiment. The live region mounts automatically. Use the controls, then press Escape to return focus to Bento. Page Up returns to the introductory slide; Page Down advances.`,
    elements
  };
}

function geometryIntroSlide(page) {
  const elements = [
    ...heading('Geometry · experiment intro', 'A measurement contracts a direction.', page, C.deep, 'In two dimensions, H selects a scalar slice. Correlation decides how far that information propagates across the state.'),
    card('geometry-intro-visual', 72, 205, 650, 408, C.panel, C.rule, 20),
    shape('geometry-intro-prior', 'ellipse', 174, 286, 380, 220, { fill: C.blueSoft, stroke: C.blue, strokeWidth: 4, rotation: 22 }),
    shape('geometry-intro-post', 'ellipse', 274, 328, 190, 134, { fill: C.deepSoft, stroke: C.deep, strokeWidth: 5, rotation: 22 }),
    shape('geometry-intro-line', 'rect', 142, 382, 470, 4, { fill: C.accent, rotation: -31 }),
    text('geometry-intro-h', 514, 240, 150, 34, 'measurement H', { fontSize: 13, fontFamily: MONO, fontWeight: 800, color: C.accent, align: 'right' }),
    text('geometry-intro-prior-label', 112, 530, 240, 28, 'prior ellipse P⁻', { fontSize: 14, fontFamily: MONO, fontWeight: 800, color: C.blue }),
    text('geometry-intro-post-label', 438, 486, 230, 28, 'contracted ellipse P⁺', { fontSize: 14, fontFamily: MONO, fontWeight: 800, color: C.deep, align: 'right' }),
    card('geometry-intro-copy', 754, 205, 454, 408, C.deepSoft, C.deep, 20),
    text('geometry-intro-copy-label', 782, 232, 398, 22, 'WHAT TO WATCH', { fontSize: 10, fontFamily: MONO, fontWeight: 900, color: C.deep, letterSpacing: 1.2 }),
    text('geometry-intro-gain', 782, 278, 398, 55, 'K = Cov(x, ν) Var(ν)⁻¹', { fontSize: 25, fontFamily: SERIF, fontWeight: 700 }),
    text('geometry-intro-copy-a', 782, 355, 398, 62, '<b>Rotate H.</b> The direction of strongest contraction rotates with it.', { fontSize: 17, lineHeight: 1.42 }),
    text('geometry-intro-copy-b', 782, 435, 398, 62, '<b>Increase R.</b> The measurement becomes less trusted, so contraction weakens.', { fontSize: 17, lineHeight: 1.42 }),
    text('geometry-intro-copy-c', 782, 515, 398, 62, '<b>Change ρ.</b> Cross-covariance carries the scalar evidence into the unmeasured coordinate.', { fontSize: 17, lineHeight: 1.42 }),
    ...chrome('Geometry', page, C.deep)
  ];
  return {
    id: 'covariance-geometry', background: C.paper, transition: 'morph',
    notes: 'Explain the covariance ellipse as a geometric object. The measurement is a one-dimensional slice; correlation transports information to the other coordinate. Ask the audience to watch rotation, noise, and correlation on the next slide.',
    elements
  };
}

function finitePrecisionIntroSlide(page) {
  const elements = [
    ...heading('Numerics · experiment intro', 'Equal algebra is not equal arithmetic.', page, C.violet, 'All formulations target the same P⁺. Finite precision decides which route reaches it most reliably.'),
    card('precision-target-card', 72, 216, 250, 370, C.violetSoft, C.violet, 20),
    text('precision-target-label', 98, 242, 198, 22, 'ONE TARGET', { fontSize: 10, fontFamily: MONO, fontWeight: 900, color: C.violet, align: 'center', letterSpacing: 1.2 }),
    shape('precision-target-orbit', 'ellipse', 118, 302, 158, 158, { fill: C.panel, stroke: C.violet, strokeWidth: 3 }),
    text('precision-target-p', 118, 344, 158, 74, 'P⁺', { fontSize: 48, fontFamily: SERIF, fontWeight: 700, color: C.violet, align: 'center', valign: 'middle' }),
    text('precision-target-copy', 98, 492, 198, 58, 'The estimator is unchanged.<br>The arithmetic path is not.', { fontSize: 15, fontFamily: SERIF, fontWeight: 700, align: 'center', lineHeight: 1.4 }),
    shape('precision-arrow-a', 'line', 340, 398, 74, 2, { fill: C.violet, lineEnd: 'arrow' }),
    card('precision-paths-card', 432, 216, 776, 370, C.panel, C.rule, 20),
    text('precision-paths-label', 458, 240, 724, 22, 'FOUR ARITHMETIC PATHS', { fontSize: 10, fontFamily: MONO, fontWeight: 900, color: C.violet, letterSpacing: 1.2 })
  ];
  const paths = [
    ['Covariance subtraction', 'fast · can lose symmetry / PSD', C.accent, C.accentSoft],
    ['Information inversion', 'natural for sparse evidence', C.blue, C.blueSoft],
    ['Joseph stabilization', 'more products · better PSD behavior', C.deep, C.deepSoft],
    ['Square root / QR', 'factorize · avoid explicit covariance products', C.violet, C.violetSoft]
  ];
  paths.forEach(([name, copy, accent, soft], index) => {
    const y = 282 + index * 66;
    elements.push(card(`precision-path-card-${index}`, 458, y, 724, 52, soft, accent, 10));
    elements.push(text(`precision-path-name-${index}`, 478, y + 7, 292, 38, name, { fontSize: 16, fontFamily: SERIF, fontWeight: 700, color: accent, valign: 'middle' }));
    elements.push(text(`precision-path-copy-${index}`, 790, y + 7, 370, 38, copy, { fontSize: 13.5, fontFamily: MONO, fontWeight: 700, color: C.muted, align: 'right', valign: 'middle' }));
  });
  elements.push(
    card('precision-watch-card', 432, 608, 776, 42, C.violetSoft, C.violet, 12),
    text('precision-watch-copy', 452, 612, 736, 34, 'WATCH NEXT · raise cond(P⁻), then lower significant digits.', { fontSize: 12.5, fontFamily: MONO, fontWeight: 900, color: C.violet, align: 'center', valign: 'middle', letterSpacing: 0.7 }),
    ...chrome('Numerics', page, C.violet)
  );
  return {
    id: 'finite-precision', background: C.paper, transition: 'morph',
    notes: 'Separate mathematical equivalence from numerical behavior. All four forms target one covariance. On the live slide, make the prior ill-conditioned and then lower the simulated precision until subtraction or inversion begins to separate.',
    elements
  };
}

function synthesisSlide(page) {
  const elements = [
    ...heading('Synthesis', 'Different languages. Same estimator.', page, C.accent, 'The derivations disagree only about which structure to expose and which arithmetic to perform.'),
    card('synthesis-prob-card', 72, 220, 350, 314, C.deepSoft, C.deep, 20),
    text('synthesis-prob-number', 96, 242, 302, 58, '7', { fontSize: 48, fontFamily: SERIF, fontWeight: 700, color: C.deep }),
    text('synthesis-prob-title', 96, 309, 302, 58, 'Probability & graphs', { fontSize: 23, fontFamily: SERIF, fontWeight: 700 }),
    text('synthesis-prob-copy', 96, 382, 302, 120, 'Explain <b>why</b> the belief remains Gaussian: conditioning, messages, temporal recursion, and closure.', { fontSize: 17, lineHeight: 1.48 }),
    card('synthesis-opt-card', 465, 220, 350, 314, C.accentSoft, C.accent, 20),
    text('synthesis-opt-number', 489, 242, 302, 58, '6', { fontSize: 48, fontFamily: SERIF, fontWeight: 700, color: C.accent }),
    text('synthesis-opt-title', 489, 309, 302, 58, 'Optimization & estimation', { fontSize: 23, fontFamily: SERIF, fontWeight: 700 }),
    text('synthesis-opt-copy', 489, 382, 302, 120, 'Explain <b>what</b> is optimized or projected: a quadratic objective, covariance, unbiased linear estimate, or innovation subspace.', { fontSize: 17, lineHeight: 1.48 }),
    card('synthesis-struct-card', 858, 220, 350, 314, C.violetSoft, C.violet, 20),
    text('synthesis-struct-number', 882, 242, 302, 58, '3', { fontSize: 48, fontFamily: SERIF, fontWeight: 700, color: C.violet }),
    text('synthesis-struct-title', 882, 309, 302, 58, 'Structure & numerics', { fontSize: 23, fontFamily: SERIF, fontWeight: 700 }),
    text('synthesis-struct-copy', 882, 382, 302, 120, 'Explain <b>how</b> to compute: additive information, Riccati duality, KL projection, or stable triangular factors.', { fontSize: 17, lineHeight: 1.48 }),
    card('synthesis-takeaway', 72, 566, 1136, 72, C.panel, C.rule, 14),
    text('synthesis-takeaway-label', 96, 588, 160, 24, 'TAKEAWAY', { fontSize: 10, fontFamily: MONO, fontWeight: 900, color: C.accent, letterSpacing: 1.2 }),
    text('synthesis-takeaway-copy', 262, 577, 920, 48, 'Choose a derivation for insight. Choose a formulation for numerical behavior. Do not mistake either choice for a different posterior.', { fontSize: 18, fontFamily: SERIF, fontWeight: 700, valign: 'middle' }),
    ...chrome('Finish', page, C.accent)
  ];
  return {
    id: 'synthesis', background: C.paper, transition: 'morph',
    notes: 'Close by separating three questions: why the posterior has its form, what objective or projection it solves, and how to compute it reliably. The key editorial choice in this deck is that the common posterior appears once rather than being repeated sixteen times.',
    elements
  };
}

const slides = [coverSlide(), modelSlide(), atlasSlide()];
routes.forEach((route, index) => slides.push(routeSlide(route, index + 4)));
slides.push(sharedPosteriorSlide(slides.length + 1));
slides.push(liveSlide({
  id: 'shared-posterior-live',
  page: slides.length + 1,
  eyebrow: 'Shared posterior · live experiment',
  title: 'Make the common posterior move.',
  prompt: 'Change means and uncertainties. One posterior moves; an agreement readout verifies that the derivations coincide.',
  accent: C.accent,
  demo: 'scalar fusion',
  fallback: scalarFallback
}));
slides.push(geometryIntroSlide(slides.length + 1));
slides.push(liveSlide({
  id: 'covariance-geometry-live',
  page: slides.length + 1,
  eyebrow: 'Covariance geometry · live experiment',
  title: 'Rotate the measurement. Watch uncertainty contract.',
  prompt: 'Vary H, R, and correlation. The state–innovation cross-covariance decides how the scalar evidence spreads.',
  accent: C.deep,
  demo: 'covariance geometry',
  fallback: geometryFallback
}));
slides.push(finitePrecisionIntroSlide(slides.length + 1));
slides.push(liveSlide({
  id: 'finite-precision-live',
  page: slides.length + 1,
  eyebrow: 'Finite precision · live experiment',
  title: 'Stress the arithmetic, not the estimator.',
  prompt: 'Increase conditioning and lower precision. Exact identities begin to separate for numerical—not statistical—reasons.',
  accent: C.violet,
  demo: 'finite-precision equivalence',
  fallback: equivalenceFallback
}));
slides.push(synthesisSlide(slides.length + 1));

export const deck = {
  format: 'bento/slides',
  version: 1,
  docId: 'kalman-filter-derivation-atlas',
  title: 'One Filter, Sixteen Derivations',
  readonly: true,
  meta: {
    author: 'Bai Liping',
    subject: 'Kalman filter measurement-update derivations',
    company: 'bailiping.com'
  },
  size: { width: WIDTH, height: HEIGHT },
  theme: {
    background: C.paper,
    color: C.ink,
    accent: C.accent,
    fontFamily: SANS
  },
  slides
};

function indexOf(id) {
  const index = slides.findIndex(slide => slide.id === id);
  if (index < 0) throw new Error(`Unknown slide ${id}`);
  return index;
}

const liveBounds = { x: 74, y: 153, width: 1132, height: 494 };

export const inlineLiveMap = [
  {
    introSlide: 'shared-posterior',
    slide: 'shared-posterior-live',
    slideIndex: indexOf('shared-posterior-live'),
    inline: true,
    layout: 'region',
    bounds: liveBounds,
    src: './live/?demo=scalar&embed=region',
    source: './live/?demo=scalar',
    title: 'Interactive scalar Kalman fusion',
    sandbox: 'allow-scripts',
    hideSource: true,
    readyMessage: true,
    unloadWhenHidden: true
  },
  {
    introSlide: 'covariance-geometry',
    slide: 'covariance-geometry-live',
    slideIndex: indexOf('covariance-geometry-live'),
    inline: true,
    layout: 'region',
    bounds: liveBounds,
    src: './live/?demo=geometry&embed=region',
    source: './live/?demo=geometry',
    title: 'Interactive covariance geometry',
    sandbox: 'allow-scripts',
    hideSource: true,
    readyMessage: true,
    unloadWhenHidden: true
  },
  {
    introSlide: 'finite-precision',
    slide: 'finite-precision-live',
    slideIndex: indexOf('finite-precision-live'),
    inline: true,
    layout: 'region',
    bounds: liveBounds,
    src: './live/?demo=equivalence&embed=region',
    source: './live/?demo=equivalence',
    title: 'Interactive finite-precision comparison',
    sandbox: 'allow-scripts',
    hideSource: true,
    readyMessage: true,
    unloadWhenHidden: true
  }
];
