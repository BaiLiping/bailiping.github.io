const WIDTH = 1280;
const HEIGHT = 720;
const SERIF = "Georgia, 'Times New Roman', serif";
const SANS = "Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
const MONO = "'SFMono-Regular', Consolas, 'Liberation Mono', monospace";

const C = {
  paper: '#F7F5EF',
  panel: '#FFFEFB',
  ink: '#203129',
  muted: '#66756E',
  faint: '#8A948F',
  rule: '#D8DED7',
  green: '#2F6B4F',
  greenSoft: '#E7F0EA',
  rust: '#A94F2A',
  rustSoft: '#F5E8DF',
  blue: '#496E87',
  blueSoft: '#E7EEF3',
  violet: '#71638B',
  violetSoft: '#ECE9F2',
  gold: '#92702E',
  goldSoft: '#F4EEDC'
};

function text(id, x, y, w, h, html, options = {}) {
  return {
    id,
    type: 'text',
    x,
    y,
    w,
    h,
    rotation: 0,
    opacity: options.opacity ?? 1,
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
    ...(options.fx ? { fx: options.fx } : {}),
    ...(options.morphId ? { morphId: options.morphId } : {})
  };
}

function shape(id, shapeName, x, y, w, h, options = {}) {
  return {
    id,
    type: 'shape',
    shape: shapeName,
    x,
    y,
    w,
    h,
    fill: options.fill ?? 'none',
    stroke: options.stroke ?? 'none',
    strokeWidth: options.strokeWidth ?? 0,
    radius: options.radius ?? 0,
    rotation: options.rotation ?? 0,
    opacity: options.opacity ?? 1,
    ...(options.lineStart ? { lineStart: options.lineStart } : {}),
    ...(options.lineEnd ? { lineEnd: options.lineEnd } : {}),
    ...(options.fx ? { fx: options.fx } : {}),
    ...(options.morphId ? { morphId: options.morphId } : {})
  };
}

function card(id, x, y, w, h, fill = C.panel, stroke = C.rule, radius = 14) {
  return shape(id, 'rect', x, y, w, h, { fill, stroke, strokeWidth: 1, radius });
}

function chrome(section, accent = C.green) {
  return [
    shape('chrome-rule', 'rect', 72, 674, 1136, 1, { fill: C.rule }),
    text('chrome-site', 72, 687, 330, 17, 'BAI LIPING · ESTIMATION NOTES', {
      fontSize: 10, fontFamily: MONO, fontWeight: 750, color: C.faint, letterSpacing: 0.7
    }),
    text('chrome-section', 430, 685, 420, 18, section.toUpperCase(), {
      fontSize: 10, fontFamily: MONO, fontWeight: 800, color: accent, align: 'center', letterSpacing: 1.1
    })
  ];
}

function heading(eyebrow, title, subtitle, accent = C.green, options = {}) {
  return [
    text('slide-eyebrow', 72, 38, 900, 22, eyebrow.toUpperCase(), {
      fontSize: 11, fontFamily: MONO, fontWeight: 850, color: accent, letterSpacing: 1.55,
      fx: options.fx === false ? undefined : { enter: 'fade-up', order: 0 }
    }),
    text('slide-title', 72, 68, 1110, options.titleHeight ?? 55, title, {
      fontSize: options.titleSize ?? 38, fontFamily: SERIF, fontWeight: 700, color: C.ink, lineHeight: 1.05,
      fx: options.fx === false ? undefined : { enter: 'fade-up', order: 1 }
    }),
    ...(subtitle ? [text('slide-subtitle', 72, options.subtitleY ?? 119, 1110, options.subtitleHeight ?? 43, subtitle, {
      fontSize: options.subtitleSize ?? 16, color: C.muted, lineHeight: 1.35,
      fx: options.fx === false ? undefined : { enter: 'fade-up', order: 2 }
    })] : [])
  ];
}

function panel(id, x, y, w, h, title, body, options = {}) {
  const accent = options.accent ?? C.green;
  const fill = options.fill ?? C.panel;
  return [
    card(`${id}-card`, x, y, w, h, fill, options.stroke ?? C.rule, options.radius ?? 14),
    text(`${id}-title`, x + 18, y + 15, w - 36, options.titleHeight ?? 22, title, {
      fontSize: options.titleSize ?? 11, fontFamily: options.titleFamily ?? MONO, fontWeight: 850,
      color: accent, letterSpacing: options.letterSpacing ?? 0.65
    }),
    text(`${id}-body`, x + 18, y + (options.bodyY ?? 48), w - 36, h - (options.bodyY ?? 48) - 14, body, {
      fontSize: options.fontSize ?? 15, fontFamily: options.fontFamily ?? SANS, fontWeight: options.fontWeight ?? 400,
      color: options.color ?? C.ink, lineHeight: options.lineHeight ?? 1.42,
      align: options.align ?? 'left', valign: options.valign ?? 'top'
    })
  ];
}

function overviewSlide() {
  const elements = [
    shape('cover-accent', 'rect', 0, 0, 16, HEIGHT, { fill: C.rust }),
    text('cover-eyebrow', 82, 65, 760, 26, 'KALMAN FILTER · CONSOLIDATED DERIVATIONS', {
      fontSize: 12, fontFamily: MONO, fontWeight: 850, color: C.rust, letterSpacing: 1.7,
      fx: { enter: 'fade-up', order: 0 }
    }),
    text('cover-title', 80, 105, 760, 130, 'One filter.<br><span style="color:#2F6B4F">Four families.</span>', {
      fontSize: 62, fontFamily: SERIF, fontWeight: 700, lineHeight: 0.98,
      fx: { enter: 'fade-up', order: 1 }
    }),
    text('cover-copy', 82, 247, 710, 75, 'Group by the object being solved for—not by every name for the same algebra. Each family keeps its distinctive insight and arrives at the same linear–Gaussian update.', {
      fontSize: 18, color: C.muted, lineHeight: 1.45,
      fx: { enter: 'fade-up', order: 2 }
    }),
    shape('cover-path', 'line', 82, 345, 698, 2, {
      fill: C.rust, lineEnd: 'arrow', fx: { loop: { type: 'dash-march' } }
    })
  ];

  const families = [
    ['01', 'Gaussian Bayes', 'distribution', C.green, C.greenSoft, 'bayes'],
    ['02', 'Minimum-MSE estimation', 'gain', C.blue, C.blueSoft, 'mse'],
    ['03', 'Weighted least squares', 'state', C.rust, C.rustSoft, 'least-squares'],
    ['04', 'KL variational updating', 'density', C.violet, C.violetSoft, 'kl']
  ];
  families.forEach(([number, name, object, accent, soft, link], index) => {
    const x = 82 + (index % 2) * 366;
    const y = 382 + Math.floor(index / 2) * 105;
    elements.push(card(`cover-family-${number}`, x, y, 344, 84, soft, accent, 14));
    elements.push(text(`cover-family-number-${number}`, x + 17, y + 17, 42, 22, number, {
      fontSize: 11, fontFamily: MONO, fontWeight: 900, color: accent, link
    }));
    elements.push(text(`cover-family-name-${number}`, x + 65, y + 13, 260, 28, name, {
      fontSize: 17, fontFamily: SERIF, fontWeight: 700, color: C.ink, link
    }));
    elements.push(text(`cover-family-object-${number}`, x + 65, y + 46, 260, 19, `solve for the ${object}`, {
      fontSize: 11, fontFamily: MONO, fontWeight: 700, color: accent, link
    }));
  });

  elements.push(
    card('cover-result-card', 842, 72, 366, 516, C.panel, C.rule, 22),
    text('cover-result-label', 876, 106, 298, 24, 'THE SHARED RESULT', {
      fontSize: 11, fontFamily: MONO, fontWeight: 850, color: C.rust, align: 'center', letterSpacing: 1.25
    }),
    shape('cover-result-ring-a', 'ellipse', 912, 166, 226, 226, { fill: C.greenSoft, stroke: C.green, strokeWidth: 2 }),
    shape('cover-result-ring-b', 'ellipse', 952, 206, 146, 146, { fill: C.panel, stroke: C.rust, strokeWidth: 2 }),
    text('cover-result-symbol', 952, 243, 146, 64, 'm⁺, P⁺', {
      fontSize: 31, fontFamily: SERIF, fontWeight: 700, align: 'center', valign: 'middle'
    }),
    text('cover-result-formula', 874, 424, 302, 88, 'm⁺ = m⁻ + K(z − Hm⁻)<br>P⁺ = P⁻ − KSKᵀ', {
      fontSize: 20, fontFamily: SERIF, fontWeight: 700, align: 'center', lineHeight: 1.55
    }),
    text('cover-result-note', 874, 529, 302, 30, 'three live experiments · deterministic defaults', {
      fontSize: 10, fontFamily: MONO, fontWeight: 800, color: C.muted, align: 'center'
    }),
    ...chrome('Overview', C.rust)
  );

  return {
    id: 'overview',
    background: C.paper,
    transition: 'none',
    notes: 'Open with the editorial claim: these are four derivation families, not four independent filters. The four cards link to the family introductions. Preview that three deterministic experiments will make the shared update, covariance geometry, and numerical conditioning operable.',
    elements
  };
}

function modelSlide() {
  const elements = [
    ...heading('Shared setup · experiment introduction', 'The model and the common recursion', 'Fix the assumptions first. Then vary confidence and evidence without changing the estimator.', C.green),
    ...panel('model-dynamics', 72, 180, 354, 176, 'LINEAR–GAUSSIAN MODEL', 'x<sub>k</sub> = F<sub>k</sub>x<sub>k−1</sub> + B<sub>k</sub>u<sub>k</sub> + w<sub>k</sub><br>z<sub>k</sub> = H<sub>k</sub>x<sub>k</sub> + v<sub>k</sub><br><br><span style="color:#66756E">w ∼ N(0,Q), v ∼ N(0,R), independent.</span>', {
      accent: C.green, fill: C.greenSoft, stroke: C.green, fontFamily: SERIF, fontSize: 18, lineHeight: 1.45
    }),
    ...panel('model-predict', 463, 180, 354, 176, 'PREDICT', 'm<sup>−</sup><sub>k</sub> = F<sub>k</sub>m<sup>+</sup><sub>k−1</sub> + B<sub>k</sub>u<sub>k</sub><br>P<sup>−</sup><sub>k</sub> = F<sub>k</sub>P<sup>+</sup><sub>k−1</sub>F<sub>k</sub><sup>T</sup> + Q<sub>k</sub>', {
      accent: C.blue, fill: C.blueSoft, stroke: C.blue, fontFamily: SERIF, fontSize: 20, lineHeight: 1.65
    }),
    ...panel('model-correct', 854, 180, 354, 176, 'CORRECT', 'ν = z − Hm<sup>−</sup><br>S = HP<sup>−</sup>H<sup>T</sup> + R<br>K = P<sup>−</sup>H<sup>T</sup>S<sup>−1</sup><br>m<sup>+</sup> = m<sup>−</sup> + Kν', {
      accent: C.rust, fill: C.rustSoft, stroke: C.rust, fontFamily: SERIF, fontSize: 18, lineHeight: 1.35
    }),
    card('model-observe', 72, 390, 1136, 214, C.panel, C.rule, 16),
    text('model-observe-label', 96, 414, 230, 22, 'WHAT TO WATCH NEXT', {
      fontSize: 11, fontFamily: MONO, fontWeight: 900, color: C.green, letterSpacing: 1
    }),
    text('model-observe-question', 96, 454, 520, 86, 'How far should the posterior move toward the measurement?', {
      fontSize: 29, fontFamily: SERIF, fontWeight: 700, lineHeight: 1.16
    }),
    text('model-observe-answer', 96, 554, 520, 28, 'The gain K is a confidence-weighted answer.', {
      fontSize: 15, color: C.muted
    }),
    shape('model-observe-divider', 'rect', 650, 418, 1, 160, { fill: C.rule }),
    text('model-observe-steps', 688, 421, 482, 170, '<b>1 · Move the prior and measurement.</b><br><span style="color:#66756E">The innovation changes direction and size.</span><br><br><b>2 · Swap their uncertainty.</b><br><span style="color:#66756E">The posterior follows the more precise source.</span><br><br><b>3 · Check four routes.</b><br><span style="color:#66756E">Bayes, WLS, information, and conditioning agree.</span>', {
      fontSize: 15, lineHeight: 1.38
    }),
    ...chrome('Shared model', C.green)
  ];
  return {
    id: 'model',
    background: C.paper,
    transition: 'morph',
    notes: 'State the model and the predict–correct schedule once. The next slide is the scalar fusion experiment. Ask the audience to predict which source the posterior will follow when the uncertainty values are swapped.',
    elements
  };
}

const LIVE_BOUNDS = { x: 74, y: 154, width: 1132, height: 494 };

function scalarFallback() {
  const { x, y, width, height } = LIVE_BOUNDS;
  return [
    card('scalar-fallback-region', x, y, width, height, '#FBFAF6', C.rule, 14),
    card('scalar-fallback-controls', x + 16, y + 16, 278, height - 32, C.greenSoft, C.green, 12),
    text('scalar-fallback-controls-label', x + 34, y + 33, 242, 22, 'DETERMINISTIC DEFAULT', {
      fontSize: 10, fontFamily: MONO, fontWeight: 900, color: C.green, letterSpacing: 1
    }),
    text('scalar-fallback-controls-copy', x + 34, y + 75, 242, 174, 'prior mean&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; m⁻ = −1.2<br>prior deviation&nbsp; σₚ = 1.35<br><br>measurement&nbsp;&nbsp;&nbsp;&nbsp; z = 2.1<br>measurement dev.&nbsp; σᵣ = 0.75', {
      fontSize: 16, fontFamily: MONO, fontWeight: 700, lineHeight: 1.55
    }),
    text('scalar-fallback-controls-hint', x + 34, y + 352, 242, 76, 'Live controls replace this region only while the Bento slide is active.', {
      fontSize: 13, color: C.muted, lineHeight: 1.45
    }),
    card('scalar-fallback-plot', x + 312, y + 16, width - 328, height - 32, C.panel, C.rule, 12),
    text('scalar-fallback-plot-title', x + 338, y + 34, 500, 22, 'PRIOR × LIKELIHOOD → POSTERIOR', {
      fontSize: 10, fontFamily: MONO, fontWeight: 900, color: C.rust, letterSpacing: 1.05
    }),
    shape('scalar-axis', 'rect', x + 358, y + 345, 470, 2, { fill: C.rule }),
    shape('scalar-prior', 'ellipse', x + 382, y + 145, 244, 200, { fill: 'rgba(73,110,135,.10)', stroke: C.blue, strokeWidth: 3 }),
    shape('scalar-likelihood', 'ellipse', x + 650, y + 100, 130, 245, { fill: 'rgba(169,79,42,.09)', stroke: C.rust, strokeWidth: 3 }),
    shape('scalar-posterior', 'ellipse', x + 570, y + 73, 106, 272, { fill: 'rgba(47,107,79,.10)', stroke: C.green, strokeWidth: 4 }),
    text('scalar-prior-label', x + 394, y + 362, 180, 22, 'prior', { fontSize: 11, fontFamily: MONO, fontWeight: 850, color: C.blue, align: 'center' }),
    text('scalar-post-label', x + 558, y + 362, 140, 22, 'posterior', { fontSize: 11, fontFamily: MONO, fontWeight: 850, color: C.green, align: 'center' }),
    text('scalar-like-label', x + 676, y + 362, 130, 22, 'likelihood', { fontSize: 11, fontFamily: MONO, fontWeight: 850, color: C.rust, align: 'center' }),
    card('scalar-metric-gain', x + 848, y + 84, 236, 106, C.greenSoft, C.green, 11),
    text('scalar-metric-gain-label', x + 866, y + 100, 200, 20, 'KALMAN GAIN', { fontSize: 9, fontFamily: MONO, fontWeight: 900, color: C.green, align: 'center' }),
    text('scalar-metric-gain-value', x + 866, y + 130, 200, 42, 'K = 0.764', { fontSize: 25, fontFamily: SERIF, fontWeight: 700, align: 'center' }),
    card('scalar-metric-post', x + 848, y + 212, 236, 142, C.rustSoft, C.rust, 11),
    text('scalar-metric-post-label', x + 866, y + 228, 200, 20, 'POSTERIOR', { fontSize: 9, fontFamily: MONO, fontWeight: 900, color: C.rust, align: 'center' }),
    text('scalar-metric-post-value', x + 866, y + 259, 200, 70, 'm⁺ ≈ 1.32<br>σ⁺ ≈ 0.66', { fontSize: 21, fontFamily: SERIF, fontWeight: 700, align: 'center', lineHeight: 1.4 }),
    text('scalar-fallback-status', x + 338, y + 421, 746, 24, 'STATIC FALLBACK · four derivations agree at the deterministic default', {
      fontSize: 10, fontFamily: MONO, fontWeight: 850, color: C.muted, align: 'center'
    })
  ];
}

function scalarLiveSlide() {
  const elements = [
    text('live-eyebrow', 74, 37, 870, 21, 'SHARED MODEL · LIVE EXPERIMENT', {
      fontSize: 10, fontFamily: MONO, fontWeight: 900, color: C.green, letterSpacing: 1.45
    }),
    text('live-title', 74, 66, 1050, 48, 'Make the common posterior move.', {
      fontSize: 36, fontFamily: SERIF, fontWeight: 700, lineHeight: 1.05
    }),
    text('live-prompt', 74, 116, 1080, 28, 'Change the two Gaussian sources. Watch confidence—not the derivation label—determine the result.', {
      fontSize: 15, color: C.muted
    }),
    ...scalarFallback(),
    shape('live-demo-mount', 'rect', LIVE_BOUNDS.x, LIVE_BOUNDS.y, LIVE_BOUNDS.width, LIVE_BOUNDS.height, {
      fill: 'rgba(255,255,255,0)', stroke: 'rgba(255,255,255,0)', strokeWidth: 0, opacity: 0
    }),
    ...chrome('Shared model · live', C.green)
  ];
  return {
    id: 'model-live',
    background: C.paper,
    transition: 'morph',
    notes: 'The experiment mounts automatically. Move either mean, swap the uncertainty values, and point out that Bayes, weighted least squares, information form, and conditioning produce the same posterior. Press Escape to return focus to Bento; Page Up returns to the model slide.',
    elements
  };
}

function ideaSlide({ id, family, eyebrow, title, subtitle, formula, leftTitle, leftBody, rightTitle, rightBody, note, accent, soft, notes, liveCue = '' }) {
  const elements = [
    ...heading(eyebrow, title, subtitle, accent),
    card(`${id}-formula-card`, 72, 184, 1136, 112, soft, accent, 16),
    text(`${id}-formula-label`, 96, 203, 230, 20, 'GOVERNING IDEA', {
      fontSize: 10, fontFamily: MONO, fontWeight: 900, color: accent, letterSpacing: 1
    }),
    text(`${id}-formula`, 96, 231, 1088, 50, formula, {
      fontSize: 23, fontFamily: SERIF, fontWeight: 700, align: 'center', valign: 'middle', lineHeight: 1.25
    }),
    ...panel(`${id}-left`, 72, 322, 552, 204, leftTitle, leftBody, {
      accent, fill: C.panel, stroke: C.rule, fontSize: 16, lineHeight: 1.46
    }),
    ...panel(`${id}-right`, 656, 322, 552, 204, rightTitle, rightBody, {
      accent, fill: C.panel, stroke: C.rule, fontSize: 16, lineHeight: 1.46
    }),
    card(`${id}-note-card`, 72, 548, 1136, 88, soft, accent, 13),
    text(`${id}-note-label`, 94, 568, 162, 18, liveCue ? 'WATCH NEXT' : 'BOUNDARY', {
      fontSize: 10, fontFamily: MONO, fontWeight: 900, color: accent, letterSpacing: 1
    }),
    text(`${id}-note`, 252, 561, 932, 54, note, {
      fontSize: 14.5, fontWeight: 650, lineHeight: 1.42, valign: 'middle'
    }),
    ...chrome(family, accent)
  ];
  return { id, background: C.paper, transition: 'morph', notes, elements };
}

function bayesSlide() {
  return ideaSlide({
    id: 'bayes',
    family: 'Family 01 · Gaussian Bayes',
    eyebrow: 'Family 01 · solve for a distribution',
    title: 'Gaussian Bayes',
    subtitle: 'Propagate the old posterior through the dynamics, then multiply the prediction by the new likelihood.',
    formula: 'p<sup>−</sup><sub>k</sub>(x) = ∫ p(x | x′)p<sup>+</sup><sub>k−1</sub>(x′)dx′ &nbsp;&nbsp;·&nbsp;&nbsp; p<sup>+</sup><sub>k</sub>(x) ∝ p(z<sub>k</sub> | x)p<sup>−</sup><sub>k</sub>(x)',
    leftTitle: 'ONE OPERATION · THREE ALGEBRAIC VIEWS',
    leftBody: '<b>Completing the square</b> reads off mean and covariance.<br><br><b>Information form</b> adds quadratic coefficients.<br><br><b>Joint conditioning</b> reads the conditional Gaussian blocks.',
    rightTitle: 'WHAT HAS BEEN CONSOLIDATED',
    rightBody: '<b>HMM forward recursion</b> and recursive Bayesian filtering use the same predict–update schedule.<br><br><b>GP conditioning</b> uses the same Gaussian identity; an arbitrary GP need not have a finite Markov realization.',
    note: 'Distinctive output: the entire posterior—not only a point estimate. Under the stated model this is the exact Bayesian filter.',
    accent: C.green,
    soft: C.greenSoft,
    notes: 'Introduce Bayes as distribution computation. Completing the square, information addition, and Gaussian conditioning are algebraic views of one posterior. Emphasize the model boundary: exactness comes from linear–Gaussian closure.'
  });
}

function mseSlide() {
  return ideaSlide({
    id: 'mse',
    family: 'Family 02 · Minimum-MSE estimation',
    eyebrow: 'Family 02 · experiment introduction',
    title: 'Minimum-MSE estimation',
    subtitle: 'Choose the gain—not a density. Make the affine correction error as small as possible.',
    formula: 'x̂(K) = m<sup>−</sup> + Kν &nbsp;&nbsp;·&nbsp;&nbsp; K<sub>⋆</sub> = arg min<sub>K</sub> E‖x − x̂(K)‖²',
    leftTitle: 'TWO PROOFS · ONE OPTIMUM',
    leftBody: '<b>Orthogonality:</b> the remaining error is uncorrelated with the innovation.<br><br><b>Covariance minimization:</b> differentiate the trace of the error covariance.<br><br>Both give K<sub>⋆</sub>S = P<sup>−</sup>H<sup>T</sup>.',
    rightTitle: 'GEOMETRIC READING',
    rightBody: 'The measurement selects a direction through H. Cross-covariance P<sup>−</sup>H<sup>T</sup> carries that scalar evidence into the state.<br><br>Noise R weakens the contraction; correlation rotates how the correction spreads.',
    note: 'Rotate H, vary R, and change correlation ρ. Watch the posterior ellipse contract mainly along the measured slice.',
    accent: C.blue,
    soft: C.blueSoft,
    liveCue: 'geometry',
    notes: 'Explain that finite second moments and the required zero cross-covariances suffice for affine/LMMSE optimality. The next slide turns the projection into geometry: rotate H, adjust R and correlation, and observe covariance contraction.'
  });
}

function leastSquaresSlide() {
  return ideaSlide({
    id: 'least-squares',
    family: 'Family 03 · Weighted least squares',
    eyebrow: 'Family 03 · solve for a state vector',
    title: 'Weighted least squares',
    subtitle: 'Penalize disagreement with the prediction and observation, each weighted by inverse uncertainty.',
    formula: 'φ(x) = ½‖x − m<sup>−</sup>‖²<sub>(P<sup>−</sup>)<sup>−1</sup></sub> + ½‖z − Hx‖²<sub>R<sup>−1</sup></sub>',
    leftTitle: 'STATISTICAL INTERPRETATIONS',
    leftBody: '<b>MAP:</b> this is the negative Gaussian log posterior; its mode equals its mean.<br><br><b>BLUE:</b> the same generalized least-squares algebra applies to independent unbiased observations of a fixed state—but the experiment differs.',
    rightTitle: 'ALGORITHMS · NOT NEW PRINCIPLES',
    rightBody: '<b>RLS</b> updates the normal equations as observations arrive; the static case has F = I and Q = 0.<br><br><b>Square-root / QR</b> solves the whitened system without forming its normal matrix.',
    note: 'BLUE requires a random unbiased observation of a fixed state. A fixed Bayesian prior mean is not unbiased for every possible state.',
    accent: C.rust,
    soft: C.rustSoft,
    notes: 'Frame least squares as state-vector optimization. Separate the MAP experiment from the BLUE experiment even though their linear algebra matches. Reserve RLS and QR for implementation, not additional derivation families.'
  });
}

function klSlide() {
  return ideaSlide({
    id: 'kl',
    family: 'Family 04 · KL variational updating',
    eyebrow: 'Family 04 · solve for a density',
    title: 'KL variational updating',
    subtitle: 'Penalize departure from the prior while rewarding densities that explain the new observation.',
    formula: 'q<sub>⋆</sub> = arg min<sub>q≥0, ∫q=1</sub> { D<sub>KL</sub>(q ‖ p<sup>−</sup>) + E<sub>q</sub>[−log p(z | x)] }',
    leftTitle: 'DIFFERENT FROM LEAST SQUARES',
    leftBody: 'Least squares varies a state or mean. This problem varies the <b>full density</b>, including covariance.<br><br>In the Gaussian reduction, the entropy term determines covariance instead of collapsing the answer to a point.',
    rightTitle: 'WHAT HAS BEEN CONSOLIDATED',
    rightBody: '“Minimum surprise” and relative-entropy language belong here only when the objective or constraints are explicit.<br><br>The displayed likelihood-loss objective recovers Bayes exactly; related maximum-entropy formulations use constraints.',
    note: 'Minimizing KL to the prior alone returns the prior. The likelihood/loss—or a specified observation constraint—is indispensable.',
    accent: C.violet,
    soft: C.violetSoft,
    notes: 'State the optimization variable clearly: a whole density. Derive the exact minimizer by rewriting the objective as KL to the posterior plus a constant. Correct the common misconception that KL-to-prior alone performs an update.'
  });
}

function equationSheetSlide({ id, family, title, context, panels, accent, soft, notes }) {
  const elements = [
    ...heading(`${family} · equation sheet`, title, context, accent, { titleSize: 35, subtitleSize: 13.5, subtitleHeight: 39 }),
  ];
  panels.forEach((item, index) => {
    const col = index % 2;
    const row = Math.floor(index / 2);
    const x = 72 + col * 576;
    const y = 180 + row * 225;
    elements.push(...panel(`${id}-panel-${index + 1}`, x, y, 552, 205, `${index + 1} · ${item.title}`, item.body, {
      accent,
      fill: index === 3 ? soft : C.panel,
      stroke: index === 3 ? accent : C.rule,
      fontFamily: item.fontFamily ?? SERIF,
      fontSize: item.fontSize ?? 16.5,
      lineHeight: item.lineHeight ?? 1.46,
      bodyY: 46,
      titleSize: 11
    }));
  });
  elements.push(...chrome(`${family} · equations`, accent));
  return { id, background: C.paper, transition: 'morph', notes, elements };
}

function bayesEquationsSlide() {
  return equationSheetSlide({
    id: 'bayes-equations', family: 'Family 01', title: 'Gaussian Bayes · equations',
    context: 'One correction: x ∼ N(m⁻,P⁻), z = Hx + v, v ∼ N(0,R), independent; P⁻ ≻ 0 and R ≻ 0.',
    accent: C.green, soft: C.greenSoft,
    panels: [
      { title: 'Multiply Gaussian factors', body: 'p<sup>+</sup>(x) ∝ p<sup>−</sup>(x)p(z | x)<br><br>−log p<sup>+</sup>(x) = ½‖x−m<sup>−</sup>‖²<sub>(P<sup>−</sup>)<sup>−1</sup></sub><br>&nbsp;&nbsp;&nbsp;&nbsp;+ ½‖z−Hx‖²<sub>R<sup>−1</sup></sub> + c' },
      { title: 'Collect information', body: 'J<sup>+</sup> = (P<sup>−</sup>)<sup>−1</sup> + H<sup>T</sup>R<sup>−1</sup>H<br><br>h<sup>+</sup> = (P<sup>−</sup>)<sup>−1</sup>m<sup>−</sup> + H<sup>T</sup>R<sup>−1</sup>z<br><br><span style="color:#66756E">Precision and information vector add.</span>' },
      { title: 'Complete the square', body: 'm<sup>+</sup> = (J<sup>+</sup>)<sup>−1</sup>h<sup>+</sup><br>P<sup>+</sup> = (J<sup>+</sup>)<sup>−1</sup><br><br>p<sup>+</sup>(x) ∝ exp{−½(x−m<sup>+</sup>)<sup>T</sup>J<sup>+</sup>(x−m<sup>+</sup>)}', fontSize: 16 },
      { title: 'Same answer by conditioning', body: 'S = HP<sup>−</sup>H<sup>T</sup> + R<br>m<sup>+</sup> = m<sup>−</sup> + P<sup>−</sup>H<sup>T</sup>S<sup>−1</sup>(z−Hm<sup>−</sup>)<br>P<sup>+</sup> = P<sup>−</sup> − P<sup>−</sup>H<sup>T</sup>S<sup>−1</sup>HP<sup>−</sup>', fontSize: 15.5 }
    ],
    notes: 'Walk clockwise: factor multiplication, information addition, completing the square, and joint conditioning. The four boxes are one derivation written in complementary coordinates.'
  });
}

function mseEquationsSlide() {
  return equationSheetSlide({
    id: 'mse-equations', family: 'Family 02', title: 'Minimum-MSE estimation · equations',
    context: 'Let e⁻ = x−m⁻, ν = z−Hm⁻, S = HP⁻Hᵀ+R; e⁻ and v are zero-mean and uncorrelated, and S ≻ 0.',
    accent: C.blue, soft: C.blueSoft,
    panels: [
      { title: 'Error for an arbitrary gain', body: 'e(K) = e<sup>−</sup> − Kν<br><br>E[e<sup>−</sup>ν<sup>T</sup>] = P<sup>−</sup>H<sup>T</sup><br><br><span style="color:#66756E">Choose K inside an affine correction.</span>' },
      { title: 'Orthogonality gives the gain', body: 'E[e(K<sub>⋆</sub>)ν<sup>T</sup>] = 0<br><br>K<sub>⋆</sub>S = P<sup>−</sup>H<sup>T</sup><br>K<sub>⋆</sub> = P<sup>−</sup>H<sup>T</sup>S<sup>−1</sup>' },
      { title: 'Equivalent covariance calculation', body: 'P(K) = (I−KH)P<sup>−</sup>(I−KH)<sup>T</sup> + KRK<sup>T</sup><br><br>∇<sub>K</sub> tr P(K) = 2(KS − P<sup>−</sup>H<sup>T</sup>)', fontSize: 16 },
      { title: 'Certify the minimum; update', body: 'P(K) − P(K<sub>⋆</sub>) = (K−K<sub>⋆</sub>)S(K−K<sub>⋆</sub>)<sup>T</sup> ⪰ 0<br><br>m<sup>+</sup> = m<sup>−</sup> + K<sub>⋆</sub>ν<br>P<sup>+</sup> = P<sup>−</sup> − K<sub>⋆</sub>SK<sub>⋆</sub><sup>T</sup>', fontSize: 15.5 }
    ],
    notes: 'Connect the previous geometric experiment back to the algebra. Orthogonality and trace minimization give the same normal equation. The final positive-semidefinite difference certifies global optimality.'
  });
}

function leastSquaresEquationsSlide() {
  return equationSheetSlide({
    id: 'least-squares-equations', family: 'Family 03', title: 'Weighted least squares · equations',
    context: 'One correction: x ∼ N(m⁻,P⁻), z = Hx + v, v ∼ N(0,R), independent; P⁻ ≻ 0 and R ≻ 0.',
    accent: C.rust, soft: C.rustSoft,
    panels: [
      { title: 'Objective', body: 'φ(x) = ½‖x−m<sup>−</sup>‖²<sub>(P<sup>−</sup>)<sup>−1</sup></sub><br>&nbsp;&nbsp;&nbsp;&nbsp;+ ½‖z−Hx‖²<sub>R<sup>−1</sup></sub><br><br><span style="color:#66756E">Gaussian MAP objective; Hessian = posterior precision.</span>' },
      { title: 'Normal equations and curvature', body: 'J<sup>+</sup> = (P<sup>−</sup>)<sup>−1</sup> + H<sup>T</sup>R<sup>−1</sup>H<br>h<sup>+</sup> = (P<sup>−</sup>)<sup>−1</sup>m<sup>−</sup> + H<sup>T</sup>R<sup>−1</sup>z<br>∇φ(x)=J<sup>+</sup>x−h<sup>+</sup>=0<br>m<sup>+</sup>=(J<sup>+</sup>)<sup>−1</sup>h<sup>+</sup>, P<sup>+</sup>=(J<sup>+</sup>)<sup>−1</sup>', fontSize: 14.5 },
      { title: 'Expose the Kalman correction', body: 'ν = z−Hm<sup>−</sup>, &nbsp; S = HP<sup>−</sup>H<sup>T</sup>+R<br><br>(J<sup>+</sup>)<sup>−1</sup>H<sup>T</sup>R<sup>−1</sup> = P<sup>−</sup>H<sup>T</sup>S<sup>−1</sup> = K<br><br>m<sup>+</sup>=m<sup>−</sup>+Kν, &nbsp; P<sup>+</sup>=P<sup>−</sup>−KSK<sup>T</sup>', fontSize: 15 },
      { title: 'BLUE: a different experiment', body: 'd = [a; z], &nbsp; G = [I; H], &nbsp; W = diag(P<sup>−</sup>,R)<br><br>x̂ = (G<sup>T</sup>W<sup>−1</sup>G)<sup>−1</sup>G<sup>T</sup>W<sup>−1</sup>d<br>&nbsp;&nbsp;= a + K(z−Ha)<br><br><span style="color:#66756E">Here a=x+ε and x is fixed.</span>', fontSize: 14.5 }
    ],
    notes: 'Derive the normal equations, then use Woodbury to expose the Kalman correction. End by restating that BLUE uses a different sampling experiment even when the estimator formula matches.'
  });
}

function klEquationsSlide() {
  return equationSheetSlide({
    id: 'kl-equations', family: 'Family 04', title: 'KL variational updating · equations',
    context: 'One correction: x ∼ N(m⁻,P⁻), z = Hx + v, v ∼ N(0,R), independent; optimize over q ≥ 0 with ∫q = 1.',
    accent: C.violet, soft: C.violetSoft,
    panels: [
      { title: 'Optimize over densities', body: 'ℓ<sub>z</sub>(x) = −log p(z | x)<br><br>F(q) = D<sub>KL</sub>(q ‖ p<sup>−</sup>) + E<sub>q</sub>[ℓ<sub>z</sub>(x)]<br><br><span style="color:#66756E">Include the likelihood normalizing constant.</span>' },
      { title: 'Identify the exact minimizer', body: 'F(q) = D<sub>KL</sub>(q ‖ p(x | z)) − log p(z)<br><br>q<sub>⋆</sub>(x) ∝ p<sup>−</sup>(x)e<sup>−ℓ<sub>z</sub>(x)</sup><br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= p<sup>−</sup>(x)p(z | x)' },
      { title: 'Gaussian parameter objective', body: 'q=N(μ,Σ), &nbsp; J=(P<sup>−</sup>)<sup>−1</sup>+H<sup>T</sup>R<sup>−1</sup>H<br><br>F(μ,Σ)=½‖μ−m<sup>−</sup>‖²<sub>(P<sup>−</sup>)<sup>−1</sup></sub> + ½‖z−Hμ‖²<sub>R<sup>−1</sup></sub><br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ ½tr(JΣ) − ½log det Σ + c', fontSize: 14.2 },
      { title: 'Recover mean and covariance', body: 'h=(P<sup>−</sup>)<sup>−1</sup>m<sup>−</sup>+H<sup>T</sup>R<sup>−1</sup>z<br><br>∇<sub>μ</sub>F=0 ⇒ Jμ=h<br>∇<sub>Σ</sub>F=0 ⇒ Σ<sup>−1</sup>=J<br><br>m<sup>+</sup>=J<sup>−1</sup>h, &nbsp; P<sup>+</sup>=J<sup>−1</sup>', fontSize: 15.5 }
    ],
    notes: 'Show that the variational objective is exactly KL to the posterior plus a constant. In the Gaussian parameterization, optimize both μ and Σ; the entropy term is what prevents covariance collapse.'
  });
}

function geometryFallback() {
  const { x, y, width, height } = LIVE_BOUNDS;
  return [
    card('geometry-fallback-region', x, y, width, height, '#FBFAF6', C.rule, 14),
    card('geometry-fallback-controls', x + 16, y + 16, 278, height - 32, C.blueSoft, C.blue, 12),
    text('geometry-fallback-label', x + 34, y + 33, 242, 22, 'DEFAULT GEOMETRY', { fontSize: 10, fontFamily: MONO, fontWeight: 900, color: C.blue, letterSpacing: 1 }),
    text('geometry-fallback-values', x + 34, y + 75, 242, 214, 'σₓ = 1.80<br>σᵧ = 1.00<br>ρ = 0.65<br><br>measurement angle φ = 28°<br>measured value z = 1.70<br>measurement σᵣ = 0.45', { fontSize: 15.5, fontFamily: MONO, fontWeight: 700, lineHeight: 1.52 }),
    text('geometry-fallback-hint', x + 34, y + 368, 242, 62, 'Rotate H and watch correlation carry evidence across coordinates.', { fontSize: 13, color: C.muted, lineHeight: 1.45 }),
    card('geometry-fallback-stage', x + 312, y + 16, width - 328, height - 32, C.panel, C.rule, 12),
    text('geometry-fallback-stage-label', x + 338, y + 34, 520, 22, 'PRIOR ELLIPSE → MEASUREMENT STRIP → POSTERIOR', { fontSize: 10, fontFamily: MONO, fontWeight: 900, color: C.blue, letterSpacing: 1 }),
    shape('geometry-prior', 'ellipse', x + 410, y + 126, 382, 236, { fill: C.blueSoft, stroke: C.blue, strokeWidth: 4, rotation: 23 }),
    shape('geometry-posterior', 'ellipse', x + 516, y + 176, 205, 132, { fill: C.greenSoft, stroke: C.green, strokeWidth: 5, rotation: 23 }),
    shape('geometry-measurement', 'rect', x + 398, y + 276, 446, 4, { fill: C.rust, rotation: -28 }),
    text('geometry-prior-label', x + 382, y + 385, 220, 24, 'prior P⁻', { fontSize: 12, fontFamily: MONO, fontWeight: 850, color: C.blue, align: 'center' }),
    text('geometry-post-label', x + 632, y + 339, 220, 24, 'posterior P⁺', { fontSize: 12, fontFamily: MONO, fontWeight: 850, color: C.green, align: 'center' }),
    card('geometry-metric', x + 874, y + 94, 194, 164, C.greenSoft, C.green, 11),
    text('geometry-metric-label', x + 892, y + 111, 158, 19, 'OBSERVE', { fontSize: 9, fontFamily: MONO, fontWeight: 900, color: C.green, align: 'center' }),
    text('geometry-metric-copy', x + 892, y + 145, 158, 86, 'Evidence contracts the measured direction most strongly.', { fontSize: 16, fontFamily: SERIF, fontWeight: 700, align: 'center', lineHeight: 1.35 }),
    text('geometry-fallback-status', x + 338, y + 421, 730, 24, 'STATIC FALLBACK · deterministic covariance geometry', { fontSize: 10, fontFamily: MONO, fontWeight: 850, color: C.muted, align: 'center' })
  ];
}

function geometryLiveSlide() {
  return {
    id: 'mse-live', background: C.paper, transition: 'morph',
    notes: 'The geometry experiment mounts automatically. Rotate H, increase R, and vary ρ. Relate the gain vector to the state–innovation cross-covariance. Press Escape to return focus to Bento; Page Up returns to the minimum-MSE introduction.',
    elements: [
      text('live-eyebrow', 74, 37, 870, 21, 'MINIMUM-MSE · LIVE EXPERIMENT', { fontSize: 10, fontFamily: MONO, fontWeight: 900, color: C.blue, letterSpacing: 1.45 }),
      text('live-title', 74, 66, 1050, 48, 'Rotate the measurement. Watch uncertainty contract.', { fontSize: 34, fontFamily: SERIF, fontWeight: 700, lineHeight: 1.05 }),
      text('live-prompt', 74, 116, 1080, 28, 'Vary H, R, and correlation. Cross-covariance decides how scalar evidence spreads through the state.', { fontSize: 15, color: C.muted }),
      ...geometryFallback(),
      shape('live-demo-mount', 'rect', LIVE_BOUNDS.x, LIVE_BOUNDS.y, LIVE_BOUNDS.width, LIVE_BOUNDS.height, { fill: 'rgba(255,255,255,0)', stroke: 'rgba(255,255,255,0)', strokeWidth: 0, opacity: 0 }),
      ...chrome('Minimum-MSE · live', C.blue)
    ]
  };
}

function graphsSlide() {
  return equationSheetSlide({
    id: 'graphs', family: 'Computational notes', title: 'Gaussian elimination · the computational form',
    context: 'a ∼ N(mₐ,Pₐ), b = Fa+w, z = Hb+v; independent w ∼ N(0,Q), v ∼ N(0,R), with Pₐ,Q,R ≻ 0.',
    accent: C.gold, soft: C.goldSoft,
    panels: [
      { title: 'Write the joint canonical Gaussian', body: 'u = [a; b], &nbsp; Λ = [A C; C<sup>T</sup> D], &nbsp; η = [ηₐ; ηᵦ]<br><br>p(u) ∝ exp{−½u<sup>T</sup>Λu + u<sup>T</sup>η}', fontSize: 16 },
      { title: 'Blocks from the transition factor', body: 'A = Pₐ<sup>−1</sup> + F<sup>T</sup>Q<sup>−1</sup>F<br>C = −F<sup>T</sup>Q<sup>−1</sup>, &nbsp; D = Q<sup>−1</sup><br><br>ηₐ = Pₐ<sup>−1</sup>mₐ, &nbsp; ηᵦ = 0' },
      { title: 'Eliminate a: prediction message', body: 'J<sup>−</sup> = D − C<sup>T</sup>A<sup>−1</sup>C<br>h<sup>−</sup> = ηᵦ − C<sup>T</sup>A<sup>−1</sup>ηₐ<br><br>P<sup>−</sup> = (J<sup>−</sup>)<sup>−1</sup> = FPₐF<sup>T</sup> + Q<br>m<sup>−</sup> = P<sup>−</sup>h<sup>−</sup> = Fmₐ', fontSize: 15 },
      { title: 'Add the observation factor', body: 'J<sup>+</sup> = J<sup>−</sup> + H<sup>T</sup>R<sup>−1</sup>H<br>h<sup>+</sup> = h<sup>−</sup> + H<sup>T</sup>R<sup>−1</sup>z<br><br>P<sup>+</sup> = (J<sup>+</sup>)<sup>−1</sup><br>m<sup>+</sup> = P<sup>+</sup>h<sup>+</sup>', fontSize: 16 }
    ],
    notes: 'Present Gaussian message passing as block elimination. The Schur complement is the prediction message; adding the observation factor is the information-form correction. This is a computational organization, not a fifth derivation family.'
  });
}

function implementationsSlide() {
  const elements = [
    ...heading('Numerical forms · experiment introduction', 'Numerical forms and control duality', 'The estimator is fixed. Arithmetic path, conditioning, and factorization determine numerical behavior.', C.violet),
    ...panel('impl-qr', 72, 184, 552, 300, 'SQUARE ROOT / QR', 'P<sup>−</sup> = L<sub>p</sub>L<sub>p</sub><sup>T</sup>, &nbsp; R = L<sub>r</sub>L<sub>r</sub><sup>T</sup><br><br>A = [L<sub>p</sub><sup>−1</sup>; L<sub>r</sub><sup>−1</sup>H], &nbsp; b = [L<sub>p</sub><sup>−1</sup>m<sup>−</sup>; L<sub>r</sub><sup>−1</sup>z]<br><br>A = UT, &nbsp; Tm<sup>+</sup> = U<sup>T</sup>b<br>P<sup>+</sup> = T<sup>−1</sup>T<sup>−T</sup><br><br><span style="color:#66756E">Use triangular solves; avoid squaring the stacked system’s condition number.</span>', {
      accent: C.violet, fill: C.violetSoft, stroke: C.violet, fontFamily: SERIF, fontSize: 16.5, lineHeight: 1.43
    }),
    ...panel('impl-riccati', 656, 184, 552, 300, 'RICCATI / LQR DUALITY', 'Π<sub>k+1</sub> = FΠ<sub>k</sub>F<sup>T</sup> + Q<br>&nbsp;&nbsp;− FΠ<sub>k</sub>H<sup>T</sup>(HΠ<sub>k</sub>H<sup>T</sup>+R)<sup>−1</sup>HΠ<sub>k</sub>F<sup>T</sup><br><br>Π<sub>k</sub> = P<sup>−</sup><sub>k</sub><br><br><span style="color:#66756E">LQR uses A<sub>c</sub>=F<sup>T</sup>, B<sub>c</sub>=H<sup>T</sup>; finite-horizon time directions reverse. Constant gain also requires convergence.</span>', {
      accent: C.gold, fill: C.goldSoft, stroke: C.gold, fontFamily: SERIF, fontSize: 16, lineHeight: 1.45
    }),
    card('impl-watch', 72, 516, 1136, 116, C.panel, C.violet, 14),
    text('impl-watch-label', 96, 538, 180, 20, 'WHAT TO WATCH NEXT', { fontSize: 10, fontFamily: MONO, fontWeight: 900, color: C.violet, letterSpacing: 1 }),
    text('impl-watch-copy', 272, 530, 912, 74, '<b>Raise cond(P⁻), then lower simulated precision.</b><br><span style="color:#66756E">Covariance subtraction, information inversion, Joseph stabilization, and square-root QR target the same P⁺—but cease to behave identically in finite arithmetic.</span>', { fontSize: 16, lineHeight: 1.5, valign: 'middle' }),
    ...chrome('Numerical forms', C.violet)
  ];
  return {
    id: 'implementations', background: C.paper, transition: 'morph',
    notes: 'Separate estimator choice from numerical implementation. QR, information form, and Joseph stabilization target the same covariance. The next slide makes conditioning and simulated precision controllable.',
    elements
  };
}

function precisionFallback() {
  const { x, y, width, height } = LIVE_BOUNDS;
  const methods = [
    ['Covariance subtraction', 'P⁻ − KSKᵀ', C.rust, C.rustSoft],
    ['Information inversion', '(P⁻¹ + HᵀR⁻¹H)⁻¹', C.blue, C.blueSoft],
    ['Joseph stabilization', '(I−KH)P⁻(I−KH)ᵀ + KRKᵀ', C.green, C.greenSoft],
    ['Square root / QR', 'whiten → QR → solve', C.violet, C.violetSoft]
  ];
  const elements = [
    card('precision-fallback-region', x, y, width, height, '#FBFAF6', C.rule, 14),
    card('precision-fallback-controls', x + 16, y + 16, 278, height - 32, C.violetSoft, C.violet, 12),
    text('precision-fallback-label', x + 34, y + 33, 242, 22, 'DEFAULT STRESS TEST', { fontSize: 10, fontFamily: MONO, fontWeight: 900, color: C.violet, letterSpacing: 1 }),
    text('precision-fallback-values', x + 34, y + 78, 242, 154, 'state dimension&nbsp;&nbsp;&nbsp; n = 3<br>measurement dim.&nbsp;&nbsp; m = 1<br>cond(P⁻)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; = 10³<br>precision&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; = double', { fontSize: 15.5, fontFamily: MONO, fontWeight: 700, lineHeight: 1.55 }),
    text('precision-fallback-hint', x + 34, y + 365, 242, 68, 'Lower significant digits until algebraic identities separate numerically.', { fontSize: 13, color: C.muted, lineHeight: 1.45 }),
    card('precision-fallback-stage', x + 312, y + 16, width - 328, height - 32, C.panel, C.rule, 12),
    text('precision-fallback-stage-label', x + 338, y + 34, 700, 22, 'ONE TARGET COVARIANCE · FOUR ARITHMETIC PATHS', { fontSize: 10, fontFamily: MONO, fontWeight: 900, color: C.violet, letterSpacing: 1 })
  ];
  methods.forEach(([name, formula, accent, soft], index) => {
    const col = index % 2;
    const row = Math.floor(index / 2);
    const mx = x + 338 + col * 374;
    const my = y + 82 + row * 158;
    elements.push(card(`precision-method-${index}`, mx, my, 348, 136, soft, accent, 11));
    elements.push(text(`precision-method-name-${index}`, mx + 17, my + 15, 314, 24, name, { fontSize: 16, fontFamily: SERIF, fontWeight: 700, color: accent }));
    elements.push(text(`precision-method-formula-${index}`, mx + 17, my + 51, 314, 40, formula, { fontSize: 12.5, fontFamily: MONO, fontWeight: 750, valign: 'middle' }));
    elements.push(text(`precision-method-status-${index}`, mx + 17, my + 104, 314, 18, 'max Δ ≈ 0 · positive definite', { fontSize: 9.5, fontFamily: MONO, fontWeight: 850, color: C.green }));
  });
  elements.push(text('precision-fallback-status', x + 338, y + 421, 730, 24, 'STATIC FALLBACK · all four paths agree at the deterministic default', { fontSize: 10, fontFamily: MONO, fontWeight: 850, color: C.muted, align: 'center' }));
  return elements;
}

function precisionLiveSlide() {
  return {
    id: 'implementations-live', background: C.paper, transition: 'morph',
    notes: 'The finite-precision experiment mounts automatically. Raise the condition number and reduce significant digits. Compare error, symmetry, and minimum eigenvalue for each formulation. Press Escape to return focus to Bento; Page Up returns to the implementation introduction.',
    elements: [
      text('live-eyebrow', 74, 37, 870, 21, 'NUMERICAL FORMS · LIVE EXPERIMENT', { fontSize: 10, fontFamily: MONO, fontWeight: 900, color: C.violet, letterSpacing: 1.45 }),
      text('live-title', 74, 66, 1050, 48, 'Stress the arithmetic, not the estimator.', { fontSize: 35, fontFamily: SERIF, fontWeight: 700, lineHeight: 1.05 }),
      text('live-prompt', 74, 116, 1080, 28, 'Increase conditioning and lower precision. Exact identities separate for numerical—not statistical—reasons.', { fontSize: 15, color: C.muted }),
      ...precisionFallback(),
      shape('live-demo-mount', 'rect', LIVE_BOUNDS.x, LIVE_BOUNDS.y, LIVE_BOUNDS.width, LIVE_BOUNDS.height, { fill: 'rgba(255,255,255,0)', stroke: 'rgba(255,255,255,0)', strokeWidth: 0, opacity: 0 }),
      ...chrome('Numerical forms · live', C.violet)
    ]
  };
}

function equivalenceSlide() {
  const elements = [
    ...heading('Synthesis', 'Same answer; different assumptions', 'Under the linear–Gaussian model, the families agree. Outside it, the distinctions matter.', C.rust),
    ...panel('equiv-linear', 72, 184, 552, 190, 'UNDER THE LINEAR–GAUSSIAN MODEL', '<span style="font-family:Georgia,serif;font-size:24px"><b>posterior mean = MAP = LMMSE</b></span><br><br>Bayes computes the posterior. Projection and least squares recover its mean. Information, graph elimination, and QR reorganize the computation. KL expresses the posterior variationally.', { accent: C.green, fill: C.greenSoft, stroke: C.green, fontSize: 14.5, lineHeight: 1.42 }),
    ...panel('equiv-outside', 656, 184, 552, 190, 'OUTSIDE THAT MODEL', 'With non-Gaussian noise, LMMSE need not equal the posterior mean; MAP need not equal it either.<br><br>Correlated noises require modified cross-covariances. Nonlinear models do not preserve the exact equivalences.', { accent: C.rust, fill: C.rustSoft, stroke: C.rust, fontSize: 15.5, lineHeight: 1.46 }),
    ...panel('equiv-scalar', 72, 402, 552, 206, 'ONE SCALAR SANITY CHECK', 'm<sup>−</sup>=0, P<sup>−</sup>=4, H=1, z=3, R=1<br><br>S=5, &nbsp; K=0.8<br>m<sup>+</sup>=2.4, &nbsp; P<sup>+</sup>=0.8<br><br><span style="color:#66756E">Every applicable route returns these numbers.</span>', { accent: C.blue, fill: C.blueSoft, stroke: C.blue, fontFamily: SERIF, fontSize: 17, lineHeight: 1.4 }),
    ...panel('equiv-choice', 656, 402, 552, 206, 'CHOOSE THE LANGUAGE FOR THE QUESTION', '<b>Probability</b> for belief and closure.<br><b>Projection</b> for optimality and geometry.<br><b>Least squares</b> for objectives and solvers.<br><b>KL</b> for distributional updating.<br><b>QR / information</b> for numerical structure.', { accent: C.violet, fill: C.violetSoft, stroke: C.violet, fontSize: 16, lineHeight: 1.48 }),
    ...chrome('Synthesis', C.rust)
  ];
  return {
    id: 'equivalence', background: C.paper, transition: 'morph',
    notes: 'Close the conceptual argument. The same estimator does not mean the assumptions or outputs are interchangeable. Use the scalar example as a sanity check and the final panel as a decision guide.',
    elements
  };
}

function referencesSlide() {
  const refs = [
    ['[1]', 'Särkkä (2013)', 'Bayesian Filtering and Smoothing · Chapter 4 and Gaussian identities.', 'https://users.aalto.fi/~ssarkka/pub/cup_book_online_20131111.pdf'],
    ['[2]', 'Kalman (1960)', 'A New Approach to Linear Filtering and Prediction Problems.', 'https://doi.org/10.1115/1.3662552'],
    ['[3]', 'Kailath (1968)', 'An Innovations Approach to Least-Squares Estimation—Part I.', 'https://doi.org/10.1109/TAC.1968.1099025'],
    ['[4]', 'Loeliger et al. (2007)', 'The Factor Graph Approach to Model-Based Signal Processing.', 'https://www.isiweb.ee.ethz.ch/papers/arch/aloe-jdau-juhu-skor-2007-1.pdf'],
    ['[5]', 'Aitken (1936)', 'On Least Squares and Linear Combination of Observations.', 'https://doi.org/10.1017/S0370164600014346'],
    ['[6]', 'Bissiri, Holmes & Walker (2016)', 'A General Framework for Updating Belief Distributions.', 'https://arxiv.org/abs/1306.6430'],
    ['[7]', 'Giffin & Urniezius (2014)', 'The Kalman Filter Revisited Using Maximum Relative Entropy.', 'https://doi.org/10.3390/e16021047'],
    ['[8]', 'Kaminski, Bryson & Schmidt (1971)', 'Discrete Square Root Filtering: A Survey of Current Techniques.', 'https://doi.org/10.1109/TAC.1971.1099816'],
    ['[9]', 'Hartikainen & Särkkä (2010)', 'Kalman solutions to temporal Gaussian-process regression models.', 'https://users.aalto.fi/~ssarkka/pub/gp-ts-kfrts.pdf'],
    ['[10]', 'Kalata & Priemer (1979)', 'Linear prediction, filtering, and smoothing: an information-theoretic approach.', 'https://doi.org/10.1016/0020-0255(79)90039-2']
  ];
  const elements = [
    ...heading('Sources', 'Foundations and further reading', 'Primary references behind the four-family consolidation and its numerical notes.', C.green),
  ];
  refs.forEach(([number, author, title, link], index) => {
    const col = index % 2;
    const row = Math.floor(index / 2);
    const x = 72 + col * 576;
    const y = 176 + row * 88;
    elements.push(card(`ref-card-${index}`, x, y, 552, 74, index % 4 < 2 ? C.panel : C.greenSoft, C.rule, 10));
    elements.push(text(`ref-number-${index}`, x + 14, y + 13, 44, 18, number, { fontSize: 9.5, fontFamily: MONO, fontWeight: 900, color: C.rust, link }));
    elements.push(text(`ref-author-${index}`, x + 62, y + 10, 465, 22, author, { fontSize: 13.5, fontFamily: SERIF, fontWeight: 700, color: C.ink, link }));
    elements.push(text(`ref-title-${index}`, x + 62, y + 35, 465, 28, title, { fontSize: 10.5, color: C.muted, lineHeight: 1.25, link }));
  });
  elements.push(
    card('ref-note-card', 72, 626, 1136, 34, C.rustSoft, C.rust, 9),
    text('ref-note', 92, 631, 1096, 23, 'Editorial boundary · four families consolidate synonymous derivations while keeping statistical principles distinct from numerical implementations.', { fontSize: 10.5, fontFamily: MONO, fontWeight: 800, color: C.rust, align: 'center', valign: 'middle' }),
    ...chrome('References', C.green)
  );
  return {
    id: 'references', background: C.paper, transition: 'morph',
    notes: 'Use these sources to mark the boundary between the original results and the editorial consolidation. The citations are clickable in Bento. End by repeating that algorithms and identities should not be counted as independent filters.',
    elements
  };
}

const slides = [
  overviewSlide(),
  modelSlide(),
  scalarLiveSlide(),
  bayesSlide(),
  bayesEquationsSlide(),
  mseSlide(),
  geometryLiveSlide(),
  mseEquationsSlide(),
  leastSquaresSlide(),
  leastSquaresEquationsSlide(),
  klSlide(),
  klEquationsSlide(),
  graphsSlide(),
  implementationsSlide(),
  precisionLiveSlide(),
  equivalenceSlide(),
  referencesSlide()
];

export const deck = {
  format: 'bento/slides',
  version: 1,
  docId: 'kalman-filter-four-families-bento',
  title: 'One Filter, Four Derivation Families',
  readonly: true,
  meta: {
    author: 'Bai Liping',
    subject: 'Kalman filter derivations and interactive experiments',
    company: 'bailiping.com'
  },
  size: { width: WIDTH, height: HEIGHT },
  theme: { background: C.paper, color: C.ink, accent: C.green, fontFamily: SANS },
  slides
};

function indexOf(id) {
  const index = slides.findIndex(slide => slide.id === id);
  if (index < 0) throw new Error(`Unknown slide ${id}`);
  return index;
}

export const inlineLiveMap = [
  {
    introSlide: 'model',
    slide: 'model-live',
    slideIndex: indexOf('model-live'),
    inline: true,
    layout: 'region',
    bounds: LIVE_BOUNDS,
    src: './live/?demo=scalar&embed=region',
    source: './live/?demo=scalar',
    title: 'Interactive scalar Kalman fusion',
    sandbox: 'allow-scripts',
    hideSource: true,
    readyMessage: true,
    unloadWhenHidden: true
  },
  {
    introSlide: 'mse',
    slide: 'mse-live',
    slideIndex: indexOf('mse-live'),
    inline: true,
    layout: 'region',
    bounds: LIVE_BOUNDS,
    src: './live/?demo=geometry&embed=region',
    source: './live/?demo=geometry',
    title: 'Interactive covariance geometry',
    sandbox: 'allow-scripts',
    hideSource: true,
    readyMessage: true,
    unloadWhenHidden: true
  },
  {
    introSlide: 'implementations',
    slide: 'implementations-live',
    slideIndex: indexOf('implementations-live'),
    inline: true,
    layout: 'region',
    bounds: LIVE_BOUNDS,
    src: './live/?demo=equivalence&embed=region',
    source: './live/?demo=equivalence',
    title: 'Interactive finite-precision comparison',
    sandbox: 'allow-scripts',
    hideSource: true,
    readyMessage: true,
    unloadWhenHidden: true
  }
];
