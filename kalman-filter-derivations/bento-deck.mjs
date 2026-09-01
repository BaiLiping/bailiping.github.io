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

function tex(strings, ...values) {
  const source = String.raw(strings, ...values);
  return `<span class="math-tex math-inline">\\(${source}\\)</span>`;
}

function texBlock(strings, ...values) {
  const source = String.raw(strings, ...values);
  return `<span class="math-tex math-display">\\[${source}\\]</span>`;
}

function mathLines(...lines) {
  return lines.join('<br>');
}

function mathParagraphs(...paragraphs) {
  return paragraphs.join('<br><br>');
}

function muted(html) {
  return `<span style="color:${C.muted}">${html}</span>`;
}

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
    shape('cover-path', 'line', 82, 286, 698, 2, {
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
    const y = 330 + Math.floor(index / 2) * 105;
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
    text('cover-result-symbol', 952, 243, 146, 64, texBlock`m^+,\;P^+`, {
      fontSize: 31, fontFamily: SERIF, fontWeight: 700, align: 'center', valign: 'middle'
    }),
    text('cover-result-formula', 874, 424, 302, 88, texBlock`\begin{aligned}m^+ &= m^- + K(z-Hm^-)\\ P^+ &= P^- - KSK^\mathsf{T}\end{aligned}`, {
      fontSize: 20, fontFamily: SERIF, fontWeight: 700, align: 'center', lineHeight: 1.55
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
    ...panel('model-dynamics', 72, 180, 354, 176, 'LINEAR–GAUSSIAN MODEL', mathParagraphs(
      mathLines(
        tex`x_k = F_k x_{k-1} + B_k u_k + w_k`,
        tex`z_k = H_k x_k + v_k`
      ),
      muted(`${tex`w_k \sim \mathcal{N}(0,Q_k)`}, ${tex`v_k \sim \mathcal{N}(0,R_k)`}, independent.`)
    ), {
      accent: C.green, fill: C.greenSoft, stroke: C.green, fontFamily: SERIF, fontSize: 18, lineHeight: 1.45
    }),
    ...panel('model-predict', 463, 180, 354, 176, 'PREDICT', mathLines(
      tex`m_k^- = F_k m_{k-1}^+ + B_k u_k`,
      tex`P_k^- = F_k P_{k-1}^+ F_k^\mathsf{T} + Q_k`
    ), {
      accent: C.blue, fill: C.blueSoft, stroke: C.blue, fontFamily: SERIF, fontSize: 20, lineHeight: 1.65
    }),
    ...panel('model-correct', 854, 180, 354, 176, 'CORRECT', mathLines(
      tex`\nu_k = z_k - H_k m_k^-`,
      tex`S_k = H_k P_k^- H_k^\mathsf{T} + R_k`,
      tex`K_k = P_k^- H_k^\mathsf{T} S_k^{-1}`,
      tex`m_k^+ = m_k^- + K_k\nu_k`
    ), {
      accent: C.rust, fill: C.rustSoft, stroke: C.rust, fontFamily: SERIF, fontSize: 18, lineHeight: 1.35
    }),
    card('model-observe', 72, 390, 1136, 214, C.panel, C.rule, 16),
    text('model-observe-label', 96, 414, 230, 22, 'WHAT TO WATCH NEXT', {
      fontSize: 11, fontFamily: MONO, fontWeight: 900, color: C.green, letterSpacing: 1
    }),
    text('model-observe-question', 96, 454, 520, 86, 'How far should the posterior move toward the measurement?', {
      fontSize: 29, fontFamily: SERIF, fontWeight: 700, lineHeight: 1.16
    }),
    text('model-observe-answer', 96, 554, 520, 28, `The gain ${tex`K`} is a confidence-weighted answer.`, {
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
    text('scalar-fallback-controls-copy', x + 34, y + 75, 242, 174, mathParagraphs(
      mathLines(tex`m^-=-1.2`, tex`\sigma_p=1.35`),
      mathLines(tex`z=2.1`, tex`\sigma_r=0.75`)
    ), {
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
    text('scalar-metric-gain-value', x + 866, y + 130, 200, 42, texBlock`K=0.764`, { fontSize: 25, fontFamily: SERIF, fontWeight: 700, align: 'center' }),
    card('scalar-metric-post', x + 848, y + 212, 236, 142, C.rustSoft, C.rust, 11),
    text('scalar-metric-post-label', x + 866, y + 228, 200, 20, 'POSTERIOR', { fontSize: 9, fontFamily: MONO, fontWeight: 900, color: C.rust, align: 'center' }),
    text('scalar-metric-post-value', x + 866, y + 259, 200, 70, texBlock`\begin{aligned}m^+&\approx1.32\\ \sigma^+&\approx0.66\end{aligned}`, { fontSize: 21, fontFamily: SERIF, fontWeight: 700, align: 'center', lineHeight: 1.4 }),
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
    formula: texBlock`p_k^-(x)=\int p(x\mid x')p_{k-1}^+(x')\,dx' \qquad p_k^+(x)\propto p(z_k\mid x)p_k^-(x)`,
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
    formula: texBlock`\widehat{x}(K)=m^-+K\nu \qquad K_\star=\underset{K}{\operatorname{arg\,min}}\;\mathbb{E}\!\left[\lVert x-\widehat{x}(K)\rVert_2^2\right]`,
    leftTitle: 'TWO PROOFS · ONE OPTIMUM',
    leftBody: `<b>Orthogonality:</b> the remaining error is uncorrelated with the innovation.<br><br><b>Covariance minimization:</b> differentiate the trace of the error covariance.<br><br>Both give ${tex`K_\star S=P^-H^\mathsf{T}`}.`,
    rightTitle: 'GEOMETRIC READING',
    rightBody: `The measurement selects a direction through ${tex`H`}. Cross-covariance ${tex`P^-H^\mathsf{T}`} carries that scalar evidence into the state.<br><br>Noise ${tex`R`} weakens the contraction; correlation rotates how the correction spreads.`,
    note: `Rotate ${tex`H`}, vary ${tex`R`}, and change correlation ${tex`\rho`}. Watch the posterior ellipse contract mainly along the measured slice.`,
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
    formula: texBlock`\phi(x)=\tfrac12\lVert x-m^-\rVert_{(P^-)^{-1}}^2+\tfrac12\lVert z-Hx\rVert_{R^{-1}}^2`,
    leftTitle: 'STATISTICAL INTERPRETATIONS',
    leftBody: '<b>MAP:</b> this is the negative Gaussian log posterior; its mode equals its mean.<br><br><b>BLUE:</b> the same generalized least-squares algebra applies to independent unbiased observations of a fixed state—but the experiment differs.',
    rightTitle: 'ALGORITHMS · NOT NEW PRINCIPLES',
    rightBody: `<b>RLS</b> updates the normal equations as observations arrive; the static case has ${tex`F=I`} and ${tex`Q=0`}.<br><br><b>Square-root / QR</b> solves the whitened system without forming its normal matrix.`,
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
    formula: texBlock`q_\star=\underset{q\ge 0,\;\int q=1}{\operatorname{arg\,min}}\left\{D_{\mathrm{KL}}(q\Vert p^-)+\mathbb{E}_q[-\log p(z\mid x)]\right\}`,
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
    context: `One correction: ${tex`x\sim\mathcal{N}(m^-,P^-)`}, ${tex`z=Hx+v`}, ${tex`v\sim\mathcal{N}(0,R)`}, independent; ${tex`P^-\succ0`} and ${tex`R\succ0`}.`,
    accent: C.green, soft: C.greenSoft,
    panels: [
      { title: 'Multiply Gaussian factors', body: mathParagraphs(
        tex`p^+(x)\propto p^-(x)p(z\mid x)`,
        mathLines(
          tex`-\log p^+(x)=\tfrac12\lVert x-m^-\rVert_{(P^-)^{-1}}^2`,
          tex`\qquad\quad+\tfrac12\lVert z-Hx\rVert_{R^{-1}}^2+c`
        )
      ) },
      { title: 'Collect information', body: mathParagraphs(
        mathLines(
          tex`J^+=(P^-)^{-1}+H^\mathsf{T}R^{-1}H`,
          tex`h^+=(P^-)^{-1}m^-+H^\mathsf{T}R^{-1}z`
        ),
        muted('Precision and information vector add.')
      ) },
      { title: 'Complete the square', body: mathParagraphs(
        mathLines(tex`m^+=(J^+)^{-1}h^+`, tex`P^+=(J^+)^{-1}`),
        tex`p^+(x)\propto\exp\!\left[-\tfrac12(x-m^+)^\mathsf{T}J^+(x-m^+)\right]`
      ), fontSize: 16 },
      { title: 'Same answer by conditioning', body: mathLines(
        tex`S=HP^-H^\mathsf{T}+R`,
        tex`m^+=m^-+P^-H^\mathsf{T}S^{-1}(z-Hm^-)`,
        tex`P^+=P^--P^-H^\mathsf{T}S^{-1}HP^-`
      ), fontSize: 15.5 }
    ],
    notes: 'Walk clockwise: factor multiplication, information addition, completing the square, and joint conditioning. The four boxes are one derivation written in complementary coordinates.'
  });
}

function mseEquationsSlide() {
  return equationSheetSlide({
    id: 'mse-equations', family: 'Family 02', title: 'Minimum-MSE estimation · equations',
    context: `Let ${tex`e^-=x-m^-`}, ${tex`\nu=z-Hm^-`}, ${tex`S=HP^-H^\mathsf{T}+R`}; ${tex`e^-`} and ${tex`v`} are zero-mean and uncorrelated, and ${tex`S\succ0`}.`,
    accent: C.blue, soft: C.blueSoft,
    panels: [
      { title: 'Error for an arbitrary gain', body: mathParagraphs(
        tex`e(K)=e^--K\nu`,
        tex`\mathbb{E}[e^-\nu^\mathsf{T}]=P^-H^\mathsf{T}`,
        muted(`Choose ${tex`K`} inside an affine correction.`)
      ) },
      { title: 'Orthogonality gives the gain', body: mathParagraphs(
        tex`\mathbb{E}[e(K_\star)\nu^\mathsf{T}]=0`,
        mathLines(tex`K_\star S=P^-H^\mathsf{T}`, tex`K_\star=P^-H^\mathsf{T}S^{-1}`)
      ) },
      { title: 'Equivalent covariance calculation', body: mathParagraphs(
        tex`P(K)=(I-KH)P^-(I-KH)^\mathsf{T}+KRK^\mathsf{T}`,
        tex`\nabla_K\operatorname{tr}P(K)=2(KS-P^-H^\mathsf{T})`
      ), fontSize: 16 },
      { title: 'Certify the minimum; update', body: mathParagraphs(
        tex`P(K)-P(K_\star)=(K-K_\star)S(K-K_\star)^\mathsf{T}\succeq0`,
        mathLines(tex`m^+=m^-+K_\star\nu`, tex`P^+=P^--K_\star SK_\star^\mathsf{T}`)
      ), fontSize: 15.5 }
    ],
    notes: 'Connect the previous geometric experiment back to the algebra. Orthogonality and trace minimization give the same normal equation. The final positive-semidefinite difference certifies global optimality.'
  });
}

function leastSquaresEquationsSlide() {
  return equationSheetSlide({
    id: 'least-squares-equations', family: 'Family 03', title: 'Weighted least squares · equations',
    context: `One correction: ${tex`x\sim\mathcal{N}(m^-,P^-)`}, ${tex`z=Hx+v`}, ${tex`v\sim\mathcal{N}(0,R)`}, independent; ${tex`P^-\succ0`} and ${tex`R\succ0`}.`,
    accent: C.rust, soft: C.rustSoft,
    panels: [
      { title: 'Objective', body: mathParagraphs(
        mathLines(
          tex`\phi(x)=\tfrac12\lVert x-m^-\rVert_{(P^-)^{-1}}^2`,
          tex`\qquad\quad+\tfrac12\lVert z-Hx\rVert_{R^{-1}}^2`
        ),
        muted('Gaussian MAP objective; Hessian = posterior precision.')
      ) },
      { title: 'Normal equations and curvature', body: mathLines(
        tex`J^+=(P^-)^{-1}+H^\mathsf{T}R^{-1}H`,
        tex`h^+=(P^-)^{-1}m^-+H^\mathsf{T}R^{-1}z`,
        tex`\nabla\phi(x)=J^+x-h^+=0`,
        tex`m^+=(J^+)^{-1}h^+,\quad P^+=(J^+)^{-1}`
      ), fontSize: 14.5 },
      { title: 'Expose the Kalman correction', body: mathParagraphs(
        tex`\nu=z-Hm^-,\qquad S=HP^-H^\mathsf{T}+R`,
        tex`(J^+)^{-1}H^\mathsf{T}R^{-1}=P^-H^\mathsf{T}S^{-1}=K`,
        tex`m^+=m^-+K\nu,\qquad P^+=P^--KSK^\mathsf{T}`
      ), fontSize: 15 },
      { title: 'BLUE: a different experiment', body: mathParagraphs(
        tex`d=\begin{bmatrix}a\\z\end{bmatrix},\quad G=\begin{bmatrix}I\\H\end{bmatrix},\quad W=\operatorname{diag}(P^-,R)`,
        mathLines(
          tex`\widehat{x}=(G^\mathsf{T}W^{-1}G)^{-1}G^\mathsf{T}W^{-1}d`,
          tex`\phantom{\widehat{x}}=a+K(z-Ha)`
        ),
        muted(`Here ${tex`a=x+\varepsilon`} and ${tex`x`} is fixed.`)
      ), fontSize: 14.5 }
    ],
    notes: 'Derive the normal equations, then use Woodbury to expose the Kalman correction. End by restating that BLUE uses a different sampling experiment even when the estimator formula matches.'
  });
}

function klEquationsSlide() {
  return equationSheetSlide({
    id: 'kl-equations', family: 'Family 04', title: 'KL variational updating · equations',
    context: `One correction: ${tex`x\sim\mathcal{N}(m^-,P^-)`}, ${tex`z=Hx+v`}, ${tex`v\sim\mathcal{N}(0,R)`}, independent; optimize over ${tex`q\ge0`} with ${tex`\int q=1`}.`,
    accent: C.violet, soft: C.violetSoft,
    panels: [
      { title: 'Optimize over densities', body: mathParagraphs(
        tex`\ell_z(x)=-\log p(z\mid x)`,
        tex`F(q)=D_{\mathrm{KL}}(q\Vert p^-)+\mathbb{E}_q[\ell_z(x)]`,
        muted('Include the likelihood normalizing constant.')
      ) },
      { title: 'Identify the exact minimizer', body: mathParagraphs(
        tex`F(q)=D_{\mathrm{KL}}(q\Vert p(x\mid z))-\log p(z)`,
        mathLines(
          tex`q_\star(x)\propto p^-(x)e^{-\ell_z(x)}`,
          tex`\phantom{q_\star(x)}=p^-(x)p(z\mid x)`
        )
      ) },
      { title: 'Gaussian parameter objective', body: mathParagraphs(
        tex`q=\mathcal{N}(\mu,\Sigma),\quad J=(P^-)^{-1}+H^\mathsf{T}R^{-1}H`,
        mathLines(
          tex`F(\mu,\Sigma)=\tfrac12\lVert\mu-m^-\rVert_{(P^-)^{-1}}^2+\tfrac12\lVert z-H\mu\rVert_{R^{-1}}^2`,
          tex`\qquad\qquad+\tfrac12\operatorname{tr}(J\Sigma)-\tfrac12\log\det\Sigma+c`
        )
      ), fontSize: 14.2 },
      { title: 'Recover mean and covariance', body: mathParagraphs(
        tex`h=(P^-)^{-1}m^-+H^\mathsf{T}R^{-1}z`,
        mathLines(tex`\nabla_\mu F=0\;\Longrightarrow\;J\mu=h`, tex`\nabla_\Sigma F=0\;\Longrightarrow\;\Sigma^{-1}=J`),
        tex`m^+=J^{-1}h,\qquad P^+=J^{-1}`
      ), fontSize: 15.5 }
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
    text('geometry-fallback-values', x + 34, y + 75, 242, 214, mathParagraphs(
      mathLines(tex`\sigma_x=1.80`, tex`\sigma_y=1.00`, tex`\rho=0.65`),
      mathLines(tex`\varphi=28^\circ`, tex`z=1.70`, tex`\sigma_r=0.45`)
    ), { fontSize: 15.5, fontFamily: MONO, fontWeight: 700, lineHeight: 1.52 }),
    text('geometry-fallback-hint', x + 34, y + 368, 242, 62, `Rotate ${tex`H`} and watch correlation carry evidence across coordinates.`, { fontSize: 13, color: C.muted, lineHeight: 1.45 }),
    card('geometry-fallback-stage', x + 312, y + 16, width - 328, height - 32, C.panel, C.rule, 12),
    text('geometry-fallback-stage-label', x + 338, y + 34, 520, 22, 'PRIOR ELLIPSE → MEASUREMENT STRIP → POSTERIOR', { fontSize: 10, fontFamily: MONO, fontWeight: 900, color: C.blue, letterSpacing: 1 }),
    shape('geometry-prior', 'ellipse', x + 410, y + 126, 382, 236, { fill: C.blueSoft, stroke: C.blue, strokeWidth: 4, rotation: 23 }),
    shape('geometry-posterior', 'ellipse', x + 516, y + 176, 205, 132, { fill: C.greenSoft, stroke: C.green, strokeWidth: 5, rotation: 23 }),
    shape('geometry-measurement', 'rect', x + 398, y + 276, 446, 4, { fill: C.rust, rotation: -28 }),
    text('geometry-prior-label', x + 382, y + 385, 220, 24, `prior ${tex`P^-`}`, { fontSize: 12, fontFamily: MONO, fontWeight: 850, color: C.blue, align: 'center' }),
    text('geometry-post-label', x + 632, y + 339, 220, 24, `posterior ${tex`P^+`}`, { fontSize: 12, fontFamily: MONO, fontWeight: 850, color: C.green, align: 'center' }),
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
      text('live-prompt', 74, 116, 1080, 28, `Vary ${tex`H`}, ${tex`R`}, and correlation. Cross-covariance decides how scalar evidence spreads through the state.`, { fontSize: 15, color: C.muted }),
      ...geometryFallback(),
      shape('live-demo-mount', 'rect', LIVE_BOUNDS.x, LIVE_BOUNDS.y, LIVE_BOUNDS.width, LIVE_BOUNDS.height, { fill: 'rgba(255,255,255,0)', stroke: 'rgba(255,255,255,0)', strokeWidth: 0, opacity: 0 }),
      ...chrome('Minimum-MSE · live', C.blue)
    ]
  };
}

function graphsSlide() {
  return equationSheetSlide({
    id: 'graphs', family: 'Computational notes', title: 'Gaussian elimination · the computational form',
    context: `${tex`a\sim\mathcal{N}(m_a,P_a)`}, ${tex`b=Fa+w`}, ${tex`z=Hb+v`}; independent ${tex`w\sim\mathcal{N}(0,Q)`}, ${tex`v\sim\mathcal{N}(0,R)`}, with ${tex`P_a,Q,R\succ0`}.`,
    accent: C.gold, soft: C.goldSoft,
    panels: [
      { title: 'Write the joint canonical Gaussian', body: mathParagraphs(
        tex`u=\begin{bmatrix}a\\b\end{bmatrix},\quad \Lambda=\begin{bmatrix}A&C\\C^\mathsf{T}&D\end{bmatrix},\quad \eta=\begin{bmatrix}\eta_a\\\eta_b\end{bmatrix}`,
        tex`p(u)\propto\exp\!\left(-\tfrac12u^\mathsf{T}\Lambda u+u^\mathsf{T}\eta\right)`
      ), fontSize: 16 },
      { title: 'Blocks from the transition factor', body: mathParagraphs(
        mathLines(
          tex`A=P_a^{-1}+F^\mathsf{T}Q^{-1}F`,
          tex`C=-F^\mathsf{T}Q^{-1},\qquad D=Q^{-1}`
        ),
        tex`\eta_a=P_a^{-1}m_a,\qquad \eta_b=0`
      ) },
      { title: 'Eliminate a: prediction message', body: mathParagraphs(
        mathLines(tex`J^-=D-C^\mathsf{T}A^{-1}C`, tex`h^-=\eta_b-C^\mathsf{T}A^{-1}\eta_a`),
        mathLines(tex`P^-=(J^-)^{-1}=FP_aF^\mathsf{T}+Q`, tex`m^-=P^-h^-=Fm_a`)
      ), fontSize: 15 },
      { title: 'Add the observation factor', body: mathParagraphs(
        mathLines(tex`J^+=J^-+H^\mathsf{T}R^{-1}H`, tex`h^+=h^-+H^\mathsf{T}R^{-1}z`),
        mathLines(tex`P^+=(J^+)^{-1}`, tex`m^+=P^+h^+`)
      ), fontSize: 16 }
    ],
    notes: 'Present Gaussian message passing as block elimination. The Schur complement is the prediction message; adding the observation factor is the information-form correction. This is a computational organization, not a fifth derivation family.'
  });
}

function implementationsSlide() {
  const elements = [
    ...heading('Numerical forms · experiment introduction', 'Numerical forms and control duality', 'The estimator is fixed. Arithmetic path, conditioning, and factorization determine numerical behavior.', C.violet),
    ...panel('impl-qr', 72, 184, 552, 300, 'SQUARE ROOT / QR', mathParagraphs(
      tex`P^-=L_pL_p^\mathsf{T},\qquad R=L_rL_r^\mathsf{T}`,
      tex`A=\begin{bmatrix}L_p^{-1}\\L_r^{-1}H\end{bmatrix},\qquad b=\begin{bmatrix}L_p^{-1}m^-\\L_r^{-1}z\end{bmatrix}`,
      mathLines(tex`A=UT,\qquad Tm^+=U^\mathsf{T}b`, tex`P^+=T^{-1}T^{-\mathsf{T}}`),
      muted('Use triangular solves; avoid squaring the stacked system’s condition number.')
    ), {
      accent: C.violet, fill: C.violetSoft, stroke: C.violet, fontFamily: SERIF, fontSize: 16.5, lineHeight: 1.43
    }),
    ...panel('impl-riccati', 656, 184, 552, 300, 'RICCATI / LQR DUALITY', mathParagraphs(
      mathLines(
        tex`\Pi_{k+1}=F\Pi_kF^\mathsf{T}+Q`,
        tex`\qquad-F\Pi_kH^\mathsf{T}(H\Pi_kH^\mathsf{T}+R)^{-1}H\Pi_kF^\mathsf{T}`
      ),
      tex`\Pi_k=P_k^-`,
      muted(`LQR uses ${tex`A_c=F^\mathsf{T}`} and ${tex`B_c=H^\mathsf{T}`}; finite-horizon time directions reverse. Constant gain also requires convergence.`)
    ), {
      accent: C.gold, fill: C.goldSoft, stroke: C.gold, fontFamily: SERIF, fontSize: 16, lineHeight: 1.45
    }),
    card('impl-watch', 72, 516, 1136, 116, C.panel, C.violet, 14),
    text('impl-watch-label', 96, 538, 180, 20, 'WHAT TO WATCH NEXT', { fontSize: 10, fontFamily: MONO, fontWeight: 900, color: C.violet, letterSpacing: 1 }),
    text('impl-watch-copy', 272, 530, 912, 74, `<b>Raise ${tex`\operatorname{cond}(P^-)`}, then lower simulated precision.</b><br>${muted(`Covariance subtraction, information inversion, Joseph stabilization, and square-root QR target the same ${tex`P^+`}—but cease to behave identically in finite arithmetic.`)}`, { fontSize: 16, lineHeight: 1.5, valign: 'middle' }),
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
    ['Covariance subtraction', texBlock`P^- - KSK^\mathsf{T}`, C.rust, C.rustSoft],
    ['Information inversion', texBlock`\left((P^-)^{-1}+H^\mathsf{T}R^{-1}H\right)^{-1}`, C.blue, C.blueSoft],
    ['Joseph stabilization', texBlock`(I-KH)P^-(I-KH)^\mathsf{T}+KRK^\mathsf{T}`, C.green, C.greenSoft],
    ['Square root / QR', texBlock`\text{whiten}\;\longrightarrow\;\mathrm{QR}\;\longrightarrow\;\text{solve}`, C.violet, C.violetSoft]
  ];
  const elements = [
    card('precision-fallback-region', x, y, width, height, '#FBFAF6', C.rule, 14),
    card('precision-fallback-controls', x + 16, y + 16, 278, height - 32, C.violetSoft, C.violet, 12),
    text('precision-fallback-label', x + 34, y + 33, 242, 22, 'DEFAULT STRESS TEST', { fontSize: 10, fontFamily: MONO, fontWeight: 900, color: C.violet, letterSpacing: 1 }),
    text('precision-fallback-values', x + 34, y + 78, 242, 154, mathLines(
      tex`n=3`,
      tex`m=1`,
      tex`\operatorname{cond}(P^-)=10^3`,
      tex`\text{precision}=\text{double}`
    ), { fontSize: 15.5, fontFamily: MONO, fontWeight: 700, lineHeight: 1.55 }),
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
    elements.push(text(`precision-method-status-${index}`, mx + 17, my + 104, 314, 18, `${tex`\max\Delta\approx0`} · positive definite`, { fontSize: 9.5, fontFamily: MONO, fontWeight: 850, color: C.green }));
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
  const choices = [
    ['PROBABILITY', 'belief and closure', C.green, C.greenSoft],
    ['PROJECTION', 'optimality and geometry', C.blue, C.blueSoft],
    ['LEAST SQUARES', 'objectives and solvers', C.rust, C.rustSoft],
    ['KL', 'distributional updating', C.violet, C.violetSoft],
    ['QR / INFORMATION', 'numerical structure', C.gold, C.goldSoft]
  ];
  const elements = [
    ...heading('Synthesis', 'Same answer; different assumptions', 'Under the linear–Gaussian model, the families agree. Outside it, the distinctions matter.', C.rust),
    ...panel('equiv-linear', 72, 184, 552, 190, 'UNDER THE LINEAR–GAUSSIAN MODEL', `<span style="font-size:24px;font-weight:700">${tex`\operatorname{posterior\ mean}=\mathrm{MAP}=\mathrm{LMMSE}`}</span><br><br>Bayes computes the posterior. Projection and least squares recover its mean. Information, graph elimination, and QR reorganize the computation. KL expresses the posterior variationally.`, { accent: C.green, fill: C.greenSoft, stroke: C.green, fontSize: 14.5, lineHeight: 1.42 }),
    ...panel('equiv-outside', 656, 184, 552, 190, 'OUTSIDE THAT MODEL', 'With non-Gaussian noise, LMMSE need not equal the posterior mean; MAP need not equal it either.<br><br>Correlated noises require modified cross-covariances. Nonlinear models do not preserve the exact equivalences.', { accent: C.rust, fill: C.rustSoft, stroke: C.rust, fontSize: 15.5, lineHeight: 1.46 }),
    card('equiv-choice-card', 72, 402, 1136, 206, C.panel, C.violet, 14),
    text('equiv-choice-title', 94, 421, 1092, 22, 'CHOOSE THE LANGUAGE FOR THE QUESTION', {
      fontSize: 11, fontFamily: MONO, fontWeight: 850, color: C.violet, letterSpacing: 0.65
    })
  ];
  choices.forEach(([label, body, accent, soft], index) => {
    const x = 92 + index * 219;
    elements.push(card(`equiv-choice-${index}`, x, 458, 208, 124, soft, accent, 11));
    elements.push(text(`equiv-choice-${index}-label`, x + 13, 475, 182, 22, label, {
      fontSize: 9.5, fontFamily: MONO, fontWeight: 900, color: accent, align: 'center', letterSpacing: 0.5
    }));
    elements.push(text(`equiv-choice-${index}-body`, x + 15, 516, 178, 44, body, {
      fontSize: 14, fontFamily: SERIF, fontWeight: 700, align: 'center', lineHeight: 1.25
    }));
  });
  elements.push(...chrome('Synthesis', C.rust));
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
