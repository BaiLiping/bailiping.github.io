const WIDTH = 1280;
const HEIGHT = 720;
const SERIF = "Georgia, 'Times New Roman', serif";
const SANS = "Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
const MONO = "'SFMono-Regular', Consolas, 'Liberation Mono', monospace";

const C = {
  paper: '#F8F6F1',
  panel: '#FFFEFB',
  ink: '#202B33',
  muted: '#66737D',
  faint: '#8A949B',
  rule: '#D8DEE2',
  violet: '#6557A7',
  violetSoft: '#ECE9F7',
  blue: '#39708C',
  blueSoft: '#E5EFF4',
  teal: '#2D7A70',
  tealSoft: '#E3F0ED',
  rust: '#A95736',
  rustSoft: '#F5E8E1',
  gold: '#92702E',
  goldSoft: '#F4EEDC',
  rose: '#A34F6A',
  roseSoft: '#F5E6EB'
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
    ...(options.fx ? { fx: options.fx } : {})
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
    ...(options.fx ? { fx: options.fx } : {})
  };
}

function card(id, x, y, w, h, fill = C.panel, stroke = C.rule, radius = 14) {
  return shape(id, 'rect', x, y, w, h, { fill, stroke, strokeWidth: 1, radius });
}

function chrome(section, accent = C.violet) {
  return [
    shape('chrome-rule', 'rect', 72, 674, 1136, 1, { fill: C.rule }),
    text('chrome-site', 72, 687, 330, 17, 'BAI LIPING · RANDOM THOUGHTS', {
      fontSize: 10,
      fontFamily: MONO,
      fontWeight: 750,
      color: C.faint,
      letterSpacing: 0.7
    }),
    text('chrome-section', 430, 685, 420, 18, section.toUpperCase(), {
      fontSize: 10,
      fontFamily: MONO,
      fontWeight: 800,
      color: accent,
      align: 'center',
      letterSpacing: 1.1
    })
  ];
}

function heading(eyebrow, title, subtitle, accent = C.violet, options = {}) {
  return [
    text('slide-eyebrow', 72, 38, 900, 22, eyebrow.toUpperCase(), {
      fontSize: 11,
      fontFamily: MONO,
      fontWeight: 850,
      color: accent,
      letterSpacing: 1.55
    }),
    text('slide-title', 72, 68, 1110, options.titleHeight ?? 55, title, {
      fontSize: options.titleSize ?? 38,
      fontFamily: SERIF,
      fontWeight: 700,
      color: C.ink,
      lineHeight: 1.05
    }),
    ...(subtitle ? [text('slide-subtitle', 72, options.subtitleY ?? 119, 1110, options.subtitleHeight ?? 43, subtitle, {
      fontSize: options.subtitleSize ?? 16,
      color: C.muted,
      lineHeight: 1.35
    })] : [])
  ];
}

function panel(id, x, y, w, h, title, body, options = {}) {
  const accent = options.accent ?? C.violet;
  return [
    card(`${id}-card`, x, y, w, h, options.fill ?? C.panel, options.stroke ?? C.rule, options.radius ?? 14),
    text(`${id}-title`, x + 18, y + 15, w - 36, options.titleHeight ?? 22, title, {
      fontSize: options.titleSize ?? 11,
      fontFamily: options.titleFamily ?? MONO,
      fontWeight: 850,
      color: accent,
      letterSpacing: options.letterSpacing ?? 0.65
    }),
    text(`${id}-body`, x + 18, y + (options.bodyY ?? 48), w - 36, h - (options.bodyY ?? 48) - 14, body, {
      fontSize: options.fontSize ?? 15,
      fontFamily: options.fontFamily ?? SANS,
      fontWeight: options.fontWeight ?? 400,
      color: options.color ?? C.ink,
      lineHeight: options.lineHeight ?? 1.42,
      align: options.align ?? 'left',
      valign: options.valign ?? 'top'
    })
  ];
}

function overviewSlide() {
  const elements = [
    shape('cover-accent', 'rect', 0, 0, 16, HEIGHT, { fill: C.violet }),
    text('cover-eyebrow', 82, 62, 720, 26, 'APPROXIMATE INFERENCE · INTERACTIVE DECK', {
      fontSize: 12,
      fontFamily: MONO,
      fontWeight: 850,
      color: C.violet,
      letterSpacing: 1.7,
      fx: { enter: 'fade-up', order: 0 }
    }),
    text('cover-title', 80, 105, 735, 146, 'Variational<br><span style="color:#2D7A70">inference.</span>', {
      fontSize: 67,
      fontFamily: SERIF,
      fontWeight: 700,
      lineHeight: 0.96,
      fx: { enter: 'fade-up', order: 1 }
    }),
    text('cover-subtitle', 82, 272, 690, 62, 'Replace an intractable posterior calculation with a tractable optimization problem—and keep track of what the approximation gives up.', {
      fontSize: 19,
      color: C.muted,
      lineHeight: 1.42
    })
  ];

  const topics = [
    ['01', 'The ELBO', 'one objective, two readings', C.violet, C.violetSoft, 'elbo'],
    ['02', 'Mean field', 'factorize, then coordinate-ascent', C.blue, C.blueSoft, 'mean-field'],
    ['03', 'EM', 'exact local inference as an E-step', C.rust, C.rustSoft, 'em'],
    ['04', 'Gradient VI', 'sample, differentiate, optimize', C.teal, C.tealSoft, 'stochastic']
  ];
  topics.forEach(([number, name, detail, accent, soft, link], index) => {
    const x = 82 + (index % 2) * 360;
    const y = 372 + Math.floor(index / 2) * 102;
    elements.push(card(`cover-topic-${number}`, x, y, 338, 82, soft, accent, 14));
    elements.push(text(`cover-topic-number-${number}`, x + 16, y + 17, 42, 20, number, {
      fontSize: 11,
      fontFamily: MONO,
      fontWeight: 900,
      color: accent,
      link
    }));
    elements.push(text(`cover-topic-name-${number}`, x + 61, y + 12, 252, 27, name, {
      fontSize: 17,
      fontFamily: SERIF,
      fontWeight: 700,
      link
    }));
    elements.push(text(`cover-topic-detail-${number}`, x + 61, y + 45, 252, 21, detail, {
      fontSize: 10.5,
      fontFamily: MONO,
      fontWeight: 750,
      color: accent,
      link
    }));
  });

  elements.push(
    card('cover-objective-card', 842, 70, 366, 522, C.panel, C.rule, 22),
    text('cover-objective-label', 874, 103, 302, 23, 'THE VARIATIONAL MOVE', {
      fontSize: 11,
      fontFamily: MONO,
      fontWeight: 850,
      color: C.violet,
      align: 'center',
      letterSpacing: 1.25
    }),
    shape('cover-prior', 'ellipse', 904, 162, 150, 150, { fill: C.violetSoft, stroke: C.violet, strokeWidth: 2 }),
    shape('cover-family', 'ellipse', 996, 218, 150, 150, { fill: C.tealSoft, stroke: C.teal, strokeWidth: 2 }),
    text('cover-p-label', 920, 209, 78, 45, texBlock`p(z\mid x)`, {
      fontSize: 19,
      fontFamily: SERIF,
      fontWeight: 700,
      align: 'center',
      valign: 'middle'
    }),
    text('cover-q-label', 1040, 263, 72, 45, texBlock`q_\lambda(z)`, {
      fontSize: 18,
      fontFamily: SERIF,
      fontWeight: 700,
      align: 'center',
      valign: 'middle'
    }),
    shape('cover-arrow', 'line', 922, 400, 235, 2, { fill: C.rust, lineEnd: 'arrow', fx: { loop: { type: 'dash-march' } } }),
    text('cover-objective', 872, 427, 306, 84, texBlock`q_\star=\underset{q\in\mathcal Q}{\operatorname{arg\,min}}\;D_{\mathrm{KL}}\!\left(q(z)\,\Vert\,p(z\mid x)\right)`, {
      fontSize: 20,
      fontFamily: SERIF,
      fontWeight: 700,
      align: 'center',
      valign: 'middle'
    }),
    text('cover-objective-note', 875, 529, 300, 31, 'Choose a family. Optimize its closest member.', {
      fontSize: 12,
      color: C.muted,
      align: 'center'
    }),
    ...chrome('Overview', C.violet)
  );

  return {
    id: 'overview',
    background: C.paper,
    transition: 'none',
    notes: 'Open with the variational move: choose a tractable family and optimize the best member. The four cards link to the ELBO, mean-field inference, the EM introduction, and stochastic-gradient VI. Preview that the EM experiment will expose each coordinate update.',
    elements
  };
}

function problemSlide() {
  const elements = [
    ...heading('01 · The inference problem', 'The posterior is easy to name—and often hard to compute', 'Latent variables explain observed data, but the normalizing evidence couples every possible explanation.', C.blue),
    card('problem-equation-card', 72, 180, 1136, 122, C.blueSoft, C.blue, 16),
    text('problem-equation-label', 96, 201, 190, 20, 'BAYES’ RULE', {
      fontSize: 10,
      fontFamily: MONO,
      fontWeight: 900,
      color: C.blue,
      letterSpacing: 1
    }),
    text('problem-equation', 284, 200, 870, 82, texBlock`p(z\mid x)=\frac{p(x,z)}{p(x)},\qquad p(x)=\int p(x,z)\,dz`, {
      fontSize: 27,
      fontFamily: SERIF,
      fontWeight: 700,
      align: 'center',
      valign: 'middle'
    }),
    ...panel('problem-known', 72, 334, 354, 238, 'WHAT WE CAN USUALLY EVALUATE', mathParagraphs(
      tex`\log p(x,z)`,
      muted('The joint score for one proposed latent configuration.'),
      '<b>Local computation</b><br>often factorizes with the model.'
    ), { accent: C.teal, fill: C.tealSoft, stroke: C.teal, fontSize: 15.5, lineHeight: 1.42 }),
    ...panel('problem-hard', 463, 334, 354, 238, 'WHAT BLOCKS EXACT INFERENCE', mathParagraphs(
      tex`\log p(x)=\log\int p(x,z)\,dz`,
      muted('A sum or integral over all latent configurations.'),
      '<b>Global normalization</b><br>can grow exponentially or lose conjugacy.'
    ), { accent: C.rust, fill: C.rustSoft, stroke: C.rust, fontSize: 15.5, lineHeight: 1.42 }),
    ...panel('problem-move', 854, 334, 354, 238, 'THE VARIATIONAL SUBSTITUTE', mathParagraphs(
      tex`q_\lambda(z)\in\mathcal Q`,
      muted('A normalized density we can evaluate, sample, and optimize.'),
      '<b>Approximate globally</b><br>while exploiting local model structure.'
    ), { accent: C.violet, fill: C.violetSoft, stroke: C.violet, fontSize: 15.5, lineHeight: 1.42 }),
    card('problem-boundary', 72, 600, 1136, 42, C.panel, C.rule, 10),
    text('problem-boundary-copy', 92, 609, 1096, 25, `The approximation is defined jointly by the family ${tex`\mathcal Q`}, the divergence, and the optimization procedure.`, {
      fontSize: 13.5,
      color: C.muted,
      align: 'center',
      valign: 'middle'
    }),
    ...chrome('The inference problem', C.blue)
  ];
  return {
    id: 'problem',
    background: C.paper,
    transition: 'none',
    notes: 'Separate the easy-to-evaluate joint from the hard evidence integral. Stress that variational inference is not one approximation: the family, divergence, and optimizer all matter.',
    elements
  };
}

function elboSlide() {
  const elements = [
    ...heading('02 · Objective', 'The ELBO turns posterior matching into optimization', 'One identity gives both a computable lower bound and a measure of approximation error.', C.violet),
    card('elbo-main-card', 72, 180, 1136, 142, C.violetSoft, C.violet, 16),
    text('elbo-main-label', 96, 201, 230, 20, 'THE CENTRAL IDENTITY', {
      fontSize: 10,
      fontFamily: MONO,
      fontWeight: 900,
      color: C.violet,
      letterSpacing: 1
    }),
    text('elbo-main-formula', 96, 226, 1088, 80, texBlock`\log p(x)=\mathcal L(q)+D_{\mathrm{KL}}\!\left(q(z)\,\Vert\,p(z\mid x)\right)`, {
      fontSize: 30,
      fontFamily: SERIF,
      fontWeight: 700,
      align: 'center',
      valign: 'middle'
    }),
    ...panel('elbo-bound', 72, 354, 552, 206, 'READING A · LOWER-BOUND THE EVIDENCE', mathParagraphs(
      tex`\mathcal L(q)=\mathbb E_q[\log p(x,z)-\log q(z)]`,
      `${tex`D_{\mathrm{KL}}\ge 0`} implies ${tex`\mathcal L(q)\le\log p(x)`}.`,
      muted('Useful for learning parameters and comparing optimization progress.')
    ), { accent: C.teal, fill: C.tealSoft, stroke: C.teal, fontSize: 16, lineHeight: 1.46 }),
    ...panel('elbo-project', 656, 354, 552, 206, 'READING B · PROJECT ONTO A FAMILY', mathParagraphs(
      tex`q_\star=\underset{q\in\mathcal Q}{\operatorname{arg\,max}}\;\mathcal L(q)`,
      tex`\phantom{q_\star}=\underset{q\in\mathcal Q}{\operatorname{arg\,min}}\;D_{\mathrm{KL}}(q\Vert p)`,
      muted('The evidence is constant with respect to the variational density.')
    ), { accent: C.blue, fill: C.blueSoft, stroke: C.blue, fontSize: 16, lineHeight: 1.46 }),
    card('elbo-caution', 72, 588, 1136, 50, C.rustSoft, C.rust, 11),
    text('elbo-caution-copy', 94, 599, 1092, 27, '<b>A higher ELBO means a better fit within the chosen family.</b> It does not certify that the family represents every posterior feature.', {
      fontSize: 14,
      color: C.rust,
      align: 'center',
      valign: 'middle'
    }),
    ...chrome('The ELBO', C.violet)
  ];
  return {
    id: 'elbo',
    background: C.paper,
    transition: 'none',
    notes: 'Use the identity twice: as a lower bound on the evidence and as reverse-KL projection. Point out that a larger ELBO is only a relative certificate inside the selected family.',
    elements
  };
}

function equationSheetSlide({ id, eyebrow, title, context, panels, accent, soft, notes }) {
  const elements = [
    ...heading(`${eyebrow} · equation sheet`, title, context, accent, { titleSize: 35, subtitleSize: 13.5, subtitleHeight: 39 })
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
  elements.push(...chrome(`${eyebrow} · equations`, accent));
  return { id, background: C.paper, transition: 'none', notes, elements };
}

function elboEquationsSlide() {
  return equationSheetSlide({
    id: 'elbo-equations',
    eyebrow: 'The ELBO',
    title: 'Derive the bound without losing the gap',
    context: `Let ${tex`q(z)`} be normalized with support compatible with ${tex`p(x,z)`}, and assume the relevant expectations are finite. The observed ${tex`x`} is fixed.`,
    accent: C.violet,
    soft: C.violetSoft,
    panels: [
      { title: 'Insert the variational density', body: mathParagraphs(
        tex`\log p(x)=\log\int p(x,z)\,dz`,
        tex`\phantom{\log p(x)}=\log\int q(z)\frac{p(x,z)}{q(z)}\,dz`
      ) },
      { title: 'Apply Jensen’s inequality', body: mathParagraphs(
        tex`\log p(x)\ge\mathbb E_q\!\left[\log\frac{p(x,z)}{q(z)}\right]`,
        tex`\mathcal L(q)=\mathbb E_q[\log p(x,z)]+\mathsf H[q]`,
        muted(`where ${tex`\mathsf H[q]=-\mathbb E_q[\log q(z)]`}.`)
      ) },
      { title: 'Expose fit and regularization', body: mathParagraphs(
        tex`\mathcal L(q)=\mathbb E_q[\log p(x\mid z)]-D_{\mathrm{KL}}\!\left(q(z)\Vert p(z)\right)`,
        muted('Expected data fit is balanced against departure from the prior.')
      ), fontSize: 16 },
      { title: 'Recover the exact gap', body: mathParagraphs(
        tex`D_{\mathrm{KL}}\!\left(q(z)\Vert p(z\mid x)\right)=\log p(x)-\mathcal L(q)`,
        tex`\mathcal L(q)=\log p(x)\iff q(z)=p(z\mid x)\;\text{a.e.}`
      ), fontSize: 15.5 }
    ],
    notes: 'Derive the lower bound using Jensen, then recover the exact KL gap. Keep the support condition explicit: the ratio requires q to cover every region where the joint contributes.'
  });
}

function meanFieldSlide() {
  const elements = [
    ...heading('03 · Structured families', 'Mean field buys tractability with conditional separation', 'Factorize the approximation, then optimize one factor while averaging over the rest.', C.blue),
    card('mf-factor-card', 72, 180, 1136, 110, C.blueSoft, C.blue, 16),
    text('mf-factor-label', 96, 200, 190, 20, 'FAMILY', { fontSize: 10, fontFamily: MONO, fontWeight: 900, color: C.blue, letterSpacing: 1 }),
    text('mf-factor-formula', 266, 197, 918, 72, texBlock`q(z)=\prod_{i=1}^{m}q_i(z_i)`, { fontSize: 30, fontFamily: SERIF, fontWeight: 700, align: 'center', valign: 'middle' }),
    ...panel('mf-update', 72, 318, 552, 228, 'THE COORDINATE OPTIMUM', mathParagraphs(
      tex`\log q_i^\star(z_i)=\mathbb E_{q_{-i}}[\log p(x,z)]+\text{constant}`,
      muted(`Hold ${tex`q_{-i}=\prod_{j\ne i}q_j`} fixed, then normalize the resulting factor.`),
      '<b>Conjugate exponential-family models</b> often make this update analytic.'
    ), { accent: C.violet, fill: C.violetSoft, stroke: C.violet, fontSize: 16, lineHeight: 1.48 }),
    ...panel('mf-cost', 656, 318, 552, 228, 'THE STRUCTURAL COST', mathParagraphs(
      tex`q(z_a,z_b)=q_a(z_a)q_b(z_b)`,
      muted('The variational density cannot retain posterior dependence across the chosen partition.'),
      '<b>The model may still be strongly coupled.</b><br>The independence is an approximation, not a claim about the data-generating process.'
    ), { accent: C.rust, fill: C.rustSoft, stroke: C.rust, fontSize: 16, lineHeight: 1.48 }),
    card('mf-observe', 72, 572, 1136, 64, C.panel, C.rule, 12),
    text('mf-observe-copy', 94, 585, 1092, 39, `A useful partition follows computational structure: retain dependence inside blocks, factorize only where the expected log joint stays manageable.`, { fontSize: 14.5, color: C.muted, align: 'center', valign: 'middle' }),
    ...chrome('Mean-field inference', C.blue)
  ];
  return {
    id: 'mean-field',
    background: C.paper,
    transition: 'none',
    notes: 'Introduce mean field as a design decision about the variational family. Derive the coordinate form conceptually and emphasize that the independence belongs to q, not necessarily to the true posterior.',
    elements
  };
}

function caviSlide() {
  const elements = [
    ...heading('04 · Coordinate ascent', 'CAVI turns mean field into an algorithm', 'Cycle through normalized factor updates; exact coordinate optima cannot decrease the ELBO.', C.teal),
    card('cavi-flow-card', 72, 178, 1136, 204, C.panel, C.rule, 16)
  ];
  const steps = [
    ['1', 'INITIALIZE', tex`q_1,\ldots,q_m`, C.blue, C.blueSoft],
    ['2', 'EXPECT', tex`\mathbb E_{q_{-i}}[\log p(x,z)]`, C.violet, C.violetSoft],
    ['3', 'NORMALIZE', tex`q_i\leftarrow q_i^\star`, C.rust, C.rustSoft],
    ['4', 'CHECK', tex`\Delta\mathcal L`, C.teal, C.tealSoft]
  ];
  steps.forEach(([number, title, body, accent, soft], index) => {
    const x = 92 + index * 278;
    elements.push(card(`cavi-step-${index}`, x, 207, 246, 142, soft, accent, 12));
    elements.push(text(`cavi-step-number-${index}`, x + 14, 222, 30, 20, number, { fontSize: 10, fontFamily: MONO, fontWeight: 900, color: accent }));
    elements.push(text(`cavi-step-title-${index}`, x + 48, 219, 177, 22, title, { fontSize: 10, fontFamily: MONO, fontWeight: 900, color: accent, letterSpacing: 0.8 }));
    elements.push(text(`cavi-step-body-${index}`, x + 16, 263, 214, 58, body, { fontSize: 17, fontFamily: SERIF, fontWeight: 700, align: 'center', valign: 'middle' }));
    if (index < steps.length - 1) elements.push(shape(`cavi-arrow-${index}`, 'line', x + 246, 278, 31, 2, { fill: C.faint, lineEnd: 'arrow' }));
  });
  elements.push(
    ...panel('cavi-guarantee', 72, 414, 552, 174, 'WHAT MONOTONICITY DOES SAY', mathParagraphs(
      tex`\mathcal L(q^{(t+1)})\ge\mathcal L(q^{(t)})`,
      'Every exact coordinate update maximizes the objective over one factor.',
      muted('Bounded ascent converges in objective value.')
    ), { accent: C.teal, fill: C.tealSoft, stroke: C.teal, fontSize: 15.5, lineHeight: 1.43 }),
    ...panel('cavi-limit', 656, 414, 552, 174, 'WHAT MONOTONICITY DOES NOT SAY', 'The ELBO is usually non-convex jointly in all factors.<br><br>Different initializations can reach different stationary points; a stationary point need not be globally optimal.', { accent: C.rust, fill: C.rustSoft, stroke: C.rust, fontSize: 15.5, lineHeight: 1.46 }),
    card('cavi-practice', 72, 610, 1136, 30, C.goldSoft, C.gold, 9),
    text('cavi-practice-copy', 92, 614, 1096, 20, 'PRACTICE · cache sufficient statistics, update in a stable order, and compare multiple starts.', { fontSize: 10.5, fontFamily: MONO, fontWeight: 850, color: C.gold, align: 'center', valign: 'middle' }),
    ...chrome('Coordinate-ascent VI', C.teal)
  );
  return {
    id: 'cavi',
    background: C.paper,
    transition: 'none',
    notes: 'Walk left to right through a CAVI sweep. Distinguish monotone improvement from global optimality, and recommend cached sufficient statistics plus multiple initializations.',
    elements
  };
}

function klDirectionSlide() {
  const elements = [
    ...heading('05 · Divergence geometry', 'KL direction changes which mistakes are expensive', 'The usual ELBO minimizes reverse KL. Forward KL answers a different projection problem.', C.rose),
    ...panel('kl-reverse', 72, 182, 552, 302, 'REVERSE KL · STANDARD VI', mathParagraphs(
      tex`D_{\mathrm{KL}}(q\Vert p)=\mathbb E_q\!\left[\log\frac{q(z)}{p(z\mid x)}\right]`,
      '<b>Expensive:</b> placing variational mass where the posterior density is tiny.<br><br><b>Often observed with restricted unimodal families:</b> selecting one mode and under-representing tails.',
      muted('The exact behavior depends on the family and optimization landscape.')
    ), { accent: C.violet, fill: C.violetSoft, stroke: C.violet, fontSize: 15.5, lineHeight: 1.45 }),
    ...panel('kl-forward', 656, 182, 552, 302, 'FORWARD KL · MASS COVERING', mathParagraphs(
      tex`D_{\mathrm{KL}}(p\Vert q)=\mathbb E_p\!\left[\log\frac{p(z\mid x)}{q(z)}\right]`,
      '<b>Expensive:</b> assigning too little variational density where the posterior has mass.<br><br><b>Often observed with restricted unimodal families:</b> broadening to cover separated modes.',
      muted('Its expectation under the unknown posterior is usually not directly tractable.')
    ), { accent: C.blue, fill: C.blueSoft, stroke: C.blue, fontSize: 15.5, lineHeight: 1.45 }),
    card('kl-zero-card', 72, 512, 1136, 126, C.panel, C.rule, 14),
    text('kl-zero-label', 96, 534, 230, 20, 'ZERO-FORCING INTUITION', { fontSize: 10, fontFamily: MONO, fontWeight: 900, color: C.rose, letterSpacing: 1 }),
    text('kl-zero-copy', 96, 567, 1088, 48, `Reverse KL heavily penalizes ${tex`q(z)>0`} where ${tex`p(z\mid x)=0`}. Forward KL heavily penalizes ${tex`q(z)=0`} where ${tex`p(z\mid x)>0`}.`, { fontSize: 16.5, fontFamily: SERIF, fontWeight: 650, align: 'center', valign: 'middle' }),
    ...chrome('KL direction', C.rose)
  ];
  return {
    id: 'kl-direction',
    background: C.paper,
    transition: 'none',
    notes: 'Avoid presenting mode-seeking and mass-covering as universal laws. Tie each intuition to a restricted family, and state why the forward-KL expectation is usually unavailable.',
    elements
  };
}

function emIntroSlide() {
  const elements = [
    ...heading('06 · EM · experiment introduction', 'EM is coordinate ascent on a bound', 'Use a free density for the latent variables, then alternate exact inference and parameter optimization.', C.rust),
    card('em-objective-card', 72, 180, 1136, 116, C.rustSoft, C.rust, 16),
    text('em-objective-label', 96, 201, 220, 20, 'FREE-ENERGY OBJECTIVE', { fontSize: 10, fontFamily: MONO, fontWeight: 900, color: C.rust, letterSpacing: 1 }),
    text('em-objective', 252, 203, 932, 76, texBlock`\mathcal F(q,\theta)=\mathbb E_q[\log p(x,z\mid\theta)]+\mathsf H[q]\le\log p(x\mid\theta)`, { fontSize: 25, fontFamily: SERIF, fontWeight: 700, align: 'center', valign: 'middle' }),
    ...panel('em-estep', 72, 326, 552, 190, 'E-STEP · CLOSE THE GAP', mathParagraphs(
      tex`q^{(t+1)}(z)=p(z\mid x,\theta^{(t)})`,
      'Infer latent responsibilities with parameters fixed.',
      muted(`The bound becomes tight at ${tex`\theta^{(t)}`}.`)
    ), { accent: C.blue, fill: C.blueSoft, stroke: C.blue, fontSize: 16, lineHeight: 1.46 }),
    ...panel('em-mstep', 656, 326, 552, 190, 'M-STEP · RAISE THE BOUND', mathParagraphs(
      tex`\theta^{(t+1)}=\underset{\theta}{\operatorname{arg\,max}}\;\mathcal F(q^{(t+1)},\theta)`,
      'Refit parameters to the expected complete-data statistics.',
      muted('The observed-data likelihood cannot decrease.')
    ), { accent: C.teal, fill: C.tealSoft, stroke: C.teal, fontSize: 16, lineHeight: 1.46 }),
    card('em-watch-card', 72, 544, 1136, 94, C.panel, C.rust, 13),
    text('em-watch-label', 94, 565, 160, 20, 'WATCH NEXT', { fontSize: 10, fontFamily: MONO, fontWeight: 900, color: C.rust, letterSpacing: 1 }),
    text('em-watch-copy', 250, 557, 932, 58, '<b>Advance one coordinate at a time.</b> In the E-step, point colors become responsibilities. In the M-step, means and mixture weights move. The log likelihood rises after each complete cycle.', { fontSize: 14.5, lineHeight: 1.44, valign: 'middle' }),
    ...chrome('EM · concept', C.rust)
  ];
  return {
    id: 'em',
    background: C.paper,
    transition: 'none',
    notes: 'Introduce EM as coordinate ascent on a variational free-energy bound. The next slide is the live Gaussian-mixture experiment. Ask the audience to predict whether ambiguous points will move the means or the weights more strongly.',
    elements
  };
}

const LIVE_BOUNDS = { x: 74, y: 154, width: 1132, height: 494 };

function emFallback() {
  const { x, y, width, height } = LIVE_BOUNDS;
  const dataXs = [370, 390, 411, 426, 447, 466, 486, 504, 523, 542, 562, 582, 602, 622, 642, 662, 682, 704, 728, 754, 778, 805];
  const elements = [
    card('em-fallback-region', x, y, width, height, '#FBFAF6', C.rule, 14),
    card('em-fallback-controls', x + 16, y + 16, 270, height - 32, C.rustSoft, C.rust, 12),
    text('em-fallback-controls-label', x + 34, y + 34, 234, 22, 'DETERMINISTIC DEFAULT', { fontSize: 10, fontFamily: MONO, fontWeight: 900, color: C.rust, letterSpacing: 1 }),
    text('em-fallback-controls-values', x + 34, y + 78, 234, 132, mathParagraphs(
      mathLines(tex`N=48`, tex`\sigma=0.72`),
      mathLines(tex`\Delta\mu=3.2`, tex`c_0=0.45`)
    ), { fontSize: 15.5, fontFamily: MONO, fontWeight: 700, lineHeight: 1.55 }),
    card('em-fallback-next', x + 34, y + 232, 234, 68, C.panel, C.rust, 10),
    text('em-fallback-next-label', x + 50, y + 244, 202, 18, 'NEXT COORDINATE', { fontSize: 9, fontFamily: MONO, fontWeight: 900, color: C.rust, align: 'center' }),
    text('em-fallback-next-value', x + 50, y + 269, 202, 22, 'E-STEP', { fontSize: 17, fontFamily: SERIF, fontWeight: 700, color: C.rust, align: 'center' }),
    text('em-fallback-controls-hint', x + 34, y + 354, 234, 72, 'Live controls replace this region only while the Bento slide is active.', { fontSize: 13, color: C.muted, lineHeight: 1.45 }),
    card('em-fallback-plot', x + 304, y + 16, 594, height - 32, C.panel, C.rule, 12),
    text('em-fallback-plot-label', x + 326, y + 34, 550, 20, 'TWO-COMPONENT GAUSSIAN MIXTURE', { fontSize: 10, fontFamily: MONO, fontWeight: 900, color: C.violet, letterSpacing: 1 }),
    shape('em-fallback-axis', 'rect', x + 344, y + 306, 514, 2, { fill: C.rule }),
    shape('em-fallback-density-a', 'ellipse', x + 462, y + 126, 220, 180, { fill: 'rgba(57,112,140,.10)', stroke: C.blue, strokeWidth: 3 }),
    shape('em-fallback-density-b', 'ellipse', x + 578, y + 110, 220, 196, { fill: 'rgba(169,87,54,.10)', stroke: C.rust, strokeWidth: 3 }),
    shape('em-fallback-mean-a', 'rect', x + 572, y + 94, 3, 215, { fill: C.blue }),
    shape('em-fallback-mean-b', 'rect', x + 688, y + 94, 3, 215, { fill: C.rust }),
    text('em-fallback-mean-a-label', x + 508, y + 75, 130, 22, tex`\mu_1=-0.45`, { fontSize: 11, fontFamily: MONO, fontWeight: 850, color: C.blue, align: 'center' }),
    text('em-fallback-mean-b-label', x + 624, y + 75, 130, 22, tex`\mu_2=1.35`, { fontSize: 11, fontFamily: MONO, fontWeight: 850, color: C.rust, align: 'center' })
  ];
  dataXs.forEach((cx, index) => {
    elements.push(shape(`em-fallback-point-${index}`, 'ellipse', x + cx, y + 316 + (index % 3) * 18, 10, 10, {
      fill: C.violet,
      stroke: C.panel,
      strokeWidth: 1
    }));
  });
  elements.push(
    text('em-fallback-plot-caption', x + 326, y + 398, 550, 34, 'Uniform point color shows the pre–E-step state; component profiles move only during the M-step.', { fontSize: 11.5, color: C.muted, align: 'center' }),
    card('em-fallback-metrics', x + 916, y + 16, width - 932, height - 32, C.violetSoft, C.violet, 12),
    text('em-fallback-metrics-label', x + 934, y + 34, 164, 20, 'INITIAL STATE', { fontSize: 9.5, fontFamily: MONO, fontWeight: 900, color: C.violet, align: 'center' }),
    text('em-fallback-metrics-values', x + 932, y + 78, 168, 154, mathParagraphs(
      tex`t=0`,
      tex`\pi_1=0.50`,
      tex`\pi_2=0.50`,
      tex`\mathcal L=-115.76`
    ), { fontSize: 15, fontFamily: MONO, fontWeight: 700, lineHeight: 1.52, align: 'center' }),
    card('em-fallback-phase', x + 934, y + 264, 164, 102, C.panel, C.teal, 10),
    text('em-fallback-phase-label', x + 948, y + 278, 136, 17, 'BOUND GAP', { fontSize: 9, fontFamily: MONO, fontWeight: 900, color: C.teal, align: 'center' }),
    text('em-fallback-phase-value', x + 948, y + 307, 136, 38, texBlock`\log p(x)-\mathcal F\approx101.14`, { fontSize: 13, fontFamily: SERIF, fontWeight: 700, align: 'center' }),
    text('em-fallback-status', x + 326, y + 438, 772, 18, 'STATIC FALLBACK · advance to reveal exact responsibilities, then refit the parameters', { fontSize: 9.5, fontFamily: MONO, fontWeight: 850, color: C.muted, align: 'center' })
  );
  return elements;
}

function emLiveSlide() {
  return {
    id: 'em-live',
    background: C.paper,
    transition: 'none',
    notes: 'The EM experiment mounts automatically. Use E-step to update responsibilities, then M-step to move means and weights. Increase overlap and reset to show slower convergence. Use Run to convergence to confirm monotone log likelihood. Escape returns focus to Bento; Page Up returns to the EM concept slide.',
    elements: [
      text('live-eyebrow', 74, 37, 870, 21, 'EM · LIVE EXPERIMENT', { fontSize: 10, fontFamily: MONO, fontWeight: 900, color: C.rust, letterSpacing: 1.45 }),
      text('live-title', 74, 66, 1050, 48, 'Alternate inference and learning—one coordinate at a time.', { fontSize: 34, fontFamily: SERIF, fontWeight: 700, lineHeight: 1.05 }),
      text('live-prompt', 74, 116, 1080, 28, 'Use the E-step to recolor responsibility, then the M-step to move parameters. Increase overlap and compare convergence.', { fontSize: 15, color: C.muted }),
      ...emFallback(),
      shape('live-demo-mount', 'rect', LIVE_BOUNDS.x, LIVE_BOUNDS.y, LIVE_BOUNDS.width, LIVE_BOUNDS.height, { fill: 'rgba(255,255,255,0)', stroke: 'rgba(255,255,255,0)', strokeWidth: 0, opacity: 0 }),
      ...chrome('EM · live', C.rust)
    ]
  };
}

function emEquationsSlide() {
  return equationSheetSlide({
    id: 'em-equations',
    eyebrow: 'EM for a Gaussian mixture',
    title: 'The live steps in equations',
    context: `Two components with fixed shared ${tex`\sigma>0`}: ${tex`p(x_n\mid\theta)=\sum_{k=1}^{2}\pi_k\mathcal N(x_n\mid\mu_k,\sigma^2)`}, ${tex`\pi_k>0`}, ${tex`\sum_k\pi_k=1`}.`,
    accent: C.rust,
    soft: C.rustSoft,
    panels: [
      { title: 'Latent assignments', body: mathParagraphs(
        tex`z_n\in\{1,2\}`,
        tex`p(x,z\mid\theta)=\prod_{n=1}^{N}\prod_{k=1}^{2}\left[\pi_k\mathcal N(x_n\mid\mu_k,\sigma^2)\right]^{\mathbb I[z_n=k]}`
      ), fontSize: 15 },
      { title: 'E-step responsibilities', body: mathParagraphs(
        tex`r_{nk}=q(z_n=k)`,
        tex`r_{nk}=\frac{\pi_k\mathcal N(x_n\mid\mu_k,\sigma^2)}{\sum_{j=1}^{2}\pi_j\mathcal N(x_n\mid\mu_j,\sigma^2)}`
      ), fontSize: 15.5 },
      { title: 'M-step sufficient statistics', body: mathParagraphs(
        tex`N_k=\sum_{n=1}^{N}r_{nk}`,
        mathLines(tex`\pi_k^{\mathrm{new}}=\frac{N_k}{N}`, tex`\mu_k^{\mathrm{new}}=\frac{1}{N_k}\sum_{n=1}^{N}r_{nk}x_n`)
      ), fontSize: 16 },
      { title: 'Monotone observed likelihood', body: mathParagraphs(
        tex`\ell(\theta)=\sum_{n=1}^{N}\log\sum_{k=1}^{2}\pi_k\mathcal N(x_n\mid\mu_k,\sigma^2)`,
        tex`\ell(\theta^{(t+1)})\ge\ell(\theta^{(t)})`,
        muted('Monotone ascent can still converge to a local optimum or saddle point.')
      ), fontSize: 14.8 }
    ],
    notes: 'Tie each equation to the preceding interaction: point colors are r_nk, weighted counts are N_k, and the mean markers are weighted averages. The shared variance is fixed in the demo to prevent variance collapse and isolate the alternating logic.'
  });
}

function stochasticSlide() {
  const elements = [
    ...heading('07 · Gradient estimators', 'Differentiate the ELBO when closed forms disappear', 'Monte Carlo estimates make the ELBO scalable—but gradient variance becomes part of the algorithm.', C.teal),
    card('stochastic-objective-card', 72, 180, 1136, 112, C.tealSoft, C.teal, 16),
    text('stochastic-objective-label', 96, 200, 220, 20, 'OPTIMIZE PARAMETERS', { fontSize: 10, fontFamily: MONO, fontWeight: 900, color: C.teal, letterSpacing: 1 }),
    text('stochastic-objective', 268, 201, 916, 72, texBlock`\lambda_\star=\underset{\lambda}{\operatorname{arg\,max}}\;\mathbb E_{q_\lambda(z)}[\log p(x,z)-\log q_\lambda(z)]`, { fontSize: 24, fontFamily: SERIF, fontWeight: 700, align: 'center', valign: 'middle' }),
    ...panel('stochastic-pathwise', 72, 322, 552, 230, 'PATHWISE / REPARAMETERIZATION GRADIENT', mathParagraphs(
      tex`\varepsilon\sim p(\varepsilon),\qquad z=g(\varepsilon,\lambda)`,
      tex`\nabla_\lambda\mathcal L=\mathbb E_\varepsilon\!\left[\nabla_\lambda\bigl(\log p(x,g)-\log q_\lambda(g)\bigr)\right]`,
      muted('Usually low variance for differentiable continuous latent variables.')
    ), { accent: C.teal, fill: C.tealSoft, stroke: C.teal, fontSize: 15.5, lineHeight: 1.44 }),
    ...panel('stochastic-score', 656, 322, 552, 230, 'SCORE-FUNCTION GRADIENT', mathParagraphs(
      tex`\nabla_\lambda\mathbb E_{q_\lambda}[f(z)]=\mathbb E_{q_\lambda}\!\left[f(z)\nabla_\lambda\log q_\lambda(z)\right]`,
      `For ${tex`f`} independent of ${tex`\lambda`}; the ELBO form follows after accounting for its explicit ${tex`\lambda`}-dependence.<br><br>Applies to discrete or non-reparameterizable samples.`,
      muted('Often needs baselines, Rao–Blackwellization, or other control variates.')
    ), { accent: C.violet, fill: C.violetSoft, stroke: C.violet, fontSize: 14.5, lineHeight: 1.38 }),
    card('stochastic-scale', 72, 580, 1136, 58, C.panel, C.rule, 11),
    text('stochastic-scale-copy', 94, 591, 1092, 34, `For independent data, minibatch the likelihood terms and scale them by ${tex`N/B`}; keep the prior and entropy terms at their correct global weight.`, { fontSize: 14.5, color: C.muted, align: 'center', valign: 'middle' }),
    ...chrome('Stochastic VI', C.teal)
  ];
  return {
    id: 'stochastic',
    background: C.paper,
    transition: 'none',
    notes: 'Contrast pathwise and score-function gradients. Explain that stochastic VI shifts difficulty from closed-form expectations to gradient variance, minibatch scaling, and optimization stability.',
    elements
  };
}

function diagnosticsSlide() {
  const elements = [
    ...heading('08 · Diagnostics', 'A converged ELBO is not a calibrated posterior', 'Separate optimization failure, approximation failure, and model failure before trusting the result.', C.gold)
  ];
  const checks = [
    ['OPTIMIZATION', 'Did the solver find a good point?', 'ELBO traces · gradient norms · multiple starts', C.violet, C.violetSoft],
    ['APPROXIMATION', 'Can the family express the posterior?', 'importance-weighted checks · richer families · simulation', C.blue, C.blueSoft],
    ['MODEL', 'Does the posterior predict useful data?', 'posterior predictive checks · held-out prediction', C.rust, C.rustSoft]
  ];
  checks.forEach(([label, question, tools, accent, soft], index) => {
    const x = 72 + index * 391;
    elements.push(card(`diag-${index}`, x, 184, 354, 292, soft, accent, 15));
    elements.push(text(`diag-${index}-label`, x + 20, 204, 314, 22, label, { fontSize: 10, fontFamily: MONO, fontWeight: 900, color: accent, letterSpacing: 0.9, align: 'center' }));
    elements.push(text(`diag-${index}-question`, x + 24, 250, 306, 70, question, { fontSize: 22, fontFamily: SERIF, fontWeight: 700, align: 'center', lineHeight: 1.2 }));
    elements.push(shape(`diag-${index}-rule`, 'rect', x + 44, 340, 266, 1, { fill: accent }));
    elements.push(text(`diag-${index}-tools`, x + 30, 369, 294, 76, tools, { fontSize: 14, color: C.muted, align: 'center', lineHeight: 1.48 }));
  });
  elements.push(
    card('diag-warning', 72, 506, 1136, 130, C.panel, C.gold, 14),
    text('diag-warning-label', 96, 528, 212, 20, 'COMMON FALSE COMFORT', { fontSize: 10, fontFamily: MONO, fontWeight: 900, color: C.gold, letterSpacing: 1 }),
    text('diag-warning-copy', 94, 558, 1092, 55, `A smooth, increasing ${tex`\mathcal L`} only shows that the chosen optimizer improved the chosen bound. It does not reveal missing modes, underestimated covariance, or a misspecified likelihood.`, { fontSize: 17, fontFamily: SERIF, fontWeight: 650, align: 'center', valign: 'middle', lineHeight: 1.38 }),
    ...chrome('Diagnostics', C.gold)
  );
  return {
    id: 'diagnostics',
    background: C.paper,
    transition: 'none',
    notes: 'Use the three columns as a debugging order: optimizer, variational family, then model. A rising ELBO is necessary evidence about optimization progress but insufficient evidence about calibration.',
    elements
  };
}

function connectionsSlide() {
  const rows = [
    ['EM', 'latent posterior', 'exact local posterior', 'closed-form or numerical maximization', C.rust, C.rustSoft],
    ['CAVI', 'factor densities', 'mean-field or structured', 'analytic coordinate updates', C.blue, C.blueSoft],
    ['Amortized VI', 'inference-network weights', 'shared across observations', 'stochastic gradients', C.teal, C.tealSoft],
    ['Laplace', 'mode and curvature', 'local Gaussian', 'optimization plus Hessian', C.violet, C.violetSoft],
    ['MCMC', 'samples', 'asymptotically exact target', 'Markov transitions', C.gold, C.goldSoft]
  ];
  const elements = [
    ...heading('09 · Connections', 'Choose the computational object for the question', 'These methods may share algebra, but they optimize or simulate different objects.', C.violet),
    card('connections-table', 72, 180, 1136, 430, C.panel, C.rule, 16),
    text('connections-h-method', 92, 202, 168, 20, 'METHOD', { fontSize: 9.5, fontFamily: MONO, fontWeight: 900, color: C.faint }),
    text('connections-h-object', 284, 202, 224, 20, 'COMPUTATIONAL OBJECT', { fontSize: 9.5, fontFamily: MONO, fontWeight: 900, color: C.faint }),
    text('connections-h-family', 536, 202, 252, 20, 'APPROXIMATION / TARGET', { fontSize: 9.5, fontFamily: MONO, fontWeight: 900, color: C.faint }),
    text('connections-h-update', 822, 202, 344, 20, 'PRIMARY UPDATE', { fontSize: 9.5, fontFamily: MONO, fontWeight: 900, color: C.faint })
  ];
  rows.forEach(([method, object, family, update, accent, soft], index) => {
    const y = 238 + index * 70;
    elements.push(card(`connections-row-${index}`, 88, y, 1104, 56, soft, accent, 9));
    elements.push(text(`connections-method-${index}`, 106, y + 15, 150, 26, method, { fontSize: 15.5, fontFamily: SERIF, fontWeight: 700, color: accent, valign: 'middle' }));
    elements.push(text(`connections-object-${index}`, 284, y + 14, 224, 28, object, { fontSize: 13.5, fontWeight: 650, valign: 'middle' }));
    elements.push(text(`connections-family-${index}`, 536, y + 14, 252, 28, family, { fontSize: 13.5, color: C.muted, valign: 'middle' }));
    elements.push(text(`connections-update-${index}`, 822, y + 14, 344, 28, update, { fontSize: 13.5, color: C.muted, valign: 'middle' }));
  });
  elements.push(
    card('connections-note', 72, 626, 1136, 34, C.violetSoft, C.violet, 9),
    text('connections-note-copy', 92, 631, 1096, 23, 'EM is a variational algorithm; MCMC is not. Laplace is local approximation; amortization changes how variational parameters are produced.', { fontSize: 10.5, fontFamily: MONO, fontWeight: 800, color: C.violet, align: 'center', valign: 'middle' }),
    ...chrome('Connections', C.violet)
  );
  return {
    id: 'connections',
    background: C.paper,
    transition: 'none',
    notes: 'Use the table to prevent method names from collapsing into one category. EM and CAVI optimize bounds; amortized VI shares an inference map; Laplace is a local Gaussian construction; MCMC targets samples.',
    elements
  };
}

function summarySlide() {
  const elements = [
    ...heading('10 · Synthesis', 'Variational inference is a sequence of choices', 'Make each choice visible; then the approximation becomes inspectable rather than magical.', C.teal)
  ];
  const choices = [
    ['1', 'TARGET', tex`p(z\mid x)`, 'Which posterior or marginal?', C.blue, C.blueSoft],
    ['2', 'FAMILY', tex`q_\lambda\in\mathcal Q`, 'Which dependence can survive?', C.violet, C.violetSoft],
    ['3', 'OBJECTIVE', tex`\mathcal L(q)`, 'Which mismatch is penalized?', C.rust, C.rustSoft],
    ['4', 'OPTIMIZER', tex`\lambda\leftarrow\lambda+\Delta`, 'Which stationary point is reached?', C.teal, C.tealSoft]
  ];
  choices.forEach(([number, label, formula, question, accent, soft], index) => {
    const x = 72 + index * 284;
    elements.push(card(`summary-choice-${index}`, x, 184, 264, 244, soft, accent, 15));
    elements.push(text(`summary-number-${index}`, x + 20, 204, 34, 22, number, { fontSize: 11, fontFamily: MONO, fontWeight: 900, color: accent }));
    elements.push(text(`summary-label-${index}`, x + 58, 202, 182, 24, label, { fontSize: 10, fontFamily: MONO, fontWeight: 900, color: accent, letterSpacing: 0.8 }));
    elements.push(text(`summary-formula-${index}`, x + 20, 252, 224, 66, formula, { fontSize: 19, fontFamily: SERIF, fontWeight: 700, align: 'center', valign: 'middle' }));
    elements.push(shape(`summary-rule-${index}`, 'rect', x + 34, 338, 196, 1, { fill: accent }));
    elements.push(text(`summary-question-${index}`, x + 26, 360, 212, 48, question, { fontSize: 13.5, color: C.muted, align: 'center', lineHeight: 1.35 }));
    if (index < choices.length - 1) elements.push(shape(`summary-arrow-${index}`, 'line', x + 264, 306, 20, 2, { fill: C.faint, lineEnd: 'arrow' }));
  });
  elements.push(
    card('summary-three-card', 72, 462, 1136, 174, C.panel, C.rule, 14),
    text('summary-three-label', 96, 483, 190, 20, 'THREE LAST CHECKS', { fontSize: 10, fontFamily: MONO, fontWeight: 900, color: C.teal, letterSpacing: 1 }),
    text('summary-three-a', 96, 524, 330, 72, '<b>Bound</b><br><span style="color:#66737D">Did the optimization improve?</span>', { fontSize: 17, fontFamily: SERIF, lineHeight: 1.5, align: 'center' }),
    text('summary-three-b', 474, 524, 330, 72, '<b>Family</b><br><span style="color:#66737D">What posterior structure is impossible?</span>', { fontSize: 17, fontFamily: SERIF, lineHeight: 1.5, align: 'center' }),
    text('summary-three-c', 852, 524, 330, 72, '<b>Prediction</b><br><span style="color:#66737D">Does the model explain held-out data?</span>', { fontSize: 17, fontFamily: SERIF, lineHeight: 1.5, align: 'center' }),
    ...chrome('Synthesis', C.teal)
  );
  return {
    id: 'summary',
    background: C.paper,
    transition: 'none',
    notes: 'Close with the four explicit choices: target, family, objective, optimizer. The final three checks separate numerical convergence, expressive adequacy, and predictive usefulness.',
    elements
  };
}

function referencesSlide() {
  const refs = [
    ['[1]', 'Jordan et al. (1999)', 'An Introduction to Variational Methods for Graphical Models.', 'https://doi.org/10.1023/A:1007665907178'],
    ['[2]', 'Bishop (2006)', 'Pattern Recognition and Machine Learning · Chapters 9–10.', 'https://link.springer.com/book/9780387310732'],
    ['[3]', 'Dempster, Laird & Rubin (1977)', 'Maximum Likelihood from Incomplete Data via the EM Algorithm.', 'https://doi.org/10.1111/j.2517-6161.1977.tb01600.x'],
    ['[4]', 'Neal & Hinton (1998)', 'A View of the EM Algorithm that Justifies Incremental, Sparse, and Other Variants.', 'https://doi.org/10.1007/978-94-011-5014-9_12'],
    ['[5]', 'Hoffman et al. (2013)', 'Stochastic Variational Inference.', 'https://jmlr.org/papers/v14/hoffman13a.html'],
    ['[6]', 'Kingma & Welling (2014)', 'Auto-Encoding Variational Bayes.', 'https://arxiv.org/abs/1312.6114'],
    ['[7]', 'Blei, Kucukelbir & McAuliffe (2017)', 'Variational Inference: A Review for Statisticians.', 'https://doi.org/10.1080/01621459.2017.1285773'],
    ['[8]', 'Yao et al. (2018)', 'Yes, but Did It Work? Evaluating Variational Inference.', 'https://proceedings.mlr.press/v80/yao18a.html']
  ];
  const elements = [
    ...heading('Sources', 'Foundations and further reading', 'Primary sources for the ELBO, EM, stochastic inference, reparameterization, and diagnostics.', C.violet)
  ];
  refs.forEach(([number, author, title, link], index) => {
    const col = index % 2;
    const row = Math.floor(index / 2);
    const x = 72 + col * 576;
    const y = 182 + row * 100;
    elements.push(card(`ref-card-${index}`, x, y, 552, 84, index % 4 < 2 ? C.panel : C.violetSoft, C.rule, 10));
    elements.push(text(`ref-number-${index}`, x + 14, y + 14, 44, 18, number, { fontSize: 9.5, fontFamily: MONO, fontWeight: 900, color: C.rust, link }));
    elements.push(text(`ref-author-${index}`, x + 62, y + 11, 465, 22, author, { fontSize: 13.5, fontFamily: SERIF, fontWeight: 700, color: C.ink, link }));
    elements.push(text(`ref-title-${index}`, x + 62, y + 38, 465, 34, title, { fontSize: 10.5, color: C.muted, lineHeight: 1.28, link }));
  });
  elements.push(
    card('ref-note-card', 72, 606, 1136, 54, C.tealSoft, C.teal, 10),
    text('ref-note', 92, 615, 1096, 36, 'The EM demo uses a deterministic two-component Gaussian mixture with fixed shared variance; it is pedagogical, not an empirical benchmark.', { fontSize: 11, fontFamily: MONO, fontWeight: 800, color: C.teal, align: 'center', valign: 'middle' }),
    ...chrome('References', C.violet)
  );
  return {
    id: 'references',
    background: C.paper,
    transition: 'none',
    notes: 'Point to the primary EM paper, the variational-methods tutorial, the modern review, and the diagnostics reference. Restate that the live mixture is a deterministic teaching model.',
    elements
  };
}

const slides = [
  overviewSlide(),
  problemSlide(),
  elboSlide(),
  elboEquationsSlide(),
  meanFieldSlide(),
  caviSlide(),
  klDirectionSlide(),
  emIntroSlide(),
  emLiveSlide(),
  emEquationsSlide(),
  stochasticSlide(),
  diagnosticsSlide(),
  connectionsSlide(),
  summarySlide(),
  referencesSlide()
];

export const deck = {
  format: 'bento/slides',
  version: 1,
  docId: 'variational-inference-bento',
  title: 'Variational Inference: Approximation by Optimization',
  readonly: true,
  meta: {
    author: 'Bai Liping',
    subject: 'Variational inference, ELBOs, EM, and stochastic optimization',
    company: 'bailiping.com'
  },
  size: { width: WIDTH, height: HEIGHT },
  theme: { background: C.paper, color: C.ink, accent: C.violet, fontFamily: SANS },
  slides
};

function indexOf(id) {
  const index = slides.findIndex(slide => slide.id === id);
  if (index < 0) throw new Error(`Unknown slide ${id}`);
  return index;
}

export const inlineLiveMap = [
  {
    introSlide: 'em',
    slide: 'em-live',
    slideIndex: indexOf('em-live'),
    inline: true,
    layout: 'region',
    bounds: LIVE_BOUNDS,
    src: './live/?demo=em&embed=region',
    source: './live/?demo=em',
    title: 'Interactive expectation-maximization experiment',
    sandbox: 'allow-scripts',
    hideSource: true,
    readyMessage: true,
    unloadWhenHidden: true
  }
];
