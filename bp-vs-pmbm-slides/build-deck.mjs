import fs from "node:fs";

const SERIF = "Georgia, 'Times New Roman', serif";
const SANS = "Arial, Helvetica, sans-serif";
const PAPER = "#F4F6F8";
const WHITE = "#FFFFFF";
const INK = "#16222E";
const SOFT = "#51606E";
const FAINT = "#8A97A3";
const LINE = "#D7DEE5";
const BP = "#1F77B4";
const BP_DEEP = "#155D8F";
const BP_WASH = "#E5EFF7";
const PM = "#E8720C";
const PM_DEEP = "#B45607";
const PM_WASH = "#FCEEDE";
const TRACKS = ["#2CA02C", "#9467BD", "#D62728"];

function text(id, x, y, w, h, html, fontSize = 24, color = INK, opts = {}) {
  return {
    id, type: "text", x, y, w, h, rotation: 0, opacity: 1, html,
    fontSize,
    fontFamily: opts.fontFamily || SERIF,
    fontWeight: opts.fontWeight === undefined ? 400 : opts.fontWeight,
    color,
    align: opts.align || "left",
    valign: opts.valign || "top",
    lineHeight: opts.lineHeight || 1.25,
    ...(opts.letterSpacing === undefined ? {} : { letterSpacing: opts.letterSpacing }),
    ...(opts.link ? { link: opts.link } : {}),
    ...(opts.fx ? { fx: opts.fx } : {})
  };
}

function rect(id, x, y, w, h, fill, opts = {}) {
  return {
    id, type: "shape", shape: opts.shape || "rect", x, y, w, h,
    fill, stroke: opts.stroke || "none", strokeWidth: opts.strokeWidth || 0,
    radius: opts.radius || 0, rotation: opts.rotation || 0,
    opacity: opts.opacity === undefined ? 1 : opts.opacity,
    ...(opts.link ? { link: opts.link } : {}),
    ...(opts.fx ? { fx: opts.fx } : {})
  };
}

function rule(id, x, y, w, color = LINE, height = 2, opts = {}) {
  return rect(id, x, y, w, height, color, opts);
}

function circle(id, x, y, size, fill, opts = {}) {
  return rect(id, x, y, size, size, fill, { ...opts, shape: "ellipse", radius: size / 2 });
}

function footer(section) {
  return [
    text("footer-l", 72, 683, 650, 20, "BP × PMBM · " + section + " · Bai Liping", 12, FAINT),
    text("footer-r", 1090, 683, 118, 20, "{{page}} / {{pages}}", 12, FAINT, { align: "right" })
  ];
}

function regular(id, section, titleHtml, subtitle, notes, elements, opts = {}) {
  const chrome = [
    text("deck-kicker", 72, 28, 760, 22, section.toUpperCase(), 13, opts.sectionColor || BP_DEEP, {
      fontWeight: 700, letterSpacing: 2.2
    }),
    text("deck-short-title", 930, 30, 278, 20, "ONE SCAN · TWO VIEWS", 11, FAINT, {
      fontFamily: SANS, fontWeight: 700, align: "right", letterSpacing: 1.2
    }),
    rule("deck-rule", 72, 66, 1136, LINE, 1),
    text("slide-title", 72, 92, 1136, 68, titleHtml, opts.titleSize || 42, INK, {
      fontWeight: 700, lineHeight: 1.08, fx: opts.titleFx || { enter: "fade-up", order: 0 }
    }),
    ...(subtitle ? [text("slide-subtitle", 72, 161, 1060, 48, subtitle, 19, SOFT, {
      lineHeight: 1.35, fx: { enter: "fade-up", order: 1 }
    })] : []),
    ...footer(section)
  ];
  return {
    id,
    background: opts.background || PAPER,
    transition: opts.transition || "morph",
    notes,
    elements: chrome.concat(elements)
  };
}

const INLINE_BOUNDS = { x: 72, y: 220, width: 1136, height: 440 };

function inlineMount() {
  return rect(
    "live-demo-mount",
    INLINE_BOUNDS.x,
    INLINE_BOUNDS.y,
    INLINE_BOUNDS.width,
    INLINE_BOUNDS.height,
    "rgba(255,255,255,0)",
    { opacity: 0 }
  );
}

function matrixWeight(track, measurement, covariance, pd = 0.9, clutter = 5e-5) {
  const dx = measurement.x - track.x;
  const dy = measurement.y - track.y;
  const det = covariance[0][0] * covariance[1][1] - covariance[0][1] * covariance[1][0];
  const q = (dx * (covariance[1][1] * dx - covariance[0][1] * dy) +
    dy * (-covariance[1][0] * dx + covariance[0][0] * dy)) / det;
  if (q > 9.21) return 0;
  return pd * Math.exp(-0.5 * q) / (2 * Math.PI * Math.sqrt(det)) / clutter;
}

const benchmarkTracks = [
  { x: 285, y: 205, S: [[520, 140], [140, 340]] },
  { x: 352, y: 232, S: [[460, -120], [-120, 480]] },
  { x: 318, y: 158, S: [[620, 0], [0, 300]] }
];
const benchmarkMeasurements = [
  { x: 318, y: 198 }, { x: 322, y: 215 }, { x: 314, y: 186 }, { x: 560, y: 120 }
];
const L = benchmarkTracks.map(track => [
  0.1,
  ...benchmarkMeasurements.map(measurement => matrixWeight(track, measurement, track.S))
]);

function bpHistory(weights) {
  const n = weights.length;
  const m = weights[0].length - 1;
  let nu = Array.from({ length: m }, () => Array(n).fill(1));
  const mu = Array.from({ length: n }, () => Array(m).fill(0));
  const marg = () => weights.map((row, i) => {
    const r = [row[0], ...Array.from({ length: m }, (_, j) => row[j + 1] * nu[j][i])];
    const z = r.reduce((a, b) => a + b, 0);
    return r.map(v => v / z);
  });
  const history = [marg()];
  for (let sweep = 1; sweep <= 50; sweep += 1) {
    for (let i = 0; i < n; i += 1) {
      let total = weights[i][0];
      for (let j = 0; j < m; j += 1) total += weights[i][j + 1] * nu[j][i];
      for (let j = 0; j < m; j += 1) {
        mu[i][j] = weights[i][j + 1] / Math.max(1e-15, total - weights[i][j + 1] * nu[j][i]);
      }
    }
    const next = Array.from({ length: m }, () => Array(n).fill(1));
    let delta = 0;
    for (let j = 0; j < m; j += 1) {
      let total = 1;
      for (let i = 0; i < n; i += 1) total += mu[i][j];
      for (let i = 0; i < n; i += 1) {
        next[j][i] = 1 / Math.max(1e-15, total - mu[i][j]);
        delta = Math.max(delta, Math.abs(next[j][i] - nu[j][i]));
      }
    }
    nu = next;
    history.push(marg());
    if (delta < 1e-10) break;
  }
  return history;
}

function enumerate(weights) {
  const n = weights.length;
  const m = weights[0].length - 1;
  const a = Array(n).fill(-1);
  const events = [];
  function rec(i, used, weight) {
    if (i === n) {
      events.push({ a: a.slice(), weight });
      return;
    }
    a[i] = -1;
    rec(i + 1, used, weight * weights[i][0]);
    for (let j = 0; j < m; j += 1) {
      if ((used & (1 << j)) || weights[i][j + 1] <= 0) continue;
      a[i] = j;
      rec(i + 1, used | (1 << j), weight * weights[i][j + 1]);
    }
    a[i] = -1;
  }
  rec(0, 0, 1);
  const z = events.reduce((sum, event) => sum + event.weight, 0);
  events.forEach(event => { event.p = event.weight / z; });
  events.sort((a, b) => b.p - a.p);
  return events;
}

function marginals(events, n = 3, m = 4) {
  const M = Array.from({ length: n }, () => Array(m + 1).fill(0));
  for (const event of events) {
    for (let i = 0; i < n; i += 1) M[i][event.a[i] < 0 ? 0 : event.a[i] + 1] += event.p;
  }
  return M;
}

const bp = bpHistory(L).at(-1);
const events = enumerate(L);
const exact = marginals(events);
let maxBpError = 0;
for (let i = 0; i < 3; i += 1) {
  for (let j = 0; j < 5; j += 1) maxBpError = Math.max(maxBpError, Math.abs(bp[i][j] - exact[i][j]));
}
const topFiveMass = events.slice(0, 5).reduce((sum, event) => sum + event.p, 0);

function fmtWeight(value) {
  if (value <= 0) return "·";
  if (value >= 10) return value.toFixed(1);
  return value.toFixed(2);
}

function story(event) {
  return event.a.map((j, i) => "T" + (i + 1) + "→" + (j < 0 ? "∅" : "z" + (j + 1))).join(" · ");
}

const slides = [];

slides.push({
  id: "s-cover",
  background: PAPER,
  transition: "none",
  notes: "Open with the shared data-association question. The deck contrasts two inference representations, not two incompatible measurement models. BP will approximate marginals; the joint-hypothesis route will preserve compatible global stories.",
  elements: [
    text("cover-kicker", 72, 68, 900, 28, "MULTI-TARGET TRACKING · DATA ASSOCIATION", 15, BP_DEEP, {
      fontWeight: 700, letterSpacing: 2.6, fx: { enter: "fade-up", order: 0 }
    }),
    text("cover-title", 72, 122, 1136, 142,
      "One association problem.\u003cbr>\u003cspan style=\"color:#155D8F\">Two\u003c/span> \u003cspan style=\"color:#B45607\">philosophies.\u003c/span>",
      65, INK, { fontWeight: 700, lineHeight: 1.02, fx: { enter: "fade-up", order: 1 } }),
    text("cover-question", 74, 292, 980, 48,
      "Which measurement belongs to which target—and how much ambiguity should survive?",
      24, SOFT, { lineHeight: 1.35, fx: { enter: "fade-up", order: 2 } }),
    rect("cover-bp-card", 72, 390, 510, 190, BP_WASH, { stroke: BP, strokeWidth: 1, radius: 12 }),
    text("cover-bp-label", 104, 420, 440, 30, "BELIEF PROPAGATION", 16, BP_DEEP, {
      fontWeight: 700, letterSpacing: 1.8
    }),
    text("cover-bp-big", 104, 466, 440, 52, "Pass messages.", 34, INK, { fontWeight: 700 }),
    text("cover-bp-small", 104, 520, 420, 52, "Approximate marginal probabilities—without listing global events.", 17, SOFT, { lineHeight: 1.25 }),
    rect("cover-pm-card", 626, 390, 582, 190, PM_WASH, { stroke: PM, strokeWidth: 1, radius: 12 }),
    text("cover-pm-label", 658, 420, 500, 30, "JOINT HYPOTHESES · PMBM VIEW", 16, PM_DEEP, {
      fontWeight: 700, letterSpacing: 1.5
    }),
    text("cover-pm-big", 658, 466, 500, 52, "Keep the stories.", 34, INK, { fontWeight: 700 }),
    text("cover-pm-small", 658, 525, 500, 38, "Weight compatible global assignments, then manage the mixture.", 17, SOFT),
    rule("cover-motion-rule", 72, 620, 1136, INK, 2, { fx: { loop: { type: "dash-march" } } }),
    text("cover-author", 72, 650, 1136, 24, "Bai Liping · bailiping.com/bp-vs-pmbm", 13, FAINT)
  ]
});

slides.push(regular(
  "s-question", "01 · THE QUESTION",
  "Every scan asks a \u003cspan style=\"color:#155D8F\">local\u003c/span> question with a \u003cspan style=\"color:#B45607\">global\u003c/span> constraint.",
  "Likelihoods score individual pairs. A legal association must also obey one target ↔ at most one measurement.",
  "Explain that high pairwise likelihood is necessary but insufficient. Two tracks cannot both claim the same measurement, so association probabilities are coupled even before any state update.",
  [
    ...[0,1,2].flatMap((i) => [
      circle("question-track-" + i, 110, 264 + i * 95, 42, TRACKS[i]),
      text("question-track-label-" + i, 117, 273 + i * 95, 28, 20, "T" + (i + 1), 15, WHITE, { fontWeight: 700, align: "center" })
    ]),
    rect("question-likelihood", 326, 280, 280, 225, WHITE, { stroke: LINE, strokeWidth: 1, radius: 12 }),
    text("question-likelihood-title", 354, 309, 225, 28, "PAIRWISE EVIDENCE", 13, BP_DEEP, { fontWeight: 700, letterSpacing: 1.5 }),
    text("question-likelihood-body", 354, 345, 220, 138, "How plausible is Tᵢ ↔ zⱼ?\u003cbr>\u003cbr>ℓᵢⱼ combines detection,\u003cbr>likelihood, and clutter.", 20, INK, { lineHeight: 1.28 }),
    rect("question-constraint", 660, 246, 470, 290, PM_WASH, { stroke: PM, strokeWidth: 2, radius: 14 }),
    text("question-constraint-title", 696, 278, 400, 30, "THE ONE-TO-ONE CONSTRAINT", 14, PM_DEEP, { fontWeight: 700, letterSpacing: 1.5 }),
    text("question-constraint-big", 696, 326, 394, 116, "A collection of good pairs can still be an impossible global story.", 30, INK, { fontWeight: 700, lineHeight: 1.14 }),
    text("question-constraint-small", 696, 456, 394, 42, "Inference is the negotiation between local evidence and global compatibility.", 17, SOFT)
  ]
));

slides.push(regular(
  "s-boundary", "02 · MODEL BOUNDARY",
  "First, name exactly what is being compared.",
  "The live computation is exact for a normalized one-scan assignment model—not for an entire PMBM filter update.",
  "This is the key accuracy slide. Exact refers only to exhaustive summation of the toy assignment events. A complete PMBM update also includes PPP-driven new-target evidence, Bernoulli existence and state densities, and the rest of the RFS recursion.",
  [
    rect("boundary-main", 72, 232, 1136, 126, WHITE, { stroke: PM, strokeWidth: 3, radius: 12 }),
    text("boundary-eq", 104, 256, 1072, 36, "EXACT HERE = Σ over every valid normalized assignment event", 25, PM_DEEP, { fontWeight: 700, align: "center" }),
    text("boundary-not", 104, 307, 1072, 28, "≠ a complete PMBM posterior update", 21, INK, { fontWeight: 700, align: "center" }),
    rect("boundary-in", 72, 394, 542, 220, BP_WASH, { radius: 12 }),
    text("boundary-in-head", 104, 422, 478, 28, "IN THE BENCHMARK", 14, BP_DEEP, { fontWeight: 700, letterSpacing: 1.7 }),
    text("boundary-in-body", 104, 464, 470, 130,
      "• existing tracks may be missed\u003cbr>• each track claims at most one measurement\u003cbr>• each measurement has at most one existing-track owner\u003cbr>• unassigned-measurement baseline weight = 1",
      18, INK, { lineHeight: 1.55 }),
    rect("boundary-out", 666, 394, 542, 220, PM_WASH, { radius: 12 }),
    text("boundary-out-head", 698, 422, 478, 28, "REQUIRED IN A FULL PMBM UPDATE", 14, PM_DEEP, { fontWeight: 700, letterSpacing: 1.4 }),
    text("boundary-out-body", 698, 464, 470, 130,
      "• undetected-target Poisson intensity\u003cbr>• measurement-specific PPP birth evidence\u003cbr>• Bernoulli existence and state densities\u003cbr>• global-hypothesis history and RFS state update",
      18, INK, { lineHeight: 1.55 })
  ],
  { sectionColor: PM_DEEP }
));

const weightRows = [
  { cells: [{ html: "" }, { html: "∅ miss" }, ...benchmarkMeasurements.map((_, j) => ({ html: "z" + (j + 1) }))] },
  ...L.map((row, i) => ({
    cells: [
      { html: "T" + (i + 1), bold: true },
      ...row.map(value => ({ html: fmtWeight(value) }))
    ]
  }))
];
slides.push(regular(
  "s-weights", "03 · SHARED INPUT",
  "Both routes consume the same gated assignment weights.",
  "Change geometry, Pᴅ, or clutter density and the entire inference problem changes.",
  "Walk through the matrix. The missed-detection column is one minus detection probability. Gated pairs use the likelihood-to-clutter ratio. A dot is exactly zero after gating. Unassigned measurements contribute the normalized baseline one.",
  [
    {
      id: "weight-table", type: "table", x: 72, y: 242, w: 694, h: 250, rotation: 0, opacity: 1,
      header: true,
      columns: [{ w: 1.0 }, { w: 1.0 }, { w: 1.0 }, { w: 1.0 }, { w: 1.0 }, { w: 1.0 }],
      rows: weightRows,
      style: {
        headerBg: INK, headerColor: WHITE, zebra: "rgba(31,119,180,0.055)",
        borderColor: "rgba(22,34,46,0.15)", borderWidth: 1,
        cellPadX: 12, cellPadY: 10, fontSize: 18, color: INK, radius: 9
      }
    },
    rect("weight-formula-card", 814, 242, 394, 250, BP_WASH, { stroke: BP, strokeWidth: 1, radius: 12 }),
    text("weight-formula-head", 846, 270, 330, 26, "NORMALIZED WEIGHTS", 14, BP_DEEP, { fontWeight: 700, letterSpacing: 1.5 }),
    text("weight-formula", 846, 320, 330, 92,
      "ℓᵢⱼ = Pᴅ · N(zⱼ; ẑᵢ,Sᵢ) / λc\u003cbr>\u003cbr>ℓᵢ∅ = 1 − Pᴅ",
      23, INK, { lineHeight: 1.35 }),
    text("weight-baseline", 846, 430, 330, 38, "Unassigned measurement: baseline weight 1", 16, SOFT, { fontWeight: 700 }),
    rect("weight-insight", 72, 526, 1136, 78, WHITE, { stroke: LINE, strokeWidth: 1, radius: 10 }),
    text("weight-insight-copy", 100, 540, 1080, 52,
      "This equality of inputs makes the marginal comparison meaningful: message passing and enumeration answer the same toy assignment question.",
      19, INK, { align: "center", valign: "middle" }),
    inlineMount()
  ]
));

const graphElements = [];
const leftYs = [290, 400, 510];
const rightYs = [250, 345, 440, 535];
const gatedPairs = [];
for (let i = 0; i < 3; i += 1) {
  for (let j = 0; j < 4; j += 1) if (L[i][j + 1] > 0) gatedPairs.push([i, j]);
}
for (const [i, j] of gatedPairs) {
  const x1 = 188, x2 = 590, y1 = leftYs[i], y2 = rightYs[j];
  graphElements.push(rect("graph-edge-" + i + "-" + j, x1, y1, Math.hypot(x2 - x1, y2 - y1), 2, TRACKS[i], {
    rotation: Math.atan2(y2 - y1, x2 - x1) * 180 / Math.PI,
    opacity: 0.38
  }));
}
for (let i = 0; i < 3; i += 1) {
  graphElements.push(circle("graph-a-" + i, 148, leftYs[i] - 20, 40, WHITE, { stroke: TRACKS[i], strokeWidth: 3 }));
  graphElements.push(text("graph-a-label-" + i, 155, leftYs[i] - 10, 26, 20, "a" + (i + 1), 16, TRACKS[i], { fontWeight: 700, align: "center" }));
}
for (let j = 0; j < 4; j += 1) {
  graphElements.push(circle("graph-b-" + j, 590, rightYs[j] - 18, 36, WHITE, { stroke: SOFT, strokeWidth: 2 }));
  graphElements.push(text("graph-b-label-" + j, 596, rightYs[j] - 9, 24, 18, "b" + (j + 1), 14, SOFT, { fontWeight: 700, align: "center" }));
}
slides.push(regular(
  "s-constraint", "04 · FACTOR GRAPH",
  "Duplicate the bookkeeping so consistency can be local.",
  "Track variable aᵢ and measurement variable bⱼ describe the same ownership decision from opposite sides.",
  "The variables are redundant by design. Pairwise consistency factors ensure that a_i equals j if and only if b_j equals i. This construction converts the global one-to-one constraint into local messages on a bipartite graph.",
  [
    ...graphElements,
    text("graph-left-label", 105, 228, 155, 22, "TRACK VARIABLES", 12, BP_DEEP, { fontWeight: 700, align: "center", letterSpacing: 1.3 }),
    text("graph-right-label", 548, 224, 155, 32, "MEASUREMENT VARIABLES", 11, SOFT, { fontWeight: 700, align: "center", letterSpacing: 1.0, lineHeight: 1.15 }),
    rect("graph-card", 754, 246, 454, 332, WHITE, { stroke: LINE, strokeWidth: 1, radius: 12 }),
    text("graph-card-head", 790, 278, 382, 28, "LOCAL FACTORS, GLOBAL LEGALITY", 14, BP_DEEP, { fontWeight: 700, letterSpacing: 1.3 }),
    text("graph-card-body", 790, 326, 370, 182,
      "Ψᵢⱼ(aᵢ,bⱼ) says:\u003cbr>\u003cbr>• if aᵢ = j, then bⱼ = i\u003cbr>• if bⱼ = i, then aᵢ = j\u003cbr>• otherwise the pair must agree that no link exists",
      19, INK, { lineHeight: 1.45 }),
    text("graph-card-foot", 790, 522, 370, 38, "Gated-out pairs carry zero weight and become inert.", 16, SOFT, { fontWeight: 700 })
  ]
));

slides.push(regular(
  "s-bp-view", "05 · BP PHILOSOPHY",
  "\u003cspan style=\"color:#155D8F\">Do not enumerate stories.\u003c/span> Negotiate their marginals.",
  "Each edge asks: how committed is this track elsewhere, and how contested is this measurement?",
  "BP never constructs the global assignment list. It repeatedly exchanges scalar messages. The result is a marginal probability per track-measurement association, suitable for a soft state update or another marginal tracker component.",
  [
    rect("bp-input", 72, 270, 290, 220, WHITE, { stroke: LINE, strokeWidth: 1, radius: 12 }),
    text("bp-input-head", 104, 300, 226, 28, "LOCAL WEIGHTS", 14, BP_DEEP, { fontWeight: 700, letterSpacing: 1.6 }),
    text("bp-input-big", 104, 342, 226, 112, "ℓᵢⱼ\u003cbr>+ one-to-one factors", 27, INK, { fontWeight: 700, lineHeight: 1.28 }),
    text("bp-input-small", 104, 459, 226, 26, "No event list.", 16, SOFT),
    rect("bp-flow", 414, 236, 410, 290, BP_WASH, { stroke: BP, strokeWidth: 2, radius: 145 }),
    text("bp-flow-title", 464, 286, 310, 42, "MESSAGE LOOP", 16, BP_DEEP, { fontWeight: 700, align: "center", letterSpacing: 2 }),
    text("bp-flow-mu", 470, 352, 300, 34, "μ · track → measurement", 23, TRACKS[0], { fontWeight: 700, align: "center" }),
    text("bp-flow-nu", 470, 408, 300, 34, "ν · measurement → track", 23, SOFT, { fontWeight: 700, align: "center" }),
    rule("bp-flow-rule", 500, 396, 238, BP, 2, { fx: { loop: { type: "dash-march" } } }),
    rect("bp-output", 876, 270, 332, 220, WHITE, { stroke: BP, strokeWidth: 1, radius: 12 }),
    text("bp-output-head", 908, 300, 268, 28, "OUTPUT", 14, BP_DEEP, { fontWeight: 700, letterSpacing: 1.6 }),
    text("bp-output-big", 908, 350, 268, 76, "p(aᵢ = j)", 34, INK, { fontWeight: 700, align: "center" }),
    text("bp-output-small", 908, 444, 268, 32, "Approximate marginals.", 17, SOFT, { align: "center" }),
    text("bp-bottom", 116, 548, 1048, 54,
      "The efficiency comes from never writing down the joint events—not from pretending the one-to-one constraint disappeared.",
      20, INK, { fontWeight: 700, align: "center" })
  ]
));

slides.push(regular(
  "s-bp-messages", "06 · WILLIAMS–LAU BP",
  "Two reciprocal messages, repeated to a fixed point.",
  "The specific association construction converges; each sweep touches every track–measurement edge.",
  "Use careful wording: the Williams–Lau data-association BP construction has a convergence guarantee. Its cost is O(nm) per sweep, or O(Tnm) for T sweeps. This statement should not be generalized to arbitrary loopy BP models.",
  [
    rect("bp-eq-one", 72, 246, 536, 164, BP_WASH, { stroke: BP, strokeWidth: 1, radius: 12 }),
    text("bp-eq-one-label", 104, 274, 470, 24, "TRACK → MEASUREMENT", 13, BP_DEEP, { fontWeight: 700, letterSpacing: 1.4 }),
    text("bp-eq-one-main", 104, 318, 470, 48, "μᵢ→ⱼ = ℓᵢⱼ / (ℓᵢ∅ + Σₖ≠ⱼ ℓᵢₖνₖ→ᵢ)", 23, INK, { fontWeight: 700 }),
    text("bp-eq-one-note", 104, 375, 470, 24, "How committed is track i elsewhere?", 16, SOFT),
    rect("bp-eq-two", 672, 246, 536, 164, WHITE, { stroke: BP, strokeWidth: 1, radius: 12 }),
    text("bp-eq-two-label", 704, 274, 470, 24, "MEASUREMENT → TRACK", 13, BP_DEEP, { fontWeight: 700, letterSpacing: 1.4 }),
    text("bp-eq-two-main", 704, 318, 470, 48, "νⱼ→ᵢ = 1 / (1 + Σₗ≠ᵢ μₗ→ⱼ)", 25, INK, { fontWeight: 700 }),
    text("bp-eq-two-note", 704, 375, 470, 24, "How contested is measurement j?", 16, SOFT),
    rect("bp-schedule", 72, 452, 1136, 142, WHITE, { stroke: LINE, strokeWidth: 1, radius: 10 }),
    text("bp-step-0", 108, 484, 200, 26, "0 · ν ≡ 1", 20, INK, { fontWeight: 700, align: "center" }),
    text("bp-arrow-1", 310, 482, 64, 28, "→", 24, BP, { fontWeight: 700, align: "center" }),
    text("bp-step-1", 376, 484, 230, 26, "1 · all μ fire", 20, TRACKS[0], { fontWeight: 700, align: "center" }),
    text("bp-arrow-2", 610, 482, 64, 28, "→", 24, BP, { fontWeight: 700, align: "center" }),
    text("bp-step-2", 680, 484, 230, 26, "2 · all ν reply", 20, SOFT, { fontWeight: 700, align: "center" }),
    text("bp-arrow-3", 914, 482, 64, 28, "→", 24, BP, { fontWeight: 700, align: "center" }),
    text("bp-step-3", 984, 484, 180, 26, "fixed point", 20, BP_DEEP, { fontWeight: 700, align: "center" }),
    text("bp-complexity", 128, 545, 1024, 30,
      "Williams–Lau construction: O(nm) per sweep · O(Tnm) for T sweeps · convergence guarantee for this association model",
      16, SOFT, { align: "center" }),
    inlineMount()
  ]
));

const comparisonCategories = ["T1→z1", "T1→z2", "T2→z2", "T3→z3"];
const comparisonIndex = [[0,1], [0,2], [1,2], [2,3]];
const comparisonBp = comparisonIndex.map(([i,j]) => +(100 * bp[i][j]).toFixed(2));
const comparisonExact = comparisonIndex.map(([i,j]) => +(100 * exact[i][j]).toFixed(2));
slides.push(regular(
  "s-bp-marginals", "07 · BP OUTPUT",
  "The output is a table of \u003cspan style=\"color:#155D8F\">marginals\u003c/span>—not a winning joint story.",
  "On a loopy graph the fixed point is generally approximate; the error depends on the actual weights.",
  "The bars use the default live tangle. Exhaustive enumeration gives the exact normalized-assignment marginal. BP is exact on an acyclic factor graph, but loopiness alone does not determine the direction or magnitude of error.",
  [
    {
      id: "bp-compare-chart", type: "chart", x: 72, y: 232, w: 790, h: 365, rotation: 0, opacity: 1,
      preset: "bar",
      option: {
        grid: { left: 70, right: 24, top: 48, bottom: 48 },
        legend: { data: ["BP fixed point", "Exact enumeration"], top: 5 },
        xAxis: { type: "category", data: comparisonCategories, axisLabel: { fontSize: 14 } },
        yAxis: { type: "value", max: 100, axisLabel: { formatter: "{value}%" } },
        series: [
          { name: "BP fixed point", type: "bar", data: comparisonBp, itemStyle: { color: BP, borderRadius: [4,4,0,0] }, barMaxWidth: 48 },
          { name: "Exact enumeration", type: "bar", data: comparisonExact, itemStyle: { color: PM, borderRadius: [4,4,0,0] }, barMaxWidth: 48 }
        ],
        tooltip: { trigger: "axis" }
      },
      fx: { enter: "fade-up" }
    },
    rect("bp-error-card", 910, 248, 298, 172, BP_WASH, { stroke: BP, strokeWidth: 1, radius: 12 }),
    text("bp-error-label", 938, 278, 242, 25, "DEFAULT TANGLE", 13, BP_DEEP, { fontWeight: 700, align: "center", letterSpacing: 1.4 }),
    text("bp-error-number", 938, 320, 242, 58, (100 * maxBpError).toFixed(2) + " pp", 44, INK, { fontWeight: 700, align: "center" }),
    text("bp-error-desc", 938, 383, 242, 25, "largest marginal difference", 15, SOFT, { align: "center" }),
    rect("bp-error-note", 910, 448, 298, 149, WHITE, { stroke: LINE, strokeWidth: 1, radius: 12 }),
    text("bp-error-note-copy", 938, 470, 242, 120,
      "Exact on acyclic association graphs.\u003cbr>\u003cbr>Approximate on this loopy tangle.\u003cbr>\u003cbr>Error is not monotone in “loopiness.”",
      16, INK, { lineHeight: 1.42 })
  ]
));

slides.push(regular(
  "s-joint-events", "08 · JOINT VIEW",
  "Now write down the compatible global stories.",
  "Each event assigns every existing track to one measurement or ∅, with no measurement claimed twice.",
  "Enumeration is feasible only because this example is tiny. The weight of a joint event is the product of its assigned-pair and missed-detection weights, with baseline one for every unassigned measurement.",
  [
    ...events.slice(0, 3).flatMap((event, rank) => {
      const y = 242 + rank * 112;
      return [
        rect("event-card-" + rank, 72, y, 760, 88, rank === 0 ? PM_WASH : WHITE, {
          stroke: rank === 0 ? PM : LINE, strokeWidth: rank === 0 ? 2 : 1, radius: 10
        }),
        text("event-rank-" + rank, 94, y + 17, 60, 25, "#" + (rank + 1), 16, PM_DEEP, { fontWeight: 700 }),
        text("event-story-" + rank, 162, y + 17, 500, 28, story(event), 20, INK, { fontWeight: 700 }),
        text("event-weight-" + rank, 676, y + 16, 128, 30, (100 * event.p).toFixed(1) + "%", 22, PM_DEEP, { fontWeight: 700, align: "right" }),
        rule("event-bar-bg-" + rank, 162, y + 59, 590, LINE, 8, { radius: 4 }),
        rule("event-bar-" + rank, 162, y + 59, Math.max(8, 590 * event.p / events[0].p), PM, 8, { radius: 4 })
      ];
    }),
    rect("event-formula-card", 878, 242, 330, 312, PM_WASH, { stroke: PM, strokeWidth: 1, radius: 12 }),
    text("event-formula-head", 910, 272, 266, 28, "JOINT WEIGHT", 14, PM_DEEP, { fontWeight: 700, align: "center", letterSpacing: 1.5 }),
    text("event-formula-main", 910, 326, 266, 110,
      "wA ∝\u003cbr>Π(i,j)∈A ℓᵢⱼ\u003cbr>· Πi missed ℓᵢ∅",
      24, INK, { fontWeight: 700, align: "center", lineHeight: 1.45 }),
    text("event-formula-note", 910, 454, 266, 78,
      "Summing weights across matching rows gives exact assignment marginals for this benchmark.",
      16, SOFT, { align: "center", lineHeight: 1.4 }),
    text("event-count", 72, 596, 1136, 28,
      events.length + " valid events in the default gated tangle · all can be enumerated here; real workloads require selection and pruning",
      17, SOFT, { align: "center" })
  ],
  { sectionColor: PM_DEEP }
));

slides.push(regular(
  "s-pmbm", "09 · PMBM CONTEXT",
  "PMBM keeps ambiguity in a global-hypothesis mixture.",
  "An undetected-target Poisson process sits beside a multi-Bernoulli mixture for detected targets.",
  "This slide places the joint-assignment table in the correct PMBM context. A full PMBM point-target update is conjugate under its assumed model. New-target Bernoulli evidence is measurement specific and comes from integrating the likelihood against the undetected-target PPP.",
  [
    rect("pmbm-ppp", 72, 250, 346, 286, PM_WASH, { stroke: PM, strokeWidth: 2, radius: 143 }),
    text("pmbm-ppp-label", 116, 300, 258, 28, "UNDETECTED", 14, PM_DEEP, { fontWeight: 700, align: "center", letterSpacing: 1.8 }),
    text("pmbm-ppp-big", 116, 342, 258, 78, "Poisson\u003cbr>intensity λᵘ(x)", 28, INK, { fontWeight: 700, align: "center", lineHeight: 1.2 }),
    text("pmbm-ppp-small", 116, 444, 258, 46, "Drives measurement-specific new-target evidence.", 16, SOFT, { align: "center", lineHeight: 1.35 }),
    text("pmbm-union", 440, 362, 76, 50, "⊎", 40, PM_DEEP, { fontWeight: 700, align: "center" }),
    rect("pmbm-mbm", 526, 246, 682, 296, WHITE, { stroke: LINE, strokeWidth: 1, radius: 12 }),
    text("pmbm-mbm-head", 558, 276, 618, 28, "DETECTED · MULTI-BERNOULLI MIXTURE", 14, PM_DEEP, { fontWeight: 700, letterSpacing: 1.4 }),
    ...[0,1,2].flatMap((rank) => {
      const y = 328 + rank * 63;
      return [
        rect("pmbm-h-" + rank, 558, y, 586, 48, rank === 0 ? PM_WASH : PAPER, { radius: 7 }),
        text("pmbm-h-label-" + rank, 576, y + 11, 410, 24, "h" + (rank + 1) + " · " + story(events[rank]), 16, INK, { fontWeight: 700 }),
        text("pmbm-h-weight-" + rank, 1010, y + 11, 112, 24, "w = " + (100 * events[rank].p).toFixed(1) + "%", 15, PM_DEEP, { fontWeight: 700, align: "right" })
      ];
    }),
    rect("pmbm-boundary", 72, 574, 1136, 60, PM_WASH, { stroke: PM, strokeWidth: 1, radius: 10 }),
    text("pmbm-boundary-copy", 96, 582, 1088, 44,
      "The live table mirrors normalized assignment/hypothesis bookkeeping only. It does not compute PPP birth evidence, Bernoulli existence, or state densities.",
      17, INK, { fontWeight: 700, align: "center", valign: "middle" })
  ],
  { sectionColor: PM_DEEP }
));

const hypBarData = events.slice(0, 10).map((event, i) => ({
  value: +(100 * event.p).toFixed(3),
  itemStyle: { color: i < 5 ? PM : "#E7C5A8", borderRadius: [4,4,0,0] }
}));
slides.push(regular(
  "s-pruning", "10 · HYPOTHESIS MANAGEMENT",
  "Retain the head. Quantify the tail.",
  "Practical joint-hypothesis filters generate selected high-weight children, then prune, recycle, merge, or cap.",
  "The chart ranks exact normalized event weights for the default scene. Top-k truncation loses probability mass and changes marginals after renormalization. The MAP row alone is one association story, not a complete multi-object estimate.",
  [
    {
      id: "prune-chart", type: "chart", x: 72, y: 236, w: 800, h: 350, rotation: 0, opacity: 1,
      preset: "bar",
      option: {
        grid: { left: 62, right: 20, top: 24, bottom: 40 },
        xAxis: { type: "category", data: events.slice(0,10).map((_, i) => "#" + (i + 1)), axisLabel: { fontSize: 14 } },
        yAxis: { type: "value", axisLabel: { formatter: "{value}%" } },
        series: [{ type: "bar", data: hypBarData, barMaxWidth: 44 }],
        tooltip: { trigger: "item", formatter: "event {b}: {c}%" }
      },
      fx: { enter: "fade-up" }
    },
    rect("prune-mass-card", 920, 244, 288, 154, PM_WASH, { stroke: PM, strokeWidth: 1, radius: 12 }),
    text("prune-mass-label", 948, 272, 232, 26, "TOP 5 RETAINS", 13, PM_DEEP, { fontWeight: 700, align: "center", letterSpacing: 1.5 }),
    text("prune-mass-number", 948, 312, 232, 56, (100 * topFiveMass).toFixed(1) + "%", 42, INK, { fontWeight: 700, align: "center", lineHeight: 1.05 }),
    rect("prune-map-card", 920, 432, 288, 154, WHITE, { stroke: LINE, strokeWidth: 1, radius: 12 }),
    text("prune-map-label", 948, 458, 232, 26, "MAP ONLY RETAINS", 13, PM_DEEP, { fontWeight: 700, align: "center", letterSpacing: 1.3 }),
    text("prune-map-number", 948, 500, 232, 56, (100 * events[0].p).toFixed(1) + "%", 42, INK, { fontWeight: 700, align: "center", lineHeight: 1.05 }),
    text("prune-foot", 112, 605, 1048, 28,
      "Orange = kept by k = 5 · pale = pruned tail · truncation changes the renormalized assignment marginals",
      16, SOFT, { align: "center" }),
    inlineMount()
  ],
  { sectionColor: PM_DEEP }
));

slides.push(regular(
  "s-head-to-head", "11 · HEAD TO HEAD",
  "Same weights. Different objects retained.",
  "The methods answer related questions—but preserve different information structures.",
  "Use the table as the main comparison. BP returns marginals directly; the joint-hypothesis view can return those same marginals only after summing events, while also preserving cross-track correlations and alternative histories.",
  [
    {
      id: "head-table", type: "table", x: 72, y: 228, w: 1136, h: 390, rotation: 0, opacity: 1,
      header: true,
      columns: [{ w: 0.9 }, { w: 1.45 }, { w: 1.45 }],
      rows: [
        { cells: [{ html: "" }, { html: "Belief propagation" }, { html: "Joint hypotheses / PMBM view" }] },
        { cells: [{ html: "Retains", bold: true }, { html: "marginal p(aᵢ=j)" }, { html: "weighted compatible global stories h" }] },
        { cells: [{ html: "Core operation", bold: true }, { html: "iterated μ and ν messages" }, { html: "assignment generation + weight + pruning" }] },
        { cells: [{ html: "Correlation", bold: true }, { html: "projected into marginals" }, { html: "preserved across tracks and hypotheses" }] },
        { cells: [{ html: "Approximation", bold: true }, { html: "Bethe fixed-point marginals on loops" }, { html: "gating, selected children, truncation, numerical state integrals" }] },
        { cells: [{ html: "Best pressure", bold: true }, { html: "scale and latency" }, { html: "ambiguity, identity, history" }] }
      ],
      style: {
        headerBg: INK, headerColor: WHITE, zebra: "rgba(22,34,46,0.04)",
        borderColor: "rgba(22,34,46,0.15)", borderWidth: 1,
        cellPadX: 18, cellPadY: 13, fontSize: 18, color: INK, radius: 10
      }
    },
    rect("head-bp-accent", 378, 228, 390, 6, BP, { radius: 3 }),
    rect("head-pm-accent", 768, 228, 440, 6, PM, { radius: 3 })
  ]
));

function partialAssignments(n) {
  let total = 0;
  function factorial(x) { let p = 1; for (let i = 2; i <= x; i += 1) p *= i; return p; }
  function choose(a, b) { return factorial(a) / (factorial(b) * factorial(a - b)); }
  for (let k = 0; k <= n; k += 1) total += choose(n, k) * factorial(n) / factorial(n - k);
  return total;
}
const scaleN = [1,2,3,4,5,6,7];
slides.push(regular(
  "s-scaling", "12 · SCALING",
  "Edges grow quadratically. Joint stories explode.",
  "Complete gating with n tracks and n measurements makes the contrast visible before any state dimension enters.",
  "The event count is the exact number of partial injective assignments under complete gating. The BP curve is edge count n squared. Williams–Lau message updates cost O(nm) per sweep, though total runtime also includes the number of sweeps.",
  [
    {
      id: "scaling-chart", type: "chart", x: 72, y: 224, w: 830, h: 390, rotation: 0, opacity: 1,
      preset: "line",
      option: {
        grid: { left: 74, right: 26, top: 48, bottom: 46 },
        legend: { data: ["BP edges n²", "valid joint events"], top: 5 },
        xAxis: { type: "category", data: scaleN.map(String), name: "n tracks = n measurements" },
        yAxis: { type: "log", logBase: 10, axisLabel: { formatter: "{value}" } },
        series: [
          { name: "BP edges n²", type: "line", data: scaleN.map(n => n * n), smooth: true, symbolSize: 7, lineStyle: { width: 3, color: BP }, itemStyle: { color: BP } },
          { name: "valid joint events", type: "line", data: scaleN.map(partialAssignments), smooth: true, symbolSize: 7, lineStyle: { width: 3, color: PM }, itemStyle: { color: PM } }
        ],
        tooltip: { trigger: "axis" }
      },
      fx: { enter: "fade-up" }
    },
    rect("scaling-bp", 948, 244, 260, 140, BP_WASH, { stroke: BP, strokeWidth: 1, radius: 12 }),
    text("scaling-bp-head", 972, 270, 212, 24, "BP SWEEP", 13, BP_DEEP, { fontWeight: 700, align: "center", letterSpacing: 1.5 }),
    text("scaling-bp-big", 972, 310, 212, 42, "O(nm)", 34, INK, { fontWeight: 700, align: "center" }),
    rect("scaling-joint", 948, 414, 260, 172, PM_WASH, { stroke: PM, strokeWidth: 1, radius: 12 }),
    text("scaling-joint-head", 972, 440, 212, 24, "n = 7 COMPLETE GATING", 12, PM_DEEP, { fontWeight: 700, align: "center", letterSpacing: 1.0 }),
    text("scaling-joint-big", 972, 480, 212, 42, partialAssignments(7).toLocaleString(), 32, INK, { fontWeight: 700, align: "center" }),
    text("scaling-joint-small", 972, 536, 212, 28, "compatible partial assignments", 14, SOFT, { align: "center" })
  ]
));

slides.push(regular(
  "s-time", "13 · ACROSS SCANS",
  "The representational choice compounds over time.",
  "Marginalize now, or preserve alternative histories until later evidence resolves them.",
  "Many BP-based filters carry one belief set forward after each scan, which is efficient but can merge modes and contribute to coalescence. PMBM-style global hypotheses preserve alternative association histories, then prune them under computational pressure.",
  [
    text("time-bp-label", 72, 236, 250, 28, "BP-BASED TRACKER", 14, BP_DEEP, { fontWeight: 700, letterSpacing: 1.5 }),
    ...[0,1,2].flatMap((i) => {
      const x = 96 + i * 250;
      return [
        rect("time-bp-box-" + i, x, 286, 178, 82, BP_WASH, { stroke: BP, strokeWidth: 1, radius: 10 }),
        text("time-bp-box-label-" + i, x + 16, 302, 146, 48, "scan " + (i + 1) + "\u003cbr>marginalize", 16, INK, { fontWeight: 700, align: "center", lineHeight: 1.3 }),
        ...(i < 2 ? [text("time-bp-arrow-" + i, x + 188, 310, 52, 28, "→", 24, BP, { fontWeight: 700, align: "center" })] : [])
      ];
    }),
    rect("time-bp-result", 862, 280, 346, 94, WHITE, { stroke: BP, strokeWidth: 2, radius: 12 }),
    text("time-bp-result-copy", 890, 302, 290, 52, "Flat cost; ambiguity projected at every scan.", 20, INK, { fontWeight: 700, align: "center", lineHeight: 1.3 }),
    rule("time-divider", 72, 410, 1136, LINE, 1),
    text("time-pm-label", 72, 434, 300, 38, "JOINT-HYPOTHESIS / PMBM TRACKER", 13, PM_DEEP, { fontWeight: 700, letterSpacing: 1.1, lineHeight: 1.15 }),
    rect("time-root", 100, 500, 142, 54, PM_WASH, { stroke: PM, strokeWidth: 2, radius: 9 }),
    text("time-root-label", 114, 515, 114, 24, "prior h", 17, INK, { fontWeight: 700, align: "center" }),
    ...[0,1,2].flatMap((i) => {
      const x = 340 + i * 185;
      return [
        rect("time-branch-a-" + i, x, 478, 128, 46, i < 2 ? PM_WASH : PAPER, { stroke: PM, strokeWidth: 1, radius: 7, opacity: i < 2 ? 1 : 0.45 }),
        text("time-branch-a-label-" + i, x + 10, 490, 108, 22, "h" + (i + 1) + "a", 15, INK, { fontWeight: 700, align: "center" }),
        rect("time-branch-b-" + i, x, 542, 128, 46, i === 0 ? PM_WASH : PAPER, { stroke: PM, strokeWidth: 1, radius: 7, opacity: i === 0 ? 1 : 0.35 }),
        text("time-branch-b-label-" + i, x + 10, 554, 108, 22, "h" + (i + 1) + "b", 15, INK, { fontWeight: 700, align: "center" })
      ];
    }),
    text("time-prune", 948, 504, 260, 62, "branch → weight → prune\u003cbr>identity ambiguity can survive", 18, PM_DEEP, { fontWeight: 700, align: "center", lineHeight: 1.45 })
  ]
));

slides.push(regular(
  "s-bridge", "14 · THE BRIDGE",
  "TOMB/P meets the two views in a specific middle.",
  "Approximate a PMBM-style mixture by one multi-Bernoulli using marginal association probabilities.",
  "The bridge is precise: TOMB/P uses marginal association probabilities to approximate a mixture by a single multi-Bernoulli. BP is one efficient method for approximating those marginals. This does not make every BP or sum-product tracker a PMBM filter with hypotheses removed.",
  [
    rect("bridge-left", 72, 272, 290, 220, PM_WASH, { stroke: PM, strokeWidth: 2, radius: 12 }),
    text("bridge-left-head", 100, 304, 234, 28, "PMBM-STYLE MIXTURE", 13, PM_DEEP, { fontWeight: 700, align: "center", letterSpacing: 1.1 }),
    text("bridge-left-big", 100, 354, 234, 82, "Σₕ wₕ · MBₕ", 30, INK, { fontWeight: 700, align: "center", valign: "middle" }),
    text("bridge-arrow-one", 370, 350, 105, 58, "→", 42, PM, { fontWeight: 700, align: "center", lineHeight: 1.0 }),
    rect("bridge-middle", 480, 242, 330, 280, WHITE, { stroke: BP, strokeWidth: 2, radius: 140 }),
    text("bridge-middle-head", 525, 285, 240, 28, "MARGINAL ASSOCIATION", 13, BP_DEEP, { fontWeight: 700, align: "center", letterSpacing: 1.0 }),
    text("bridge-middle-big", 525, 345, 240, 58, "p(aᵢ = j)", 35, INK, { fontWeight: 700, align: "center" }),
    text("bridge-middle-note", 525, 420, 240, 54, "exact by summing events\u003cbr>or approximated efficiently by BP", 16, SOFT, { align: "center", lineHeight: 1.4 }),
    text("bridge-arrow-two", 818, 350, 95, 58, "→", 42, BP, { fontWeight: 700, align: "center", lineHeight: 1.0 }),
    rect("bridge-right", 920, 272, 288, 220, BP_WASH, { stroke: BP, strokeWidth: 2, radius: 12 }),
    text("bridge-right-head", 950, 304, 228, 28, "TOMB/P PROJECTION", 13, BP_DEEP, { fontWeight: 700, align: "center", letterSpacing: 1.1 }),
    text("bridge-right-big", 950, 354, 228, 82, "one multi-\u003cbr>Bernoulli", 29, INK, { fontWeight: 700, align: "center", lineHeight: 1.25 }),
    rect("bridge-boundary", 160, 560, 960, 58, INK, { radius: 10 }),
    text("bridge-boundary-copy", 188, 576, 904, 28,
      "A specific bridge—not a universal equivalence between BP/SPA trackers and PMBM.",
      18, WHITE, { fontWeight: 700, align: "center" })
  ]
));

slides.push(regular(
  "s-decision", "15 · DECISION",
  "Choose what you can afford to forget.",
  "Neither representation dominates; the right approximation depends on scale, latency, identity, and how long ambiguity matters.",
  "Turn the comparison into a decision. If the scene is large and latency strict, BP's marginal route is compelling. If alternative histories and identity matter, preserve hypotheses as long as budget permits. Hybrid approximations deliberately choose where to project.",
  [
    rect("decision-bp", 72, 244, 516, 322, BP_WASH, { stroke: BP, strokeWidth: 2, radius: 14 }),
    text("decision-bp-head", 108, 278, 444, 30, "REACH FOR BP WHEN…", 16, BP_DEEP, { fontWeight: 700, letterSpacing: 1.4 }),
    text("decision-bp-body", 108, 332, 430, 172,
      "• n and m are large\u003cbr>• latency is a hard constraint\u003cbr>• marginals are the required output\u003cbr>• event enumeration is hopeless anyway\u003cbr>• message-parallel implementation matters",
      21, INK, { lineHeight: 1.55 }),
    text("decision-bp-foot", 108, 520, 430, 28, "Cost paid: loopy marginal approximation.", 16, BP_DEEP, { fontWeight: 700 }),
    rect("decision-pm", 692, 244, 516, 322, PM_WASH, { stroke: PM, strokeWidth: 2, radius: 14 }),
    text("decision-pm-head", 728, 278, 444, 30, "PRESERVE HYPOTHESES WHEN…", 16, PM_DEEP, { fontWeight: 700, letterSpacing: 1.1 }),
    text("decision-pm-body", 728, 332, 430, 172,
      "• crossings and close encounters dominate\u003cbr>• identity and history are first-class\u003cbr>• later evidence may settle ambiguity\u003cbr>• accuracy outweighs flat cost\u003cbr>• pruning mass can be monitored",
      21, INK, { lineHeight: 1.55 }),
    text("decision-pm-foot", 728, 520, 430, 28, "Cost paid: hypothesis management.", 16, PM_DEEP, { fontWeight: 700 }),
    text("decision-bottom", 160, 602, 960, 26, "The design question is not “which is best?” It is “which correlations must survive this scan?”", 19, INK, { fontWeight: 700, align: "center" })
  ]
));

slides.push(regular(
  "s-takeaways", "16 · TAKEAWAYS",
  "Four boundaries worth carrying forward.",
  "",
  "Close by restating the accuracy boundaries. The source papers are Williams and Lau 2014 for association BP, Williams 2015 for marginal multi-Bernoulli filters, Meyer et al. 2018 for scalable message-passing trackers, and García-Fernández et al. 2018 for PMBM.",
  [
    ...[
      ["01", "Same question, same weights.", "The benchmark isolates inference representation by feeding BP and enumeration the identical normalized assignment model.", BP_WASH, BP_DEEP],
      ["02", "BP returns approximate marginals.", "It passes Williams–Lau messages without enumerating joint events; O(nm) is per sweep for this construction.", WHITE, BP_DEEP],
      ["03", "PMBM is more than an assignment table.", "Its full update includes PPP-driven birth evidence and Bernoulli existence/state densities alongside global hypotheses.", PM_WASH, PM_DEEP],
      ["04", "TOMB/P is a specific bridge.", "Marginal projection can connect the views; it does not make every BP tracker equivalent to PMBM.", WHITE, PM_DEEP]
    ].flatMap((item, i) => {
      const col = i % 2;
      const row = Math.floor(i / 2);
      const x = 72 + col * 568;
      const y = 220 + row * 164;
      return [
        rect("take-card-" + i, x, y, 536, 140, item[3], { stroke: LINE, strokeWidth: 1, radius: 11 }),
        text("take-num-" + i, x + 24, y + 24, 54, 30, item[0], 18, item[4], { fontWeight: 700 }),
        text("take-head-" + i, x + 86, y + 22, 420, 28, item[1], 21, INK, { fontWeight: 700 }),
        text("take-body-" + i, x + 86, y + 59, 420, 64, item[2], 15, SOFT, { lineHeight: 1.4 })
      ];
    }),
    rule("take-ref-rule", 72, 570, 1136, LINE, 1),
    text("take-refs", 72, 590, 1136, 58,
      "Primary sources · Williams & Lau (2014), Approximate evaluation of marginal association probabilities with BP · Williams (2015), Marginal multi-Bernoulli filters · Meyer et al. (2018), Message passing for scalable MTT · García-Fernández et al. (2018), PMBM filter",
      13, SOFT, { fontFamily: SANS, lineHeight: 1.5, align: "center" })
  ]
));

const doc = {
  format: "bento/slides",
  version: 1,
  docId: "bp-vs-pmbm-data-association-deck",
  title: "BP × PMBM — One Association Problem, Two Philosophies",
  readonly: true,
  meta: {
    author: "Bai Liping",
    subject: "Belief propagation, joint hypotheses, and PMBM data-association context",
    company: "bailiping.com"
  },
  size: { width: 1280, height: 720 },
  theme: { background: PAPER, color: INK, accent: BP_DEEP, fontFamily: SERIF },
  slides
};

const inlineLiveMap = [
  {
    slide: "s-weights",
    slideIndex: slides.findIndex(slide => slide.id === "s-weights"),
    inline: true,
    layout: "region",
    bounds: INLINE_BOUNDS,
    src: "./live/?demo=assignment&embed=region",
    source: "./live/?demo=assignment",
    title: "Shared normalized assignment weights",
    sandbox: "allow-scripts",
    hideSource: true,
    readyMessage: true,
    unloadWhenHidden: true
  },
  {
    slide: "s-bp-messages",
    slideIndex: slides.findIndex(slide => slide.id === "s-bp-messages"),
    inline: true,
    layout: "region",
    bounds: INLINE_BOUNDS,
    src: "./live/?demo=bp&embed=region",
    source: "./live/?demo=bp",
    title: "Williams–Lau BP marginal approximation",
    sandbox: "allow-scripts",
    hideSource: true,
    readyMessage: true,
    unloadWhenHidden: true
  },
  {
    slide: "s-pruning",
    slideIndex: slides.findIndex(slide => slide.id === "s-pruning"),
    inline: true,
    layout: "region",
    bounds: INLINE_BOUNDS,
    src: "./live/?demo=hypotheses&embed=region",
    source: "./live/?demo=hypotheses",
    title: "Joint assignments and top-k pruning",
    sandbox: "allow-scripts",
    hideSource: true,
    readyMessage: true,
    unloadWhenHidden: true
  }
];

const referenceUrl = new URL("../frame-registration-slides/index.html", import.meta.url);
const outputUrl = new URL("./index.html", import.meta.url);
let html = fs.readFileSync(referenceUrl, "utf8");
const serializedDoc = JSON.stringify(doc, null, 1).replace(/</g, "\\u003c");
const serializedConfig = JSON.stringify(inlineLiveMap, null, 2).replace(/</g, "\\u003c");

html = html.replace("<title>bento/slides</title>", "<title>BP × PMBM — data association slides | Bai Liping</title>");
html = html.replace(
  /(<script type="application\/bento\+json" id="bento-doc">\s*)[\s\S]*?(\s*<\/script>)/,
  "$1" + serializedDoc + "$2"
);
html = html.replace(
  /<script type="application\/json" id="(?:bento-live-config|bento-inline-live-map)">[\s\S]*?<\/script>/,
  '<script type="application/json" id="bento-inline-live-map">\n' + serializedConfig + "\n    </script>"
);
html = html.replaceAll("../assets/bento-live.css", "../assets/bento-inline-live.css");
html = html.replaceAll("../assets/bento-live.js", "../assets/bento-inline-live.js");

if (!html.includes('"docId": "bp-vs-pmbm-data-association-deck"')) {
  throw new Error("Bento document replacement failed.");
}
if (!html.includes('id="bento-inline-live-map"') || html.includes('id="bento-live-config"')) {
  throw new Error("Inline-live map replacement failed.");
}
fs.writeFileSync(outputUrl, html);
console.log("Built", slides.length, "regular Bento slides with", inlineLiveMap.length, "inline demos at", outputUrl.pathname);
