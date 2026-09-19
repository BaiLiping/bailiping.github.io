import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const templatePath = path.resolve(here, "../frame-registration-slides/index.html");
const outputPath = path.join(here, "index.html");

const PAPER = "#FBFAF7";
const WHITE = "#FFFFFF";
const INK = "#202628";
const MUTED = "#69736F";
const ACCENT = "#2E7564";
const ACCENT_DARK = "#245A4E";
const SLAM_FILL = "#E5EFEA";
const SOFT_LINE = "#7D8984";
const SERIF = "Georgia, 'Times New Roman', serif";
const SANS = "ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";

function text(id, x, y, w, h, html, fontSize, options = {}) {
  return {
    id,
    type: "text",
    x,
    y,
    w,
    h,
    rotation: options.rotation ?? 0,
    opacity: options.opacity ?? 1,
    html,
    fontSize,
    fontFamily: options.fontFamily ?? SERIF,
    fontWeight: options.fontWeight ?? 400,
    color: options.color ?? INK,
    align: options.align ?? "left",
    valign: "top",
    lineHeight: options.lineHeight ?? 1.18,
    ...(options.letterSpacing !== undefined ? { letterSpacing: options.letterSpacing } : {}),
    ...(options.role ? { role: options.role } : {}),
    ...(options.morphId ? { morphId: options.morphId } : {}),
    ...(options.fx ? { fx: options.fx } : {})
  };
}

function rect(id, x, y, w, h, options = {}) {
  return {
    id,
    type: "shape",
    shape: "rect",
    x,
    y,
    w,
    h,
    fill: options.fill ?? "transparent",
    stroke: options.stroke ?? "none",
    strokeWidth: options.strokeWidth ?? 0,
    radius: options.radius ?? 0,
    rotation: options.rotation ?? 0,
    opacity: options.opacity ?? 1,
    ...(options.strokeStyle ? { strokeStyle: options.strokeStyle } : {}),
    ...(options.morphId ? { morphId: options.morphId } : {}),
    ...(options.fx ? { fx: options.fx } : {})
  };
}

function ellipse(id, x, y, w, h, options = {}) {
  return {
    id,
    type: "shape",
    shape: "ellipse",
    x,
    y,
    w,
    h,
    fill: options.fill ?? "transparent",
    stroke: options.stroke ?? INK,
    strokeWidth: options.strokeWidth ?? 2,
    radius: 0,
    rotation: 0,
    opacity: options.opacity ?? 1,
    ...(options.fx ? { fx: options.fx } : {})
  };
}

function wire(id, x, y, w, h, color = INK, opacity = 1) {
  return rect(id, x, y, w, h, { fill: color, opacity });
}

function arrowhead(id, direction, x, y, color = INK, size = 18) {
  const glyph = { right: "▶", left: "◀", up: "▲", down: "▼" }[direction];
  return text(id, x, y, size + 6, size + 8, glyph, size, {
    fontFamily: SANS,
    fontWeight: 700,
    color,
    align: "center",
    lineHeight: 1
  });
}

function graphEdge(id, x1, y1, x2, y2, options = {}) {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const length = Math.hypot(dx, dy);
  const angle = Math.atan2(dy, dx) * 180 / Math.PI;
  const thickness = options.thickness ?? 2;
  const color = options.color ?? INK;

  if (!options.dashed) {
    return [rect(id, (x1 + x2) / 2 - length / 2, (y1 + y2) / 2 - thickness / 2, length, thickness, {
      fill: color,
      rotation: angle,
      radius: thickness / 2,
      opacity: options.opacity ?? 1
    })];
  }

  const dash = options.dash ?? 9;
  const gap = options.gap ?? 7;
  const ux = dx / length;
  const uy = dy / length;
  const pieces = [];
  let offset = 0;
  let index = 0;
  while (offset < length) {
    const segmentLength = Math.min(dash, length - offset);
    const centerDistance = offset + segmentLength / 2;
    const cx = x1 + ux * centerDistance;
    const cy = y1 + uy * centerDistance;
    pieces.push(rect(`${id}-dash-${index}`, cx - segmentLength / 2, cy - thickness / 2, segmentLength, thickness, {
      fill: color,
      rotation: angle,
      radius: thickness / 2,
      opacity: options.opacity ?? 1
    }));
    offset += dash + gap;
    index += 1;
  }
  return pieces;
}

function graphVariable(id, cx, cy, label, options = {}) {
  const radius = options.radius ?? 23;
  return [
    ellipse(`${id}-node`, cx - radius, cy - radius, radius * 2, radius * 2, {
      fill: WHITE,
      stroke: INK,
      strokeWidth: 2.1,
      fx: options.fx
    }),
    text(`${id}-label`, cx - radius, cy - 13, radius * 2, 28, label, options.fontSize ?? 21, {
      fontWeight: 500,
      align: "center",
      lineHeight: 1,
      fx: options.fx
    })
  ];
}

function graphFactor(id, cx, cy, options = {}) {
  const radius = options.radius ?? 6;
  return ellipse(id, cx - radius, cy - radius, radius * 2, radius * 2, {
    fill: INK,
    stroke: INK,
    strokeWidth: 1,
    fx: options.fx
  });
}

function buildFactorGraph(prefix, variables, edges, priors, order) {
  const edgeElements = [];
  const factorElements = [];
  const nodeElements = [];
  const fx = { enter: "fade-up", order };

  for (const edge of edges) {
    const a = variables[edge.a];
    const b = variables[edge.b];
    edgeElements.push(...graphEdge(`${prefix}-${edge.id}`, a.x, a.y, b.x, b.y, {
      dashed: edge.dashed,
      thickness: edge.thickness ?? 2,
      color: edge.color ?? INK,
      opacity: edge.opacity ?? 1
    }));
    const t = edge.t ?? 0.5;
    factorElements.push(graphFactor(`${prefix}-${edge.id}-factor`, a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, { fx }));
  }

  for (const prior of priors) {
    const target = variables[prior.target];
    edgeElements.push(...graphEdge(`${prefix}-${prior.id}`, prior.x, prior.y, target.x, target.y, {
      thickness: 2,
      color: INK
    }));
    factorElements.push(graphFactor(`${prefix}-${prior.id}-factor`, prior.x, prior.y, { fx }));
  }

  for (const [key, variable] of Object.entries(variables)) {
    nodeElements.push(...graphVariable(`${prefix}-${key}`, variable.x, variable.y, variable.label, {
      radius: variable.radius,
      fontSize: variable.fontSize,
      fx
    }));
  }

  return [...edgeElements, ...factorElements, ...nodeElements];
}

const slide1Elements = [
  text("section-kicker", 72, 28, 520, 22, "SECTION 1 · AUTONOMY ARCHITECTURE", 12, {
    fontFamily: SANS,
    fontWeight: 800,
    color: ACCENT,
    letterSpacing: 2.1,
    role: "kicker",
    morphId: "header-kicker",
    fx: { enter: "fade-up", order: 0 }
  }),
  text("slide-title", 72, 52, 1090, 50, "The Role of SLAM in the Autonomy Architecture", 35, {
    fontWeight: 700,
    lineHeight: 1.08,
    role: "title",
    morphId: "header-title",
    fx: { enter: "fade-up", order: 1 }
  }),
  text("slide-number", 1165, 33, 44, 26, "01", 14, {
    fontFamily: SANS,
    fontWeight: 800,
    color: MUTED,
    align: "right",
    letterSpacing: 1.2,
    morphId: "header-number"
  }),
  rect("title-rule", 72, 104, 1136, 2, { fill: "#D9DEDB", morphId: "header-rule" }),

  // SLAM system boundary: intentionally animated as the diagram's quiet focal cue.
  rect("slam-boundary", 338, 280, 690, 372, {
    fill: "transparent",
    stroke: SOFT_LINE,
    strokeWidth: 2,
    strokeStyle: "dashed",
    opacity: 0.86,
    fx: { loop: { type: "dash-march", distance: 14, duration: 5 } }
  }),

  // Top-level autonomy loop wiring.
  wire("wire-goal", 80, 191, 102, 3),
  arrowhead("arrow-goal", "right", 170, 181),
  wire("wire-plan", 380, 191, 88, 3),
  arrowhead("arrow-plan", "right", 456, 181),
  wire("wire-error", 525, 191, 69, 3),
  arrowhead("arrow-error", "right", 582, 181),
  wire("wire-control", 800, 191, 124, 3),
  arrowhead("arrow-control", "right", 912, 181),

  // Robot/environment sensor loop and branch to both SLAM front-ends.
  wire("sensor-top", 1146, 191, 49, 3),
  wire("sensor-down", 1192, 191, 3, 186),
  wire("sensor-odometry", 935, 374, 260, 3),
  arrowhead("sensor-odometry-head", "left", 925, 364),
  wire("sensor-branch", 1004, 374, 3, 182),
  wire("sensor-loop-closure", 935, 553, 72, 3),
  arrowhead("sensor-loop-head", "left", 925, 543),

  // Fast odometry feedback to the motion controller.
  wire("odometry-horizontal", 500, 374, 220, 3),
  wire("odometry-up", 500, 207, 3, 170),
  arrowhead("odometry-head", "up", 490, 203),

  // Globally corrected map and robot state feedback to planning.
  wire("map-horizontal", 290, 473, 100, 3),
  wire("map-up", 290, 239, 3, 237),
  arrowhead("map-head", "up", 280, 231),

  // Bidirectional front-end/back-end exchanges.
  wire("backend-odom-link", 590, 409, 120, 3, SOFT_LINE),
  arrowhead("backend-odom-left", "left", 582, 399, SOFT_LINE, 17),
  arrowhead("backend-odom-right", "right", 699, 399, SOFT_LINE, 17),
  wire("backend-loop-link", 590, 509, 120, 3, SOFT_LINE),
  arrowhead("backend-loop-left", "left", 582, 499, SOFT_LINE, 17),
  arrowhead("backend-loop-right", "right", 699, 499, SOFT_LINE, 17),

  // Main autonomy modules.
  rect("motion-planning-box", 190, 142, 190, 100, {
    fill: WHITE,
    stroke: INK,
    strokeWidth: 2.5,
    fx: { enter: "fade-up", order: 2 }
  }),
  text("motion-planning-label", 207, 164, 156, 62, "motion<br>planning", 25, {
    fontWeight: 500,
    align: "center",
    lineHeight: 1.08,
    fx: { enter: "fade-up", order: 2 }
  }),
  ellipse("sum-node", 475, 167, 50, 50, {
    fill: WHITE,
    stroke: INK,
    strokeWidth: 2.5,
    fx: { enter: "fade-up", order: 3 }
  }),
  text("sum-plus", 486, 170, 28, 22, "+", 22, { align: "center", lineHeight: 1 }),
  text("sum-minus", 486, 194, 28, 22, "−", 24, { align: "center", lineHeight: 1 }),
  rect("motion-controller-box", 600, 142, 200, 100, {
    fill: WHITE,
    stroke: INK,
    strokeWidth: 2.5,
    fx: { enter: "fade-up", order: 4 }
  }),
  text("motion-controller-label", 616, 164, 168, 62, "motion<br>controller", 25, {
    fontWeight: 500,
    align: "center",
    lineHeight: 1.08,
    fx: { enter: "fade-up", order: 4 }
  }),
  rect("system-box", 930, 142, 216, 100, {
    fill: WHITE,
    stroke: INK,
    strokeWidth: 2.5,
    fx: { enter: "fade-up", order: 5 }
  }),
  text("system-label", 946, 154, 184, 78, "system:<br>robot and<br>environment", 22, {
    fontWeight: 500,
    align: "center",
    lineHeight: 1.12,
    fx: { enter: "fade-up", order: 5 }
  }),

  // SLAM modules.
  rect("slam-backend-box", 390, 375, 200, 160, {
    fill: SLAM_FILL,
    stroke: ACCENT_DARK,
    strokeWidth: 2.5,
    fx: { enter: "fade-up", order: 6 }
  }),
  text("slam-backend-label", 408, 430, 164, 74, "SLAM<br>back-end", 25, {
    fontWeight: 600,
    align: "center",
    lineHeight: 1.1,
    fx: { enter: "fade-up", order: 6 }
  }),
  rect("slam-odometry-box", 720, 320, 215, 110, {
    fill: SLAM_FILL,
    stroke: ACCENT_DARK,
    strokeWidth: 2.5,
    fx: { enter: "fade-up", order: 7 }
  }),
  text("slam-odometry-label", 737, 337, 181, 82, "SLAM<br>front-end:<br>odometry", 22, {
    fontWeight: 600,
    align: "center",
    lineHeight: 1.12,
    fx: { enter: "fade-up", order: 7 }
  }),
  rect("slam-loop-box", 720, 500, 215, 110, {
    fill: SLAM_FILL,
    stroke: ACCENT_DARK,
    strokeWidth: 2.5,
    fx: { enter: "fade-up", order: 8 }
  }),
  text("slam-loop-label", 736, 517, 183, 82, "SLAM<br>front-end:<br>loop closures", 22, {
    fontWeight: 600,
    align: "center",
    lineHeight: 1.12,
    fx: { enter: "fade-up", order: 8 }
  }),

  // Flow labels.
  text("motion-goal-label", 66, 139, 108, 56, "motion<br>goal", 20, {
    align: "center",
    lineHeight: 1.14
  }),
  text("motion-plan-label", 384, 126, 88, 58, "motion<br>plan", 20, {
    align: "center",
    lineHeight: 1.12
  }),
  text("error-label", 532, 150, 64, 28, "error", 20, { align: "center" }),
  text("control-inputs-label", 810, 126, 108, 58, "control<br>inputs", 20, {
    align: "center",
    lineHeight: 1.12
  }),
  text("sensor-data-label", 1060, 300, 112, 58, "sensor<br>data", 20, {
    align: "center",
    lineHeight: 1.12
  }),
  text("odometry-label", 514, 328, 118, 30, "odometry", 20, { align: "left" }),
  text("map-state-label", 140, 425, 136, 64, "map and<br>robot state", 20, {
    align: "center",
    lineHeight: 1.15
  }),
  text("slam-boundary-label", 354, 610, 132, 34, "SLAM", 27, {
    fontWeight: 600,
    color: ACCENT,
    lineHeight: 1
  }),

  text("figure-caption", 72, 674, 1060, 22,
    "SLAM turns sensor data into odometry, loop closures, and a globally consistent map and robot state for control and planning.",
    13.5,
    { fontFamily: SANS, color: MUTED, lineHeight: 1.2 }
  ),
  text("footer-mark", 1167, 673, 41, 22, "SLAM", 11.5, {
    fontFamily: SANS,
    fontWeight: 800,
    color: ACCENT,
    align: "right",
    letterSpacing: 1.3,
    morphId: "footer-mark"
  })
];

const landmarkSlamVariables = {
  p1: { x: 160, y: 246, label: "<i>p</i><sub>1</sub>" },
  p2: { x: 300, y: 246, label: "<i>p</i><sub>2</sub>" },
  p3: { x: 400, y: 190, label: "<i>p</i><sub>3</sub>" },
  p4: { x: 530, y: 205, label: "<i>p</i><sub>4</sub>" },
  l1: { x: 210, y: 143, label: "ℓ<sub>1</sub>" },
  l2: { x: 455, y: 135, label: "ℓ<sub>2</sub>" },
  l3: { x: 440, y: 290, label: "ℓ<sub>3</sub>" }
};

const landmarkObservationEdges = [
  { id: "l1-p1", a: "l1", b: "p1", dashed: true },
  { id: "l1-p2", a: "l1", b: "p2", dashed: true },
  { id: "l1-p3", a: "l1", b: "p3", dashed: true },
  { id: "l2-p3", a: "l2", b: "p3", dashed: true },
  { id: "l2-p4", a: "l2", b: "p4", dashed: true },
  { id: "l3-p2", a: "l3", b: "p2", dashed: true },
  { id: "l3-p4", a: "l3", b: "p4", dashed: true }
];

const canonicalSlamEdges = [
  ...landmarkObservationEdges,
  { id: "p1-p2", a: "p1", b: "p2" },
  { id: "p2-p3", a: "p2", b: "p3" },
  { id: "p3-p4", a: "p3", b: "p4" }
];

const bundleAdjustmentVariables = {
  p1: { x: 760, y: 246, label: "<i>p</i><sub>1</sub>" },
  p2: { x: 895, y: 246, label: "<i>p</i><sub>2</sub>" },
  p3: { x: 1000, y: 190, label: "<i>p</i><sub>3</sub>" },
  p4: { x: 1125, y: 205, label: "<i>p</i><sub>4</sub>" },
  l1: { x: 805, y: 143, label: "ℓ<sub>1</sub>" },
  l2: { x: 1055, y: 135, label: "ℓ<sub>2</sub>" },
  l3: { x: 1040, y: 290, label: "ℓ<sub>3</sub>" }
};

const bundleAdjustmentEdges = [
  { id: "l1-p1", a: "l1", b: "p1", dashed: true },
  { id: "l1-p2", a: "l1", b: "p2", dashed: true },
  { id: "l1-p3", a: "l1", b: "p3", dashed: true },
  { id: "l2-p3", a: "l2", b: "p3", dashed: true },
  { id: "l2-p4", a: "l2", b: "p4", dashed: true },
  { id: "l3-p2", a: "l3", b: "p2", dashed: true },
  { id: "l3-p4", a: "l3", b: "p4", dashed: true }
];

const poseGraphVariables = {
  p1: { x: 160, y: 475, label: "<i>p</i><sub>1</sub>", radius: 21, fontSize: 19 },
  p2: { x: 295, y: 475, label: "<i>p</i><sub>2</sub>", radius: 21, fontSize: 19 },
  p3: { x: 395, y: 418, label: "<i>p</i><sub>3</sub>", radius: 21, fontSize: 19 },
  p4: { x: 515, y: 425, label: "<i>p</i><sub>4</sub>", radius: 21, fontSize: 19 },
  p5: { x: 515, y: 500, label: "<i>p</i><sub>5</sub>", radius: 21, fontSize: 19 },
  p6: { x: 405, y: 550, label: "<i>p</i><sub>6</sub>", radius: 21, fontSize: 19 },
  p7: { x: 280, y: 545, label: "<i>p</i><sub>7</sub>", radius: 21, fontSize: 19 },
  p8: { x: 160, y: 565, label: "<i>p</i><sub>8</sub>", radius: 21, fontSize: 19 }
};

const poseGraphEdges = [
  { id: "p1-p2", a: "p1", b: "p2" },
  { id: "p2-p3", a: "p2", b: "p3" },
  { id: "p3-p4", a: "p3", b: "p4" },
  { id: "p4-p5", a: "p4", b: "p5" },
  { id: "p5-p6", a: "p5", b: "p6" },
  { id: "p6-p7", a: "p6", b: "p7" },
  { id: "p7-p8", a: "p7", b: "p8" },
  { id: "loop-p1-p8", a: "p1", b: "p8", dashed: true },
  { id: "loop-p2-p6", a: "p2", b: "p6", dashed: true }
];

const steamVariables = {
  x1: { x: 760, y: 500, label: "<i>x</i><sub>1</sub>", radius: 21, fontSize: 19 },
  x2: { x: 895, y: 500, label: "<i>x</i><sub>2</sub>", radius: 21, fontSize: 19 },
  x3: { x: 1000, y: 445, label: "<i>x</i><sub>3</sub>", radius: 21, fontSize: 19 },
  x4: { x: 1125, y: 455, label: "<i>x</i><sub>4</sub>", radius: 21, fontSize: 19 },
  l1: { x: 805, y: 408, label: "ℓ<sub>1</sub>", radius: 21, fontSize: 19 },
  l2: { x: 1055, y: 395, label: "ℓ<sub>2</sub>", radius: 21, fontSize: 19 },
  l3: { x: 1045, y: 565, label: "ℓ<sub>3</sub>", radius: 21, fontSize: 19 }
};

const steamEdges = [
  { id: "l1-x1", a: "l1", b: "x1", dashed: true },
  { id: "l1-x2", a: "l1", b: "x2", dashed: true },
  { id: "l1-x3", a: "l1", b: "x3", dashed: true },
  { id: "l2-x3", a: "l2", b: "x3", dashed: true },
  { id: "l2-x4", a: "l2", b: "x4", dashed: true },
  { id: "l3-x2", a: "l3", b: "x2", dashed: true },
  { id: "l3-x4", a: "l3", b: "x4", dashed: true },
  { id: "x1-x2", a: "x1", b: "x2" },
  { id: "x2-x3", a: "x2", b: "x3" },
  { id: "x3-x4", a: "x3", b: "x4" }
];

const slide2Elements = [
  text("section-kicker", 72, 28, 520, 22, "SECTION 1 · FACTOR-GRAPH MODELS", 12, {
    fontFamily: SANS,
    fontWeight: 800,
    color: ACCENT,
    letterSpacing: 2.1,
    role: "kicker",
    morphId: "header-kicker"
  }),
  text("slide-title", 72, 52, 1090, 50, "Four Factor-Graph Views of SLAM", 35, {
    fontWeight: 700,
    lineHeight: 1.08,
    role: "title",
    morphId: "header-title"
  }),
  text("slide-number", 1165, 33, 44, 26, "02", 14, {
    fontFamily: SANS,
    fontWeight: 800,
    color: MUTED,
    align: "right",
    letterSpacing: 1.2,
    morphId: "header-number"
  }),
  rect("title-rule", 72, 104, 1136, 2, { fill: "#D9DEDB", morphId: "header-rule" }),

  ...buildFactorGraph("canonical", landmarkSlamVariables, canonicalSlamEdges, [
    { id: "prior", target: "p1", x: 92, y: 246 }
  ], 2),
  text("canonical-odometry-arrow", 219, 255, 24, 28, "↑", 24, { align: "center", lineHeight: 1 }),
  text("canonical-odometry-label", 181, 282, 100, 22, "odometry", 15.5, { align: "center", color: MUTED }),
  text("canonical-panel-letter", 76, 329, 24, 24, "A", 13, {
    fontFamily: SANS,
    fontWeight: 900,
    color: ACCENT,
    align: "center"
  }),
  text("canonical-caption", 100, 326, 500, 28, "canonical landmark-based SLAM", 19, {
    align: "center",
    fontWeight: 500,
    fx: { enter: "fade-up", order: 2 }
  }),

  ...buildFactorGraph("ba", bundleAdjustmentVariables, bundleAdjustmentEdges, [
    { id: "prior", target: "p1", x: 692, y: 246 }
  ], 3),
  text("ba-no-prior-arrow", 819, 253, 24, 30, "↑", 24, { align: "center", lineHeight: 1 }),
  text("ba-no-prior-label", 789, 281, 108, 42, "no motion<br>prior", 15.5, {
    align: "center",
    color: MUTED,
    lineHeight: 1.08
  }),
  text("ba-panel-letter", 662, 329, 24, 24, "B", 13, {
    fontFamily: SANS,
    fontWeight: 900,
    color: ACCENT,
    align: "center"
  }),
  text("ba-caption", 690, 326, 500, 52, "bundle adjustment (BA)<br><span style=\"color:#69736F\">structure from motion</span>", 18.5, {
    align: "center",
    fontWeight: 500,
    lineHeight: 1.18,
    fx: { enter: "fade-up", order: 3 }
  }),

  ...buildFactorGraph("pgo", poseGraphVariables, poseGraphEdges, [
    { id: "prior", target: "p1", x: 92, y: 475 }
  ], 4),
  text("pgo-loop-label", 420, 472, 112, 42, "loop<br>closure", 15, {
    align: "center",
    color: MUTED,
    lineHeight: 1.08
  }),
  text("pgo-loop-arrow", 380, 496, 34, 30, "↙", 25, { align: "center", lineHeight: 1 }),
  text("pgo-odometry-arrow", 210, 554, 24, 28, "↑", 23, { align: "center", lineHeight: 1 }),
  text("pgo-odometry-label", 176, 581, 94, 22, "odometry", 15, { align: "center", color: MUTED }),
  text("pgo-panel-letter", 76, 613, 24, 24, "C", 13, {
    fontFamily: SANS,
    fontWeight: 900,
    color: ACCENT,
    align: "center"
  }),
  text("pgo-caption", 100, 607, 500, 54, "pose-graph optimization (PGO)<br><span style=\"color:#69736F\">pose-graph SLAM</span>", 18.5, {
    align: "center",
    fontWeight: 500,
    lineHeight: 1.18,
    fx: { enter: "fade-up", order: 4 }
  }),

  ...buildFactorGraph("steam", steamVariables, steamEdges, [
    { id: "prior", target: "x1", x: 692, y: 500 }
  ], 5),
  text("steam-state-equation", 666, 552, 126, 26, "<i>x</i><sub>i</sub> = [<i>p</i><sub>i</sub>, <i>v</i><sub>i</sub>]ᵀ", 15.5, {
    align: "center",
    color: INK,
    lineHeight: 1
  }),
  text("steam-motion-arrow", 824, 520, 24, 28, "↑", 23, { align: "center", lineHeight: 1 }),
  text("steam-motion-label", 805, 550, 125, 45, "continuous-time<br>motion prior", 15, {
    align: "center",
    color: MUTED,
    lineHeight: 1.08
  }),
  text("steam-panel-letter", 662, 613, 24, 24, "D", 13, {
    fontFamily: SANS,
    fontWeight: 900,
    color: ACCENT,
    align: "center"
  }),
  text("steam-caption", 690, 607, 500, 54, "simultaneous trajectory estimation<br>and mapping (STEAM)", 18.5, {
    align: "center",
    fontWeight: 500,
    lineHeight: 1.18,
    fx: { enter: "fade-up", order: 5 }
  }),

  text("factor-graph-legend", 72, 681, 1030, 18,
    "White circles: variables · black dots: factors · solid edges: motion/process priors · dashed edges: landmark observations",
    11.5,
    { fontFamily: SANS, color: MUTED, lineHeight: 1.1 }
  ),
  text("footer-mark", 1167, 680, 41, 20, "SLAM", 11.5, {
    fontFamily: SANS,
    fontWeight: 800,
    color: ACCENT,
    align: "right",
    letterSpacing: 1.3,
    morphId: "footer-mark"
  })
];

const doc = {
  format: "bento/slides",
  version: 1,
  docId: "slam-autonomy-architecture-deck",
  title: "SLAM — Autonomy Architecture",
  readonly: true,
  meta: {
    author: "Liping Bai",
    subject: "Simultaneous localization and mapping",
    company: "bailiping.com"
  },
  size: { width: 1280, height: 720 },
  theme: {
    background: PAPER,
    color: INK,
    accent: ACCENT,
    fontFamily: SERIF
  },
  slides: [
    {
      id: "slam-autonomy-architecture",
      background: PAPER,
      transition: "none",
      notes: "SLAM sits inside the robot autonomy loop. Sensor data feeds a fast front-end for odometry and a second front-end for loop closures. The back-end fuses those constraints into a consistent map and robot state. Fast odometry closes the inner control loop; the globally corrected state and map feed motion planning.",
      elements: slide1Elements
    },
    {
      id: "slam-factor-graph-views",
      background: PAPER,
      transition: "morph",
      notes: "The same estimation problem can be organized with different state and process assumptions. Canonical landmark SLAM combines odometry and landmark observations. Bundle adjustment removes the motion prior and relies on image-to-landmark constraints. Pose-graph SLAM marginalizes landmarks and adds loop-closure constraints between poses. STEAM augments the state with trajectory derivatives and uses a continuous-time motion prior while retaining landmark observations.",
      elements: slide2Elements
    }
  ]
};

const template = await readFile(templatePath, "utf8");
const docJson = JSON.stringify(doc, null, 1).replaceAll("<", "\\u003c");
const withTitle = template.replace(
  /<title>[\s\S]*?<\/title>/,
  "<title>SLAM — Autonomy Architecture | Bai Liping</title>"
);
const withDoc = withTitle.replace(
  /(<script type="application\/bento\+json" id="bento-doc">)[\s\S]*?(<\/script>)/,
  `$1\n${docJson}\n$2`
);

if (withDoc === withTitle) {
  throw new Error("Could not locate the Bento document block in the template.");
}

// The source runtime may belong to a deck with inline demos. This deck is
// fully native Bento content, so inherited live-demo hooks must not survive.
const output = withDoc
  .replace(/\s*<link[^>]+bento-inline-live\.css[^>]*>/g, "")
  .replace(/\s*<script type="application\/json" id="bento-inline-live-map">[\s\S]*?<\/script>/g, "")
  .replace(/\s*<script[^>]+bento-inline-live\.js[^>]*><\/script>/g, "");

await writeFile(outputPath, output, "utf8");
console.log(`Wrote ${outputPath}`);
