(function () {
  "use strict";

  const STEPS = [
    {
      kicker: "INPUT · RGB-D KEYFRAME",
      title: "A new observation anchors the story.",
      body: "The first camera pose receives a strong prior. Its color and depth seed a small set of 3-D Gaussians—the beginning of the map.",
      insightLead: "What changed:",
      insight: "Depth lets us back-project pixels into 3-D; color initializes each Gaussian.",
      equation: "Z₀ = { I₀, D₀ }  →  𝒢₀",
      world: "seed map",
      graph: "one pose + prior",
      pixelTitle: "Image evidence",
      pixelHint: "The observed keyframe initializes color.",
      metricLabel: "MAP SIZE",
      metricValue: "128 splats",
      metricWidth: 28,
      next: "Next: render a guess"
    },
    {
      kicker: "TRACK · MAP FIXED",
      title: "Render what this pose should see.",
      body: "Hold the Gaussian map fixed, choose a candidate camera pose, and rasterize the map from that viewpoint. Drag the pose hypothesis to move the synthetic render.",
      insightLead: "The useful signal:",
      insight: "A wrong pose shifts projected splats, producing structured pixel error instead of a single opaque score.",
      equation: "Îₖ = R(𝒢, Tₖ)",
      world: "pose hypothesis",
      graph: "candidate x₁",
      pixelTitle: "Pose → predicted image",
      pixelHint: "Move the hypothesis; the map itself stays still.",
      metricLabel: "PHOTOMETRIC COST",
      metricValue: "0.148",
      metricWidth: 72,
      next: "Next: make a factor"
    },
    {
      kicker: "LINEARIZE · SE(3)",
      title: "Package the pixels as a factor.",
      body: "Sample RGB residuals and compute their Jacobian with respect to a six-dimensional pose perturbation. That is the contract GTSAM needs.",
      insightLead: "The bridge:",
      insight: "The renderer does not have to understand a Bayes tree; it only returns r and ∂r/∂ξ for the pose node.",
      equation: "rₖ = vec(Îₖ[P] − Iₖ[P]),   Jₖ = ∂rₖ/∂ξ",
      world: "linearize at Tₖ",
      graph: "SplatFactor → x₁",
      pixelTitle: "A vector-valued measurement",
      pixelHint: "Each sampled pixel contributes three residuals: R, G, B.",
      metricLabel: "PHOTOMETRIC COST",
      metricValue: "0.148",
      metricWidth: 72,
      next: "Next: update iSAM2"
    },
    {
      kicker: "INCREMENT · NEW KEYFRAMES",
      title: "Let one graph combine every clue.",
      body: "Each keyframe adds a pose variable. Odometry links neighbors; the SplatFactor pulls a pose toward image alignment; IMU or other standard factors can join unchanged.",
      insightLead: "Why iSAM2:",
      insight: "New evidence updates only the affected Bayes-tree cliques instead of rebuilding the entire trajectory from scratch.",
      equation: "p(X | Z) ∝ φprior · ∏φodom · ∏φsplat · ∏φimu",
      world: "local trajectory",
      graph: "incremental graph",
      pixelTitle: "Rendering is now one sensor",
      pixelHint: "Its factor shares xₖ with odometry and IMU.",
      metricLabel: "ACTIVE POSES",
      metricValue: "5 nodes",
      metricWidth: 55,
      next: "Next: revisit a place"
    },
    {
      kicker: "DETECT · VERIFY · CONNECT",
      title: "The camera returns—but odometry does not.",
      body: "Appearance matching proposes a revisit; geometry estimates the relative pose. The final camera should meet the first, yet accumulated drift leaves a visible gap.",
      insightLead: "A loop is just another factor:",
      insight: "A robust BetweenFactorPose3 connects x₇ to x₀ and says where those two poses must sit relative to one another.",
      equation: "φloop(x₇, x₀) = ρ( ‖ Log(Z₇₀⁻¹ · x₇⁻¹x₀) ‖² )",
      world: "drift exposed",
      graph: "loop candidate",
      pixelTitle: "A familiar view reappears",
      pixelHint: "Appearance proposes; geometry verifies.",
      metricLabel: "REVISIT SCORE",
      metricValue: "0.92 match",
      metricWidth: 92,
      next: "Next: correct globally"
    },
    {
      kicker: "OPTIMIZE · GLOBAL CONSISTENCY",
      title: "One loop moves the whole trajectory.",
      body: "Insert the loop factor. iSAM2 relinearizes where needed, propagates the correction through connected poses, and tightens their marginal uncertainty.",
      insightLead: "This is the payoff:",
      insight: "The photometric factors and the loop constraint act on the same pose variables—there is no second correction pipeline to synchronize.",
      equation: "X* = arg minₓ  Σᵢ ‖rᵢ(X)‖²Σᵢ",
      world: "global correction",
      graph: "re-eliminate affected cliques",
      pixelTitle: "The graph agrees again",
      pixelHint: "Local appearance and global consistency now share X.",
      metricLabel: "LOOP GAP",
      metricValue: "37 cm",
      metricWidth: 82,
      next: "Next: refine the map"
    },
    {
      kicker: "ALTERNATE · POSES ⇄ GAUSSIANS",
      title: "Refine one side while freezing the other.",
      body: "Pose updates happen in GTSAM with the map fixed. Map updates happen in PyTorch with poses fixed. Alternating the two keeps the scene and trajectory mutually consistent.",
      insightLead: "The division of labor:",
      insight: "iSAM2 owns pose belief and covariance; Adam owns Gaussian parameters. The SplatFactor is their narrow, reusable interface.",
      equation: "fix 𝒢 → optimize X   ⇄   fix X → optimize 𝒢",
      world: "map + poses agree",
      graph: "two optimizers, one system",
      pixelTitle: "Render, compare, improve",
      pixelHint: "Alternate until pose and appearance settle.",
      metricLabel: "POSE σ",
      metricValue: "low",
      metricWidth: 24,
      next: "Restart walkthrough"
    }
  ];

  const COLORS = {
    ink: "#132422",
    inkSoft: "#2c403d",
    muted: "#667572",
    paper: "#f1efe7",
    line: "#d6d4ca",
    teal: "#1ca794",
    tealDark: "#0f7166",
    coral: "#ee6e55",
    amber: "#e8a931",
    violet: "#7658df",
    blue: "#3d78e6",
    lime: "#b7d95a",
    graphBg: "#172a28"
  };

  const els = {
    progressFill: document.getElementById("progressFill"),
    labCount: document.getElementById("labCount"),
    stageKicker: document.getElementById("stageKicker"),
    stageTitle: document.getElementById("stageTitle"),
    stageBody: document.getElementById("stageBody"),
    stageInsight: document.getElementById("stageInsight"),
    insightLead: document.getElementById("insightLead"),
    stageEquation: document.getElementById("stageEquation"),
    worldState: document.getElementById("worldState"),
    graphState: document.getElementById("graphState"),
    pixelTitle: document.getElementById("pixelTitle"),
    pixelHint: document.getElementById("pixelHint"),
    metricLabel: document.getElementById("metricLabel"),
    metricValue: document.getElementById("metricValue"),
    metricBar: document.getElementById("metricBar"),
    prevStep: document.getElementById("prevStep"),
    nextStep: document.getElementById("nextStep"),
    playSteps: document.getElementById("playSteps"),
    poseControl: document.getElementById("poseControl"),
    poseError: document.getElementById("poseError"),
    poseValue: document.getElementById("poseValue"),
    optimizePose: document.getElementById("optimizePose"),
    loopControl: document.getElementById("loopControl"),
    closeLoop: document.getElementById("closeLoop"),
    alternateControl: document.getElementById("alternateControl"),
    worldCanvas: document.getElementById("worldCanvas"),
    graphSvg: document.getElementById("graphSvg"),
    observedCanvas: document.getElementById("observedCanvas"),
    renderedCanvas: document.getElementById("renderedCanvas"),
    residualCanvas: document.getElementById("residualCanvas"),
    mixerGraph: document.getElementById("mixerGraph"),
    factorCount: document.getElementById("factorCount"),
    trackingValue: document.getElementById("trackingValue"),
    driftValue: document.getElementById("driftValue"),
    sigmaValue: document.getElementById("sigmaValue")
  };

  const stepButtons = Array.from(document.querySelectorAll(".step-button"));
  const alternateButtons = Array.from(document.querySelectorAll("[data-mode]"));
  const factorInputs = Array.from(document.querySelectorAll("[data-factor]"));

  let currentStep = 0;
  let poseError = 18;
  let loopClosed = false;
  let correctionProgress = 0;
  let alternateMode = "poses";
  let playTimer = null;
  let animationFrame = null;

  const groundPath = [
    { x: 102, y: 304 },
    { x: 220, y: 327 },
    { x: 365, y: 280 },
    { x: 493, y: 165 },
    { x: 400, y: 88 },
    { x: 254, y: 82 },
    { x: 118, y: 151 },
    { x: 105, y: 281 }
  ];

  const driftOffsets = [
    [0, 0], [3, -2], [8, -4], [15, -8], [23, -11],
    [31, -7], [38, 0], [39, 22]
  ];

  const gaussianSeeds = [
    [118, 250, 17, 7, -0.4, COLORS.coral], [150, 224, 10, 23, 0.35, COLORS.teal],
    [205, 270, 23, 8, 0.15, COLORS.amber], [250, 230, 13, 28, -0.7, COLORS.violet],
    [310, 263, 26, 9, -0.15, COLORS.teal], [365, 217, 12, 27, 0.4, COLORS.coral],
    [430, 208, 28, 9, 0.55, COLORS.amber], [454, 140, 11, 26, -0.25, COLORS.violet],
    [381, 127, 27, 9, 0.2, COLORS.teal], [321, 125, 12, 25, 0.8, COLORS.coral],
    [255, 139, 23, 8, -0.25, COLORS.amber], [191, 164, 11, 26, 0.45, COLORS.violet],
    [147, 189, 24, 8, 0.2, COLORS.teal], [544, 250, 18, 7, -0.6, COLORS.coral],
    [535, 113, 10, 23, 0.5, COLORS.teal], [85, 196, 19, 7, 0.4, COLORS.amber],
    [287, 190, 12, 10, 0.2, COLORS.coral], [349, 174, 14, 8, -0.4, COLORS.violet]
  ];

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function lerp(a, b, t) {
    return a + (b - a) * t;
  }

  function eased(t) {
    return 1 - Math.pow(1 - t, 3);
  }

  function getDriftPath(progress) {
    return groundPath.map(function (point, index) {
      const offset = driftOffsets[index];
      return {
        x: point.x + offset[0] * (1 - progress),
        y: point.y + offset[1] * (1 - progress)
      };
    });
  }

  function drawGrid(ctx, width, height, dark) {
    ctx.save();
    ctx.strokeStyle = dark ? "rgba(255,255,255,0.055)" : "rgba(19,36,34,0.07)";
    ctx.lineWidth = 1;
    for (let x = 0.5; x < width; x += 36) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let y = 0.5; y < height; y += 36) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }
    ctx.restore();
  }

  function drawGaussian(ctx, item, alpha, crisp) {
    const x = item[0];
    const y = item[1];
    const rx = item[2] * (crisp ? 0.9 : 1.12);
    const ry = item[3] * (crisp ? 0.9 : 1.12);
    const angle = item[4];
    const color = item[5];
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(angle);
    ctx.globalAlpha = alpha * (crisp ? 0.86 : 0.57);
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.ellipse(0, 0, rx, ry, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalAlpha = alpha * 0.9;
    ctx.fillStyle = "rgba(255,255,255,0.66)";
    ctx.beginPath();
    ctx.ellipse(-rx * 0.2, -ry * 0.12, Math.max(1.7, rx * 0.12), Math.max(1.5, ry * 0.12), 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  function pathStroke(ctx, points, color, width, dashed) {
    if (!points.length) return;
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    if (dashed) ctx.setLineDash(dashed);
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i += 1) ctx.lineTo(points[i].x, points[i].y);
    ctx.stroke();
    ctx.restore();
  }

  function cameraAngle(points, index) {
    const next = points[Math.min(index + 1, points.length - 1)];
    const previous = points[Math.max(0, index - 1)];
    return Math.atan2(next.y - previous.y, next.x - previous.x);
  }

  function drawCamera(ctx, point, angle, label, active, corrected) {
    ctx.save();
    ctx.translate(point.x, point.y);
    ctx.rotate(angle);
    if (active) {
      ctx.fillStyle = corrected ? "rgba(28,167,148,0.12)" : "rgba(238,110,85,0.12)";
      ctx.beginPath();
      ctx.moveTo(2, 0);
      ctx.lineTo(72, -34);
      ctx.lineTo(72, 34);
      ctx.closePath();
      ctx.fill();
      ctx.strokeStyle = corrected ? "rgba(28,167,148,0.45)" : "rgba(238,110,85,0.38)";
      ctx.lineWidth = 1;
      ctx.stroke();
    }
    ctx.fillStyle = active ? (corrected ? COLORS.teal : COLORS.coral) : COLORS.ink;
    ctx.beginPath();
    ctx.moveTo(14, 0);
    ctx.lineTo(-9, -8);
    ctx.lineTo(-5, 0);
    ctx.lineTo(-9, 8);
    ctx.closePath();
    ctx.fill();
    ctx.restore();

    ctx.save();
    ctx.fillStyle = active ? (corrected ? COLORS.tealDark : "#b44735") : COLORS.muted;
    ctx.font = "700 11px " + getComputedStyle(document.documentElement).getPropertyValue("--mono");
    ctx.fillText(label, point.x + 11, point.y - 12);
    ctx.restore();
  }

  function drawUncertainty(ctx, point, radius, color) {
    ctx.save();
    ctx.strokeStyle = color;
    ctx.fillStyle = color.replace("1)", "0.055)");
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.ellipse(point.x, point.y, radius * 1.4, radius, -0.25, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.restore();
  }

  function drawLegend(ctx, items) {
    let x = 20;
    const y = 370;
    ctx.save();
    ctx.font = "700 10px " + getComputedStyle(document.documentElement).getPropertyValue("--mono");
    items.forEach(function (item) {
      ctx.strokeStyle = item[1];
      ctx.lineWidth = 3;
      ctx.setLineDash(item[2] || []);
      ctx.beginPath();
      ctx.moveTo(x, y - 3);
      ctx.lineTo(x + 19, y - 3);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = COLORS.muted;
      ctx.fillText(item[0], x + 25, y);
      x += 25 + ctx.measureText(item[0]).width + 24;
    });
    ctx.restore();
  }

  function drawWorld() {
    const canvas = els.worldCanvas;
    const ctx = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = COLORS.paper;
    ctx.fillRect(0, 0, width, height);
    drawGrid(ctx, width, height, false);

    const initialCount = currentStep === 0 ? 8 : gaussianSeeds.length;
    const crispMap = currentStep === 6 && alternateMode === "map";
    gaussianSeeds.slice(0, initialCount).forEach(function (item, index) {
      drawGaussian(ctx, item, currentStep === 0 && index > 7 ? 0.25 : 1, crispMap);
    });

    ctx.save();
    ctx.fillStyle = "rgba(19,36,34,0.055)";
    ctx.font = "800 58px " + getComputedStyle(document.documentElement).getPropertyValue("--mono");
    ctx.fillText("𝒢", 618, 76);
    ctx.font = "700 10px " + getComputedStyle(document.documentElement).getPropertyValue("--mono");
    ctx.fillStyle = COLORS.muted;
    ctx.fillText("GAUSSIAN MAP", 584, 96);
    ctx.restore();

    let path;
    let count;
    if (currentStep === 0) {
      path = groundPath;
      count = 1;
    } else if (currentStep === 1 || currentStep === 2) {
      path = groundPath.slice(0, 2).map(function (p, i) {
        return i === 1 ? { x: p.x + poseError * 0.75, y: p.y - poseError * 0.32 } : p;
      });
      count = 2;
      pathStroke(ctx, groundPath.slice(0, 2), "rgba(19,36,34,0.22)", 2, [5, 5]);
    } else if (currentStep === 3) {
      path = getDriftPath(0).slice(0, 5);
      count = 5;
      pathStroke(ctx, groundPath.slice(0, 5), "rgba(19,36,34,0.20)", 2, [5, 5]);
    } else if (currentStep === 4) {
      path = getDriftPath(0);
      count = path.length;
      pathStroke(ctx, groundPath, "rgba(19,36,34,0.23)", 2, [5, 5]);
    } else {
      path = getDriftPath(currentStep === 6 ? 0.96 : correctionProgress);
      count = path.length;
      pathStroke(ctx, groundPath, "rgba(19,36,34,0.22)", 2, [5, 5]);
    }

    if (count > 1) {
      const activeColor = currentStep >= 5 && (loopClosed || currentStep === 6) ? COLORS.teal : COLORS.coral;
      pathStroke(ctx, path.slice(0, count), activeColor, 3.2, null);
    }

    if (currentStep === 4 || (currentStep === 5 && !loopClosed)) {
      const first = path[0];
      const last = path[path.length - 1];
      ctx.save();
      ctx.strokeStyle = COLORS.violet;
      ctx.lineWidth = 2;
      ctx.setLineDash([7, 6]);
      ctx.beginPath();
      ctx.moveTo(last.x, last.y);
      ctx.lineTo(first.x, first.y);
      ctx.stroke();
      ctx.fillStyle = COLORS.violet;
      ctx.font = "800 10px " + getComputedStyle(document.documentElement).getPropertyValue("--mono");
      ctx.fillText("LOOP CANDIDATE", first.x + 26, first.y + 5);
      ctx.restore();
    }

    for (let i = 0; i < count; i += 1) {
      const isLast = i === count - 1;
      if (currentStep >= 3) {
        const shrink = currentStep === 6 ? 0.25 : (loopClosed ? 1 - correctionProgress * 0.75 : 1);
        drawUncertainty(ctx, path[i], (5 + i * 2.5) * shrink, "rgba(118,88,223,1)");
      }
      drawCamera(ctx, path[i], cameraAngle(path, i), "x" + i, isLast, currentStep >= 5 && (loopClosed || currentStep === 6));
    }

    if (currentStep === 1 || currentStep === 2) {
      const target = groundPath[1];
      ctx.save();
      ctx.strokeStyle = "rgba(28,167,148,0.7)";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.arc(target.x, target.y, 13, 0, Math.PI * 2);
      ctx.stroke();
      ctx.fillStyle = COLORS.tealDark;
      ctx.font = "700 10px " + getComputedStyle(document.documentElement).getPropertyValue("--mono");
      ctx.fillText("image-aligned pose", target.x + 18, target.y + 4);
      ctx.restore();
    }

    if (currentStep === 6) {
      ctx.save();
      const activeX = alternateMode === "poses" ? 536 : 555;
      ctx.fillStyle = "rgba(255,255,255,0.92)";
      ctx.strokeStyle = alternateMode === "poses" ? COLORS.teal : COLORS.coral;
      ctx.lineWidth = 2;
      roundedRect(ctx, 515, 300, 175, 46, 10);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = COLORS.muted;
      ctx.font = "700 9px " + getComputedStyle(document.documentElement).getPropertyValue("--mono");
      ctx.fillText(alternateMode === "poses" ? "MAP FROZEN" : "POSES FROZEN", activeX, 319);
      ctx.fillStyle = alternateMode === "poses" ? COLORS.tealDark : "#b44735";
      ctx.font = "800 11px " + getComputedStyle(document.documentElement).getPropertyValue("--mono");
      ctx.fillText(alternateMode === "poses" ? "iSAM2 updates X" : "Adam updates 𝒢", activeX, 336);
      ctx.restore();
    }

    if (currentStep >= 3) {
      drawLegend(ctx, [
        ["ground truth", "rgba(19,36,34,0.38)", [5, 5]],
        [currentStep >= 5 && (loopClosed || currentStep === 6) ? "optimized" : "odometry", currentStep >= 5 && (loopClosed || currentStep === 6) ? COLORS.teal : COLORS.coral]
      ]);
    } else {
      drawLegend(ctx, [["camera path", currentStep === 0 ? COLORS.ink : COLORS.coral]]);
    }
  }

  function roundedRect(ctx, x, y, width, height, radius) {
    const r = Math.min(radius, width / 2, height / 2);
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + width, y, x + width, y + height, r);
    ctx.arcTo(x + width, y + height, x, y + height, r);
    ctx.arcTo(x, y + height, x, y, r);
    ctx.arcTo(x, y, x + width, y, r);
    ctx.closePath();
  }

  function svgText(x, y, text, className, anchor) {
    return '<text x="' + x + '" y="' + y + '" class="' + className + '" text-anchor="' + (anchor || "middle") + '">' + text + "</text>";
  }

  function svgPose(x, y, index, options) {
    const active = options && options.active;
    const corrected = options && options.corrected;
    const uncertainty = options && options.uncertainty;
    let out = '<g class="pose-node' + (active ? " active" : "") + (corrected ? " corrected" : "") + '">';
    if (uncertainty) {
      out += '<ellipse class="uncertainty" cx="' + x + '" cy="' + y + '" rx="' + uncertainty * 1.45 + '" ry="' + uncertainty + '"></ellipse>';
    }
    out += '<circle cx="' + x + '" cy="' + y + '" r="18"></circle>';
    out += svgText(x, y + 4, "x" + index, "node-label");
    out += "</g>";
    return out;
  }

  function svgDiamond(x, y, colorClass, label) {
    let out = '<g class="factor-node ' + colorClass + '"><rect x="' + (x - 7) + '" y="' + (y - 7) + '" width="14" height="14" transform="rotate(45 ' + x + " " + y + ')"></rect>';
    if (label) out += svgText(x, y - 16, label, "factor-label");
    out += "</g>";
    return out;
  }

  function graphDefs() {
    return '<defs><pattern id="graphGrid" width="36" height="36" patternUnits="userSpaceOnUse"><path d="M 36 0 L 0 0 0 36" fill="none" stroke="rgba(255,255,255,.045)" stroke-width="1"/></pattern><filter id="glow"><feGaussianBlur stdDeviation="4" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>' +
      '<style>' +
      '.boundary{fill:none;stroke:rgba(255,255,255,.11);stroke-dasharray:6 6}.boundary-label,.tiny-label,.factor-label,.node-label,.map-label,.graph-callout{font-family:SFMono-Regular,Consolas,monospace}.boundary-label{fill:#617773;font-size:9px;font-weight:700;letter-spacing:1.2px}.edge{stroke:#69807b;stroke-width:1.6}.edge.odom{stroke:#53bfae}.edge.splat{stroke:#ed806b;stroke-dasharray:4 4}.edge.loop{stroke:#9b82ee;stroke-width:3}.edge.imu{stroke:#dfb450;stroke-dasharray:2 4}.pose-node circle{fill:#1d3633;stroke:#8ca09c;stroke-width:1.5}.pose-node.active circle{fill:#b54835;stroke:#ffad9d;stroke-width:2}.pose-node.corrected circle{fill:#107468;stroke:#8be0d3;stroke-width:2}.node-label{fill:#e4efed;font-size:11px;font-weight:800}.uncertainty{fill:rgba(118,88,223,.08);stroke:#8269dc;stroke-width:1;stroke-dasharray:3 3}.factor-node rect{stroke-width:1.5}.factor-node.prior rect{fill:#355da9;stroke:#7ba6ff}.factor-node.odom rect{fill:#126e64;stroke:#70d7c8}.factor-node.splat rect{fill:#9d4939;stroke:#ff947e}.factor-node.loop rect{fill:#5940b4;stroke:#b19cff}.factor-label,.tiny-label{fill:#849a95;font-size:8px;font-weight:700}.map-card{fill:#203936;stroke:#50645f}.map-label{fill:#dce9e6;font-size:10px;font-weight:800}.map-sub{fill:#78908b;font-size:8px;font-family:SFMono-Regular,Consolas,monospace}.splat-shape{opacity:.8}.graph-callout{fill:#f0c06b;font-size:9px;font-weight:800}.highlight-ring{fill:none;stroke:#ff947e;stroke-width:2;stroke-dasharray:4 3;filter:url(#glow)}' +
      "</style>";
  }

  function mapCard(x, y, active) {
    return '<g class="map-group' + (active ? " active" : "") + '"><rect class="map-card" x="' + x + '" y="' + y + '" width="154" height="70" rx="12"></rect>' +
      '<ellipse class="splat-shape" cx="' + (x + 29) + '" cy="' + (y + 30) + '" rx="17" ry="6" transform="rotate(-22 ' + (x + 29) + " " + (y + 30) + ')" fill="#1ca794"></ellipse>' +
      '<ellipse class="splat-shape" cx="' + (x + 49) + '" cy="' + (y + 40) + '" rx="7" ry="17" transform="rotate(28 ' + (x + 49) + " " + (y + 40) + ')" fill="#ee6e55"></ellipse>' +
      '<ellipse class="splat-shape" cx="' + (x + 25) + '" cy="' + (y + 48) + '" rx="14" ry="5" transform="rotate(15 ' + (x + 25) + " " + (y + 48) + ')" fill="#7658df"></ellipse>' +
      svgText(x + 68, y + 29, "Gaussian map 𝒢", "map-label", "start") +
      svgText(x + 68, y + 46, "fixed for pose solve", "map-sub", "start") +
      "</g>";
  }

  function graphFrame() {
    return '<rect width="720" height="390" fill="#172a28"></rect><rect width="720" height="390" fill="url(#graphGrid)"></rect>' +
      '<rect class="boundary" x="28" y="38" width="664" height="314" rx="16"></rect>' +
      svgText(45, 58, "GTSAM NONLINEAR FACTOR GRAPH", "boundary-label", "start");
  }

  function renderChainGraph(nodeCount, factorized) {
    let out = graphDefs() + graphFrame();
    const startX = 88;
    const endX = nodeCount <= 2 ? 300 : 500;
    const y = 252;
    const spacing = nodeCount === 1 ? 0 : (endX - startX) / (nodeCount - 1);
    out += mapCard(520, 70, currentStep === 2);

    out += '<line class="edge" x1="88" y1="235" x2="88" y2="145"></line>';
    out += svgDiamond(88, 132, "prior", "PRIOR");

    for (let i = 0; i < nodeCount; i += 1) {
      const x = startX + spacing * i;
      if (i > 0) {
        const previousX = startX + spacing * (i - 1);
        out += '<line class="edge odom" x1="' + (previousX + 18) + '" y1="' + y + '" x2="' + (x - 18) + '" y2="' + y + '"></line>';
        out += svgDiamond((previousX + x) / 2, y, "odom", "ODO");
      }
      const uncertainty = currentStep >= 3 ? 7 + i * 4 : 0;
      out += svgPose(x, y, i, { active: i === nodeCount - 1, uncertainty: uncertainty });

      if (i > 0 && (factorized || currentStep >= 3)) {
        const fx = x + (i % 2 ? 12 : -12);
        const fy = 183 - i * 4;
        out += '<line class="edge splat" x1="' + x + '" y1="' + (y - 18) + '" x2="' + fx + '" y2="' + (fy + 8) + '"></line>';
        out += '<path class="edge splat" d="M ' + fx + ' ' + (fy - 8) + ' Q ' + (fx + 48) + ' 110 520 112"></path>';
        out += svgDiamond(fx, fy, "splat", "SPLAT");
      }
    }

    if (currentStep === 1) {
      out += svgText(300, 306, "candidate pose; not yet image-aligned", "graph-callout");
    }
    if (currentStep === 2) {
      out += '<circle class="highlight-ring" cx="300" cy="183" r="23"></circle>';
      out += svgText(300, 134, "r + J", "graph-callout");
      out += svgText(300, 322, "same x₁ is also constrained by odometry", "tiny-label");
    }
    if (currentStep === 3) {
      out += svgText(365, 329, "only affected Bayes-tree cliques are re-eliminated", "graph-callout");
    }
    return out;
  }

  function renderLoopGraph() {
    const points = [
      [110, 272], [225, 304], [360, 270], [454, 154],
      [370, 95], [255, 90], [142, 145], [111, 222]
    ];
    let out = graphDefs() + graphFrame();
    out += mapCard(520, 70, false);

    for (let i = 0; i < points.length; i += 1) {
      const p = points[i];
      if (i > 0) {
        const previous = points[i - 1];
        out += '<line class="edge odom" x1="' + previous[0] + '" y1="' + previous[1] + '" x2="' + p[0] + '" y2="' + p[1] + '"></line>';
        out += svgDiamond((previous[0] + p[0]) / 2, (previous[1] + p[1]) / 2, "odom", "");
      }
      const corrected = currentStep === 5 && loopClosed;
      const uncertainty = Math.max(3, (7 + i * 2.2) * (corrected ? 1 - correctionProgress * 0.72 : 1));
      out += svgPose(p[0], p[1], i, { active: i === points.length - 1, corrected: corrected && i > 0, uncertainty: uncertainty });
      if (i === 2 || i === 4 || i === 6 || i === 7) {
        const factorX = p[0] + 24;
        const factorY = p[1] - 36;
        out += '<line class="edge splat" x1="' + p[0] + '" y1="' + p[1] + '" x2="' + factorX + '" y2="' + factorY + '"></line>';
        out += svgDiamond(factorX, factorY, "splat", "");
      }
    }

    out += '<line class="edge" x1="110" y1="254" x2="110" y2="195"></line>' + svgDiamond(110, 182, "prior", "PRIOR");

    const loopVisible = currentStep === 4 || currentStep === 5;
    if (loopVisible) {
      out += '<path class="edge loop" d="M 108 257 Q 63 225 111 222"></path>';
      out += svgDiamond(76, 238, "loop", currentStep === 5 && loopClosed ? "LOOP ✓" : "LOOP?");
      out += svgText(245, 339, currentStep === 5 && loopClosed ? "one loop factor tightens the connected trajectory" : "x₇ should meet x₀, but drift leaves a gap", "graph-callout");
    }
    return out;
  }

  function renderAlternatingGraph() {
    const posesActive = alternateMode === "poses";
    let out = graphDefs() + graphFrame();
    out += '<rect x="64" y="92" width="260" height="210" rx="16" fill="' + (posesActive ? "#1d3a36" : "#1b312f") + '" stroke="' + (posesActive ? "#70d7c8" : "#425b56") + '" stroke-width="2"></rect>';
    out += svgText(194, 121, posesActive ? "GTSAM ACTIVE" : "GTSAM FROZEN", "graph-callout");
    for (let i = 0; i < 4; i += 1) {
      const x = 103 + i * 58;
      if (i > 0) out += '<line class="edge odom" x1="' + (x - 40) + '" y1="205" x2="' + (x - 18) + '" y2="205"></line>';
      out += svgPose(x, 205, i, { corrected: posesActive, uncertainty: posesActive ? 5 : 10 });
    }
    out += svgText(194, 270, posesActive ? "optimize X with iSAM2" : "hold X fixed", "tiny-label");

    out += '<rect x="396" y="92" width="260" height="210" rx="16" fill="' + (!posesActive ? "#3a2c2a" : "#1b312f") + '" stroke="' + (!posesActive ? "#ff947e" : "#425b56") + '" stroke-width="2"></rect>';
    out += svgText(526, 121, !posesActive ? "PYTORCH ACTIVE" : "PYTORCH FROZEN", "graph-callout");
    const splatPositions = [[456, 181, 26, 8, -18, "#1ca794"], [514, 218, 13, 31, 28, "#ee6e55"], [564, 172, 29, 9, 16, "#e8a931"], [598, 227, 14, 34, -24, "#7658df"], [462, 244, 22, 7, 25, "#7658df"]];
    splatPositions.forEach(function (s) {
      out += '<ellipse cx="' + s[0] + '" cy="' + s[1] + '" rx="' + s[2] + '" ry="' + s[3] + '" transform="rotate(' + s[4] + " " + s[0] + " " + s[1] + ')" fill="' + s[5] + '" opacity=".82"></ellipse>';
    });
    out += svgText(526, 270, !posesActive ? "optimize 𝒢 with Adam" : "hold 𝒢 fixed", "tiny-label");
    out += '<path d="M 324 164 C 353 134 368 134 396 164" fill="none" stroke="#ff947e" stroke-width="2" marker-end="url(#none)"></path>';
    out += '<path d="M 396 240 C 368 270 353 270 324 240" fill="none" stroke="#70d7c8" stroke-width="2"></path>';
    out += svgText(360, 196, "r, J", "graph-callout");
    out += svgText(360, 215, "⇄", "graph-callout");
    out += svgText(360, 336, "alternate—not one giant joint update", "boundary-label");
    return out;
  }

  function drawGraph() {
    let markup;
    if (currentStep === 0) markup = renderChainGraph(1, false);
    else if (currentStep === 1) markup = renderChainGraph(2, false);
    else if (currentStep === 2) markup = renderChainGraph(2, true);
    else if (currentStep === 3) markup = renderChainGraph(5, true);
    else if (currentStep === 4 || currentStep === 5) markup = renderLoopGraph();
    else markup = renderAlternatingGraph();
    els.graphSvg.innerHTML = markup;
  }

  function drawSceneBase(ctx, shiftX, shiftY, alpha, mapVariant) {
    const width = ctx.canvas.width;
    const height = ctx.canvas.height;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#172724";
    ctx.fillRect(0, 0, width, height);

    ctx.save();
    ctx.translate(shiftX, shiftY);
    ctx.globalAlpha = alpha;
    ctx.fillStyle = "#243d39";
    ctx.beginPath();
    ctx.moveTo(0, 102);
    ctx.lineTo(74, 68);
    ctx.lineTo(194, 68);
    ctx.lineTo(260, 101);
    ctx.lineTo(260, 136);
    ctx.lineTo(0, 136);
    ctx.closePath();
    ctx.fill();
    ctx.fillStyle = "#31534d";
    ctx.fillRect(30, 33, 61, 68);
    ctx.fillStyle = "#99c7b4";
    ctx.fillRect(38, 42, 45, 44);
    ctx.fillStyle = "#1d322f";
    ctx.fillRect(113, 47, 96, 56);

    const sceneSplats = [
      [50, 74, 25, 10, -0.2, "#1ca794"], [73, 54, 11, 27, 0.5, "#e8a931"],
      [127, 91, 35, 11, 0.1, "#ee6e55"], [157, 71, 15, 31, -0.6, "#7658df"],
      [194, 90, 28, 9, -0.3, "#1ca794"], [220, 58, 11, 25, 0.25, "#e8a931"],
      [102, 116, 32, 10, 0.15, "#7658df"], [207, 117, 37, 11, -0.18, "#ee6e55"]
    ];
    sceneSplats.forEach(function (s, index) {
      const scale = mapVariant && index % 2 ? 0.82 : 1;
      ctx.save();
      ctx.translate(s[0], s[1]);
      ctx.rotate(s[4]);
      ctx.globalAlpha = 0.72;
      ctx.fillStyle = s[5];
      ctx.beginPath();
      ctx.ellipse(0, 0, s[2] * scale, s[3] * scale, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    });
    ctx.restore();

    ctx.fillStyle = "rgba(255,255,255,0.72)";
    ctx.font = "700 8px SFMono-Regular,Consolas,monospace";
    ctx.fillText("SYNTHETIC KEYFRAME", 9, 13);
  }

  function renderShiftForStep() {
    if (currentStep === 0) return 0;
    if (currentStep === 1 || currentStep === 2) return poseError * 0.9;
    if (currentStep === 4) return 11;
    if (currentStep === 5) return loopClosed ? 11 * (1 - correctionProgress) : 11;
    return currentStep === 6 && alternateMode === "map" ? 1.6 : 1;
  }

  function drawResidual(ctx, shift) {
    const width = ctx.canvas.width;
    const height = ctx.canvas.height;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#172724";
    ctx.fillRect(0, 0, width, height);
    const magnitude = clamp(Math.abs(shift) / 27, 0.035, 1);

    drawGrid(ctx, width, height, true);
    const hotspots = [[49, 73, 27, 14], [126, 91, 39, 16], [159, 70, 19, 34], [205, 113, 41, 17], [220, 58, 15, 29]];
    hotspots.forEach(function (spot, index) {
      ctx.save();
      ctx.globalAlpha = magnitude * (0.48 + index * 0.07);
      ctx.fillStyle = index % 2 ? COLORS.coral : COLORS.amber;
      ctx.beginPath();
      ctx.ellipse(spot[0] + shift * 0.18, spot[1], spot[2] * magnitude + 2, spot[3] * magnitude + 1, index * 0.25, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    });

    ctx.fillStyle = "rgba(255,255,255,0.72)";
    ctx.font = "700 8px SFMono-Regular,Consolas,monospace";
    ctx.fillText("MEAN |r|  " + (0.006 + magnitude * 0.142).toFixed(3), 9, 13);

    if (currentStep === 0) {
      ctx.fillStyle = "rgba(23,39,36,0.82)";
      ctx.fillRect(0, 0, width, height);
      ctx.fillStyle = "#829892";
      ctx.font = "700 9px SFMono-Regular,Consolas,monospace";
      ctx.textAlign = "center";
      ctx.fillText("NO RESIDUAL YET", width / 2, height / 2 + 3);
      ctx.textAlign = "start";
    }
  }

  function drawPixels() {
    const observedCtx = els.observedCanvas.getContext("2d");
    const renderedCtx = els.renderedCanvas.getContext("2d");
    const residualCtx = els.residualCanvas.getContext("2d");
    const shift = renderShiftForStep();
    drawSceneBase(observedCtx, 0, 0, 1, false);
    drawSceneBase(renderedCtx, shift, -shift * 0.18, currentStep === 0 ? 0.35 : 1, currentStep === 6 && alternateMode === "map");
    drawResidual(residualCtx, shift);

    if (currentStep === 0) {
      renderedCtx.fillStyle = "rgba(23,39,36,0.78)";
      renderedCtx.fillRect(0, 0, renderedCtx.canvas.width, renderedCtx.canvas.height);
      renderedCtx.fillStyle = "#829892";
      renderedCtx.font = "700 9px SFMono-Regular,Consolas,monospace";
      renderedCtx.textAlign = "center";
      renderedCtx.fillText("MAP INITIALIZING", renderedCtx.canvas.width / 2, renderedCtx.canvas.height / 2 + 3);
      renderedCtx.textAlign = "start";
    }
  }

  function updatePoseReadout() {
    const sign = poseError > 0 ? "+" : "";
    els.poseValue.textContent = sign + Math.round(poseError) + " px";
    const normalized = (Number(els.poseError.value) + 30) / 60 * 100;
    els.poseError.style.background = "linear-gradient(90deg, #d5d7d1 0 " + Math.min(normalized, 50) + "%, #1ca794 " + Math.min(normalized, 50) + "% " + Math.max(normalized, 50) + "%, #d5d7d1 " + Math.max(normalized, 50) + "% 100%)";
    if (currentStep === 1 || currentStep === 2) {
      const cost = 0.008 + Math.pow(Math.abs(poseError) / 30, 1.35) * 0.224;
      els.metricValue.textContent = cost.toFixed(3);
      els.metricBar.style.width = clamp(cost / 0.232 * 100, 4, 100) + "%";
      els.metricBar.style.background = Math.abs(poseError) < 5 ? COLORS.teal : COLORS.coral;
    }
  }

  function updateLoopReadout() {
    if (currentStep !== 5) return;
    const gap = 37 * (1 - correctionProgress) + 2.4 * correctionProgress;
    els.metricValue.textContent = gap.toFixed(gap < 10 ? 1 : 0) + " cm";
    els.metricBar.style.width = clamp(gap / 37 * 82, 7, 82) + "%";
    els.metricBar.style.background = correctionProgress > 0.65 ? COLORS.teal : COLORS.coral;
    els.worldState.textContent = correctionProgress >= 0.98 ? "loop closed" : "global correction";
    els.graphState.textContent = correctionProgress >= 0.98 ? "posterior updated" : "re-eliminate affected cliques";
  }

  function renderAll() {
    drawWorld();
    drawGraph();
    drawPixels();
    updatePoseReadout();
    updateLoopReadout();
  }

  function updateStageCopy() {
    const step = STEPS[currentStep];
    els.stageKicker.textContent = step.kicker;
    els.stageTitle.textContent = step.title;
    els.stageBody.textContent = step.body;
    els.insightLead.textContent = step.insightLead;
    els.stageInsight.textContent = step.insight;
    els.stageEquation.textContent = step.equation;
    els.worldState.textContent = step.world;
    els.graphState.textContent = step.graph;
    els.pixelTitle.textContent = step.pixelTitle;
    els.pixelHint.textContent = step.pixelHint;
    els.metricLabel.textContent = step.metricLabel;
    els.metricValue.textContent = step.metricValue;
    els.metricBar.style.width = step.metricWidth + "%";
    els.metricBar.style.background = currentStep === 4 ? COLORS.violet : (currentStep === 5 ? COLORS.coral : COLORS.teal);
    els.labCount.textContent = String(currentStep + 1).padStart(2, "0") + " / 07";
    els.progressFill.style.width = ((currentStep + 1) / STEPS.length * 100) + "%";

    stepButtons.forEach(function (button, index) {
      button.classList.toggle("is-active", index === currentStep);
      button.classList.toggle("is-complete", index < currentStep);
      if (index === currentStep) button.setAttribute("aria-current", "step");
      else button.removeAttribute("aria-current");
    });

    els.prevStep.disabled = currentStep === 0;
    els.nextStep.innerHTML = (currentStep === STEPS.length - 1 ? "Restart walkthrough" : step.next) + ' <span aria-hidden="true">→</span>';
    els.poseControl.hidden = !(currentStep === 1 || currentStep === 2);
    els.loopControl.hidden = currentStep !== 5;
    els.alternateControl.hidden = currentStep !== 6;

    if (currentStep === 5) {
      els.closeLoop.disabled = loopClosed;
      els.closeLoop.textContent = loopClosed ? "Loop factor inserted ✓" : "Insert loop factor";
    }

    if (currentStep === 6) updateAlternateReadout();
  }

  function setStep(nextIndex, source) {
    if (animationFrame) {
      cancelAnimationFrame(animationFrame);
      animationFrame = null;
    }
    if (loopClosed && correctionProgress < 1) correctionProgress = 1;
    currentStep = clamp(nextIndex, 0, STEPS.length - 1);
    if (currentStep === 1 && source === "next" && Math.abs(poseError) < 4) {
      poseError = 18;
      els.poseError.value = String(poseError);
    }
    updateStageCopy();
    renderAll();
    if (source === "rail") stopPlaying();
    if (playTimer && currentStep === 5 && !loopClosed) {
      window.setTimeout(function () {
        if (currentStep === 5 && playTimer) insertLoop();
      }, 650);
    }
  }

  function animatePoseOptimization() {
    if (animationFrame) cancelAnimationFrame(animationFrame);
    const startError = poseError;
    const targetError = startError * 0.18;
    const start = performance.now();
    const duration = 720;
    els.optimizePose.disabled = true;
    els.optimizePose.textContent = "Linearizing…";

    function tick(now) {
      const t = clamp((now - start) / duration, 0, 1);
      poseError = lerp(startError, targetError, eased(t));
      els.poseError.value = String(poseError);
      renderAll();
      if (t < 1) {
        animationFrame = requestAnimationFrame(tick);
      } else {
        animationFrame = null;
        els.optimizePose.disabled = false;
        els.optimizePose.innerHTML = 'Run another LM step <span aria-hidden="true">→</span>';
      }
    }
    animationFrame = requestAnimationFrame(tick);
  }

  function insertLoop() {
    if (loopClosed) return;
    loopClosed = true;
    correctionProgress = 0;
    els.closeLoop.disabled = true;
    els.closeLoop.textContent = "Updating Bayes tree…";
    const start = performance.now();
    const duration = 1350;

    function tick(now) {
      const t = clamp((now - start) / duration, 0, 1);
      correctionProgress = eased(t);
      renderAll();
      if (t < 1) {
        animationFrame = requestAnimationFrame(tick);
      } else {
        animationFrame = null;
        els.closeLoop.textContent = "Loop factor inserted ✓";
        updateLoopReadout();
      }
    }
    animationFrame = requestAnimationFrame(tick);
  }

  function updateAlternateReadout() {
    const posesActive = alternateMode === "poses";
    els.metricLabel.textContent = posesActive ? "POSE σ" : "MAP LOSS";
    els.metricValue.textContent = posesActive ? "low" : "decreasing";
    els.metricBar.style.width = posesActive ? "24%" : "38%";
    els.metricBar.style.background = posesActive ? COLORS.teal : COLORS.coral;
    els.worldState.textContent = posesActive ? "map frozen" : "poses frozen";
    els.graphState.textContent = posesActive ? "iSAM2 active" : "Adam active";
  }

  function startPlaying() {
    if (playTimer) return;
    if (currentStep === STEPS.length - 1) {
      loopClosed = false;
      correctionProgress = 0;
      poseError = 18;
      els.poseError.value = "18";
      setStep(0, "play");
    }
    els.playSteps.setAttribute("aria-pressed", "true");
    els.playSteps.innerHTML = '<span class="play-icon" aria-hidden="true">Ⅱ</span> Pause';
    if (currentStep === 5 && !loopClosed) {
      window.setTimeout(function () {
        if (currentStep === 5 && playTimer) insertLoop();
      }, 400);
    }
    playTimer = window.setInterval(function () {
      if (currentStep >= STEPS.length - 1) {
        stopPlaying();
      } else {
        setStep(currentStep + 1, "play");
      }
    }, 3300);
  }

  function stopPlaying() {
    if (playTimer) window.clearInterval(playTimer);
    playTimer = null;
    els.playSteps.setAttribute("aria-pressed", "false");
    els.playSteps.innerHTML = '<span class="play-icon" aria-hidden="true">▶</span> Play steps';
  }

  function togglePlaying() {
    if (playTimer) stopPlaying();
    else startPlaying();
  }

  function mixerDefs() {
    return '<defs><pattern id="mixerGrid" width="36" height="36" patternUnits="userSpaceOnUse"><path d="M36 0H0V36" fill="none" stroke="rgba(255,255,255,.04)"/></pattern></defs><style>' +
      '.mix-pose circle{fill:#1b3431;stroke:#829994;stroke-width:1.7}.mix-pose text,.mix-label,.mix-map text{font-family:SFMono-Regular,Consolas,monospace}.mix-pose text{fill:#e0ebe8;font-size:10px;font-weight:800}.mix-edge{fill:none;stroke-width:2}.mix-edge.odom{stroke:#70d7c8}.mix-edge.splat{stroke:#ff947e;stroke-dasharray:4 4}.mix-edge.imu{stroke:#eac365;stroke-dasharray:3 4}.mix-edge.loop{stroke:#a890ff;stroke-width:3}.mix-factor{stroke-width:1.5}.mix-factor.odom{fill:#126e64;stroke:#70d7c8}.mix-factor.splat{fill:#9d4939;stroke:#ff947e}.mix-factor.prior{fill:#355da9;stroke:#7ba6ff}.mix-label{fill:#78908b;font-size:8px;font-weight:700}.mix-map rect{fill:#203936;stroke:#536b66}.mix-map text{fill:#dce9e6;font-size:10px;font-weight:800}' +
      "</style>";
  }

  function renderMixer() {
    const enabled = {};
    factorInputs.forEach(function (input) { enabled[input.dataset.factor] = input.checked; });
    const points = [[83, 244], [188, 283], [305, 264], [418, 211], [492, 129], [596, 181]];
    let out = mixerDefs() + '<rect width="720" height="360" fill="#10211f"></rect><rect width="720" height="360" fill="url(#mixerGrid)"></rect>';

    if (enabled.odom) {
      for (let i = 1; i < points.length; i += 1) {
        const a = points[i - 1];
        const b = points[i];
        out += '<line class="mix-edge odom" x1="' + a[0] + '" y1="' + a[1] + '" x2="' + b[0] + '" y2="' + b[1] + '"></line>';
        const mx = (a[0] + b[0]) / 2;
        const my = (a[1] + b[1]) / 2;
        out += '<rect class="mix-factor odom" x="' + (mx - 6) + '" y="' + (my - 6) + '" width="12" height="12" transform="rotate(45 ' + mx + " " + my + ')"></rect>';
      }
    }

    if (enabled.imu) {
      out += '<path class="mix-edge imu" d="M 83 228 Q 188 166 305 248"></path><path class="mix-edge imu" d="M 305 248 Q 430 121 596 165"></path>';
      out += svgText(350, 142, "IMU PREINTEGRATION", "mix-label");
    }

    if (enabled.loop) {
      out += '<path class="mix-edge loop" d="M 82 229 Q 280 25 595 166"></path>';
      out += svgText(332, 56, "LOOP CLOSURE", "mix-label");
      out += '<rect x="328" y="66" width="14" height="14" transform="rotate(45 335 73)" fill="#5940b4" stroke="#a890ff" stroke-width="1.5"></rect>';
    }

    if (enabled.splat) {
      out += '<g class="mix-map"><rect x="277" y="76" width="166" height="48" rx="10"></rect>' + svgText(360, 97, "Gaussian map 𝒢", "", "middle") + svgText(360, 113, "fixed measurement model", "mix-label", "middle") + "</g>";
      [1, 3, 5].forEach(function (index) {
        const p = points[index];
        const fx = p[0] - 8;
        const fy = p[1] - 45;
        out += '<path class="mix-edge splat" d="M ' + p[0] + " " + (p[1] - 18) + " L " + fx + " " + (fy + 8) + '"></path>';
        out += '<rect class="mix-factor splat" x="' + (fx - 6) + '" y="' + (fy - 6) + '" width="12" height="12" transform="rotate(45 ' + fx + " " + fy + ')"></rect>';
      });
    }

    out += '<line x1="83" y1="226" x2="83" y2="178" stroke="#7ba6ff" stroke-width="1.7"></line><rect class="mix-factor prior" x="77" y="160" width="12" height="12" transform="rotate(45 83 166)"></rect>';

    points.forEach(function (p, index) {
      out += '<g class="mix-pose"><circle cx="' + p[0] + '" cy="' + p[1] + '" r="18"></circle>' + svgText(p[0], p[1] + 4, "x" + index, "", "middle") + "</g>";
    });
    out += svgText(84, 326, "diamond = factor", "mix-label", "start") + svgText(636, 326, "circle = pose", "mix-label", "end");
    els.mixerGraph.innerHTML = out;

    let count = 1;
    if (enabled.odom) count += 5;
    if (enabled.splat) count += 3;
    if (enabled.imu) count += 2;
    if (enabled.loop) count += 1;
    els.factorCount.textContent = count + (count === 1 ? " factor" : " factors");

    els.trackingValue.textContent = enabled.splat ? "strong" : (enabled.odom || enabled.imu ? "relative only" : "anchored only");
    els.driftValue.textContent = enabled.loop ? "closed" : (enabled.imu ? "slower" : (enabled.odom ? "accumulating" : "unobserved"));
    els.sigmaValue.textContent = enabled.loop ? "low" : (enabled.splat && enabled.imu ? "medium–low" : (enabled.splat || enabled.imu ? "medium" : "high"));
  }

  stepButtons.forEach(function (button) {
    button.addEventListener("click", function () {
      setStep(Number(button.dataset.step), "rail");
    });
  });

  els.prevStep.addEventListener("click", function () {
    stopPlaying();
    setStep(currentStep - 1, "previous");
  });

  els.nextStep.addEventListener("click", function () {
    stopPlaying();
    if (currentStep === STEPS.length - 1) {
      loopClosed = false;
      correctionProgress = 0;
      poseError = 18;
      els.poseError.value = "18";
      alternateMode = "poses";
      alternateButtons.forEach(function (button) { button.classList.toggle("is-active", button.dataset.mode === "poses"); });
      setStep(0, "restart");
    } else {
      setStep(currentStep + 1, "next");
    }
  });

  els.playSteps.addEventListener("click", togglePlaying);

  els.poseError.addEventListener("input", function () {
    stopPlaying();
    poseError = Number(els.poseError.value);
    renderAll();
  });

  els.optimizePose.addEventListener("click", function () {
    stopPlaying();
    animatePoseOptimization();
  });

  els.closeLoop.addEventListener("click", function () {
    stopPlaying();
    insertLoop();
  });

  alternateButtons.forEach(function (button) {
    button.addEventListener("click", function () {
      alternateMode = button.dataset.mode;
      alternateButtons.forEach(function (candidate) {
        candidate.classList.toggle("is-active", candidate === button);
      });
      updateAlternateReadout();
      renderAll();
    });
  });

  factorInputs.forEach(function (input) {
    input.addEventListener("change", renderMixer);
  });

  document.addEventListener("visibilitychange", function () {
    if (document.hidden) stopPlaying();
  });

  updateStageCopy();
  renderAll();
  renderMixer();
})();
