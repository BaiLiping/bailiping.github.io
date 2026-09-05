(function () {
  "use strict";

  var TAU = Math.PI * 2;
  var DEG = 180 / Math.PI;
  var WORLD = { xmin: -6.4, xmax: 6.4, ymin: -3.9, ymax: 3.9 };
  var COLORS = {
    paper: "#f8f4eb",
    grid: "#ddd6ca",
    ink: "#20232b",
    muted: "#756f65",
    accent: "#8a2d1c",
    coral: "#ff9e8a",
    source: "#486a9a",
    sourceSoft: "#9bb2cf",
    success: "#39705a",
    truth: "#857d71"
  };
  var truth = { th: rad(14), tx: 1.15, ty: -0.65 };

  var canvas = document.getElementById("registration-canvas");
  var ctx = canvas.getContext("2d");
  var statusEl = document.getElementById("status");
  var hintEl = document.getElementById("canvas-hint");
  var stageHeading = document.getElementById("stage-heading");
  var metricA = document.getElementById("metric-a");
  var metricB = document.getElementById("metric-b");
  var metricPose = document.getElementById("metric-pose");
  var metricALabel = document.getElementById("metric-a-label");
  var metricBLabel = document.getElementById("metric-b-label");
  var methodKicker = document.getElementById("method-kicker");
  var methodTitle = document.getElementById("method-title");
  var methodCopy = document.getElementById("method-copy");
  var outlierRange = document.getElementById("outlier-range");
  var gateRange = document.getElementById("gate-range");
  var angleRange = document.getElementById("angle-range");
  var outlierValue = document.getElementById("outlier-value");
  var gateValue = document.getElementById("gate-value");
  var angleValue = document.getElementById("angle-value");
  var showMatches = document.getElementById("show-matches");

  var copy = {
    ransac: {
      kicker: "GLOBAL SEARCH",
      title: "Find consensus among outliers",
      body: "Each step proposes rigid transforms from compatible point pairs. The pose with the most geometric support survives.",
      heading: "PAIR-CONGRUENCE SEARCH",
      hint: "Pair segments show the latest proposal; green points support the best pose so far."
    },
    icp: {
      kicker: "LOCAL REFINEMENT",
      title: "Alternate matching and fitting",
      body: "ICP chooses nearest neighbours, solves the least-squares rigid fit, and repeats. Its basin of attraction is the lesson.",
      heading: "NEAREST NEIGHBOURS + KABSCH FIT",
      hint: "Drag to translate · Shift-drag to rotate · Focus the canvas and use arrows to nudge."
    },
    ndt: {
      kicker: "DISTRIBUTION MATCHING",
      title: "Navigate a likelihood surface",
      body: "NDT replaces discrete target points with Gaussian cells. Registration becomes optimization over a smooth-ish score landscape.",
      heading: "NORMAL DISTRIBUTIONS TRANSFORM",
      hint: "The bright ridges are high-likelihood translations at the selected rotation; click anywhere to move the pose."
    }
  };

  var requestedMode = new URLSearchParams(window.location.search).get("demo");
  if (!Object.prototype.hasOwnProperty.call(copy, requestedMode)) requestedMode = "ransac";
  var mode = "ransac";
  var scene = makeScene(Number(outlierRange.value));
  var ransac = null;
  var icp = null;
  var ndt = null;
  var drag = null;
  var lastLayout = null;
  var timerId = 0;
  var runIntent = false;
  var externallyPaused = false;

  function rad(degrees) {
    return degrees * Math.PI / 180;
  }

  function wrapAngle(value) {
    return Math.atan2(Math.sin(value), Math.cos(value));
  }

  function mulberry32(seed) {
    return function () {
      seed |= 0;
      seed = seed + 0x6D2B79F5 | 0;
      var t = Math.imul(seed ^ seed >>> 15, 1 | seed);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }

  function gaussian(random) {
    var u = 0;
    var v = 0;
    while (u === 0) u = random();
    while (v === 0) v = random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(TAU * v);
  }

  function cloneT(T) {
    return { th: T.th, tx: T.tx, ty: T.ty };
  }

  function applyT(T, p) {
    var c = Math.cos(T.th);
    var s = Math.sin(T.th);
    return [c * p[0] - s * p[1] + T.tx, s * p[0] + c * p[1] + T.ty];
  }

  function inverseApply(T, p) {
    var x = p[0] - T.tx;
    var y = p[1] - T.ty;
    var c = Math.cos(T.th);
    var s = Math.sin(T.th);
    return [c * x + s * y, -s * x + c * y];
  }

  function composeT(A, B) {
    var c = Math.cos(A.th);
    var s = Math.sin(A.th);
    return {
      th: wrapAngle(A.th + B.th),
      tx: c * B.tx - s * B.ty + A.tx,
      ty: s * B.tx + c * B.ty + A.ty
    };
  }

  function transformError(T) {
    return {
      angle: Math.abs(wrapAngle(T.th - truth.th)) * DEG,
      translation: Math.hypot(T.tx - truth.tx, T.ty - truth.ty)
    };
  }

  function makeScene(outlierPercent) {
    var random = mulberry32(73019 + outlierPercent * 97);
    var base = [];
    var i;
    for (i = 0; i < 45; i += 1) {
      var floorX = -4.85 + i * 9.5 / 44;
      base.push([floorX, -2.25 + 0.07 * Math.sin(floorX * 1.7)]);
    }
    for (i = 1; i < 31; i += 1) {
      var wallY = -2.25 + i * 5.35 / 30;
      base.push([-4.48 + 0.05 * Math.sin(wallY * 2.1), wallY]);
    }
    for (i = 0; i < 34; i += 1) {
      var roofX = -3.95 + i * 7.7 / 33;
      base.push([roofX, 2.52 - 0.13 * roofX + 0.08 * Math.cos(roofX * 2.4)]);
    }
    for (i = 0; i < 26; i += 1) {
      var a = -1.3 + i * 2.55 / 25;
      base.push([1.35 + 1.48 * Math.cos(a), 0.03 + 1.12 * Math.sin(a)]);
    }

    var target = base.map(function (p) {
      return [p[0] + gaussian(random) * 0.018, p[1] + gaussian(random) * 0.018];
    });
    var source = [];
    base.forEach(function (p, index) {
      if (index % 7 === 3) return;
      var local = inverseApply(truth, p);
      source.push({
        p: [local[0] + gaussian(random) * 0.025, local[1] + gaussian(random) * 0.025],
        outlier: false
      });
    });
    var goodCount = source.length;
    var requested = Math.round(goodCount * outlierPercent / Math.max(1, 100 - outlierPercent));
    for (i = 0; i < requested; i += 1) {
      source.push({
        p: [-5.8 + random() * 11.6, -3.45 + random() * 6.9],
        outlier: true
      });
    }
    return {
      target: target,
      source: source,
      goodCount: goodCount,
      centroid: centroid(source.slice(0, goodCount).map(function (item) { return item.p; }))
    };
  }

  function centroid(points) {
    var x = 0;
    var y = 0;
    points.forEach(function (p) {
      x += p[0];
      y += p[1];
    });
    return [x / points.length, y / points.length];
  }

  function nearest(point, targets) {
    var bestIndex = -1;
    var bestSquared = Infinity;
    for (var i = 0; i < targets.length; i += 1) {
      var dx = point[0] - targets[i][0];
      var dy = point[1] - targets[i][1];
      var squared = dx * dx + dy * dy;
      if (squared < bestSquared) {
        bestSquared = squared;
        bestIndex = i;
      }
    }
    return { index: bestIndex, distance: Math.sqrt(bestSquared) };
  }

  function weightedKabsch(P, Q) {
    var px = 0;
    var py = 0;
    var qx = 0;
    var qy = 0;
    var i;
    for (i = 0; i < P.length; i += 1) {
      px += P[i][0];
      py += P[i][1];
      qx += Q[i][0];
      qy += Q[i][1];
    }
    px /= P.length;
    py /= P.length;
    qx /= Q.length;
    qy /= Q.length;
    var cosineTerm = 0;
    var sineTerm = 0;
    for (i = 0; i < P.length; i += 1) {
      var ax = P[i][0] - px;
      var ay = P[i][1] - py;
      var bx = Q[i][0] - qx;
      var by = Q[i][1] - qy;
      cosineTerm += ax * bx + ay * by;
      sineTerm += ax * by - ay * bx;
    }
    var th = Math.atan2(sineTerm, cosineTerm);
    var c = Math.cos(th);
    var s = Math.sin(th);
    return {
      th: th,
      tx: qx - (c * px - s * py),
      ty: qy - (s * px + c * py)
    };
  }

  function fitTwoPoints(p1, p2, q1, q2) {
    var th = Math.atan2(q2[1] - q1[1], q2[0] - q1[0]) -
      Math.atan2(p2[1] - p1[1], p2[0] - p1[0]);
    var c = Math.cos(th);
    var s = Math.sin(th);
    return {
      th: wrapAngle(th),
      tx: q1[0] - (c * p1[0] - s * p1[1]),
      ty: q1[1] - (s * p1[0] + c * p1[1])
    };
  }

  function scoreConsensus(T, gate) {
    var matches = [];
    scene.source.forEach(function (item, index) {
      var moved = applyT(T, item.p);
      var match = nearest(moved, scene.target);
      if (match.distance < gate) {
        matches.push({ source: index, target: match.index, distance: match.distance });
      }
    });
    return matches;
  }

  function targetPairCatalog() {
    var pairs = [];
    for (var i = 0; i < scene.target.length; i += 1) {
      for (var j = i + 1; j < scene.target.length; j += 1) {
        var distance = Math.hypot(
          scene.target[j][0] - scene.target[i][0],
          scene.target[j][1] - scene.target[i][1]
        );
        if (distance > 1.15) pairs.push({ a: i, b: j, distance: distance });
      }
    }
    return pairs;
  }

  function resetRansac() {
    ransac = {
      T: { th: 0, tx: 0, ty: 0 },
      bestT: null,
      bestCount: 0,
      inliers: [],
      hypotheses: 0,
      random: mulberry32(99173 + Number(outlierRange.value)),
      targetPairs: targetPairCatalog(),
      lastProposal: null,
      done: false
    };
    if (mode === "ransac") {
      setStatus("Ready — test the first batch", false);
      updateMetrics();
      render();
    }
  }

  function polishRansac(T) {
    var current = cloneT(T);
    for (var round = 0; round < 2; round += 1) {
      var matches = scoreConsensus(current, 0.25);
      if (matches.length < 3) break;
      var P = matches.map(function (match) { return scene.source[match.source].p; });
      var Q = matches.map(function (match) { return scene.target[match.target]; });
      current = weightedKabsch(P, Q);
    }
    return current;
  }

  function stepRansac() {
    if (ransac.done) return true;
    var improved = false;
    var tested = 0;
    while (tested < 36) {
      tested += 1;
      ransac.hypotheses += 1;
      var i1 = Math.floor(ransac.random() * scene.source.length);
      var i2 = Math.floor(ransac.random() * scene.source.length);
      var p1 = scene.source[i1].p;
      var p2 = scene.source[i2].p;
      var sourceDistance = Math.hypot(p2[0] - p1[0], p2[1] - p1[1]);
      if (i1 === i2 || sourceDistance < 1.15) continue;

      var compatible = [];
      for (var k = 0; k < ransac.targetPairs.length; k += 1) {
        if (Math.abs(ransac.targetPairs[k].distance - sourceDistance) < 0.065) {
          compatible.push(ransac.targetPairs[k]);
        }
      }
      if (!compatible.length) continue;
      var pair = compatible[Math.floor(ransac.random() * compatible.length)];
      var flip = ransac.random() < 0.5;
      var q1 = scene.target[flip ? pair.b : pair.a];
      var q2 = scene.target[flip ? pair.a : pair.b];
      var hypothesis = fitTwoPoints(p1, p2, q1, q2);
      var support = scoreConsensus(hypothesis, 0.25);
      ransac.lastProposal = {
        source: [p1, p2],
        target: [q1, q2],
        T: hypothesis,
        count: support.length
      };
      if (support.length > ransac.bestCount) {
        ransac.bestT = polishRansac(hypothesis);
        ransac.inliers = scoreConsensus(ransac.bestT, 0.25);
        ransac.bestCount = ransac.inliers.length;
        ransac.T = cloneT(ransac.bestT);
        improved = true;
      }
    }
    var error = transformError(ransac.T);
    ransac.done = ransac.bestCount > scene.goodCount * 0.72 &&
      error.angle < 1.25 && error.translation < 0.09;
    if (ransac.done) {
      setStatus("Consensus found — outliers rejected", false);
    } else if (improved) {
      setStatus("New best: " + ransac.bestCount + " supporting points", true);
    } else {
      setStatus("No stronger consensus in this batch", true);
    }
    updateMetrics();
    render();
    return ransac.done;
  }

  function resetIcp() {
    icp = {
      T: { th: rad(5), tx: 0.38, ty: 0.03 },
      iterations: 0,
      pairs: [],
      rms: null,
      done: false
    };
    icp.pairs = computeIcpPairs(icp.T);
    if (mode === "icp") {
      setStatus("Drag the blue scan into the attraction basin", false);
      updateMetrics();
      render();
    }
  }

  function computeIcpPairs(T) {
    var gate = Number(gateRange.value) / 100;
    var pairs = [];
    scene.source.forEach(function (item, index) {
      var moved = applyT(T, item.p);
      var match = nearest(moved, scene.target);
      if (match.distance < gate) {
        pairs.push({
          source: index,
          target: match.index,
          moved: moved,
          distance: match.distance
        });
      }
    });
    pairs.sort(function (a, b) { return a.distance - b.distance; });
    var keep = Math.max(3, Math.floor(pairs.length * 0.84));
    return pairs.slice(0, keep);
  }

  function stepIcp() {
    if (icp.done) return true;
    var pairs = computeIcpPairs(icp.T);
    if (pairs.length < 3) {
      icp.done = true;
      setStatus("Too few matches — drag the scan closer", false);
      updateMetrics();
      render();
      return true;
    }
    var P = pairs.map(function (pair) { return pair.moved; });
    var Q = pairs.map(function (pair) { return scene.target[pair.target]; });
    var correction = weightedKabsch(P, Q);
    icp.T = composeT(correction, icp.T);
    icp.iterations += 1;
    icp.pairs = computeIcpPairs(icp.T);
    var squareSum = icp.pairs.reduce(function (sum, pair) {
      return sum + pair.distance * pair.distance;
    }, 0);
    var nextRms = Math.sqrt(squareSum / Math.max(1, icp.pairs.length));
    var change = icp.rms === null ? Infinity : Math.abs(icp.rms - nextRms);
    icp.rms = nextRms;
    icp.done = change < 0.00008 || icp.iterations >= 35;
    var error = transformError(icp.T);
    if (icp.done && error.translation < 0.12 && error.angle < 1.8) {
      setStatus("Converged inside the correct basin", false);
    } else if (icp.done) {
      setStatus("Settled in a local minimum — reset and reposition", false);
    } else {
      setStatus("Re-matched " + icp.pairs.length + " pairs, then refit", true);
    }
    updateMetrics();
    render();
    return icp.done;
  }

  function buildNdtModel(points, cellSize) {
    var offsets = [[0, 0], [cellSize / 2, 0], [0, cellSize / 2], [cellSize / 2, cellSize / 2]];
    var grids = offsets.map(function (offset) {
      var bins = new Map();
      points.forEach(function (p) {
        var ix = Math.floor((p[0] - WORLD.xmin - offset[0]) / cellSize);
        var iy = Math.floor((p[1] - WORLD.ymin - offset[1]) / cellSize);
        var key = ix + "," + iy;
        if (!bins.has(key)) bins.set(key, []);
        bins.get(key).push(p);
      });
      var cells = new Map();
      bins.forEach(function (pts, key) {
        if (pts.length < 3) return;
        var mean = centroid(pts);
        var xx = 0;
        var xy = 0;
        var yy = 0;
        pts.forEach(function (p) {
          var dx = p[0] - mean[0];
          var dy = p[1] - mean[1];
          xx += dx * dx;
          xy += dx * dy;
          yy += dy * dy;
        });
        xx = xx / pts.length + 0.026;
        xy = xy / pts.length;
        yy = yy / pts.length + 0.026;
        var determinant = xx * yy - xy * xy;
        if (determinant <= 0.000001) return;
        cells.set(key, {
          mean: mean,
          b11: yy / determinant,
          b12: -xy / determinant,
          b22: xx / determinant
        });
      });
      return cells;
    });
    return { size: cellSize, offsets: offsets, grids: grids };
  }

  function ndtCell(model, gridIndex, point) {
    var offset = model.offsets[gridIndex];
    var ix = Math.floor((point[0] - WORLD.xmin - offset[0]) / model.size);
    var iy = Math.floor((point[1] - WORLD.ymin - offset[1]) / model.size);
    return model.grids[gridIndex].get(ix + "," + iy);
  }

  function ndtScore(T) {
    var total = 0;
    scene.source.forEach(function (item) {
      var moved = applyT(T, item.p);
      for (var g = 0; g < ndt.model.grids.length; g += 1) {
        var cell = ndtCell(ndt.model, g, moved);
        if (!cell) continue;
        var dx = moved[0] - cell.mean[0];
        var dy = moved[1] - cell.mean[1];
        var mahal = cell.b11 * dx * dx + 2 * cell.b12 * dx * dy + cell.b22 * dy * dy;
        if (mahal < 28) total += Math.exp(-0.5 * mahal);
      }
    });
    return total / Math.max(1, scene.goodCount);
  }

  function rebuildLandscape() {
    var cols = 52;
    var rows = 40;
    var bounds = { xmin: -1.0, xmax: 3.3, ymin: -2.75, ymax: 1.45 };
    var values = [];
    var min = Infinity;
    var max = -Infinity;
    for (var row = 0; row < rows; row += 1) {
      for (var col = 0; col < cols; col += 1) {
        var tx = bounds.xmin + (col + 0.5) / cols * (bounds.xmax - bounds.xmin);
        var ty = bounds.ymax - (row + 0.5) / rows * (bounds.ymax - bounds.ymin);
        var value = ndtScore({ th: ndt.T.th, tx: tx, ty: ty });
        values.push(value);
        min = Math.min(min, value);
        max = Math.max(max, value);
      }
    }
    ndt.landscape = { cols: cols, rows: rows, bounds: bounds, values: values, min: min, max: max };
  }

  function resetNdt() {
    ndt = {
      T: { th: rad(Number(angleRange.value)), tx: 0.50, ty: -0.20 },
      model: buildNdtModel(scene.target, 1.3),
      landscape: null,
      iterations: 0,
      stepSize: 0.28,
      score: 0,
      done: false
    };
    ndt.score = ndtScore(ndt.T);
    rebuildLandscape();
    if (mode === "ndt") {
      setStatus("Choose a start on the likelihood map", false);
      updateMetrics();
      render();
    }
  }

  function stepNdt() {
    if (ndt.done) return true;
    var best = { T: cloneT(ndt.T), score: ndtScore(ndt.T) };
    var directions = [[1, 0], [0.707, 0.707], [0, 1], [-0.707, 0.707], [-1, 0], [-0.707, -0.707], [0, -1], [0.707, -0.707]];
    directions.forEach(function (direction) {
      var candidate = {
        th: ndt.T.th,
        tx: ndt.T.tx + direction[0] * ndt.stepSize,
        ty: ndt.T.ty + direction[1] * ndt.stepSize
      };
      var score = ndtScore(candidate);
      if (score > best.score + 0.00001) best = { T: candidate, score: score };
    });
    if (best.T.tx === ndt.T.tx && best.T.ty === ndt.T.ty) {
      ndt.stepSize *= 0.55;
    } else {
      ndt.T = best.T;
      ndt.score = best.score;
    }
    ndt.iterations += 1;
    ndt.done = ndt.stepSize < 0.018 || ndt.iterations >= 55;
    if (ndt.done) {
      var error = transformError(ndt.T);
      if (error.translation < 0.15) {
        setStatus("Peak found near the correct translation", false);
      } else {
        setStatus("Local peak found — try another bright basin", false);
      }
    } else {
      setStatus("Direct search climbed to score " + ndt.score.toFixed(2), true);
    }
    updateMetrics();
    render();
    return ndt.done;
  }

  function activeTransform() {
    if (mode === "ransac") return ransac.T;
    if (mode === "icp") return icp.T;
    return ndt.T;
  }

  function updateMetrics() {
    var T = activeTransform();
    metricPose.textContent = (T.th * DEG).toFixed(1) + "° / " + Math.hypot(T.tx, T.ty).toFixed(2) + " u";
    if (mode === "ransac") {
      metricALabel.textContent = "Hypotheses";
      metricBLabel.textContent = "Inliers";
      metricA.textContent = String(ransac.hypotheses);
      metricB.textContent = ransac.bestCount ? ransac.bestCount + " / " + scene.source.length : "—";
    } else if (mode === "icp") {
      metricALabel.textContent = "Iterations";
      metricBLabel.textContent = "RMS residual";
      metricA.textContent = String(icp.iterations);
      metricB.textContent = icp.rms === null ? "—" : icp.rms.toFixed(3) + " u";
    } else {
      metricALabel.textContent = "Climb steps";
      metricBLabel.textContent = "NDT score";
      metricA.textContent = String(ndt.iterations);
      metricB.textContent = ndt.score.toFixed(2);
    }
  }

  function setStatus(text, running) {
    statusEl.lastChild.nodeValue = " " + text;
    statusEl.classList.toggle("is-running", Boolean(running));
  }

  function selectedRunButton() {
    return document.getElementById(mode + "-run");
  }

  function pauseTimer() {
    if (timerId) {
      window.clearInterval(timerId);
      timerId = 0;
    }
    updateRunButton();
  }

  function stopAuto(clearIntent) {
    pauseTimer();
    if (clearIntent !== false) runIntent = false;
    updateRunButton();
  }

  function autoTick() {
    var done = false;
    if (mode === "ransac") done = stepRansac();
    if (mode === "icp") done = stepIcp();
    if (mode === "ndt") done = stepNdt();
    if (done) stopAuto(true);
  }

  function resumeTimer() {
    if (!runIntent || timerId || externallyPaused || document.hidden) return;
    timerId = window.setInterval(autoTick, mode === "ransac" ? 130 : 210);
    setStatus("Running — pause any time", true);
    updateRunButton();
  }

  function startAuto() {
    runIntent = true;
    resumeTimer();
  }

  function toggleAuto() {
    if (runIntent) {
      stopAuto(true);
      setStatus("Paused — step manually or resume", false);
    } else {
      startAuto();
    }
  }

  function updateRunButton() {
    document.querySelectorAll("[id$='-run']").forEach(function (button) {
      var active = button === selectedRunButton() && runIntent && Boolean(timerId);
      button.setAttribute("aria-pressed", String(active));
      button.textContent = active ? "Pause" : "Run";
    });
  }

  function switchMode(nextMode, focusTab) {
    if (!copy[nextMode] || nextMode === mode) return;
    stopAuto(true);
    drag = null;
    mode = nextMode;
    document.body.classList.toggle("mode-icp", mode === "icp");
    document.querySelectorAll(".method-tab").forEach(function (tab) {
      var selected = tab.dataset.mode === mode;
      tab.classList.toggle("is-active", selected);
      tab.setAttribute("aria-selected", String(selected));
      tab.tabIndex = selected ? 0 : -1;
      if (selected && focusTab) tab.focus();
    });
    document.querySelectorAll(".control-panel").forEach(function (panel) {
      panel.hidden = panel.id !== "panel-" + mode;
    });
    var text = copy[mode];
    methodKicker.textContent = text.kicker;
    methodTitle.textContent = text.title;
    methodCopy.textContent = text.body;
    stageHeading.textContent = text.heading;
    hintEl.textContent = text.hint;
    if (mode === "ransac") setStatus("Ready — test the first batch", false);
    if (mode === "icp") setStatus("Drag the blue scan into the attraction basin", false);
    if (mode === "ndt") setStatus("Choose a start on the likelihood map", false);
    updateMetrics();
    render();
  }

  function resizeCanvas() {
    var rect = canvas.getBoundingClientRect();
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    var width = Math.max(320, Math.round(rect.width));
    var height = Math.max(260, Math.round(rect.height));
    if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { width: width, height: height };
  }

  function plotLayout(size) {
    if (mode === "ndt") {
      var heatWidth = Math.max(230, size.width * 0.34);
      return {
        plot: { x: 18, y: 20, w: size.width - heatWidth - 48, h: size.height - 40 },
        heat: { x: size.width - heatWidth - 18, y: 43, w: heatWidth, h: size.height - 83 }
      };
    }
    return { plot: { x: 22, y: 20, w: size.width - 44, h: size.height - 40 }, heat: null };
  }

  function worldToScreen(point, plot) {
    return [
      plot.x + (point[0] - WORLD.xmin) / (WORLD.xmax - WORLD.xmin) * plot.w,
      plot.y + (WORLD.ymax - point[1]) / (WORLD.ymax - WORLD.ymin) * plot.h
    ];
  }

  function screenToWorld(point, plot) {
    return [
      WORLD.xmin + (point[0] - plot.x) / plot.w * (WORLD.xmax - WORLD.xmin),
      WORLD.ymax - (point[1] - plot.y) / plot.h * (WORLD.ymax - WORLD.ymin)
    ];
  }

  function inside(point, rect) {
    return point[0] >= rect.x && point[0] <= rect.x + rect.w &&
      point[1] >= rect.y && point[1] <= rect.y + rect.h;
  }

  function drawPlotBackground(plot) {
    ctx.save();
    ctx.fillStyle = COLORS.paper;
    ctx.fillRect(plot.x, plot.y, plot.w, plot.h);
    ctx.strokeStyle = COLORS.grid;
    ctx.lineWidth = 1;
    for (var x = -6; x <= 6; x += 2) {
      var sx = worldToScreen([x, 0], plot)[0];
      ctx.beginPath();
      ctx.moveTo(sx, plot.y);
      ctx.lineTo(sx, plot.y + plot.h);
      ctx.stroke();
    }
    for (var y = -3; y <= 3; y += 1.5) {
      var sy = worldToScreen([0, y], plot)[1];
      ctx.beginPath();
      ctx.moveTo(plot.x, sy);
      ctx.lineTo(plot.x + plot.w, sy);
      ctx.stroke();
    }
    var origin = worldToScreen([0, 0], plot);
    ctx.strokeStyle = "rgba(32,35,43,.23)";
    ctx.beginPath();
    ctx.moveTo(origin[0] - 5, origin[1]);
    ctx.lineTo(origin[0] + 5, origin[1]);
    ctx.moveTo(origin[0], origin[1] - 5);
    ctx.lineTo(origin[0], origin[1] + 5);
    ctx.stroke();
    ctx.restore();
  }

  function drawTruthBox(plot) {
    var points = scene.source.slice(0, scene.goodCount).map(function (item) {
      return applyT(truth, item.p);
    });
    var xmin = Infinity;
    var xmax = -Infinity;
    var ymin = Infinity;
    var ymax = -Infinity;
    points.forEach(function (p) {
      xmin = Math.min(xmin, p[0]);
      xmax = Math.max(xmax, p[0]);
      ymin = Math.min(ymin, p[1]);
      ymax = Math.max(ymax, p[1]);
    });
    var topLeft = worldToScreen([xmin - 0.12, ymax + 0.12], plot);
    var bottomRight = worldToScreen([xmax + 0.12, ymin - 0.12], plot);
    ctx.save();
    ctx.strokeStyle = "rgba(117,111,101,.48)";
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 5]);
    ctx.strokeRect(topLeft[0], topLeft[1], bottomRight[0] - topLeft[0], bottomRight[1] - topLeft[1]);
    ctx.restore();
  }

  function drawPoints(plot, T) {
    ctx.save();
    ctx.beginPath();
    ctx.rect(plot.x, plot.y, plot.w, plot.h);
    ctx.clip();
    scene.target.forEach(function (p) {
      var s = worldToScreen(p, plot);
      ctx.beginPath();
      ctx.arc(s[0], s[1], 2.7, 0, TAU);
      ctx.fillStyle = "rgba(138,45,28,.82)";
      ctx.fill();
    });
    scene.source.forEach(function (item) {
      var moved = applyT(T, item.p);
      var s = worldToScreen(moved, plot);
      ctx.beginPath();
      ctx.arc(s[0], s[1], item.outlier ? 2.2 : 2.8, 0, TAU);
      if (item.outlier) {
        ctx.strokeStyle = "rgba(72,106,154,.58)";
        ctx.lineWidth = 1.2;
        ctx.stroke();
      } else {
        ctx.fillStyle = "rgba(72,106,154,.82)";
        ctx.fill();
      }
    });
    ctx.restore();
  }

  function drawPoseFrame(plot, T, color, dashed) {
    var origin = worldToScreen([T.tx, T.ty], plot);
    var xEnd = worldToScreen(applyT(T, [0.65, 0]), plot);
    var yEnd = worldToScreen(applyT(T, [0, 0.65]), plot);
    ctx.save();
    if (dashed) ctx.setLineDash([4, 4]);
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(origin[0], origin[1]);
    ctx.lineTo(xEnd[0], xEnd[1]);
    ctx.moveTo(origin[0], origin[1]);
    ctx.lineTo(yEnd[0], yEnd[1]);
    ctx.stroke();
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(origin[0], origin[1], 2.6, 0, TAU);
    ctx.fill();
    ctx.restore();
  }

  function drawRansacOverlay(plot) {
    ctx.save();
    ctx.beginPath();
    ctx.rect(plot.x, plot.y, plot.w, plot.h);
    ctx.clip();
    ransac.inliers.forEach(function (match) {
      var p = worldToScreen(applyT(ransac.T, scene.source[match.source].p), plot);
      ctx.beginPath();
      ctx.arc(p[0], p[1], 4.6, 0, TAU);
      ctx.strokeStyle = "rgba(57,112,90,.82)";
      ctx.lineWidth = 1.4;
      ctx.stroke();
    });
    if (ransac.lastProposal) {
      var sourceA = worldToScreen(applyT(ransac.lastProposal.T, ransac.lastProposal.source[0]), plot);
      var sourceB = worldToScreen(applyT(ransac.lastProposal.T, ransac.lastProposal.source[1]), plot);
      var targetA = worldToScreen(ransac.lastProposal.target[0], plot);
      var targetB = worldToScreen(ransac.lastProposal.target[1], plot);
      ctx.lineWidth = 3;
      ctx.lineCap = "round";
      ctx.strokeStyle = "rgba(72,106,154,.72)";
      ctx.beginPath();
      ctx.moveTo(sourceA[0], sourceA[1]);
      ctx.lineTo(sourceB[0], sourceB[1]);
      ctx.stroke();
      ctx.strokeStyle = "rgba(138,45,28,.72)";
      ctx.beginPath();
      ctx.moveTo(targetA[0], targetA[1]);
      ctx.lineTo(targetB[0], targetB[1]);
      ctx.stroke();
    }
    ctx.restore();
  }

  function drawIcpOverlay(plot) {
    if (!showMatches.checked) return;
    ctx.save();
    ctx.beginPath();
    ctx.rect(plot.x, plot.y, plot.w, plot.h);
    ctx.clip();
    ctx.strokeStyle = "rgba(72,106,154,.19)";
    ctx.lineWidth = 1;
    var skip = Math.max(1, Math.ceil(icp.pairs.length / 38));
    for (var i = 0; i < icp.pairs.length; i += skip) {
      var pair = icp.pairs[i];
      var a = worldToScreen(applyT(icp.T, scene.source[pair.source].p), plot);
      var b = worldToScreen(scene.target[pair.target], plot);
      ctx.beginPath();
      ctx.moveTo(a[0], a[1]);
      ctx.lineTo(b[0], b[1]);
      ctx.stroke();
    }
    ctx.restore();
  }

  function heatColor(unit) {
    var t = Math.max(0, Math.min(1, Math.pow(unit, 0.72)));
    var stops = [
      [247, 243, 234],
      [240, 181, 148],
      [190, 79, 57],
      [77, 36, 44]
    ];
    var scaled = t * (stops.length - 1);
    var index = Math.min(stops.length - 2, Math.floor(scaled));
    var local = scaled - index;
    var a = stops[index];
    var b = stops[index + 1];
    return "rgb(" +
      Math.round(a[0] + (b[0] - a[0]) * local) + "," +
      Math.round(a[1] + (b[1] - a[1]) * local) + "," +
      Math.round(a[2] + (b[2] - a[2]) * local) + ")";
  }

  function heatToScreen(T, heat) {
    var bounds = ndt.landscape.bounds;
    return [
      heat.x + (T.tx - bounds.xmin) / (bounds.xmax - bounds.xmin) * heat.w,
      heat.y + (bounds.ymax - T.ty) / (bounds.ymax - bounds.ymin) * heat.h
    ];
  }

  function screenToHeat(point, heat) {
    var bounds = ndt.landscape.bounds;
    return {
      tx: bounds.xmin + (point[0] - heat.x) / heat.w * (bounds.xmax - bounds.xmin),
      ty: bounds.ymax - (point[1] - heat.y) / heat.h * (bounds.ymax - bounds.ymin)
    };
  }

  function drawNdtLandscape(heat) {
    var land = ndt.landscape;
    var cellW = heat.w / land.cols;
    var cellH = heat.h / land.rows;
    ctx.save();
    ctx.fillStyle = COLORS.muted;
    ctx.font = "700 10px system-ui, sans-serif";
    ctx.fillText("TRANSLATION SCORE", heat.x, heat.y - 13);
    for (var row = 0; row < land.rows; row += 1) {
      for (var col = 0; col < land.cols; col += 1) {
        var value = land.values[row * land.cols + col];
        var unit = (value - land.min) / Math.max(0.00001, land.max - land.min);
        ctx.fillStyle = heatColor(unit);
        ctx.fillRect(heat.x + col * cellW, heat.y + row * cellH, cellW + 0.55, cellH + 0.55);
      }
    }
    ctx.strokeStyle = "rgba(32,35,43,.25)";
    ctx.lineWidth = 1;
    ctx.strokeRect(heat.x, heat.y, heat.w, heat.h);

    var current = heatToScreen(ndt.T, heat);
    var answer = heatToScreen(truth, heat);
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = "rgba(255,255,255,.9)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(answer[0], answer[1], 8, 0, TAU);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#fff";
    ctx.strokeStyle = COLORS.ink;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(current[0], current[1], 6, 0, TAU);
    ctx.fill();
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(current[0] - 11, current[1]);
    ctx.lineTo(current[0] + 11, current[1]);
    ctx.moveTo(current[0], current[1] - 11);
    ctx.lineTo(current[0], current[1] + 11);
    ctx.stroke();

    ctx.fillStyle = COLORS.muted;
    ctx.font = "10px Georgia, serif";
    ctx.fillText("tx", heat.x + heat.w - 10, heat.y + heat.h + 17);
    ctx.fillText("ty", heat.x - 16, heat.y + 4);
    ctx.fillText("low", heat.x, heat.y + heat.h + 17);
    ctx.textAlign = "right";
    ctx.fillText("high", heat.x + heat.w - 20, heat.y + heat.h + 17);
    ctx.restore();
  }

  function render() {
    var size = resizeCanvas();
    var layout = plotLayout(size);
    lastLayout = layout;
    ctx.clearRect(0, 0, size.width, size.height);
    drawPlotBackground(layout.plot);
    drawTruthBox(layout.plot);
    drawPoints(layout.plot, activeTransform());
    drawPoseFrame(layout.plot, truth, COLORS.truth, true);
    drawPoseFrame(layout.plot, activeTransform(), COLORS.source, false);
    if (mode === "ransac") drawRansacOverlay(layout.plot);
    if (mode === "icp") drawIcpOverlay(layout.plot);
    if (mode === "ndt") drawNdtLandscape(layout.heat);
  }

  function canvasPoint(event) {
    var rect = canvas.getBoundingClientRect();
    return [event.clientX - rect.left, event.clientY - rect.top];
  }

  function beginCanvasInteraction(event) {
    if (!lastLayout) return;
    var point = canvasPoint(event);
    if (mode === "icp" && inside(point, lastLayout.plot)) {
      stopAuto(true);
      var world = screenToWorld(point, lastLayout.plot);
      var pivot = applyT(icp.T, scene.centroid);
      drag = {
        pointerId: event.pointerId,
        kind: event.shiftKey ? "rotate" : "translate",
        startWorld: world,
        startT: cloneT(icp.T),
        pivot: pivot,
        startAngle: Math.atan2(world[1] - pivot[1], world[0] - pivot[0])
      };
      canvas.setPointerCapture(event.pointerId);
      canvas.classList.add("is-dragging");
      setStatus(drag.kind === "rotate" ? "Rotating the initial pose" : "Positioning the initial pose", true);
      event.preventDefault();
    } else if (mode === "ndt" && inside(point, lastLayout.heat)) {
      stopAuto(true);
      drag = { pointerId: event.pointerId, kind: "landscape" };
      canvas.setPointerCapture(event.pointerId);
      moveOnLandscape(point);
      event.preventDefault();
    }
  }

  function moveCanvasInteraction(event) {
    if (!drag || event.pointerId !== drag.pointerId || !lastLayout) return;
    var point = canvasPoint(event);
    if (drag.kind === "landscape") {
      moveOnLandscape(point);
      return;
    }
    var world = screenToWorld(point, lastLayout.plot);
    if (drag.kind === "translate") {
      icp.T.tx = drag.startT.tx + world[0] - drag.startWorld[0];
      icp.T.ty = drag.startT.ty + world[1] - drag.startWorld[1];
    } else {
      var angle = Math.atan2(world[1] - drag.pivot[1], world[0] - drag.pivot[0]);
      icp.T.th = wrapAngle(drag.startT.th + angle - drag.startAngle);
      var c = Math.cos(icp.T.th);
      var s = Math.sin(icp.T.th);
      icp.T.tx = drag.pivot[0] - (c * scene.centroid[0] - s * scene.centroid[1]);
      icp.T.ty = drag.pivot[1] - (s * scene.centroid[0] + c * scene.centroid[1]);
    }
    icp.iterations = 0;
    icp.rms = null;
    icp.done = false;
    icp.pairs = computeIcpPairs(icp.T);
    updateMetrics();
    render();
    event.preventDefault();
  }

  function moveOnLandscape(point) {
    var heat = lastLayout.heat;
    var clamped = [
      Math.max(heat.x, Math.min(heat.x + heat.w, point[0])),
      Math.max(heat.y, Math.min(heat.y + heat.h, point[1]))
    ];
    var translation = screenToHeat(clamped, heat);
    ndt.T.tx = translation.tx;
    ndt.T.ty = translation.ty;
    ndt.score = ndtScore(ndt.T);
    ndt.iterations = 0;
    ndt.stepSize = 0.28;
    ndt.done = false;
    setStatus("New start selected — score " + ndt.score.toFixed(2), false);
    updateMetrics();
    render();
  }

  function endCanvasInteraction(event) {
    if (!drag || event.pointerId !== drag.pointerId) return;
    if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
    if (mode === "icp") setStatus("Start set — iterate nearest-neighbour fits", false);
    drag = null;
    canvas.classList.remove("is-dragging");
  }

  function nudgeIcp(event) {
    if (mode !== "icp") return;
    var handled = true;
    if (event.shiftKey && (event.key === "ArrowLeft" || event.key === "ArrowRight")) {
      var delta = event.key === "ArrowLeft" ? -rad(1) : rad(1);
      icp.T.th = wrapAngle(icp.T.th + delta);
    } else if (event.key === "ArrowLeft") {
      icp.T.tx -= 0.08;
    } else if (event.key === "ArrowRight") {
      icp.T.tx += 0.08;
    } else if (event.key === "ArrowUp") {
      icp.T.ty += 0.08;
    } else if (event.key === "ArrowDown") {
      icp.T.ty -= 0.08;
    } else {
      handled = false;
    }
    if (!handled) return;
    event.preventDefault();
    stopAuto(true);
    icp.iterations = 0;
    icp.rms = null;
    icp.done = false;
    icp.pairs = computeIcpPairs(icp.T);
    setStatus("Pose nudged — ready to iterate", false);
    updateMetrics();
    render();
  }

  function sendBackMessage() {
    if (window.parent !== window) {
      window.parent.postMessage({ type: "bento-live-back" }, "*");
      return true;
    }
    return false;
  }

  document.querySelectorAll(".method-tab").forEach(function (tab) {
    tab.addEventListener("click", function () { switchMode(tab.dataset.mode, false); });
    tab.addEventListener("keydown", function (event) {
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      event.preventDefault();
      var order = ["ransac", "icp", "ndt"];
      var delta = event.key === "ArrowRight" ? 1 : -1;
      var next = order[(order.indexOf(mode) + delta + order.length) % order.length];
      switchMode(next, true);
    });
  });

  document.getElementById("ransac-step").addEventListener("click", function () {
    stopAuto(true);
    stepRansac();
  });
  document.getElementById("icp-step").addEventListener("click", function () {
    stopAuto(true);
    stepIcp();
  });
  document.getElementById("ndt-step").addEventListener("click", function () {
    stopAuto(true);
    stepNdt();
  });
  document.querySelectorAll("[id$='-run']").forEach(function (button) {
    button.addEventListener("click", toggleAuto);
  });
  document.getElementById("ransac-reset").addEventListener("click", function () {
    stopAuto(true);
    resetRansac();
  });
  document.getElementById("icp-reset").addEventListener("click", function () {
    stopAuto(true);
    resetIcp();
  });
  document.getElementById("ndt-reset").addEventListener("click", function () {
    stopAuto(true);
    resetNdt();
  });

  outlierRange.addEventListener("input", function () {
    outlierValue.textContent = outlierRange.value + "%";
  });
  outlierRange.addEventListener("change", function () {
    stopAuto(true);
    scene = makeScene(Number(outlierRange.value));
    resetRansac();
    resetIcp();
    resetNdt();
    setStatus("New scan generated with " + outlierRange.value + "% outliers", false);
    updateMetrics();
    render();
  });
  gateRange.addEventListener("input", function () {
    gateValue.textContent = (Number(gateRange.value) / 100).toFixed(2) + " u";
    icp.pairs = computeIcpPairs(icp.T);
    updateMetrics();
    render();
  });
  angleRange.addEventListener("input", function () {
    stopAuto(true);
    angleValue.textContent = angleRange.value + "°";
    ndt.T.th = rad(Number(angleRange.value));
    ndt.score = ndtScore(ndt.T);
    ndt.iterations = 0;
    ndt.stepSize = 0.28;
    ndt.done = false;
    rebuildLandscape();
    setStatus("Landscape recomputed at " + angleRange.value + "°", false);
    updateMetrics();
    render();
  });
  showMatches.addEventListener("change", render);

  canvas.addEventListener("pointerdown", beginCanvasInteraction);
  canvas.addEventListener("pointermove", moveCanvasInteraction);
  canvas.addEventListener("pointerup", endCanvasInteraction);
  canvas.addEventListener("pointercancel", endCanvasInteraction);
  canvas.addEventListener("keydown", nudgeIcp);

  document.getElementById("back-link").addEventListener("click", function (event) {
    if (sendBackMessage()) event.preventDefault();
  });

  document.addEventListener("keydown", function (event) {
    if (event.key !== "Escape") return;
    stopAuto(true);
    if (!sendBackMessage()) window.location.href = "../";
  });

  document.addEventListener("visibilitychange", function () {
    if (document.hidden) {
      pauseTimer();
      if (runIntent) setStatus("Paused while the slide is hidden", false);
    } else if (runIntent && !externallyPaused) {
      resumeTimer();
    }
  });

  window.addEventListener("message", function (event) {
    var type = event.data && event.data.type;
    if (type === "bento-live-pause") {
      externallyPaused = true;
      pauseTimer();
      if (runIntent) setStatus("Paused by the slide deck", false);
    }
    if (type === "bento-live-resume") {
      externallyPaused = false;
      if (runIntent) resumeTimer();
    }
  });

  window.addEventListener("pagehide", function () {
    stopAuto(true);
  });

  if ("ResizeObserver" in window) {
    new ResizeObserver(render).observe(canvas);
  } else {
    window.addEventListener("resize", render);
  }

  resetRansac();
  resetIcp();
  resetNdt();
  if (requestedMode === "ransac") {
    document.body.classList.remove("mode-icp");
    updateMetrics();
    render();
  } else {
    switchMode(requestedMode, false);
  }
}());
