(function () {
  "use strict";

  var GATE = 9.21;
  var TRACK_COLORS = ["#2ca02c", "#9467bd", "#d62728"];
  var COLORS = {
    ink: "#16222e",
    soft: "#51606e",
    faint: "#8a97a3",
    line: "#d7dee5",
    grid: "#eef2f5",
    bp: "#1f77b4",
    bpDeep: "#155d8f",
    pm: "#e8720c",
    pmDeep: "#b45607",
    white: "#ffffff"
  };

  function mathTex(source, display) {
    return '<span class="math-tex ' + (display ? 'math-display' : 'math-inline') + '">' +
      (display ? '\\[' : '\\(') + source + (display ? '\\]' : '\\)') + '</span>';
  }
  var DEFAULT_TRACKS = [
    { x: 285, y: 205, S: [[520, 140], [140, 340]] },
    { x: 352, y: 232, S: [[460, -120], [-120, 480]] },
    { x: 318, y: 158, S: [[620, 0], [0, 300]] }
  ];
  var DEFAULT_MEASUREMENTS = [
    { x: 318, y: 198 },
    { x: 322, y: 215 },
    { x: 314, y: 186 },
    { x: 560, y: 120 }
  ];
  var SEPARATE_TRACKS = [
    { x: 165, y: 250, S: [[380, 20], [20, 300]] },
    { x: 360, y: 125, S: [[360, -30], [-30, 320]] },
    { x: 535, y: 275, S: [[400, 0], [0, 310]] }
  ];
  var SEPARATE_MEASUREMENTS = [
    { x: 174, y: 245 },
    { x: 351, y: 133 },
    { x: 527, y: 268 },
    { x: 650, y: 80 }
  ];

  var canvas = document.getElementById("association-canvas");
  var ctx = canvas.getContext("2d");
  var visualView = document.getElementById("visual-view");
  var hypothesisView = document.getElementById("hypothesis-view");
  var dataCard = document.getElementById("data-card");
  var hypothesisList = document.getElementById("hypothesis-list");
  var marginalCard = document.getElementById("marginal-card");
  var methodKicker = document.getElementById("method-kicker");
  var methodTitle = document.getElementById("method-title");
  var methodCopy = document.getElementById("method-copy");
  var stageHeading = document.getElementById("stage-heading");
  var statusEl = document.getElementById("status");
  var hintEl = document.getElementById("canvas-hint");
  var pdRange = document.getElementById("pd-range");
  var clutterRange = document.getElementById("clutter-range");
  var kRange = document.getElementById("k-range");
  var pdValue = document.getElementById("pd-value");
  var clutterValue = document.getElementById("clutter-value");
  var kValue = document.getElementById("k-value");
  var metricA = document.getElementById("metric-a");
  var metricB = document.getElementById("metric-b");
  var metricC = document.getElementById("metric-c");
  var metricALabel = document.getElementById("metric-a-label");
  var metricBLabel = document.getElementById("metric-b-label");
  var metricCLabel = document.getElementById("metric-c-label");

  var copy = {
    assignment: {
      kicker: "COMMON INPUT",
      title: "Shape the association problem",
      body: "Drag a measurement through overlapping validation gates. Both inference routes consume the same normalized weights.",
      heading: "GATED LIKELIHOODS " + mathTex("\\ell"),
      hint: "Dashed ellipses are 99% validation gates. A dot in the matrix means a gated-out pair."
    },
    bp: {
      kicker: "MARGINAL VIEW",
      title: "Negotiate without enumeration",
      body: "Loopy sum–product passes local competition messages until approximate association marginals settle.",
      heading: "WILLIAMS–LAU MESSAGE PASSING",
      hint: "Line width follows the current BP marginal. Edge labels show " + mathTex("\\mu_{\\mathrm{track}\\to\\mathrm{measurement}}") + " and " + mathTex("\\nu_{\\mathrm{measurement}\\to\\mathrm{track}}") + "."
    },
    hypotheses: {
      kicker: "JOINT VIEW",
      title: "Rank compatible stories",
      body: "Exhaustive enumeration exposes every valid global assignment in this small benchmark; " + mathTex("k") + " controls truncation.",
      heading: "EXACT JOINT ASSIGNMENTS + PRUNING",
      hint: "Orange bars are normalized joint-event weights. Dim rows are pruned; retained marginals are renormalized over the kept set."
    }
  };

  var requestedMode = new URLSearchParams(window.location.search).get("demo");
  if (!Object.prototype.hasOwnProperty.call(copy, requestedMode)) requestedMode = "assignment";
  var mode = "assignment";
  var state = {
    tracks: clone(DEFAULT_TRACKS),
    measurements: clone(DEFAULT_MEASUREMENTS),
    PD: 0.90,
    clutter: 5e-5,
    k: 5,
    bpIndex: 0,
    selectedMeasurement: 0
  };
  var result = null;
  var canvasMap = null;
  var drag = null;
  var timerId = 0;
  var runIntent = false;
  var externallyPaused = false;

  function clone(value) {
    return JSON.parse(JSON.stringify(value));
  }

  function gaussian2(dx, dy, S) {
    var determinant = S[0][0] * S[1][1] - S[0][1] * S[1][0];
    var q = (
      dx * (S[1][1] * dx - S[0][1] * dy) +
      dy * (-S[1][0] * dx + S[0][0] * dy)
    ) / determinant;
    return {
      pdf: Math.exp(-0.5 * q) / (2 * Math.PI * Math.sqrt(determinant)),
      d2: q
    };
  }

  function buildWeights() {
    var L = [];
    var gate = [];
    for (var i = 0; i < state.tracks.length; i += 1) {
      L.push([1 - state.PD]);
      gate.push([]);
      for (var j = 0; j < state.measurements.length; j += 1) {
        var track = state.tracks[i];
        var measurement = state.measurements[j];
        var likelihood = gaussian2(measurement.x - track.x, measurement.y - track.y, track.S);
        var inside = likelihood.d2 <= GATE;
        gate[i].push(inside);
        L[i].push(inside ? state.PD * likelihood.pdf / state.clutter : 0);
      }
    }
    return { L: L, gate: gate, n: state.tracks.length, m: state.measurements.length };
  }

  function marginalsFromMessages(L, nu) {
    var M = [];
    for (var i = 0; i < L.length; i += 1) {
      var row = [L[i][0]];
      for (var j = 0; j < nu.length; j += 1) row.push(L[i][j + 1] * nu[j][i]);
      var normalizer = row.reduce(function (sum, value) { return sum + value; }, 0);
      M.push(row.map(function (value) { return value / normalizer; }));
    }
    return M;
  }

  function runBeliefPropagation(L, maxIterations, tolerance) {
    var n = L.length;
    var m = L[0].length - 1;
    var nu = Array.from({ length: m }, function () { return Array(n).fill(1); });
    var mu = Array.from({ length: n }, function () { return Array(m).fill(0); });
    var history = [{
      sweep: 0,
      mu: mu.map(function (row) { return row.slice(); }),
      nu: nu.map(function (row) { return row.slice(); }),
      marginals: marginalsFromMessages(L, nu),
      delta: null
    }];

    for (var sweep = 1; sweep <= maxIterations; sweep += 1) {
      for (var i = 0; i < n; i += 1) {
        var total = L[i][0];
        for (var k = 0; k < m; k += 1) total += L[i][k + 1] * nu[k][i];
        for (var j = 0; j < m; j += 1) {
          mu[i][j] = L[i][j + 1] / Math.max(1e-15, total - L[i][j + 1] * nu[j][i]);
        }
      }

      var nextNu = Array.from({ length: m }, function () { return Array(n).fill(1); });
      var delta = 0;
      for (var measurement = 0; measurement < m; measurement += 1) {
        var contested = 1;
        for (var trackIndex = 0; trackIndex < n; trackIndex += 1) contested += mu[trackIndex][measurement];
        for (var target = 0; target < n; target += 1) {
          var value = 1 / Math.max(1e-15, contested - mu[target][measurement]);
          delta = Math.max(delta, Math.abs(value - nu[measurement][target]));
          nextNu[measurement][target] = value;
        }
      }
      nu = nextNu;
      history.push({
        sweep: sweep,
        mu: mu.map(function (row) { return row.slice(); }),
        nu: nu.map(function (row) { return row.slice(); }),
        marginals: marginalsFromMessages(L, nu),
        delta: delta
      });
      if (delta < tolerance) break;
    }
    return history;
  }

  function enumerateAssignments(L) {
    var n = L.length;
    var m = L[0].length - 1;
    var assignment = Array(n).fill(-1);
    var events = [];
    function recurse(index, used, weight) {
      if (index === n) {
        events.push({ assignment: assignment.slice(), weight: weight });
        return;
      }
      assignment[index] = -1;
      recurse(index + 1, used, weight * L[index][0]);
      for (var j = 0; j < m; j += 1) {
        if ((used & (1 << j)) || L[index][j + 1] <= 0) continue;
        assignment[index] = j;
        recurse(index + 1, used | (1 << j), weight * L[index][j + 1]);
      }
      assignment[index] = -1;
    }
    recurse(0, 0, 1);
    var normalizer = events.reduce(function (sum, event) { return sum + event.weight; }, 0);
    events.forEach(function (event) { event.probability = event.weight / normalizer; });
    events.sort(function (a, b) { return b.probability - a.probability; });
    return events;
  }

  function eventMarginals(events, n, m, limit) {
    var use = events.slice(0, limit === undefined ? events.length : limit);
    var retainedMass = use.reduce(function (sum, event) { return sum + event.probability; }, 0);
    var M = Array.from({ length: n }, function () { return Array(m + 1).fill(0); });
    use.forEach(function (event) {
      for (var i = 0; i < n; i += 1) {
        M[i][event.assignment[i] < 0 ? 0 : event.assignment[i] + 1] += event.probability;
      }
    });
    if (retainedMass > 0) {
      M = M.map(function (row) {
        return row.map(function (value) { return value / retainedMass; });
      });
    }
    return { marginals: M, mass: retainedMass };
  }

  function maxDifference(A, B) {
    var maximum = 0;
    for (var i = 0; i < A.length; i += 1) {
      for (var j = 0; j < A[i].length; j += 1) {
        maximum = Math.max(maximum, Math.abs(A[i][j] - B[i][j]));
      }
    }
    return maximum;
  }

  function recompute(resetBp) {
    var built = buildWeights();
    var history = runBeliefPropagation(built.L, 50, 1e-10);
    var events = enumerateAssignments(built.L);
    var exact = eventMarginals(events, built.n, built.m).marginals;
    var pairs = [];
    for (var i = 0; i < built.n; i += 1) {
      for (var j = 0; j < built.m; j += 1) if (built.gate[i][j]) pairs.push([i, j]);
    }
    result = {
      L: built.L,
      gate: built.gate,
      n: built.n,
      m: built.m,
      history: history,
      events: events,
      exact: exact,
      pairs: pairs
    };
    result.finalError = maxDifference(history[history.length - 1].marginals, exact);
    if (resetBp !== false) state.bpIndex = 0;
    state.bpIndex = Math.min(state.bpIndex, history.length - 1);
    state.k = Math.max(1, Math.min(state.k, events.length));
    kRange.max = String(events.length);
    kRange.value = String(state.k);
    kValue.value = String(state.k);
    renderAll();
  }

  function formatWeight(value) {
    if (value <= 0) return "·";
    if (value >= 100) return value.toFixed(0);
    if (value >= 10) return value.toFixed(1);
    return value.toFixed(2);
  }

  function formatProbability(value) {
    if (value < 0.0005) return "·";
    if (value < 0.095) return (value * 100).toFixed(1) + "%";
    return Math.round(value * 100) + "%";
  }

  function formatMessage(value) {
    if (!Number.isFinite(value)) return "—";
    if (value < 9.995) return value.toFixed(2);
    if (value < 99.5) return value.toFixed(1);
    return String(Math.round(value));
  }

  function currentBp() {
    return result.history[state.bpIndex];
  }

  function setStatus(text, running) {
    statusEl.innerHTML = "<i></i> " + text;
    statusEl.classList.toggle("is-running", Boolean(running));
  }

  function weightTable() {
    var html = "<h3>Existing-track weights</h3><table class=\"mini-matrix\"><tr><th></th><th>" + mathTex("\\varnothing") + "</th>";
    for (var j = 0; j < result.m; j += 1) html += "<th>" + mathTex("z_{" + (j + 1) + "}") + "</th>";
    html += "</tr>";
    for (var i = 0; i < result.n; i += 1) {
      html += "<tr><td class=\"row-head\" style=\"color:" + TRACK_COLORS[i] + "\">" + mathTex("T_{" + (i + 1) + "}") + "</td>";
      result.L[i].forEach(function (value, column) {
        var cellClass = value <= 0 ? "dim" : (column === state.selectedMeasurement + 1 ? "hot" : "");
        html += "<td class=\"" + cellClass + "\">" + formatWeight(value) + "</td>";
      });
      html += "</tr>";
    }
    html += "</table>";
    html += "<p class=\"card-note\">" + mathTex("\\ell_{ij}=P_{\\mathrm D}\\,\\mathcal N(z_j;\\widehat z_i,S_i)/\\lambda_c", true) + mathTex("\\ell_{i\\varnothing}=1-P_{\\mathrm D}", true) + "unassigned measurement baseline = 1</p>";
    return html;
  }

  function bpTable() {
    var current = currentBp();
    var html = "<h3>BP track marginals</h3><table class=\"mini-matrix\"><tr><th></th><th>" + mathTex("\\varnothing") + "</th>";
    for (var j = 0; j < result.m; j += 1) html += "<th>" + mathTex("z_{" + (j + 1) + "}") + "</th>";
    html += "</tr>";
    for (var i = 0; i < result.n; i += 1) {
      html += "<tr><td class=\"row-head\" style=\"color:" + TRACK_COLORS[i] + "\">" + mathTex("T_{" + (i + 1) + "}") + "</td>";
      current.marginals[i].forEach(function (value) {
        html += "<td>" + formatProbability(value) + "</td>";
      });
      html += "</tr>";
    }
    var error = maxDifference(current.marginals, result.exact);
    var width = Math.min(100, error * 1000);
    html += "</table><p class=\"card-note\">Exact-reference max difference: <strong>" + (100 * error).toFixed(2) + " pp</strong></p>";
    html += "<div class=\"delta-track\"><i style=\"width:" + width.toFixed(1) + "%\"></i></div>";
    html += "<p class=\"card-note\">Sweep " + current.sweep + " of " + (result.history.length - 1) + " · " + mathTex("\\Delta") + " " + (current.delta === null ? "initial" : current.delta.toExponential(1)) + "</p>";
    return html;
  }

  function updateMetrics() {
    if (mode === "assignment") {
      metricALabel.textContent = "Gated pairs";
      metricBLabel.textContent = "Valid events";
      metricCLabel.textContent = "BP final error";
      metricA.textContent = result.pairs.length + " / " + (result.n * result.m);
      metricB.textContent = String(result.events.length);
      metricC.textContent = (100 * result.finalError).toFixed(2) + " pp";
    } else if (mode === "bp") {
      var current = currentBp();
      metricALabel.textContent = "Sweep";
      metricBLabel.textContent = "Message Δ";
      metricCLabel.textContent = "Marginal error";
      metricA.textContent = current.sweep + " / " + (result.history.length - 1);
      metricB.textContent = current.delta === null ? "initial" : current.delta.toExponential(1);
      metricC.textContent = (100 * maxDifference(current.marginals, result.exact)).toFixed(2) + " pp";
    } else {
      var truncated = eventMarginals(result.events, result.n, result.m, state.k);
      metricALabel.textContent = "Kept";
      metricBLabel.textContent = "Retained mass";
      metricCLabel.textContent = "Discarded mass";
      metricA.textContent = state.k + " / " + result.events.length;
      metricB.textContent = (100 * truncated.mass).toFixed(1) + "%";
      metricC.textContent = (100 * Math.max(0, 1 - truncated.mass)).toFixed(1) + "%";
    }
  }

  function resizeCanvas() {
    var rect = canvas.getBoundingClientRect();
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    var width = Math.max(300, Math.round(rect.width));
    var height = Math.max(220, Math.round(rect.height));
    if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { width: width, height: height };
  }

  function prepareSceneMap(size) {
    var scale = Math.min((size.width - 22) / 720, (size.height - 22) / 420);
    canvasMap = {
      scale: scale,
      ox: (size.width - 720 * scale) / 2,
      oy: (size.height - 420 * scale) / 2
    };
  }

  function sceneToCanvas(point) {
    return [
      canvasMap.ox + point.x * canvasMap.scale,
      canvasMap.oy + point.y * canvasMap.scale
    ];
  }

  function canvasToScene(point) {
    return {
      x: (point[0] - canvasMap.ox) / canvasMap.scale,
      y: (point[1] - canvasMap.oy) / canvasMap.scale
    };
  }

  function ellipseParameters(S) {
    var a = S[0][0];
    var b = S[0][1];
    var c = S[1][1];
    var angle = 0.5 * Math.atan2(2 * b, a - c);
    var middle = (a + c) / 2;
    var spread = Math.sqrt(Math.pow((a - c) / 2, 2) + b * b);
    return {
      rx: Math.sqrt(GATE * (middle + spread)),
      ry: Math.sqrt(GATE * (middle - spread)),
      angle: angle
    };
  }

  function drawScene() {
    var size = resizeCanvas();
    prepareSceneMap(size);
    ctx.clearRect(0, 0, size.width, size.height);
    ctx.fillStyle = COLORS.white;
    ctx.fillRect(0, 0, size.width, size.height);
    ctx.save();
    ctx.translate(canvasMap.ox, canvasMap.oy);
    ctx.scale(canvasMap.scale, canvasMap.scale);
    ctx.strokeStyle = COLORS.grid;
    ctx.lineWidth = 1 / canvasMap.scale;
    for (var x = 100; x < 720; x += 100) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, 420);
      ctx.stroke();
    }
    for (var y = 100; y < 420; y += 100) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(720, y);
      ctx.stroke();
    }

    result.pairs.forEach(function (pair) {
      var track = state.tracks[pair[0]];
      var measurement = state.measurements[pair[1]];
      ctx.beginPath();
      ctx.moveTo(track.x, track.y);
      ctx.lineTo(measurement.x, measurement.y);
      ctx.setLineDash([3, 4]);
      ctx.strokeStyle = "rgba(81,96,110,.45)";
      ctx.lineWidth = 1 / canvasMap.scale;
      ctx.stroke();
    });
    ctx.setLineDash([]);

    state.tracks.forEach(function (track, index) {
      var ellipse = ellipseParameters(track.S);
      ctx.save();
      ctx.translate(track.x, track.y);
      ctx.rotate(ellipse.angle);
      ctx.beginPath();
      ctx.ellipse(0, 0, ellipse.rx, ellipse.ry, 0, 0, Math.PI * 2);
      ctx.fillStyle = colorAlpha(TRACK_COLORS[index], 0.07);
      ctx.strokeStyle = TRACK_COLORS[index];
      ctx.lineWidth = 1.5 / canvasMap.scale;
      ctx.setLineDash([6, 5]);
      ctx.fill();
      ctx.stroke();
      ctx.restore();
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.arc(track.x, track.y, 5.5, 0, Math.PI * 2);
      ctx.fillStyle = TRACK_COLORS[index];
      ctx.fill();
      ctx.fillStyle = TRACK_COLORS[index];
      ctx.font = "700 12px ui-monospace, monospace";
      ctx.fillText("T" + (index + 1), track.x + 10, track.y - 9);
    });

    state.measurements.forEach(function (measurement, index) {
      ctx.save();
      ctx.translate(measurement.x, measurement.y);
      if (index === state.selectedMeasurement) {
        ctx.beginPath();
        ctx.arc(0, 0, 13, 0, Math.PI * 2);
        ctx.strokeStyle = COLORS.bp;
        ctx.lineWidth = 2 / canvasMap.scale;
        ctx.stroke();
      }
      ctx.beginPath();
      ctx.moveTo(-6, -6);
      ctx.lineTo(6, 6);
      ctx.moveTo(-6, 6);
      ctx.lineTo(6, -6);
      ctx.strokeStyle = COLORS.ink;
      ctx.lineWidth = 2.4 / canvasMap.scale;
      ctx.stroke();
      ctx.fillStyle = COLORS.ink;
      ctx.font = "11px ui-monospace, monospace";
      ctx.fillText("z" + (index + 1), 9, 16);
      ctx.restore();
    });
    ctx.restore();
  }

  function colorAlpha(hex, alpha) {
    var value = hex.replace("#", "");
    var r = parseInt(value.slice(0, 2), 16);
    var g = parseInt(value.slice(2, 4), 16);
    var b = parseInt(value.slice(4, 6), 16);
    return "rgba(" + r + "," + g + "," + b + "," + alpha + ")";
  }

  function drawBpGraph() {
    var size = resizeCanvas();
    ctx.clearRect(0, 0, size.width, size.height);
    ctx.fillStyle = COLORS.white;
    ctx.fillRect(0, 0, size.width, size.height);
    var current = currentBp();
    var leftX = Math.max(85, size.width * 0.18);
    var rightX = size.width * 0.77;
    var top = 45;
    var bottom = size.height - 38;
    var trackY = state.tracks.map(function (_, index) {
      return top + index * (bottom - top) / Math.max(1, result.n - 1);
    });
    var measurementY = state.measurements.map(function (_, index) {
      return 24 + index * (size.height - 48) / Math.max(1, result.m - 1);
    });

    for (var i = 0; i < result.n; i += 1) {
      for (var j = 0; j < result.m; j += 1) {
        var live = result.gate[i][j];
        var marginal = current.marginals[i][j + 1];
        ctx.save();
        ctx.beginPath();
        ctx.moveTo(leftX + 18, trackY[i]);
        ctx.lineTo(rightX - 16, measurementY[j]);
        ctx.strokeStyle = live ? colorAlpha(TRACK_COLORS[i], 0.34 + 0.45 * marginal) : "rgba(138,151,163,.32)";
        ctx.lineWidth = live ? 1 + marginal * 7 : 1;
        if (!live) ctx.setLineDash([3, 5]);
        ctx.stroke();
        if (live) {
          var dx = rightX - leftX;
          var dy = measurementY[j] - trackY[i];
          ctx.fillStyle = TRACK_COLORS[i];
          ctx.font = "700 8px ui-monospace, monospace";
          ctx.fillText("μ " + formatMessage(current.mu[i][j]), leftX + dx * 0.31, trackY[i] + dy * 0.31 - 4);
          ctx.fillStyle = COLORS.soft;
          ctx.fillText("ν " + formatMessage(current.nu[j][i]), leftX + dx * 0.64, trackY[i] + dy * 0.64 + 10);
        }
        ctx.restore();
      }
    }

    trackY.forEach(function (y, index) {
      ctx.beginPath();
      ctx.arc(leftX, y, 18, 0, Math.PI * 2);
      ctx.fillStyle = COLORS.white;
      ctx.fill();
      ctx.strokeStyle = TRACK_COLORS[index];
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.fillStyle = TRACK_COLORS[index];
      ctx.font = "italic 700 12px Georgia, serif";
      ctx.textAlign = "center";
      ctx.fillText("a" + (index + 1), leftX, y + 4);
    });
    measurementY.forEach(function (y, index) {
      ctx.beginPath();
      ctx.arc(rightX, y, 16, 0, Math.PI * 2);
      ctx.fillStyle = COLORS.white;
      ctx.fill();
      ctx.strokeStyle = COLORS.soft;
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.fillStyle = COLORS.soft;
      ctx.font = "italic 700 11px Georgia, serif";
      ctx.textAlign = "center";
      ctx.fillText("b" + (index + 1), rightX, y + 4);
    });
    ctx.textAlign = "left";
    ctx.fillStyle = COLORS.faint;
    ctx.font = "800 9px ui-monospace, monospace";
    ctx.fillText("TRACK VARIABLES", leftX - 46, 17);
    ctx.fillText("MEASUREMENT VARIABLES", rightX - 63, 17);
    ctx.fillStyle = COLORS.bpDeep;
    ctx.fillText("sweep " + current.sweep + " / " + (result.history.length - 1), 12, size.height - 12);
  }

  function assignmentStory(event) {
    var parts = event.assignment.map(function (measurement, track) {
      if (measurement < 0) return "<span class=\"miss\">" + mathTex("T_{" + (track + 1) + "}\\to\\varnothing") + "</span>";
      return mathTex("T_{" + (track + 1) + "}\\to z_{" + (measurement + 1) + "}");
    });
    var assigned = new Set(event.assignment.filter(function (value) { return value >= 0; }));
    var free = [];
    for (var j = 0; j < result.m; j += 1) if (!assigned.has(j)) free.push(mathTex("z_{" + (j + 1) + "}"));
    if (free.length) parts.push("<span class=\"miss\">" + free.join(",") + " unassigned</span>");
    return parts.join(" · ");
  }

  function renderHypotheses() {
    var topProbability = result.events[0].probability;
    var visible = Math.min(12, result.events.length);
    var html = "";
    for (var i = 0; i < visible; i += 1) {
      var event = result.events[i];
      var relative = 100 * event.probability / topProbability;
      html += "<div class=\"hyp-row" + (i >= state.k ? " is-pruned" : "") + "\">";
      html += "<div class=\"hyp-rank\">#" + (i + 1) + "</div>";
      html += "<div class=\"hyp-story\">" + assignmentStory(event) + "</div>";
      html += "<div class=\"weight-cell\"><i style=\"width:" + Math.max(1.2, relative).toFixed(1) + "%\"></i><span>" + (100 * event.probability).toFixed(event.probability < 0.01 ? 2 : 1) + "%</span></div>";
      html += "</div>";
    }
    if (result.events.length > visible) {
      html += "<div class=\"hyp-row is-pruned\"><div class=\"hyp-rank\">…</div><div class=\"hyp-story\">" + (result.events.length - visible) + " additional compatible events</div><div></div></div>";
    }
    hypothesisList.innerHTML = html;

    var truncated = eventMarginals(result.events, result.n, result.m, state.k);
    var card = "<div class=\"mass-callout\"><strong>" + (100 * truncated.mass).toFixed(1) + "%</strong><span>normalized joint mass retained by top " + state.k + "</span></div>";
    card += "<h3>Marginals after truncation</h3>";
    truncated.marginals.forEach(function (row, track) {
      card += "<div class=\"track-marginal\"><h4 style=\"color:" + TRACK_COLORS[track] + "\">" + mathTex("T_{" + (track + 1) + "}") + "</h4>";
      row.forEach(function (value, column) {
        card += "<div class=\"bar-row\"><span>" + mathTex(column === 0 ? "\\varnothing" : "z_{" + column + "}") + "</span><div class=\"bar-track\"><i style=\"width:" + (100 * value).toFixed(1) + "%\"></i></div><span>" + formatProbability(value) + "</span></div>";
      });
      card += "</div>";
    });
    card += "<p class=\"card-note\">These are assignment marginals from the retained rows—not Bernoulli existence/state updates from an undetected-target Poisson intensity.</p>";
    marginalCard.innerHTML = card;
  }

  function renderStage() {
    if (mode === "hypotheses") {
      renderHypotheses();
      return;
    }
    if (mode === "assignment") {
      drawScene();
      dataCard.innerHTML = weightTable();
    } else {
      drawBpGraph();
      dataCard.innerHTML = bpTable();
    }
  }

  function renderAll() {
    pdValue.value = state.PD.toFixed(2);
    clutterValue.innerHTML = mathTex(formatScientificTex(state.clutter));
    kValue.value = String(state.k);
    updateMetrics();
    renderStage();
  }

  function formatScientificTex(value) {
    var parts = value.toExponential(1).split("e");
    return parts[0] + "\\times10^{" + Number(parts[1]) + "}";
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

  function updateRunButton() {
    var button = document.getElementById("bp-run");
    var active = mode === "bp" && runIntent && Boolean(timerId);
    button.setAttribute("aria-pressed", String(active));
    button.textContent = active ? "Pause" : "Run";
  }

  function stepBp() {
    if (state.bpIndex < result.history.length - 1) state.bpIndex += 1;
    var current = currentBp();
    if (state.bpIndex >= result.history.length - 1) {
      stopAuto(true);
      setStatus("Fixed point reached — compare with exhaustive marginals", false);
    } else {
      setStatus("Sweep " + current.sweep + " landed · Δ " + current.delta.toExponential(1), true);
    }
    renderAll();
    return state.bpIndex >= result.history.length - 1;
  }

  function resumeTimer() {
    if (!runIntent || timerId || mode !== "bp" || externallyPaused || document.hidden) return;
    timerId = window.setInterval(function () {
      if (stepBp()) stopAuto(true);
    }, 260);
    setStatus("Passing messages sweep by sweep", true);
    updateRunButton();
  }

  function toggleAuto() {
    if (runIntent) {
      stopAuto(true);
      setStatus("Paused — step manually or resume", false);
    } else {
      runIntent = true;
      if (state.bpIndex >= result.history.length - 1) state.bpIndex = 0;
      resumeTimer();
      renderAll();
    }
  }

  function switchMode(nextMode, focusTab) {
    if (!copy[nextMode] || nextMode === mode) return;
    stopAuto(true);
    drag = null;
    mode = nextMode;
    document.body.classList.toggle("mode-assignment", mode === "assignment");
    document.body.classList.toggle("mode-hypotheses", mode === "hypotheses");
    document.querySelectorAll(".mode-tab").forEach(function (tab) {
      var selected = tab.dataset.mode === mode;
      tab.classList.toggle("is-active", selected);
      tab.setAttribute("aria-selected", String(selected));
      tab.tabIndex = selected ? 0 : -1;
      if (selected && focusTab) tab.focus();
    });
    document.querySelectorAll(".control-panel").forEach(function (panel) {
      panel.hidden = panel.id !== "panel-" + mode;
    });
    visualView.hidden = mode === "hypotheses";
    hypothesisView.hidden = mode !== "hypotheses";
    var text = copy[mode];
    methodKicker.textContent = text.kicker;
    methodTitle.textContent = text.title;
    methodCopy.innerHTML = text.body;
    stageHeading.innerHTML = text.heading;
    hintEl.innerHTML = text.hint;
    if (mode === "assignment") setStatus("Drag a measurement to recompute", false);
    if (mode === "bp") setStatus("Start at uncoupled " + mathTex("\\nu=1"), false);
    if (mode === "hypotheses") setStatus("Top " + state.k + " of " + result.events.length + " events retained", false);
    renderAll();
  }

  function applyPreset(kind) {
    stopAuto(true);
    if (kind === "separate") {
      state.tracks = clone(SEPARATE_TRACKS);
      state.measurements = clone(SEPARATE_MEASUREMENTS);
      setStatus("Separated scene — the graph nearly decomposes", false);
    } else {
      state.tracks = clone(DEFAULT_TRACKS);
      state.measurements = clone(DEFAULT_MEASUREMENTS);
      setStatus("Three-way tangle restored", false);
    }
    state.selectedMeasurement = 0;
    recompute(true);
  }

  function canvasPoint(event) {
    var rect = canvas.getBoundingClientRect();
    return [event.clientX - rect.left, event.clientY - rect.top];
  }

  function beginDrag(event) {
    if (mode !== "assignment" || !canvasMap) return;
    var point = canvasPoint(event);
    var nearestIndex = -1;
    var nearestDistance = Infinity;
    state.measurements.forEach(function (measurement, index) {
      var screen = sceneToCanvas(measurement);
      var distance = Math.hypot(point[0] - screen[0], point[1] - screen[1]);
      if (distance < nearestDistance) {
        nearestDistance = distance;
        nearestIndex = index;
      }
    });
    if (nearestDistance > 24) return;
    drag = { pointerId: event.pointerId, measurement: nearestIndex };
    state.selectedMeasurement = nearestIndex;
    canvas.setPointerCapture(event.pointerId);
    canvas.classList.add("is-dragging");
    moveDrag(event);
    event.preventDefault();
  }

  function moveDrag(event) {
    if (!drag || event.pointerId !== drag.pointerId || !canvasMap) return;
    var point = canvasToScene(canvasPoint(event));
    state.measurements[drag.measurement].x = Math.max(8, Math.min(712, point.x));
    state.measurements[drag.measurement].y = Math.max(8, Math.min(412, point.y));
    recompute(true);
    setStatus("z" + (drag.measurement + 1) + " moved · all weights and inferences recomputed", true);
    event.preventDefault();
  }

  function endDrag(event) {
    if (!drag || event.pointerId !== drag.pointerId) return;
    if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
    setStatus("Scene updated — inspect weights or switch inference view", false);
    drag = null;
    canvas.classList.remove("is-dragging");
  }

  function nudgeMeasurement(event) {
    if (mode !== "assignment") return;
    if (/^[1-4]$/.test(event.key)) {
      event.preventDefault();
      state.selectedMeasurement = Number(event.key) - 1;
      setStatus("Selected z" + event.key + " · use arrow keys to move it", false);
      renderAll();
      return;
    }
    var delta = event.shiftKey ? 10 : 3;
    var measurement = state.measurements[state.selectedMeasurement];
    var handled = true;
    if (event.key === "ArrowLeft") measurement.x -= delta;
    else if (event.key === "ArrowRight") measurement.x += delta;
    else if (event.key === "ArrowUp") measurement.y -= delta;
    else if (event.key === "ArrowDown") measurement.y += delta;
    else handled = false;
    if (!handled) return;
    event.preventDefault();
    measurement.x = Math.max(8, Math.min(712, measurement.x));
    measurement.y = Math.max(8, Math.min(412, measurement.y));
    recompute(true);
    setStatus("z" + (state.selectedMeasurement + 1) + " nudged · recomputed", false);
  }

  function setK(value) {
    state.k = Math.max(1, Math.min(Number(value), result.events.length));
    kRange.value = String(state.k);
    kValue.value = String(state.k);
    setStatus("Top " + state.k + " retains " + (100 * eventMarginals(result.events, result.n, result.m, state.k).mass).toFixed(1) + "% of joint mass", false);
    renderAll();
  }

  function sendBackMessage() {
    if (window.parent !== window) {
      window.parent.postMessage({ type: "bento-live-back" }, "*");
      return true;
    }
    return false;
  }

  document.querySelectorAll(".mode-tab").forEach(function (tab) {
    tab.addEventListener("click", function () { switchMode(tab.dataset.mode, false); });
    tab.addEventListener("keydown", function (event) {
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      event.preventDefault();
      var order = ["assignment", "bp", "hypotheses"];
      var delta = event.key === "ArrowRight" ? 1 : -1;
      switchMode(order[(order.indexOf(mode) + delta + order.length) % order.length], true);
    });
  });

  pdRange.addEventListener("input", function () {
    state.PD = Number(pdRange.value) / 100;
    recompute(true);
    setStatus("Detection probability changed · shared weights recomputed", false);
  });
  clutterRange.addEventListener("input", function () {
    state.clutter = Math.pow(10, Number(clutterRange.value));
    recompute(true);
    setStatus("Clutter density changed · shared weights recomputed", false);
  });
  kRange.addEventListener("input", function () { setK(kRange.value); });

  document.getElementById("tangle-preset").addEventListener("click", function () { applyPreset("tangle"); });
  document.getElementById("separate-preset").addEventListener("click", function () { applyPreset("separate"); });
  document.getElementById("assignment-reset").addEventListener("click", function () { applyPreset("tangle"); });
  document.getElementById("bp-step").addEventListener("click", function () {
    stopAuto(true);
    stepBp();
  });
  document.getElementById("bp-run").addEventListener("click", toggleAuto);
  document.getElementById("bp-reset").addEventListener("click", function () {
    stopAuto(true);
    state.bpIndex = 0;
    setStatus("Reset to uncoupled " + mathTex("\\nu=1"), false);
    renderAll();
  });
  document.getElementById("bp-end").addEventListener("click", function () {
    stopAuto(true);
    state.bpIndex = result.history.length - 1;
    setStatus("Fixed point reached — exact benchmark remains separate", false);
    renderAll();
  });
  document.getElementById("keep-one").addEventListener("click", function () { setK(1); });
  document.getElementById("keep-five").addEventListener("click", function () { setK(Math.min(5, result.events.length)); });
  document.getElementById("keep-all").addEventListener("click", function () { setK(result.events.length); });

  canvas.addEventListener("pointerdown", beginDrag);
  canvas.addEventListener("pointermove", moveDrag);
  canvas.addEventListener("pointerup", endDrag);
  canvas.addEventListener("pointercancel", endDrag);
  canvas.addEventListener("keydown", nudgeMeasurement);

  document.getElementById("back-link").addEventListener("click", function (event) {
    stopAuto(true);
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
    if (event.source !== window.parent || event.origin !== window.location.origin) return;
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
  window.addEventListener("pagehide", function () { stopAuto(true); });

  if ("ResizeObserver" in window) {
    new ResizeObserver(function () {
      if (mode !== "hypotheses") renderStage();
    }).observe(canvas);
  } else {
    window.addEventListener("resize", renderStage);
  }

  recompute(true);
  document.body.classList.add("mode-assignment");
  if (requestedMode !== "assignment") switchMode(requestedMode, false);
}());
