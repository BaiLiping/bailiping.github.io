(function () {
  "use strict";

  var S1 = window.RadioSlamS1;
  if (!S1) throw new Error("Shared radio-SLAM setup S1 failed to load");
  var GATE = 9.21;
  var TRACK_COLORS = ["#e8720c", "#0e8f7e", "#1874b8"];
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
  var SIGNATURE_BOUNDS = { left: 52, right: 674, top: 34, bottom: 384, rangeMax: 14 };

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
  var scanRange = document.getElementById("scan-range");
  var scanValue = document.getElementById("scan-value");
  var setupScan = document.getElementById("setup-scan");
  var setupKnown = document.getElementById("setup-known");
  var setupLatent = document.getElementById("setup-latent");
  var setupOutput = document.getElementById("setup-output");
  var pdRange = document.getElementById("pd-range");
  var clutterRange = document.getElementById("clutter-range");
  var particleRange = document.getElementById("particle-range");
  var pdValue = document.getElementById("pd-value");
  var clutterValue = document.getElementById("clutter-value");
  var particleValue = document.getElementById("particle-value");
  var metricA = document.getElementById("metric-a");
  var metricB = document.getElementById("metric-b");
  var metricC = document.getElementById("metric-c");
  var metricALabel = document.getElementById("metric-a-label");
  var metricBLabel = document.getElementById("metric-b-label");
  var metricCLabel = document.getElementById("metric-c-label");

  var copy = {
    assignment: {
      kicker: "COMMON INPUT",
      title: "Inspect the common evidence",
      body: "Drag an MPC in the path-length/AoA plane. Both state–map factorizations consume the same BS, odometry, and measured tuples.",
      heading: "SHARED S1 RADIO EVIDENCE",
      hint: "This optional view exposes the common scan evidence. Pages 12 and 14 use it to explain different factorizations of the same latent trajectory and map."
    },
    bp: {
      kicker: "BP-SLAM FACTORIZATION",
      title: "Couple state and map in one graph",
      body: "Motion factors connect " + mathTex("\\mathbf x_{1:5}") + "; radio factors couple every UE state to the shared map features " + mathTex("\\mathbf m_A,\\mathbf m_B") + ".",
      heading: "FACTOR GRAPH FOR " + mathTex("p(\\mathbf X,\\mathcal M\\mid Z,U,\\mathbf b,A)"),
      hint: "Squares are factors; circles are latent variables. The S1 path labels are conditioned in this structural walkthrough so association does not obscure the state–map coupling."
    },
    pmbm: {
      kicker: "PMBM-SLAM FACTORIZATION",
      title: "Condition the map on the trajectory",
      body: "A weighted trajectory-particle mixture represents " + mathTex("f(\\mathbf X\\mid Z,U)") + "; each particle carries its own conditional PMBM map.",
      heading: "RAO–BLACKWELLIZED STATE × CONDITIONAL MAP",
      hint: "Select a trajectory particle. Its conditional map splits into a PPP for undetected features and an MBM of detected Bernoulli features."
    }
  };

  var BP_STAGES = [
    { label: "Joint variables", family: "priors", detail: "The UE trajectory and map features are both latent; priors enter as factors." },
    { label: "Motion chain", family: "motion", detail: "Odometry factors propagate information along the five-state UE trajectory." },
    { label: "Radio coupling", family: "radio", detail: "Each scan factor depends jointly on its UE state and the shared map features." },
    { label: "Map update", family: "state→map", detail: "Radio messages combine evidence from all five states into each map belief." },
    { label: "State feedback", family: "map→state", detail: "Updated map messages return through the radio factors to refine every UE state." },
    { label: "Marginal beliefs", family: "beliefs", detail: "Products of incoming messages return separate approximate marginals for state and map." }
  ];

  var PMBM_PARTICLES = [
    { weight: 0.58, shift: "0.00\\,\\mathrm m", lambda: 0.36, rA: 0.96, rB: 0.91, h1: 0.73, h2: 0.27, mA: "[2.1,12.0]^{\\mathsf T}", mB: "[15.0,2.1]^{\\mathsf T}" },
    { weight: 0.27, shift: "+0.12\\,\\mathrm m", lambda: 0.52, rA: 0.88, rB: 0.79, h1: 0.61, h2: 0.39, mA: "[2.2,11.8]^{\\mathsf T}", mB: "[14.8,2.2]^{\\mathsf T}" },
    { weight: 0.15, shift: "-0.18\\,\\mathrm m", lambda: 0.81, rA: 0.71, rB: 0.64, h1: 0.54, h2: 0.46, mA: "[1.9,12.3]^{\\mathsf T}", mB: "[15.3,1.9]^{\\mathsf T}" }
  ];

  var requestedMode = new URLSearchParams(window.location.search).get("demo");
  if (requestedMode === "hypotheses") requestedMode = "pmbm";
  if (!Object.prototype.hasOwnProperty.call(copy, requestedMode)) requestedMode = "assignment";
  var mode = "assignment";
  var initialScene = sceneForScan(3);
  var state = {
    scanIndex: 3,
    scanData: initialScene.scan,
    tracks: initialScene.tracks,
    measurements: initialScene.measurements,
    PD: 0.90,
    clutter: 5e-5,
    factorStage: 0,
    particleIndex: 0,
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

  function degrees(angle) {
    return angle / S1.DEG;
  }

  function signaturePoint(value) {
    var bounds = SIGNATURE_BOUNDS;
    return {
      x: bounds.left + Math.max(0, Math.min(bounds.rangeMax, value.range)) / bounds.rangeMax * (bounds.right - bounds.left),
      y: bounds.top + (180 - degrees(value.aoa)) / 360 * (bounds.bottom - bounds.top)
    };
  }

  function routeTex(index) {
    if (index === 0) return "H_{\\mathrm{LoS}}";
    return "H_{" + (index === 1 ? "A" : "B") + "}";
  }

  function sceneForScan(scanIndex) {
    var scan = S1.scan(scanIndex, 1);
    var covariances = [
      [[760, 0], [0, 900]],
      [[1250, 180], [180, 2200]],
      [[1250, -180], [-180, 2200]]
    ];
    var tracks = scan.predictions.map(function (prediction, index) {
      var point = signaturePoint(prediction);
      return {
        x: point.x,
        y: point.y,
        S: clone(covariances[index]),
        label: prediction.label,
        tex: prediction.tex,
        range: prediction.range,
        aoa: prediction.aoa
      };
    });
    var measurements = scan.measurements.map(function (measurement) {
      var point = signaturePoint(measurement);
      return {
        x: point.x,
        y: point.y,
        label: measurement.label,
        isClutter: measurement.isClutter,
        range: measurement.range,
        aoa: measurement.aoa,
        gainDb: measurement.gainDb
      };
    });
    return { scan: scan, tracks: tracks, measurements: measurements };
  }

  function updateSetupPanel() {
    var scanNumber = state.scanIndex + 1;
    setupScan.textContent = "scan " + scanNumber + " / 5";
    scanValue.innerHTML = mathTex("\\mathbf z_{" + scanNumber + "}") + " / 5";
    if (mode === "bp") {
      setupKnown.innerHTML = mathTex("\\mathbf b,U,Z,A") + " · S1 labels conditioned";
      setupLatent.innerHTML = mathTex("\\mathbf X,\\mathcal M") + " · trajectory and map";
      setupOutput.innerHTML = mathTex("b(\\mathbf x_t),b(\\mathbf m_j)") + " marginals";
    } else if (mode === "pmbm") {
      setupKnown.innerHTML = mathTex("\\mathbf b,U,Z") + " · common S1 evidence";
      setupLatent.innerHTML = mathTex("\\mathbf X,\\mathcal M") + " · trajectory and RFS map";
      setupOutput.textContent = "trajectory mixture + conditional PMBM map";
    } else {
      setupKnown.innerHTML = mathTex("\\mathbf b,U,Z") + " · common S1 evidence";
      setupLatent.innerHTML = mathTex("\\mathbf X,\\mathcal M") + " · trajectory and map";
      setupOutput.textContent = "same evidence for both factorizations";
    }
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
    if (resetBp !== false) state.factorStage = 0;
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

  function setStatus(text, running) {
    statusEl.innerHTML = "<i></i> " + text;
    statusEl.classList.toggle("is-running", Boolean(running));
  }

  function weightTable() {
    var html = "<h3>S1 route-hypothesis weights</h3><table class=\"mini-matrix\"><tr><th></th><th>" + mathTex("\\varnothing") + "</th>";
    for (var j = 0; j < result.m; j += 1) html += "<th>" + mathTex("z_{" + (j + 1) + "}") + "</th>";
    html += "</tr>";
    for (var i = 0; i < result.n; i += 1) {
      html += "<tr><td class=\"row-head\" style=\"color:" + TRACK_COLORS[i] + "\">" + mathTex(routeTex(i)) + "</td>";
      result.L[i].forEach(function (value, column) {
        var cellClass = value <= 0 ? "dim" : (column === state.selectedMeasurement + 1 ? "hot" : "");
        html += "<td class=\"" + cellClass + "\">" + formatWeight(value) + "</td>";
      });
      html += "</tr>";
    }
    html += "</table>";
    html += "<p class=\"card-note\">" + mathTex("\\ell_{ij}=P_{\\mathrm D}\\,\\mathcal N(z_j;\\widehat z_i,S_i)/\\lambda_c", true) + mathTex("\\ell_{i\\varnothing}=1-P_{\\mathrm D}", true) + "gates include the current S1 map-prediction uncertainty</p>";
    return html;
  }

  function bpStageCard() {
    var stage = BP_STAGES[state.factorStage];
    var html = "<div class=\"stage-callout\"><span>STEP " + (state.factorStage + 1) + " / " + BP_STAGES.length + "</span><strong>" + stage.label + "</strong><p>" + stage.detail + "</p></div>";
    if (state.factorStage < 5) {
      html += "<h3>Joint factorization</h3>";
      html += mathTex("p(\\mathbf X,\\mathcal M\\mid Z,U,\\mathbf b,A)\\propto p(\\mathbf x_1)\\prod_{t=2}^{5} f_t^{\\mathrm{mot}}(\\mathbf x_{t-1},\\mathbf x_t;\\mathbf u_t)\\prod_{j\\in\\{A,B\\}}p(\\mathbf m_j)\\prod_{t=1}^{5}f_t^{\\mathrm{rad}}(\\mathbf x_t,\\mathbf m_A,\\mathbf m_B;Z_t,A_t,\\mathbf b)", true);
    } else {
      html += "<h3>Returned marginal beliefs</h3>";
      html += mathTex("b(\\mathbf x_t)\\propto\\prod_{f\\in\\mathcal N(\\mathbf x_t)}\\mu_{f\\to\\mathbf x_t}(\\mathbf x_t)", true);
      html += mathTex("b(\\mathbf m_j)\\propto p(\\mathbf m_j)\\prod_{t=1}^{5}\\mu_{f_t^{\\mathrm{rad}}\\to\\mathbf m_j}(\\mathbf m_j)", true);
    }
    html += "<p class=\"card-note\"><strong>Conditioning note.</strong> " + mathTex("A") + " denotes the fixed S1 route labels in this diagram. Full BP-SLAM also represents feature existence and association variables; they are suppressed here to expose the " + mathTex("\\mathbf X") + "–" + mathTex("\\mathcal M") + " coupling.</p>";
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
      metricALabel.textContent = "Stage";
      metricBLabel.textContent = "Active family";
      metricCLabel.textContent = "Return";
      metricA.textContent = (state.factorStage + 1) + " / " + BP_STAGES.length;
      metricB.textContent = BP_STAGES[state.factorStage].family;
      metricC.textContent = state.factorStage === 5 ? "b(x), b(m)" : "pending";
    } else {
      var particle = PMBM_PARTICLES[state.particleIndex];
      metricALabel.textContent = "Particle";
      metricBLabel.textContent = "Weight";
      metricCLabel.textContent = "Map";
      metricA.textContent = (state.particleIndex + 1) + " / " + PMBM_PARTICLES.length;
      metricB.textContent = (100 * particle.weight).toFixed(0) + "%";
      metricC.textContent = "PPP + MBM";
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
      ctx.fillText(index === 0 ? "LoS" : (index === 1 ? "wall A" : "wall B"), track.x + 10, track.y - 9);
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
      ctx.fillText("z" + (index + 1) + (measurement.isClutter ? " · FA" : ""), 9, 16);
      ctx.restore();
    });
    ctx.fillStyle = COLORS.faint;
    ctx.font = "800 9px ui-monospace, monospace";
    ctx.textAlign = "right";
    ctx.fillText("path length cτ (m) →", 704, 410);
    ctx.save();
    ctx.translate(15, 226);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.fillText("UE-frame AoA φ (deg)", 0, 0);
    ctx.restore();
    ctx.textAlign = "left";
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
      ctx.fillText(index === 0 ? "L" : (index === 1 ? "A" : "B"), leftX, y + 4);
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
      ctx.fillText("z" + (index + 1), rightX, y + 4);
    });
    ctx.textAlign = "left";
    ctx.fillStyle = COLORS.faint;
    ctx.font = "800 9px ui-monospace, monospace";
    ctx.fillText("ROUTE HYPOTHESES", leftX - 46, 17);
    ctx.fillText("MPC VARIABLES", rightX - 48, 17);
    ctx.fillStyle = COLORS.bpDeep;
    ctx.fillText("sweep " + current.sweep + " / " + (result.history.length - 1), 12, size.height - 12);
  }

  function assignmentStory(event) {
    var parts = event.assignment.map(function (measurement, track) {
      if (measurement < 0) return "<span class=\"miss\">" + mathTex(routeTex(track) + "\\to\\varnothing") + "</span>";
      return mathTex(routeTex(track) + "\\to z_{" + (measurement + 1) + "}");
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
      card += "<div class=\"track-marginal\"><h4 style=\"color:" + TRACK_COLORS[track] + "\">" + mathTex(routeTex(track)) + "</h4>";
      row.forEach(function (value, column) {
        card += "<div class=\"bar-row\"><span>" + mathTex(column === 0 ? "\\varnothing" : "z_{" + column + "}") + "</span><div class=\"bar-track\"><i style=\"width:" + (100 * value).toFixed(1) + "%\"></i></div><span>" + formatProbability(value) + "</span></div>";
      });
      card += "</div>";
    });
    card += "<p class=\"card-note\">These are assignment marginals from the retained rows—not Bernoulli existence/state updates from an undetected-target Poisson intensity.</p>";
    marginalCard.innerHTML = card;
  }

  function renderBpFactorGraph() {
    var stage = state.factorStage;
    var poseX = [120, 310, 500, 690, 880];
    var motionX = [215, 405, 595, 785];
    var graph = "<div class=\"factor-graph\" role=\"img\" aria-label=\"BP-SLAM factor graph coupling five UE-state variables to two map-feature variables through motion and radio factors\">";
    graph += "<div class=\"fg-axis-label fg-map-label\">LATENT MAP " + mathTex("\\mathcal M") + "</div>";
    graph += "<div class=\"fg-axis-label fg-state-label\">LATENT UE TRAJECTORY " + mathTex("\\mathbf X=\\mathbf x_{1:5}") + "</div>";
    graph += "<svg class=\"fg-lines\" viewBox=\"0 0 1000 500\" preserveAspectRatio=\"none\" aria-hidden=\"true\"><defs>";
    graph += "<marker id=\"bp-arrow-blue\" viewBox=\"0 0 10 10\" refX=\"8\" refY=\"5\" markerWidth=\"5\" markerHeight=\"5\" orient=\"auto-start-reverse\"><path d=\"M 0 0 L 10 5 L 0 10 z\"></path></marker>";
    graph += "<marker id=\"bp-arrow-green\" viewBox=\"0 0 10 10\" refX=\"8\" refY=\"5\" markerWidth=\"5\" markerHeight=\"5\" orient=\"auto-start-reverse\"><path d=\"M 0 0 L 10 5 L 0 10 z\"></path></marker></defs>";
    graph += "<line class=\"fg-edge prior-edge" + (stage === 0 ? " is-active" : "") + "\" x1=\"270\" y1=\"24\" x2=\"270\" y2=\"80\"></line>";
    graph += "<line class=\"fg-edge prior-edge" + (stage === 0 ? " is-active" : "") + "\" x1=\"730\" y1=\"24\" x2=\"730\" y2=\"80\"></line>";
    graph += "<line class=\"fg-edge prior-edge" + (stage === 0 ? " is-active" : "") + "\" x1=\"52\" y1=\"410\" x2=\"120\" y2=\"410\"></line>";
    motionX.forEach(function (x, index) {
      var active = stage === 1 ? " is-active" : "";
      graph += "<line class=\"fg-edge motion-edge" + active + "\" x1=\"" + poseX[index] + "\" y1=\"410\" x2=\"" + x + "\" y2=\"410\"></line>";
      graph += "<line class=\"fg-edge motion-edge" + active + "\" x1=\"" + x + "\" y1=\"410\" x2=\"" + poseX[index + 1] + "\" y2=\"410\"></line>";
    });
    poseX.forEach(function (x) {
      var radioActive = stage === 2 || stage === 3 || stage === 4 ? " is-active" : "";
      var mapArrow = stage === 3 ? " marker-end=\"url(#bp-arrow-green)\"" : "";
      var stateArrow = stage === 4 ? " marker-end=\"url(#bp-arrow-blue)\"" : "";
      graph += "<line class=\"fg-edge radio-edge map-a-edge" + radioActive + "\" x1=\"" + x + "\" y1=\"250\" x2=\"270\" y2=\"80\"" + mapArrow + "></line>";
      graph += "<line class=\"fg-edge radio-edge map-b-edge" + radioActive + "\" x1=\"" + x + "\" y1=\"250\" x2=\"730\" y2=\"80\"" + mapArrow + "></line>";
      graph += "<line class=\"fg-edge radio-edge state-radio-edge" + radioActive + "\" x1=\"" + x + "\" y1=\"250\" x2=\"" + x + "\" y2=\"410\"" + stateArrow + "></line>";
    });
    graph += "</svg>";
    graph += "<div class=\"fg-node fg-prior prior-map-a" + (stage === 0 ? " is-active" : "") + "\">" + mathTex("p(\\mathbf m_A)") + "</div>";
    graph += "<div class=\"fg-node fg-prior prior-map-b" + (stage === 0 ? " is-active" : "") + "\">" + mathTex("p(\\mathbf m_B)") + "</div>";
    graph += "<div class=\"fg-node fg-prior prior-state" + (stage === 0 ? " is-active" : "") + "\">" + mathTex("p(\\mathbf x_1)") + "</div>";
    graph += "<div class=\"fg-node fg-variable map-node map-a" + (stage === 3 || stage === 5 ? " is-active" : "") + "\">" + mathTex("\\mathbf m_A") + "</div>";
    graph += "<div class=\"fg-node fg-variable map-node map-b" + (stage === 3 || stage === 5 ? " is-active" : "") + "\">" + mathTex("\\mathbf m_B") + "</div>";
    poseX.forEach(function (_, index) {
      graph += "<div class=\"fg-node fg-factor radio-factor radio-" + (index + 1) + (stage === 2 || stage === 3 || stage === 4 ? " is-active" : "") + "\">" + mathTex("f_{" + (index + 1) + "}^{\\mathrm{rad}}") + "</div>";
      graph += "<div class=\"fg-node fg-variable state-node state-" + (index + 1) + (stage === 4 || stage === 5 ? " is-active" : "") + "\">" + mathTex("\\mathbf x_{" + (index + 1) + "}") + "</div>";
    });
    motionX.forEach(function (_, index) {
      graph += "<div class=\"fg-node fg-factor motion-factor motion-" + (index + 2) + (stage === 1 ? " is-active" : "") + "\">" + mathTex("f_{" + (index + 2) + "}^{\\mathrm{mot}}") + "</div>";
    });
    graph += "<div class=\"fg-legend\"><span><i class=\"legend-variable\"></i>variable</span><span><i class=\"legend-factor\"></i>factor</span><span>" + mathTex("A") + " conditioned</span></div></div>";
    hypothesisList.innerHTML = graph;
    marginalCard.innerHTML = bpStageCard();
  }

  function renderPmbmFactorization() {
    var particle = PMBM_PARTICLES[state.particleIndex];
    var html = "<div class=\"pmbm-tree\">";
    html += "<div class=\"posterior-node posterior-root\"><span>JOINT POSTERIOR</span>" + mathTex("f(\\mathbf X,\\mathcal M\\mid Z,U)", true) + "</div>";
    html += "<div class=\"tree-operator\">" + mathTex("=") + "</div>";
    html += "<div class=\"state-map-row\"><div class=\"posterior-node state-density\"><span>VEHICLE TRAJECTORY</span>" + mathTex("f(\\mathbf X\\mid Z,U)", true) + "</div><div class=\"tree-times\">" + mathTex("\\times") + "</div><div class=\"posterior-node conditional-density\"><span>CONDITIONAL RFS MAP</span>" + mathTex("f(\\mathcal M\\mid\\mathbf X,Z,U)", true) + "</div></div>";
    html += "<div class=\"particle-strip\">";
    PMBM_PARTICLES.forEach(function (item, index) {
      html += "<div class=\"particle-card" + (index === state.particleIndex ? " is-selected" : "") + "\"><span>" + mathTex("\\mathbf X^{(" + (index + 1) + ")}") + "</span><strong>" + mathTex("w^{(" + (index + 1) + ")}=" + item.weight.toFixed(2)) + "</strong></div>";
    });
    html += "</div>";
    html += "<div class=\"conditional-banner\">Selected " + mathTex("\\mathbf X^{(" + (state.particleIndex + 1) + ")}") + " carries " + mathTex("f_{\\mathrm{PMBM}}^{(" + (state.particleIndex + 1) + ")}(\\mathcal M)") + "</div>";
    html += "<div class=\"pmbm-split\"><div class=\"component-card ppp-card\"><span>UNDETECTED MAP</span><strong>PPP</strong>" + mathTex("f_{\\mathrm P}^{u}(\\mathcal M^u;\\lambda^{u})", true) + "<small>expected count " + mathTex("\\Lambda^u=" + particle.lambda.toFixed(2)) + "</small></div><div class=\"tree-plus\">" + mathTex("\\uplus") + "</div><div class=\"component-card mbm-card\"><span>DETECTED MAP</span><strong>MBM</strong>" + mathTex("\\sum_{h\\in\\mathcal H}w_h\\prod_i f_{\\mathrm B}^{h,i}(\\mathcal M^i)", true) + "<small>global hypotheses mix Bernoulli map components</small></div></div>";
    html += "<div class=\"bernoulli-row\"><div><span>FEATURE A</span>" + mathTex("r_A=" + particle.rA.toFixed(2)) + "</div><div><span>FEATURE B</span>" + mathTex("r_B=" + particle.rB.toFixed(2)) + "</div><div><span>GLOBAL MIXTURE</span>" + mathTex("(w_{h_1},w_{h_2})=(" + particle.h1.toFixed(2) + "," + particle.h2.toFixed(2) + ")") + "</div></div>";
    html += "</div>";
    hypothesisList.innerHTML = html;

    var card = "<div class=\"particle-callout\"><span>SELECTED TRAJECTORY PARTICLE</span><strong>" + mathTex("n=" + (state.particleIndex + 1) + ",\\quad w^{(n)}=" + particle.weight.toFixed(2)) + "</strong><p>illustrative lateral shift " + mathTex(particle.shift) + "</p></div>";
    card += "<h3>Rao–Blackwellized factorization</h3>";
    card += mathTex("f(\\mathbf X,\\mathcal M\\mid Z,U)\\approx\\sum_{n=1}^{N}w^{(n)}\\delta(\\mathbf X-\\mathbf X^{(n)})f_{\\mathrm{PMBM}}^{(n)}(\\mathcal M)", true);
    card += "<h3>Conditional map for particle " + (state.particleIndex + 1) + "</h3>";
    card += mathTex("f_{\\mathrm{PMBM}}^{(n)}(\\mathcal M)=\\sum_{\\mathcal M^u\\uplus\\mathcal M^d=\\mathcal M}f_{\\mathrm P}^{u,(n)}(\\mathcal M^u)\\sum_{h\\in\\mathcal H^{(n)}}w_h^{(n)}\\sum_{\\biguplus_i\\mathcal M^i=\\mathcal M^d}\\prod_i f_{\\mathrm B}^{(n,h,i)}(\\mathcal M^i)", true);
    card += "<div class=\"map-readout\"><div><span>" + mathTex("\\mathbf m_A") + "</span><strong>" + mathTex(particle.mA + "\\,\\mathrm m") + "</strong></div><div><span>" + mathTex("\\mathbf m_B") + "</span><strong>" + mathTex(particle.mB + "\\,\\mathrm m") + "</strong></div></div>";
    card += "<p class=\"card-note\">Particle weights and component parameters are illustrative S1 values. The structural split—trajectory density times a trajectory-conditioned PMBM map—is the method-level result.</p>";
    marginalCard.innerHTML = card;
  }

  function renderStage() {
    if (mode === "assignment") {
      drawScene();
      dataCard.innerHTML = weightTable();
    } else if (mode === "bp") {
      renderBpFactorGraph();
    } else {
      renderPmbmFactorization();
    }
  }

  function renderAll() {
    scanRange.value = String(state.scanIndex);
    pdValue.value = state.PD.toFixed(2);
    clutterValue.innerHTML = mathTex(formatScientificTex(state.clutter));
    particleRange.value = String(state.particleIndex);
    particleValue.innerHTML = mathTex("n=" + (state.particleIndex + 1));
    ["particle-one", "particle-two", "particle-three"].forEach(function (id, index) {
      var button = document.getElementById(id);
      var selected = index === state.particleIndex;
      button.classList.toggle("primary", selected);
      button.setAttribute("aria-pressed", String(selected));
    });
    updateSetupPanel();
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
    button.textContent = active ? "Pause" : "Play";
  }

  function stepBp() {
    if (state.factorStage < BP_STAGES.length - 1) state.factorStage += 1;
    if (state.factorStage >= BP_STAGES.length - 1) {
      stopAuto(true);
      setStatus("Marginal state and map beliefs assembled", false);
    } else {
      setStatus(BP_STAGES[state.factorStage].label + " active", true);
    }
    renderAll();
    return state.factorStage >= BP_STAGES.length - 1;
  }

  function resumeTimer() {
    if (!runIntent || timerId || mode !== "bp" || externallyPaused || document.hidden) return;
    timerId = window.setInterval(function () {
      if (stepBp()) stopAuto(true);
    }, 720);
    setStatus("Walking through factor families", true);
    updateRunButton();
  }

  function toggleAuto() {
    if (runIntent) {
      stopAuto(true);
      setStatus("Paused — step manually or continue", false);
    } else {
      runIntent = true;
      if (state.factorStage >= BP_STAGES.length - 1) state.factorStage = 0;
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
    document.body.classList.toggle("mode-hypotheses", mode === "pmbm");
    document.body.classList.toggle("mode-pmbm", mode === "pmbm");
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
    visualView.hidden = mode !== "assignment";
    hypothesisView.hidden = mode === "assignment";
    var text = copy[mode];
    methodKicker.textContent = text.kicker;
    methodTitle.textContent = text.title;
    methodCopy.innerHTML = text.body;
    stageHeading.innerHTML = text.heading;
    hintEl.innerHTML = text.hint;
    if (mode === "assignment") setStatus("Scan " + (state.scanIndex + 1) + " loaded · drag an MPC to perturb it", false);
    if (mode === "bp") setStatus("Start with joint latent variables and priors", false);
    if (mode === "pmbm") setStatus("Particle " + (state.particleIndex + 1) + " selected · inspect its conditional PMBM map", false);
    renderAll();
  }

  function setScan(scanIndex, message) {
    stopAuto(true);
    var next = Math.max(0, Math.min(S1.setup.poses.length - 1, Number(scanIndex)));
    var scene = sceneForScan(next);
    state.scanIndex = next;
    state.scanData = scene.scan;
    state.tracks = scene.tracks;
    state.measurements = scene.measurements;
    state.selectedMeasurement = 0;
    recompute(true);
    setStatus(message || ("S1 scan " + (next + 1) + " loaded" + (scene.scan.clutter ? " · fixed clutter MPC present" : "")), false);
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
    if (/^[1-9]$/.test(event.key) && Number(event.key) <= state.measurements.length) {
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

  function setParticle(value) {
    state.particleIndex = Math.max(0, Math.min(Number(value), PMBM_PARTICLES.length - 1));
    particleRange.value = String(state.particleIndex);
    particleValue.innerHTML = mathTex("n=" + (state.particleIndex + 1));
    setStatus("Particle " + (state.particleIndex + 1) + " selected · weight " + PMBM_PARTICLES[state.particleIndex].weight.toFixed(2), false);
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
      var order = ["assignment", "bp", "pmbm"];
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
  particleRange.addEventListener("input", function () { setParticle(particleRange.value); });

  scanRange.addEventListener("input", function () { setScan(scanRange.value); });
  document.getElementById("previous-scan").addEventListener("click", function () {
    setScan((state.scanIndex + S1.setup.poses.length - 1) % S1.setup.poses.length);
  });
  document.getElementById("next-scan").addEventListener("click", function () {
    setScan((state.scanIndex + 1) % S1.setup.poses.length);
  });
  document.getElementById("assignment-reset").addEventListener("click", function () { setScan(state.scanIndex, "Shared S1 measurement restored"); });
  document.getElementById("bp-step").addEventListener("click", function () {
    stopAuto(true);
    stepBp();
  });
  document.getElementById("bp-run").addEventListener("click", toggleAuto);
  document.getElementById("bp-reset").addEventListener("click", function () {
    stopAuto(true);
    state.factorStage = 0;
    setStatus("Reset to joint latent variables and priors", false);
    renderAll();
  });
  document.getElementById("bp-end").addEventListener("click", function () {
    stopAuto(true);
    state.factorStage = BP_STAGES.length - 1;
    setStatus("Marginal state and map beliefs assembled", false);
    renderAll();
  });
  document.getElementById("particle-one").addEventListener("click", function () { setParticle(0); });
  document.getElementById("particle-two").addEventListener("click", function () { setParticle(1); });
  document.getElementById("particle-three").addEventListener("click", function () { setParticle(2); });

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
      if (mode === "assignment") renderStage();
    }).observe(canvas);
  } else {
    window.addEventListener("resize", renderStage);
  }

  recompute(true);
  document.body.classList.add("mode-assignment");
  if (requestedMode !== "assignment") switchMode(requestedMode, false);
}());
