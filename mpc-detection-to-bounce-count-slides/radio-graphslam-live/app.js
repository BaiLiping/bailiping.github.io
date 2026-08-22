(function () {
  "use strict";

  var S1 = window.RadioSlamS1;
  if (!S1) throw new Error("Shared radio-SLAM setup S1 failed to load");
  var COLORS = {
    ink: "#16222e",
    soft: "#51606e",
    faint: "#8a97a3",
    line: "#d7dee5",
    grid: "#e9eef2",
    pose: "#1874b8",
    poseDeep: "#0d568c",
    poseSoft: "#e8f2fa",
    map: "#0e8f7e",
    mapDeep: "#0a6b5e",
    mapSoft: "#e3f2ef",
    radio: "#e8720c",
    radioDeep: "#b45607",
    danger: "#c22f2f",
    paper: "#ffffff"
  };

  function mathTex(source) {
    return '<span class="math-tex math-inline">\\(' + source + '\\)</span>';
  }

  var BS = S1.clone(S1.setup.bs);
  var TRUE_POSES = S1.clone(S1.setup.poses);
  var TRUE_VAS = S1.clone(S1.setup.virtualAnchors);
  var PRIOR = { x: 2.64, y: 6.34, heading: -15 * S1.DEG };
  var T = TRUE_POSES.length;
  var J = TRUE_VAS.length;
  var PATH_COUNT = S1.setup.paths.length;
  var DIM = T * 3 + J * 2;
  var WRONG_T = 3;
  var WRONG_PATH = 1;

  var sceneCanvas = document.getElementById("scene-canvas");
  var graphCanvas = document.getElementById("graph-canvas");
  var sceneCtx = sceneCanvas.getContext("2d");
  var graphCtx = graphCanvas.getContext("2d");
  var modelRange = document.getElementById("model-range");
  var modelFull = document.getElementById("model-full");
  var wrongAssociation = document.getElementById("wrong-association");
  var robustLoss = document.getElementById("robust-loss");
  var noiseRange = document.getElementById("noise-range");
  var noiseValue = document.getElementById("noise-value");
  var epochRange = document.getElementById("epoch-range");
  var epochValue = document.getElementById("epoch-value");
  var stepButton = document.getElementById("step-button");
  var solveButton = document.getElementById("solve-button");
  var resetButton = document.getElementById("reset-button");
  var costValue = document.getElementById("cost-value");
  var rmseValue = document.getElementById("rmse-value");
  var iterationValue = document.getElementById("iteration-value");
  var conditionValue = document.getElementById("condition-value");
  var statusEl = document.getElementById("status");
  var hintEl = document.getElementById("canvas-hint");

  var state = {
    vector: null,
    initial: null,
    odometry: null,
    measurements: null,
    iteration: 0,
    condition: Infinity,
    running: false,
    lastStep: 0
  };

  function wrap(angle) {
    return S1.wrap(angle);
  }

  function hypot2(x, y) {
    return Math.sqrt(x * x + y * y);
  }

  function cloneVector(vector) {
    return vector.slice();
  }

  function poseOffset(t) {
    return t * 3;
  }

  function vaOffset(j) {
    return T * 3 + j * 2;
  }

  function getPose(vector, t) {
    var k = poseOffset(t);
    return { x: vector[k], y: vector[k + 1], heading: vector[k + 2] };
  }

  function getVA(vector, j) {
    var k = vaOffset(j);
    return { x: vector[k], y: vector[k + 1] };
  }

  function setPose(vector, t, pose) {
    var k = poseOffset(t);
    vector[k] = pose.x;
    vector[k + 1] = pose.y;
    vector[k + 2] = wrap(pose.heading);
  }

  function setVA(vector, j, va) {
    var k = vaOffset(j);
    vector[k] = va.x;
    vector[k + 1] = va.y;
  }

  function relativeMotion(a, b) {
    var dx = b.x - a.x;
    var dy = b.y - a.y;
    var c = Math.cos(a.heading);
    var s = Math.sin(a.heading);
    return {
      x: c * dx + s * dy,
      y: -s * dx + c * dy,
      heading: wrap(b.heading - a.heading)
    };
  }

  function predictRadio(pose, va) {
    var dx = va.x - pose.x;
    var dy = va.y - pose.y;
    var range = hypot2(dx, dy);
    var aoa = wrap(Math.atan2(dy, dx) - pose.heading);
    var nx = va.x - BS.x;
    var ny = va.y - BS.y;
    var mx = 0.5 * (va.x + BS.x);
    var my = 0.5 * (va.y + BS.y);
    var denominator = nx * dx + ny * dy;
    var numerator = nx * (mx - pose.x) + ny * (my - pose.y);
    var s = Math.abs(denominator) < 1e-9 ? 0.5 : numerator / denominator;
    var point = { x: pose.x + s * dx, y: pose.y + s * dy };
    var aod = wrap(Math.atan2(point.y - BS.y, point.x - BS.x) - BS.heading);
    return { range: range, aoa: aoa, aod: aod, point: point, fold: s };
  }

  function predictDirect(pose) {
    var dx = BS.x - pose.x;
    var dy = BS.y - pose.y;
    return {
      range: hypot2(dx, dy),
      aoa: wrap(Math.atan2(dy, dx) - pose.heading),
      aod: wrap(Math.atan2(pose.y - BS.y, pose.x - BS.x) - BS.heading),
      point: null,
      fold: null
    };
  }

  function makeOdometry() {
    var noise = S1.odometryNoise;
    var result = [];
    for (var t = 0; t < T - 1; t += 1) {
      var motion = relativeMotion(TRUE_POSES[t], TRUE_POSES[t + 1]);
      result.push({
        x: motion.x + noise[t].x,
        y: motion.y + noise[t].y,
        heading: wrap(motion.heading + noise[t].heading)
      });
    }
    return result;
  }

  function makeMeasurements() {
    var scale = Number(noiseRange.value) / 100;
    return S1.allScans(scale).map(function (scan) { return scan.measurements; });
  }

  function makeInitialVector() {
    var vector = new Array(DIM).fill(0);
    var pose = { x: PRIOR.x, y: PRIOR.y, heading: PRIOR.heading };
    setPose(vector, 0, pose);
    for (var t = 0; t < T - 1; t += 1) {
      var u = state.odometry[t];
      var c = Math.cos(pose.heading);
      var s = Math.sin(pose.heading);
      pose = {
        x: pose.x + c * u.x - s * u.y + 0.10 * (t + 1),
        y: pose.y + s * u.x + c * u.y - 0.055 * (t + 1),
        heading: wrap(pose.heading + u.heading + 0.025)
      };
      setPose(vector, t + 1, pose);
    }
    setVA(vector, 0, { x: 2.75, y: 11.15 });
    setVA(vector, 1, { x: 14.15, y: 2.75 });
    return vector;
  }

  function associationFor(t, pathIndex) {
    if (pathIndex === 0) return -1;
    var vaIndex = pathIndex - 1;
    if (wrongAssociation.checked && t === WRONG_T && pathIndex === WRONG_PATH) return 1 - vaIndex;
    return vaIndex;
  }

  function predictForPath(vector, pose, t, pathIndex) {
    if (pathIndex === 0) return predictDirect(pose);
    return predictRadio(pose, getVA(vector, associationFor(t, pathIndex)));
  }

  function huberScale(group) {
    if (!robustLoss.checked) return 1;
    var norm = Math.sqrt(group.reduce(function (sum, value) { return sum + value * value; }, 0));
    var delta = 2.6;
    return norm <= delta || norm < 1e-12 ? 1 : Math.sqrt(delta / norm);
  }

  function residualVector(vector) {
    var residuals = [];
    var pose0 = getPose(vector, 0);
    residuals.push((pose0.x - PRIOR.x) / 0.32);
    residuals.push((pose0.y - PRIOR.y) / 0.32);
    residuals.push(wrap(pose0.heading - PRIOR.heading) / 0.16);

    for (var t = 0; t < T - 1; t += 1) {
      var a = getPose(vector, t);
      var b = getPose(vector, t + 1);
      var predictedMotion = relativeMotion(a, b);
      var measuredMotion = state.odometry[t];
      residuals.push((predictedMotion.x - measuredMotion.x) / 0.10);
      residuals.push((predictedMotion.y - measuredMotion.y) / 0.10);
      residuals.push(wrap(predictedMotion.heading - measuredMotion.heading) / 0.045);
    }

    for (var ti = 0; ti < T; ti += 1) {
      var pose = getPose(vector, ti);
      for (var pathIndex = 0; pathIndex < PATH_COUNT; pathIndex += 1) {
        var prediction = predictForPath(vector, pose, ti, pathIndex);
        var measurement = state.measurements[ti][pathIndex];
        var group = [(prediction.range - measurement.range) / S1.setup.noise.rangeSigma];
        if (modelFull.checked) {
          group.push(wrap(prediction.aoa - measurement.aoa) / S1.setup.noise.aoaSigma);
          group.push(wrap(prediction.aod - measurement.aod) / S1.setup.noise.aodSigma);
        }
        var scale = huberScale(group);
        for (var g = 0; g < group.length; g += 1) residuals.push(scale * group[g]);
      }
    }
    return residuals;
  }

  function cost(vector) {
    var residuals = residualVector(vector);
    var value = 0;
    for (var i = 0; i < residuals.length; i += 1) value += residuals[i] * residuals[i];
    return 0.5 * value;
  }

  function symmetricCondition(matrix) {
    var A = matrix.map(function (row) { return row.slice(); });
    var n = A.length;
    for (var iteration = 0; iteration < 90; iteration += 1) {
      var p = 0, q = 1, maximum = 0;
      for (var i = 0; i < n; i += 1) {
        for (var j = i + 1; j < n; j += 1) {
          var magnitude = Math.abs(A[i][j]);
          if (magnitude > maximum) { maximum = magnitude; p = i; q = j; }
        }
      }
      if (maximum < 1e-8) break;
      var angle = 0.5 * Math.atan2(2 * A[p][q], A[q][q] - A[p][p]);
      var c = Math.cos(angle), s = Math.sin(angle);
      var app = A[p][p], aqq = A[q][q], apq = A[p][q];
      for (var k = 0; k < n; k += 1) {
        if (k === p || k === q) continue;
        var akp = A[k][p], akq = A[k][q];
        A[k][p] = A[p][k] = c * akp - s * akq;
        A[k][q] = A[q][k] = s * akp + c * akq;
      }
      A[p][p] = c * c * app - 2 * s * c * apq + s * s * aqq;
      A[q][q] = s * s * app + 2 * s * c * apq + c * c * aqq;
      A[p][q] = A[q][p] = 0;
    }
    var eigenvalues = A.map(function (row, index) { return Math.max(0, row[index]); });
    var maximumEigenvalue = Math.max.apply(null, eigenvalues);
    var threshold = Math.max(1e-10, maximumEigenvalue * 1e-10);
    var positive = eigenvalues.filter(function (value) { return value > threshold; });
    if (!positive.length) return Infinity;
    return maximumEigenvalue / Math.min.apply(null, positive);
  }

  function normalEquations(vector, damping) {
    var base = residualVector(vector);
    var m = base.length;
    var jacobian = Array.from({ length: m }, function () { return new Array(DIM).fill(0); });
    for (var column = 0; column < DIM; column += 1) {
      var perturbed = cloneVector(vector);
      var isAngle = column < T * 3 && column % 3 === 2;
      var epsilon = isAngle ? 1e-5 : 1e-4;
      perturbed[column] += epsilon;
      if (isAngle) perturbed[column] = wrap(perturbed[column]);
      var shifted = residualVector(perturbed);
      for (var row = 0; row < m; row += 1) jacobian[row][column] = (shifted[row] - base[row]) / epsilon;
    }

    var H = Array.from({ length: DIM }, function () { return new Array(DIM).fill(0); });
    var g = new Array(DIM).fill(0);
    for (var r = 0; r < m; r += 1) {
      for (var c1 = 0; c1 < DIM; c1 += 1) {
        var j1 = jacobian[r][c1];
        g[c1] += j1 * base[r];
        for (var c2 = c1; c2 < DIM; c2 += 1) H[c1][c2] += j1 * jacobian[r][c2];
      }
    }
    for (var i = 0; i < DIM; i += 1) {
      for (var j = 0; j < i; j += 1) H[i][j] = H[j][i];
    }
    var condition = symmetricCondition(H);
    for (var diagonalIndex = 0; diagonalIndex < DIM; diagonalIndex += 1) {
      H[diagonalIndex][diagonalIndex] += damping * Math.max(1, H[diagonalIndex][diagonalIndex]);
    }
    return { H: H, g: g, condition: condition };
  }

  function solveLinear(A, b) {
    var n = b.length;
    var matrix = A.map(function (row, index) { return row.slice().concat([b[index]]); });
    for (var column = 0; column < n; column += 1) {
      var pivot = column;
      var pivotMagnitude = Math.abs(matrix[pivot][column]);
      for (var row = column + 1; row < n; row += 1) {
        var magnitude = Math.abs(matrix[row][column]);
        if (magnitude > pivotMagnitude) {
          pivot = row;
          pivotMagnitude = magnitude;
        }
      }
      if (pivotMagnitude < 1e-10) return null;
      if (pivot !== column) {
        var swap = matrix[column];
        matrix[column] = matrix[pivot];
        matrix[pivot] = swap;
      }
      var diagonal = matrix[column][column];
      for (var c = column; c <= n; c += 1) matrix[column][c] /= diagonal;
      for (var r = 0; r < n; r += 1) {
        if (r === column) continue;
        var factor = matrix[r][column];
        if (Math.abs(factor) < 1e-15) continue;
        for (var cc = column; cc <= n; cc += 1) matrix[r][cc] -= factor * matrix[column][cc];
      }
    }
    return matrix.map(function (row) { return row[n]; });
  }

  function applyStep(vector, step, alpha) {
    var candidate = cloneVector(vector);
    for (var i = 0; i < DIM; i += 1) candidate[i] += alpha * step[i];
    for (var t = 0; t < T; t += 1) candidate[poseOffset(t) + 2] = wrap(candidate[poseOffset(t) + 2]);
    return candidate;
  }

  function gaussNewtonStep() {
    var oldCost = cost(state.vector);
    var dampingValues = [1e-4, 1e-3, 1e-2, 1e-1, 1];
    for (var d = 0; d < dampingValues.length; d += 1) {
      var system = normalEquations(state.vector, dampingValues[d]);
      var right = system.g.map(function (value) { return -value; });
      var step = solveLinear(system.H, right);
      if (!step) continue;
      var norm = Math.sqrt(step.reduce(function (sum, value) { return sum + value * value; }, 0));
      var alphas = [1, 0.5, 0.25, 0.125, 0.0625];
      for (var a = 0; a < alphas.length; a += 1) {
        var candidate = applyStep(state.vector, step, alphas[a]);
        var candidateCost = cost(candidate);
        if (isFinite(candidateCost) && candidateCost < oldCost - 1e-7) {
          state.vector = candidate;
          state.iteration += 1;
          state.condition = system.condition;
          state.lastStep = norm * alphas[a];
          return { accepted: true, cost: candidateCost, step: state.lastStep };
        }
      }
    }
    return { accepted: false, cost: oldCost, step: 0 };
  }

  function poseRMSE() {
    var sum = 0;
    for (var t = 0; t < T; t += 1) {
      var pose = getPose(state.vector, t);
      var dx = pose.x - TRUE_POSES[t].x;
      var dy = pose.y - TRUE_POSES[t].y;
      sum += dx * dx + dy * dy;
    }
    return Math.sqrt(sum / T);
  }

  function setStatus(text, warning) {
    statusEl.classList.toggle("is-warning", Boolean(warning));
    statusEl.lastChild.nodeValue = " " + text;
  }

  function updateMetrics() {
    var currentCost = cost(state.vector);
    costValue.textContent = currentCost < 1000 ? currentCost.toFixed(2) : currentCost.toExponential(2);
    rmseValue.textContent = poseRMSE().toFixed(2) + " m";
    iterationValue.textContent = String(state.iteration);
    conditionValue.textContent = isFinite(state.condition) ? state.condition.toExponential(1) : "—";
    noiseValue.textContent = (Number(noiseRange.value) / 100).toFixed(1) + "×";
    epochValue.innerHTML = mathTex("x_{" + (Number(epochRange.value) + 1) + "}");
  }

  function line(ctx, x1, y1, x2, y2, color, width, dash, alpha) {
    ctx.save();
    ctx.globalAlpha = alpha === undefined ? 1 : alpha;
    ctx.strokeStyle = color;
    ctx.lineWidth = width || 1;
    ctx.setLineDash(dash || []);
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
    ctx.restore();
  }

  function circle(ctx, x, y, radius, fill, stroke, width) {
    ctx.save();
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, 2 * Math.PI);
    ctx.fillStyle = fill;
    ctx.fill();
    if (stroke) {
      ctx.strokeStyle = stroke;
      ctx.lineWidth = width || 1;
      ctx.stroke();
    }
    ctx.restore();
  }

  function diamond(ctx, x, y, radius, fill, stroke) {
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(x, y - radius);
    ctx.lineTo(x + radius, y);
    ctx.lineTo(x, y + radius);
    ctx.lineTo(x - radius, y);
    ctx.closePath();
    ctx.fillStyle = fill;
    ctx.fill();
    ctx.strokeStyle = stroke || fill;
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.restore();
  }

  function label(ctx, text, x, y, color, size, align, weight) {
    ctx.save();
    ctx.fillStyle = color || COLORS.soft;
    ctx.font = (weight || 700) + " " + (size || 10) + "px Menlo, Consolas, monospace";
    ctx.textAlign = align || "left";
    ctx.textBaseline = "middle";
    ctx.fillText(text, x, y);
    ctx.restore();
  }

  function sceneTransform() {
    var minX = 0, maxX = 16.2, minY = 0, maxY = 13.4;
    var padding = 28;
    var width = sceneCanvas.width - 2 * padding;
    var height = sceneCanvas.height - 2 * padding;
    var scale = Math.min(width / (maxX - minX), height / (maxY - minY));
    var usedWidth = scale * (maxX - minX);
    var usedHeight = scale * (maxY - minY);
    var ox = padding + 0.5 * (width - usedWidth);
    var oy = padding + 0.5 * (height - usedHeight);
    return {
      point: function (x, y) {
        return { x: ox + (x - minX) * scale, y: sceneCanvas.height - oy - (y - minY) * scale };
      },
      scale: scale,
      minX: minX, maxX: maxX, minY: minY, maxY: maxY
    };
  }

  function wallSegment(va) {
    var nx = va.x - BS.x;
    var ny = va.y - BS.y;
    var norm = Math.max(hypot2(nx, ny), 1e-9);
    var dx = -ny / norm;
    var dy = nx / norm;
    var mx = 0.5 * (va.x + BS.x);
    var my = 0.5 * (va.y + BS.y);
    return {
      a: { x: mx - 18 * dx, y: my - 18 * dy },
      b: { x: mx + 18 * dx, y: my + 18 * dy }
    };
  }

  function drawPose(ctx, transform, pose, color, radius, alpha) {
    var point = transform.point(pose.x, pose.y);
    ctx.save();
    ctx.globalAlpha = alpha === undefined ? 1 : alpha;
    circle(ctx, point.x, point.y, radius, COLORS.paper, color, 2);
    var length = 13;
    line(ctx, point.x, point.y, point.x + length * Math.cos(-pose.heading), point.y + length * Math.sin(-pose.heading), color, 2);
    ctx.restore();
  }

  function drawScene() {
    var ctx = sceneCtx;
    var transform = sceneTransform();
    ctx.clearRect(0, 0, sceneCanvas.width, sceneCanvas.height);
    ctx.fillStyle = COLORS.paper;
    ctx.fillRect(0, 0, sceneCanvas.width, sceneCanvas.height);

    for (var gx = 0; gx <= 16; gx += 2) {
      var gxa = transform.point(gx, 0);
      var gxb = transform.point(gx, 13.4);
      line(ctx, gxa.x, gxa.y, gxb.x, gxb.y, COLORS.grid, 1);
      label(ctx, String(gx), gxa.x, sceneCanvas.height - 10, COLORS.faint, 8, "center", 400);
    }
    for (var gy = 0; gy <= 12; gy += 2) {
      var gya = transform.point(0, gy);
      var gyb = transform.point(16.2, gy);
      line(ctx, gya.x, gya.y, gyb.x, gyb.y, COLORS.grid, 1);
      label(ctx, String(gy), 10, gya.y, COLORS.faint, 8, "left", 400);
    }

    for (var trueWallIndex = 0; trueWallIndex < J; trueWallIndex += 1) {
      var trueWall = wallSegment(TRUE_VAS[trueWallIndex]);
      var twa = transform.point(trueWall.a.x, trueWall.a.y);
      var twb = transform.point(trueWall.b.x, trueWall.b.y);
      line(ctx, twa.x, twa.y, twb.x, twb.y, COLORS.faint, 2, [7, 6], 0.48);
    }

    for (var wallIndex = 0; wallIndex < J; wallIndex += 1) {
      var estimatedWall = wallSegment(getVA(state.vector, wallIndex));
      var ewa = transform.point(estimatedWall.a.x, estimatedWall.a.y);
      var ewb = transform.point(estimatedWall.b.x, estimatedWall.b.y);
      line(ctx, ewa.x, ewa.y, ewb.x, ewb.y, wallIndex === 0 ? COLORS.map : COLORS.poseDeep, 3, [], 0.86);
    }

    for (var pathType = 0; pathType < 2; pathType += 1) {
      var sourceVector = pathType === 0 ? null : state.initial;
      var color = pathType === 0 ? COLORS.faint : COLORS.poseSoft;
      var dash = pathType === 0 ? [5, 5] : [3, 4];
      var width = pathType === 0 ? 1.5 : 2;
      var previous = null;
      for (var t = 0; t < T; t += 1) {
        var pose = sourceVector ? getPose(sourceVector, t) : TRUE_POSES[t];
        var p = transform.point(pose.x, pose.y);
        if (previous) line(ctx, previous.x, previous.y, p.x, p.y, color, width, dash, pathType === 0 ? 0.62 : 0.9);
        previous = p;
      }
    }

    var selectedEpoch = Number(epochRange.value);
    var selectedPose = getPose(state.vector, selectedEpoch);
    var bsPoint = transform.point(BS.x, BS.y);
    var uePoint = transform.point(selectedPose.x, selectedPose.y);
    for (var pathIndex = 0; pathIndex < PATH_COUNT; pathIndex += 1) {
      var prediction = predictForPath(state.vector, selectedPose, selectedEpoch, pathIndex);
      var expectedVA = pathIndex - 1;
      var associatedVA = associationFor(selectedEpoch, pathIndex);
      var wrong = pathIndex > 0 && associatedVA !== expectedVA;
      var routeColor = wrong ? COLORS.danger : (pathIndex === 0 ? COLORS.radio : (pathIndex === 1 ? COLORS.map : COLORS.pose));
      if (pathIndex === 0) {
        line(ctx, bsPoint.x, bsPoint.y, uePoint.x, uePoint.y, routeColor, 4, [], 0.9);
        label(ctx, "LoS", 0.5 * (bsPoint.x + uePoint.x) + 5, 0.5 * (bsPoint.y + uePoint.y) - 7, routeColor, 8, "left", 700);
        continue;
      }
      var va = getVA(state.vector, associatedVA);
      var reflectionPoint = transform.point(prediction.point.x, prediction.point.y);
      var vaPoint = transform.point(va.x, va.y);
      line(ctx, bsPoint.x, bsPoint.y, reflectionPoint.x, reflectionPoint.y, routeColor, 4, [], 0.9);
      line(ctx, reflectionPoint.x, reflectionPoint.y, uePoint.x, uePoint.y, routeColor, 4, [], 0.9);
      line(ctx, uePoint.x, uePoint.y, vaPoint.x, vaPoint.y, routeColor, 1.5, [5, 4], 0.45);
      circle(ctx, reflectionPoint.x, reflectionPoint.y, 4.5, COLORS.paper, routeColor, 2);
      label(ctx, "P" + (pathIndex === 1 ? "A" : "B"), reflectionPoint.x + 7, reflectionPoint.y - 8, routeColor, 8, "left", 700);
    }
    if (state.measurements[selectedEpoch].length > PATH_COUNT) {
      label(ctx, "× clutter z4 · no geometry factor", sceneCanvas.width - 12, 13, COLORS.danger, 8, "right", 700);
    }

    var previousEstimate = null;
    for (var estimateIndex = 0; estimateIndex < T; estimateIndex += 1) {
      var estimatePose = getPose(state.vector, estimateIndex);
      var estimatePoint = transform.point(estimatePose.x, estimatePose.y);
      if (previousEstimate) line(ctx, previousEstimate.x, previousEstimate.y, estimatePoint.x, estimatePoint.y, COLORS.pose, 3);
      drawPose(ctx, transform, estimatePose, COLORS.pose, estimateIndex === selectedEpoch ? 6.5 : 5, 1);
      label(ctx, "x" + (estimateIndex + 1), estimatePoint.x + 8, estimatePoint.y - 9, COLORS.poseDeep, 8, "left", 700);
      previousEstimate = estimatePoint;
    }

    var bs = transform.point(BS.x, BS.y);
    ctx.fillStyle = COLORS.ink;
    ctx.fillRect(bs.x - 6, bs.y - 6, 12, 12);
    label(ctx, "known BS", bs.x + 10, bs.y + 12, COLORS.ink, 9, "left", 700);

    for (var j = 0; j < J; j += 1) {
      var trueVA = transform.point(TRUE_VAS[j].x, TRUE_VAS[j].y);
      line(ctx, trueVA.x - 5, trueVA.y, trueVA.x + 5, trueVA.y, COLORS.faint, 1.5, [], 0.65);
      line(ctx, trueVA.x, trueVA.y - 5, trueVA.x, trueVA.y + 5, COLORS.faint, 1.5, [], 0.65);
      var estimatedVA = transform.point(getVA(state.vector, j).x, getVA(state.vector, j).y);
      diamond(ctx, estimatedVA.x, estimatedVA.y, 7, j === 0 ? COLORS.map : COLORS.poseDeep, COLORS.paper);
      label(ctx, j === 0 ? "vA" : "vB", estimatedVA.x + 10, estimatedVA.y - 9, j === 0 ? COLORS.mapDeep : COLORS.poseDeep, 9, "left", 700);
    }

    label(ctx, modelFull.checked ? "S1 geometry residual: [cτ, φ, ψ] · α grades hypotheses" : "S1 geometry residual: [cτ]", 10, 13, COLORS.poseDeep, 9, "left", 700);
  }

  function drawGraph() {
    var ctx = graphCtx;
    ctx.clearRect(0, 0, graphCanvas.width, graphCanvas.height);
    ctx.fillStyle = COLORS.paper;
    ctx.fillRect(0, 0, graphCanvas.width, graphCanvas.height);
    label(ctx, "FACTOR GRAPH", 14, 18, COLORS.poseDeep, 9, "left", 700);
    label(ctx, "continuous solve | fixed A,Q", 14, 34, COLORS.faint, 8, "left", 400);

    var poseY = [68, 128, 188, 248, 308];
    var poseX = 72;
    var radioX = 130;
    var vaX = 218;
    var targetY = [72, 190, 304];

    for (var t = 0; t < T; t += 1) {
      if (t < T - 1) {
        var motionY = 0.5 * (poseY[t] + poseY[t + 1]);
        line(ctx, poseX, poseY[t] + 10, poseX, motionY - 5, COLORS.soft, 1.5, [], 0.65);
        line(ctx, poseX, motionY + 5, poseX, poseY[t + 1] - 10, COLORS.soft, 1.5, [], 0.65);
        ctx.fillStyle = COLORS.paper;
        ctx.strokeStyle = COLORS.soft;
        ctx.lineWidth = 1.5;
        ctx.fillRect(poseX - 5, motionY - 5, 10, 10);
        ctx.strokeRect(poseX - 5, motionY - 5, 10, 10);
      }
    }

    ctx.fillStyle = COLORS.poseSoft;
    ctx.strokeStyle = COLORS.poseDeep;
    ctx.lineWidth = 1.5;
    ctx.fillRect(18, poseY[0] - 6, 12, 12);
    ctx.strokeRect(18, poseY[0] - 6, 12, 12);
    line(ctx, 30, poseY[0], poseX - 10, poseY[0], COLORS.poseDeep, 1.5);
    label(ctx, "prior", 13, poseY[0] - 14, COLORS.poseDeep, 7, "left", 700);

    var selectedEpoch = Number(epochRange.value);
    for (var ti = 0; ti < T; ti += 1) {
      circle(ctx, poseX, poseY[ti], ti === selectedEpoch ? 10 : 8, ti === selectedEpoch ? COLORS.pose : COLORS.paper, COLORS.poseDeep, 2);
      label(ctx, "x" + (ti + 1), poseX, poseY[ti] + 1, ti === selectedEpoch ? COLORS.paper : COLORS.poseDeep, 8, "center", 700);
      for (var pathIndex = 0; pathIndex < PATH_COUNT; pathIndex += 1) {
        var associatedVA = associationFor(ti, pathIndex);
        var wrong = pathIndex > 0 && associatedVA !== pathIndex - 1;
        var targetIndex = pathIndex === 0 ? 0 : associatedVA + 1;
        var yTarget = targetY[targetIndex];
        var yFactor = poseY[ti] + (pathIndex - 1) * 7;
        var factorColor = wrong ? COLORS.danger : (pathIndex === 0 ? COLORS.radio : (pathIndex === 1 ? COLORS.map : COLORS.pose));
        var alpha = ti === selectedEpoch ? 0.95 : 0.18;
        line(ctx, poseX + 9, yFactor, radioX - 5, yFactor, factorColor, ti === selectedEpoch ? 2 : 1, [], alpha);
        line(ctx, radioX + 5, yFactor, vaX - 11, yTarget, factorColor, ti === selectedEpoch ? 2 : 1, [], alpha);
        ctx.save();
        ctx.globalAlpha = alpha;
        ctx.fillStyle = COLORS.paper;
        ctx.strokeStyle = factorColor;
        ctx.lineWidth = ti === selectedEpoch ? 2 : 1;
        ctx.fillRect(radioX - 5, yFactor - 5, 10, 10);
        ctx.strokeRect(radioX - 5, yFactor - 5, 10, 10);
        ctx.restore();
      }
    }

    ctx.fillStyle = COLORS.ink;
    ctx.fillRect(vaX - 8, targetY[0] - 8, 16, 16);
    label(ctx, "b", vaX, targetY[0] + 1, COLORS.paper, 8, "center", 700);
    for (var j = 0; j < J; j += 1) {
      diamond(ctx, vaX, targetY[j + 1], 11, j === 0 ? COLORS.map : COLORS.poseDeep, COLORS.paper);
      label(ctx, j === 0 ? "vA" : "vB", vaX, targetY[j + 1] + 1, COLORS.paper, 8, "center", 700);
    }

    label(ctx, "■ known BS · ○ pose", 16, 354, COLORS.soft, 8, "left", 400);
    label(ctx, "◇ VA variable · □ factor", 16, 372, COLORS.soft, 8, "left", 400);
    label(ctx, "clutter scans: x2, x4 · no edge", 16, 390, COLORS.danger, 8, "left", 400);
    label(ctx, modelFull.checked ? "radio: τ + φ + ψ" : "radio: τ only", 16, 410, COLORS.poseDeep, 8, "left", 700);
  }

  function render() {
    updateMetrics();
    drawScene();
    drawGraph();
    var wrong = wrongAssociation.checked;
    hintEl.textContent = wrong
      ? "Red edge: the wall-A MPC at x" + (WRONG_T + 1) + " is attached to vB. Robust loss can protect the rest of the S1 graph."
      : "Setup S1: orange is LoS; green/blue are the two folded specular routes. Clutter at x2 and x4 has no geometry edge.";
  }

  function resetEstimate(message) {
    state.measurements = makeMeasurements();
    state.vector = makeInitialVector();
    state.initial = cloneVector(state.vector);
    state.iteration = 0;
    state.condition = Infinity;
    state.lastStep = 0;
    setStatus(message || "Estimate reset", wrongAssociation.checked);
    render();
  }

  function runOneStep() {
    if (state.running) return;
    var result = gaussNewtonStep();
    if (result.accepted) {
      setStatus(result.step < 1e-4 ? "Converged" : "Accepted Gauss–Newton step", false);
    } else {
      setStatus("No lower-cost step found", true);
    }
    render();
  }

  function runSolve() {
    if (state.running) return;
    state.running = true;
    stepButton.disabled = true;
    solveButton.disabled = true;
    var remaining = 14;
    function iterate() {
      var result = gaussNewtonStep();
      render();
      if (!result.accepted || result.step < 2e-5 || remaining <= 1) {
        state.running = false;
        stepButton.disabled = false;
        solveButton.disabled = false;
        var finishedText = result.accepted ? "Optimization converged" : (state.iteration > 0 ? "Converged — no lower-cost step" : "No lower-cost step found");
        setStatus(finishedText, state.iteration === 0 || (!robustLoss.checked && wrongAssociation.checked));
        render();
        return;
      }
      remaining -= 1;
      setStatus("Optimizing… iteration " + state.iteration, false);
      window.requestAnimationFrame(iterate);
    }
    window.requestAnimationFrame(iterate);
  }

  function onModelChange() {
    resetEstimate(modelFull.checked ? "Full MPC factors enabled" : "Delay-only factors enabled");
  }

  modelRange.addEventListener("change", onModelChange);
  modelFull.addEventListener("change", onModelChange);
  wrongAssociation.addEventListener("change", function () {
    resetEstimate(wrongAssociation.checked ? "Wrong association injected" : "Associations restored");
  });
  robustLoss.addEventListener("change", function () {
    resetEstimate(robustLoss.checked ? "Huber loss enabled" : "Quadratic loss enabled");
  });
  noiseRange.addEventListener("input", function () {
    resetEstimate("Measurements regenerated");
  });
  epochRange.addEventListener("input", render);
  stepButton.addEventListener("click", runOneStep);
  solveButton.addEventListener("click", runSolve);
  resetButton.addEventListener("click", function () { resetEstimate("Estimate reset"); });
  sceneCanvas.addEventListener("keydown", function (event) {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    event.preventDefault();
    var next = Number(epochRange.value) + (event.key === "ArrowRight" ? 1 : -1);
    epochRange.value = String(Math.max(0, Math.min(T - 1, next)));
    render();
  });

  state.odometry = makeOdometry();
  resetEstimate("Ready");
})();
