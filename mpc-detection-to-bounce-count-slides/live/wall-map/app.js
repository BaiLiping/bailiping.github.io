(function () {
  "use strict";

  var COLORS = {
    stage: "#091724",
    stageSoft: "#8ba1b3",
    grid: "#183044",
    pose: "#4ca6e8",
    poseDeep: "#1874b8",
    radio: "#ff9b47",
    radioDeep: "#e8720c",
    double: "#f06d9b",
    loop: "#ffd166",
    loopDeep: "#b57d00",
    map: ["#2bc8ae", "#b184eb", "#62a9e6", "#e8c25d"],
    truth: "#718697",
    white: "#eef6fb",
    ink: "#16222e",
    soft: "#51606e",
    faint: "#8a97a3",
    paper: "#fbfcfd",
    line: "#d7dee5"
  };

  var BS = [1.0, 0.9, 3.15];
  var TRAJECTORY = [
    [1.55, 1.45, 1.28],
    [2.95, 1.48, 1.34],
    [4.35, 1.88, 1.40],
    [5.75, 2.78, 1.47],
    [7.15, 4.05, 1.53],
    [8.55, 5.55, 1.58],
    [8.25, 7.05, 1.52],
    [6.55, 7.62, 1.44],
    [4.65, 6.95, 1.37],
    [3.05, 5.70, 1.31],
    [1.82, 3.82, 1.27],
    [1.62, 1.72, 1.29]
  ];

  var SURFACES = [
    {
      id: "A", name: "East wall", type: "vertical plane", first: 0,
      center: [11.7, 4.50, 2.05], u: [0, -1, 0], v: [0, 0, 1], hu: 4.50, hv: 2.05,
      error: [-1.35, 0.36, 0.22], angleError: 0.17
    },
    {
      id: "B", name: "North wall", type: "vertical plane", first: 1,
      center: [6.0, 8.8, 2.05], u: [1, 0, 0], v: [0, 0, 1], hu: 6.00, hv: 2.05,
      error: [0.34, -1.18, -0.20], angleError: -0.14
    }
  ];

  var OBSERVATIONS = [
    [0],
    [0, 1],
    [0, 1],
    [0, 1],
    [0, 1],
    [0, 1],
    [0, 1],
    [0, 1],
    [0, 1],
    [0, 1],
    [0, 1],
    [0, 1]
  ];

  var DOUBLE_OBSERVATIONS = [
    null, [1, 0], null, null, [1, 0], null,
    [0, 1], null, [0, 1], null, null, [0, 1]
  ];

  var state = {
    time: 0,
    phase: 0,
    playing: false,
    timer: null,
    camera: { yaw: -0.78, pitch: 0.48, distance: 18.5 },
    drag: null
  };

  var sceneCanvas = document.getElementById("scene-canvas");
  var graphCanvas = document.getElementById("graph-canvas");
  var sceneCtx = sceneCanvas.getContext("2d");
  var graphCtx = graphCanvas.getContext("2d");
  var timeRange = document.getElementById("time-range");
  var truthToggle = document.getElementById("truth-toggle");
  var raysToggle = document.getElementById("rays-toggle");
  var normalToggle = document.getElementById("normal-toggle");

  function clamp(value, low, high) {
    return Math.max(low, Math.min(high, value));
  }

  function add(a, b) {
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
  }

  function sub(a, b) {
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  }

  function scale(a, s) {
    return [a[0] * s, a[1] * s, a[2] * s];
  }

  function dot(a, b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  }

  function cross(a, b) {
    return [
      a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0]
    ];
  }

  function norm(a) {
    return Math.sqrt(dot(a, a));
  }

  function normalize(a) {
    var length = Math.max(norm(a), 1e-9);
    return scale(a, 1 / length);
  }

  function mix(a, b, t) {
    return [
      a[0] + (b[0] - a[0]) * t,
      a[1] + (b[1] - a[1]) * t,
      a[2] + (b[2] - a[2]) * t
    ];
  }

  function rotateZ(vector, angle) {
    var c = Math.cos(angle);
    var s = Math.sin(angle);
    return [c * vector[0] - s * vector[1], s * vector[0] + c * vector[1], vector[2]];
  }

  function withAlpha(hex, alpha) {
    var value = hex.replace("#", "");
    var r = parseInt(value.slice(0, 2), 16);
    var g = parseInt(value.slice(2, 4), 16);
    var b = parseInt(value.slice(4, 6), 16);
    return "rgba(" + r + "," + g + "," + b + "," + alpha + ")";
  }

  function resizeCanvas(canvas, context) {
    var rect = canvas.getBoundingClientRect();
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    var width = Math.max(1, Math.round(rect.width));
    var height = Math.max(1, Math.round(rect.height));
    if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
    }
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { width: width, height: height };
  }

  function observationCount(surfaceIndex) {
    var count = 0;
    for (var t = 0; t < state.time; t += 1) {
      if (OBSERVATIONS[t].indexOf(surfaceIndex) !== -1) count += 1;
      if (DOUBLE_OBSERVATIONS[t] && DOUBLE_OBSERVATIONS[t].indexOf(surfaceIndex) !== -1) count += 1;
    }
    if (state.phase >= 2) {
      if (OBSERVATIONS[state.time].indexOf(surfaceIndex) !== -1) count += 1;
      if (DOUBLE_OBSERVATIONS[state.time] && DOUBLE_OBSERVATIONS[state.time].indexOf(surfaceIndex) !== -1) count += 1;
    }
    return count;
  }

  function insertedScan(scanIndex) {
    return scanIndex < state.time || (scanIndex === state.time && state.phase >= 2);
  }

  function loopClosureInserted() {
    return state.time === TRAJECTORY.length - 1 && state.phase >= 2;
  }

  function optimizedScans() {
    return state.time + (state.phase >= 3 ? 1 : 0);
  }

  function surfaceQuality(surfaceIndex) {
    var count = observationCount(surfaceIndex);
    if (!count) return 0;
    var quality = 0.28 + 0.145 * count + 0.018 * optimizedScans();
    if (state.phase >= 3 && (OBSERVATIONS[state.time].indexOf(surfaceIndex) !== -1 ||
        (DOUBLE_OBSERVATIONS[state.time] && DOUBLE_OBSERVATIONS[state.time].indexOf(surfaceIndex) !== -1))) quality += 0.08;
    if (state.time === TRAJECTORY.length - 1 && state.phase >= 3) quality += 0.06;
    return clamp(quality, 0.30, 0.97);
  }

  function estimateSurface(surfaceIndex) {
    var surface = SURFACES[surfaceIndex];
    var quality = surfaceQuality(surfaceIndex);
    var center = add(surface.center, scale(surface.error, 1 - quality));
    var angle = surface.angleError * (1 - quality);
    return {
      center: center,
      u: normalize(rotateZ(surface.u, angle)),
      v: normalize(rotateZ(surface.v, angle)),
      hu: surface.hu,
      hv: surface.hv,
      quality: quality,
      uncertainty: clamp(1.35 - quality, 0.14, 1.1)
    };
  }

  function poseQuality() {
    var quality = 0.30 + optimizedScans() * 0.052;
    if (state.time >= 8) quality += 0.05;
    if (state.time === TRAJECTORY.length - 1 && state.phase >= 3) quality += 0.08;
    return clamp(quality, 0.30, 0.97);
  }

  function estimatePose(index) {
    var truth = TRAJECTORY[index];
    if (index === 0) return add(truth, [0.02, -0.015, 0.01]);
    var drift = [
      0.12 * index + 0.18 * Math.sin(index * 0.78),
      -0.075 * index + 0.22 * Math.cos(index * 0.61),
      0.045 * Math.sin(index * 0.84)
    ];
    var quality = poseQuality();
    var currentPenalty = index === state.time && state.phase < 3 ? 1.22 : 1;
    return add(truth, scale(drift, (1 - quality) * currentPenalty));
  }

  function surfaceCorners(surface) {
    return [
      add(add(surface.center, scale(surface.u, -surface.hu)), scale(surface.v, -surface.hv)),
      add(add(surface.center, scale(surface.u, surface.hu)), scale(surface.v, -surface.hv)),
      add(add(surface.center, scale(surface.u, surface.hu)), scale(surface.v, surface.hv)),
      add(add(surface.center, scale(surface.u, -surface.hu)), scale(surface.v, surface.hv))
    ];
  }

  function planeFromSurface(surface) {
    var normal = normalize(cross(surface.u, surface.v));
    return { normal: normal, d: dot(normal, surface.center) };
  }

  function mirrorAcrossPlane(point, surface) {
    var plane = planeFromSurface(surface);
    return sub(point, scale(plane.normal, 2 * (dot(plane.normal, point) - plane.d)));
  }

  function specularBounce(bs, ue, surface) {
    var mirroredTransmitter = mirrorAcrossPlane(bs, surface);
    var direction = sub(mirroredTransmitter, ue);
    var plane = planeFromSurface(surface);
    var denominator = dot(plane.normal, direction);
    if (Math.abs(denominator) < 1e-7) return surface.center;
    var lambda = (plane.d - dot(plane.normal, ue)) / denominator;
    return add(ue, scale(direction, lambda));
  }

  function linePlaneIntersection(a, b, surface) {
    var plane = planeFromSurface(surface);
    var direction = sub(b, a);
    var denominator = dot(plane.normal, direction);
    if (Math.abs(denominator) < 1e-8) return null;
    return add(a, scale(direction, (plane.d - dot(plane.normal, a)) / denominator));
  }

  function specularPath(bs, ue, orderedSurfaceIndices, surfaceStates) {
    var images = [bs];
    orderedSurfaceIndices.forEach(function (surfaceIndex) {
      images.push(mirrorAcrossPlane(images[images.length - 1], surfaceStates[surfaceIndex]));
    });
    var interactions = new Array(orderedSurfaceIndices.length);
    var nextPoint = ue;
    for (var r = orderedSurfaceIndices.length - 1; r >= 0; r -= 1) {
      var interaction = linePlaneIntersection(nextPoint, images[r + 1], surfaceStates[orderedSurfaceIndices[r]]);
      if (!interaction) return null;
      interactions[r] = interaction;
      nextPoint = interaction;
    }
    return [bs].concat(interactions, [ue]);
  }

  function makeProjector(size) {
    var center = [6.1, 4.55, 1.7];
    var camera = state.camera;
    var eye = add(center, [
      camera.distance * Math.cos(camera.pitch) * Math.cos(camera.yaw),
      camera.distance * Math.cos(camera.pitch) * Math.sin(camera.yaw),
      camera.distance * Math.sin(camera.pitch)
    ]);
    var forward = normalize(sub(center, eye));
    var right = normalize(cross(forward, [0, 0, 1]));
    var up = normalize(cross(right, forward));
    var focal = Math.min(size.width, size.height) * 1.28;
    return function (point) {
      var relative = sub(point, eye);
      var depth = dot(relative, forward);
      if (depth < 0.2) return null;
      return {
        x: size.width * 0.5 + focal * dot(relative, right) / depth,
        y: size.height * 0.52 - focal * dot(relative, up) / depth,
        depth: depth
      };
    };
  }

  function line3D(context, project, a, b, color, width, dash, alpha) {
    var pa = project(a);
    var pb = project(b);
    if (!pa || !pb) return;
    context.save();
    context.globalAlpha = alpha === undefined ? 1 : alpha;
    context.strokeStyle = color;
    context.lineWidth = width || 1;
    context.setLineDash(dash || []);
    context.beginPath();
    context.moveTo(pa.x, pa.y);
    context.lineTo(pb.x, pb.y);
    context.stroke();
    context.restore();
  }

  function arrow3D(context, project, a, b, color, width, alpha) {
    var pa = project(a);
    var pb = project(b);
    if (!pa || !pb) return;
    context.save();
    context.globalAlpha = alpha === undefined ? 1 : alpha;
    context.strokeStyle = color;
    context.fillStyle = color;
    context.lineWidth = width || 1.5;
    context.beginPath();
    context.moveTo(pa.x, pa.y);
    context.lineTo(pb.x, pb.y);
    context.stroke();
    var angle = Math.atan2(pb.y - pa.y, pb.x - pa.x);
    var head = 6;
    context.beginPath();
    context.moveTo(pb.x, pb.y);
    context.lineTo(pb.x - head * Math.cos(angle - 0.48), pb.y - head * Math.sin(angle - 0.48));
    context.lineTo(pb.x - head * Math.cos(angle + 0.48), pb.y - head * Math.sin(angle + 0.48));
    context.closePath();
    context.fill();
    context.restore();
  }

  function polyline3D(context, project, points, color, width, dash, alpha) {
    var projected = points.map(project).filter(Boolean);
    if (projected.length < 2) return;
    context.save();
    context.globalAlpha = alpha === undefined ? 1 : alpha;
    context.strokeStyle = color;
    context.lineWidth = width || 1;
    context.lineJoin = "round";
    context.lineCap = "round";
    context.setLineDash(dash || []);
    context.beginPath();
    context.moveTo(projected[0].x, projected[0].y);
    for (var i = 1; i < projected.length; i += 1) context.lineTo(projected[i].x, projected[i].y);
    context.stroke();
    context.restore();
  }

  function drawLabel(context, text, point, color, align, size) {
    if (!point) return;
    context.save();
    context.fillStyle = color;
    context.font = "700 " + (size || 10) + "px ui-monospace, Menlo, monospace";
    context.textAlign = align || "left";
    context.textBaseline = "middle";
    context.fillText(text, point.x, point.y);
    context.restore();
  }

  function drawSceneLoop(context, project, firstPose, lastPose) {
    var first = project(firstPose);
    var last = project(lastPose);
    if (!first || !last) return;
    var controlY = Math.min(first.y, last.y) - 52;
    var factor = {
      x: (first.x + last.x) / 2,
      y: (first.y + last.y) * 0.125 + controlY * 0.75
    };
    context.save();
    context.strokeStyle = COLORS.loop;
    context.lineWidth = 2.4;
    context.setLineDash([7, 4]);
    context.beginPath();
    context.moveTo(first.x, first.y);
    context.bezierCurveTo(first.x, controlY, last.x, controlY, last.x, last.y);
    context.stroke();
    context.setLineDash([]);
    context.fillStyle = COLORS.stage;
    context.strokeStyle = COLORS.loop;
    context.lineWidth = 1.6;
    context.fillRect(factor.x - 4, factor.y - 4, 8, 8);
    context.strokeRect(factor.x - 4, factor.y - 4, 8, 8);
    context.restore();
    drawLabel(context, "floop(x" + (TRAJECTORY.length - 1) + ", x0)", { x: factor.x, y: factor.y - 12 }, COLORS.loop, "center", 9);
  }

  function drawDot(context, point, radius, fill, stroke, width) {
    if (!point) return;
    context.save();
    context.beginPath();
    context.arc(point.x, point.y, radius, 0, Math.PI * 2);
    context.fillStyle = fill;
    context.fill();
    if (stroke) {
      context.strokeStyle = stroke;
      context.lineWidth = width || 1;
      context.stroke();
    }
    context.restore();
  }

  function drawSurface(context, project, surface, color, fillAlpha, lineWidth, dash) {
    var projected = surfaceCorners(surface).map(project);
    if (projected.some(function (point) { return !point; })) return;
    context.save();
    context.fillStyle = withAlpha(color, fillAlpha);
    context.strokeStyle = color;
    context.lineWidth = lineWidth || 1.5;
    context.setLineDash(dash || []);
    context.beginPath();
    context.moveTo(projected[0].x, projected[0].y);
    for (var i = 1; i < projected.length; i += 1) context.lineTo(projected[i].x, projected[i].y);
    context.closePath();
    context.fill();
    context.stroke();
    context.restore();
  }

  function drawScene() {
    var size = resizeCanvas(sceneCanvas, sceneCtx);
    var project = makeProjector(size);
    sceneCtx.clearRect(0, 0, size.width, size.height);
    sceneCtx.fillStyle = COLORS.stage;
    sceneCtx.fillRect(0, 0, size.width, size.height);

    var gradient = sceneCtx.createRadialGradient(size.width * 0.54, size.height * 0.42, 10, size.width * 0.54, size.height * 0.42, size.width * 0.68);
    gradient.addColorStop(0, "rgba(31,67,91,.42)");
    gradient.addColorStop(1, "rgba(9,23,36,0)");
    sceneCtx.fillStyle = gradient;
    sceneCtx.fillRect(0, 0, size.width, size.height);

    for (var x = 0; x <= 12; x += 1) line3D(sceneCtx, project, [x, 0, 0], [x, 9, 0], COLORS.grid, x % 2 === 0 ? 1.1 : 0.65, [], x % 2 === 0 ? 0.9 : 0.55);
    for (var y = 0; y <= 9; y += 1) line3D(sceneCtx, project, [0, y, 0], [12, y, 0], COLORS.grid, y % 2 === 0 ? 1.1 : 0.65, [], y % 2 === 0 ? 0.9 : 0.55);
    [[0, 0], [12, 0], [12, 9], [0, 9]].forEach(function (corner) {
      line3D(sceneCtx, project, [corner[0], corner[1], 0], [corner[0], corner[1], 4.2], COLORS.grid, 1, [4, 5], 0.62);
    });

    line3D(sceneCtx, project, [0, 0, 0], [2.0, 0, 0], "#dc5f5f", 2, [], 0.9);
    line3D(sceneCtx, project, [0, 0, 0], [0, 2.0, 0], "#67c889", 2, [], 0.9);
    line3D(sceneCtx, project, [0, 0, 0], [0, 0, 2.0], "#6fa8ed", 2, [], 0.9);
    var xLabel = project([2.2, 0, 0]);
    var yLabel = project([0, 2.2, 0]);
    var zLabel = project([0, 0, 2.2]);
    drawLabel(sceneCtx, "x", xLabel, "#dc7a7a", "center", 10);
    drawLabel(sceneCtx, "y", yLabel, "#7bd39a", "center", 10);
    drawLabel(sceneCtx, "z", zLabel, "#86b9f2", "center", 10);

    if (truthToggle.checked) {
      SURFACES.forEach(function (surface) {
        drawSurface(sceneCtx, project, surface, COLORS.truth, 0.025, 1.1, [5, 5]);
      });
      var trueVisible = TRAJECTORY.slice(0, state.time + 1);
      polyline3D(sceneCtx, project, trueVisible, COLORS.truth, 1.2, [5, 5], 0.62);
    }

    SURFACES.forEach(function (surface, surfaceIndex) {
      var count = observationCount(surfaceIndex);
      if (!count) return;
      var estimate = estimateSurface(surfaceIndex);
      var uncertainty = {
        center: estimate.center,
        u: estimate.u,
        v: estimate.v,
        hu: estimate.hu + estimate.uncertainty,
        hv: estimate.hv + estimate.uncertainty * 0.55
      };
      drawSurface(sceneCtx, project, uncertainty, COLORS.map[surfaceIndex], 0.018, 1, [3, 5]);
      drawSurface(sceneCtx, project, estimate, COLORS.map[surfaceIndex], 0.16, 2.1, []);
      var surfacePoint = project(add(estimate.center, scale(estimate.v, estimate.hv + 0.28)));
      drawLabel(sceneCtx, "π" + surface.id + " · " + count + " hits", surfacePoint, COLORS.map[surfaceIndex], "center", 10);
      if (normalToggle.checked) {
        var plane = planeFromSurface(estimate);
        var normalEnd = add(estimate.center, scale(plane.normal, 1.05));
        arrow3D(sceneCtx, project, estimate.center, normalEnd, COLORS.map[surfaceIndex], 1.5, 0.92);
        var normalPoint = project(normalEnd);
        drawLabel(sceneCtx, "n" + surface.id, normalPoint ? { x: normalPoint.x + 6, y: normalPoint.y - 6 } : null, COLORS.map[surfaceIndex], "left", 8);
      }
    });

    var estimatedTrajectory = [];
    for (var t = 0; t <= state.time; t += 1) estimatedTrajectory.push(estimatePose(t));
    polyline3D(sceneCtx, project, estimatedTrajectory, COLORS.pose, 3.4, [], 0.95);
    if (loopClosureInserted()) drawSceneLoop(sceneCtx, project, estimatedTrajectory[0], estimatedTrajectory[estimatedTrajectory.length - 1]);

    if (raysToggle.checked && state.phase >= 1) {
      var ueTruth = TRAJECTORY[state.time];
      line3D(sceneCtx, project, BS, ueTruth, COLORS.radio, 2.4, [], 0.88);
      OBSERVATIONS[state.time].forEach(function (surfaceIndex) {
        var bounce = specularBounce(BS, ueTruth, SURFACES[surfaceIndex]);
        polyline3D(sceneCtx, project, [BS, bounce, ueTruth], COLORS.radio, 2.7, [], 0.92);
        var bouncePoint = project(bounce);
        drawDot(sceneCtx, bouncePoint, 3.5, COLORS.stage, COLORS.radio, 1.5);
        drawLabel(sceneCtx, "q" + state.time + "," + SURFACES[surfaceIndex].id, bouncePoint ? { x: bouncePoint.x + 6, y: bouncePoint.y - 7 } : null, COLORS.radio, "left", 8);
        if (state.phase >= 2 && observationCount(surfaceIndex)) {
          var estimatedWall = estimateSurface(surfaceIndex);
          var estimatedUE = estimatePose(state.time);
          var predictedBounce = specularBounce(BS, estimatedUE, estimatedWall);
          polyline3D(sceneCtx, project, [BS, predictedBounce, estimatedUE], COLORS.map[surfaceIndex], 1.5, [5, 4], 0.82);
        }
      });
      var orderedPair = DOUBLE_OBSERVATIONS[state.time];
      if (orderedPair) {
        var trueDoublePath = specularPath(BS, ueTruth, orderedPair, SURFACES);
        if (trueDoublePath) {
          polyline3D(sceneCtx, project, trueDoublePath, COLORS.double, 3.2, [], 0.96);
          trueDoublePath.slice(1, -1).forEach(function (bounce, bounceIndex) {
            var bouncePoint = project(bounce);
            drawDot(sceneCtx, bouncePoint, 4.2, COLORS.stage, COLORS.double, 1.8);
            drawLabel(sceneCtx, "q" + (bounceIndex + 1), bouncePoint ? { x: bouncePoint.x + 7, y: bouncePoint.y - 8 } : null, COLORS.double, "left", 9);
          });
          var orderPoint = project(trueDoublePath[2]);
          var orderLabel = SURFACES[orderedPair[0]].id + "→" + SURFACES[orderedPair[1]].id;
          drawLabel(sceneCtx, "S2 " + orderLabel, orderPoint ? { x: orderPoint.x + 9, y: orderPoint.y + 11 } : null, COLORS.double, "left", 9);

          if (state.phase >= 2) {
            var estimatedWalls = SURFACES.map(function (_, surfaceIndex) { return estimateSurface(surfaceIndex); });
            var estimatedDoublePath = specularPath(BS, estimatePose(state.time), orderedPair, estimatedWalls);
            if (estimatedDoublePath) polyline3D(sceneCtx, project, estimatedDoublePath, COLORS.double, 1.6, [5, 4], 0.76);
          }
        }
      }
    }

    estimatedTrajectory.forEach(function (pose, poseIndex) {
      var point = project(pose);
      var isCurrent = poseIndex === state.time;
      drawDot(sceneCtx, point, isCurrent ? 6.2 : 4.3, isCurrent ? COLORS.pose : COLORS.stage, COLORS.pose, isCurrent ? 2.2 : 1.6);
      if (isCurrent || poseIndex === 0 || poseIndex % 3 === 0) {
        drawLabel(sceneCtx, "x" + poseIndex, point ? { x: point.x + 8, y: point.y - 9 } : null, isCurrent ? COLORS.white : COLORS.pose, "left", 9);
      }
    });

    var bsPoint = project(BS);
    if (bsPoint) {
      sceneCtx.save();
      sceneCtx.fillStyle = COLORS.white;
      sceneCtx.fillRect(bsPoint.x - 5, bsPoint.y - 5, 10, 10);
      sceneCtx.strokeStyle = COLORS.stage;
      sceneCtx.strokeRect(bsPoint.x - 5, bsPoint.y - 5, 10, 10);
      sceneCtx.restore();
      drawLabel(sceneCtx, "known BS", { x: bsPoint.x + 9, y: bsPoint.y + 10 }, COLORS.white, "left", 9);
    }
  }

  function graphLine(context, a, b, color, width, alpha) {
    context.save();
    context.globalAlpha = alpha === undefined ? 1 : alpha;
    context.strokeStyle = color;
    context.lineWidth = width || 1;
    context.beginPath();
    context.moveTo(a.x, a.y);
    context.lineTo(b.x, b.y);
    context.stroke();
    context.restore();
  }

  function graphFactor(context, a, b, color, alpha, squareSize) {
    var factor = { x: a.x + (b.x - a.x) * 0.42, y: a.y + (b.y - a.y) * 0.42 };
    graphLine(context, a, factor, color, 1, alpha);
    graphLine(context, factor, b, color, 1, alpha);
    var side = squareSize || 6;
    context.save();
    context.globalAlpha = alpha;
    context.fillStyle = COLORS.paper;
    context.strokeStyle = color;
    context.lineWidth = 1.2;
    context.fillRect(factor.x - side / 2, factor.y - side / 2, side, side);
    context.strokeRect(factor.x - side / 2, factor.y - side / 2, side, side);
    context.restore();
  }

  function graphPairFactor(context, pose, firstWall, secondWall, label, alpha, squareSize) {
    var mapMidpoint = {
      x: (firstWall.x + secondWall.x) / 2,
      y: (firstWall.y + secondWall.y) / 2
    };
    var factor = {
      x: pose.x + (mapMidpoint.x - pose.x) * 0.48,
      y: pose.y + (mapMidpoint.y - pose.y) * 0.48
    };
    graphLine(context, pose, factor, COLORS.double, 1.3, alpha);
    graphLine(context, factor, firstWall, COLORS.double, 1.3, alpha);
    graphLine(context, factor, secondWall, COLORS.double, 1.3, alpha);
    var side = squareSize || 7;
    context.save();
    context.globalAlpha = alpha;
    context.fillStyle = COLORS.paper;
    context.strokeStyle = COLORS.double;
    context.lineWidth = 1.5;
    context.fillRect(factor.x - side / 2, factor.y - side / 2, side, side);
    context.strokeRect(factor.x - side / 2, factor.y - side / 2, side, side);
    context.restore();
    if (alpha > 0.5) canvasLabel(context, "S2 " + label, factor.x + 7, factor.y - 8, COLORS.double, 7, "left");
  }

  function graphLoopFactor(context, firstPose, lastPose) {
    var controlY = Math.max(8, Math.min(firstPose.y, lastPose.y) - 52);
    var factor = {
      x: (firstPose.x + lastPose.x) / 2,
      y: (firstPose.y + lastPose.y) * 0.125 + controlY * 0.75
    };
    context.save();
    context.strokeStyle = COLORS.loopDeep;
    context.lineWidth = 2.1;
    context.setLineDash([6, 4]);
    context.beginPath();
    context.moveTo(firstPose.x, firstPose.y);
    context.bezierCurveTo(firstPose.x, controlY, lastPose.x, controlY, lastPose.x, lastPose.y);
    context.stroke();
    context.setLineDash([]);
    context.fillStyle = COLORS.paper;
    context.strokeStyle = COLORS.loopDeep;
    context.lineWidth = 1.5;
    context.fillRect(factor.x - 4, factor.y - 4, 8, 8);
    context.strokeRect(factor.x - 4, factor.y - 4, 8, 8);
    context.restore();
    canvasLabel(context, "floop", factor.x, factor.y - 10, COLORS.loopDeep, 7);
  }

  function graphCircle(context, point, radius, fill, stroke) {
    context.save();
    context.beginPath();
    context.arc(point.x, point.y, radius, 0, Math.PI * 2);
    context.fillStyle = fill;
    context.fill();
    context.strokeStyle = stroke;
    context.lineWidth = 1.6;
    context.stroke();
    context.restore();
  }

  function graphWallNode(context, point, fill, stroke) {
    context.save();
    context.fillStyle = fill;
    context.strokeStyle = stroke;
    context.lineWidth = 1.5;
    context.fillRect(point.x - 10, point.y - 6, 20, 12);
    context.strokeRect(point.x - 10, point.y - 6, 20, 12);
    context.beginPath();
    context.moveTo(point.x - 6, point.y + 4);
    context.lineTo(point.x - 1, point.y - 4);
    context.moveTo(point.x + 1, point.y + 4);
    context.lineTo(point.x + 6, point.y - 4);
    context.stroke();
    context.restore();
  }

  function canvasLabel(context, text, x, y, color, size, align, weight) {
    context.save();
    context.fillStyle = color;
    context.font = (weight || 700) + " " + (size || 10) + "px ui-monospace, Menlo, monospace";
    context.textAlign = align || "center";
    context.textBaseline = "middle";
    context.fillText(text, x, y);
    context.restore();
  }

  function drawGraph() {
    var size = resizeCanvas(graphCanvas, graphCtx);
    graphCtx.clearRect(0, 0, size.width, size.height);
    graphCtx.fillStyle = COLORS.paper;
    graphCtx.fillRect(0, 0, size.width, size.height);

    var margin = 24;
    var poseY = Math.max(42, size.height * 0.25);
    var mapY = Math.max(poseY + 85, size.height - 46);
    var available = Math.max(1, size.width - margin * 2);
    var poseNodes = [];
    for (var t = 0; t <= state.time; t += 1) {
      poseNodes.push({ x: margin + available * (t / (TRAJECTORY.length - 1)), y: poseY });
    }

    for (var odom = 0; odom < state.time; odom += 1) {
      graphFactor(graphCtx, poseNodes[odom], poseNodes[odom + 1], COLORS.poseDeep, 0.72, 6);
    }

    var bsNode = { x: margin, y: mapY };
    var mapNodes = SURFACES.map(function (_, index) {
      var fraction = SURFACES.length === 1 ? 0.5 : index / (SURFACES.length - 1);
      return { x: margin + 72 + (available - 88) * fraction, y: mapY };
    });

    for (var scan = 0; scan <= state.time; scan += 1) {
      if (!insertedScan(scan)) continue;
      var alpha = scan === state.time ? 0.86 : 0.12;
      graphFactor(graphCtx, poseNodes[scan], bsNode, COLORS.radioDeep, alpha, scan === state.time ? 7 : 5);
      OBSERVATIONS[scan].forEach(function (surfaceIndex) {
        graphFactor(graphCtx, poseNodes[scan], mapNodes[surfaceIndex], COLORS.radioDeep, alpha, scan === state.time ? 7 : 5);
      });
      var orderedPair = DOUBLE_OBSERVATIONS[scan];
      if (orderedPair) {
        graphPairFactor(
          graphCtx,
          poseNodes[scan],
          mapNodes[orderedPair[0]],
          mapNodes[orderedPair[1]],
          SURFACES[orderedPair[0]].id + "→" + SURFACES[orderedPair[1]].id,
          alpha,
          scan === state.time ? 8 : 6
        );
      }
    }

    if (loopClosureInserted()) graphLoopFactor(graphCtx, poseNodes[0], poseNodes[poseNodes.length - 1]);

    graphCtx.save();
    graphCtx.fillStyle = COLORS.ink;
    graphCtx.fillRect(bsNode.x - 6, bsNode.y - 6, 12, 12);
    graphCtx.restore();
    canvasLabel(graphCtx, "b", bsNode.x, bsNode.y + 1, "#ffffff", 8);
    canvasLabel(graphCtx, "fixed", bsNode.x, bsNode.y + 17, COLORS.faint, 8);

    SURFACES.forEach(function (surface, surfaceIndex) {
      var count = observationCount(surfaceIndex);
      var fill = count ? withAlpha(COLORS.map[surfaceIndex], 0.16) : "#f2f4f6";
      var stroke = count ? COLORS.map[surfaceIndex] : COLORS.line;
      graphWallNode(graphCtx, mapNodes[surfaceIndex], fill, stroke);
      canvasLabel(graphCtx, "π" + surface.id, mapNodes[surfaceIndex].x, mapNodes[surfaceIndex].y + 1, count ? COLORS.ink : COLORS.faint, 7);
      canvasLabel(graphCtx, count ? String(count) + " obs" : "unseen", mapNodes[surfaceIndex].x, mapNodes[surfaceIndex].y + 17, COLORS.faint, 7);
    });

    poseNodes.forEach(function (node, index) {
      var current = index === state.time;
      graphCircle(graphCtx, node, current ? 7 : 5.5, current ? COLORS.poseDeep : "#ffffff", COLORS.poseDeep);
      if (current || index === 0 || index % 3 === 0) canvasLabel(graphCtx, "x" + index, node.x, node.y - 13, current ? COLORS.poseDeep : COLORS.soft, 8);
    });

    var priorTarget = { x: poseNodes[0].x, y: poseNodes[0].y - 29 };
    graphLine(graphCtx, priorTarget, poseNodes[0], COLORS.poseDeep, 1.2, 0.8);
    graphCtx.save();
    graphCtx.fillStyle = "#ffffff";
    graphCtx.strokeStyle = COLORS.poseDeep;
    graphCtx.fillRect(priorTarget.x - 4, priorTarget.y - 4, 8, 8);
    graphCtx.strokeRect(priorTarget.x - 4, priorTarget.y - 4, 8, 8);
    graphCtx.restore();
    canvasLabel(graphCtx, "prior", priorTarget.x, priorTarget.y - 10, COLORS.poseDeep, 7);

    if (state.phase < 2) canvasLabel(graphCtx, "radio edges appear in CONNECT", size.width - 10, 14, COLORS.faint, 8, "right", 600);
  }

  function factorCount() {
    var count = 1 + state.time;
    for (var scan = 0; scan <= state.time; scan += 1) {
      if (insertedScan(scan)) count += 1 + OBSERVATIONS[scan].length + (DOUBLE_OBSERVATIONS[scan] ? 1 : 0);
    }
    if (loopClosureInserted()) count += 1;
    return count;
  }

  function poseRMSE() {
    var sum = 0;
    for (var i = 0; i <= state.time; i += 1) {
      var delta = sub(estimatePose(i), TRAJECTORY[i]);
      sum += dot(delta, delta);
    }
    return Math.sqrt(sum / (state.time + 1));
  }

  function wallOffsetError(surfaceIndex) {
    var truth = planeFromSurface(SURFACES[surfaceIndex]);
    var estimate = planeFromSurface(estimateSurface(surfaceIndex));
    if (dot(truth.normal, estimate.normal) < 0) estimate.d *= -1;
    return Math.abs(estimate.d - truth.d);
  }

  function wallOffsetMAE() {
    var sum = 0;
    var count = 0;
    SURFACES.forEach(function (_, surfaceIndex) {
      if (!observationCount(surfaceIndex)) return;
      sum += wallOffsetError(surfaceIndex);
      count += 1;
    });
    return count ? sum / count : null;
  }

  function confidence(surfaceIndex) {
    var count = observationCount(surfaceIndex);
    return count ? Math.round(surfaceQuality(surfaceIndex) * 100) : 0;
  }

  function stageContent() {
    var scan = state.time;
    var surfaces = OBSERVATIONS[scan].map(function (index) { return SURFACES[index]; });
    var orderedPair = DOUBLE_OBSERVATIONS[scan];
    var orderedLabel = orderedPair ? SURFACES[orderedPair[0]].id + "→" + SURFACES[orderedPair[1]].id : "";
    var radioPathCount = 1 + surfaces.length + (orderedPair ? 1 : 0);
    var newSurfaces = surfaces.filter(function (surface) { return surface.first === scan; });
    var revisited = surfaces.filter(function (surface) { return surface.first < scan; });

    if (state.phase === 0) {
      if (scan === 0) {
        return {
          state: "ANCHOR", number: "STEP 1 OF 4", title: "Anchor the first pose",
          description: "Insert the prior on the first UE pose. The calibrated base station is a fixed argument, not an optimized node.",
          equation: "<span class=\"factor prior\">f<sup>prior</sup></span>(x<sub>0</sub>)",
          changeTitle: "Prior fixes the gauge",
          changeDetail: "Without one anchored pose, the entire trajectory and map can move together."
        };
      }
      return {
        state: "PREDICT", number: "STEP 1 OF 4", title: "Add pose x" + scan,
        description: "Propagate the previous estimate with the measured relative motion, then connect the two poses with an odometry factor.",
        equation: "<span class=\"factor prior\">f<sup>odom</sup><sub>" + scan + "</sub></span>(x<sub>" + (scan - 1) + "</sub>, x<sub>" + scan + "</sub>)",
        changeTitle: "Trajectory grows by one node",
        changeDetail: "The map is untouched until the new radio scan is associated and connected."
      };
    }

    if (state.phase === 1) {
      return {
        state: "MEASURE", number: "STEP 2 OF 4", title: "Ingest scan z" + scan,
        description: "The channel estimator reports one LoS component, " + surfaces.length + " single-bounce MPC" + (surfaces.length === 1 ? "" : "s") + (orderedPair ? ", and one ordered double-bounce MPC (" + orderedLabel + ")" : "") + ". Delay, AoA, and AoD constrain the 3D path geometry." + (scan === TRAJECTORY.length - 1 ? " Place recognition also flags x0 as a loop-closure candidate." : ""),
        equation: "z<sub>" + scan + "ℓ</sub> = [ τ, u<sup>A</sup>, u<sup>D</sup> ]<sup>T</sup> + ε",
        changeTitle: radioPathCount + " radio paths enter the front end",
        changeDetail: "Orange paths are S1; pink paths are ordered S2 with q1 and q2. No persistent wall variable is changed yet."
      };
    }

    if (state.phase === 2) {
      if (scan === TRAJECTORY.length - 1) {
        return {
          state: "CONNECT", number: "STEP 3 OF 4", title: "Add the loop closure",
          description: "The verified revisit adds a non-sequential factor between x" + scan + " and x0. The final S1 and S2 observations still connect x" + scan + " to the shared walls.",
          equation: "<span class=\"factor prior\">f<sup>loop</sup><sub>" + scan + ",0</sub></span>(x<sub>" + scan + "</sub>, x<sub>0</sub>), &nbsp; <span class=\"factor\">f<sup>S2</sup></span>(x<sub>" + scan + "</sub>, π<sub>A</sub>, π<sub>B</sub>)",
          changeTitle: "Graph receives a non-sequential edge",
          changeDetail: "The yellow arc closes the cycle instead of extending the odometry chain."
        };
      }
      var title = newSurfaces.length ? "Create " + newSurfaces.map(function (surface) { return "π" + surface.id; }).join(" and ") : "Reuse the persistent wall map";
      var detail = newSurfaces.length
        ? newSurfaces.map(function (surface) { return surface.name; }).join(" and ") + " enter as finite plane variables."
        : revisited.map(function (surface) { return "π" + surface.id; }).join(", ") + " receive new reflection constraints instead of duplicate features.";
      return {
        state: "CONNECT", number: "STEP 3 OF 4", title: title,
        description: "Fixed associations select the finite wall touched by each MPC. S1 solves one bounce point and connects x" + scan + " to one wall." + (orderedPair ? " The ordered S2 " + orderedLabel + " factor solves q1 and q2 and connects the pose to both wall features." : "") + " The known BS remains a parameter.",
        equation: orderedPair
          ? "(q<sub>1</sub>,q<sub>2</sub>) = SpecularSolve(BS, x<sub>" + scan + "</sub>, (π<sub>" + SURFACES[orderedPair[0]].id + "</sub>,π<sub>" + SURFACES[orderedPair[1]].id + "</sub>)), &nbsp; <span class=\"factor\">f<sup>S2</sup></span>(x<sub>" + scan + "</sub>,π<sub>" + SURFACES[orderedPair[0]].id + "</sub>,π<sub>" + SURFACES[orderedPair[1]].id + "</sub>)"
          : "q = SpecularSolve(BS, x<sub>" + scan + "</sub>, π<sub>j</sub>), &nbsp; <span class=\"factor\">f<sup>S1</sup><sub>" + scan + "ℓ</sub></span>(x<sub>" + scan + "</sub>, π<sub>j</sub>)",
        changeTitle: newSurfaces.length ? "Map acquires a wall feature" : "Same wall, another constraint",
        changeDetail: detail
      };
    }

    if (scan === TRAJECTORY.length - 1) {
      return {
        state: "CONVERGED", number: "STEP 4 OF 4", title: "Close the trajectory and refine the map",
        description: "The loop factor pulls the terminal pose into agreement with x0. Re-linearization propagates that correction through the odometry chain and the shared wall factors.",
        equation: "(X̂, Π̂) = arg min Σ ‖r<sup>odom</sup>‖² + Σ ρ(‖r<sup>wall</sup>‖²) + ‖r<sup>loop</sup><sub>" + scan + ",0</sub>‖²",
        changeTitle: "Global correction propagates through the graph",
        changeDetail: "The final scan reduces accumulated drift and tightens wall normals, offsets, and finite support."
      };
    }

    return {
      state: "OPTIMIZE", number: "STEP 4 OF 4", title: "Re-linearize the active graph",
      description: "Whitened prior, odometry, and direct wall-reflection residuals are solved jointly. Shared planes pull poses into agreement, while improved poses sharpen wall geometry.",
      equation: "(X̂, Π̂) = arg min Σ ‖r<sup>odom</sup>‖² + Σ ρ(‖r<sup>wall</sup>‖²)",
      changeTitle: "Trajectory and map update together",
      changeDetail: "The same reflection factor corrects x" + scan + " and its connected wall normal, offset, and support."
    };
  }

  function updateLedger() {
    var ledger = document.getElementById("map-ledger");
    ledger.innerHTML = "";
    SURFACES.forEach(function (surface, surfaceIndex) {
      var count = observationCount(surfaceIndex);
      var row = document.createElement("div");
      row.className = "map-row" + (count ? "" : " unseen");
      var pct = confidence(surfaceIndex);
      row.innerHTML =
        "<div class=\"map-row-head\">" +
          "<span class=\"map-badge\">π" + surface.id + "</span>" +
          "<span class=\"map-name\"><strong>" + surface.name + "</strong><small>" + (count ? count + " observation" + (count === 1 ? "" : "s") + " · |Δd| " + wallOffsetError(surfaceIndex).toFixed(2) + " m" : surface.type + " · unseen") + "</small></span>" +
          "<span class=\"map-confidence\">" + (count ? count + "×" : "—") + "</span>" +
        "</div>" +
        "<div class=\"confidence-track\"><i style=\"width:" + pct + "%\"></i></div>";
      ledger.appendChild(row);
    });
  }

  function updateActivity() {
    var entries;
    var t = state.time;
    if (state.phase === 0) {
      entries = t === 0
        ? ["Create x0 = (p0, R0, b0)", "Attach the pose prior", "Keep calibrated Tᴡʙ fixed"]
        : ["Create x" + t, "Read relative motion ΔT" + (t - 1) + "," + t, "Attach odometry edge"];
    } else if (state.phase === 1) {
      entries = ["Detect " + (1 + OBSERVATIONS[t].length + (DOUBLE_OBSERVATIONS[t] ? 1 : 0)) + " resolvable paths", "Transform AoA/AoD bearings", "Gate S1/S2 path hypotheses"];
      if (t === TRAJECTORY.length - 1) entries.push("Verify revisit candidate x0");
    } else if (state.phase === 2) {
      entries = OBSERVATIONS[t].map(function (surfaceIndex) {
        return (SURFACES[surfaceIndex].first === t ? "Initialize " : "Reconnect ") + "π" + SURFACES[surfaceIndex].id;
      });
      entries.unshift("Attach LoS factor to x" + t);
      if (DOUBLE_OBSERVATIONS[t]) {
        entries.push("Attach ordered S2 " + SURFACES[DOUBLE_OBSERVATIONS[t][0]].id + "→" + SURFACES[DOUBLE_OBSERVATIONS[t][1]].id + " factor");
      }
      if (t === TRAJECTORY.length - 1) entries.push("Attach loop factor x" + t + " ↔ x0");
    } else {
      entries = t === TRAJECTORY.length - 1
        ? ["Linearize graph including floop", "Solve sparse normal equations", "Propagate closure correction", "Update trajectory and map uncertainty"]
        : ["Linearize at current estimate", "Solve sparse normal equations", "Retract poses and planes", "Update map uncertainty proxy"];
    }
    var list = document.getElementById("activity-log");
    list.innerHTML = entries.map(function (entry) { return "<li>" + entry + "</li>"; }).join("");
  }

  function updateUI() {
    var content = stageContent();
    document.getElementById("scan-label").textContent = "SCAN " + String(state.time + 1).padStart(2, "0") + " / " + TRAJECTORY.length;
    document.getElementById("step-state").textContent = content.state;
    document.getElementById("step-number").textContent = content.number;
    document.getElementById("step-title").textContent = content.title;
    document.getElementById("step-description").textContent = content.description;
    document.getElementById("step-equation").innerHTML = content.equation;
    document.getElementById("change-title").textContent = content.changeTitle;
    document.getElementById("change-detail").textContent = content.changeDetail;
    document.getElementById("time-output").textContent = "t = " + state.time;
    timeRange.value = String(state.time);

    var counts = SURFACES.map(function (_, index) { return observationCount(index); });
    document.getElementById("pose-count").textContent = String(state.time + 1);
    document.getElementById("surface-count").textContent = counts.filter(Boolean).length + " / " + SURFACES.length;
    document.getElementById("factor-count").textContent = String(factorCount());
    var mapError = wallOffsetMAE();
    document.getElementById("map-rmse").textContent = mapError === null ? "— m mean |Δd|" : mapError.toFixed(2) + " m mean |Δd|";
    var cost = 6.5 + 137 * Math.exp(-0.072 * factorCount()) + 18 * poseRMSE() + (mapError || 1.1) * 7;
    if (state.phase === 3) cost *= 0.72;
    document.getElementById("cost-value").textContent = "χ² " + cost.toFixed(1);

    document.querySelectorAll(".phase").forEach(function (button) {
      var phase = Number(button.dataset.phase);
      button.classList.toggle("active", phase === state.phase);
      button.classList.toggle("complete", phase < state.phase);
    });

    var globalIndex = state.time * 4 + state.phase;
    var maxIndex = (TRAJECTORY.length - 1) * 4 + 3;
    document.getElementById("back-button").disabled = globalIndex === 0;
    document.getElementById("next-button").disabled = globalIndex === maxIndex;
    document.getElementById("next-button").textContent = globalIndex === maxIndex ? "Map complete" : "Next step →";

    var playButton = document.getElementById("play-button");
    playButton.setAttribute("aria-pressed", String(state.playing));
    playButton.innerHTML = state.playing ? "<span aria-hidden=\"true\">Ⅱ</span> Pause" : "<span aria-hidden=\"true\">▶</span> Play";

    updateLedger();
    updateActivity();
    drawScene();
    drawGraph();
  }

  function setGlobalIndex(index) {
    var maxIndex = (TRAJECTORY.length - 1) * 4 + 3;
    var bounded = clamp(index, 0, maxIndex);
    state.time = Math.floor(bounded / 4);
    state.phase = bounded % 4;
    updateUI();
  }

  function advance() {
    var index = state.time * 4 + state.phase;
    var maxIndex = (TRAJECTORY.length - 1) * 4 + 3;
    if (index >= maxIndex) {
      pause();
      return;
    }
    setGlobalIndex(index + 1);
  }

  function retreat() {
    setGlobalIndex(state.time * 4 + state.phase - 1);
  }

  function play() {
    if (state.playing) {
      pause();
      return;
    }
    if (state.time === TRAJECTORY.length - 1 && state.phase === 3) setGlobalIndex(0);
    state.playing = true;
    updateUI();
    state.timer = window.setInterval(advance, 820);
  }

  function pause() {
    if (state.timer) window.clearInterval(state.timer);
    state.timer = null;
    state.playing = false;
    updateUI();
  }

  function reset() {
    pause();
    state.camera = { yaw: -0.78, pitch: 0.48, distance: 18.5 };
    setGlobalIndex(0);
  }

  function publicState() {
    var visibleWalls = SURFACES.filter(function (_, index) { return observationCount(index) > 0; });
    var error = wallOffsetMAE();
    var orderedPair = DOUBLE_OBSERVATIONS[state.time];
    return {
      time: state.time,
      phase: ["predict", "measure", "connect", "optimize"][state.phase],
      poses: state.time + 1,
      walls: visibleWalls.map(function (surface) { return surface.id; }),
      currentDoubleBounceOrder: orderedPair ? orderedPair.map(function (surfaceIndex) { return SURFACES[surfaceIndex].id; }) : null,
      loopClosure: loopClosureInserted(),
      factors: factorCount(),
      meanWallOffsetErrorMeters: error === null ? null : Number(error.toFixed(3))
    };
  }

  function registerWebMCP() {
    var context = document.modelContext;
    if (!context || typeof context.registerTool !== "function") return;
    var lifecycle = new AbortController();
    function register(tool) {
      try {
        Promise.resolve(context.registerTool(tool, { signal: lifecycle.signal })).catch(function (error) {
          console.warn("WebMCP registration skipped", error);
        });
      } catch (error) {
        console.warn("WebMCP registration skipped", error);
      }
    }
    try {
      register({
        name: "set_wall_graphslam_demo_step",
        title: "Set wall GraphSLAM demo step",
        description: "Navigate the visible 3D wall-feature radio GraphSLAM demo to one trajectory time and processing phase.",
        inputSchema: {
          type: "object",
          properties: {
            time: { type: "integer", minimum: 0, maximum: TRAJECTORY.length - 1 },
            phase: { type: "string", enum: ["predict", "measure", "connect", "optimize"] }
          },
          required: ["time", "phase"],
          additionalProperties: false
        },
        annotations: { readOnlyHint: false, untrustedContentHint: false },
        execute: function (input) {
          var phaseIndex = ["predict", "measure", "connect", "optimize"].indexOf(input && input.phase);
          if (!input || !Number.isInteger(input.time) || input.time < 0 || input.time >= TRAJECTORY.length || phaseIndex < 0) {
            throw new Error("Expected time 0–" + (TRAJECTORY.length - 1) + " and phase predict, measure, connect, or optimize.");
          }
          pause();
          state.time = input.time;
          state.phase = phaseIndex;
          updateUI();
          return publicState();
        }
      });
      register({
        name: "read_wall_graphslam_demo_state",
        title: "Read wall GraphSLAM demo state",
        description: "Read the current visible time, phase, graph size, active ordered S2 path, loop-closure status, mapped walls, and wall-offset error from the 3D demo.",
        inputSchema: { type: "object", properties: {}, additionalProperties: false },
        annotations: { readOnlyHint: true, untrustedContentHint: false },
        execute: function () { return publicState(); }
      });
      window.addEventListener("pagehide", function () { lifecycle.abort(); }, { once: true });
    } catch (error) { console.warn("WebMCP registration skipped", error); }
  }

  document.getElementById("next-button").addEventListener("click", function () { pause(); advance(); });
  document.getElementById("back-button").addEventListener("click", function () { pause(); retreat(); });
  document.getElementById("play-button").addEventListener("click", play);
  document.querySelectorAll(".phase").forEach(function (button) {
    button.addEventListener("click", function () {
      pause();
      state.phase = Number(button.dataset.phase);
      updateUI();
    });
  });
  timeRange.addEventListener("input", function () {
    var selectedTime = Number(timeRange.value);
    pause();
    state.time = selectedTime;
    state.phase = 3;
    updateUI();
  });
  truthToggle.addEventListener("change", drawScene);
  raysToggle.addEventListener("change", drawScene);
  normalToggle.addEventListener("change", drawScene);

  sceneCanvas.addEventListener("pointerdown", function (event) {
    state.drag = { x: event.clientX, y: event.clientY, yaw: state.camera.yaw, pitch: state.camera.pitch };
    sceneCanvas.setPointerCapture(event.pointerId);
  });
  sceneCanvas.addEventListener("pointermove", function (event) {
    if (!state.drag) return;
    state.camera.yaw = state.drag.yaw - (event.clientX - state.drag.x) * 0.007;
    state.camera.pitch = clamp(state.drag.pitch + (event.clientY - state.drag.y) * 0.006, 0.12, 1.12);
    drawScene();
  });
  sceneCanvas.addEventListener("pointerup", function (event) {
    state.drag = null;
    if (sceneCanvas.hasPointerCapture(event.pointerId)) sceneCanvas.releasePointerCapture(event.pointerId);
  });
  sceneCanvas.addEventListener("pointercancel", function () { state.drag = null; });
  sceneCanvas.addEventListener("wheel", function (event) {
    event.preventDefault();
    state.camera.distance = clamp(state.camera.distance * Math.exp(event.deltaY * 0.001), 11, 30);
    drawScene();
  }, { passive: false });
  sceneCanvas.addEventListener("dblclick", function () {
    state.camera = { yaw: -0.78, pitch: 0.48, distance: 18.5 };
    drawScene();
  });

  window.addEventListener("keydown", function (event) {
    var tag = document.activeElement && document.activeElement.tagName;
    if (tag === "INPUT" && document.activeElement !== timeRange) return;
    if (event.key === "ArrowRight") {
      event.preventDefault();
      pause();
      advance();
    } else if (event.key === "ArrowLeft") {
      event.preventDefault();
      pause();
      retreat();
    } else if (event.key === " ") {
      event.preventDefault();
      play();
    } else if (event.key.toLowerCase() === "r") {
      event.preventDefault();
      reset();
    }
  });

  var resizeObserver = new ResizeObserver(function () {
    drawScene();
    drawGraph();
  });
  resizeObserver.observe(sceneCanvas);
  resizeObserver.observe(graphCanvas);

  updateUI();
  registerWebMCP();
})();
