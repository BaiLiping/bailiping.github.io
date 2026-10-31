(function (root, factory) {
  "use strict";
  var api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.RadioSlamS1 = api;
})(typeof window !== "undefined" ? window : null, function () {
  "use strict";

  var C = 299792458;
  var DEG = Math.PI / 180;

  var setup = {
    id: "S1",
    bs: { x: 2.0, y: 2.0, heading: 0 },
    walls: [
      { id: "A", axis: "y", value: 7.0, label: "wall A" },
      { id: "B", axis: "x", value: 8.5, label: "wall B" }
    ],
    virtualAnchors: [
      { x: 2.0, y: 12.0, wallId: "A" },
      { x: 15.0, y: 2.0, wallId: "B" }
    ],
    poses: [
      { x: 2.8, y: 6.2, headingDeg: -18 },
      { x: 3.9, y: 5.6, headingDeg: -13 },
      { x: 5.0, y: 5.0, headingDeg: -6 },
      { x: 6.1, y: 4.3, headingDeg: 3 },
      { x: 7.2, y: 3.5, headingDeg: 12 }
    ],
    paths: [
      { id: "los", label: "LoS", tex: "\\mathrm{LoS}", kind: "los", bounceCount: 0, color: "#e8720c", reflectionLossDb: 0 },
      { id: "wall-a", label: "wall A", tex: "A", kind: "reflection", bounceCount: 1, vaIndex: 0, color: "#0e8f7e", reflectionLossDb: 6 },
      { id: "wall-b", label: "wall B", tex: "B", kind: "reflection", bounceCount: 1, vaIndex: 1, color: "#1874b8", reflectionLossDb: 8 }
    ],
    noise: {
      rangeSigma: 0.08,
      aoaSigma: 1.4 * DEG,
      aodSigma: 1.4 * DEG,
      gainSigmaDb: 1.0
    },
    clutterScans: [1, 3]
  };

  setup.poses.forEach(function (pose) { pose.heading = pose.headingDeg * DEG; });

  var rangeNoise = [
    [-0.04, -0.07, 0.05],
    [0.03, 0.04, -0.06],
    [-0.06, 0.07, 0.02],
    [0.05, -0.05, 0.08],
    [-0.02, 0.03, -0.04]
  ];
  var aoaNoiseDeg = [
    [0.5, -0.8, 0.7],
    [-0.6, 0.5, -0.9],
    [0.8, -0.4, 0.6],
    [-0.7, 0.9, -0.5],
    [0.4, -0.6, 0.5]
  ];
  var aodNoiseDeg = [
    [-0.4, 0.7, -0.6],
    [0.6, -0.5, 0.8],
    [-0.7, 0.4, -0.5],
    [0.9, -0.6, 0.7],
    [-0.5, 0.6, -0.4]
  ];
  var gainNoiseDb = [
    [0.3, -0.5, 0.6],
    [-0.4, 0.7, -0.3],
    [0.5, -0.6, 0.4],
    [-0.7, 0.5, -0.4],
    [0.4, -0.3, 0.6]
  ];
  var odometryNoise = [
    { x: 0.025, y: -0.018, heading: 0.35 * DEG },
    { x: -0.020, y: 0.022, heading: -0.30 * DEG },
    { x: 0.030, y: 0.012, heading: 0.40 * DEG },
    { x: -0.018, y: -0.020, heading: -0.25 * DEG }
  ];

  function wrap(angle) {
    while (angle > Math.PI) angle -= 2 * Math.PI;
    while (angle <= -Math.PI) angle += 2 * Math.PI;
    return angle;
  }

  function clone(value) {
    return JSON.parse(JSON.stringify(value));
  }

  function reflectionPoint(pose, virtualAnchor, wall) {
    var denominator = wall.axis === "x" ? virtualAnchor.x - pose.x : virtualAnchor.y - pose.y;
    var numerator = wall.value - (wall.axis === "x" ? pose.x : pose.y);
    var fraction = Math.abs(denominator) < 1e-12 ? 0.5 : numerator / denominator;
    return {
      x: pose.x + fraction * (virtualAnchor.x - pose.x),
      y: pose.y + fraction * (virtualAnchor.y - pose.y),
      fraction: fraction
    };
  }

  function pathLossDb(range, path) {
    return 46 + 20 * Math.log10(Math.max(range, 0.25)) + path.reflectionLossDb;
  }

  function predictPath(pose, path) {
    var source;
    var point = null;
    var aod;
    if (path.kind === "los") {
      source = setup.bs;
      aod = wrap(Math.atan2(pose.y - setup.bs.y, pose.x - setup.bs.x) - setup.bs.heading);
    } else {
      source = setup.virtualAnchors[path.vaIndex];
      point = reflectionPoint(pose, source, setup.walls[path.vaIndex]);
      aod = wrap(Math.atan2(point.y - setup.bs.y, point.x - setup.bs.x) - setup.bs.heading);
    }
    var dx = source.x - pose.x;
    var dy = source.y - pose.y;
    var range = Math.hypot(dx, dy);
    var aoa = wrap(Math.atan2(dy, dx) - pose.heading);
    var lossDb = pathLossDb(range, path);
    return {
      pathId: path.id,
      pathIndex: setup.paths.indexOf(path),
      label: path.label,
      tex: path.tex,
      kind: path.kind,
      bounceCount: path.bounceCount,
      range: range,
      tau: range / C,
      aoa: aoa,
      aod: aod,
      gainDb: -lossDb,
      pathLossDb: lossDb,
      alphaMagnitude: Math.pow(10, -lossDb / 20),
      reflectionPoint: point
    };
  }

  function noisyPath(prediction, scanIndex, pathIndex, scale) {
    var gainDb = prediction.gainDb + scale * gainNoiseDb[scanIndex][pathIndex];
    var range = prediction.range + scale * rangeNoise[scanIndex][pathIndex];
    return {
      pathId: prediction.pathId,
      pathIndex: pathIndex,
      label: prediction.label,
      tex: prediction.tex,
      kind: prediction.kind,
      bounceCount: prediction.bounceCount,
      range: range,
      tau: range / C,
      aoa: wrap(prediction.aoa + scale * aoaNoiseDeg[scanIndex][pathIndex] * DEG),
      aod: wrap(prediction.aod + scale * aodNoiseDeg[scanIndex][pathIndex] * DEG),
      gainDb: gainDb,
      pathLossDb: -gainDb,
      alphaMagnitude: Math.pow(10, gainDb / 20),
      isClutter: false
    };
  }

  function clutterMeasurement(scanIndex, predictions) {
    if (scanIndex === 1) {
      var nearA = predictions[1];
      return {
        pathId: "clutter",
        pathIndex: -1,
        label: "clutter",
        tex: "\\mathrm{FA}",
        kind: "clutter",
        bounceCount: null,
        range: nearA.range + 0.22,
        tau: (nearA.range + 0.22) / C,
        aoa: wrap(nearA.aoa - 5.5 * DEG),
        aod: wrap(nearA.aod + 4.0 * DEG),
        gainDb: nearA.gainDb - 4.5,
        pathLossDb: -(nearA.gainDb - 4.5),
        alphaMagnitude: Math.pow(10, (nearA.gainDb - 4.5) / 20),
        isClutter: true
      };
    }
    if (scanIndex === 3) {
      var a = predictions[1];
      var b = predictions[2];
      var range = 0.5 * (a.range + b.range) + 0.08;
      var aoa = wrap(a.aoa + 0.5 * wrap(b.aoa - a.aoa));
      var aod = wrap(a.aod + 0.5 * wrap(b.aod - a.aod));
      var gainDb = 0.5 * (a.gainDb + b.gainDb) - 3.0;
      return {
        pathId: "clutter",
        pathIndex: -1,
        label: "clutter",
        tex: "\\mathrm{FA}",
        kind: "clutter",
        bounceCount: null,
        range: range,
        tau: range / C,
        aoa: aoa,
        aod: aod,
        gainDb: gainDb,
        pathLossDb: -gainDb,
        alphaMagnitude: Math.pow(10, gainDb / 20),
        isClutter: true
      };
    }
    return null;
  }

  function scan(scanIndex, noiseScale) {
    var index = Math.max(0, Math.min(setup.poses.length - 1, Number(scanIndex) || 0));
    var scale = noiseScale === undefined ? 1 : Number(noiseScale);
    var pose = setup.poses[index];
    var predictions = setup.paths.map(function (path) { return predictPath(pose, path); });
    var measurements = predictions.map(function (prediction, pathIndex) {
      return noisyPath(prediction, index, pathIndex, scale);
    });
    var clutter = clutterMeasurement(index, predictions);
    if (clutter) measurements.push(clutter);
    return {
      id: setup.id,
      scanIndex: index,
      pose: clone(pose),
      predictions: predictions,
      measurements: measurements,
      clutter: clutter
    };
  }

  function allScans(noiseScale) {
    return setup.poses.map(function (_, index) { return scan(index, noiseScale); });
  }

  return {
    C: C,
    DEG: DEG,
    setup: setup,
    odometryNoise: odometryNoise,
    wrap: wrap,
    clone: clone,
    reflectionPoint: reflectionPoint,
    predictPath: predictPath,
    scan: scan,
    allScans: allScans
  };
});
