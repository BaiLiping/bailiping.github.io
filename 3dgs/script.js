(() => {
  "use strict";

  const COLORS = {
    ink: "#0a0c0e",
    panel: "#121719",
    panel2: "#192023",
    paper: "#f2eee5",
    muted: "#747c79",
    faint: "rgba(242, 238, 229, 0.12)",
    line: "rgba(242, 238, 229, 0.18)",
    coral: "#ff725e",
    cyan: "#61dce8",
    acid: "#d9ff6f",
    violet: "#c7a6ff",
  };

  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const dpr = () => Math.min(window.devicePixelRatio || 1, 2);
  const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
  const lerp = (a, b, t) => a + (b - a) * t;
  const $ = (selector, parent = document) => parent.querySelector(selector);
  const $$ = (selector, parent = document) => [...parent.querySelectorAll(selector)];

  function mulberry32(seed) {
    return function random() {
      let value = (seed += 0x6d2b79f5);
      value = Math.imul(value ^ (value >>> 15), value | 1);
      value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
      return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
    };
  }

  function prepareCanvas(canvas) {
    const rect = canvas.getBoundingClientRect();
    const scale = dpr();
    const pixelWidth = Math.max(1, Math.round(rect.width * scale));
    const pixelHeight = Math.max(1, Math.round(rect.height * scale));

    if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
      canvas.width = pixelWidth;
      canvas.height = pixelHeight;
    }

    const context = canvas.getContext("2d");
    context.setTransform(scale, 0, 0, scale, 0, 0);
    context.imageSmoothingEnabled = true;

    return { context, width: rect.width, height: rect.height };
  }

  function roundedRect(context, x, y, width, height, radius = 8) {
    const r = Math.min(radius, width / 2, height / 2);
    context.beginPath();
    context.moveTo(x + r, y);
    context.lineTo(x + width - r, y);
    context.quadraticCurveTo(x + width, y, x + width, y + r);
    context.lineTo(x + width, y + height - r);
    context.quadraticCurveTo(x + width, y + height, x + width - r, y + height);
    context.lineTo(x + r, y + height);
    context.quadraticCurveTo(x, y + height, x, y + height - r);
    context.lineTo(x, y + r);
    context.quadraticCurveTo(x, y, x + r, y);
    context.closePath();
  }

  function drawLabel(context, text, x, y, color = COLORS.muted, align = "left") {
    context.save();
    context.fillStyle = color;
    context.font = '700 8px "SFMono-Regular", Consolas, monospace';
    context.textAlign = align;
    context.textBaseline = "middle";
    context.fillText(text.toUpperCase(), x, y);
    context.restore();
  }

  function drawArrow(context, x1, y1, x2, y2, color, width = 1.4, head = 6) {
    const angle = Math.atan2(y2 - y1, x2 - x1);
    context.save();
    context.strokeStyle = color;
    context.fillStyle = color;
    context.lineWidth = width;
    context.beginPath();
    context.moveTo(x1, y1);
    context.lineTo(x2, y2);
    context.stroke();
    context.beginPath();
    context.moveTo(x2, y2);
    context.lineTo(x2 - head * Math.cos(angle - Math.PI / 6), y2 - head * Math.sin(angle - Math.PI / 6));
    context.lineTo(x2 - head * Math.cos(angle + Math.PI / 6), y2 - head * Math.sin(angle + Math.PI / 6));
    context.closePath();
    context.fill();
    context.restore();
  }

  function drawGrid(context, box, size = 28, alpha = 0.05) {
    context.save();
    context.strokeStyle = `rgba(242, 238, 229, ${alpha})`;
    context.lineWidth = 1;
    context.beginPath();
    for (let x = box.x; x <= box.x + box.w; x += size) {
      context.moveTo(x, box.y);
      context.lineTo(x, box.y + box.h);
    }
    for (let y = box.y; y <= box.y + box.h; y += size) {
      context.moveTo(box.x, y);
      context.lineTo(box.x + box.w, y);
    }
    context.stroke();
    context.restore();
  }

  function drawPanel(context, box, title, trailing) {
    context.save();
    context.fillStyle = COLORS.panel;
    context.strokeStyle = COLORS.line;
    context.lineWidth = 1;
    roundedRect(context, box.x, box.y, box.w, box.h, 7);
    context.fill();
    context.stroke();
    drawLabel(context, title, box.x + 16, box.y + 19, "#9ba19f");
    if (trailing) drawLabel(context, trailing, box.x + box.w - 16, box.y + 19, "#59615f", "right");
    context.strokeStyle = COLORS.faint;
    context.beginPath();
    context.moveTo(box.x, box.y + 38);
    context.lineTo(box.x + box.w, box.y + 38);
    context.stroke();
    context.restore();
  }

  function gaussianFill(context, x, y, rx, ry, rotation, color, alpha = 0.55) {
    context.save();
    context.translate(x, y);
    context.rotate(rotation);
    context.scale(rx, ry);
    const gradient = context.createRadialGradient(0, 0, 0, 0, 0, 1);
    gradient.addColorStop(0, color);
    gradient.addColorStop(0.42, color);
    gradient.addColorStop(1, "rgba(0,0,0,0)");
    context.globalAlpha = alpha;
    context.fillStyle = gradient;
    context.beginPath();
    context.arc(0, 0, 1, 0, Math.PI * 2);
    context.fill();
    context.restore();
  }

  // Header behavior
  const header = $("[data-header]");
  const updateHeader = () => header.classList.toggle("is-scrolled", window.scrollY > 24);
  updateHeader();
  window.addEventListener("scroll", updateHeader, { passive: true });

  // Shared scene data
  const random = mulberry32(3042023);
  const pointPalette = [COLORS.coral, COLORS.cyan, COLORS.acid, COLORS.violet, "#f2b45f"];
  const scenePoints = [];

  for (let index = 0; index < 54; index += 1) {
    const band = index % 3;
    const angle = random() * Math.PI * 2;
    const radius = band === 0 ? 0.17 + random() * 0.24 : band === 1 ? 0.34 + random() * 0.3 : 0.56 + random() * 0.2;
    scenePoints.push({
      x: Math.cos(angle) * radius * (0.85 + random() * 0.35),
      y: Math.sin(angle) * radius * (0.65 + random() * 0.25),
      z: random() * 2 - 1,
      scale: 0.5 + random() * 1.3,
      stretch: 0.45 + random() * 1.55,
      rotation: angle * 0.45 + (random() - 0.5) * 1.2,
      color: pointPalette[Math.floor(random() * pointPalette.length)],
      phase: random() * Math.PI * 2,
    });
  }

  const cameraAngles = [-2.55, -1.72, -0.72, 0.12, 1.18, 2.28];
  let selectedCamera = 2;

  function cameraGeometry(index, box, radiusScale = 0.4) {
    const centerX = box.x + box.w * 0.5;
    const centerY = box.y + box.h * 0.55;
    const radius = Math.min(box.w, box.h) * radiusScale;
    const angle = cameraAngles[index];
    return {
      x: centerX + Math.cos(angle) * radius,
      y: centerY + Math.sin(angle) * radius * 0.72,
      angle: angle + Math.PI,
      centerX,
      centerY,
    };
  }

  function drawCameraIcon(context, geometry, active, scale = 1) {
    const { x, y, angle } = geometry;
    context.save();
    context.translate(x, y);
    context.rotate(angle);
    context.strokeStyle = active ? COLORS.cyan : "#737c79";
    context.fillStyle = active ? COLORS.cyan : COLORS.panel2;
    context.lineWidth = active ? 1.8 : 1;
    roundedRect(context, -7 * scale, -5 * scale, 12 * scale, 10 * scale, 2);
    context.fill();
    context.stroke();
    context.beginPath();
    context.moveTo(5 * scale, -4 * scale);
    context.lineTo(12 * scale, -8 * scale);
    context.lineTo(12 * scale, 8 * scale);
    context.lineTo(5 * scale, 4 * scale);
    context.closePath();
    context.stroke();
    if (active) {
      context.fillStyle = COLORS.acid;
      context.beginPath();
      context.arc(-2 * scale, 0, 1.6 * scale, 0, Math.PI * 2);
      context.fill();
    }
    context.restore();
  }

  // Hero particle field
  const heroCanvas = $("#hero-canvas");
  const heroRandom = mulberry32(42);
  const heroParticles = [];
  for (let index = 0; index < 155; index += 1) {
    const theta = heroRandom() * Math.PI * 2;
    const ring = 0.18 + Math.pow(heroRandom(), 0.72) * 0.82;
    heroParticles.push({
      x: Math.cos(theta) * ring,
      y: Math.sin(theta) * ring * (0.54 + heroRandom() * 0.22),
      z: (heroRandom() - 0.5) * 1.8,
      size: 0.45 + heroRandom() * 1.25,
      stretch: 0.35 + heroRandom() * 1.8,
      rotation: theta * 0.65 + heroRandom(),
      phase: heroRandom() * Math.PI * 2,
      color: pointPalette[Math.floor(heroRandom() * pointPalette.length)],
    });
  }

  let heroPointerX = 0;
  let heroPointerY = 0;
  let heroVisible = true;

  heroCanvas.addEventListener("pointermove", (event) => {
    const rect = heroCanvas.getBoundingClientRect();
    heroPointerX = event.clientX / rect.width - 0.5;
    heroPointerY = event.clientY / rect.height - 0.5;
  });

  heroCanvas.addEventListener("pointerleave", () => {
    heroPointerX = 0;
    heroPointerY = 0;
  });

  if ("IntersectionObserver" in window) {
    const heroObserver = new IntersectionObserver(([entry]) => {
      heroVisible = entry.isIntersecting;
    });
    heroObserver.observe(heroCanvas);
  }

  function drawHero(timestamp) {
    if (heroVisible) {
      const { context, width, height } = prepareCanvas(heroCanvas);
      context.clearRect(0, 0, width, height);
      const mobile = width < 760;
      const centerX = mobile ? width * 0.65 : width * 0.73;
      const centerY = mobile ? height * 0.68 : height * 0.48;
      const extent = Math.min(width * (mobile ? 0.66 : 0.43), height * 0.58);
      const time = reducedMotion ? 0 : timestamp * 0.00012;
      const rotationY = time + heroPointerX * 0.45;
      const cosY = Math.cos(rotationY);
      const sinY = Math.sin(rotationY);

      context.save();
      context.translate(centerX, centerY);
      context.strokeStyle = "rgba(97, 220, 232, 0.12)";
      context.lineWidth = 1;
      for (let ring = 1; ring <= 3; ring += 1) {
        context.beginPath();
        context.ellipse(0, 0, extent * 0.24 * ring, extent * 0.1 * ring, -0.13, 0, Math.PI * 2);
        context.stroke();
      }
      context.restore();

      const projected = heroParticles
        .map((particle) => {
          const rotatedX = particle.x * cosY - particle.z * sinY;
          const rotatedZ = particle.x * sinY + particle.z * cosY;
          const perspective = 1 / (1.25 + rotatedZ * 0.2);
          return {
            ...particle,
            sx: centerX + rotatedX * extent * perspective + heroPointerX * 12,
            sy:
              centerY +
              (particle.y + Math.sin(time * 2 + particle.phase) * 0.007 - heroPointerY * 0.04) *
                extent *
                perspective,
            depth: rotatedZ,
            perspective,
          };
        })
        .sort((a, b) => b.depth - a.depth);

      projected.forEach((particle) => {
        const size = (4.2 + particle.size * 8) * particle.perspective;
        context.save();
        context.translate(particle.sx, particle.sy);
        context.rotate(particle.rotation + time * 0.08);
        context.scale(particle.stretch, 1 / Math.max(0.7, particle.stretch));
        context.globalAlpha = clamp(0.1 + (1 - (particle.depth + 1) / 2) * 0.32, 0.08, 0.42);
        context.fillStyle = particle.color;
        context.beginPath();
        context.ellipse(0, 0, size, size * 0.58, 0, 0, Math.PI * 2);
        context.fill();
        context.globalAlpha = 0.72;
        context.fillStyle = particle.color;
        context.beginPath();
        context.arc(0, 0, Math.max(0.7, size * 0.09), 0, Math.PI * 2);
        context.fill();
        context.restore();
      });

      context.save();
      context.translate(centerX, centerY);
      context.rotate(-0.1);
      context.strokeStyle = "rgba(217, 255, 111, 0.34)";
      context.setLineDash([4, 8]);
      context.beginPath();
      context.ellipse(0, 0, extent * 0.85, extent * 0.28, 0, 0, Math.PI * 2);
      context.stroke();
      context.setLineDash([]);
      [0.25, 2.25, 4.18].forEach((angle, index) => {
        const geometry = {
          x: Math.cos(angle + time * 0.08) * extent * 0.84,
          y: Math.sin(angle + time * 0.08) * extent * 0.28,
          angle: angle + Math.PI + time * 0.08,
        };
        drawCameraIcon(context, geometry, index === 0, 0.85);
      });
      context.restore();
    }
    window.requestAnimationFrame(drawHero);
  }
  window.requestAnimationFrame(drawHero);

  // Training loop
  const processCanvas = $("#process-canvas");
  const stepData = [
    {
      title: "Calibrate the capture",
      copy:
        "Structure from Motion recovers a sparse 3D point cloud and one calibrated camera per registered image: intrinsics, rotation, and translation.",
      label: "SfM gives us",
      equation: "{ points, Kᵢ, Rᵢ, tᵢ, imageᵢ }",
      status: "COLMAP OUTPUT · FIXED",
      pills: ["camera poses: fixed", "seed points: created"],
    },
    {
      title: "Turn points into Gaussians",
      copy:
        "Every sparse SfM point becomes a 3D Gaussian. Its mean and color come from the point; its initial covariance is isotropic, scaled from nearby-point distances.",
      label: "Initialize each primitive",
      equation: "Gᵢ = { μᵢ, sᵢ, qᵢ, αᵢ, SHᵢ }",
      status: "SPARSE SEEDS → TRAINABLE ELLIPSOIDS",
      pills: ["opacity: 0.1", "rotation: identity", "SH: point color"],
    },
    {
      title: "Choose a training view",
      copy:
        "The implementation draws one camera from the training-camera stack. Its calibrated pose and matching photograph define this iteration’s viewpoint and target.",
      label: "Selected supervision pair",
      equation: "cameraᵢ ↔ ground_truth_imageᵢ",
      status: "ONE CALIBRATED VIEW · THIS ITERATION",
      pills: ["camera: fixed", "photo: target", "Gaussians: shared"],
    },
    {
      title: "Differentiably splat",
      copy:
        "Project visible 3D Gaussians to 2D ellipses, bin them into 16 × 16 pixel tiles, sort by tile and depth, then alpha-composite front-to-back.",
      label: "Projected covariance",
      equation: "Σ′ = J W Σ Wᵀ Jᵀ",
      status: "TILE RASTERIZER · FORWARD PASS",
      pills: ["frustum cull", "depth sort", "α composite"],
    },
    {
      title: "Compare render and photo",
      copy:
        "The rendered pixels are compared with the selected camera’s ground-truth photo using a weighted combination of L1 error and structural dissimilarity.",
      label: "Paper objective",
      equation: "ℒ = 0.8 ℒ₁ + 0.2 (1 − SSIM)",
      status: "LOSS · PIXELS + LOCAL STRUCTURE",
      pills: ["L1: 80%", "D-SSIM: 20%", "scalar loss"],
    },
    {
      title: "Send gradients backward",
      copy:
        "Calling backward differentiates through compositing and projection. Adam updates the visible Gaussians’ position, scale, rotation, opacity, and SH coefficients.",
      label: "Gradient targets",
      equation: "∂ℒ / ∂{ μ, s, q, α, SH }",
      status: "BACKWARD PASS · OPTIMIZER STEP",
      pills: ["render is differentiable", "poses stay fixed", "parameters update"],
    },
    {
      title: "Adapt the representation",
      copy:
        "At scheduled intervals, high-gradient small Gaussians are cloned, large ones are split, and low-opacity or oversized Gaussians are pruned.",
      label: "Adaptive density control",
      equation: "clone · split · prune · reset α",
      status: "DENSIFY EVERY 100 ITERATIONS",
      pills: ["starts: 500", "stops: 15k", "opacity reset: 3k"],
    },
  ];

  let currentStep = 0;
  let loopTimer = null;
  let processHitboxes = [];

  function panelLayout(width, height) {
    const gap = 14;
    const margin = 16;
    const top = 16;
    if (width < 720) {
      const available = height - top - margin - gap;
      return {
        world: { x: margin, y: top, w: width - margin * 2, h: available * 0.48 },
        image: {
          x: margin,
          y: top + available * 0.48 + gap,
          w: width - margin * 2,
          h: available * 0.52,
        },
      };
    }
    return {
      world: { x: margin, y: top, w: (width - margin * 2 - gap) * 0.55, h: height - top - margin },
      image: {
        x: margin + (width - margin * 2 - gap) * 0.55 + gap,
        y: top,
        w: (width - margin * 2 - gap) * 0.45,
        h: height - top - margin,
      },
    };
  }

  function drawSceneCore(context, box, step, timestamp, includeLabels = true) {
    const content = { x: box.x, y: box.y + (includeLabels ? 38 : 0), w: box.w, h: box.h - (includeLabels ? 38 : 0) };
    const centerX = content.x + content.w * 0.5;
    const centerY = content.y + content.h * 0.55;
    const pointExtent = Math.min(content.w, content.h) * 0.27;
    const pulse = reducedMotion ? 0 : (Math.sin(timestamp * 0.004) + 1) / 2;

    drawGrid(context, content, 30, 0.035);

    context.save();
    context.strokeStyle = "rgba(242, 238, 229, 0.08)";
    context.setLineDash([3, 6]);
    context.beginPath();
    context.ellipse(centerX, centerY, pointExtent * 1.82, pointExtent * 1.15, 0, 0, Math.PI * 2);
    context.stroke();
    context.restore();

    const selectedGeometry = cameraGeometry(selectedCamera, content, 0.41);
    if (step >= 2) {
      context.save();
      const direction = Math.atan2(centerY - selectedGeometry.y, centerX - selectedGeometry.x);
      const spread = 0.32;
      const distance = Math.hypot(centerX - selectedGeometry.x, centerY - selectedGeometry.y) * 0.93;
      context.fillStyle = "rgba(97, 220, 232, 0.055)";
      context.strokeStyle = "rgba(97, 220, 232, 0.28)";
      context.lineWidth = 1;
      context.beginPath();
      context.moveTo(selectedGeometry.x, selectedGeometry.y);
      context.lineTo(
        selectedGeometry.x + Math.cos(direction - spread) * distance,
        selectedGeometry.y + Math.sin(direction - spread) * distance,
      );
      context.lineTo(
        selectedGeometry.x + Math.cos(direction + spread) * distance,
        selectedGeometry.y + Math.sin(direction + spread) * distance,
      );
      context.closePath();
      context.fill();
      context.stroke();
      context.restore();
    }

    const pointRenderData = scenePoints
      .map((point, index) => ({
        ...point,
        index,
        sx: centerX + point.x * pointExtent,
        sy: centerY + point.y * pointExtent,
      }))
      .sort((a, b) => a.z - b.z);

    pointRenderData.forEach((point) => {
      if (step === 0) {
        context.save();
        context.globalAlpha = 0.55 + (point.z + 1) * 0.14;
        context.fillStyle = point.color;
        context.beginPath();
        context.arc(point.sx, point.sy, 1.8 + point.scale * 0.8, 0, Math.PI * 2);
        context.fill();
        context.restore();
        return;
      }

      const highlight = step === 5 && point.index % 9 === 0;
      const densityPulse = step === 6 && point.index % 7 === 0 ? 1 + pulse * 0.18 : 1;
      context.save();
      context.translate(point.sx, point.sy);
      context.rotate(point.rotation);
      context.scale(point.stretch, 1 / Math.max(0.7, point.stretch));
      context.globalAlpha = highlight ? 0.68 : 0.26 + (point.z + 1) * 0.08;
      context.fillStyle = highlight ? COLORS.paper : point.color;
      context.beginPath();
      context.ellipse(0, 0, (4 + point.scale * 4.5) * densityPulse, 3.4 + point.scale * 2.6, 0, 0, Math.PI * 2);
      context.fill();
      context.globalAlpha = 0.72;
      context.beginPath();
      context.arc(0, 0, 1.1, 0, Math.PI * 2);
      context.fill();
      context.restore();

      if (step === 5 && point.index % 9 === 0) {
        const angle = point.rotation + Math.sin(point.index) * 0.5;
        const length = 12 + pulse * 8;
        drawArrow(
          context,
          point.sx,
          point.sy,
          point.sx + Math.cos(angle) * length,
          point.sy + Math.sin(angle) * length,
          COLORS.coral,
          1,
          4,
        );
      }

      if (step === 6 && point.index % 7 === 0) {
        const offset = 7 + pulse * 5;
        context.save();
        context.fillStyle = point.color;
        context.globalAlpha = 0.46;
        context.beginPath();
        context.ellipse(point.sx + offset, point.sy - offset * 0.35, 3.2, 2, point.rotation, 0, Math.PI * 2);
        context.fill();
        context.restore();
      }
    });

    processHitboxes = [];
    cameraAngles.forEach((angle, index) => {
      const geometry = cameraGeometry(index, content, 0.41);
      const active = index === selectedCamera;
      context.save();
      context.strokeStyle = active ? "rgba(97, 220, 232, 0.45)" : "rgba(242, 238, 229, 0.075)";
      context.setLineDash(active ? [4, 5] : [2, 8]);
      context.beginPath();
      context.moveTo(geometry.x, geometry.y);
      context.lineTo(centerX, centerY);
      context.stroke();
      context.restore();
      drawCameraIcon(context, geometry, active, active ? 1.05 : 0.85);
      drawLabel(
        context,
        String(index + 1).padStart(2, "0"),
        geometry.x,
        geometry.y + 18,
        active ? COLORS.cyan : "#545c59",
        "center",
      );
      processHitboxes.push({ x: geometry.x, y: geometry.y, radius: 22, index });
    });

    if (includeLabels) {
      const chipText = step === 0 ? `${scenePoints.length} SPARSE POINTS` : `${scenePoints.length + (step === 6 ? 8 : 0)} GAUSSIANS`;
      context.save();
      roundedRect(context, box.x + 14, box.y + box.h - 34, 116, 20, 10);
      context.fillStyle = "rgba(10, 12, 14, 0.7)";
      context.fill();
      drawLabel(context, chipText, box.x + 72, box.y + box.h - 24, step === 0 ? COLORS.acid : COLORS.violet, "center");
      context.restore();
    }
  }

  function drawTargetScene(context, box, mode, cameraIndex, timestamp = 0) {
    const shift = (cameraIndex - 2.5) * box.w * 0.018;
    context.save();
    roundedRect(context, box.x, box.y, box.w, box.h, 5);
    context.clip();
    context.fillStyle = mode === "residual" ? "#080b0c" : "#172226";
    context.fillRect(box.x, box.y, box.w, box.h);

    if (mode === "target") {
      context.fillStyle = "#243337";
      context.fillRect(box.x, box.y + box.h * 0.67, box.w, box.h * 0.33);
      context.fillStyle = "rgba(217, 255, 111, 0.15)";
      context.beginPath();
      context.arc(box.x + box.w * 0.76 - shift, box.y + box.h * 0.22, box.w * 0.1, 0, Math.PI * 2);
      context.fill();
      context.fillStyle = COLORS.cyan;
      context.beginPath();
      context.moveTo(box.x + box.w * 0.37 + shift, box.y + box.h * 0.42);
      context.bezierCurveTo(
        box.x + box.w * 0.3 + shift,
        box.y + box.h * 0.62,
        box.x + box.w * 0.34 + shift,
        box.y + box.h * 0.86,
        box.x + box.w * 0.5 + shift,
        box.y + box.h * 0.86,
      );
      context.bezierCurveTo(
        box.x + box.w * 0.64 + shift,
        box.y + box.h * 0.83,
        box.x + box.w * 0.66 + shift,
        box.y + box.h * 0.56,
        box.x + box.w * 0.59 + shift,
        box.y + box.h * 0.42,
      );
      context.closePath();
      context.fill();
      context.fillStyle = "#0d1315";
      context.fillRect(box.x + box.w * 0.485 + shift, box.y + box.h * 0.25, 2, box.h * 0.25);
      const flowers = [
        [0.48, 0.23, COLORS.coral],
        [0.38, 0.31, COLORS.acid],
        [0.58, 0.3, COLORS.violet],
        [0.44, 0.37, "#f5b861"],
        [0.64, 0.22, COLORS.coral],
      ];
      flowers.forEach(([fx, fy, color], index) => {
        const x = box.x + box.w * fx + shift;
        const y = box.y + box.h * fy;
        context.strokeStyle = "#638170";
        context.lineWidth = 1;
        context.beginPath();
        context.moveTo(box.x + box.w * 0.5 + shift, box.y + box.h * 0.48);
        context.lineTo(x, y);
        context.stroke();
        context.fillStyle = color;
        for (let petal = 0; petal < 5; petal += 1) {
          const angle = (petal / 5) * Math.PI * 2 + index * 0.15;
          context.beginPath();
          context.ellipse(
            x + Math.cos(angle) * box.w * 0.018,
            y + Math.sin(angle) * box.w * 0.018,
            box.w * 0.018,
            box.w * 0.01,
            angle,
            0,
            Math.PI * 2,
          );
          context.fill();
        }
      });
      context.fillStyle = "rgba(242, 238, 229, 0.12)";
      context.fillRect(box.x + box.w * 0.12, box.y + box.h * 0.75, box.w * 0.16, 2);
      context.fillRect(box.x + box.w * 0.69, box.y + box.h * 0.82, box.w * 0.18, 2);
    } else if (mode === "render") {
      gaussianFill(context, box.x + box.w * 0.5 + shift, box.y + box.h * 0.68, box.w * 0.19, box.h * 0.3, 0, COLORS.cyan, 0.7);
      gaussianFill(context, box.x + box.w * 0.48 + shift, box.y + box.h * 0.5, box.w * 0.11, box.h * 0.2, 0.08, COLORS.cyan, 0.48);
      const flowers = [
        [0.48, 0.23, COLORS.coral],
        [0.38, 0.31, COLORS.acid],
        [0.58, 0.3, COLORS.violet],
        [0.44, 0.37, "#f5b861"],
        [0.64, 0.22, COLORS.coral],
      ];
      flowers.forEach(([fx, fy, color], index) => {
        gaussianFill(
          context,
          box.x + box.w * fx + shift,
          box.y + box.h * fy,
          box.w * (0.05 + (index % 2) * 0.01),
          box.h * 0.055,
          index * 0.7,
          color,
          0.72,
        );
      });
      gaussianFill(context, box.x + box.w * 0.73 - shift, box.y + box.h * 0.24, box.w * 0.13, box.h * 0.12, 0, COLORS.acid, 0.14);
      context.fillStyle = "rgba(242, 238, 229, 0.05)";
      context.fillRect(box.x, box.y + box.h * 0.72, box.w, box.h * 0.28);
    } else {
      const pulse = reducedMotion ? 0.5 : (Math.sin(timestamp * 0.004) + 1) / 2;
      gaussianFill(context, box.x + box.w * 0.38 + shift, box.y + box.h * 0.3, box.w * 0.11, box.h * 0.13, 0.3, COLORS.coral, 0.5 + pulse * 0.16);
      gaussianFill(context, box.x + box.w * 0.56 + shift, box.y + box.h * 0.58, box.w * 0.15, box.h * 0.24, -0.2, "#f5f0e8", 0.18);
      gaussianFill(context, box.x + box.w * 0.7 - shift, box.y + box.h * 0.24, box.w * 0.08, box.h * 0.09, 0, COLORS.coral, 0.4);
      context.strokeStyle = "rgba(255, 114, 94, 0.35)";
      context.lineWidth = 1;
      for (let y = box.y; y < box.y + box.h; y += 10) {
        context.beginPath();
        context.moveTo(box.x, y);
        context.lineTo(box.x + box.w, y);
        context.stroke();
      }
    }

    context.restore();
    context.save();
    context.strokeStyle = COLORS.line;
    roundedRect(context, box.x, box.y, box.w, box.h, 5);
    context.stroke();
    context.restore();
  }

  function drawImageStage(context, box, step, timestamp) {
    const inner = { x: box.x + 14, y: box.y + 52, w: box.w - 28, h: box.h - 70 };
    const titleY = box.y + 20;

    if (step === 0) {
      const rows = [
        ["cameras.bin", "Kᵢ · Rᵢ · tᵢ"],
        ["images.bin", "2D ↔ 3D tracks"],
        ["points3D.bin", "XYZ · RGB"],
      ];
      rows.forEach(([name, detail], index) => {
        const y = inner.y + index * 48;
        context.fillStyle = index === 2 ? "rgba(217, 255, 111, 0.07)" : "rgba(242, 238, 229, 0.025)";
        context.fillRect(inner.x, y, inner.w, 38);
        drawLabel(context, name, inner.x + 10, y + 14, index === 2 ? COLORS.acid : COLORS.paper);
        drawLabel(context, detail, inner.x + 10, y + 27, "#626a67");
      });
      drawLabel(context, "MULTI-VIEW CAPTURE", inner.x, inner.y + 170, COLORS.cyan);
      const thumbGap = 7;
      const thumbWidth = (inner.w - thumbGap * 2) / 3;
      for (let index = 0; index < 3; index += 1) {
        drawTargetScene(
          context,
          { x: inner.x + index * (thumbWidth + thumbGap), y: inner.y + 184, w: thumbWidth, h: Math.min(100, inner.h - 190) },
          "target",
          index * 2,
          timestamp,
        );
      }
      return;
    }

    if (step === 1) {
      const centerX = inner.x + inner.w * 0.5;
      const centerY = inner.y + inner.h * 0.42;
      gaussianFill(context, centerX, centerY, inner.w * 0.27, inner.h * 0.16, -0.42, COLORS.violet, 0.52);
      context.save();
      context.translate(centerX, centerY);
      context.rotate(-0.42);
      context.strokeStyle = COLORS.violet;
      context.lineWidth = 1.2;
      context.beginPath();
      context.ellipse(0, 0, inner.w * 0.2, inner.h * 0.11, 0, 0, Math.PI * 2);
      context.stroke();
      context.fillStyle = COLORS.paper;
      context.beginPath();
      context.arc(0, 0, 2.5, 0, Math.PI * 2);
      context.fill();
      context.restore();

      const properties = [
        ["μ", "SfM XYZ"],
        ["s", "nearest-neighbor scale"],
        ["q", "identity"],
        ["α", "0.1"],
        ["c", "SfM RGB → SH₀"],
      ];
      const columns = 2;
      properties.forEach(([symbol, value], index) => {
        const column = index % columns;
        const row = Math.floor(index / columns);
        const cellWidth = (inner.w - 8) / columns;
        const x = inner.x + column * (cellWidth + 8);
        const y = inner.y + inner.h * 0.68 + row * 38;
        context.fillStyle = "rgba(242, 238, 229, 0.035)";
        context.fillRect(x, y, cellWidth, 30);
        context.fillStyle = index === 0 ? COLORS.coral : COLORS.cyan;
        context.font = '400 16px "Iowan Old Style", serif';
        context.textBaseline = "middle";
        context.fillText(symbol, x + 9, y + 15);
        drawLabel(context, value, x + 30, y + 15, "#9ba19f");
      });
      return;
    }

    if (step === 2) {
      drawTargetScene(context, inner, "target", selectedCamera, timestamp);
      const cardWidth = Math.min(148, inner.w * 0.52);
      context.fillStyle = "rgba(10, 12, 14, 0.84)";
      context.fillRect(inner.x + 10, inner.y + inner.h - 52, cardWidth, 40);
      drawLabel(context, `CAMERA ${String(selectedCamera + 1).padStart(2, "0")}`, inner.x + 20, inner.y + inner.h - 38, COLORS.cyan);
      drawLabel(context, `image_${String(selectedCamera + 1).padStart(3, "0")}.jpg`, inner.x + 20, inner.y + inner.h - 24, "#979e9b");
      return;
    }

    if (step === 3) {
      drawTargetScene(context, inner, "render", selectedCamera, timestamp);
      context.save();
      context.strokeStyle = "rgba(217, 255, 111, 0.2)";
      context.lineWidth = 0.8;
      const tile = Math.max(20, Math.round(Math.min(inner.w, inner.h) / 7));
      for (let x = inner.x; x <= inner.x + inner.w; x += tile) {
        context.beginPath();
        context.moveTo(x, inner.y);
        context.lineTo(x, inner.y + inner.h);
        context.stroke();
      }
      for (let y = inner.y; y <= inner.y + inner.h; y += tile) {
        context.beginPath();
        context.moveTo(inner.x, y);
        context.lineTo(inner.x + inner.w, y);
        context.stroke();
      }
      const highlightX = inner.x + tile * 3;
      const highlightY = inner.y + tile * 2;
      context.fillStyle = "rgba(217, 255, 111, 0.09)";
      context.fillRect(highlightX, highlightY, tile, tile);
      context.strokeStyle = COLORS.acid;
      context.strokeRect(highlightX, highlightY, tile, tile);
      context.restore();
      context.fillStyle = "rgba(10, 12, 14, 0.84)";
      context.fillRect(inner.x + 10, inner.y + 10, 116, 42);
      drawLabel(context, "16 × 16 TILES", inner.x + 20, inner.y + 24, COLORS.acid);
      drawLabel(context, "SORT: TILE | DEPTH", inner.x + 20, inner.y + 39, "#939a97");
      return;
    }

    if (step === 4) {
      const gap = 7;
      const cardWidth = (inner.w - gap * 2) / 3;
      const labels = ["GROUND TRUTH", "RENDER", "RESIDUAL"];
      const modes = ["target", "render", "residual"];
      for (let index = 0; index < 3; index += 1) {
        const x = inner.x + index * (cardWidth + gap);
        drawLabel(context, labels[index], x, inner.y + 8, index === 2 ? COLORS.coral : "#909795");
        drawTargetScene(
          context,
          { x, y: inner.y + 20, w: cardWidth, h: inner.h * 0.62 },
          modes[index],
          selectedCamera,
          timestamp,
        );
      }
      const lossY = inner.y + inner.h * 0.72;
      context.fillStyle = "rgba(255, 114, 94, 0.08)";
      context.fillRect(inner.x, lossY, inner.w, inner.h - (lossY - inner.y));
      drawLabel(context, "COMBINED LOSS", inner.x + 12, lossY + 18, COLORS.coral);
      context.fillStyle = COLORS.paper;
      context.font = '400 24px "Iowan Old Style", serif';
      context.fillText("0.8 L₁  +  0.2 (1 − SSIM)", inner.x + 12, lossY + 49);
      return;
    }

    if (step === 5) {
      drawTargetScene(context, inner, "render", selectedCamera, timestamp);
      const pulse = reducedMotion ? 0.6 : (Math.sin(timestamp * 0.005) + 1) / 2;
      const arrows = [
        [0.38, 0.31, -1.1],
        [0.58, 0.29, -0.25],
        [0.49, 0.67, 0.45],
        [0.64, 0.58, 2.3],
      ];
      arrows.forEach(([px, py, angle], index) => {
        const x = inner.x + inner.w * px;
        const y = inner.y + inner.h * py;
        const length = 16 + pulse * 13 + index * 2;
        drawArrow(context, x, y, x + Math.cos(angle) * length, y + Math.sin(angle) * length, COLORS.coral, 1.4, 5);
      });
      const labels = ["∂μ", "∂s", "∂q", "∂α", "∂SH"];
      labels.forEach((label, index) => {
        const chipW = (inner.w - 8 * 4) / 5;
        const x = inner.x + index * (chipW + 8);
        const y = inner.y + inner.h - 34;
        context.fillStyle = index === 0 ? "rgba(255, 114, 94, 0.16)" : "rgba(242, 238, 229, 0.06)";
        context.fillRect(x, y, chipW, 28);
        drawLabel(context, label, x + chipW / 2, y + 14, index === 0 ? COLORS.coral : "#aab0ad", "center");
      });
      return;
    }

    const gap = 10;
    const cardWidth = (inner.w - gap) / 2;
    const cardHeight = inner.h * 0.62;
    drawLabel(context, "BEFORE", inner.x, inner.y + 8, "#8a918e");
    drawLabel(context, "AFTER", inner.x + cardWidth + gap, inner.y + 8, COLORS.acid);
    [0, 1].forEach((side) => {
      const card = {
        x: inner.x + side * (cardWidth + gap),
        y: inner.y + 20,
        w: cardWidth,
        h: cardHeight,
      };
      context.save();
      roundedRect(context, card.x, card.y, card.w, card.h, 5);
      context.clip();
      context.fillStyle = "#0c1113";
      context.fillRect(card.x, card.y, card.w, card.h);
      const count = side === 0 ? 9 : 19;
      for (let index = 0; index < count; index += 1) {
        const angle = index * 2.17;
        const radius = (0.08 + ((index * 0.39) % 0.36)) * Math.min(card.w, card.h);
        const x = card.x + card.w * 0.5 + Math.cos(angle) * radius;
        const y = card.y + card.h * 0.52 + Math.sin(angle) * radius * 0.72;
        gaussianFill(context, x, y, side === 0 ? 17 : 9, side === 0 ? 9 : 5, angle, pointPalette[index % pointPalette.length], 0.55);
      }
      context.restore();
      context.strokeStyle = COLORS.line;
      roundedRect(context, card.x, card.y, card.w, card.h, 5);
      context.stroke();
    });
    const actions = ["CLONE SMALL", "SPLIT LARGE", "PRUNE α < 0.005"];
    actions.forEach((action, index) => {
      const actionW = (inner.w - 8 * 2) / 3;
      const x = inner.x + index * (actionW + 8);
      const y = inner.y + inner.h - 38;
      context.fillStyle = index === 1 ? "rgba(217, 255, 111, 0.11)" : "rgba(242, 238, 229, 0.04)";
      context.fillRect(x, y, actionW, 30);
      drawLabel(context, action, x + actionW / 2, y + 15, index === 1 ? COLORS.acid : "#929996", "center");
    });
  }

  function drawProcess(timestamp = 0) {
    const { context, width, height } = prepareCanvas(processCanvas);
    context.clearRect(0, 0, width, height);
    context.fillStyle = "#0d1113";
    context.fillRect(0, 0, width, height);
    const layout = panelLayout(width, height);
    drawPanel(context, layout.world, currentStep === 0 ? "SFM WORLD" : "SHARED 3D MODEL", currentStep < 2 ? "ALL CAMERAS" : `VIEW ${String(selectedCamera + 1).padStart(2, "0")}`);
    drawPanel(context, layout.image, currentStep < 3 ? "TRAINING DATA" : "IMAGE SPACE", currentStep === 3 ? "FORWARD" : currentStep === 5 ? "BACKWARD" : "");
    drawSceneCore(context, layout.world, currentStep, timestamp, true);
    drawImageStage(context, layout.image, currentStep, timestamp);
  }

  function processAnimation(timestamp) {
    drawProcess(timestamp);
    window.requestAnimationFrame(processAnimation);
  }
  window.requestAnimationFrame(processAnimation);

  function updateStep(index) {
    currentStep = (index + stepData.length) % stepData.length;
    const data = stepData[currentStep];
    let activeTab = null;
    $$(".pipeline-tab").forEach((button, buttonIndex) => {
      const active = buttonIndex === currentStep;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-selected", String(active));
      if (active) activeTab = button;
    });
    if (window.innerWidth < 781 && activeTab) {
      const tabStrip = activeTab.parentElement;
      tabStrip.scrollTo({
        left: activeTab.offsetLeft - (tabStrip.clientWidth - activeTab.offsetWidth) / 2,
        behavior: reducedMotion ? "auto" : "smooth",
      });
    }
    $("#step-count").textContent = `STEP ${String(currentStep + 1).padStart(2, "0")} / 07`;
    $("#step-title").textContent = data.title;
    $("#step-copy").textContent = data.copy;
    $("#step-detail-label").textContent = data.label;
    $("#step-equation").textContent = data.equation;
    $("#lab-status").textContent = data.status;
    $("#change-pills").replaceChildren(...data.pills.map((text) => {
      const pill = document.createElement("span");
      pill.textContent = text;
      return pill;
    }));
  }

  $$(".pipeline-tab").forEach((button) => {
    button.addEventListener("click", () => {
      stopLoop();
      updateStep(Number(button.dataset.step));
    });
  });

  $("#previous-step").addEventListener("click", () => {
    stopLoop();
    updateStep(currentStep - 1);
  });

  $("#next-step").addEventListener("click", () => {
    stopLoop();
    updateStep(currentStep + 1);
  });

  function startLoop() {
    if (loopTimer) return;
    const button = $("#play-loop");
    button.setAttribute("aria-pressed", "true");
    $(".play-label", button).textContent = "Pause loop";
    $(".play-icon", button).textContent = "Ⅱ";
    loopTimer = window.setInterval(() => updateStep(currentStep + 1), 2100);
  }

  function stopLoop() {
    if (loopTimer) window.clearInterval(loopTimer);
    loopTimer = null;
    const button = $("#play-loop");
    button.setAttribute("aria-pressed", "false");
    $(".play-label", button).textContent = "Play loop";
    $(".play-icon", button).textContent = "▶";
  }

  $("#play-loop").addEventListener("click", () => {
    if (loopTimer) stopLoop();
    else startLoop();
  });

  processCanvas.addEventListener("click", (event) => {
    const rect = processCanvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const hit = processHitboxes.find((item) => Math.hypot(x - item.x, y - item.y) <= item.radius);
    if (hit) selectCamera(hit.index);
  });

  // Camera selection and loss
  const l1Values = [0.164, 0.151, 0.142, 0.156, 0.173, 0.148];
  const dssimValues = [0.241, 0.234, 0.226, 0.238, 0.257, 0.229];
  const lambdaSlider = $("#lambda-slider");

  function selectCamera(index) {
    selectedCamera = index;
    $$("[data-camera]").forEach((button) => {
      button.classList.toggle("is-active", Number(button.dataset.camera) === selectedCamera);
    });
    $("#camera-readout-id").textContent = `CAM ${String(selectedCamera + 1).padStart(2, "0")}`;
    $("#camera-readout-pose").textContent = `R${toSubscript(selectedCamera + 1)}, t${toSubscript(selectedCamera + 1)}`;
    $("#camera-readout-image").textContent = `image_${String(selectedCamera + 1).padStart(3, "0")}.jpg`;
    $("#loss-camera").textContent = String(selectedCamera + 1).padStart(2, "0");
    updateLoss();
  }

  function toSubscript(number) {
    const digits = "₀₁₂₃₄₅₆₇₈₉";
    return String(number)
      .split("")
      .map((digit) => digits[Number(digit)])
      .join("");
  }

  $$("[data-camera]").forEach((button) => {
    button.addEventListener("click", () => selectCamera(Number(button.dataset.camera)));
  });

  function updateLoss() {
    const lambda = Number(lambdaSlider.value);
    const l1 = l1Values[selectedCamera];
    const dssim = dssimValues[selectedCamera];
    const total = (1 - lambda) * l1 + lambda * dssim;
    $("#lambda-output").textContent = lambda.toFixed(2);
    $("#lambda-value").textContent = lambda.toFixed(2).replace(/^0/, "0");
    $("#one-minus-lambda").textContent = (1 - lambda).toFixed(2).replace(/^0/, "0");
    $("#l1-value").textContent = l1.toFixed(3);
    $("#ssim-value").textContent = dssim.toFixed(3);
    $("#total-loss").textContent = total.toFixed(3);
    $("#l1-meter").style.width = `${l1 * 200}%`;
    $("#ssim-meter").style.width = `${dssim * 200}%`;
    lambdaSlider.style.background = `linear-gradient(90deg, #b94536 0 ${lambda * 100}%, rgba(10, 12, 14, 0.16) ${lambda * 100}% 100%)`;
  }

  lambdaSlider.addEventListener("input", updateLoss);
  $("#reset-lambda").addEventListener("click", () => {
    lambdaSlider.value = "0.2";
    updateLoss();
  });
  updateLoss();

  // Dedicated camera diagram
  const cameraCanvas = $("#camera-canvas");
  let cameraHitboxes = [];

  function drawCameraDiagram(timestamp = 0) {
    const { context, width, height } = prepareCanvas(cameraCanvas);
    context.clearRect(0, 0, width, height);
    context.fillStyle = "#0c1113";
    context.fillRect(0, 0, width, height);
    const content = { x: 0, y: 0, w: width, h: height };
    drawGrid(context, content, 32, 0.045);
    const centerX = width * 0.5;
    const centerY = height * 0.52;
    const extent = Math.min(width, height) * 0.2;
    const pulse = reducedMotion ? 0.4 : (Math.sin(timestamp * 0.003) + 1) / 2;

    context.save();
    context.strokeStyle = "rgba(242, 238, 229, 0.1)";
    context.setLineDash([3, 7]);
    context.beginPath();
    context.ellipse(centerX, centerY, extent * 2.05, extent * 1.45, 0, 0, Math.PI * 2);
    context.stroke();
    context.restore();

    scenePoints.forEach((point, index) => {
      const x = centerX + point.x * extent;
      const y = centerY + point.y * extent;
      context.save();
      context.translate(x, y);
      context.rotate(point.rotation);
      context.fillStyle = point.color;
      context.globalAlpha = 0.22 + ((point.z + 1) / 2) * 0.22;
      context.beginPath();
      context.ellipse(0, 0, 3 + point.scale * 3.4, 2 + point.scale * 2, 0, 0, Math.PI * 2);
      context.fill();
      context.restore();
    });

    cameraHitboxes = [];
    cameraAngles.forEach((angle, index) => {
      const geometry = cameraGeometry(index, content, 0.4);
      const active = index === selectedCamera;
      if (active) {
        const direction = Math.atan2(centerY - geometry.y, centerX - geometry.x);
        const distance = Math.hypot(centerX - geometry.x, centerY - geometry.y) * 0.94;
        context.save();
        context.fillStyle = "rgba(97, 220, 232, 0.05)";
        context.strokeStyle = `rgba(97, 220, 232, ${0.3 + pulse * 0.2})`;
        context.beginPath();
        context.moveTo(geometry.x, geometry.y);
        context.lineTo(geometry.x + Math.cos(direction - 0.3) * distance, geometry.y + Math.sin(direction - 0.3) * distance);
        context.lineTo(geometry.x + Math.cos(direction + 0.3) * distance, geometry.y + Math.sin(direction + 0.3) * distance);
        context.closePath();
        context.fill();
        context.stroke();
        context.restore();
      } else {
        context.save();
        context.strokeStyle = "rgba(242, 238, 229, 0.07)";
        context.setLineDash([2, 7]);
        context.beginPath();
        context.moveTo(geometry.x, geometry.y);
        context.lineTo(centerX, centerY);
        context.stroke();
        context.restore();
      }
      drawCameraIcon(context, geometry, active, active ? 1.3 : 1);
      drawLabel(
        context,
        `CAM ${String(index + 1).padStart(2, "0")}`,
        geometry.x,
        geometry.y + 24,
        active ? COLORS.cyan : "#59615f",
        "center",
      );
      cameraHitboxes.push({ x: geometry.x, y: geometry.y, radius: 28, index });
    });

    context.fillStyle = "rgba(10, 12, 14, 0.74)";
    roundedRect(context, centerX - 64, centerY - 16, 128, 32, 16);
    context.fill();
    drawLabel(context, "ONE SHARED MODEL", centerX, centerY, COLORS.paper, "center");
  }

  function cameraAnimation(timestamp) {
    drawCameraDiagram(timestamp);
    window.requestAnimationFrame(cameraAnimation);
  }
  window.requestAnimationFrame(cameraAnimation);

  cameraCanvas.addEventListener("click", (event) => {
    const rect = cameraCanvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const hit = cameraHitboxes.find((item) => Math.hypot(x - item.x, y - item.y) <= item.radius);
    if (hit) selectCamera(hit.index);
  });

  // Rasterization explainer
  const rasterCanvas = $("#raster-canvas");
  let rasterStep = 0;

  function rasterBox(width, height) {
    return { x: 48, y: 66, w: width - 96, h: height - 106 };
  }

  function drawWireEllipsoid(context, x, y, rx, ry, rotation, color, alpha = 1) {
    context.save();
    context.translate(x, y);
    context.rotate(rotation);
    context.strokeStyle = color;
    context.globalAlpha = alpha;
    context.lineWidth = 1.2;
    context.beginPath();
    context.ellipse(0, 0, rx, ry, 0, 0, Math.PI * 2);
    context.stroke();
    context.beginPath();
    context.ellipse(0, 0, rx * 0.32, ry, 0, 0, Math.PI * 2);
    context.stroke();
    context.beginPath();
    context.ellipse(0, 0, rx, ry * 0.32, 0, 0, Math.PI * 2);
    context.stroke();
    context.restore();
  }

  function drawRasterProject(context, box, timestamp) {
    const narrow = box.w < 620;
    const cameraX = narrow ? box.x + box.w * 0.5 : box.x + box.w * 0.12;
    const cameraY = narrow ? box.y + box.h * 0.86 : box.y + box.h * 0.56;
    const worldX = narrow ? box.x + box.w * 0.5 : box.x + box.w * 0.43;
    const worldY = narrow ? box.y + box.h * 0.55 : box.y + box.h * 0.49;
    const screen = narrow
      ? { x: box.x + box.w * 0.16, y: box.y + 16, w: box.w * 0.68, h: box.h * 0.25 }
      : { x: box.x + box.w * 0.68, y: box.y + box.h * 0.18, w: box.w * 0.26, h: box.h * 0.64 };
    const pulse = reducedMotion ? 0.6 : (Math.sin(timestamp * 0.004) + 1) / 2;

    drawCameraIcon(context, { x: cameraX, y: cameraY, angle: narrow ? -Math.PI / 2 : 0 }, true, 1.3);
    drawLabel(context, "CALIBRATED CAMERA", cameraX, cameraY + (narrow ? 28 : 34), COLORS.cyan, "center");

    drawWireEllipsoid(context, worldX - 22, worldY - 12, 34, 18, -0.55, COLORS.coral, 0.9);
    drawWireEllipsoid(context, worldX + 22, worldY + 20, 25, 36, 0.32, COLORS.violet, 0.75);
    gaussianFill(context, worldX - 22, worldY - 12, 30, 15, -0.55, COLORS.coral, 0.2);
    gaussianFill(context, worldX + 22, worldY + 20, 22, 32, 0.32, COLORS.violet, 0.18);
    drawLabel(context, "3D COVARIANCE Σ", worldX, worldY + 64, "#929997", "center");

    context.save();
    context.strokeStyle = `rgba(217, 255, 111, ${0.2 + pulse * 0.2})`;
    context.setLineDash([5, 7]);
    context.beginPath();
    context.moveTo(cameraX, cameraY);
    context.lineTo(screen.x, screen.y + screen.h);
    context.moveTo(cameraX, cameraY);
    context.lineTo(screen.x + screen.w, screen.y + screen.h);
    context.stroke();
    context.restore();

    context.fillStyle = "#11191b";
    context.strokeStyle = COLORS.line;
    context.fillRect(screen.x, screen.y, screen.w, screen.h);
    context.strokeRect(screen.x, screen.y, screen.w, screen.h);
    gaussianFill(context, screen.x + screen.w * 0.42, screen.y + screen.h * 0.42, screen.w * 0.18, screen.h * 0.09, -0.35, COLORS.coral, 0.7);
    gaussianFill(context, screen.x + screen.w * 0.57, screen.y + screen.h * 0.58, screen.w * 0.13, screen.h * 0.18, 0.2, COLORS.violet, 0.62);
    drawLabel(context, "2D FOOTPRINT Σ′", screen.x + screen.w * 0.5, screen.y - 12, COLORS.acid, "center");

    const arrowStartX = narrow ? worldX : worldX + 62;
    const arrowStartY = narrow ? worldY - 58 : worldY;
    const arrowEndX = narrow ? screen.x + screen.w * 0.5 : screen.x - 10;
    const arrowEndY = narrow ? screen.y + screen.h + 10 : screen.y + screen.h * 0.5;
    drawArrow(context, arrowStartX, arrowStartY, arrowEndX, arrowEndY, COLORS.acid, 1.2, 6);
  }

  function drawRasterTiles(context, box, timestamp) {
    const screen = { x: box.x + box.w * 0.08, y: box.y + box.h * 0.08, w: box.w * 0.62, h: box.h * 0.82 };
    const tile = Math.max(30, Math.min(52, screen.w / 7));
    const cols = Math.floor(screen.w / tile);
    const rows = Math.floor(screen.h / tile);
    screen.w = cols * tile;
    screen.h = rows * tile;
    context.fillStyle = "#121a1d";
    context.fillRect(screen.x, screen.y, screen.w, screen.h);
    const activeCol = Math.min(3, cols - 1);
    const activeRow = Math.min(2, rows - 1);
    context.fillStyle = "rgba(217, 255, 111, 0.08)";
    context.fillRect(screen.x + activeCol * tile, screen.y + activeRow * tile, tile, tile);
    context.strokeStyle = "rgba(242, 238, 229, 0.12)";
    for (let col = 0; col <= cols; col += 1) {
      context.beginPath();
      context.moveTo(screen.x + col * tile, screen.y);
      context.lineTo(screen.x + col * tile, screen.y + screen.h);
      context.stroke();
    }
    for (let row = 0; row <= rows; row += 1) {
      context.beginPath();
      context.moveTo(screen.x, screen.y + row * tile);
      context.lineTo(screen.x + screen.w, screen.y + row * tile);
      context.stroke();
    }
    context.strokeStyle = COLORS.acid;
    context.strokeRect(screen.x + activeCol * tile, screen.y + activeRow * tile, tile, tile);

    const pulse = reducedMotion ? 0.5 : (Math.sin(timestamp * 0.004) + 1) / 2;
    const splats = [
      [0.38, 0.38, 76, 38, -0.25, COLORS.coral],
      [0.56, 0.54, 55, 88, 0.35, COLORS.violet],
      [0.65, 0.28, 64, 33, 0.65, COLORS.cyan],
      [0.27, 0.66, 48, 34, -0.7, COLORS.acid],
    ];
    splats.forEach(([px, py, rx, ry, rotation, color], index) => {
      gaussianFill(context, screen.x + screen.w * px, screen.y + screen.h * py, rx, ry, rotation, color, 0.28 + (index === 0 ? pulse * 0.1 : 0));
      context.save();
      context.translate(screen.x + screen.w * px, screen.y + screen.h * py);
      context.rotate(rotation);
      context.strokeStyle = color;
      context.globalAlpha = 0.55;
      context.beginPath();
      context.ellipse(0, 0, rx * 0.72, ry * 0.72, 0, 0, Math.PI * 2);
      context.stroke();
      context.restore();
    });

    const listX = screen.x + screen.w + 34;
    const available = box.x + box.w - listX;
    drawLabel(context, "GAUSSIAN INSTANCES", listX, screen.y + 4, COLORS.cyan);
    drawLabel(context, "ONE ENTRY / OVERLAPPED TILE", listX, screen.y + 20, "#626a68");
    for (let index = 0; index < 6; index += 1) {
      const y = screen.y + 46 + index * 38;
      context.fillStyle = index < 3 ? "rgba(217, 255, 111, 0.07)" : "rgba(242, 238, 229, 0.035)";
      context.fillRect(listX, y, available, 28);
      context.fillStyle = pointPalette[index % pointPalette.length];
      context.beginPath();
      context.arc(listX + 10, y + 14, 3, 0, Math.PI * 2);
      context.fill();
      drawLabel(context, `G${index + 1}  →  TILE ${18 + index}`, listX + 22, y + 14, index < 3 ? COLORS.paper : "#7f8784");
    }
  }

  function drawRasterSort(context, box, timestamp) {
    const listX = box.x + box.w * 0.12;
    const listW = box.w * 0.76;
    const rowH = Math.min(54, box.h * 0.105);
    const startY = box.y + box.h * 0.12;
    const pulse = reducedMotion ? 0 : (Math.sin(timestamp * 0.003) + 1) / 2;
    drawLabel(context, "64-BIT SORT KEY", listX, startY - 24, COLORS.cyan);
    drawLabel(context, "HIGH BITS: TILE ID", listX + listW * 0.58, startY - 24, "#737b78");
    drawLabel(context, "LOW BITS: DEPTH", listX + listW, startY - 24, "#737b78", "right");
    const rows = [
      { tile: "TILE 018", depth: "z 1.24", color: COLORS.coral },
      { tile: "TILE 018", depth: "z 2.07", color: COLORS.cyan },
      { tile: "TILE 018", depth: "z 3.81", color: COLORS.violet },
      { tile: "TILE 019", depth: "z 0.94", color: COLORS.acid },
      { tile: "TILE 019", depth: "z 2.44", color: "#f2b45f" },
      { tile: "TILE 020", depth: "z 1.65", color: COLORS.coral },
    ];
    rows.forEach((row, index) => {
      const y = startY + index * (rowH + 5);
      context.fillStyle = index < 3 ? "rgba(97, 220, 232, 0.055)" : "rgba(242, 238, 229, 0.025)";
      context.fillRect(listX, y, listW, rowH);
      context.fillStyle = row.color;
      context.globalAlpha = 0.72;
      context.beginPath();
      context.ellipse(listX + 22, y + rowH / 2, 10 + index * 0.6, 5, index * 0.32, 0, Math.PI * 2);
      context.fill();
      context.globalAlpha = 1;
      drawLabel(context, row.tile, listX + listW * 0.58, y + rowH / 2, index < 3 ? COLORS.cyan : "#8c9491", "right");
      drawLabel(context, row.depth, listX + listW - 18, y + rowH / 2, COLORS.paper, "right");
      if (index === 0) {
        context.fillStyle = `rgba(217, 255, 111, ${0.12 + pulse * 0.08})`;
        context.fillRect(listX, y, 2, rowH);
      }
    });
    drawLabel(context, "RADIX SORT → CONTIGUOUS RANGE PER TILE", listX, startY + rows.length * (rowH + 5) + 20, COLORS.acid);
  }

  function drawRasterComposite(context, box, timestamp) {
    const narrow = box.w < 650;
    const centerX = narrow ? box.x + box.w * 0.5 : box.x + box.w * 0.37;
    const centerY = box.y + box.h * 0.45;
    const pixelSize = Math.min(170, box.h * 0.36, box.w * (narrow ? 0.4 : 0.28));
    const pulse = reducedMotion ? 0.5 : ((timestamp * 0.0003) % 1);
    const layers = [
      { color: COLORS.coral, alpha: 0.48, z: 1.2 },
      { color: COLORS.cyan, alpha: 0.36, z: 2.1 },
      { color: COLORS.violet, alpha: 0.28, z: 3.4 },
      { color: COLORS.acid, alpha: 0.18, z: 4.0 },
    ];

    context.fillStyle = "#151c1f";
    context.fillRect(centerX - pixelSize / 2, centerY - pixelSize / 2, pixelSize, pixelSize);
    context.strokeStyle = COLORS.line;
    context.strokeRect(centerX - pixelSize / 2, centerY - pixelSize / 2, pixelSize, pixelSize);
    drawLabel(context, "ONE PIXEL", centerX, centerY - pixelSize / 2 - 15, COLORS.paper, "center");
    context.fillStyle = "rgba(242, 238, 229, 0.65)";
    context.beginPath();
    context.arc(centerX, centerY, 3, 0, Math.PI * 2);
    context.fill();

    const layerStartX = narrow ? box.x + box.w * 0.13 : box.x + box.w * 0.07;
    const layerEndX = narrow ? box.x + box.w * 0.87 : box.x + box.w * 0.65;
    layers.forEach((layer, index) => {
      const progress = clamp(pulse * 5 - index * 0.85, 0, 1);
      const x = lerp(layerStartX, layerEndX, index / (layers.length - 1));
      const y = centerY + pixelSize * 0.72;
      gaussianFill(context, x, y, 30, 13, index * 0.35, layer.color, 0.58);
      context.strokeStyle = layer.color;
      context.globalAlpha = index <= pulse * 4 ? 0.9 : 0.3;
      context.beginPath();
      context.moveTo(x, y - 9);
      context.lineTo(centerX + (index - 1.5) * 9, centerY + pixelSize / 2);
      context.stroke();
      context.globalAlpha = 1;
      drawLabel(context, `z ${layer.z.toFixed(1)}`, x, y + 23, layer.color, "center");
      if (progress > 0) {
        context.fillStyle = layer.color;
        context.globalAlpha = layer.alpha * progress;
        context.fillRect(centerX - pixelSize / 2, centerY - pixelSize / 2, pixelSize, pixelSize);
        context.globalAlpha = 1;
      }
    });

    const infoX = narrow ? box.x + 12 : box.x + box.w * 0.72;
    const infoY = narrow ? box.y + 14 : box.y + box.h * 0.19;
    const infoW = narrow ? box.w - 24 : box.w * 0.24;
    drawLabel(context, "FRONT → BACK", infoX, infoY, COLORS.acid);
    let transmittance = 1;
    layers.forEach((layer, index) => {
      const y = infoY + 28 + index * 40;
      context.fillStyle = "rgba(242, 238, 229, 0.035)";
      context.fillRect(infoX, y, infoW, 30);
      context.fillStyle = layer.color;
      context.fillRect(infoX, y, 3, 30);
      drawLabel(context, `α${index + 1} = ${layer.alpha.toFixed(2)}`, infoX + 13, y + 11, layer.color);
      drawLabel(context, `T = ${transmittance.toFixed(2)}`, infoX + 13, y + 22, "#929996");
      transmittance *= 1 - layer.alpha;
    });
    drawLabel(context, `FINAL TRANSMITTANCE ${transmittance.toFixed(2)}`, infoX, infoY + 204, COLORS.cyan);
  }

  function drawRaster(timestamp = 0) {
    const { context, width, height } = prepareCanvas(rasterCanvas);
    context.clearRect(0, 0, width, height);
    context.fillStyle = "#0c1012";
    context.fillRect(0, 0, width, height);
    drawGrid(context, { x: 0, y: 0, w: width, h: height }, 34, 0.035);
    const box = rasterBox(width, height);
    if (rasterStep === 0) drawRasterProject(context, box, timestamp);
    else if (rasterStep === 1) drawRasterTiles(context, box, timestamp);
    else if (rasterStep === 2) drawRasterSort(context, box, timestamp);
    else drawRasterComposite(context, box, timestamp);
  }

  function rasterAnimation(timestamp) {
    drawRaster(timestamp);
    window.requestAnimationFrame(rasterAnimation);
  }
  window.requestAnimationFrame(rasterAnimation);

  const rasterNames = ["Project", "Bin into tiles", "Sort by depth", "Composite"];
  $$(".raster-step").forEach((button) => {
    button.addEventListener("click", () => {
      rasterStep = Number(button.dataset.rasterStep);
      $$(".raster-step").forEach((item) => item.classList.toggle("is-active", item === button));
      $("#raster-stage-number").textContent = String(rasterStep + 1).padStart(2, "0");
      $("#raster-stage-name").textContent = rasterNames[rasterStep];
    });
  });

  // From fixed-camera 3DGS to Gaussian-splatting SLAM
  const slamCanvas = $("#slam-canvas");
  const slamGraph = $("#slam-graph");
  const slamStages = [
    {
      kicker: "ORIGINAL PAPER · OFFLINE",
      title: "SfM has already solved the cameras.",
      body:
        "All calibrated views supervise one shared Gaussian scene. A training iteration chooses a camera, but that camera is data—not a variable.",
      equation: "freeze {Tᵢ} · optimize 𝒢",
      pose: "fixed input",
      map: "learning",
      image: "calibrated set",
      global: "not needed",
      world: "SfM poses known",
      graph: "map-only optimization",
      next: "Next: track pose",
      canvasLabel: "Top-down Gaussian map surrounded by fixed structure-from-motion camera poses",
      graphLabel: "Fixed camera poses supervising one optimized Gaussian map",
    },
    {
      kicker: "SLAM · TRACKING",
      title: "Freeze the map; solve the new pose.",
      body:
        "Render the current Gaussian map from a pose hypothesis and compare it with the incoming frame. Pose gradients move the camera estimate until prediction and observation align.",
      equation: "T̂ₜ = argmin_T ℒ(R(𝒢; T), Iₜ)",
      pose: "optimizing",
      map: "frozen",
      image: "current frame",
      global: "local factor",
      world: "candidate pose moves",
      graph: "image constrains xₜ",
      next: "Next: update map",
      canvasLabel: "Current camera pose hypothesis moving into alignment with a fixed Gaussian map",
      graphLabel: "Current pose node connected to the Gaussian map through a splat image factor",
    },
    {
      kicker: "SLAM · MAPPING",
      title: "Accept poses; then improve the map.",
      body:
        "Selected keyframes become supervision. With their poses held still, the system inserts, densifies, prunes, and refines Gaussians in regions the current map explains poorly.",
      equation: "𝒢̂ = argmin_𝒢 Σₖ∈𝒦 ℒ(R(𝒢; T̂ₖ), Iₖ)",
      pose: "accepted",
      map: "optimizing",
      image: "keyframe window",
      global: "optional",
      world: "new splats appear",
      graph: "keyframes update 𝒢",
      next: "Next: close loop",
      canvasLabel: "Accepted keyframe poses supervising new and refined Gaussians",
      graphLabel: "Several accepted keyframe poses connected to the evolving Gaussian map",
    },
    {
      kicker: "GRAPHSLAM · GLOBAL CONSISTENCY",
      title: "A revisit can correct the whole trajectory.",
      body:
        "A loop factor joins poses that see the same place. Graph optimization distributes the correction through the trajectory; the Gaussian map must then be reconciled with the corrected poses.",
      equation: "argmin_{x₀…xₜ} Σ φprior + φodom + φsplat + φloop",
      pose: "joint update",
      map: "reconcile",
      image: "revisits + sensors",
      global: "active",
      world: "drift corrected",
      graph: "loop closes globally",
      next: "Restart chapter",
      canvasLabel: "Drifted camera path corrected globally after a loop closure",
      graphLabel: "Pose factor graph with odometry, splat factors, and a loop closure",
    },
  ];

  const slamPath = [
    { x: 0.14, y: 0.74 },
    { x: 0.2, y: 0.47 },
    { x: 0.37, y: 0.27 },
    { x: 0.62, y: 0.22 },
    { x: 0.82, y: 0.4 },
    { x: 0.8, y: 0.7 },
    { x: 0.55, y: 0.82 },
    { x: 0.28, y: 0.82 },
  ];

  let slamStep = 0;
  let slamTimer = null;

  function slamPoint(point, width, height) {
    return { x: point.x * width, y: point.y * height };
  }

  function drawSlamPath(context, points, width, height, color, dashed = false, lineWidth = 1.5) {
    context.save();
    context.strokeStyle = color;
    context.lineWidth = lineWidth;
    if (dashed) context.setLineDash([7, 7]);
    context.beginPath();
    points.forEach((point, index) => {
      const projected = slamPoint(point, width, height);
      if (index === 0) context.moveTo(projected.x, projected.y);
      else context.lineTo(projected.x, projected.y);
    });
    context.stroke();
    context.restore();
  }

  function drawSlamCamera(context, x, y, targetX, targetY, color, label, ghost = false) {
    const angle = Math.atan2(targetY - y, targetX - x);
    context.save();
    context.translate(x, y);
    context.rotate(angle);
    context.globalAlpha = ghost ? 0.38 : 1;
    context.fillStyle = "#111719";
    context.strokeStyle = color;
    context.lineWidth = ghost ? 1 : 1.8;
    roundedRect(context, -8, -6, 14, 12, 2);
    context.fill();
    context.stroke();
    context.beginPath();
    context.moveTo(6, -4);
    context.lineTo(14, -9);
    context.lineTo(14, 9);
    context.lineTo(6, 4);
    context.closePath();
    context.stroke();
    context.restore();
    if (label) drawLabel(context, label, x, y + 22, color, "center");
  }

  function drawSlamMap(context, width, height, timestamp, mapping = false) {
    const centerX = width * 0.5;
    const centerY = height * 0.52;
    const spreadX = width * 0.22;
    const spreadY = height * 0.22;
    const pulse = reducedMotion ? 0.65 : (Math.sin(timestamp * 0.004) + 1) / 2;
    scenePoints.slice(0, 34).forEach((point, index) => {
      const x = centerX + point.x * spreadX;
      const y = centerY + point.y * spreadY;
      const scaleBoost = mapping && index > 25 ? 0.65 + pulse * 0.55 : 1;
      gaussianFill(
        context,
        x,
        y,
        (6 + point.scale * 5) * scaleBoost,
        (3 + point.scale * point.stretch * 2.8) * scaleBoost,
        point.rotation,
        point.color,
        mapping && index > 25 ? 0.28 + pulse * 0.38 : 0.48
      );
    });
    context.save();
    context.strokeStyle = mapping ? "rgba(217, 255, 111, 0.24)" : "rgba(199, 166, 255, 0.18)";
    context.beginPath();
    context.ellipse(centerX, centerY, spreadX * 0.9, spreadY * 0.82, 0, 0, Math.PI * 2);
    context.stroke();
    context.restore();
    drawLabel(context, mapping ? "𝒢 · MAP UPDATE" : "𝒢 · FIXED WORLD MAP", centerX, centerY + spreadY + 24, mapping ? COLORS.acid : COLORS.violet, "center");
  }

  function drawSlamWorld(timestamp = 0) {
    const { context, width, height } = prepareCanvas(slamCanvas);
    context.clearRect(0, 0, width, height);
    context.fillStyle = "#0e1416";
    context.fillRect(0, 0, width, height);
    drawGrid(context, { x: 0, y: 0, w: width, h: height }, 30, 0.035);
    drawSlamMap(context, width, height, timestamp, slamStep === 2);
    const centerX = width * 0.5;
    const centerY = height * 0.52;

    if (slamStep === 0) {
      slamPath.slice(0, 6).forEach((point, index) => {
        const camera = slamPoint(point, width, height);
        context.save();
        context.strokeStyle = "rgba(97, 220, 232, 0.12)";
        context.setLineDash([4, 7]);
        context.beginPath();
        context.moveTo(camera.x, camera.y);
        context.lineTo(centerX, centerY);
        context.stroke();
        context.restore();
        drawSlamCamera(context, camera.x, camera.y, centerX, centerY, COLORS.cyan, `T${index}`);
      });
      drawLabel(context, "CALIBRATED ONCE BY SfM", 16, 18, COLORS.cyan);
    } else if (slamStep === 1) {
      drawSlamPath(context, slamPath.slice(0, 5), width, height, "rgba(97, 220, 232, 0.35)");
      slamPath.slice(0, 4).forEach((point) => {
        const camera = slamPoint(point, width, height);
        drawSlamCamera(context, camera.x, camera.y, centerX, centerY, "#6d7774", "");
      });
      const target = slamPoint(slamPath[4], width, height);
      const drift = { x: target.x + width * 0.09, y: target.y - height * 0.12 };
      const fit = reducedMotion ? 0.62 : (Math.sin(timestamp * 0.0024 - Math.PI / 2) + 1) / 2;
      const estimate = { x: lerp(drift.x, target.x, fit), y: lerp(drift.y, target.y, fit) };
      drawSlamCamera(context, target.x, target.y, centerX, centerY, COLORS.cyan, "target", true);
      drawSlamCamera(context, estimate.x, estimate.y, centerX, centerY, COLORS.coral, "T̂ₜ");
      drawArrow(context, drift.x, drift.y, target.x, target.y, COLORS.acid, 1.2, 6);
      drawLabel(context, "RENDER → RESIDUAL → POSE GRADIENT", 16, 18, COLORS.acid);
    } else if (slamStep === 2) {
      drawSlamPath(context, slamPath.slice(0, 6), width, height, "rgba(97, 220, 232, 0.38)");
      slamPath.slice(1, 6).forEach((point, index) => {
        const camera = slamPoint(point, width, height);
        drawSlamCamera(context, camera.x, camera.y, centerX, centerY, index === 4 ? COLORS.acid : "#75807d", index === 4 ? "keyframe" : "");
      });
      drawLabel(context, "POSES ACCEPTED · DENSIFY WHERE ERROR REMAINS", 16, 18, COLORS.acid);
    } else {
      const corrected = [...slamPath, slamPath[0]];
      const drifted = slamPath.map((point, index) => ({
        x: point.x + index * 0.012,
        y: point.y - index * 0.006 + Math.sin(index * 1.4) * 0.018,
      }));
      drawSlamPath(context, drifted, width, height, "rgba(255, 114, 94, 0.58)", true, 1.4);
      drawSlamPath(context, corrected, width, height, "rgba(97, 220, 232, 0.72)", false, 2);
      const start = slamPoint(slamPath[0], width, height);
      const end = slamPoint(slamPath[slamPath.length - 1], width, height);
      context.save();
      context.strokeStyle = COLORS.acid;
      context.lineWidth = 2.4;
      context.setLineDash([5, 5]);
      context.beginPath();
      context.moveTo(end.x, end.y);
      context.lineTo(start.x, start.y);
      context.stroke();
      context.restore();
      drawSlamCamera(context, start.x, start.y, centerX, centerY, COLORS.cyan, "x₀");
      drawSlamCamera(context, end.x, end.y, centerX, centerY, COLORS.acid, "xₜ");
      drawLabel(context, "CORAL: DRIFT · CYAN: CORRECTED · ACID: LOOP", 16, 18, COLORS.acid);
    }
  }

  function graphPose(x, y, label, className = "") {
    return `<circle class="slam-graph-node ${className}" cx="${x}" cy="${y}" r="24"></circle><text class="slam-graph-label" x="${x}" y="${y}">${label}</text>`;
  }

  function graphMap(x, y) {
    return `<ellipse class="slam-graph-node is-map" cx="${x}" cy="${y}" rx="48" ry="31"></ellipse><text class="slam-graph-label" x="${x}" y="${y}">𝒢</text>`;
  }

  function graphFactor(x, y) {
    return `<rect class="slam-factor-diamond" x="${x - 9}" y="${y - 9}" width="18" height="18" transform="rotate(45 ${x} ${y})"></rect>`;
  }

  function updateSlamGraph() {
    let markup = "";
    if (slamStep === 0) {
      markup = `
        <path class="slam-graph-edge is-splat" d="M112 92 L238 170 M408 92 L282 170 M112 276 L238 202 M408 276 L282 202"></path>
        ${graphPose(92, 80, "T₀", "is-fixed")}${graphPose(428, 80, "T₁", "is-fixed")}
        ${graphPose(92, 286, "T₂", "is-fixed")}${graphPose(428, 286, "T₃", "is-fixed")}
        ${graphMap(260, 186)}
        <text class="slam-graph-caption" x="260" y="338">LOCKED POSES · OPTIMIZE MAP</text>`;
    } else if (slamStep === 1) {
      markup = `
        <path class="slam-graph-edge" d="M92 112 L212 112 L352 112"></path>
        <path class="slam-graph-edge is-splat" d="M372 136 L344 212 M328 232 L282 260"></path>
        ${graphPose(72, 112, "x₀", "is-fixed")}${graphPose(212, 112, "x₁", "is-fixed")}${graphPose(372, 112, "xₜ", "is-current")}
        ${graphFactor(336, 224)}${graphMap(260, 286)}
        <text class="slam-graph-caption" x="336" y="196">SPLAT FACTOR</text>
        <text class="slam-graph-caption" x="260" y="338">MAP FROZEN · SOLVE xₜ</text>`;
    } else if (slamStep === 2) {
      markup = `
        <path class="slam-graph-edge is-splat" d="M104 108 L218 172 M260 108 L260 154 M416 108 L302 172"></path>
        ${graphPose(84, 96, "x₀", "is-fixed")}${graphPose(260, 84, "x₁", "is-fixed")}${graphPose(436, 96, "x₂", "is-fixed")}
        ${graphFactor(218, 172)}${graphFactor(260, 172)}${graphFactor(302, 172)}${graphMap(260, 254)}
        <text class="slam-graph-caption" x="260" y="322">POSES ACCEPTED · UPDATE 𝒢</text>`;
    } else {
      markup = `
        <path class="slam-graph-edge" d="M96 104 L210 62 L336 82 L422 172 L366 286 L212 294 L104 220"></path>
        <path class="slam-graph-edge is-loop" d="M96 104 Q34 166 104 220"></path>
        <path class="slam-graph-edge is-splat" d="M116 112 L234 178 M222 80 L246 166 M324 100 L276 168 M352 270 L276 208 M222 278 L246 214"></path>
        ${graphPose(78, 96, "x₀", "is-loop")}${graphPose(212, 54, "x₁")}${graphPose(350, 78, "x₂")}${graphPose(438, 174, "x₃")}
        ${graphPose(366, 296, "x₄")}${graphPose(212, 304, "x₅")}${graphPose(88, 226, "xₜ", "is-loop")}
        ${graphMap(260, 190)}
        <text class="slam-graph-caption" x="53" y="169">LOOP</text>
        <text class="slam-graph-caption" x="260" y="344">JOINT POSE UPDATE · THEN RECONCILE MAP</text>`;
    }
    slamGraph.innerHTML = markup;
  }

  function stopSlamPlayback() {
    if (slamTimer !== null) window.clearInterval(slamTimer);
    slamTimer = null;
    $("#slam-play").setAttribute("aria-pressed", "false");
    $("#slam-play").innerHTML = '<span aria-hidden="true">▶</span> Play chapter';
  }

  function updateSlamStep(nextStep, keepPlaying = false) {
    slamStep = (nextStep + slamStages.length) % slamStages.length;
    const stage = slamStages[slamStep];
    $$(".slam-tab").forEach((button) => {
      const active = Number(button.dataset.slamStep) === slamStep;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-selected", String(active));
    });
    $("#slam-step-count").textContent = `${String(slamStep + 1).padStart(2, "0")} / 04`;
    $("#slam-kicker").textContent = stage.kicker;
    $("#slam-stage-title").textContent = stage.title;
    $("#slam-stage-body").textContent = stage.body;
    $("#slam-equation").textContent = stage.equation;
    $("#slam-pose-state").textContent = stage.pose;
    $("#slam-map-state").textContent = stage.map;
    $("#slam-image-state").textContent = stage.image;
    $("#slam-global-state").textContent = stage.global;
    $("#slam-world-state").textContent = stage.world;
    $("#slam-graph-state").textContent = stage.graph;
    slamCanvas.setAttribute("aria-label", stage.canvasLabel);
    slamGraph.setAttribute("aria-label", stage.graphLabel);
    $("#slam-previous").disabled = slamStep === 0;
    $("#slam-next").innerHTML = `${stage.next} <span aria-hidden="true">${slamStep === slamStages.length - 1 ? "↺" : "→"}</span>`;
    updateSlamGraph();
    drawSlamWorld(performance.now());
    if (!keepPlaying) stopSlamPlayback();
  }

  $$(".slam-tab").forEach((button) => {
    button.addEventListener("click", () => updateSlamStep(Number(button.dataset.slamStep)));
  });

  $("#slam-previous").addEventListener("click", () => updateSlamStep(slamStep - 1));
  $("#slam-next").addEventListener("click", () => updateSlamStep(slamStep + 1));
  $("#slam-play").addEventListener("click", () => {
    if (slamTimer !== null) {
      stopSlamPlayback();
      return;
    }
    $("#slam-play").setAttribute("aria-pressed", "true");
    $("#slam-play").innerHTML = '<span aria-hidden="true">Ⅱ</span> Pause chapter';
    slamTimer = window.setInterval(() => updateSlamStep(slamStep + 1, true), 2300);
  });

  function slamAnimation(timestamp) {
    drawSlamWorld(timestamp);
    window.requestAnimationFrame(slamAnimation);
  }
  updateSlamStep(0);
  window.requestAnimationFrame(slamAnimation);

  // Keyboard navigation for the main lab when it has focus context.
  document.addEventListener("keydown", (event) => {
    if (event.target.matches("input, button, a")) return;
    if (event.key === "ArrowRight" && window.location.hash === "#loop") updateStep(currentStep + 1);
    if (event.key === "ArrowLeft" && window.location.hash === "#loop") updateStep(currentStep - 1);
  });

  selectCamera(selectedCamera);
  updateStep(0);
})();
