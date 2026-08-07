(() => {
  "use strict";

  const COLORS = {
    ink: "#f4f6f8",
    panel: "#ffffff",
    panel2: "#f7f9fa",
    paper: "#16222e",
    muted: "#667582",
    faint: "rgba(22, 34, 46, 0.08)",
    line: "rgba(22, 34, 46, 0.17)",
    coral: "#e8720c",
    cyan: "#0e8f7e",
    acid: "#7c4dbe",
    violet: "#1874b8",
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
    context.strokeStyle = `rgba(22, 34, 46, ${Math.min(alpha * 1.8, 0.13)})`;
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
    drawLabel(context, title, box.x + 16, box.y + 19, "#51606e");
    if (trailing) drawLabel(context, trailing, box.x + box.w - 16, box.y + 19, "#8a97a3", "right");
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
  const pointPalette = [COLORS.coral, COLORS.cyan, COLORS.acid, COLORS.violet, "#d39a42"];
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
      context.strokeStyle = "rgba(14, 143, 126, 0.14)";
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
      context.strokeStyle = "rgba(124, 77, 190, 0.3)";
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
    context.strokeStyle = "rgba(22, 34, 46, 0.09)";
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
      context.fillStyle = "rgba(14, 143, 126, 0.05)";
      context.strokeStyle = "rgba(14, 143, 126, 0.28)";
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
      context.strokeStyle = active ? "rgba(14, 143, 126, 0.45)" : "rgba(22, 34, 46, 0.1)";
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
        active ? COLORS.cyan : "#7b8994",
        "center",
      );
      processHitboxes.push({ x: geometry.x, y: geometry.y, radius: 22, index });
    });

    if (includeLabels) {
      const chipText = step === 0 ? `${scenePoints.length} SPARSE POINTS` : `${scenePoints.length + (step === 6 ? 8 : 0)} GAUSSIANS`;
      context.save();
      roundedRect(context, box.x + 14, box.y + box.h - 34, 116, 20, 10);
      context.fillStyle = "rgba(255, 255, 255, 0.9)";
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
    context.fillStyle = mode === "residual" ? "#f8ece8" : "#edf3f2";
    context.fillRect(box.x, box.y, box.w, box.h);

    if (mode === "target") {
      context.fillStyle = "#dce8e6";
      context.fillRect(box.x, box.y + box.h * 0.67, box.w, box.h * 0.33);
      context.fillStyle = "rgba(124, 77, 190, 0.13)";
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
      context.fillStyle = "#51606e";
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
      context.fillStyle = "rgba(22, 34, 46, 0.12)";
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
      context.fillStyle = "rgba(22, 34, 46, 0.045)";
      context.fillRect(box.x, box.y + box.h * 0.72, box.w, box.h * 0.28);
    } else {
      const pulse = reducedMotion ? 0.5 : (Math.sin(timestamp * 0.004) + 1) / 2;
      gaussianFill(context, box.x + box.w * 0.38 + shift, box.y + box.h * 0.3, box.w * 0.11, box.h * 0.13, 0.3, COLORS.coral, 0.5 + pulse * 0.16);
      gaussianFill(context, box.x + box.w * 0.56 + shift, box.y + box.h * 0.58, box.w * 0.15, box.h * 0.24, -0.2, "#51606e", 0.18);
      gaussianFill(context, box.x + box.w * 0.7 - shift, box.y + box.h * 0.24, box.w * 0.08, box.h * 0.09, 0, COLORS.coral, 0.4);
      context.strokeStyle = "rgba(194, 47, 47, 0.3)";
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
        context.fillStyle = index === 2 ? "rgba(124, 77, 190, 0.07)" : "rgba(22, 34, 46, 0.025)";
        context.fillRect(inner.x, y, inner.w, 38);
        drawLabel(context, name, inner.x + 10, y + 14, index === 2 ? COLORS.acid : COLORS.paper);
        drawLabel(context, detail, inner.x + 10, y + 27, "#7a8893");
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
        context.fillStyle = "rgba(22, 34, 46, 0.035)";
        context.fillRect(x, y, cellWidth, 30);
        context.fillStyle = index === 0 ? COLORS.coral : COLORS.cyan;
        context.font = '400 16px "Iowan Old Style", serif';
        context.textBaseline = "middle";
        context.fillText(symbol, x + 9, y + 15);
        drawLabel(context, value, x + 30, y + 15, "#667582");
      });
      return;
    }

    if (step === 2) {
      drawTargetScene(context, inner, "target", selectedCamera, timestamp);
      const cardWidth = Math.min(148, inner.w * 0.52);
      context.fillStyle = "rgba(255, 255, 255, 0.9)";
      context.fillRect(inner.x + 10, inner.y + inner.h - 52, cardWidth, 40);
      drawLabel(context, `CAMERA ${String(selectedCamera + 1).padStart(2, "0")}`, inner.x + 20, inner.y + inner.h - 38, COLORS.cyan);
      drawLabel(context, `image_${String(selectedCamera + 1).padStart(3, "0")}.jpg`, inner.x + 20, inner.y + inner.h - 24, "#667582");
      return;
    }

    if (step === 3) {
      drawTargetScene(context, inner, "render", selectedCamera, timestamp);
      context.save();
      context.strokeStyle = "rgba(124, 77, 190, 0.18)";
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
      context.fillStyle = "rgba(124, 77, 190, 0.08)";
      context.fillRect(highlightX, highlightY, tile, tile);
      context.strokeStyle = COLORS.acid;
      context.strokeRect(highlightX, highlightY, tile, tile);
      context.restore();
      context.fillStyle = "rgba(255, 255, 255, 0.9)";
      context.fillRect(inner.x + 10, inner.y + 10, 116, 42);
      drawLabel(context, "16 × 16 TILES", inner.x + 20, inner.y + 24, COLORS.acid);
      drawLabel(context, "SORT: TILE | DEPTH", inner.x + 20, inner.y + 39, "#667582");
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
      context.fillStyle = "rgba(232, 114, 12, 0.07)";
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
        context.fillStyle = index === 0 ? "rgba(232, 114, 12, 0.13)" : "rgba(22, 34, 46, 0.05)";
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
      context.fillStyle = "#fbfcfd";
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
      context.fillStyle = index === 1 ? "rgba(124, 77, 190, 0.1)" : "rgba(22, 34, 46, 0.04)";
      context.fillRect(x, y, actionW, 30);
      drawLabel(context, action, x + actionW / 2, y + 15, index === 1 ? COLORS.acid : "#929996", "center");
    });
  }

  function drawProcess(timestamp = 0) {
    const { context, width, height } = prepareCanvas(processCanvas);
    context.clearRect(0, 0, width, height);
    context.fillStyle = "#fbfcfd";
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
    context.fillStyle = "#fbfcfd";
    context.fillRect(0, 0, width, height);
    const content = { x: 0, y: 0, w: width, h: height };
    drawGrid(context, content, 32, 0.045);
    const centerX = width * 0.5;
    const centerY = height * 0.52;
    const extent = Math.min(width, height) * 0.2;
    const pulse = reducedMotion ? 0.4 : (Math.sin(timestamp * 0.003) + 1) / 2;

    context.save();
    context.strokeStyle = "rgba(22, 34, 46, 0.1)";
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
        context.fillStyle = "rgba(14, 143, 126, 0.05)";
        context.strokeStyle = `rgba(14, 143, 126, ${0.3 + pulse * 0.2})`;
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
        context.strokeStyle = "rgba(22, 34, 46, 0.08)";
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

    context.fillStyle = "rgba(255, 255, 255, 0.9)";
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
    context.strokeStyle = `rgba(124, 77, 190, ${0.2 + pulse * 0.2})`;
    context.setLineDash([5, 7]);
    context.beginPath();
    context.moveTo(cameraX, cameraY);
    context.lineTo(screen.x, screen.y + screen.h);
    context.moveTo(cameraX, cameraY);
    context.lineTo(screen.x + screen.w, screen.y + screen.h);
    context.stroke();
    context.restore();

    context.fillStyle = "#f7f9fa";
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
    context.fillStyle = "#f7f9fa";
    context.fillRect(screen.x, screen.y, screen.w, screen.h);
    const activeCol = Math.min(3, cols - 1);
    const activeRow = Math.min(2, rows - 1);
    context.fillStyle = "rgba(124, 77, 190, 0.08)";
    context.fillRect(screen.x + activeCol * tile, screen.y + activeRow * tile, tile, tile);
    context.strokeStyle = "rgba(22, 34, 46, 0.12)";
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
      context.fillStyle = index < 3 ? "rgba(124, 77, 190, 0.07)" : "rgba(22, 34, 46, 0.035)";
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
      context.fillStyle = index < 3 ? "rgba(14, 143, 126, 0.055)" : "rgba(22, 34, 46, 0.025)";
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
        context.fillStyle = `rgba(124, 77, 190, ${0.12 + pulse * 0.08})`;
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

    context.fillStyle = "#eef2f4";
    context.fillRect(centerX - pixelSize / 2, centerY - pixelSize / 2, pixelSize, pixelSize);
    context.strokeStyle = COLORS.line;
    context.strokeRect(centerX - pixelSize / 2, centerY - pixelSize / 2, pixelSize, pixelSize);
    drawLabel(context, "ONE PIXEL", centerX, centerY - pixelSize / 2 - 15, COLORS.paper, "center");
    context.fillStyle = "rgba(22, 34, 46, 0.65)";
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
      context.fillStyle = "rgba(22, 34, 46, 0.035)";
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
    context.fillStyle = "#fbfcfd";
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
    context.fillStyle = "#ffffff";
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
    context.strokeStyle = mapping ? "rgba(124, 77, 190, 0.24)" : "rgba(24, 116, 184, 0.2)";
    context.beginPath();
    context.ellipse(centerX, centerY, spreadX * 0.9, spreadY * 0.82, 0, 0, Math.PI * 2);
    context.stroke();
    context.restore();
    drawLabel(context, mapping ? "𝒢 · MAP UPDATE" : "𝒢 · FIXED WORLD MAP", centerX, centerY + spreadY + 24, mapping ? COLORS.acid : COLORS.violet, "center");
  }

  function drawSlamWorld(timestamp = 0) {
    const { context, width, height } = prepareCanvas(slamCanvas);
    context.clearRect(0, 0, width, height);
    context.fillStyle = "#fbfcfd";
    context.fillRect(0, 0, width, height);
    drawGrid(context, { x: 0, y: 0, w: width, h: height }, 30, 0.035);
    drawSlamMap(context, width, height, timestamp, slamStep === 2);
    const centerX = width * 0.5;
    const centerY = height * 0.52;

    if (slamStep === 0) {
      slamPath.slice(0, 6).forEach((point, index) => {
        const camera = slamPoint(point, width, height);
        context.save();
        context.strokeStyle = "rgba(14, 143, 126, 0.14)";
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
      drawSlamPath(context, slamPath.slice(0, 5), width, height, "rgba(14, 143, 126, 0.38)");
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
      drawSlamPath(context, slamPath.slice(0, 6), width, height, "rgba(14, 143, 126, 0.4)");
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
      drawSlamPath(context, drifted, width, height, "rgba(232, 114, 12, 0.62)", true, 1.4);
      drawSlamPath(context, corrected, width, height, "rgba(14, 143, 126, 0.76)", false, 2);
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

  // Radio multipath optimization with an unknown UE pose
  const radioSvg = $("#radio-opt-svg");
  const radioHeading = $("#radio-heading");
  const radioBounce = $("#radio-bounce");
  const radioSweep = $("#radio-sweep");

  const RADIO = {
    ink: "#16222e",
    soft: "#51606e",
    muted: "#8a97a3",
    line: "#d7dee5",
    va: "#7c4dbe",
    vaDeep: "#5d3691",
    incidence: "#0e8f7e",
    incidenceDeep: "#0a6b5e",
    measurement: "#e8720c",
    measurementDeep: "#b45607",
    scatter: "#1874b8",
    ue: "#2ca02c",
    ueDeep: "#1d7a1d",
    error: "#c22f2f",
  };

  const radioCases = [
    {
      kicker: "§3.1 · SINGLE BOUNCE",
      title: "One path leaves a family, not a point.",
      body:
        "Walk the measured length along the departure ray to E. Pick a bounce P, rotate the body-frame AoA by a heading hypothesis, and the candidate UE and wall follow. Every P and θ remains coherent.",
      equation: "E = BS + Ldψ · state = (P, θ)",
      caption:
        "The endpoint E is the mirror image of the UE for this path. Re-pinning the remaining length at P and rotating φbody by a candidate θ constructs a valid UE and an implied wall. One bounce cannot identify position, wall, or heading by itself.",
      figure: "two-parameter family",
      family: "2D family",
      rank: "rank deficient",
      aria: "Single-bounce construction showing a family of UE and wall hypotheses for unknown heading",
    },
    {
      kicker: "§3.2 · DOUBLE BOUNCE",
      title: "A second path rejects slices, but does not fix heading.",
      body:
        "Path 1 proposes wall A. Path 2 must hit that wall in the forward direction, reflect, and spend a positive remaining delay before its second incidence point. Invalid ray order is rejected in red.",
      equation: "forward hits + positive segments · family (P¹, θ)",
      caption:
        "The second measured path strips its first bounce at wall A, then constructs wall B from the remaining delay and AoA. It prunes nonphysical hypotheses, but the feasible subset still moves with P¹ and θ; another path is a constraint, not automatically a unique pose.",
      figure: "ordered two-bounce test",
      family: "2D subset",
      rank: "still deficient",
      aria: "Corner scene with single- and double-bounce paths testing forward-order feasibility",
    },
    {
      kicker: "§3.3 · TRIPLE BOUNCE",
      title: "The same construction climbs one rung further.",
      body:
        "The path-1 hypothesis gives wall A; a valid two-bounce prefix gives wall B. Path 3 must hit both walls in order before its delay budget can form the last incidence point and wall C.",
      equation: "A → B → C · require every segment > 0",
      caption:
        "A triple-bounce path adds another forward-order and remaining-range test. It can remove more impossible slices, yet the surviving wall set and UE still change together with the unknown heading. More bounces do not, by themselves, guarantee observability.",
      figure: "recursive feasibility",
      family: "2D subset",
      rank: "still deficient",
      aria: "Corner scene with a recursively constructed triple-bounce path and three wall hypotheses",
    },
    {
      kicker: "§3.4 · CORRIDOR DOUBLE BOUNCE",
      title: "Parallel walls collapse two candidate lines into one.",
      body:
        "At the reference heading, both radio paths produce the same cross-corridor UE rail. There is no line crossing to solve. A heading hypothesis wedges the wall estimates, but the admissible crossing continues to slide.",
      equation: "line₂ ≡ line₁ at θ = 0 · continuum survives",
      caption:
        "In a corridor the double-bounce construction does not create an independent transverse direction. Clean data makes the two candidate lines coincide; changing θ re-dresses the parallel corridor as a wedge while keeping a continuous family of valid poses.",
      figure: "coincident candidate rails",
      family: "continuum",
      rank: "rank deficient",
      aria: "Parallel-wall corridor showing coincident UE candidate rails for two-bounce radio paths",
    },
    {
      kicker: "§3.5 · CORRIDOR TRIPLE BOUNCE",
      title: "Path parity checks the model, not the null direction.",
      body:
        "The third path alternates right–left–right. Its ordered bounces can reject bad slices and verify the two-wall explanation, while translation across the corridor remains coupled to the wall offsets.",
      equation: "R → L → R · parity valid, gauge remains",
      caption:
        "A higher-order path is valuable evidence: it checks bounce order, delay budget, and the alternating-wall model. But if every recovered normal is parallel, those measurements still lack the independent direction needed to pin the corridor and trajectory absolutely.",
      figure: "parity-valid continuum",
      family: "1D continuum",
      rank: "one null mode",
      aria: "Corridor triple-bounce path alternating between parallel wall hypotheses",
    },
    {
      kicker: "§3.6 · SPECIAL OBSERVABILITY TEST",
      title: "A rank test says whether optimization can return a point.",
      body:
        "Now add globally referenced UE displacements. Nonparallel wall normals give a full-rank cross-family system; parallel corridor normals give rank one, so the solver should report a line of answers.",
      equation: "rank[−2nA | 2nB] = 2 (corner), 1 (corridor)",
      caption:
        "This final case changes the sensor assumption. With global displacement, a corner supplies two independent wall normals and closes the offsets. A corridor supplies only one normal direction: walls and trajectory can slide together along the remaining null direction.",
      figure: "full rank vs rank one",
      family: "point / line",
      rank: "rank 2 / rank 1",
      aria: "Side-by-side observability comparison of a full-rank corner and rank-one corridor",
    },
  ];

  let radioCase = 0;
  let radioSweepFrame = null;

  const rAdd = (a, b) => ({ x: a.x + b.x, y: a.y + b.y });
  const rSub = (a, b) => ({ x: a.x - b.x, y: a.y - b.y });
  const rMul = (a, scalar) => ({ x: a.x * scalar, y: a.y * scalar });
  const rDot = (a, b) => a.x * b.x + a.y * b.y;
  const rCross = (a, b) => a.x * b.y - a.y * b.x;
  const rLength = (a) => Math.hypot(a.x, a.y);
  const rDistance = (a, b) => rLength(rSub(a, b));
  const rUnit = (a) => {
    const length = rLength(a);
    return length > 1e-9 ? rMul(a, 1 / length) : { x: 1, y: 0 };
  };
  const rPerp = (a) => ({ x: -a.y, y: a.x });
  const rReflect = (direction, normal) => rSub(direction, rMul(normal, 2 * rDot(direction, normal)));
  const rDirection = (angle) => ({ x: Math.cos(angle), y: Math.sin(angle) });
  const rAngle = (direction) => Math.atan2(direction.y, direction.x);
  const rPointOn = (a, b, t) => rAdd(a, rMul(rSub(b, a), t));
  const rNumber = (value) => Number(value.toFixed(2));

  function rIntersectLines(originA, directionA, originB, directionB) {
    const denominator = rCross(directionA, directionB);
    if (Math.abs(denominator) < 1e-7) return null;
    const delta = rSub(originB, originA);
    const t = rCross(delta, directionB) / denominator;
    const s = rCross(delta, directionA) / denominator;
    return { point: rAdd(originA, rMul(directionA, t)), t, s };
  }

  function rHitRayLine(origin, direction, wallPoint, wallNormal) {
    const denominator = rDot(wallNormal, direction);
    if (Math.abs(denominator) < 1e-7) return null;
    const t = rDot(wallNormal, rSub(wallPoint, origin)) / denominator;
    return { point: rAdd(origin, rMul(direction, t)), t };
  }

  function rReflectPoint(point, wall) {
    const signedDistance = rDot(wall.n, rSub(point, wall.p));
    return rSub(point, rMul(wall.n, 2 * signedDistance));
  }

  function rTraceReference(origin, destination, walls) {
    let image = destination;
    for (let index = walls.length - 1; index >= 0; index -= 1) image = rReflectPoint(image, walls[index]);
    let current = origin;
    let direction = rUnit(rSub(image, origin));
    const points = [origin];
    walls.forEach((wall) => {
      const hit = rHitRayLine(current, direction, wall.p, wall.n);
      if (!hit || hit.t <= 0) return;
      points.push(hit.point);
      current = hit.point;
      direction = rReflect(direction, wall.n);
    });
    points.push(destination);
    return points;
  }

  function rMeasurePath(points) {
    let length = 0;
    for (let index = 1; index < points.length; index += 1) length += rDistance(points[index - 1], points[index]);
    return {
      L: length,
      psi: rAngle(rSub(points[1], points[0])),
      phi: rAngle(rSub(points[points.length - 2], points[points.length - 1])),
      points,
    };
  }

  function rWallFromBounce(previous, bounce, next) {
    const towardPrevious = rUnit(rSub(previous, bounce));
    const towardNext = rUnit(rSub(next, bounce));
    const tangent = rUnit(rSub(towardPrevious, towardNext));
    return { p: bounce, t: tangent, n: rPerp(tangent) };
  }

  const cornerReference = (() => {
    const B = { x: 120, y: 390 };
    const U = { x: 550, y: 390 };
    const wallA = { p: { x: 335, y: 200 }, n: { x: 0, y: 1 }, t: { x: 1, y: 0 }, label: "A" };
    const wallB = { p: { x: 620, y: 330 }, n: { x: 1, y: 0 }, t: { x: 0, y: 1 }, label: "B" };
    const path1 = rTraceReference(B, U, [wallA]);
    const path2 = rTraceReference(B, U, [wallA, wallB]);
    const first3 = { x: 400, y: 200 };
    const d3 = rUnit(rSub(first3, B));
    const afterA = rReflect(d3, wallA.n);
    const secondHit = rHitRayLine(first3, afterA, wallB.p, wallB.n).point;
    const afterB = rReflect(afterA, wallB.n);
    const thirdHit = rAdd(secondHit, rMul(afterB, 170));
    const path3 = [B, first3, secondHit, thirdHit, U];
    const wallC = { ...rWallFromBounce(secondHit, thirdHit, U), label: "C" };
    return {
      B,
      U,
      walls: [wallA, wallB, wallC],
      measures: [rMeasurePath(path1), rMeasurePath(path2), rMeasurePath(path3)],
    };
  })();

  const corridorReference = (() => {
    const B = { x: 360, y: 105 };
    const U = { x: 390, y: 420 };
    const wallR = { p: { x: 610, y: 260 }, n: { x: 1, y: 0 }, t: { x: 0, y: 1 }, label: "R" };
    const wallL = { p: { x: 150, y: 260 }, n: { x: 1, y: 0 }, t: { x: 0, y: 1 }, label: "L" };
    const path1 = rTraceReference(B, U, [wallR]);
    const path2 = rTraceReference(B, U, [wallR, wallL]);
    const path3 = rTraceReference(B, U, [wallR, wallL, wallR]);
    return {
      B,
      U,
      walls: [wallR, wallL],
      measures: [rMeasurePath(path1), rMeasurePath(path2), rMeasurePath(path3)],
    };
  })();

  function rConstructSingle(scene, theta, bounceFraction) {
    const measurement = scene.measures[0];
    const departure = rDirection(measurement.psi);
    const arrival = rDirection(measurement.phi + theta);
    const E = rAdd(scene.B, rMul(departure, measurement.L));
    const travelled = bounceFraction * measurement.L;
    const P = rAdd(scene.B, rMul(departure, travelled));
    const remaining = measurement.L - travelled;
    const U = rSub(P, rMul(arrival, remaining));
    const normal = rUnit(rSub(U, E));
    return {
      B: scene.B,
      E,
      P,
      U,
      n: normal,
      t: rPerp(normal),
      line: { o: rSub(scene.B, rMul(arrival, measurement.L)), d: rAdd(departure, arrival) },
      departure,
      arrival,
      remaining,
      valid: true,
    };
  }

  function rConstructDouble(scene, single, theta) {
    const measurement = scene.measures[1];
    const departure = rDirection(measurement.psi);
    const arrival = rDirection(measurement.phi + theta);
    const firstHit = rHitRayLine(scene.B, departure, single.P, single.n);
    if (!firstHit || firstHit.t <= 0 || firstHit.t >= measurement.L) return { valid: false, reason: "path 2 misses wall A in forward order" };
    const reflected = rReflect(departure, single.n);
    const remaining = measurement.L - firstHit.t;
    const E = rAdd(firstHit.point, rMul(reflected, remaining));
    const line = { o: rSub(firstHit.point, rMul(arrival, remaining)), d: rAdd(reflected, arrival) };
    const crossing = rIntersectLines(single.line.o, single.line.d, line.o, line.d);
    let U;
    let distanceToBounce;
    let coincident = false;
    if (crossing) {
      U = crossing.point;
      distanceToBounce = crossing.s;
    } else {
      coincident = true;
      U = single.U;
      distanceToBounce = rDot(rSub(U, line.o), line.d) / Math.max(1e-9, rDot(line.d, line.d));
    }
    const valid = distanceToBounce > 0 && distanceToBounce < remaining && rDistance(U, single.U) < 8;
    if (!valid) return { valid: false, reason: "a line crossing exists, but a bounce lies behind its ray or exceeds the delay" };
    const P = rAdd(firstHit.point, rMul(reflected, distanceToBounce));
    const recoveredU = rSub(P, rMul(arrival, remaining - distanceToBounce));
    const normal = rUnit(rSub(recoveredU, E));
    return {
      valid: true,
      first: firstHit.point,
      P,
      U: recoveredU,
      E,
      n: normal,
      t: rPerp(normal),
      line,
      reflected,
      coincident,
    };
  }

  function rConstructTriple(scene, single, double, theta) {
    if (!double.valid) return { valid: false, reason: "the two-bounce prefix is already infeasible" };
    const measurement = scene.measures[2];
    const departure = rDirection(measurement.psi);
    const arrival = rDirection(measurement.phi + theta);
    const firstHit = rHitRayLine(scene.B, departure, single.P, single.n);
    if (!firstHit || firstHit.t <= 0) return { valid: false, reason: "path 3 cannot reach wall A in forward order" };
    const reflectedA = rReflect(departure, single.n);
    const secondHit = rHitRayLine(firstHit.point, reflectedA, double.P, double.n);
    if (!secondHit || secondHit.t <= 0) return { valid: false, reason: "path 3 reaches wall B only on a backward extension" };
    const spent = firstHit.t + secondHit.t;
    const remaining = measurement.L - spent;
    if (remaining <= 0) return { valid: false, reason: "the prefix exhausts the measured delay" };
    const reflectedB = rReflect(reflectedA, double.n);
    const E = rAdd(secondHit.point, rMul(reflectedB, remaining));
    const line = { o: rSub(secondHit.point, rMul(arrival, remaining)), d: rAdd(reflectedB, arrival) };
    const crossing = rIntersectLines(single.line.o, single.line.d, line.o, line.d);
    let U;
    let distanceToBounce;
    if (crossing) {
      U = crossing.point;
      distanceToBounce = crossing.s;
    } else {
      U = single.U;
      distanceToBounce = rDot(rSub(U, line.o), line.d) / Math.max(1e-9, rDot(line.d, line.d));
    }
    const valid =
      distanceToBounce > 0 &&
      distanceToBounce < remaining &&
      rDistance(U, single.U) < 10 &&
      rDistance(U, double.U) < 10;
    if (!valid) return { valid: false, reason: "the final intersection violates order or does not close at the same UE" };
    const P = rAdd(secondHit.point, rMul(reflectedB, distanceToBounce));
    const recoveredU = rSub(P, rMul(arrival, remaining - distanceToBounce));
    const normal = rUnit(rSub(recoveredU, E));
    return {
      valid: true,
      first: firstHit.point,
      second: secondHit.point,
      P,
      U: recoveredU,
      E,
      n: normal,
      t: rPerp(normal),
    };
  }

  function rSvgDefs() {
    const markers = [
      ["measure", RADIO.measurement],
      ["va", RADIO.va],
      ["teal", RADIO.incidence],
      ["green", RADIO.ue],
      ["red", RADIO.error],
      ["ink", RADIO.ink],
    ];
    return `<defs>${markers
      .map(
        ([id, color]) =>
          `<marker id="radio-arrow-${id}" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L8 4L0 8Z" fill="${color}"/></marker>`,
      )
      .join("")}</defs>`;
  }

  function rSvgLine(a, b, color, width = 1.5, dash = "", opacity = 1, marker = "") {
    return `<line x1="${rNumber(a.x)}" y1="${rNumber(a.y)}" x2="${rNumber(b.x)}" y2="${rNumber(b.y)}" stroke="${color}" stroke-width="${width}"${dash ? ` stroke-dasharray="${dash}"` : ""} opacity="${opacity}"${marker ? ` marker-end="url(#radio-arrow-${marker})"` : ""}/>`;
  }

  function rSvgPath(points, color, width = 2, dash = "", opacity = 1, marker = "") {
    const path = points.map((point, index) => `${index ? "L" : "M"}${rNumber(point.x)} ${rNumber(point.y)}`).join(" ");
    return `<path d="${path}" fill="none" stroke="${color}" stroke-width="${width}" stroke-linejoin="round" stroke-linecap="round"${dash ? ` stroke-dasharray="${dash}"` : ""} opacity="${opacity}"${marker ? ` marker-end="url(#radio-arrow-${marker})"` : ""}/>`;
  }

  function rSvgText(point, label, color = RADIO.soft, size = 10, anchor = "start", weight = 500) {
    return `<text x="${rNumber(point.x)}" y="${rNumber(point.y)}" fill="${color}" font-size="${size}" font-weight="${weight}" text-anchor="${anchor}">${label}</text>`;
  }

  function rSvgWall(point, tangent, label, color = RADIO.ink, opacity = 0.9, length = 170) {
    const a = rSub(point, rMul(tangent, length));
    const b = rAdd(point, rMul(tangent, length));
    return `${rSvgLine(a, b, color, 4, "", opacity)}${rSvgText(rAdd(b, { x: 7, y: 4 }), label, color, 10)}`;
  }

  function rSvgReferenceWall(wall, length = 260) {
    const a = rSub(wall.p, rMul(wall.t, length));
    const b = rAdd(wall.p, rMul(wall.t, length));
    return `${rSvgLine(a, b, RADIO.muted, 2, "7 6", 0.36)}${rSvgText(rAdd(b, { x: -5, y: -8 }), `${wall.label} reference`, RADIO.muted, 8, "end")}`;
  }

  function rSvgIncidence(point, label) {
    return `<circle cx="${rNumber(point.x)}" cy="${rNumber(point.y)}" r="5" fill="${RADIO.incidence}" stroke="#fff" stroke-width="1.5"/>${rSvgText(rAdd(point, { x: 9, y: -8 }), label, RADIO.incidenceDeep, 9)}`;
  }

  function rSvgUE(point, theta, label = "UE") {
    const heading = rDirection(theta);
    const arrowEnd = rAdd(point, rMul(heading, 34));
    return `<circle cx="${rNumber(point.x)}" cy="${rNumber(point.y)}" r="7" fill="${RADIO.ue}" stroke="#fff" stroke-width="2"/>${rSvgLine(point, arrowEnd, RADIO.va, 1.8, "", 1, "va")}${rSvgText(rAdd(point, { x: 11, y: 4 }), label, RADIO.ueDeep, 10, "start", 700)}`;
  }

  function rSvgBS(point) {
    return `<rect x="${rNumber(point.x - 7)}" y="${rNumber(point.y - 7)}" width="14" height="14" transform="rotate(45 ${rNumber(point.x)} ${rNumber(point.y)})" fill="${RADIO.scatter}" stroke="#fff" stroke-width="1.5"/>${rSvgText(rAdd(point, { x: -12, y: 22 }), "BS", RADIO.scatter, 10, "middle", 700)}`;
  }

  function rSvgEndpoint(point) {
    return `<rect x="${rNumber(point.x - 5)}" y="${rNumber(point.y - 5)}" width="10" height="10" transform="rotate(45 ${rNumber(point.x)} ${rNumber(point.y)})" fill="${RADIO.va}"/>${rSvgText(rAdd(point, { x: 8, y: -9 }), "E = BS + Ldψ", RADIO.vaDeep, 9)}`;
  }

  function rSvgMeasurementChips(bounces, thetaDegrees) {
    return `<g transform="translate(18 18)">
      <rect width="172" height="28" rx="3" fill="#fff" stroke="${RADIO.line}"/>
      ${rSvgText({ x: 12, y: 18 }, "DATA  L=cτ · ψ · φbody", RADIO.measurementDeep, 8, "start", 700)}
      <rect x="180" width="128" height="28" rx="3" fill="#fff" stroke="${RADIO.line}"/>
      ${rSvgText({ x: 192, y: 18 }, `${bounces} BOUNCE${bounces === 1 ? "" : "S"}`, RADIO.soft, 8, "start", 700)}
      <rect x="316" width="146" height="28" rx="3" fill="#fff" stroke="${RADIO.line}"/>
      ${rSvgText({ x: 328, y: 18 }, `θ HYPOTHESIS ${thetaDegrees}°`, RADIO.vaDeep, 8, "start", 700)}
    </g>`;
  }

  function rRenderRadioGeometry(scene, depth, theta, bounceFraction) {
    const single = rConstructSingle(scene, theta, bounceFraction);
    const double = depth >= 2 ? rConstructDouble(scene, single, theta) : null;
    const triple = depth >= 3 ? rConstructTriple(scene, single, double, theta) : null;
    const thetaDegrees = Math.round((theta * 180) / Math.PI);
    let markup = rSvgDefs() + rSvgMeasurementChips(depth, thetaDegrees);

    scene.walls.forEach((wall) => {
      markup += rSvgReferenceWall(wall, scene === corridorReference ? 245 : 235);
    });

    const samples = [];
    for (let fraction = 0.2; fraction <= 0.8; fraction += 0.075) {
      samples.push(rConstructSingle(scene, theta, fraction).U);
    }
    markup += rSvgPath(samples, RADIO.incidence, 1.5, "6 5", 0.55);
    samples.forEach((point) => {
      markup += `<circle cx="${rNumber(point.x)}" cy="${rNumber(point.y)}" r="2.2" fill="${RADIO.incidence}" opacity="0.42"/>`;
    });

    markup += rSvgLine(scene.B, single.E, RADIO.measurement, 1.6, "7 5", 0.78, "measure");
    markup += rSvgEndpoint(single.E);
    markup += rSvgPath([scene.B, single.P, single.U], RADIO.scatter, 2.2, "", 0.86);
    markup += rSvgIncidence(single.P, "P¹");
    markup += rSvgWall(single.P, single.t, scene === corridorReference ? "wall R hypothesis" : "wall A hypothesis");
    markup += rSvgUE(single.U, theta);
    markup += rSvgBS(scene.B);

    let valid = true;
    let reason = "all ordered segments are positive";

    if (depth >= 2) {
      if (double.valid) {
        markup += rSvgPath([scene.B, double.first, double.P, double.U], RADIO.va, 2, "4 4", 0.82);
        markup += rSvgIncidence(double.first, "P₁");
        markup += rSvgIncidence(double.P, "P₂");
        markup += rSvgWall(
          double.P,
          double.t,
          scene === corridorReference ? "wall L hypothesis" : "wall B hypothesis",
          RADIO.ink,
          0.82,
        );
        markup += rSvgText(
          rAdd(double.U, { x: 10, y: 19 }),
          double.coincident ? "candidate lines coincide" : "candidate lines cross here",
          RADIO.vaDeep,
          8,
        );
      } else {
        valid = false;
        reason = double.reason;
      }
    }

    if (depth >= 3) {
      if (triple.valid) {
        markup += rSvgPath([scene.B, triple.first, triple.second, triple.P, triple.U], RADIO.measurement, 2.2, "2 4", 0.88);
        markup += rSvgIncidence(triple.first, "Q₁");
        markup += rSvgIncidence(triple.second, "Q₂");
        markup += rSvgIncidence(triple.P, "Q₃");
        markup += rSvgWall(
          triple.P,
          triple.t,
          scene === corridorReference ? "wall R recovered again" : "wall C hypothesis",
          scene === corridorReference ? RADIO.vaDeep : RADIO.ink,
          0.78,
        );
      } else {
        valid = false;
        reason = triple.reason;
      }
    }

    if (!valid) {
      markup += `<rect x="25" y="438" width="710" height="56" rx="4" fill="#fff4f3" stroke="${RADIO.error}"/>`;
      markup += rSvgText({ x: 45, y: 460 }, "REJECTED HYPOTHESIS", RADIO.error, 10, "start", 700);
      markup += rSvgText({ x: 45, y: 479 }, reason, RADIO.error, 9);
      markup += rSvgLine({ x: 700, y: 451 }, { x: 720, y: 481 }, RADIO.error, 3);
      markup += rSvgLine({ x: 720, y: 451 }, { x: 700, y: 481 }, RADIO.error, 3);
    } else {
      markup += `<rect x="25" y="455" width="258" height="34" rx="3" fill="#f2faf7" stroke="#b9ddd5"/>`;
      markup += rSvgText({ x: 42, y: 476 }, `FEASIBLE SLICE · ${reason}`, RADIO.incidenceDeep, 8, "start", 700);
    }

    return { markup, valid, reason };
  }

  function rRenderObservability(nullSlide) {
    const shift = (nullSlide - 0.5) * 115;
    let markup = rSvgDefs();
    markup += `<rect x="18" y="18" width="350" height="484" rx="5" fill="#fff" stroke="${RADIO.line}"/>`;
    markup += `<rect x="392" y="18" width="350" height="484" rx="5" fill="#fff" stroke="${RADIO.line}"/>`;
    markup += rSvgText({ x: 38, y: 46 }, "CORNER · INDEPENDENT NORMALS", RADIO.vaDeep, 9, "start", 700);
    markup += rSvgText({ x: 412, y: 46 }, "CORRIDOR · PARALLEL NORMALS", RADIO.vaDeep, 9, "start", 700);

    const cornerA = [{ x: 62, y: 118 }, { x: 326, y: 118 }];
    const cornerB = [{ x: 326, y: 118 }, { x: 326, y: 406 }];
    markup += rSvgLine(cornerA[0], cornerA[1], RADIO.ink, 4);
    markup += rSvgLine(cornerB[0], cornerB[1], RADIO.ink, 4);
    const cu0 = { x: 126, y: 330 };
    const cu1 = { x: 226, y: 254 };
    markup += rSvgUE(cu0, -0.25, "u₀");
    markup += rSvgUE(cu1, -0.25, "u₁");
    markup += rSvgLine(cu0, cu1, RADIO.ue, 2.2, "", 1, "green");
    markup += rSvgText({ x: 172, y: 306 }, "global Δu", RADIO.ueDeep, 8, "middle", 700);
    markup += rSvgLine({ x: 82, y: 222 }, { x: 286, y: 222 }, RADIO.measurement, 1.6, "7 5", 0.75);
    markup += rSvgLine({ x: 222, y: 76 }, { x: 222, y: 424 }, RADIO.incidence, 1.6, "7 5", 0.75);
    markup += `<rect x="48" y="430" width="290" height="48" rx="3" fill="#f2faf7" stroke="#b9ddd5"/>`;
    markup += rSvgText({ x: 193, y: 450 }, "rank = 2 · σmin > 0", RADIO.incidenceDeep, 10, "middle", 700);
    markup += rSvgText({ x: 193, y: 468 }, "two offsets close at one solution", RADIO.soft, 8, "middle");

    const leftX = 438 + shift;
    const rightX = 690 + shift;
    markup += rSvgLine({ x: leftX, y: 92 }, { x: leftX, y: 420 }, RADIO.ink, 4);
    markup += rSvgLine({ x: rightX, y: 92 }, { x: rightX, y: 420 }, RADIO.ink, 4);
    const u0 = { x: 532 + shift, y: 336 };
    const u1 = { x: 584 + shift, y: 250 };
    [-34, 0, 34].forEach((ghostShift) => {
      markup += `<circle cx="${rNumber(u0.x + ghostShift)}" cy="${rNumber(u0.y)}" r="5" fill="none" stroke="${RADIO.incidence}" stroke-width="1" opacity="0.28"/>`;
      markup += rSvgLine(
        { x: leftX + ghostShift, y: 106 },
        { x: leftX + ghostShift, y: 406 },
        RADIO.incidence,
        1,
        "3 6",
        0.2,
      );
    });
    markup += rSvgUE(u0, -0.5, "u₀");
    markup += rSvgUE(u1, -0.5, "u₁");
    markup += rSvgLine(u0, u1, RADIO.ue, 2.2, "", 1, "green");
    markup += rSvgText({ x: 558 + shift, y: 296 }, "global Δu", RADIO.ueDeep, 8, "middle", 700);
    markup += rSvgLine({ x: 420, y: 210 }, { x: 720, y: 210 }, RADIO.measurement, 1.6, "7 5", 0.72);
    markup += rSvgLine({ x: 420, y: 360 }, { x: 720, y: 360 }, RADIO.incidence, 1.6, "7 5", 0.72);
    markup += rSvgText({ x: 570, y: 199 }, "both candidate rails point across the corridor", RADIO.measurementDeep, 8, "middle");
    markup += `<rect x="422" y="430" width="290" height="48" rx="3" fill="#fff4f3" stroke="#e6b4ae"/>`;
    markup += rSvgText({ x: 567, y: 450 }, "rank = 1 · σmin = 0", RADIO.error, 10, "middle", 700);
    markup += rSvgText({ x: 567, y: 468 }, "walls + trajectory slide together", RADIO.soft, 8, "middle");
    return { markup, valid: true, reason: "rank test exposes the null direction" };
  }

  function stopRadioSweep() {
    if (radioSweepFrame !== null) window.cancelAnimationFrame(radioSweepFrame);
    radioSweepFrame = null;
    radioSweep.setAttribute("aria-pressed", "false");
    radioSweep.innerHTML = '<span aria-hidden="true">▶</span> Sweep the surviving family';
  }

  function renderRadio() {
    const headingDegrees = Number(radioHeading.value);
    const theta = (headingDegrees * Math.PI) / 180;
    const bounceFraction = Number(radioBounce.value) / 100;
    const data = radioCases[radioCase];
    let result;

    if (radioCase === 5) result = rRenderObservability(bounceFraction);
    else {
      const scene = radioCase >= 3 ? corridorReference : cornerReference;
      const depth = radioCase === 0 ? 1 : radioCase === 1 || radioCase === 3 ? 2 : 3;
      result = rRenderRadioGeometry(scene, depth, theta, bounceFraction);
    }

    radioSvg.innerHTML = result.markup;
    radioSvg.setAttribute("aria-label", data.aria);
    $("#radio-heading-out").textContent = radioCase === 5 ? "global" : `${headingDegrees > 0 ? "+" : ""}${headingDegrees}°`;
    $("#radio-bounce-out").textContent = radioCase === 5 ? `${Math.round((bounceFraction - 0.5) * 100)} cm` : `${Math.round(bounceFraction * 100)}%`;
    $("#radio-slice-state").textContent = result.valid ? "feasible" : "rejected";
    $("#radio-slice-state").classList.toggle("is-rejected", !result.valid);
    $("#radio-family-state").textContent = data.family;
    $("#radio-rank-state").textContent = data.rank;
  }

  function updateRadioCase(nextCase, resetControls = true) {
    radioCase = (nextCase + radioCases.length) % radioCases.length;
    const data = radioCases[radioCase];
    stopRadioSweep();
    if (resetControls) {
      radioHeading.value = "0";
      radioBounce.value = "50";
    }
    $$(".radio-tab").forEach((button) => {
      const active = Number(button.dataset.radioCase) === radioCase;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-selected", String(active));
    });
    $("#radio-case-count").textContent = `§3.${radioCase + 1} / §3.6`;
    $("#radio-kicker").textContent = data.kicker;
    $("#radio-case-title").textContent = data.title;
    $("#radio-case-body").textContent = data.body;
    $("#radio-case-equation").textContent = data.equation;
    $("#radio-figure-state").textContent = data.figure;
    $("#radio-caption").innerHTML = `<strong>Read it.</strong> ${data.caption}`;
    radioHeading.disabled = radioCase === 5;
    $("#radio-heading-label").textContent = radioCase === 5 ? "Displacement coordinate frame" : "Candidate UE heading θ";
    $("#radio-heading-hint").textContent =
      radioCase === 5
        ? "This case explicitly assumes globally referenced displacement."
        : "One hypothesis slice—not a heading measurement.";
    $("#radio-bounce-label").textContent = radioCase === 5 ? "Corridor null-direction slide" : "Path-1 bounce P along AoD";
    $("#radio-bounce-hint").textContent =
      radioCase === 5
        ? "Move it: corridor walls and trajectory translate without changing the factors."
        : "Move P: finding the incidence point is finding the first wall.";
    renderRadio();
  }

  $$(".radio-tab").forEach((button) => {
    button.addEventListener("click", () => updateRadioCase(Number(button.dataset.radioCase)));
  });
  radioHeading.addEventListener("input", renderRadio);
  radioBounce.addEventListener("input", renderRadio);
  radioSweep.addEventListener("click", () => {
    if (radioSweepFrame !== null) {
      stopRadioSweep();
      return;
    }
    radioSweep.setAttribute("aria-pressed", "true");
    radioSweep.innerHTML = '<span aria-hidden="true">Ⅱ</span> Pause family sweep';
    const started = performance.now();
    const duration = radioCase === 5 ? 2600 : 5200;
    const animate = (now) => {
      const progress = ((now - started) % duration) / duration;
      if (radioCase === 5) radioBounce.value = String(Math.round(15 + progress * 70));
      else radioHeading.value = String(Math.round(-180 + progress * 360));
      renderRadio();
      radioSweepFrame = window.requestAnimationFrame(animate);
    };
    if (reducedMotion) {
      if (radioCase === 5) radioBounce.value = "72";
      else radioHeading.value = "90";
      renderRadio();
      stopRadioSweep();
    } else {
      radioSweepFrame = window.requestAnimationFrame(animate);
    }
  });

  updateRadioCase(0, false);

  // Keyboard navigation for the main lab when it has focus context.
  document.addEventListener("keydown", (event) => {
    if (event.target.matches("input, button, a")) return;
    if (event.key === "ArrowRight" && window.location.hash === "#loop") updateStep(currentStep + 1);
    if (event.key === "ArrowLeft" && window.location.hash === "#loop") updateStep(currentStep - 1);
  });

  selectCamera(selectedCamera);
  updateStep(0);
})();
