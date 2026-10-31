(() => {
  "use strict";

  const SVG_NS = "http://www.w3.org/2000/svg";
  const TAU = 2 * Math.PI;
  const MAX_PATHS = 6;
  const PATH_COLORS = ["#e56f3d", "#0d8f87", "#8f70d8", "#edb949", "#4e8fdb", "#d95850"];

  const DEFAULT_PATHS = [
    { id: 1, delayNs: 11.5, magnitude: 0.95, phaseDeg: 20, dopplerHz: 18, color: PATH_COLORS[0] },
    { id: 2, delayNs: 27.8, magnitude: 0.68, phaseDeg: 115, dopplerHz: -32, color: PATH_COLORS[1] },
    { id: 3, delayNs: 43.1, magnitude: 0.48, phaseDeg: -80, dopplerHz: 9, color: PATH_COLORS[2] },
  ];

  const state = {
    bandwidthMHz: 200,
    timeMs: 0,
    normalizeDelays: true,
    selectedPathId: 1,
    selectedTap: 3,
    playing: false,
    paths: clonePaths(DEFAULT_PATHS),
  };

  let nextPathId = 4;
  let currentModel = null;
  let animationFrame = null;
  let lastAnimationTime = null;
  let dragState = null;
  let resizeTimer = null;

  const dom = {};

  function clonePaths(paths) {
    return paths.map((path) => ({ ...path }));
  }

  function clamp(value, minimum, maximum) {
    return Math.min(maximum, Math.max(minimum, value));
  }

  function normalizedSinc(value) {
    if (Math.abs(value) < 1e-10) {
      return 1;
    }
    return Math.sin(Math.PI * value) / (Math.PI * value);
  }

  function complexMagnitude(value) {
    return Math.hypot(value.re, value.im);
  }

  function complexPhase(value) {
    return Math.atan2(value.im, value.re);
  }

  function phaseDegrees(value) {
    let degrees = complexPhase(value) * 180 / Math.PI;
    if (degrees <= -180) {
      degrees += 360;
    }
    return degrees;
  }

  function complexString(value, digits = 3) {
    const real = Math.abs(value.re) < 0.5 * 10 ** -digits ? 0 : value.re;
    const imaginary = Math.abs(value.im) < 0.5 * 10 ** -digits ? 0 : value.im;
    const sign = imaginary < 0 ? "−" : "+";
    return `${real.toFixed(digits)} ${sign} j${Math.abs(imaginary).toFixed(digits)}`;
  }

  function computeModel(inputState) {
    const bandwidthMHz = Number(inputState.bandwidthMHz);
    const tapSpacingNs = 1000 / bandwidthMHz;
    const minimumRawDelay = Math.min(...inputState.paths.map((path) => path.delayNs));
    const delayOffsetNs = inputState.normalizeDelays ? minimumRawDelay : 0;
    const timeSeconds = inputState.timeMs / 1000;

    const paths = inputState.paths.map((path) => {
      const delayNs = Math.max(0, path.delayNs - delayOffsetNs);
      const phaseRad = path.phaseDeg * Math.PI / 180 + TAU * path.dopplerHz * timeSeconds;
      return {
        ...path,
        effectiveDelayNs: delayNs,
        fractionalTap: delayNs / tapSpacingNs,
        phaseRad,
        coefficient: {
          re: path.magnitude * Math.cos(phaseRad),
          im: path.magnitude * Math.sin(phaseRad),
        },
      };
    });

    const maximumDelay = Math.max(...paths.map((path) => path.effectiveDelayNs));
    const lMax = Math.max(4, Math.ceil(maximumDelay / tapSpacingNs) + 2);
    const tapMaximumDelay = lMax * tapSpacingNs;
    const axisQuantum = tapMaximumDelay > 80 ? 20 : tapMaximumDelay > 35 ? 10 : 5;
    const axisMaxNs = Math.max(
      tapSpacingNs * 4,
      Math.ceil(tapMaximumDelay / axisQuantum) * axisQuantum,
    );

    const taps = [];
    for (let l = 0; l <= lMax; l += 1) {
      const contributions = paths.map((path) => {
        const kernelArgument = l - path.fractionalTap;
        const weight = normalizedSinc(kernelArgument);
        return {
          pathId: path.id,
          pathName: `P${path.id}`,
          color: path.color,
          kernelArgument,
          weight,
          re: path.coefficient.re * weight,
          im: path.coefficient.im * weight,
        };
      });
      const value = contributions.reduce(
        (sum, contribution) => ({
          re: sum.re + contribution.re,
          im: sum.im + contribution.im,
        }),
        { re: 0, im: 0 },
      );
      taps.push({
        index: l,
        delayNs: l * tapSpacingNs,
        contributions,
        re: value.re,
        im: value.im,
        magnitude: complexMagnitude(value),
        phaseRad: complexPhase(value),
      });
    }

    return {
      bandwidthMHz,
      tapSpacingNs,
      minimumRawDelay,
      delayOffsetNs,
      timeSeconds,
      paths,
      taps,
      lMax,
      axisMaxNs,
    };
  }

  function svgElement(tag, attributes = {}, text = null) {
    const node = document.createElementNS(SVG_NS, tag);
    Object.entries(attributes).forEach(([name, value]) => {
      node.setAttribute(name, String(value));
    });
    if (text !== null) {
      node.textContent = text;
    }
    return node;
  }

  function appendSvg(parent, tag, attributes = {}, text = null) {
    const node = svgElement(tag, attributes, text);
    parent.appendChild(node);
    return node;
  }

  function clearSvg(svg) {
    while (svg.firstChild) {
      svg.removeChild(svg.firstChild);
    }
  }

  function setRangeProgress(input) {
    const minimum = Number(input.min);
    const maximum = Number(input.max);
    const value = Number(input.value);
    const progress = maximum === minimum ? 0 : (value - minimum) / (maximum - minimum) * 100;
    input.style.setProperty("--range-progress", `${clamp(progress, 0, 100)}%`);
  }

  function formatField(field, value) {
    if (field === "delayNs") return `${Number(value).toFixed(1)} ns`;
    if (field === "magnitude") return Number(value).toFixed(2);
    if (field === "phaseDeg") return `${Number(value).toFixed(0)}°`;
    if (field === "dopplerHz") return `${Number(value).toFixed(0)} Hz`;
    return String(value);
  }

  function rangeMarkup(path, field, label, minimum, maximum, step) {
    const value = path[field];
    const progress = (value - minimum) / (maximum - minimum) * 100;
    return `
      <div class="mini-control">
        <label for="path-${path.id}-${field}">${label}</label>
        <input
          id="path-${path.id}-${field}"
          type="range"
          min="${minimum}"
          max="${maximum}"
          step="${step}"
          value="${value}"
          data-path-id="${path.id}"
          data-path-field="${field}"
          style="--range-progress: ${progress}%"
          aria-label="Path ${path.id} ${label}"
        >
        <output data-output-for="${field}">${formatField(field, value)}</output>
      </div>`;
  }

  function renderPathCards() {
    dom.pathList.innerHTML = state.paths.map((path) => `
      <article class="path-card ${path.id === state.selectedPathId ? "selected" : ""}" style="--path-color: ${path.color}" data-card-path-id="${path.id}">
        <div class="path-card-header">
          <button class="path-select-button" type="button" data-action="select-path" data-path-id="${path.id}" aria-pressed="${path.id === state.selectedPathId}">
            <span class="path-dot" aria-hidden="true"></span>
            <strong>P${path.id}</strong>
            <small>complex impulse</small>
          </button>
          <button
            class="remove-path"
            type="button"
            aria-label="Remove path ${path.id}"
            data-action="remove-path"
            data-path-id="${path.id}"
            ${state.paths.length === 1 ? "disabled" : ""}
          >×</button>
        </div>
        <div class="path-card-controls">
          ${rangeMarkup(path, "delayNs", "delay", 0, 80, 0.1)}
          ${rangeMarkup(path, "magnitude", "|aᵇ|", 0.05, 1.4, 0.01)}
          ${rangeMarkup(path, "phaseDeg", "phase", -180, 180, 1)}
          ${rangeMarkup(path, "dopplerHz", "Doppler", -100, 100, 1)}
        </div>
      </article>
    `).join("");

    dom.pathCountBadge.textContent = `${state.paths.length} ${state.paths.length === 1 ? "path" : "paths"}`;
    dom.addPathButton.disabled = state.paths.length >= MAX_PATHS;
    dom.addPathButton.title = state.paths.length >= MAX_PATHS ? `Maximum ${MAX_PATHS} paths` : "";
  }

  function renderPathSelector() {
    dom.pathSelector.innerHTML = state.paths.map((path) => `
      <button
        type="button"
        class="${path.id === state.selectedPathId ? "active" : ""}"
        style="--path-color: ${path.color}"
        data-kernel-path-id="${path.id}"
        aria-pressed="${path.id === state.selectedPathId}"
      >P${path.id}</button>
    `).join("");
  }

  function drawAxes(svg, options) {
    const {
      width, height, margin, xMax, yMax, xTicks = 6, yTicks = 4,
      xLabel, yLabel, yMin = 0, zeroY = null,
    } = options;
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    const xScale = (value) => margin.left + value / xMax * innerWidth;
    const yScale = (value) => margin.top + (yMax - value) / (yMax - yMin) * innerHeight;

    for (let i = 0; i <= yTicks; i += 1) {
      const value = yMin + (yMax - yMin) * i / yTicks;
      const y = yScale(value);
      appendSvg(svg, "line", {
        x1: margin.left,
        y1: y,
        x2: width - margin.right,
        y2: y,
        class: value === 0 && zeroY !== null ? "zero-line" : "grid-line",
      });
      appendSvg(svg, "text", {
        x: margin.left - 10,
        y: y + 3,
        "text-anchor": "end",
        class: "tick-label",
      }, Math.abs(value) < 1e-9 ? "0" : value.toFixed(yMax <= 2 ? 1 : 2));
    }

    for (let i = 0; i <= xTicks; i += 1) {
      const value = xMax * i / xTicks;
      const x = xScale(value);
      appendSvg(svg, "line", {
        x1: x,
        y1: margin.top,
        x2: x,
        y2: height - margin.bottom,
        class: "grid-line",
      });
      appendSvg(svg, "text", {
        x,
        y: height - margin.bottom + 18,
        "text-anchor": "middle",
        class: "tick-label",
      }, value.toFixed(value < 10 && xMax < 25 ? 1 : 0));
    }

    appendSvg(svg, "line", {
      x1: margin.left,
      y1: height - margin.bottom,
      x2: width - margin.right,
      y2: height - margin.bottom,
      class: "axis-line",
    });

    appendSvg(svg, "text", {
      x: margin.left + innerWidth / 2,
      y: height - 5,
      "text-anchor": "middle",
      class: "axis-label",
    }, xLabel);

    appendSvg(svg, "text", {
      x: 14,
      y: margin.top + innerHeight / 2,
      transform: `rotate(-90 14 ${margin.top + innerHeight / 2})`,
      "text-anchor": "middle",
      class: "axis-label",
    }, yLabel);

    return { xScale, yScale, innerWidth, innerHeight };
  }

  function attachTooltip(node, contentBuilder) {
    node.addEventListener("pointerenter", (event) => showTooltip(event, contentBuilder()));
    node.addEventListener("pointermove", (event) => positionTooltip(event));
    node.addEventListener("pointerleave", hideTooltip);
  }

  function showTooltip(event, content) {
    dom.tooltip.innerHTML = content;
    dom.tooltip.hidden = false;
    positionTooltip(event);
  }

  function positionTooltip(event) {
    if (dom.tooltip.hidden) return;
    const padding = 14;
    const rect = dom.tooltip.getBoundingClientRect();
    let x = event.clientX + 14;
    let y = event.clientY + 14;
    if (x + rect.width > window.innerWidth - padding) x = event.clientX - rect.width - 14;
    if (y + rect.height > window.innerHeight - padding) y = event.clientY - rect.height - 14;
    dom.tooltip.style.left = `${Math.max(padding, x)}px`;
    dom.tooltip.style.top = `${Math.max(padding, y)}px`;
  }

  function hideTooltip() {
    dom.tooltip.hidden = true;
  }

  function drawCirPlot(model) {
    const svg = dom.cirPlot;
    clearSvg(svg);
    const width = window.innerWidth <= 620 ? 600 : 900;
    const height = 300;
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    const margin = { top: 30, right: 28, bottom: 52, left: 62 };
    const maximumMagnitude = Math.max(...model.paths.map((path) => path.magnitude), 0.1);
    const yMax = Math.max(1, maximumMagnitude * 1.25);
    const { xScale, yScale } = drawAxes(svg, {
      width,
      height,
      margin,
      xMax: model.axisMaxNs,
      yMax,
      xTicks: width < 700 ? 4 : 6,
      xLabel: state.normalizeDelays ? "normalized path delay τᵢ (ns)" : "absolute path delay τᵢ (ns)",
      yLabel: "|aᵇᵢ(tₙ)|",
    });
    const baseline = yScale(0);

    model.paths.forEach((path) => {
      const x = xScale(path.effectiveDelayNs);
      const y = yScale(path.magnitude);
      const group = appendSvg(svg, "g", {
        "data-draggable-path": path.id,
        tabindex: "0",
        role: "button",
        "aria-label": `Path ${path.id}, delay ${path.effectiveDelayNs.toFixed(2)} nanoseconds, magnitude ${path.magnitude.toFixed(3)}`,
      });

      appendSvg(group, "line", {
        x1: x,
        y1: baseline,
        x2: x,
        y2: y,
        stroke: path.color,
        class: "cir-stem",
      });
      appendSvg(group, "circle", {
        cx: x,
        cy: y,
        r: 30,
        fill: "transparent",
        class: "cir-hit-target",
      });
      const point = appendSvg(group, "circle", {
        cx: x,
        cy: y,
        r: path.id === state.selectedPathId ? 12 : 10,
        fill: path.color,
        class: `cir-point ${path.id === state.selectedPathId ? "selected" : ""}`,
      });
      const arrowLength = path.id === state.selectedPathId ? 9 : 7;
      appendSvg(group, "line", {
        x1: x,
        y1: y,
        x2: x + arrowLength * Math.cos(path.phaseRad),
        y2: y - arrowLength * Math.sin(path.phaseRad),
        class: "phase-arrow",
      });
      appendSvg(group, "text", {
        x: x + 15,
        y: y - 13,
        fill: path.color,
        class: "path-label",
      }, `P${path.id}`);

      group.addEventListener("click", () => selectPath(path.id));
      group.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          selectPath(path.id);
        }
      });
      attachTooltip(point, () => `
        <strong>P${path.id} · CIR impulse</strong><br>
        effective τ = ${path.effectiveDelayNs.toFixed(2)} ns<br>
        raw τ = ${path.delayNs.toFixed(2)} ns<br>
        |aᵇ| = ${path.magnitude.toFixed(3)}<br>
        ∠aᵇᵢ(tₙ) = ${(path.phaseRad * 180 / Math.PI).toFixed(1)}°
      `);
    });

    appendSvg(svg, "text", {
      x: width - margin.right,
      y: margin.top + 3,
      "text-anchor": "end",
      class: "annotation",
    }, `${model.paths.length} separate ${model.paths.length === 1 ? "impulse" : "impulses"}`);
  }

  function drawKernelPlot(model) {
    const svg = dom.kernelPlot;
    clearSvg(svg);
    const selectedPath = model.paths.find((path) => path.id === state.selectedPathId) || model.paths[0];
    const width = window.innerWidth <= 620 ? 600 : 900;
    const height = 250;
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    const margin = { top: 20, right: 28, bottom: 49, left: 62 };
    const { xScale, yScale } = drawAxes(svg, {
      width,
      height,
      margin,
      xMax: model.axisMaxNs,
      yMin: -1.1,
      yMax: 1.1,
      yTicks: 4,
      xTicks: width < 700 ? 4 : 6,
      xLabel: "tap delay ℓ/W (ns)",
      yLabel: "sinc weight",
      zeroY: 0,
    });
    const baseline = yScale(0);

    const curvePoints = [];
    const sampleCount = 400;
    for (let sample = 0; sample <= sampleCount; sample += 1) {
      const delayNs = model.axisMaxNs * sample / sampleCount;
      const tapCoordinate = delayNs / model.tapSpacingNs;
      const weight = normalizedSinc(tapCoordinate - selectedPath.fractionalTap);
      curvePoints.push({ x: xScale(delayNs), y: yScale(weight) });
    }
    const curve = curvePoints.map((point, index) => `${index === 0 ? "M" : "L"}${point.x.toFixed(2)},${point.y.toFixed(2)}`).join(" ");
    const area = `${curve} L${curvePoints[curvePoints.length - 1].x.toFixed(2)},${baseline.toFixed(2)} L${curvePoints[0].x.toFixed(2)},${baseline.toFixed(2)} Z`;
    appendSvg(svg, "path", {
      d: area,
      fill: selectedPath.color,
      class: "kernel-area-positive",
    });
    appendSvg(svg, "path", {
      d: curve,
      stroke: selectedPath.color,
      class: "kernel-line",
    });

    appendSvg(svg, "line", {
      x1: xScale(selectedPath.effectiveDelayNs),
      y1: margin.top,
      x2: xScale(selectedPath.effectiveDelayNs),
      y2: height - margin.bottom,
      stroke: selectedPath.color,
      "stroke-width": 1.5,
      "stroke-dasharray": "4 5",
      opacity: 0.65,
    });

    model.taps.forEach((tap) => {
      if (tap.delayNs > model.axisMaxNs + 1e-6) return;
      const contribution = tap.contributions.find((item) => item.pathId === selectedPath.id);
      const circle = appendSvg(svg, "circle", {
        cx: xScale(tap.delayNs),
        cy: yScale(contribution.weight),
        r: tap.index === state.selectedTap ? 6 : model.taps.length > 45 ? 2.5 : 4,
        fill: selectedPath.color,
        class: `kernel-sample ${tap.index === state.selectedTap ? "selected" : ""}`,
        "data-tap-index": tap.index,
        role: "button",
        tabindex: "0",
      });
      circle.addEventListener("click", () => selectTap(tap.index));
      circle.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          selectTap(tap.index);
        }
      });
      attachTooltip(circle, () => `
        <strong>P${selectedPath.id} → tap ${tap.index}</strong><br>
        sinc(${contribution.kernelArgument.toFixed(3)}) = ${contribution.weight.toFixed(4)}<br>
        tap delay = ${tap.delayNs.toFixed(2)} ns
      `);
    });

    appendSvg(svg, "text", {
      x: xScale(selectedPath.effectiveDelayNs) + 7,
      y: margin.top + 12,
      fill: selectedPath.color,
      class: "annotation",
    }, `τ${selectedPath.id} = ${selectedPath.effectiveDelayNs.toFixed(2)} ns`);
  }

  function drawTapsPlot(model) {
    const svg = dom.tapsPlot;
    clearSvg(svg);
    const width = window.innerWidth <= 620 ? 600 : 900;
    const height = 310;
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    const margin = { top: 28, right: 28, bottom: 58, left: 62 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    const maximumMagnitude = Math.max(...model.taps.map((tap) => tap.magnitude), 1e-6);
    const yMax = maximumMagnitude * 1.18;
    const yScale = (value) => margin.top + (yMax - value) / yMax * innerHeight;
    const baseline = height - margin.bottom;
    const slotWidth = innerWidth / model.taps.length;
    const xCenter = (index) => margin.left + slotWidth * (index + 0.5);

    for (let i = 0; i <= 4; i += 1) {
      const value = yMax * i / 4;
      const y = yScale(value);
      appendSvg(svg, "line", {
        x1: margin.left,
        y1: y,
        x2: width - margin.right,
        y2: y,
        class: "grid-line",
      });
      appendSvg(svg, "text", {
        x: margin.left - 10,
        y: y + 3,
        "text-anchor": "end",
        class: "tick-label",
      }, value.toFixed(2));
    }
    appendSvg(svg, "line", {
      x1: margin.left,
      y1: baseline,
      x2: width - margin.right,
      y2: baseline,
      class: "axis-line",
    });

    const tickStep = Math.max(1, Math.ceil(model.taps.length / 11));
    model.taps.forEach((tap, index) => {
      const x = xCenter(index);
      const y = yScale(tap.magnitude);
      let interactiveNode;
      if (model.taps.length <= 38) {
        const barWidth = Math.max(4, slotWidth * 0.62);
        interactiveNode = appendSvg(svg, "rect", {
          x: x - barWidth / 2,
          y,
          width: barWidth,
          height: Math.max(1, baseline - y),
          rx: Math.min(4, barWidth / 3),
          class: `tap-bar ${tap.index === state.selectedTap ? "selected" : ""}`,
          tabindex: "0",
          role: "button",
          "data-tap-index": tap.index,
        });
      } else {
        const group = appendSvg(svg, "g", {
          tabindex: "0",
          role: "button",
          "data-tap-index": tap.index,
        });
        appendSvg(group, "line", {
          x1: x,
          y1: baseline,
          x2: x,
          y2: y,
          class: `tap-stem ${tap.index === state.selectedTap ? "selected" : ""}`,
        });
        interactiveNode = appendSvg(group, "circle", {
          cx: x,
          cy: y,
          r: tap.index === state.selectedTap ? 4.5 : 2.5,
          class: `tap-head ${tap.index === state.selectedTap ? "selected" : ""}`,
        });
        interactiveNode = group;
      }

      interactiveNode.addEventListener("click", () => selectTap(tap.index));
      interactiveNode.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          selectTap(tap.index);
        }
      });
      attachTooltip(interactiveNode, () => `
        <strong>Tap ℓ = ${tap.index}</strong><br>
        delay = ${tap.delayNs.toFixed(2)} ns<br>
        |h| = ${tap.magnitude.toFixed(4)}<br>
        ∠h = ${phaseDegrees(tap).toFixed(1)}°<br>
        ${complexString(tap, 4)}
      `);

      if (index % tickStep === 0 || index === model.taps.length - 1) {
        appendSvg(svg, "line", {
          x1: x,
          y1: baseline,
          x2: x,
          y2: baseline + 5,
          class: "axis-line",
        });
        appendSvg(svg, "text", {
          x,
          y: baseline + 18,
          "text-anchor": "middle",
          class: "tick-label",
        }, String(tap.index));
      }
    });

    appendSvg(svg, "text", {
      x: margin.left + innerWidth / 2,
      y: height - 7,
      "text-anchor": "middle",
      class: "axis-label",
    }, "tap index ℓ  ·  delay = ℓ/W");
    appendSvg(svg, "text", {
      x: 14,
      y: margin.top + innerHeight / 2,
      transform: `rotate(-90 14 ${margin.top + innerHeight / 2})`,
      "text-anchor": "middle",
      class: "axis-label",
    }, "|hₙ,ℓ|");
  }

  function renderTapInspection(model) {
    const tap = model.taps.find((item) => item.index === state.selectedTap) || model.taps[0];
    state.selectedTap = tap.index;
    dom.selectedTapTitle.textContent = `Tap ℓ = ${tap.index}`;
    dom.selectedTapDelay.textContent = `delay ${tap.delayNs.toFixed(2)} ns`;
    dom.selectedMagnitude.textContent = tap.magnitude.toFixed(4);
    dom.selectedPhase.textContent = `${phaseDegrees(tap).toFixed(1)}°`;
    dom.selectedComplex.textContent = complexString(tap, 4);

    const maximumContribution = Math.max(
      ...tap.contributions.map((contribution) => Math.hypot(contribution.re, contribution.im)),
      1e-9,
    );
    dom.contributionList.innerHTML = tap.contributions.map((contribution) => {
      const magnitude = Math.hypot(contribution.re, contribution.im);
      const barWidth = magnitude / maximumContribution * 100;
      return `
        <div class="contribution-row">
          <span class="contribution-path" style="--path-color: ${contribution.color}"><i></i>${contribution.pathName}</span>
          <span class="weight-value ${contribution.weight < 0 ? "negative" : ""}">${contribution.weight >= 0 ? "+" : "−"}${Math.abs(contribution.weight).toFixed(4)}</span>
          <span class="contribution-vector" style="--path-color: ${contribution.color}; --bar-width: ${barWidth}%">
            <span class="complex-value">${complexString(contribution, 4)}</span>
          </span>
        </div>`;
    }).join("");
  }

  function updateGlobalControls(model) {
    dom.bandwidth.value = String(state.bandwidthMHz);
    dom.bandwidthValue.textContent = state.bandwidthMHz >= 1000
      ? `${(state.bandwidthMHz / 1000).toFixed(state.bandwidthMHz % 1000 === 0 ? 0 : 2)} GHz`
      : `${state.bandwidthMHz.toFixed(0)} MHz`;
    dom.time.value = String(state.timeMs);
    dom.timeValue.textContent = `${state.timeMs.toFixed(1)} ms`;
    dom.normalizeDelays.checked = state.normalizeDelays;
    setRangeProgress(dom.bandwidth);
    setRangeProgress(dom.time);

    document.querySelectorAll("[data-bandwidth]").forEach((button) => {
      button.classList.toggle("active", Number(button.dataset.bandwidth) === state.bandwidthMHz);
    });

    dom.earliestDelay.textContent = state.normalizeDelays
      ? `${model.minimumRawDelay.toFixed(2)} ns → 0 ns`
      : `${model.minimumRawDelay.toFixed(2)} ns`;
    dom.tapSpacing.textContent = `${model.tapSpacingNs.toFixed(model.tapSpacingNs < 2 ? 2 : 1)} ns`;
    dom.tapCount.textContent = `${model.taps.length} (ℓ = 0…${model.lMax})`;

    const selectedPath = model.paths.find((path) => path.id === state.selectedPathId) || model.paths[0];
    dom.kernelPathName.textContent = `P${selectedPath.id}`;
    dom.kernelPathName.parentElement.style.setProperty("--selected-path-color", selectedPath.color);
    dom.fractionalTap.textContent = `center = tap ${selectedPath.fractionalTap.toFixed(2)}`;
  }

  function renderVisuals() {
    if (!state.paths.some((path) => path.id === state.selectedPathId)) {
      state.selectedPathId = state.paths[0].id;
    }
    currentModel = computeModel(state);
    state.selectedTap = clamp(state.selectedTap, 0, currentModel.lMax);
    updateGlobalControls(currentModel);
    renderPathSelector();
    drawCirPlot(currentModel);
    drawKernelPlot(currentModel);
    drawTapsPlot(currentModel);
    renderTapInspection(currentModel);
  }

  function selectPath(pathId) {
    state.selectedPathId = Number(pathId);
    renderPathCards();
    renderVisuals();
  }

  function selectTap(index) {
    state.selectedTap = Number(index);
    renderVisuals();
  }

  function removePath(pathId) {
    if (state.paths.length === 1) return;
    const index = state.paths.findIndex((path) => path.id === Number(pathId));
    if (index < 0) return;
    const wasSelected = state.paths[index].id === state.selectedPathId;
    state.paths.splice(index, 1);
    if (wasSelected) {
      state.selectedPathId = state.paths[Math.min(index, state.paths.length - 1)].id;
    }
    renderPathCards();
    renderVisuals();
  }

  function addPath() {
    if (state.paths.length >= MAX_PATHS) return;
    const usedColors = new Set(state.paths.map((path) => path.color));
    const color = PATH_COLORS.find((candidate) => !usedColors.has(candidate)) || PATH_COLORS[state.paths.length % PATH_COLORS.length];
    const latestDelay = Math.max(...state.paths.map((path) => path.delayNs));
    const path = {
      id: nextPathId,
      delayNs: clamp(latestDelay + 8, 0, 80),
      magnitude: 0.35,
      phaseDeg: 45,
      dopplerHz: 12,
      color,
    };
    nextPathId += 1;
    state.paths.push(path);
    state.selectedPathId = path.id;
    renderPathCards();
    renderVisuals();
    const card = dom.pathList.querySelector(`[data-card-path-id="${path.id}"]`);
    if (card) card.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function resetState() {
    setPlaying(false);
    state.bandwidthMHz = 200;
    state.timeMs = 0;
    state.normalizeDelays = true;
    state.selectedPathId = 1;
    state.selectedTap = 3;
    state.paths = clonePaths(DEFAULT_PATHS);
    nextPathId = 4;
    renderPathCards();
    renderVisuals();
  }

  function setPlaying(playing) {
    state.playing = playing;
    dom.playButton.classList.toggle("playing", playing);
    dom.playButton.setAttribute("aria-pressed", String(playing));
    dom.playButton.setAttribute("aria-label", playing ? "Pause Doppler evolution" : "Play Doppler evolution");
    lastAnimationTime = null;
    if (playing && animationFrame === null) {
      animationFrame = requestAnimationFrame(animate);
    }
  }

  function animate(timestamp) {
    if (!state.playing) {
      animationFrame = null;
      return;
    }
    if (lastAnimationTime === null) lastAnimationTime = timestamp;
    const elapsed = Math.min(50, timestamp - lastAnimationTime);
    lastAnimationTime = timestamp;
    state.timeMs = (state.timeMs + elapsed * 0.008) % 50;
    renderVisuals();
    animationFrame = requestAnimationFrame(animate);
  }

  function onPathListInput(event) {
    const input = event.target.closest("[data-path-field]");
    if (!input) return;
    const path = state.paths.find((item) => item.id === Number(input.dataset.pathId));
    if (!path) return;
    const field = input.dataset.pathField;
    path[field] = Number(input.value);
    setRangeProgress(input);
    const output = input.parentElement.querySelector(`[data-output-for="${field}"]`);
    if (output) output.textContent = formatField(field, input.value);
    state.selectedPathId = path.id;
    dom.pathList.querySelectorAll(".path-card").forEach((card) => {
      const selected = Number(card.dataset.cardPathId) === path.id;
      card.classList.toggle("selected", selected);
      const header = card.querySelector("[data-action='select-path']");
      if (header) header.setAttribute("aria-pressed", String(selected));
    });
    renderVisuals();
  }

  function onPathListClick(event) {
    const action = event.target.closest("[data-action]");
    if (!action) return;
    const pathId = Number(action.dataset.pathId);
    if (action.dataset.action === "remove-path") {
      event.preventDefault();
      event.stopPropagation();
      if (!action.disabled) removePath(pathId);
      return;
    }
    if (action.dataset.action === "select-path") selectPath(pathId);
  }

  function pointerToSvgX(event, svg) {
    const rect = svg.getBoundingClientRect();
    const viewBox = svg.viewBox.baseVal;
    return (event.clientX - rect.left) / rect.width * viewBox.width;
  }

  function selectNearestTapFromPointer(event, svg) {
    if (!currentModel || !currentModel.taps.length) return;
    if (event.pointerType === "mouse" && event.button !== 0) return;
    const marginLeft = 62;
    const marginRight = 28;
    const plotWidth = svg.viewBox.baseVal.width - marginLeft - marginRight;
    const x = pointerToSvgX(event, svg);
    const delayNs = clamp(
      (x - marginLeft) / plotWidth * currentModel.axisMaxNs,
      0,
      currentModel.axisMaxNs,
    );
    const nearestTap = currentModel.taps.reduce((nearest, tap) => (
      Math.abs(tap.delayNs - delayNs) < Math.abs(nearest.delayNs - delayNs)
        ? tap
        : nearest
    ));
    selectTap(nearestTap.index);
  }

  function startCirDrag(event) {
    const target = event.target.closest("[data-draggable-path]");
    if (!target || event.button !== 0) return;
    dragState = { pathId: Number(target.dataset.draggablePath) };
    state.selectedPathId = dragState.pathId;
    target.setPointerCapture?.(event.pointerId);
    event.preventDefault();
  }

  function moveCirDrag(event) {
    if (!dragState || !currentModel) return;
    const marginLeft = 62;
    const marginRight = 28;
    const plotWidth = dom.cirPlot.viewBox.baseVal.width - marginLeft - marginRight;
    const x = pointerToSvgX(event, dom.cirPlot);
    const effectiveDelay = clamp((x - marginLeft) / plotWidth * currentModel.axisMaxNs, 0, 80);
    const rawDelay = effectiveDelay + (state.normalizeDelays ? currentModel.minimumRawDelay : 0);
    const path = state.paths.find((item) => item.id === dragState.pathId);
    if (!path) return;
    path.delayNs = clamp(rawDelay, 0, 80);
    const input = dom.pathList.querySelector(`[data-path-id="${path.id}"][data-path-field="delayNs"]`);
    if (input) {
      input.value = String(path.delayNs);
      setRangeProgress(input);
      const output = input.parentElement.querySelector("[data-output-for='delayNs']");
      if (output) output.textContent = formatField("delayNs", path.delayNs);
    }
    renderVisuals();
  }

  function endCirDrag() {
    dragState = null;
  }

  async function copyCode() {
    const text = dom.codeSample.textContent;
    try {
      await navigator.clipboard.writeText(text);
    } catch (error) {
      const textarea = document.createElement("textarea");
      textarea.value = text;
      textarea.style.position = "fixed";
      textarea.style.opacity = "0";
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand("copy");
      textarea.remove();
    }
    dom.copyCodeButton.textContent = "Copied";
    window.setTimeout(() => { dom.copyCodeButton.textContent = "Copy"; }, 1400);
  }

  function runSelfTests() {
    const assertClose = (actual, expected, tolerance, label) => {
      if (Math.abs(actual - expected) > tolerance) {
        throw new Error(`${label}: expected ${expected}, got ${actual}`);
      }
    };
    assertClose(normalizedSinc(0), 1, 1e-12, "sinc at zero");
    assertClose(normalizedSinc(1), 0, 1e-12, "sinc at integer one");

    const aligned = computeModel({
      bandwidthMHz: 100,
      timeMs: 0,
      normalizeDelays: false,
      paths: [{ id: 1, delayNs: 10, magnitude: 2, phaseDeg: 0, dopplerHz: 0, color: "#000" }],
    });
    assertClose(aligned.taps[1].re, 2, 1e-10, "grid-aligned path amplitude");
    assertClose(aligned.taps[0].magnitude, 0, 1e-10, "grid-aligned adjacent tap");

    const cancellation = computeModel({
      bandwidthMHz: 100,
      timeMs: 0,
      normalizeDelays: true,
      paths: [
        { id: 1, delayNs: 4, magnitude: 1, phaseDeg: 0, dopplerHz: 0, color: "#000" },
        { id: 2, delayNs: 4, magnitude: 1, phaseDeg: 180, dopplerHz: 0, color: "#000" },
      ],
    });
    assertClose(cancellation.taps[0].magnitude, 0, 1e-10, "complex cancellation");
    return true;
  }

  function cacheDom() {
    [
      "modelStatus", "bandwidth", "bandwidthValue", "time", "timeValue",
      "normalizeDelays", "playButton", "resetButton", "pathList",
      "pathCountBadge", "addPathButton", "pathSelector", "cirPlot",
      "kernelPlot", "tapsPlot", "earliestDelay", "tapSpacing", "tapCount",
      "kernelPathName", "fractionalTap", "selectedTapTitle", "selectedTapDelay",
      "selectedMagnitude", "selectedPhase", "selectedComplex", "contributionList",
      "copyCodeButton", "codeSample", "tooltip",
    ].forEach((id) => { dom[id] = document.getElementById(id); });
  }

  function bindEvents() {
    dom.bandwidth.addEventListener("input", () => {
      state.bandwidthMHz = Number(dom.bandwidth.value);
      renderVisuals();
    });
    dom.time.addEventListener("input", () => {
      setPlaying(false);
      state.timeMs = Number(dom.time.value);
      renderVisuals();
    });
    dom.normalizeDelays.addEventListener("change", () => {
      state.normalizeDelays = dom.normalizeDelays.checked;
      renderVisuals();
    });
    dom.playButton.addEventListener("click", () => setPlaying(!state.playing));
    dom.resetButton.addEventListener("click", resetState);
    dom.addPathButton.addEventListener("click", addPath);
    dom.pathList.addEventListener("input", onPathListInput);
    dom.pathList.addEventListener("click", onPathListClick);
    dom.pathSelector.addEventListener("click", (event) => {
      const button = event.target.closest("[data-kernel-path-id]");
      if (button) selectPath(Number(button.dataset.kernelPathId));
    });
    document.querySelectorAll("[data-bandwidth]").forEach((button) => {
      button.addEventListener("click", () => {
        state.bandwidthMHz = Number(button.dataset.bandwidth);
        renderVisuals();
      });
    });
    dom.cirPlot.addEventListener("pointerdown", startCirDrag);
    dom.kernelPlot.addEventListener("pointerdown", (event) => {
      selectNearestTapFromPointer(event, dom.kernelPlot);
    });
    dom.tapsPlot.addEventListener("pointerdown", (event) => {
      selectNearestTapFromPointer(event, dom.tapsPlot);
    });
    window.addEventListener("pointermove", moveCirDrag);
    window.addEventListener("pointerup", endCirDrag);
    window.addEventListener("pointercancel", endCirDrag);
    window.addEventListener("resize", () => {
      window.clearTimeout(resizeTimer);
      resizeTimer = window.setTimeout(renderVisuals, 100);
    });
    dom.copyCodeButton.addEventListener("click", copyCode);
  }

  function initialize() {
    cacheDom();
    bindEvents();
    renderPathCards();
    renderVisuals();
    try {
      runSelfTests();
      dom.modelStatus.classList.add("passed");
      dom.modelStatus.lastChild.textContent = " Math checked";
      document.documentElement.dataset.selfTest = "passed";
    } catch (error) {
      console.error(error);
      dom.modelStatus.classList.add("failed");
      dom.modelStatus.lastChild.textContent = " Model error";
      document.documentElement.dataset.selfTest = "failed";
    }
  }

  window.CirTapsExplorer = {
    normalizedSinc,
    computeModel,
    getState: () => ({ ...state, paths: clonePaths(state.paths) }),
  };

  document.addEventListener("DOMContentLoaded", initialize);
})();
