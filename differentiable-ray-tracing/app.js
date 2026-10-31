(function () {
  "use strict";

  var reducedMotion = window.matchMedia &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function number(value, digits) {
    return Number(value).toFixed(digits);
  }

  function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
  }

  function setupRendererFamilyLab() {
    var picker = document.querySelectorAll('input[name="rendererFamily"]');
    var scene = document.getElementById("familyScene");
    var objectSlider = document.getElementById("familyObjectSlider");
    var stageSlider = document.getElementById("familyStageSlider");
    var sensitivityButton = document.getElementById("familySensitivityButton");
    if (!picker.length || !scene || !objectSlider || !stageSlider || !sensitivityButton) return;

    var modes = {
      raster: {
        label: "triangle pipeline",
        kicker: "surface representation",
        title: "Rasterizer",
        summary: "Triangles are projected into screen space. Coverage and depth select a surface sample before interpolation and shading.",
        caption: "Triangles are projected into pixel space; the closest covering fragment supplies attributes for shading.",
        description: "A rasterizer projects triangles, tests pixel coverage and depth, interpolates attributes, and shades the visible fragment.",
        steps: ["Project triangles", "Test coverage and depth", "Interpolate attributes", "Shade the visible fragment"],
        formula: "C(p) = shade(interpolated attributes of the nearest covering triangle)",
        representation: "triangles + vertex attributes",
        gradient: "projection, interpolation, textures, and shading; geometry visibility needs coverage or antialias treatment",
        caveat: "coverage and the winning depth can switch discretely"
      },
      volume: {
        label: "ray integration pipeline",
        kicker: "continuous field representation",
        title: "Volume renderer",
        summary: "Camera rays query density and radiance. Opacity and transmittance weight the samples from front to back.",
        caption: "Density samples become opacity; accumulated transmittance discounts samples hidden behind earlier density.",
        description: "A volume renderer marches camera rays, queries density and radiance, computes transmittance, and composites samples front to back.",
        steps: ["Construct camera rays", "Query density and radiance", "Compute opacity + transmittance", "Composite front to back"],
        formula: "αᵢ = 1 − exp(−σᵢΔᵢ),  Tᵢ = ∏_{j<i}(1 − αⱼ),  C = Σᵢ Tᵢ αᵢ cᵢ + T_end C_bg",
        representation: "a sampled or continuous density/radiance field",
        gradient: "field values or network parameters, samples, and camera",
        caveat: "resampling, acceleration structures, and sharp density changes need care"
      },
      splat: {
        label: "project-and-composite pipeline",
        kicker: "Gaussian primitive representation",
        title: "Gaussian splatter",
        summary: "Each 3-D Gaussian projects to a 2-D elliptical footprint. Overlapping footprints contribute color and opacity in visibility order.",
        caption: "Projected Gaussian footprints are ordered and alpha-composited; their centers, covariances, opacity, and color can carry gradients.",
        description: "A Gaussian splatter projects 3-D Gaussians to elliptical footprints, orders contributors, and alpha-composites them into pixels.",
        steps: ["Project 3-D Gaussians", "Form 2-D ellipses", "Order contributors", "Alpha-composite footprints"],
        formula: "αᵢ(p) = oᵢ Gᵢ²ᴰ(p),  Tᵢ = ∏_{j<i}(1 − αⱼ),  C(p) = Σᵢ Tᵢ αᵢ(p)cᵢ",
        representation: "Gaussian centers, covariance, opacity, and color",
        gradient: "projection, Gaussian shape, opacity, color, and camera",
        caveat: "culling, depth reordering, splitting, and pruning are discrete"
      },
      ray: {
        label: "surface-transport pipeline",
        kicker: "explicit path representation",
        title: "Ray / path tracer",
        summary: "Rays intersect explicit surfaces. Path tracing continues through sampled scattering events and averages path contributions.",
        caption: "Primary rays find surfaces; shadow and bounce segments evaluate transport before many samples are accumulated into a pixel.",
        description: "A ray or path tracer launches rays, intersects surfaces, follows shadow or bounce paths, and accumulates their contributions.",
        steps: ["Launch camera rays", "Intersect surfaces", "Evaluate shadow / bounce paths", "Accumulate path samples"],
        formula: "Ĉ(p) = (1/N) Σₖ W(Xₖ; θ)",
        representation: "surfaces, materials, lights, and sampled paths",
        gradient: "fixed-topology intersections and path contributions",
        caveat: "visibility and path-topology changes need specialized estimators"
      }
    };

    var stageLabel = document.getElementById("familyStageLabel");
    var modeKicker = document.getElementById("familyModeKicker");
    var modeTitle = document.getElementById("familyModeTitle");
    var modeSummary = document.getElementById("familyModeSummary");
    var stageCaption = document.getElementById("familyStageCaption");
    var description = document.getElementById("family-scene-desc");
    var objectOutput = document.getElementById("familyObjectOutput");
    var stageOutput = document.getElementById("familyStageOutput");
    var pipeline = document.getElementById("familyPipeline");
    var formula = document.getElementById("familyFormula");
    var representation = document.getElementById("familyRepresentation");
    var gradient = document.getElementById("familyGradient");
    var caveat = document.getElementById("familyCaveat");
    var sensitivity = document.getElementById("familySensitivityOverlay");
    var rasterProjection = document.getElementById("familyRasterProjection");
    var splatProjection = document.getElementById("familySplatProjection");
    var announcement = document.getElementById("familyAnnouncement");
    var currentMode = "raster";
    var sensitivityVisible = false;

    function setObjectPosition() {
      var shift = Number(objectSlider.value);
      var outputShift = shift * 0.62;
      scene.querySelectorAll(".family-scene-shift").forEach(function (node) {
        node.setAttribute("transform", "translate(" + shift + " 0)");
      });
      scene.querySelectorAll(".family-output-shift").forEach(function (node) {
        node.setAttribute("transform", "translate(" + number(outputShift, 1) + " 0)");
      });
      scene.querySelectorAll('[data-ray-end="object"]').forEach(function (node) {
        node.setAttribute("x2", 293 + shift);
      });
      scene.querySelectorAll('[data-ray-hit="object"]').forEach(function (node) {
        node.setAttribute("cx", 293 + shift);
      });
      scene.querySelectorAll('[data-shadow-start="object"]').forEach(function (node) {
        node.setAttribute("x1", 293 + shift);
      });
      if (rasterProjection) {
        rasterProjection.setAttribute("d", "M" + (294 + shift) + " 270L137 177M" + (397 + shift) + " 270L137 247M" + (345 + shift) + " 218L137 203");
      }
      if (splatProjection) {
        splatProjection.setAttribute("d", "M" + (312 + shift) + " 240L" + number(588 + outputShift, 1) + " 208M" + (380 + shift) + " 236L" + number(659 + outputShift, 1) + " 205M" + (315 + shift) + " 298L" + number(594 + outputShift, 1) + " 264M" + (385 + shift) + " 294L" + number(662 + outputShift, 1) + " 260");
      }

      var positionText = shift === 0 ? "center" : (shift < 0 ? Math.abs(shift) + " left" : shift + " right");
      objectOutput.value = positionText;
      objectOutput.textContent = positionText;
      objectSlider.setAttribute("aria-valuetext", positionText);
    }

    function setStage() {
      var stage = Number(stageSlider.value);
      var activeLayer = scene.querySelector('[data-family-layer="' + currentMode + '"]');
      if (activeLayer) {
        activeLayer.querySelectorAll("[data-stage]").forEach(function (node) {
          node.classList.toggle("is-revealed", Number(node.getAttribute("data-stage")) <= stage);
        });
      }
      pipeline.querySelectorAll("li").forEach(function (item, index) {
        item.classList.toggle("is-revealed", index < stage);
        item.classList.toggle("is-current", index === stage - 1);
      });
      stageOutput.value = stage + " / 4";
      stageOutput.textContent = stage + " / 4";
      stageSlider.setAttribute("aria-valuetext", "stage " + stage + " of 4, " + modes[currentMode].steps[stage - 1]);
    }

    function setMode(mode, announce) {
      if (!modes[mode]) return;
      currentMode = mode;
      var data = modes[mode];
      scene.querySelectorAll("[data-family-layer]").forEach(function (layer) {
        layer.classList.toggle("is-active", layer.getAttribute("data-family-layer") === mode);
      });
      stageLabel.textContent = data.label;
      modeKicker.textContent = data.kicker;
      modeTitle.textContent = data.title;
      modeSummary.textContent = data.summary;
      stageCaption.textContent = data.caption;
      description.textContent = data.description;
      formula.textContent = data.formula;
      representation.textContent = data.representation;
      gradient.textContent = data.gradient;
      caveat.textContent = data.caveat;
      pipeline.querySelectorAll("strong").forEach(function (item, index) {
        item.textContent = data.steps[index];
      });
      setStage();
      setObjectPosition();
      if (announce && announcement) announcement.textContent = data.title + " selected. " + data.summary;
    }

    picker.forEach(function (radio) {
      radio.addEventListener("change", function () {
        if (radio.checked) setMode(radio.value, true);
      });
    });
    objectSlider.addEventListener("input", setObjectPosition);
    stageSlider.addEventListener("input", setStage);
    sensitivityButton.addEventListener("click", function () {
      sensitivityVisible = !sensitivityVisible;
      sensitivity.classList.toggle("is-visible", sensitivityVisible);
      sensitivityButton.setAttribute("aria-pressed", sensitivityVisible ? "true" : "false");
      sensitivityButton.textContent = sensitivityVisible ? "Hide illustrative perturbation" : "Show illustrative perturbation";
      if (announcement) announcement.textContent = sensitivityVisible ? "Illustrative nearby-state comparison shown; it is not a computed derivative." : "Illustrative nearby-state comparison hidden.";
    });

    setMode("raster", false);
  }

  function setupSionnaSampling() {
    var slider = document.getElementById("sionnaDirectionSlider");
    var output = document.getElementById("sionnaDirectionOutput");
    var fan = document.getElementById("sionnaRayFan");
    var description = document.getElementById("sionna-figure-desc");
    if (!slider || !output || !fan) return;

    var svgNamespace = "http://www.w3.org/2000/svg";
    var sourceX = 122;
    var sourceY = 196;
    var goldenAngle = Math.PI * (3 - Math.sqrt(5));

    function update() {
      var count = Number(slider.value);
      while (fan.firstChild) fan.removeChild(fan.firstChild);
      for (var i = 0; i < count; i += 1) {
        var angle = i * goldenAngle;
        var radius = 255;
        var line = document.createElementNS(svgNamespace, "line");
        line.setAttribute("x1", sourceX);
        line.setAttribute("y1", sourceY);
        line.setAttribute("x2", number(sourceX + Math.cos(angle) * radius, 2));
        line.setAttribute("y2", number(sourceY + Math.sin(angle) * radius, 2));
        line.setAttribute("class", "sionna-sampled-ray");
        fan.appendChild(line);
      }
      output.value = String(count);
      output.textContent = String(count);
      slider.setAttribute("aria-valuetext", count + " displayed directions in the two-dimensional teaching view");
      if (description) {
        description.textContent = count + " displayed directions radiate from a source in a two-dimensional explanatory cross-section. A dashed provisional reflection chain misses the point target; a solid image-method-refined path connects it.";
      }
    }

    slider.addEventListener("input", update);
    update();
  }

  function setupReflectionLab() {
    // Forward-only specular lab: the wall is known and fixed; the slider moves
    // the receiver and every path quantity is derived by the image method.
    var slider = document.getElementById("surfaceSlider");
    if (!slider) return;

    var source = { x: 82, y: 82 };
    var receiver = { x: 612, y: 96 };
    var wallH = 286;

    var reflectionPath = document.getElementById("reflectionPath");
    var reflectionGlow = document.getElementById("reflectionGlow");
    var bouncePoint = document.getElementById("bouncePoint");
    var normalLine = document.getElementById("normalLine");
    var receiverPoint = document.getElementById("receiverPoint");
    var sceneDescription = document.getElementById("reflection-desc");
    var surfaceOutput = document.getElementById("surfaceOutput");
    var predictedLength = document.getElementById("predictedLength");
    var reflectionHit = document.getElementById("reflectionHit");
    var reflectionAngle = document.getElementById("reflectionAngle");

    // pathLength: emitter-to-receiver length via one specular bounce on the
    // wall at height wallH, computed by unfolding. Takes receiver x; returns units.
    function pathLength(receiverX) {
      var dx = receiverX - source.x;
      var unfoldedY = 2 * wallH - receiver.y - source.y;
      return Math.hypot(dx, unfoldedY);
    }

    // bounceX: horizontal coordinate of the specular hit from the image
    // method. Takes receiver x; returns the hit's x coordinate.
    function bounceX(receiverX) {
      var mirroredReceiverY = 2 * wallH - receiver.y;
      var u = (wallH - source.y) / (mirroredReceiverY - source.y);
      return source.x + u * (receiverX - source.x);
    }

    // angleFromNormal: incidence/reflection angle in degrees for the current
    // geometry. Takes receiver x; returns degrees from the wall normal.
    function angleFromNormal(receiverX) {
      var dx = Math.abs(receiverX - source.x);
      var unfoldedY = Math.abs(2 * wallH - receiver.y - source.y);
      return Math.atan2(dx, unfoldedY) * 180 / Math.PI;
    }

    // update: redraw the scene and readouts for the slider's receiver
    // position. Takes nothing; returns nothing.
    function update() {
      receiver.x = Number(slider.value);
      var hitX = bounceX(receiver.x);
      var length = pathLength(receiver.x);
      var currentAngle = angleFromNormal(receiver.x);
      var path = "M" + source.x + " " + source.y + "L" + number(hitX, 2) + " " + wallH + "L" + receiver.x + " " + receiver.y;

      receiverPoint.setAttribute("transform", "translate(" + receiver.x + " " + receiver.y + ")");
      reflectionPath.setAttribute("d", path);
      reflectionGlow.setAttribute("d", path);
      bouncePoint.setAttribute("transform", "translate(" + number(hitX, 2) + " " + wallH + ")");
      normalLine.setAttribute("x1", hitX);
      normalLine.setAttribute("x2", hitX);

      surfaceOutput.value = number(receiver.x, 0);
      surfaceOutput.textContent = number(receiver.x, 0);
      predictedLength.textContent = number(length, 2) + " units";
      reflectionHit.textContent = number(hitX, 2);
      reflectionAngle.textContent = number(currentAngle, 2) + "\u00b0";

      if (sceneDescription) {
        sceneDescription.textContent = "The wall position is known at " + wallH + ". The receiver is at horizontal position " + number(receiver.x, 0) + ". Geometry derives a reflection hit at " + number(hitX, 2) + " and an angle of " + number(currentAngle, 2) + " degrees from the normal.";
      }
    }

    slider.addEventListener("input", update);
    update();
  }

  function setupVisibilityLab() {
    var slider = document.getElementById("edgeSlider");
    if (!slider) return;

    var rayY = 135;
    var minEdge = Number(slider.min);
    var maxEdge = Number(slider.max);
    var occluder = document.getElementById("occluder");
    var edgeHandle = document.getElementById("edgeHandle");
    var edgeLabel = document.getElementById("edgeLabel");
    var visibleRay = document.getElementById("visibleRay");
    var blockedTail = document.getElementById("blockedTail");
    var blockedSymbol = document.getElementById("blockedSymbol");
    var rayStatus = document.getElementById("rayStatus");
    var edgeOutput = document.getElementById("edgeOutput");
    var badge = document.getElementById("visibilityBadge");
    var visibilityValue = document.getElementById("visibilityValue");
    var derivative = document.getElementById("visibilityDerivative");
    var stepDot = document.getElementById("stepDot");
    var sceneDescription = document.getElementById("visibility-scene-desc");

    function xGraph(edge) {
      return 42 + (edge - minEdge) / (maxEdge - minEdge) * 360;
    }

    function update() {
      var edge = Number(slider.value);
      var blocked = edge <= rayY;
      var exactSwitch = edge === rayY;

      occluder.setAttribute("y", edge);
      occluder.setAttribute("height", Math.max(0, 330 - edge));
      edgeHandle.setAttribute("cy", edge);
      edgeLabel.setAttribute("y", edge - 8);
      edgeLabel.textContent = "edge e = " + number(edge, 0);
      edgeOutput.value = number(edge, 0);
      edgeOutput.textContent = number(edge, 0);

      visibleRay.setAttribute("x2", blocked ? 350 : 612);
      visibleRay.classList.toggle("blocked", blocked);
      blockedTail.classList.toggle("active", blocked);
      blockedSymbol.classList.toggle("active", blocked);
      rayStatus.classList.toggle("blocked", blocked);
      rayStatus.textContent = exactSwitch ? "PATH SWITCH" : (blocked ? "BLOCKED" : "VISIBLE");

      badge.classList.toggle("blocked", blocked);
      badge.textContent = exactSwitch ? "path switches here" : (blocked ? "path blocked" : "path visible");
      visibilityValue.textContent = blocked ? "0" : "1";
      derivative.textContent = exactSwitch ? "undefined" : "0";

      stepDot.setAttribute("cx", xGraph(edge));
      stepDot.setAttribute("cy", blocked ? 126 : 42);

      if (sceneDescription) {
        sceneDescription.textContent = "The occluder edge is at " + number(edge, 0) +
          ". The direct ray is " + (blocked ? "blocked" : "visible") +
          (exactSwitch ? " at the visibility boundary." : ".");
      }

    }

    slider.addEventListener("input", update);
    update();
  }

  setupRendererFamilyLab();
  setupSionnaSampling();
  setupReflectionLab();
  setupVisibilityLab();
}());
