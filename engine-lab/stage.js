// stage.js: the scene. Loads the engine, lights it on a pale stage and draws it like a technical illustration:
// one lit pass with a soft shadow, then a screen pass that adds ink lines from depth, contact shading, haze
// and the paper background. Also owns picking, the cutaway planes and the hover highlight.
import * as T from './vendor/three.bundle.min.js';

const PAPER = new T.Color('#f7f5ef');
const INK = new T.Color('#2a2622');
const CAP = new T.Color('#b5382a');             // cut faces, painted red like a museum cutaway
const ACCENT = new T.Color('#2f7d5b');
const FLOOR_Y = -0.292;
const CUTS = {
  // name: plane that keeps one side. Scene axes: +X front of car, +Y up, -Z the car's left.
  top: new T.Plane(new T.Vector3(0, -1, 0), 0.0005),
  side: new T.Plane(new T.Vector3(0, 0, 1), 0.0005),
  cross: new T.Plane(new T.Vector3(1, 0, 0), 0.0885),
};

const POST_VERT = /* glsl */`
varying vec2 vUv;
void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`;

const POST_FRAG = /* glsl */`
uniform sampler2D tColor;
uniform sampler2D tDepth;
uniform vec2 uTexel;
uniform mat4 uProjInv;
uniform vec3 uPaper;
uniform vec3 uInk;
uniform float uGhost;
uniform float uFogNear;
uniform float uFogFar;
uniform float uAoScale;
uniform float uQuality;
uniform float uFloorY;
uniform mat4 uViewInv;
varying vec2 vUv;

// View-space position of the surface seen at a screen point.
vec3 viewPos(vec2 uv) {
  float d = texture2D(tDepth, uv).r;
  vec4 p = uProjInv * vec4(uv * 2.0 - 1.0, d * 2.0 - 1.0, 1.0);
  return p.xyz / p.w;
}

float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }

void main() {
  vec3 col = texture2D(tColor, vUv).rgb;
  float depth = texture2D(tDepth, vUv).r;
  vec3 p = viewPos(vUv);
  bool sky = depth > 0.99999;
  // paper with a soft vignette
  vec2 q = vUv - 0.5;
  vec3 paper = uPaper * (1.0 - 0.34 * dot(q, q));

  // ink: silhouettes where depth jumps, creases where the rebuilt normal turns
  vec3 pr = viewPos(vUv + vec2(uTexel.x, 0.0)), pl = viewPos(vUv - vec2(uTexel.x, 0.0));
  vec3 pu = viewPos(vUv + vec2(0.0, uTexel.y)), pd = viewPos(vUv - vec2(0.0, uTexel.y));
  float zc = sky ? min(min(pr.z, pl.z), min(pu.z, pd.z)) : p.z;
  float lap = abs(pr.z + pl.z + pu.z + pd.z - 4.0 * p.z) / max(0.2, -max(zc, p.z));
  float edge = smoothstep(0.0016, 0.0052, lap);
  float crease = 0.0;
  float ao = 1.0;
  if (!sky) {
    // the surface normal rebuilt from depth, taking on each axis the neighbour that lies on the same surface
    vec3 dx = abs(pr.z - p.z) < abs(p.z - pl.z) ? pr - p : p - pl;
    vec3 dy = abs(pu.z - p.z) < abs(p.z - pd.z) ? pu - p : p - pd;
    vec3 n = normalize(cross(dx, dy));
    crease = smoothstep(0.6, 1.35, length(fwidth(n))) * (1.0 - edge);
    if (uQuality > 0.5) {
      // contact shading: how much nearby geometry stands in front of this surface
      float rad = uAoScale * 0.028 / max(0.3, -p.z);
      float rot = hash(gl_FragCoord.xy) * 6.2831853;
      float occ = 0.0;
      for (int i = 0; i < 6; i++) {
        float a = rot + float(i) * 2.399963;
        float rr = rad * (0.3 + 0.7 * float(i + 1) / 6.0);
        vec3 s = viewPos(vUv + vec2(cos(a), sin(a) * uTexel.y / uTexel.x) * rr);
        vec3 v = s - p;
        float d2 = dot(v, v);
        occ += max(0.0, dot(n, v) / sqrt(d2 + 1e-8) - 0.18) / (1.0 + d2 * 900.0);
      }
      ao = 1.0 - min(1.0, occ / 6.0 * 2.1);
    }
  }
  // the lit scene is rendered in linear light: bring its highlights down gently, leave the paper alone
  col = col * (1.0 + col / 9.0) / (1.0 + col * 0.42);
  bool ground = !sky && abs((uViewInv * vec4(p, 1.0)).y - uFloorY) < 0.004;
  if (sky) col = paper;
  if (ground) {                 // the floor is only the shadow it receives, laid on the same paper as the sky
    float shade = clamp(dot(texture2D(tColor, vUv).rgb, vec3(0.333)) / dot(uPaper, vec3(0.333)), 0.0, 1.0);
    col = paper * mix(1.0, shade, 0.9);
    ao = mix(1.0, ao, 0.55);
    crease = 0.0;
  }
  col *= mix(1.0, ao, 0.62);
  // gas view: the engine fades to a line drawing so the flow reads through it
  float lum = dot(col, vec3(0.3, 0.55, 0.15));
  vec3 ghost = mix(paper, vec3(lum) * 0.6 + paper * 0.5, 0.42);
  if (!sky && !ground) col = mix(col, ghost, uGhost);
  float line = max(edge, crease * 0.42) * mix(0.9, 0.5, uGhost);
  // lines fade with distance, so the far edge of the floor never draws a horizon
  float reach = sky ? -max(max(pr.z, pl.z), max(pu.z, pd.z)) : -p.z;
  line *= 1.0 - smoothstep(uFogNear, uFogFar, reach);
  col = mix(col, uInk, line);
  // haze toward the horizon
  float fog = sky ? 1.0 : smoothstep(uFogNear, uFogFar, -p.z);
  col = mix(col, paper, fog * (1.0 - line * 0.5));
  gl_FragColor = vec4(col, 1.0);
  #include <colorspace_fragment>
  gl_FragColor.rgb += (hash(gl_FragCoord.xy + 0.37) - 0.5) / 255.0;      // dither: the vignette would band otherwise
}`;

/**
 * Wrap a loaded engine part with the state the page keeps for it.
 * Takes the mesh. Returns the record: its group, label, exploded offset, spin and parameter data, and the offset
 * a hand or the exploded view has given it.
 */
function partRecord(mesh) {
  const d = mesh.userData;
  const ex = d.ex || [0, 0, 0];
  const spin = d.spin ? { axis: new T.Vector3(d.spin[0], d.spin[2], -d.spin[1]).normalize(), ratio: d.spin[3] } : null;
  const param = d.param ? JSON.parse(d.param) : null;
  if (param && param.axis) param.axisV = new T.Vector3(param.axis[0], param.axis[2], -param.axis[1]).normalize();
  if (param && param.move) param.moveV = new T.Vector3(param.move[0], param.move[2], -param.move[1]).multiplyScalar(0.001);
  if (param && param.pivot) param.pivotV = new T.Vector3(param.pivot[0], param.pivot[2], -param.pivot[1]).multiplyScalar(0.001);
  return {
    mesh, name: mesh.name, group: d.g || '', label: d.label || mesh.name, cut: d.cut === 1,
    explode: new T.Vector3(ex[0], ex[2], -ex[1]).multiplyScalar(0.001),
    spin, param,
    restPos: mesh.position.clone(), restQuat: mesh.quaternion.clone(),
    manual: new T.Vector3(), offset: new T.Vector3(), vel: new T.Vector3(), held: false,
  };
}

/**
 * Give a material its cutaway behaviour: clipped by the active plane, with the inside faces painted as a flat cap.
 * Takes the material and the shared uniforms that carry the cap colour and normal. Returns the material.
 */
function makeCuttable(material, shared) {
  material.onBeforeCompile = (shader) => {
    shader.uniforms.uCapColor = shared.capColor;
    shader.uniforms.uCapNormal = shared.capNormal;
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', '#include <common>\nuniform vec3 uCapColor;\nuniform vec3 uCapNormal;')
      .replace('#include <color_fragment>', '#include <color_fragment>\nif (!gl_FrontFacing) diffuseColor.rgb = uCapColor;')
      .replace('#include <metalnessmap_fragment>', '#include <metalnessmap_fragment>\nif (!gl_FrontFacing) metalnessFactor = 0.0;')
      .replace('#include <roughnessmap_fragment>', '#include <roughnessmap_fragment>\nif (!gl_FrontFacing) roughnessFactor = 0.95;')
      .replace('#include <normal_fragment_maps>', '#include <normal_fragment_maps>\nif (!gl_FrontFacing) normal = normalize(uCapNormal);');
  };
  material.customProgramCacheKey = () => 'cuttable';
  return material;
}

/**
 * Build the stage.
 * Takes the canvas, the model URL and a progress callback (fraction 0 to 1). Returns the stage: its camera, the
 * list of part records, the baked clip, and the functions the page calls each frame.
 */
export async function createStage(canvas, url, onProgress) {
  const renderer = new T.WebGLRenderer({ canvas, antialias: false, alpha: false, powerPreference: 'high-performance' });
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = T.PCFShadowMap;
  renderer.localClippingEnabled = true;
  renderer.toneMapping = T.NoToneMapping;
  const coarse = matchMedia('(pointer: coarse)').matches;

  const scene = new T.Scene();
  scene.background = PAPER.clone();
  const pmrem = new T.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new T.RoomEnvironment(), 0.03).texture;
  scene.environmentIntensity = 0.72;
  const camera = new T.PerspectiveCamera(28, 1, 0.25, 40);

  const key = new T.DirectionalLight(0xfff4e6, 2.3);
  key.position.set(-1.4, 3.2, 1.9);
  key.castShadow = true;
  key.shadow.mapSize.set(coarse ? 1024 : 1536, coarse ? 1024 : 1536);
  Object.assign(key.shadow.camera, { left: -1.5, right: 1.5, top: 1.5, bottom: -1.5, near: 0.5, far: 8 });
  key.shadow.bias = -0.0006;
  key.shadow.normalBias = 0.004;
  key.shadow.radius = 5;
  key.shadow.blurSamples = 12;
  scene.add(key, key.target);
  scene.add(new T.HemisphereLight(0xffffff, 0xcfc8ba, 0.55));

  const floor = new T.Mesh(new T.PlaneGeometry(30, 30), new T.ShadowMaterial({ opacity: 0.2, color: 0x2a2622 }));
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = FLOOR_Y;
  floor.receiveShadow = true;
  scene.add(floor);

  const gltf = await new Promise((resolve, reject) => new T.GLTFLoader().load(url, resolve,
    (e) => onProgress(e.total ? e.loaded / e.total : 0), reject));
  scene.add(gltf.scene);

  // materials: one plain and one cuttable copy of each, so the cutaway spares the moving parts
  const shared = { capColor: { value: CAP.clone() }, capNormal: { value: new T.Vector3(0, 0, 1) } };
  const plain = new Map(), cuttable = new Map();
  const parts = [];
  gltf.scene.traverse((o) => {
    if (!o.isMesh) return;
    const src = o.material;
    if (!plain.has(src)) {
      src.envMapIntensity = 1.0;
      src.side = T.FrontSide;
      plain.set(src, src);
      cuttable.set(src, makeCuttable(src.clone(), shared));
    }
    const rec = partRecord(o);
    o.material = rec.cut ? cuttable.get(src) : plain.get(src);
    o.castShadow = true;
    o.receiveShadow = true;
    o.userData.part = rec;
    parts.push(rec);
  });
  // each timing chain is 98 links of two shapes: draw it as two instanced meshes and treat it as one part
  const chains = {};
  for (const tag of ['L', 'R']) {
    const links = parts.filter((p) => p.name.startsWith(`Chain_${tag}_`)).sort((p, q) => p.name.localeCompare(q.name));
    if (!links.length) continue;
    const group = new T.Group();
    group.name = `Chain_${tag}`;
    const rec = { ...links[0], mesh: group, name: group.name, restPos: new T.Vector3(), restQuat: new T.Quaternion(),
      manual: new T.Vector3(), offset: new T.Vector3(), vel: new T.Vector3(), explode: new T.Vector3() };
    const byGeometry = new Map(), slots = [];
    for (const link of links) {
      if (!byGeometry.has(link.mesh.geometry)) byGeometry.set(link.mesh.geometry, []);
      slots.push([link.mesh.geometry, byGeometry.get(link.mesh.geometry).push(link) - 1]);
      link.mesh.parent.remove(link.mesh);
      parts.splice(parts.indexOf(link), 1);
    }
    const instanced = new Map();
    for (const [geometry, list] of byGeometry) {
      const im = new T.InstancedMesh(geometry, list[0].mesh.material, list.length);
      im.instanceMatrix.setUsage(T.DynamicDrawUsage);
      im.castShadow = true;
      im.receiveShadow = true;
      im.frustumCulled = false;
      im.userData.part = rec;
      group.add(im);
      instanced.set(geometry, im);
    }
    gltf.scene.add(group);
    parts.push(rec);
    chains[tag] = { slots, instanced, group, count: links.length };
  }
  const byName = new Map(parts.map((p) => [p.name, p]));
  const meshes = [];
  for (const p of parts) p.mesh.traverse((o) => { if (o.isMesh) meshes.push(o); });
  const linkMatrix = new T.Matrix4(), linkPos = new T.Vector3(), linkQuat = new T.Quaternion(), ONE = new T.Vector3(1, 1, 1), X = new T.Vector3(1, 0, 0);

  /**
   * Place one chain link.
   * Takes the chain's tag, the link's number, its position in scene metres and its turn about the crank axis.
   * Returns nothing.
   */
  function setLink(tag, k, x, y, z, angle) {
    const [geometry, index] = chains[tag].slots[k];
    linkMatrix.compose(linkPos.set(x, y, z), linkQuat.setFromAxisAngle(X, angle), ONE);
    chains[tag].instanced.get(geometry).setMatrixAt(index, linkMatrix);
  }

  /**
   * Tell the renderer the chain links have moved.
   * Takes nothing. Returns nothing.
   */
  function commitLinks() {
    for (const c of Object.values(chains)) {
      for (const im of c.instanced.values()) {
        im.instanceMatrix.needsUpdate = true;
        if (!im.boundingSphere) im.computeBoundingSphere();
      }
    }
  }

  const clip = gltf.animations[0];

  // hover highlight: the same geometry drawn once more as a tinted film
  const glow = new T.Mesh(undefined, new T.MeshBasicMaterial({ color: ACCENT, transparent: true, opacity: 0.3, depthWrite: false,
    polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2 }));
  glow.matrixAutoUpdate = false;
  glow.visible = false;
  glow.renderOrder = 5;
  scene.add(glow);

  // two render targets: the lit scene with its depth, and nothing else; lines and haze come from the depth
  const target = new T.WebGLRenderTarget(4, 4, { type: T.HalfFloatType, depthTexture: new T.DepthTexture(4, 4, T.UnsignedIntType), samples: 0 });
  const post = new T.ShaderMaterial({
    vertexShader: POST_VERT, fragmentShader: POST_FRAG, depthTest: false, depthWrite: false,
    uniforms: {
      tColor: { value: target.texture }, tDepth: { value: target.depthTexture }, uTexel: { value: new T.Vector2() },
      uProjInv: { value: new T.Matrix4() }, uPaper: { value: PAPER.clone() }, uInk: { value: INK.clone() }, uGhost: { value: 0 },
      uFogNear: { value: 3.4 }, uFogFar: { value: 12.0 }, uAoScale: { value: 1 }, uQuality: { value: coarse ? 0 : 1 },
      uFloorY: { value: FLOOR_Y }, uViewInv: { value: new T.Matrix4() },
    },
  });
  const quad = new T.Mesh(new T.PlaneGeometry(2, 2), post);
  quad.frustumCulled = false;
  const postScene = new T.Scene();
  postScene.add(quad);
  const postCamera = new T.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  const overlay = new T.Scene();          // gas particles and motes, drawn over the finished picture

  const raycaster = new T.Raycaster();
  const ndc = new T.Vector2();
  let cut = null;

  const quality = { level: 0, scales: [1, 0.8, 0.64, 0.5] };
  let inset = 0;              // CSS pixels at the bottom of the canvas covered by the control bar

  /**
   * Match the drawing buffers to the canvas's size on the page, within a pixel budget.
   * Takes nothing. Returns nothing.
   */
  function resize() {
    const w = Math.max(1, canvas.clientWidth), h = Math.max(1, canvas.clientHeight);
    const dpr = window.devicePixelRatio || 1;
    const wanted = coarse ? Math.min(dpr, 2) : (dpr >= 2 ? Math.min(dpr, 2) : 1.5);
    const budget = Math.sqrt((coarse ? 1.5e6 : 2.4e6) / (w * h));
    const ratio = Math.max(0.5, Math.min(wanted, budget) * quality.scales[quality.level]);
    renderer.setPixelRatio(ratio);
    renderer.setSize(w, h, false);
    target.setSize(Math.round(w * ratio), Math.round(h * ratio));
    post.uniforms.uTexel.value.set(1 / (w * ratio), 1 / (h * ratio));
    post.uniforms.uQuality.value = coarse || quality.level >= 2 ? 0 : 1;
    camera.aspect = w / h;
    // centre the picture in the part of the canvas the control bar leaves free
    if (inset > 0) camera.setViewOffset(w, h, 0, inset / 2, w, h); else camera.clearViewOffset();
    camera.updateProjectionMatrix();
    post.uniforms.uAoScale.value = camera.projectionMatrix.elements[5] * 0.5;
  }

  /**
   * Say how much of the canvas's bottom the control bar covers.
   * Takes the height in CSS pixels. Returns nothing.
   */
  function setInset(pixels) {
    if (Math.abs(pixels - inset) < 2) return;
    inset = pixels;
    resize();
  }

  /**
   * Measure what a frame costs on this machine and step the picture quality down until it fits, so a modest
   * graphics chip still turns the engine smoothly. Frames are timed to completion, not by the display's
   * callbacks, which browsers slow down for windows they consider hidden.
   * Takes nothing. Returns the quality level chosen (0 is best).
   */
  function calibrate() {
    const gl = renderer.getContext(), pixel = new Uint8Array(4);
    const timed = (frames) => {
      const t0 = performance.now();
      for (let i = 0; i < frames; i++) { render(0); gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixel); }
      return (performance.now() - t0) / frames;
    };
    timed(1);                                   // the first frame compiles the shaders
    while (quality.level < quality.scales.length - 1 && timed(3) > 24) {
      quality.level += 1;
      if (quality.level >= 2) { key.shadow.mapSize.set(1024, 1024); if (key.shadow.map) { key.shadow.map.dispose(); key.shadow.map = null; } }
      resize();
      timed(1);
    }
    return quality.level;
  }

  /**
   * Find the part under a point on the canvas.
   * Takes page-relative x and y in CSS pixels. Returns { part, point } for the nearest visible part, or null.
   */
  function pick(x, y) {
    ndc.set((x / canvas.clientWidth) * 2 - 1, -(y / canvas.clientHeight) * 2 + 1);
    raycaster.setFromCamera(ndc, camera);
    for (const hit of raycaster.intersectObjects(meshes, false)) {
      const part = hit.object.userData.part;
      if (part.ghost) continue;                                                // faded out by the focus view
      if (cut && part.cut && cut.distanceToPoint(hit.point) < 0) continue;      // that surface has been cut away
      return { part, point: hit.point.clone() };
    }
    return null;
  }

  /**
   * Choose the cutaway.
   * Takes a key of CUTS or null for none. Returns nothing.
   */
  function setCut(name) {
    cut = name ? CUTS[name] : null;
    for (const m of cuttable.values()) {
      m.clippingPlanes = cut ? [cut] : null;
      m.clipShadows = true;
      m.side = cut ? T.DoubleSide : T.FrontSide;
      m.needsUpdate = true;
    }
  }

  /**
   * Show or clear the hover highlight.
   * Takes a part record or null. Returns nothing.
   */
  function setHover(part) {
    glow.visible = Boolean(part && part.mesh.geometry);
    if (!glow.visible) return;
    glow.geometry = part.mesh.geometry;
    glow.userData.follow = part.mesh;
  }

  /**
   * Draw one frame.
   * Takes the ghost amount for the gas view (0 to 1). Returns nothing.
   */
  function render(ghost) {
    if (glow.visible) { glow.matrix.copy(glow.userData.follow.matrixWorld); glow.matrixWorldNeedsUpdate = true; }
    if (cut) shared.capNormal.value.copy(cut.normal).negate().transformDirection(camera.matrixWorldInverse);
    post.uniforms.uGhost.value = ghost;
    post.uniforms.uProjInv.value.copy(camera.projectionMatrixInverse);
    post.uniforms.uViewInv.value.copy(camera.matrixWorld);
    renderer.setRenderTarget(target);
    renderer.render(scene, camera);
    renderer.setRenderTarget(null);
    renderer.render(postScene, postCamera);
    renderer.autoClear = false;
    renderer.render(overlay, camera);
    renderer.autoClear = true;
  }

  // focus view: the parts of one system stay solid, housings named as clear turn to glass, the rest fades
  const ghost = new T.MeshBasicMaterial({ color: 0x7d776c, transparent: true, opacity: 0.05, depthWrite: false });
  const glass = new T.MeshStandardMaterial({ color: 0xd4d8da, metalness: 0.1, roughness: 0.2, transparent: true, opacity: 0.26, depthWrite: false, side: T.DoubleSide });

  /**
   * Show one system on its own.
   * Takes the set of part names to keep solid and the set to draw as glass, or null to show everything again.
   * Returns nothing.
   */
  function setFocus(solid, clear) {
    for (const p of parts) {
      const mode = !solid ? 'base' : clear.has(p.name) ? 'glass' : solid.has(p.name) ? 'base' : 'ghost';
      p.ghost = mode === 'ghost';
      p.mesh.traverse((o) => {
        if (!o.isMesh) return;
        o.userData.base ??= o.material;
        o.material = mode === 'ghost' ? ghost : mode === 'glass' ? glass : o.userData.base;
        o.castShadow = mode === 'base';
      });
    }
  }

  /**
   * Where a part, or a point in the scene, is on the page.
   * Takes a part record or a Vector3. Returns { x, y } in CSS pixels and whether it is in front of the camera.
   */
  function screenPoint(what) {
    const m = what.mesh, v = new T.Vector3();
    if (what.isVector3) v.copy(what);
    else if (m.geometry) {
      if (!m.geometry.boundingSphere) m.geometry.computeBoundingSphere();
      v.copy(m.geometry.boundingSphere.center).applyMatrix4(m.matrixWorld);
    } else m.getWorldPosition(v);
    v.project(camera);
    return { x: (v.x * 0.5 + 0.5) * canvas.clientWidth, y: (-v.y * 0.5 + 0.5) * canvas.clientHeight, front: v.z < 1 };
  }

  resize();
  return { setFocus, screenPoint, setInset, renderer, scene, overlay, camera, parts, byName, clip, resize, pick, setCut, setHover, render, calibrate, setLink, commitLinks, quality, coarse, cutNames: Object.keys(CUTS) };
}
