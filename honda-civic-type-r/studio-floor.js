import * as THREE from './vendor/three-r170.js?v=31c84364';

// A capture from beneath the floor locates the actual tires and chassis (1:1 metric meters).
// Each channel records occlusion at a different height above the ground plane (Y = 0).
export function createStudioFloor(scene, renderer, backgroundColor) {
  const size = 512;
  const target = new THREE.WebGLRenderTarget(size, size, {
    generateMipmaps: true, minFilter: THREE.LinearMipmapLinearFilter,
    depthBuffer: true, stencilBuffer: false,
  });
  const footprint = new THREE.OrthographicCamera(-4.0, 4.0, 4.0, -4.0, 0.02, 40.0);
  footprint.position.y = -12.0;
  footprint.up.set(0, 0, 1);
  footprint.lookAt(0, 0, 0);
  footprint.layers.set(1);

  const mask = new THREE.ShaderMaterial({
    side: THREE.DoubleSide, toneMapped: false,
    vertexShader: `
      varying float height;
      void main() {
        vec4 world = modelMatrix * vec4(position, 1.0);
        height = world.y;
        gl_Position = projectionMatrix * viewMatrix * world;
      }`,
    fragmentShader: `
      varying float height;
      void main() {
        if (height < -0.004) discard;
        vec3 visibility = exp(-max(height, 0.0) / vec3(0.072, 0.612, 1.35));
        gl_FragColor = vec4(visibility, 1.0);
      }`,
  });

  const span = { value: 8.0 };
  // Neutral studio cyclorama floor matching the MR2 and GT3 studios.
  const material = new THREE.MeshBasicMaterial({
    color: 0xc1c2c4, toneMapped: false, fog: false,
  });
  material.onBeforeCompile = shader => {
    shader.uniforms.footprintMap = { value: target.texture };
    shader.uniforms.footprintSpan = span;
    shader.uniforms.horizonColor = { value: new THREE.Color(backgroundColor) };
    shader.vertexShader = 'varying vec2 floorPosition;\n' + shader.vertexShader.replace(
      '#include <begin_vertex>', '#include <begin_vertex>\nfloorPosition = vec2(position.x, -position.y);',
    );
    shader.fragmentShader = `
      varying vec2 floorPosition;
      uniform sampler2D footprintMap;
      uniform float footprintSpan;
      uniform vec3 horizonColor;
      // Four staggered taps of a filtered mip produce a soft penumbra
      // without another full-screen pass. Radius is in meters.
      vec4 blurredFootprint(vec2 uv, float radius) {
        vec2 d = vec2(radius / footprintSpan, 0.0);
        float lod = max(0.0, log2(radius * ${size.toFixed(1)} / footprintSpan));
        return .25 * (
          textureLod(footprintMap, uv + d, lod) + textureLod(footprintMap, uv - d, lod)
          + textureLod(footprintMap, uv + d.yx, lod) + textureLod(footprintMap, uv - d.yx, lod));
      }
    ` + shader.fragmentShader.replace('#include <opaque_fragment>', `
      vec2 uvGround = floorPosition / footprintSpan + .5;
      float edge = max(abs(uvGround.x - .5), abs(uvGround.y - .5));
      if (edge < .5) {
        float contact = blurredFootprint(uvGround, 0.025).r;
        float middle = blurredFootprint(uvGround, 0.10).g;
        float broad = blurredFootprint(uvGround, 0.31).b;
        float shade = min(.97, .22 * contact + .68 * middle + .34 * broad);
        shade *= 1.0 - smoothstep(.47, .5, edge);
        outgoingLight *= 1.0 - shade;
      }
      outgoingLight = mix(outgoingLight, horizonColor, smoothstep(2.8, 27.0, length(floorPosition)));
      #include <opaque_fragment>
    `);
  };
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(450, 450), material);
  floor.position.y = -0.0022;
  floor.rotation.x = -Math.PI / 2;
  scene.add(floor);

  const casters = [];
  const center = new THREE.Vector3();
  const scale = new THREE.Vector3();
  const clearColor = new THREE.Color();

  function addCaster(mesh) {
    mesh.layers.enable(1);
    if (!mesh.geometry.boundingSphere) mesh.geometry.computeBoundingSphere();
    casters.push(mesh);
  }

  function update() {
    if (!floor.visible || !casters.length) return;
    scene.updateMatrixWorld();
    let reach = 3.6;
    for (const caster of casters) {
      if (!caster.visible) continue;
      const sphere = caster.geometry.boundingSphere;
      center.copy(sphere.center).applyMatrix4(caster.matrixWorld);
      scale.setFromMatrixScale(caster.matrixWorld);
      const radius = sphere.radius * Math.max(scale.x, scale.y, scale.z);
      reach = Math.max(reach, Math.abs(center.x) + radius, Math.abs(center.z) + radius);
    }
    const half = Math.ceil((reach + 0.36) / 0.32) * 0.32;
    if (Math.abs(half - footprint.right) > 1e-4) {
      footprint.left = footprint.bottom = -half;
      footprint.right = footprint.top = half;
      footprint.updateProjectionMatrix();
      span.value = half * 2;
    }
    const previousTarget = renderer.getRenderTarget();
    const previousBackground = scene.background;
    const previousOverride = scene.overrideMaterial;
    const previousShadowUpdate = renderer.shadowMap.needsUpdate;
    const clearAlpha = renderer.getClearAlpha();
    renderer.getClearColor(clearColor);
    try {
      scene.overrideMaterial = mask;
      scene.background = null;
      renderer.shadowMap.needsUpdate = false;
      renderer.setClearColor(0, 0);
      renderer.setRenderTarget(target);
      renderer.render(scene, footprint);
    } finally {
      renderer.setRenderTarget(previousTarget);
      renderer.setClearColor(clearColor, clearAlpha);
      renderer.shadowMap.needsUpdate = previousShadowUpdate;
      scene.background = previousBackground;
      scene.overrideMaterial = previousOverride;
    }
  }
  return { floor, addCaster, update };
}
