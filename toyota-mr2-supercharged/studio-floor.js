import * as THREE from './vendor/three-r170.js?v=31c84364';

// Cycles supplies the empty studio's floor illumination. A ground-facing
// capture sees the lowest surface of each moving assembly, including tires.
export async function createStudioFloor(scene, renderer, backgroundColor) {
  const size = 512;
  const target = new THREE.WebGLRenderTarget(size, size, {
    generateMipmaps: true, minFilter: THREE.LinearMipmapLinearFilter,
    depthBuffer: true, stencilBuffer: false,
  });
  const footprint = new THREE.OrthographicCamera(-220, 220, 220, -220, 1, 2200);
  footprint.position.y = -600;
  footprint.up.set(0, 0, 1);
  footprint.lookAt(0, 0, 0);
  footprint.layers.set(1);

  const mask = new THREE.ShaderMaterial({
    side: THREE.DoubleSide, toneMapped: false,
    uniforms: { painted: { value: 0 } },
    vertexShader: `
      varying float height;
      void main() {
        vec4 world = modelMatrix * vec4(position, 1.0);
        height = world.y;
        gl_Position = projectionMatrix * viewMatrix * world;
      }`,
    fragmentShader: `
      varying float height;
      uniform float painted;
      void main() {
        if (height < -0.2) discard;
        vec3 visibility = exp(-max(height, 0.0) / vec3(4.0, 34.0, 75.0));
        gl_FragColor = vec4(visibility, painted * visibility.g);
      }`,
  });
  mask.onBeforeRender = (renderer, scene, camera, geometry, object) => {
    mask.uniforms.painted.value = object.material.name === 'Mat_Super_Red_3E5' ? 1 : 0;
    mask.uniformsNeedUpdate = true;
  };

  const span = { value: 440 };
  const loader = new THREE.TextureLoader();
  const [illumination, assembled] = await Promise.all(
    ['./studio-floor.webp?v=331e17c02f', './studio-contact.webp?v=4fe5b6c9cc'].map(url => loader.loadAsync(url).catch(() => null))
  );
  for (const texture of [illumination, assembled]) {
    if (texture) texture.colorSpace = THREE.SRGBColorSpace;
  }
  const assembledWeight = { value: 0 };
  const material = new THREE.MeshBasicMaterial({
    color: illumination ? 0xffffff : 0xe7e8ea, map: illumination, toneMapped: false, fog: false,
  });
  material.onBeforeCompile = shader => {
    shader.uniforms.footprintMap = { value: target.texture };
    shader.uniforms.assembledMap = { value: assembled };
    shader.uniforms.assembledWeight = assembledWeight;
    shader.uniforms.footprintSpan = span;
    shader.uniforms.horizonColor = { value: new THREE.Color(backgroundColor) };
    shader.vertexShader = 'varying vec2 floorPosition;\n' + shader.vertexShader.replace(
      '#include <begin_vertex>', '#include <begin_vertex>\nfloorPosition = vec2(position.x, -position.y);',
    );
    shader.fragmentShader = `
      varying vec2 floorPosition;
      uniform sampler2D footprintMap;
      uniform sampler2D assembledMap;
      uniform float assembledWeight;
      uniform float footprintSpan;
      uniform vec3 horizonColor;
      // Four staggered taps of a filtered mip produce a soft penumbra
      // without another full-screen pass. Radius is in scene units.
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
        float contact = blurredFootprint(uvGround, 1.4).r;
        vec4 middle = blurredFootprint(uvGround, 5.5);
        vec4 broad = blurredFootprint(uvGround, 14.0);
        float shade = min(.96, .18 * contact + .64 * middle.g + .28 * broad.b);
        shade *= 1.0 - smoothstep(.47, .5, edge);
        outgoingLight *= 1.0 - shade;
        outgoingLight += broad.a * vec3(.044, .003, .001);
      }
      float edgeDistance = max(abs(floorPosition.x), abs(floorPosition.y));
      if (assembledWeight > 0.0 && edgeDistance < 200.0) {
        vec2 bakeUV = vec2(floorPosition.x, -floorPosition.y) / 400.0 + .5;
        vec3 baked = texture2D(assembledMap, bakeUV).rgb;
        float weight = assembledWeight * (1.0 - smoothstep(160.0, 200.0, edgeDistance));
        outgoingLight = mix(outgoingLight, baked, weight);
      }
      outgoingLight = mix(outgoingLight, horizonColor, smoothstep(120.0, 410.0, edgeDistance));
      #include <opaque_fragment>
    `);
  };
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(28000 / 18, 28000 / 18), material);
  floor.position.y = -.12;
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

  function update(closedWeight) {
    assembledWeight.value = assembled ? closedWeight : 0;
    if (!floor.visible || !casters.length) return;
    if (assembledWeight.value === 1) return;
    scene.updateMatrixWorld();
    let reach = 208;
    for (const caster of casters) {
      if (!caster.visible) continue;
      const sphere = caster.geometry.boundingSphere;
      center.copy(sphere.center).applyMatrix4(caster.matrixWorld);
      scale.setFromMatrixScale(caster.matrixWorld);
      const radius = sphere.radius * Math.max(scale.x, scale.y, scale.z);
      reach = Math.max(reach, Math.abs(center.x) + radius, Math.abs(center.z) + radius);
    }
    const half = Math.ceil((reach + 20) / 16) * 16;
    if (half !== footprint.right) {
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
