// Reuse the key light's shadow texture for the soft daylight shadow on the sweep.
// The studio floor already supplies the smaller contact shadow under each tire.
export function receiveDaylight(floor, light) {
  const material = floor.material;
  const buildFloor = material.onBeforeCompile;
  const shadowMap = {value:null};
  floor.onBeforeRender = () => { shadowMap.value = light.shadow.map?.texture || null; };
  material.onBeforeCompile = shader => {
    buildFloor(shader);
    shader.uniforms.daylightDepth = shadowMap;
    shader.uniforms.daylightProjection = {value:light.shadow.matrix};
    shader.fragmentShader = `
      #include <packing>
      uniform sampler2D daylightDepth;
      uniform mat4 daylightProjection;
      float daylightOcclusion(vec2 floorPoint) {
        vec4 p = daylightProjection * vec4(floorPoint.x, 0.0, floorPoint.y, 1.);
        vec3 q = p.xyz / p.w;
        if (any(lessThan(q,vec3(0.))) || any(greaterThan(q,vec3(1.)))) return 0.;
        float blocked = 0.;
        for (int i = 0; i < 16; i++) {
          float a = float(i)*2.399963, radius = .009*sqrt((float(i)+.5)/16.);
          float depth = unpackRGBAToDepth(texture2D(daylightDepth, q.xy + radius*vec2(cos(a),sin(a))));
          blocked += step(depth + .0011, q.z);
        }
        return blocked / 16.;
      }
    ` + shader.fragmentShader;
    shader.fragmentShader = shader.fragmentShader.replace(
      'outgoingLight *= 1.0 - shade;',
      'outgoingLight *= (1.0 - shade*.89) * (1.0 - .31*daylightOcclusion(floorPosition));'
    );
  };
}
