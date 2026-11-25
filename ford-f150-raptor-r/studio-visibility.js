// The two vertex values give the fraction of the studio visible from the
// surface with the closures shut and open. Zero weights expose the assembly
// to the whole room for exploded, pulled-apart and isolated inspections.
export function applyStudioVisibility(material, weights) {
  const previousCompile = material.onBeforeCompile;
  const previousKey = material.customProgramCacheKey();
  material.onBeforeCompile = function(shader, renderer) {
    previousCompile.call(this, shader, renderer);
    shader.uniforms.enclosureWeights = weights;
    shader.vertexShader = `
      attribute vec2 _studio_visibility;
      uniform vec2 enclosureWeights;
      varying float studioVisibility;
    ` + shader.vertexShader.replace('#include <begin_vertex>', `
      #include <begin_vertex>
      studioVisibility = 1.0 - dot(vec2(1.0) - _studio_visibility, enclosureWeights);
    `);
    shader.fragmentShader = 'varying float studioVisibility;\n' +
      shader.fragmentShader.replace('#include <aomap_fragment>', `
        #include <aomap_fragment>
        // A little bounced light remains even where no part of the room is visible.
        reflectedLight.indirectDiffuse *= mix(0.14, 1.0, studioVisibility);
        float specularVisibility = mix(0.06, 1.0, studioVisibility);
        reflectedLight.indirectSpecular *= specularVisibility;
        #ifdef USE_CLEARCOAT
          clearcoatSpecularIndirect *= specularVisibility;
        #endif
      `);
  };
  material.customProgramCacheKey = () => previousKey + '|studio-visibility';
}
