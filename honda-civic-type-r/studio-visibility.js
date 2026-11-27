import { BufferAttribute } from './vendor/three-r170.js?v=655c60e5';

export async function loadStudioVisibility(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`HTTP ${response.status} loading studio visibility`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  // A server can also deliver the file with Content-Encoding: gzip.
  if (bytes[0] !== 0x1f || bytes[1] !== 0x8b) return bytes;
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

export function attachStudioVisibility(gltf, bytes, modelVersion) {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const primitiveCount = dv.getUint32(4, true);
  const modelHash = Array.from(bytes.subarray(8, 40), b => b.toString(16).padStart(2, '0')).join('');
  if (dv.getUint32(0) !== 0x464b5631 || !modelVersion || !modelHash.startsWith(modelVersion)) {
    throw new Error('Studio visibility was baked for a different model');
  }
  const { meshes, accessors } = gltf.parser.json;
  let offset = 40 + 4 * primitiveCount;
  let index = 0;
  const attributes = meshes.map(mesh => mesh.primitives.map(primitive => {
    const count = dv.getUint32(40 + 4 * index++, true);
    if (count !== accessors[primitive.attributes.POSITION].count) throw new Error('Studio visibility vertex count mismatch');
    const values = bytes.subarray(offset, offset + 2 * count);
    offset += 2 * count;
    return new BufferAttribute(values, 2, true);
  }));
  if (index !== primitiveCount || offset !== bytes.length) throw new Error('Incomplete studio visibility data');

  const bindings = new Map();
  gltf.scene.traverse(obj => {
    if (!obj.isMesh) return;
    const source = gltf.parser.associations.get(obj);
    const attribute = attributes[source?.meshes]?.[source?.primitives];
    if (attribute?.count !== obj.geometry.getAttribute('position').count) throw new Error('Studio visibility geometry mismatch');
    bindings.set(obj.geometry, attribute);
  });
  for (const [geometry, attribute] of bindings) geometry.setAttribute('_studio_visibility', attribute);
}

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
        #ifdef USE_SHEEN
          sheenSpecularIndirect *= specularVisibility;
        #endif
        #ifdef USE_CLEARCOAT
          clearcoatSpecularIndirect *= specularVisibility;
        #endif
      `);
  };
  material.customProgramCacheKey = () => previousKey + '|studio-visibility';
}
