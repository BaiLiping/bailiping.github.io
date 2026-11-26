// Draco-compress a GLB for the site (16-bit positions); keeps empty leaf nodes the viewer looks up by name.
// Takes: input.glb output.glb. Writes output.glb. Returns nothing.
import { NodeIO } from '@gltf-transform/core';
import { ALL_EXTENSIONS } from '@gltf-transform/extensions';
import { draco, prune } from '@gltf-transform/functions';
import draco3d from 'draco3d';

const [input, output] = process.argv.slice(2);
if (!input || !output) throw new Error('Usage: node compress_model.mjs input.glb output.glb');
const io = new NodeIO().registerExtensions(ALL_EXTENSIONS).registerDependencies({
  'draco3d.encoder': await draco3d.createEncoderModule(),
  'draco3d.decoder': await draco3d.createDecoderModule(),
});
const document = await io.read(input);
await document.transform(prune({ keepLeaves: true, keepAttributes: true, keepIndices: true }), draco({
  method: 'edgebreaker', encodeSpeed: 5, decodeSpeed: 5,
  quantizePosition: 16, quantizeNormal: 12, quantizeTexcoord: 14, quantizeGeneric: 12,
}));
await io.write(output, document);
