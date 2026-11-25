The viewer serves its rendering dependencies from this directory so a download of the web folder also works on a local HTTP server.

- `three-r170.js` contains the viewer's Three.js imports, GLTFLoader, DRACOLoader and RoomEnvironment from the `three@0.170.0` npm package. The bundle was built with `esbuild@0.25.12 --bundle --minify --format=esm`. The entry module is included for reproducibility.
- The three `draco_*` decoder files are the unchanged glTF decoders in that package's `examples/jsm/libs/draco/gltf` directory. Most browsers use the WebAssembly build. The JavaScript decoder supplies the fallback.

Three.js is distributed under the MIT license. Draco is distributed under the Apache License, Version 2.0. Their license texts are included in this directory.
