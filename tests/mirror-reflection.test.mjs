import assert from 'node:assert/strict';
import * as THREE from '../porsche-911-gt3/vendor/three-r170.js';
import { reflectPoint, setMirrorCamera, fitMirrorSurface, createPlanarMirror } from '../porsche-911-gt3/mirror-reflection.js';

const close = (a, b, tolerance = 1e-8) => assert.ok(Math.abs(a - b) < tolerance, `${a} != ${b}`);
const vectorClose = (a, b) => a.toArray().forEach((n, i) => close(n, b.toArray()[i]));

for (const tilt of [0, .4, -.7]) {
  const normal = new THREE.Vector3(Math.sin(tilt), 0, Math.cos(tilt));
  const up = new THREE.Vector3(0, 1, 0);
  const right = normal.clone().cross(up);
  const center = new THREE.Vector3(3, -1, 2);
  const surface = { center, normal, right, up, width: 4, height: 2, deviation: 0 };
  const eye = center.clone().addScaledVector(normal, 5).addScaledVector(right, 1.7).addScaledVector(up, .8);
  const virtualEye = reflectPoint(eye, center, normal);
  close(virtualEye.clone().sub(center).dot(normal), -5);
  vectorClose(reflectPoint(virtualEye, center, normal), eye);
  const camera = new THREE.PerspectiveCamera();
  assert.equal(setMirrorCamera(camera, surface, eye, 100), true);
  for (const x of [-2, -.8, 0, 1.3, 2]) for (const y of [-1, .25, 1]) {
    const point = center.clone().addScaledVector(right, x).addScaledVector(up, y);
    const projection = point.clone().project(camera);
    close(projection.x, x / 2);
    close(projection.y, y);
    // A scene point on the virtual eye ray projects to that same mirror UV.
    const scenePoint = virtualEye.clone().lerp(point, 1.6);
    const sceneProjection = scenePoint.clone().project(camera);
    close(sceneProjection.x, projection.x);
    close(sceneProjection.y, projection.y);
    const incident = point.clone().sub(eye).normalize();
    const reflected = incident.clone().addScaledVector(normal, -2 * incident.dot(normal));
    vectorClose(reflected, scenePoint.clone().sub(point).normalize());
  }
  // Virtual geometry behind the glass must be clipped. Real scene geometry
  // on the viewer's side of the glass must remain inside the depth range.
  assert.ok(center.clone().addScaledVector(normal, -.1).project(camera).z < -1);
  const depth = center.clone().addScaledVector(normal, .1).project(camera).z;
  assert.ok(depth > -1 && depth < 1);
  assert.equal(setMirrorCamera(camera, surface, virtualEye, 100), false);
}

const geometry = new THREE.PlaneGeometry(4, 2);
const mesh = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial());
mesh.rotation.set(.2, -.3, .1);
mesh.position.set(1, 2, 3);
mesh.updateMatrixWorld(true);
const eye = new THREE.Vector3(0, 0, 5).applyMatrix4(mesh.matrixWorld);
const fitted = fitMirrorSurface(geometry, mesh.matrixWorld, eye);
assert.ok(fitted);
close(fitted.width, 4, 1e-7);
close(fitted.height, 2, 1e-7);
close(fitted.deviation, 0);
vectorClose(fitted.center, mesh.position);
close(fitted.right.clone().cross(fitted.up).dot(fitted.normal), -1);

// Render caching and try/finally state restoration without a GPU.
let currentTarget = { name: 'main-pass' }, renders = 0, fail = false;
const originalTarget = currentTarget;
const renderer = {
  shadowMap: { autoUpdate: true, needsUpdate: true },
  getRenderTarget: () => currentTarget,
  getActiveCubeFace: () => 2,
  getActiveMipmapLevel: () => 1,
  setRenderTarget(target) { currentTarget = target; },
  clear() {},
  render() { ++renders; if (fail) throw new Error('simulated render failure'); }
};
const originalMaterial = mesh.material;
const mirror = createPlanarMirror(mesh, renderer);
assert.equal(mirror.update({}, eye), true);
assert.equal(mirror.update({}, eye), false);
assert.equal(renders, 1);
assert.equal(mirror.update({}, eye.clone().add(new THREE.Vector3(.1, 0, 0))), true);
assert.equal(renders, 2);
fail = true;
assert.throws(() => mirror.update({}, eye, true), /simulated render failure/);
assert.equal(currentTarget, originalTarget);
assert.equal(mesh.visible, true);
assert.equal(renderer.shadowMap.autoUpdate, true);
assert.equal(renderer.shadowMap.needsUpdate, true);
mirror.restore();
assert.equal(mesh.material, originalMaterial);
mirror.dispose();
console.log('Planar mirror: reflection law, tilted off-axis projection, clipping, mesh fit, caching, and render-state restoration passed.');
