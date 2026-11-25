import * as THREE from './vendor/three-r170.js?v=31c84364';

// The reflection camera is behind the glass. Its off-axis frustum passes
// through the glass boundaries, so each surface point samples its reflected
// sightline. No arbitrary rear-facing camera angle or extra image flip is used.
export function reflectPoint(point, center, normal) {
  return point.clone().addScaledVector(normal, -2 * point.clone().sub(center).dot(normal));
}

export function setMirrorCamera(view, surface, eye, far = 4000) {
  const { center, normal, right, up, width, height } = surface;
  const distance = eye.clone().sub(center).dot(normal);
  if (distance <= .005 || width <= 0 || height <= 0) return false;
  const virtualEye = reflectPoint(eye, center, normal);
  const offset = center.clone().sub(virtualEye);
  // Clip the virtual space behind the mirror, including the rear of its
  // housing. A small allowance covers residual non-planarity after the warp.
  const near = distance + (surface.deviation || 0) + .002;
  const scale = near / distance;
  const cx = offset.dot(right), cy = offset.dot(up);
  view.position.copy(virtualEye);
  view.matrix.makeBasis(right, up, normal.clone().negate());
  view.quaternion.setFromRotationMatrix(view.matrix);
  view.up.copy(up);
  view.near = near;
  view.far = Math.max(far, near + 100);
  view.updateMatrixWorld(true);
  view.projectionMatrix.makePerspective(
    (cx - width / 2) * scale, (cx + width / 2) * scale,
    (cy + height / 2) * scale, (cy - height / 2) * scale, near, view.far
  );
  view.projectionMatrixInverse.copy(view.projectionMatrix).invert();
  return true;
}

export function fitMirrorSurface(geometry, matrixWorld, eye) {
  const attribute = geometry.getAttribute('position');
  if (!attribute || attribute.count < 3) return null;
  const points = Array.from({ length: attribute.count }, (_, i) =>
    new THREE.Vector3().fromBufferAttribute(attribute, i).applyMatrix4(matrixWorld));
  const index = geometry.index;
  const count = index ? index.count : attribute.count;
  const triangles = [];
  let dominant = null;
  for (let i = 0; i + 2 < count; i += 3) {
    const ids = [0, 1, 2].map(k => index ? index.getX(i + k) : i + k);
    const [a, b, c] = ids.map(k => points[k]);
    const cross = b.clone().sub(a).cross(c.clone().sub(a));
    const area = cross.length();
    if (area < 1e-10) continue;
    const normal = cross.divideScalar(area);
    const center = a.clone().add(b).add(c).divideScalar(3);
    if (normal.dot(eye.clone().sub(center)) <= 0) continue;
    const triangle = { ids, area, normal, center };
    triangles.push(triangle);
    if (!dominant || area > dominant.area) dominant = triangle;
  }
  if (!dominant) return null;
  const center = new THREE.Vector3(), normal = new THREE.Vector3();
  const vertices = new Map();
  let total = 0;
  for (const triangle of triangles) {
    if (triangle.normal.dot(dominant.normal) < .96) continue;
    center.addScaledVector(triangle.center, triangle.area);
    normal.addScaledVector(triangle.normal, triangle.area);
    total += triangle.area;
    for (const id of triangle.ids) {
      const point = points[id];
      vertices.set(point.toArray().map(x => x.toFixed(6)).join(','), point);
    }
  }
  if (total < 1e-10) return null;
  center.divideScalar(total);
  normal.normalize();
  // The long principal axis follows the actual warped face instead of a
  // hard-coded pre-warp mirror orientation. Power iteration is well conditioned
  // because the mirror is substantially wider than it is tall.
  let right = new THREE.Vector3(0, 0, 1);
  if (Math.abs(right.dot(normal)) > .9) right.set(1, 0, 0);
  right.addScaledVector(normal, -right.dot(normal)).normalize();
  for (let i = 0; i < 16; ++i) {
    const next = new THREE.Vector3();
    for (const point of vertices.values()) {
      const delta = point.clone().sub(center);
      next.addScaledVector(delta, delta.dot(right));
    }
    next.addScaledVector(normal, -next.dot(normal));
    if (next.lengthSq() < 1e-12) break;
    right = next.normalize();
  }
  const up = right.clone().cross(normal).normalize();
  if (up.y < 0) { right.negate(); up.negate(); }
  let left = Infinity, bottom = Infinity, rightEdge = -Infinity, top = -Infinity, deviation = 0;
  for (const point of vertices.values()) {
    const delta = point.clone().sub(center);
    const x = delta.dot(right), y = delta.dot(up);
    left = Math.min(left, x); rightEdge = Math.max(rightEdge, x);
    bottom = Math.min(bottom, y); top = Math.max(top, y);
    deviation = Math.max(deviation, Math.abs(delta.dot(normal)));
  }
  center.addScaledVector(right, (left + rightEdge) / 2).addScaledVector(up, (bottom + top) / 2);
  return { center, normal, right, up, width: rightEdge - left, height: top - bottom, deviation, points };
}

export function createPlanarMirror(mesh, renderer) {
  const original = mesh.material;
  const target = new THREE.WebGLRenderTarget(768, 208);
  const surfaceMaterial = original.clone();
  surfaceMaterial.color.setRGB(0, 0, 0);
  surfaceMaterial.emissive.setRGB(.94, .97, 1);
  surfaceMaterial.emissiveIntensity = 1;
  surfaceMaterial.emissiveMap = target.texture;
  surfaceMaterial.metalness = 0;
  surfaceMaterial.roughness = 1;
  surfaceMaterial.envMapIntensity = 0;
  surfaceMaterial.clearcoat = 0;
  const view = new THREE.PerspectiveCamera();
  const lastEye = new THREE.Vector3(Infinity, Infinity, Infinity);
  let lastMatrix = null, surface = null;
  return {
    restore() { if (mesh.material === surfaceMaterial) mesh.material = original; },
    update(scene, eye, dirty = false, far = 4000) {
      mesh.updateWorldMatrix(true, false);
      const matrix = mesh.matrixWorld.elements;
      const moved = !lastMatrix || matrix.some((n, i) => Math.abs(n - lastMatrix[i]) > 1e-8);
      const eyeMoved = lastEye.distanceToSquared(eye) > 1e-8;
      mesh.material = surfaceMaterial;
      if (!moved && !eyeMoved && !dirty) return false;
      if (moved || !surface) {
        surface = fitMirrorSurface(mesh.geometry, mesh.matrixWorld, eye);
        if (!surface) { mesh.material = original; return false; }
        const uv = [];
        for (const point of surface.points) {
          const delta = point.clone().sub(surface.center);
          uv.push(.5 + delta.dot(surface.right) / surface.width,
            .5 + delta.dot(surface.up) / surface.height);
        }
        mesh.geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
        target.setSize(768, Math.max(64, Math.round(768 * surface.height / surface.width)));
      }
      if (!setMirrorCamera(view, surface, eye, far)) { mesh.material = original; return false; }
      const savedTarget = renderer.getRenderTarget();
      const savedFace = renderer.getActiveCubeFace();
      const savedMipmap = renderer.getActiveMipmapLevel();
      const wasVisible = mesh.visible;
      const shadowAutoUpdate = renderer.shadowMap.autoUpdate;
      const shadowNeedsUpdate = renderer.shadowMap.needsUpdate;
      try {
        mesh.visible = false;
        renderer.shadowMap.autoUpdate = false;
        renderer.shadowMap.needsUpdate = false;
        renderer.setRenderTarget(target);
        renderer.clear();
        renderer.render(scene, view);
      } finally {
        mesh.visible = wasVisible;
        renderer.shadowMap.autoUpdate = shadowAutoUpdate;
        renderer.shadowMap.needsUpdate = shadowNeedsUpdate;
        renderer.setRenderTarget(savedTarget, savedFace, savedMipmap);
      }
      lastMatrix = matrix.slice();
      lastEye.copy(eye);
      return true;
    },
    dispose() { this.restore(); target.dispose(); surfaceMaterial.dispose(); }
  };
}
