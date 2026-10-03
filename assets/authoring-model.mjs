// Shared by the editor, local server, and builders. No browser or Node globals.
const clone = value => value === undefined ? undefined : structuredClone(value);
const object = value => value !== null && typeof value === 'object' && !Array.isArray(value);
const keyed = value => Array.isArray(value) && value.every(item => object(item) && typeof item.id === 'string');
const equal = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const ignored = new Set(['readonly', 'collab', 'docId', 'template']);
const forbidden = new Set(['__proto__', 'prototype', 'constructor']);
export const pathLabel = path => path.map(part => typeof part === 'string' ? part : `[${part.id}]`).join('/');

export function changesBetween(before, after, path = []) {
  if (equal(before, after)) return [];
  if (keyed(before) && keyed(after)) {
    const a = new Map(before.map(item => [item.id, item]));
    const b = new Map(after.map(item => [item.id, item]));
    const changes = [...new Set([...a.keys(), ...b.keys()])].flatMap(id => changesBetween(a.get(id), b.get(id), [...path, {id}]));
    const oldOrder = before.map(item => item.id), newOrder = after.map(item => item.id);
    if (!equal(oldOrder, newOrder)) changes.push({kind: 'order', path, before: oldOrder, after: newOrder});
    return changes;
  }
  if (object(before) && object(after)) {
    return [...new Set([...Object.keys(before), ...Object.keys(after)])]
      .filter(key => path.length || !ignored.has(key))
      .flatMap(key => {
        if (forbidden.has(key)) throw new Error('Unsupported property: ' + key);
        return changesBetween(before[key], after[key], [...path, key]);
      });
  }
  return [{kind: 'value', path, hadBefore: before !== undefined, hadAfter: after !== undefined,
    ...(before === undefined ? {} : {before: clone(before)}), ...(after === undefined ? {} : {after: clone(after)})}];
}

function locate(doc, path) {
  let value = doc;
  for (const part of path) {
    if (typeof part === 'string' && forbidden.has(part)) throw new Error('Unsupported property');
    value = typeof part === 'string' ? value?.[part] : Array.isArray(value) ? value.find(item => item.id === part.id) : undefined;
  }
  return value;
}

export function applyChanges(source, changes, {strict = true} = {}) {
  const doc = clone(source), conflicts = [];
  // Apply membership changes before ordering, including when reversing a patch.
  for (const change of [...changes.filter(c => c.kind !== 'order'), ...changes.filter(c => c.kind === 'order')]) {
    const current = locate(doc, change.path);
    const conflict = () => conflicts.push(pathLabel(change.path));
    if (change.kind === 'order') {
      if (!Array.isArray(current)) { conflict(); continue; }
      const ids = current.map(item => item.id);
      const shared = new Set(change.before.filter(id => change.after.includes(id) && ids.includes(id)));
      const observed = ids.filter(id => shared.has(id));
      if (strict && !equal(observed, change.before.filter(id => shared.has(id))) && !equal(observed, change.after.filter(id => shared.has(id)))) {
        conflict(); continue;
      }
      const byId = new Map(current.map(item => [item.id, item]));
      const order = change.after.filter(id => byId.has(id));
      // Keep upstream additions next to their previous source neighbor.
      for (let i = 0; i < ids.length; i++) {
        if (order.includes(ids[i])) continue;
        const previous = ids.slice(0, i).reverse().find(id => order.includes(id));
        const next = ids.slice(i + 1).find(id => order.includes(id));
        const at = previous ? order.indexOf(previous) + 1 : next ? order.indexOf(next) : order.length;
        order.splice(at, 0, ids[i]);
      }
      current.splice(0, current.length, ...order.map(id => byId.get(id)));
      continue;
    }
    if (strict && !equal(current, change.before) && !equal(current, change.after)) { conflict(); continue; }
    const parent = locate(doc, change.path.slice(0, -1)), key = change.path.at(-1);
    if (!parent || !key) { conflict(); continue; }
    if (typeof key === 'string') {
      if (forbidden.has(key)) throw new Error('Unsupported property');
      if (change.hadAfter) parent[key] = clone(change.after); else delete parent[key];
    } else if (Array.isArray(parent)) {
      const index = parent.findIndex(item => item.id === key.id);
      if (!change.hadAfter) { if (index >= 0) parent.splice(index, 1); }
      else if (index < 0) parent.push(clone(change.after));
      else parent[index] = clone(change.after);
    } else conflict();
  }
  if (conflicts.length) throw new Error('Your saved edits overlap source changes at: ' + [...new Set(conflicts)].join(', ') + '. Ask Codex to reconcile them; your saved edits are intact.');
  return doc;
}

export function reverseChanges(changes) {
  return [...changes].reverse().map(change => ({...change, before: change.after, after: change.before,
    ...(change.kind === 'value' ? {hadBefore: change.hadAfter, hadAfter: change.hadBefore} : {})}));
}

export function validateDocument(doc, previous = doc) {
  if (doc?.format !== 'bento/slides' || !Array.isArray(doc.slides) || !doc.slides.length) throw new Error('Keep at least one slide in the presentation.');
  const ids = new Set();
  for (const slide of doc.slides) {
    if (!slide.id || ids.has(slide.id)) throw new Error('Every slide needs a unique ID.');
    ids.add(slide.id);
    if (!Array.isArray(slide.elements)) throw new Error('Invalid slide: ' + slide.id);
    const elements = new Set();
    for (const element of slide.elements) {
      if (!element.id || elements.has(element.id)) throw new Error('Duplicate object on slide ' + slide.id);
      elements.add(element.id);
      for (const key of ['x', 'y', 'w', 'h']) if (element[key] !== undefined && !Number.isFinite(element[key])) throw new Error('Invalid object size on slide ' + slide.id);
      if (/<mjx-container\b/i.test(element.html || '')) throw new Error('An equation lost its editable LaTeX on slide ' + slide.id + '. Undo the text edit and try again.');
    }
  }
  const removed = new Set(previous.slides.filter(slide => !ids.has(slide.id)).map(slide => slide.id));
  for (const slide of doc.slides) {
    if (slide.stateOf && !ids.has(slide.stateOf)) throw new Error('Slide ' + slide.id + ' needs its parent slide.');
    for (const element of slide.elements) if (removed.has(element.link)) throw new Error('Slide ' + slide.id + ' still links to the removed slide ' + element.link + '. Update that link before saving.');
  }
}

export function updateLiveMap(map, doc, previous) {
  const slides = new Map(doc.slides.map((slide, index) => [slide.id, {slide, index}]));
  return map.filter(entry => slides.has(entry.slide)).map(entry => {
    const {slide, index} = slides.get(entry.slide);
    const intro = slides.get(entry.introSlide);
    if (entry.introSlide && (!intro || intro.index !== index - 1)) {
      // Some older decks have several consecutive labs after one introduction.
      // Preserve that existing sequence without forcing a content refactor.
      const oldIntro = previous?.slides.findIndex(s => s.id === entry.introSlide) ?? -1;
      const oldLive = previous?.slides.findIndex(s => s.id === entry.slide) ?? -1;
      const oldGroup = oldIntro >= 0 && oldLive > oldIntro ? previous.slides.slice(oldIntro, oldLive + 1).map(s => s.id) : [];
      const group = intro ? doc.slides.slice(intro.index, index + 1).map(s => s.id) : [];
      if (!oldGroup.length || !equal(oldGroup, group)) throw new Error('Keep the introduction and live slide sequence together: ' + entry.introSlide + ' → ' + entry.slide + '. Move the group, then save again.');
    }
    const marker = slide.elements.find(element => element.id === 'live-demo-mount');
    const updated = {...entry, slideIndex: index};
    if (marker) updated.bounds = {x: marker.x, y: marker.y, width: marker.w, height: marker.h};
    if (entry.parentIndex !== undefined && intro) updated.parentIndex = intro.index;
    return updated;
  });
}

export function syncLiveMounts(doc, previous) {
  const box = element => element && [element.x, element.y, element.w, element.h];
  const sameBox = (a,b) => a && b && equal(box(a),box(b));
  for (const slide of doc.slides) {
    const old = previous.slides.find(s => s.id === slide.id);
    const fallback = slide.elements.find(e => e.id === 'fallback');
    const marker = slide.elements.find(e => e.id === 'live-demo-mount');
    const oldFallback = old?.elements.find(e => e.id === 'fallback');
    const oldMarker = old?.elements.find(e => e.id === 'live-demo-mount');
    if (sameBox(oldFallback,oldMarker) && sameBox(marker,oldMarker) && fallback && !sameBox(fallback,oldFallback)) {
      for (const key of ['x','y','w','h']) marker[key] = fallback[key];
    }
  }
}

export function createManifest(current, previousManifest, editorBaseline, edited) {
  const base = previousManifest ? applyChanges(current, reverseChanges(previousManifest.changes), {strict: false}) : clone(current);
  const result = applyChanges(current, changesBetween(editorBaseline, edited), {strict: false});
  result.readonly = true;
  result.docId = current.docId;
  delete result.collab;
  syncLiveMounts(result, current);
  validateDocument(result, current);
  return {manifest: {version: 1, documentId: current.docId, changes: changesBetween(base, result)}, doc: result};
}
