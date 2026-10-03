import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {applyChanges, validateDocument, updateLiveMap, syncLiveMounts} from '../assets/authoring-model.mjs';

export const docPattern = /(<script\b[^>]*\bid=["']bento-doc["'][^>]*>)([\s\S]*?)(<\/script>)/;
export const safeJSON = value => JSON.stringify(value, null, 1).replaceAll('<', '\\u003c');
export function readDocument(html) {
  const match = html.match(docPattern);
  if (!match) throw new Error('This page does not contain a Bento presentation.');
  return JSON.parse(match[2]);
}
export function readManifest(directory) {
  const file = path.join(directory, 'authoring.json');
  if (!fs.existsSync(file)) return null;
  const manifest = JSON.parse(fs.readFileSync(file, 'utf8'));
  if (manifest.version !== 1 || !Array.isArray(manifest.changes)) throw new Error('Unsupported authoring.json in ' + directory);
  return manifest;
}

export function renderDocument(html, doc, previous = readDocument(html)) {
  validateDocument(doc, previous);
  doc = structuredClone(doc);
  // Moving a lab's visible fallback should move its live mount as well.
  syncLiveMounts(doc, previous);
  for (const [index, slide] of doc.slides.entries()) {
    for (const element of slide.elements) {
      if (element.id === 'slide-number' && /^\d+\s*\/\s*\d+$/.test(element.html || '')) {
        const padded = /^0\d/.test(element.html);
        element.html = (padded ? String(index + 1).padStart(2, '0') : String(index + 1)) + ' / ' + doc.slides.length;
      }
      if (typeof element.html === 'string' && doc.slides.length !== previous.slides.length) element.html = element.html.replace(new RegExp('\\b' + previous.slides.length + '(?=\\s+SLIDES\\b)', 'gi'), String(doc.slides.length));
    }
  }
  html = html.replace(docPattern, (_, open, body, close) => open + '\n' + safeJSON(doc) + '\n' + close);
  html = html.replace(/(<script\b[^>]*\bid=["'](?:bento-inline-live-map|companion-demo-map)["'][^>]*>)([\s\S]*?)(<\/script>)/g,
    (_, open, body, close) => open + '\n' + safeJSON(updateLiveMap(JSON.parse(body), doc, previous)) + '\n' + close);
  // These are project-owned route aliases, never the compressed Bento runtime.
  html = html.replace(/<script\b([^>]*)>([\s\S]*?)<\/script>/g, (tag, attributes, body) => {
    if (attributes.trim()) return tag;
    const changed = body.replace(/((?:const|let)\s+(?:routes|aliases|indexes|routeMap)\s*=\s*)(\{[^{}]*\})/g, (original, prefix, json) => {
      let routes; try { routes = JSON.parse(json); } catch { return original; }
      if (!Object.values(routes).every(Number.isInteger)) return original;
      const next = {};
      for (const [key, index] of Object.entries(routes)) {
        const id = doc.slides.some(s => s.id === key) ? key : previous.slides[index]?.id;
        const target = doc.slides.findIndex(s => s.id === id);
        if (target >= 0) next[key] = target;
      }
      return prefix + JSON.stringify(next);
    });
    return `<script${attributes}>${changed}</script>`;
  });
  return html;
}

export function withAuthoring(html, builderURL) {
  const directory = path.dirname(fileURLToPath(builderURL));
  const source = readDocument(html), manifest = readManifest(directory);
  if (manifest) {
    if (manifest.documentId !== source.docId) throw new Error('Saved edits belong to a different document: ' + directory);
    html = renderDocument(html, applyChanges(source, manifest.changes), source);
  }
  if (!html.includes('src="../assets/slide-annotations.js"')) html = html.replace('</body>', '<script src="../assets/slide-annotations.js"></script>\n</body>');
  return html;
}

// Builders with a separate static print layout can merge before rendering it.
export function authoringDocument(doc, builderURL) {
  const manifest = readManifest(path.dirname(fileURLToPath(builderURL)));
  if (!manifest) return doc;
  if (manifest.documentId !== doc.docId) throw new Error('Saved edits belong to a different document.');
  const result = applyChanges(doc, manifest.changes);
  validateDocument(result, doc);
  return result;
}
