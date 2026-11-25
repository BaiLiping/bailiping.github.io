// Run after editing a static Bento document. Generated decks also call the helper.
import fs from 'node:fs';
import { applyDeckExtensions, deckExtensions, installExtensionAssets } from '../assets/deck-extensions.mjs';
const root = new URL('../', import.meta.url);
for (const path of Object.keys(deckExtensions)) {
  if (path === 'multidensity-fusion') continue; // Its authoring module runs in the browser.
  const file = new URL(`${path}/index.html`, root);
  let html = fs.readFileSync(file, 'utf8');
  const pattern = /(<script[^>]+id="bento-doc"[^>]*>)([\s\S]*?)(<\/script>)/;
  const match = html.match(pattern);
  if (!match) throw new Error('Missing Bento document: ' + path);
  const doc = applyDeckExtensions(JSON.parse(match[2]), path);
  const serialized = JSON.stringify(doc, null, match[2].includes('\n') ? 1 : undefined).replaceAll('<', '\\u003c');
  html = html.replace(pattern, (_, open, content, close) => open + serialized + close);
  fs.writeFileSync(file, installExtensionAssets(html));
  console.log(`${path}: ${doc.slides.length} slides, companion links installed`);
}
