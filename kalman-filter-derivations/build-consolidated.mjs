// index.html is the canonical slide deck; keep the old consolidated URL in sync.
// Run from any directory: node kalman-filter-derivations/build-consolidated.mjs
import { readFile, writeFile } from 'node:fs/promises';
const source = new URL('./index.html', import.meta.url);
const target = new URL('./consolidated-slides.html', import.meta.url);
const html = await readFile(source, 'utf8');
if (!html.includes('id="kl-equations"')) throw new Error('Expected consolidated content');
await writeFile(target, html);
console.log('Synchronized consolidated-slides.html from index.html.');
