// Canonical entry point retained for existing build commands.
// Content lives in lesson-deck.mjs; this adapter makes links native to Bento,
// whose text sanitizer intentionally removes ordinary HTML anchor elements.
import { deck as authoredDeck, inlineLiveMap, references } from './lesson-deck.mjs';
export const deck = structuredClone(authoredDeck);
export { inlineLiveMap, references };

const overview = deck.slides.find(s => s.id === 'overview');
for (const e of overview.elements) {
  if (!e.id.endsWith('-body')) continue;
  const match = e.html?.match(/<a href="([^"]+)">/);
  if (match) e.link = match[1].startsWith('#/') ? match[1].slice(2) : match[1];
}

const sourceSlide = deck.slides.find(s => s.id === 'references');
const sourceBody = sourceSlide.elements.find(e => e.id === 'refs-left-body');
sourceSlide.elements = sourceSlide.elements.filter(e => e.id !== 'refs-left-body');
Object.values(references).forEach((ref, i) => {
  sourceSlide.elements.push({
    ...sourceBody,
    id: 'reference-' + i,
    y: 248 + 58 * i,
    h: 55,
    fontSize: 14,
    lineHeight: 1.35,
    html: ref.full,
    link: ref.url,
  });
});
sourceSlide.sources = Object.keys(references);
const scope = sourceSlide.elements.find(e => e.id === 'refs-right-body');
scope.html = scope.html.replace(/<p><a href="[^"]+">[^<]+<\/a><\/p>/g, '');
for (const [i, label, link] of [
  [0, 'Scrollable study notes →', './study.html'],
  [1, 'Mathematical audit and test guide →', './math-audit.md'],
]) {
  sourceSlide.elements.push({
    ...scope,
    id: 'source-navigation-' + i,
    y: 548 + 42 * i,
    h: 34,
    fontSize: 15,
    html: label,
    color: '#16736E',
    link,
  });
}
