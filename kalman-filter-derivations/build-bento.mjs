import { readFile, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import { deck, inlineLiveMap } from './bento-deck.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const templatePath = resolve(here, '../frame-registration-slides/index.html');
const preview = process.argv.includes('--preview');
const outputPath = resolve(here, preview ? 'bento-preview.html' : 'index.html');
const template = await readFile(templatePath, 'utf8');
const safeJson = value => JSON.stringify(value, null, 1).replaceAll('<', '\\u003c');

const legacyAliases = {
  main: 'overview',
  routes: 'overview',
  result: 'model',
  conditioning: 'bayes',
  information: 'bayes',
  'bayes-closure': 'bayes',
  projection: 'mse',
  covariance: 'mse',
  blue: 'least-squares',
  'square-root': 'least-squares',
  'graphs-equations': 'graphs',
  connections: 'implementations'
};
const slideIndexes = Object.fromEntries(deck.slides.map((slide, index) => [slide.id, index]));
const routeIndexes = Object.fromEntries(
  Object.entries({ ...slideIndexes, ...legacyAliases }).map(([route, target]) => [
    route,
    Number.isInteger(target) ? target : slideIndexes[target]
  ])
);
const routeScript = `    <script>
      (() => {
        const routes = ${JSON.stringify(routeIndexes)};
        const routeToSlide = () => {
          const route = decodeURIComponent(location.hash.replace(/^#\\/?/, ''));
          if (route && !/^\\d+$/.test(route) && Number.isInteger(routes[route])) {
            history.replaceState(null, '', location.pathname + location.search + '#/' + routes[route]);
          }
        };
        addEventListener('hashchange', routeToSlide);
        routeToSlide();
      })();
    </script>`;

const mathHead = String.raw`    <style id="deck-math-style">
      .math-tex{position:relative;white-space:nowrap}
      .math-inline{display:inline-block;vertical-align:-.14em;line-height:1}
      .math-display{display:flex;width:100%;height:100%;align-items:center;justify-content:center;line-height:1}
      .math-tex mjx-container{color:inherit!important;margin:0!important}
      .math-inline mjx-container{display:inline-block!important}
      .math-display mjx-container[display="true"]{display:block!important;width:100%;margin:0!important;text-align:center}
      .math-tex mjx-container[jax="SVG"]>svg{overflow:visible}
      .math-display mjx-container[jax="SVG"]>svg{max-width:100%;max-height:100%;width:auto;height:auto}
      mjx-assistive-mml{position:absolute!important;top:0!important;left:0!important;clip:rect(1px,1px,1px,1px)!important;clip-path:inset(50%)!important;padding:1px 0 0!important;border:0!important;display:block!important;width:1px!important;height:1px!important;overflow:hidden!important;white-space:nowrap!important}
    </style>
    <script>
      window.MathJax = {
        tex: {
          inlineMath: [['\\(', '\\)']],
          displayMath: [['\\[', '\\]']],
          processEscapes: true
        },
        svg: { fontCache: 'local' },
        options: { skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'] },
        startup: {
          typeset: false,
          ready: () => {
            MathJax.startup.defaultReady();
            MathJax.startup.promise.then(() => window.dispatchEvent(new Event('mathjax-ready')));
          }
        }
      };
    </script>
    <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-svg-full.js"></script>
    <script defer src="../assets/mathjax-dynamic.js?v=2"></script>`;

const docPattern = /(<script type="application\/bento\+json" id="bento-doc">\s*)[\s\S]*?(\s*<\/script>)/;
if (!docPattern.test(template)) throw new Error(`Bento document block not found in ${templatePath}`);

let html = template.replace(docPattern, `$1${safeJson(deck)}$2`);
const pageTitle = '<title>One Filter, Four Derivation Families | Bai Liping</title>';
const pageDescription = '<meta name="description" content="An interactive Bento deck presenting four Kalman filter derivation families with equation sheets and deterministic experiments." />';
const canonical = '<link rel="canonical" href="https://bailiping.com/kalman-filter-derivations/" />';
const liveStylesheet = '<link rel="stylesheet" href="../assets/bento-inline-live.css" />';

html = html.replace(/<title>[\s\S]*?<\/title>/, pageTitle);
if (/<meta name="description"\b[^>]*>/.test(html)) {
  html = html.replace(/<meta name="description"\b[^>]*>/, pageDescription);
} else {
  html = html.replace(pageTitle, `${pageTitle}\n    ${pageDescription}`);
}
if (!html.includes(canonical)) html = html.replace(pageDescription, `${pageDescription}\n    ${canonical}`);
if (!html.includes(liveStylesheet)) html = html.replace(canonical, `${canonical}\n    ${liveStylesheet}`);
if (!html.includes('const routes =')) html = html.replace('</head>', `${routeScript}\n  </head>`);
if (!html.includes('id="deck-math-style"')) html = html.replace('</head>', `${mathHead}\n  </head>`);

const liveMap = `    <script type="application/json" id="bento-inline-live-map">\n${safeJson(inlineLiveMap)}\n    </script>`;
const mapPattern = /\s*<script type="application\/json" id="(?:bento-live-config|bento-inline-live-map)">[\s\S]*?<\/script>/;
if (mapPattern.test(html)) {
  html = html.replace(mapPattern, `\n${liveMap}`);
} else {
  html = html.replace('</body>', `${liveMap}\n  </body>`);
}

const liveScript = '<script src="../assets/bento-inline-live.js"></script>';
if (!html.includes(liveScript)) html = html.replace('</body>', `    ${liveScript}\n  </body>`);
html = html.replaceAll('../assets/bento-live.css', '../assets/bento-inline-live.css');
html = html.replaceAll('../assets/bento-live.js', '../assets/bento-inline-live.js');
html = html.replaceAll('../assets/bento-inline-live.css"', '../assets/bento-inline-live.css?v=2"');
html = html.replace(
  '<!DOCTYPE html>',
  '<!DOCTYPE html>\n<!-- Generated by kalman-filter-derivations/build-bento.mjs. Edit bento-deck.mjs, then rebuild. -->'
);
html = html.replace(/[ \t]+$/gm, '').replace(/\n*$/, '\n');

if (!html.includes('"docId": "kalman-filter-four-families-bento"')) throw new Error('Bento document replacement failed');
if (!html.includes('id="bento-inline-live-map"') || html.includes('id="bento-live-config"')) throw new Error('Inline live map replacement failed');
if (!html.includes('mathjax-dynamic.js') || !html.includes('tex-svg-full.js')) throw new Error('MathJax host injection failed');
if (!html.includes("fontCache: 'local'") || !html.includes('mjx-assistive-mml')) {
  throw new Error('MathJax SVG glyph cache or assistive-MathML styling is missing');
}
if (deck.slides.some(slide => slide.elements.some(element => element.type === 'text' && /<\/?(?:sup|sub)\b/i.test(element.html)))) {
  throw new Error('Hand-built sup/sub math found; use LaTeX helpers instead');
}

await writeFile(outputPath, html);
console.log(`Built ${outputPath}`);
console.log(`${deck.slides.length} regular Bento slides; ${inlineLiveMap.length} inline live regions`);

if (!preview) {
  const redirect = `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta http-equiv="refresh" content="0; url=./">
    <title>One Filter, Four Derivation Families</title>
    <script>location.replace('./' + location.search + location.hash);</script>
  </head>
  <body><p><a href="./">Open the interactive Bento deck</a>.</p></body>
</html>
`;
  await Promise.all([
    writeFile(resolve(here, 'consolidated-slides.html'), redirect),
    writeFile(resolve(here, 'slides.html'), redirect)
  ]);
  console.log('Updated compatibility redirects.');
}
