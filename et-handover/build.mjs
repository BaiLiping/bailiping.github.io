// Rebuild the authored content while retaining the checked-in Bento runtime.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import config from './deck.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const target = path.join(here, 'index.html');
const rawSlides = config.build({ ...config });
const demos = rawSlides.filter(slide => slide.__demo).map(slide => slide.__demo);
const slides = rawSlides.map(({ __demo, ...slide }) => slide);
const doc = {
  format: 'bento/slides', version: 1, docId: config.docId,
  title: config.title, readonly: true,
  meta: {
    subject: config.subject,
    company: 'Chalmers University of Technology'
  },
  size: { width: 1280, height: 720 },
  theme: {
    background: config.paper, color: config.ink,
    accent: config.accent, fontFamily: config.fontFamily
  },
  slides
};
const json = value => JSON.stringify(value, null, 1).replaceAll('<', '\\u003c');
let html = fs.readFileSync(target, 'utf8')
  .replace(/\s*<!-- eo-host-start -->[\s\S]*?<!-- eo-host-end -->/g, '')
  .replace(/\s*<link[^>]+inline-live\.css[^>]*>/g, '')
  .replace(/\s*<meta name="description" content="[^"]*"\s*\/?>/g, '')
  .replace(/\s*<link rel="canonical"[^>]*>/g, '')
  .replace(/\s*<script type="application\/json" id="companion-demo-map">[\s\S]*?<\/script>\s*/g, '')
  .replace(/\s*<script src="\.\/inline-live\.js"><\/script>\s*/g, '');
const pattern = /(<script type="application\/bento\+json" id="bento-doc">\s*)[\s\S]*?(\s*<\/script>)/;
if (!pattern.test(html)) throw new Error('Bento shell is missing #bento-doc');
html = html.replace(pattern, (_, open, close) => open + json(doc) + close)
  .replace(/<title>[\s\S]*?<\/title>/, '<title>' + config.title + '</title>')
  .replace('</head>', `
    <meta name="description" content="${config.description.replaceAll('"', '&quot;')}" />
    <link rel="canonical" href="https://bailiping.com/et-handover/" />
    <link rel="stylesheet" href="./inline-live.css" />
    <!-- eo-host-start -->
    <link rel="stylesheet" href="./slides.css">
    <script defer src="./topic-links.js?v=20260923-paper"></script>
    <link rel="stylesheet" href="./assets/materials.css">
    <style>.companion-demo-stage{border:0!important;border-radius:0!important;box-shadow:none!important;background:#fff!important}.companion-demo-stage:before{content:none!important}.math-display{height:100%}.math-display mjx-container[jax="SVG"]>svg{max-height:100%}</style>
    <script src="./assets/math-config.js"></script>
    <script defer src="./assets/vendor/mathjax-3.2.2-tex-svg-full.js"></script>
    <script defer src="./assets/mathjax-dynamic.js"></script>
    <!-- eo-host-end --></head>`)
  .replace('</body>', `
    <script type="application/json" id="companion-demo-map">\n${json(demos)}\n</script>
    <script src="./inline-live.js"></script>
  </body>`);
fs.writeFileSync(target, html);
console.log(`${config.title}: ${slides.length} slides, ${demos.length} live demos`);
