// Idempotent integration. Preserve the original and reviewed equations.
import {readFileSync,writeFileSync} from 'node:fs';
function edit(path,before,after){const s=readFileSync(path,'utf8');if(s.includes(after))return;if(!s.includes(before))throw new Error(`Unrecognized source in ${path}: ${before.slice(0,90)}`);writeFileSync(path,s.replace(before,after));}
const base='kalman-filter-derivations/';
edit(base+'bento-deck.mjs',"import { applyMathReview } from './math-review.mjs';","import { applyMathReview } from './math-review.mjs';\nimport { applyFamilyGuide, familyLiveEntries } from './family-guide.mjs';\nimport { applyVisualPolish } from './visual-polish.mjs';");
edit(base+'bento-deck.mjs','export const deck = {','applyFamilyGuide(slides);\napplyVisualPolish(slides);\n\nexport const deck = {');
edit(base+'bento-deck.mjs',"title: 'One Filter, Four Derivation Families',","title: 'One Filter, Many Derivations',");
edit(base+'bento-deck.mjs','export const inlineLiveMap = [','export const inlineLiveMap = [\n  ...familyLiveEntries(slides),');
edit(base+'live/families.js','draw();}}catch(e)','draw();}};}catch(e)');
edit(base+'live/families.js',"const state={...settings[mode]};let result;","document.body.dataset.family=mode;\n const state={...settings[mode]};let result;");
edit(base+'live/families.html','</style><script defer src="families.js">','</style><link rel="stylesheet" href="../visual.css?v=20260905"><script defer src="families.js">');
edit(base+'live/families.html','<body><main','<body class="kf-family-lab"><main');
// tests/ is two levels below the repository, not three.
edit(base+'tests/browser-smoke.cjs',"const root=path.resolve(__dirname,'../../..');","const root=path.resolve(__dirname,'..','..');");
let build=readFileSync(base+'build-bento.mjs','utf8');
build=build.replaceAll('One Filter, Four Derivation Families','One Filter, Many Derivations').replace('An interactive Bento deck presenting four Kalman filter derivation families with equation sheets and deterministic experiments.','Six viewpoints on Kalman filter derivations: a governing-equation overview, group summaries, reviewed equation sheets, and eight interactive experiments.');
if(!build.includes('visual.css?v=20260905'))build=build.replace("const pageTitle =", "html = html.replace('</head>', '<link rel=\"stylesheet\" href=\"visual.css?v=20260905\">\\n</head>');\nconst pageTitle =");
writeFileSync(base+'build-bento.mjs',build);
writeFileSync(base+'README.md',`# One Filter, Many Derivations

The canonical page is a 33-slide Bento presentation with eight inline experiments.

## Editing and building

- \`bento-deck.mjs\`: retained base slides and live mappings.
- \`math-review.mjs\`: reviewed equations and assumption sheets (preserved).
- \`family-guide.mjs\`: six viewpoints, 30 catalogue entries, five new lab slides, and ordering.
- \`visual-polish.mjs\`: native Bento layout, cover, comparison table, route summaries, and progress. Applied after the mathematical-review layer; does not change algorithms.
- \`visual.css\`: scoped finishing and coordinated light-theme controls.
- \`live/families.js\`: pure numerical models shared by the browser and tests.
- \`live/families.html\`: responsive experiment shell.
- Build: \`node kalman-filter-derivations/build-bento.mjs\`.
- Test: \`node --test kalman-filter-derivations/tests/family-guide.test.cjs\`.
- Browser QA: \`node kalman-filter-derivations/tests/browser-smoke.cjs\` with Playwright installed.

## Reading map

Four derivation principles: probability/conditioning, second-moment projection, quadratic objectives, and variational/KL updating. Two computational viewpoints: recursive elimination/messages and numerical representations/algebraic bridges. The catalogue lists principal routes, reformulations, and extensions, not 30 independent filters or an exhaustive taxonomy. Unrestricted KL updating is Bayes; QR is an implementation rather than another statistical optimality principle.

The opening compares questions, unknowns, governing equations, outputs, and overlap. Every group has a route summary and a distinctive experiment:

- Bayes: Gaussian versus mixture priors with equal first two moments; full posterior versus the Kalman Gaussian.
- Projection: gain-risk and orthogonality; Gaussian versus two-point variables with identical moments.
- Least squares: prior/data costs, stationarity, curvature, and one Newton step.
- KL: candidate mean/variance, entropy, evidence, and the exact free-energy gap.
- Recursive inference: stepwise Schur elimination; batch, filtering, and RTS smoothing.
- Numerical forms: retained comparison of covariance, information, Joseph, and QR under finite precision.

The original scalar-fusion and covariance-geometry labs remain. Every lab has a static/print fallback. Existing named deep links remain available; every slide links back to the group map. The legacy deck is unchanged.

## Visual system

Warm paper, high-contrast ink, Georgia display headings, system sans-serif text, and a consistent accent per group. Equations have dedicated rendering space. The six-card opening separates principles from computational views; a progress line tracks position. No new font or image download is required. New controls support keyboard navigation, mobile stacking, and reduced motion.
`);
console.log('Integrated six viewpoints, five new experiments, and visual finishing.');
