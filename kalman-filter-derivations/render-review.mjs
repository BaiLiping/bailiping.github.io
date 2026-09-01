import { createRequire } from 'node:module';
import { mkdir, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { deck } from './bento-deck.mjs';
const require = createRequire('/tmp/kalman-review-node/package.json');
const { chromium } = require('playwright');
const out = resolve('kalman-filter-derivations/review-results');
await mkdir(resolve(out, 'screenshots'), { recursive: true });
let browser;
try { browser = await chromium.launch({ channel: 'chrome', headless: true }); }
catch { browser = await chromium.launch({ headless: true }); }
const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
await context.route('https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/**', async route => {
  const suffix = new URL(route.request().url()).pathname.split('/es5/')[1];
  try { await route.fulfill({ path: '/tmp/kalman-review-node/node_modules/mathjax/es5/' + suffix, headers: { 'Access-Control-Allow-Origin': '*' } }); }
  catch { await route.continue(); }
});
const page = await context.newPage();
const errors = [];
page.on('pageerror', error => errors.push(error.message));
const report = { slides: [], live: [], pageErrors: errors };
const url = 'http://127.0.0.1:8765/kalman-filter-derivations/';
try {
  await page.goto(url, { waitUntil: 'networkidle' });
  await page.waitForFunction(() => document.querySelectorAll('.math-tex').length > 0, { timeout: 30000 });
  for (let i = 0; i < deck.slides.length; i++) {
    await page.evaluate(index => { location.hash = '/' + index; }, i);
    await page.waitForTimeout(700);
    await page.evaluate(async () => { if (window.MathJax?.startup?.promise) await window.MathJax.startup.promise; window.typesetDynamicMath?.(); });
    await page.waitForTimeout(600);
    const result = await page.evaluate(() => ({
      mathNodes: document.querySelectorAll('mjx-container').length,
      mathErrors: Array.from(document.querySelectorAll('[data-mjx-error], mjx-merror')).map(e => e.getAttribute('data-mjx-error') || e.textContent),
      textExcerpt: document.body.innerText.slice(0, 300)
    }));
    const filename = `${String(i + 1).padStart(2, '0')}-${deck.slides[i].id}.png`;
    await page.screenshot({ path: resolve(out, 'screenshots', filename) });
    report.slides.push({ number: i + 1, id: deck.slides[i].id, screenshot: filename, ...result });
  }
  for (const demo of ['scalar', 'geometry', 'equivalence']) {
    await page.goto(url + `live/?demo=${demo}`, { waitUntil: 'networkidle' });
    await page.waitForTimeout(1200);
    const input = page.locator('input[type=range]').first();
    await input.evaluate(e => { e.value = String((Number(e.min) + Number(e.max)) / 2); e.dispatchEvent(new Event('input', { bubbles: true })); });
    await page.waitForTimeout(600);
    await page.screenshot({ path: resolve(out, 'screenshots', `live-${demo}.png`) });
    report.live.push({ demo, canvasCount: await page.locator('canvas').count(), status: await page.locator('#status').innerText(), mathErrors: await page.locator('[data-mjx-error], mjx-merror').count() });
  }
} catch (error) {
  report.browserFailure = String(error.stack || error);
} finally {
  await writeFile(resolve(out, 'browser-checks.json'), JSON.stringify(report, null, 2));
  await browser.close();
}
console.log(JSON.stringify(report, null, 2));
if (report.browserFailure || errors.length || report.slides.some(s => s.mathErrors.length) || report.live.some(s => s.mathErrors)) process.exitCode = 1;
