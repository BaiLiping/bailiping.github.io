// Cold-load regression: a delayed demo must leave the correct slide preview visible.
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');

const base = (process.argv[2] || 'http://127.0.0.1:8767').replace(/\/$/, '');
const outDir = process.env.BENTO_QA_DIR || '/tmp/et-handover-loading-qa';
const activeSlide = 'section.present .bento-slide';
const activeFrame = 'section.present iframe';

async function waitForSlide(page, id) {
  await page.waitForFunction(id =>
    document.querySelector('section.present .bento-slide')?.dataset.slideId === id, id);
}

async function waitForDemo(page, target) {
  await page.waitForFunction(() =>
    document.querySelector('section.present iframe')?.dataset.ready === 'true');
  const frame = await (await page.locator(activeFrame).elementHandle()).contentFrame();
  const state = await frame.evaluate(target => {
    const widget = document.querySelector(target);
    return {
      position: getComputedStyle(widget).position,
      visibility: getComputedStyle(widget).visibility,
      hero: getComputedStyle(document.querySelector('.hero')).visibility,
      mathReady: [...widget.querySelectorAll('.math-tex')].every(n => n.querySelector('mjx-container')),
      overflow: widget.scrollWidth > widget.clientWidth + 2
    };
  }, target);
  assert.deepEqual(state, { position: 'fixed', visibility: 'visible', hero: 'hidden', mathReady: true, overflow: false });
  assert.equal(await page.locator('section.present [role="status"]').isVisible(), false);
  return frame;
}

(async () => {
  await fs.mkdir(outDir, { recursive: true });
  const browser = await chromium.launch({
    executablePath: process.env.CHROME_BIN || undefined,
    headless: true
  });
  const report = [];
  try {
    for (const width of [1440, 390]) {
      for (const asset of ['embed-page.css', 'embed.css', 'demo.js']) {
        const context = await browser.newContext({ viewport: { width, height: 900 } });
        const page = await context.newPage();
        const errors = [];
        page.on('pageerror', error => errors.push(error.message));
        page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
        let release;
        const gate = new Promise(resolve => { release = resolve; });
        let intercepted = false;
        await page.route(url => url.pathname === '/et-handover/live/' + asset, async route => {
          intercepted = true;
          await gate;
          await route.continue();
        });
        try {
          await page.goto(base + '/et-handover/#/11', { waitUntil: 'domcontentloaded' });
          await waitForSlide(page, 's-score-live');
          await page.locator(activeSlide + ' img[src*="score-fallback"]').evaluate(img => img.decode());
          await page.waitForTimeout(500);
          assert.equal(intercepted, true, 'The intended slow resource was requested');
          const loading = await page.evaluate(() => {
            const root = document.querySelector('section.present');
            const frame = root.querySelector('iframe');
            const stage = root.querySelector('.companion-demo-stage');
            const preview = root.querySelector('img[src*="score-fallback"]');
            return {
              ready: frame.dataset.ready === 'true',
              visibility: getComputedStyle(frame).visibility,
              background: getComputedStyle(stage).backgroundColor,
              busy: stage.getAttribute('aria-busy'),
              preview: preview.complete && preview.naturalWidth > 0 && preview.getBoundingClientRect().width > 0,
              overflow: document.documentElement.scrollWidth > innerWidth + 2
            };
          });
          assert.deepEqual(loading, {
            ready: false, visibility: 'hidden', background: 'rgba(0, 0, 0, 0)',
            busy: 'true', preview: true, overflow: false
          });
          assert.equal(await page.getByRole('status').filter({ hasText: 'Loading interactive demo' }).isVisible(), true);
          await page.screenshot({ path: path.join(outDir, `loading-${width}-${asset}.png`) });

          // Leaving a pending load must clear it; a late ready message cannot revive it.
          if (asset === 'demo.js') {
            await page.keyboard.press('PageUp');
            await waitForSlide(page, 's-score');
            await page.waitForFunction(() => [...document.querySelectorAll('iframe')]
              .every(frame => !frame.hasAttribute('src') && !frame.dataset.ready));
            release();
            await page.unrouteAll({ behavior: 'wait' });
            await page.keyboard.press('ArrowRight');
          } else {
            release();
            await page.unrouteAll({ behavior: 'wait' });
          }

          const frame = await waitForDemo(page, '#handover-score-demo');
          const area = await frame.locator('#sr-area').textContent();
          await frame.locator('#sr-extent').press('End');
          assert.notEqual(await frame.locator('#sr-area').textContent(), area, 'Extent control updates the widget');
          await page.screenshot({ path: path.join(outDir, `ready-${width}-${asset}.png`) });
          await frame.locator('#sr-extent').press('Escape');
          await page.waitForFunction(() => document.activeElement === document.querySelector('.reveal'));
          await page.keyboard.press('PageDown');
          await waitForSlide(page, 's-protocol');
          assert.equal(await page.locator('[data-slide-id="s-score-live"] iframe').getAttribute('src'), null);
          await page.keyboard.press('ArrowRight');
          const toy = await waitForDemo(page, '#toy');
          const before = await toy.locator('#frame-label').textContent();
          await toy.locator('#btn-next').click();
          assert.notEqual(await toy.locator('#frame-label').textContent(), before, 'Slide 14 controls still work');
          await toy.locator('#btn-next').press('PageUp');
          await waitForSlide(page, 's-protocol');
          await page.keyboard.press('ArrowLeft');
          await waitForDemo(page, '#handover-score-demo');
          await page.emulateMedia({ media: 'print' });
          assert.equal(await page.locator('section.present .companion-demo-stage').isVisible(), false);
          assert.equal(await page.locator(activeSlide + ' img[src*="score-fallback"]')
            .evaluate(img => img.complete && img.naturalWidth > 0), true);
          assert.deepEqual(errors, [], 'No browser errors');
          report.push({ width, delayed: asset, preview: true, ready: true, navigation: true, controls: true, print: true });
        } finally {
          release();
          await context.close();
        }
      }
    }
    console.log(JSON.stringify(report, null, 2));
    await fs.writeFile(path.join(outDir, 'report.json'), JSON.stringify(report, null, 2) + '\n');
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
