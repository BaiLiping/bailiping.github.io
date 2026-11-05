#!/usr/bin/env python3
"""Browser checks for the sampling deck and its live lab.

Run from a full site checkout after `pip install playwright` (set PW_CHANNEL=chrome
to drive an installed Chrome instead of `playwright install chromium`).
Writes screenshots and a JSON report to test-results/sampling-playground.
"""
import functools
import http.server
import json
import os
from pathlib import Path
import re
import threading
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'test-results/sampling-playground'
OUT.mkdir(parents=True, exist_ok=True)
class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_): pass
server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(QuietHandler, directory=str(ROOT)))
threading.Thread(target=server.serve_forever, daemon=True).start()
base = f'http://127.0.0.1:{server.server_port}'
report = {'errors': [], 'missingLocalResources': [], 'slideTextOverflow': [], 'rawMath': [], 'labs': {}}
DEMOS = ['gibbs', 'mh', 'hmc', 'slice', 'rejection', 'importance', 'smc']

VISIBLE = '''() => [...document.querySelectorAll('.reveal .slides .bento-slide[data-slide-id]')]
  .filter(e => { const r = e.getBoundingClientRect(); return r.width > 100 && r.bottom > 0 && r.top < innerHeight && e.offsetParent !== null; })
  .map(e => e.dataset.slideId)[0]'''

def lab_frame(page, demo='gibbs'):
    return next(f for f in page.frames if f'/sampling-playground/live/?demo={demo}' in f.url)

def click_in_frame(page, selector):
    # Bento scales the slide, so click at the element's on-screen centre.
    frame = lab_frame(page)
    box = frame.locator(selector).bounding_box()
    page.mouse.click(box['x'] + box['width'] / 2, box['y'] + box['height'] / 2)

with sync_playwright() as pw:
    browser = pw.chromium.launch(headless=True, channel=os.environ.get('PW_CHANNEL') or None)
    context = browser.new_context(viewport={'width': 1280, 'height': 810})
    page = context.new_page()
    page.on('pageerror', lambda e: report['errors'].append(str(e)))
    page.on('console', lambda m: report['errors'].append(m.text) if m.type == 'error' else None)
    page.on('response', lambda r: report['missingLocalResources'].append({'url': r.url, 'status': r.status})
            if r.url.startswith(base) and r.status >= 400 else None)
    try:
        page.goto(base + '/sampling-playground/', wait_until='networkidle')
        deck = json.loads(page.locator('#bento-doc').text_content())
        live_map = json.loads(page.locator('#bento-inline-live-map').text_content())
        live_ids = {e['slide'] for e in live_map}
        assert len(deck['slides']) == 27 and len(live_map) == 7
        for i, slide in enumerate(deck['slides']):
            page.goto(f'{base}/sampling-playground/#/{i}')
            page.wait_for_timeout(1500 if slide['id'] in live_ids else 700)
            current = page.locator(f'.reveal .slides .bento-slide[data-slide-id="{slide["id"]}"]').first
            frames = current.locator('iframe.companion-demo-frame')
            if slide['id'] in live_ids:
                assert frames.count() == 1, f'no lab mounted on {slide["id"]}'
            else:
                assert frames.count() == 0, f'unexpected iframe on {slide["id"]}'
            text = current.inner_text()
            if re.search(r'\\[\(\[]', text): report['rawMath'].append(slide['id'])
            assert current.locator('mjx-merror').count() == 0, f'MathJax error on {slide["id"]}'
            overflow = current.locator('.bento-el-text').evaluate_all('''(els) => els.flatMap(el => {
                const inner = el.querySelector('.bento-text-inner');
                if (!inner || !inner.textContent.trim()) return [];
                const r = document.createRange(); r.selectNodeContents(inner);
                const box = el.getBoundingClientRect(); const t = r.getBoundingClientRect();
                return t.bottom > box.bottom + 5 || t.right > box.right + 5 ? [{id: el.dataset.elId, text: inner.textContent.slice(0, 80)}] : [];
            })''')
            for item in overflow: item['slide'] = slide['id']
            report['slideTextOverflow'].extend(overflow)
            page.screenshot(path=str(OUT / f'slide-{i+1:02d}.png'))

        # Intro → live, controls, Escape and Page Up inside the lab.
        gibbs = next(i for i, s in enumerate(deck['slides']) if s['id'] == 'gibbs')
        page.goto(f'{base}/sampling-playground/#/{gibbs}')
        page.wait_for_timeout(800)
        page.keyboard.press('ArrowRight')
        page.wait_for_timeout(1800)
        assert page.evaluate(VISIBLE) == 'gibbs-live', page.evaluate(VISIBLE)
        frame = lab_frame(page)
        before = frame.locator('#metrics').inner_text()
        click_in_frame(page, '#step')
        page.wait_for_timeout(300)
        after = frame.locator('#metrics').inner_text()
        assert '13' in after and before != after, (before, after)
        page.keyboard.press('Escape')
        page.wait_for_timeout(300)
        assert page.evaluate("document.activeElement && !!document.activeElement.closest('.reveal')"), 'Escape did not return focus'
        click_in_frame(page, '#stage')
        page.keyboard.press('PageUp')
        page.wait_for_timeout(800)
        assert page.evaluate(VISIBLE) == 'gibbs', 'Page Up did not return to the introduction'

        # Every standalone lab: step, change each slider, reset, reseed; no NaN or Infinity.
        lab = context.new_page()
        lab.on('pageerror', lambda e: report['errors'].append(str(e)))
        for demo in DEMOS:
            lab.goto(f'{base}/sampling-playground/live/?demo={demo}', wait_until='networkidle')
            lab.locator('#step').click()
            for slider in lab.locator('#controls input[type=range]').all():
                slider.focus(); lab.keyboard.press('End'); lab.wait_for_timeout(80); lab.keyboard.press('Home'); lab.wait_for_timeout(80)
            for button in lab.locator('.segmented button').all():
                button.click(); lab.wait_for_timeout(80)
            lab.locator('#auto').click(); lab.wait_for_timeout(900); lab.locator('#auto').click()
            lab.locator('#reseed').click(); lab.locator('#reset').click(); lab.locator('#step').click()
            metrics = lab.locator('#metrics').inner_text()
            assert not re.search(r'NaN|Infinity|undefined', metrics), (demo, metrics)
            report['labs'][demo] = metrics.replace('\n', ' ')
            lab.screenshot(path=str(OUT / f'lab-{demo}.png'))
        lab.set_viewport_size({'width': 390, 'height': 844})
        lab.goto(f'{base}/sampling-playground/live/?demo=gibbs', wait_until='networkidle')
        assert lab.evaluate('document.documentElement.scrollWidth <= innerWidth + 1'), 'horizontal overflow on a phone'
        lab.screenshot(path=str(OUT / 'lab-mobile.png'), full_page=True)

        assert not report['errors'], report['errors']
        assert not report['missingLocalResources'], report['missingLocalResources']
        assert not report['rawMath'], report['rawMath']
        assert not report['slideTextOverflow'], report['slideTextOverflow']
        report['passed'] = True
    finally:
        (OUT / 'browser-report.json').write_text(json.dumps(report, indent=2))
        browser.close()
        server.shutdown()
print('27 slides, 7 intro → live pairs, lab focus and navigation, and all seven lab controls passed.')
