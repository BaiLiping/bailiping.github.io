#!/usr/bin/env python3
"""Browser smoke tests for the deployed teaching pages.

Run from a full site checkout after `pip install playwright` and
`playwright install chromium`. Outputs screenshots and a JSON audit report.
"""
import functools
import http.server
import json
from pathlib import Path
import re
import threading
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'test-results/frame-registration'
OUT.mkdir(parents=True, exist_ok=True)
class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_): pass
server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(QuietHandler, directory=str(ROOT)))
threading.Thread(target=server.serve_forever, daemon=True).start()
base = f'http://127.0.0.1:{server.server_port}'
report = {'pages': [], 'errors': [], 'missingLocalResources': [], 'slideTextOverflow': [], 'benchmark': []}

with sync_playwright() as pw:
    browser = pw.chromium.launch(headless=True)
    context = browser.new_context(viewport={'width': 1440, 'height': 1000}, reduced_motion='reduce')
    def page_for(path):
        page = context.new_page()
        page.on('pageerror', lambda e: report['errors'].append({'path': path, 'message': str(e)}))
        page.on('response', lambda r: report['missingLocalResources'].append({'url': r.url, 'status': r.status})
                if r.url.startswith(base) and r.status >= 400 else None)
        response = page.goto(base + path, wait_until='networkidle')
        assert response and response.ok, path
        report['pages'].append(path)
        return page
    try:
        page = page_for('/frame-registration/')
        page.wait_for_function('window.MathJax && MathJax.startup && MathJax.startup.promise', timeout=60000)
        page.evaluate('async () => { await MathJax.startup.promise; }')
        assert page.locator('mjx-container').count() > 50, 'MathJax did not render the article'
        assert page.locator('mjx-merror').count() == 0, 'MathJax parse errors'
        page.screenshot(path=str(OUT / 'article-top.png'))
        for control in ['kabSolve','ranStep','icpStep','gicStep','fldNudge','regStep','raceStep',
                        'memNext1','memNext2','memNext3','memNext4','memReset']:
            page.locator('#' + control).click()
            page.wait_for_timeout(120)
        # Exercise the coordinated NDT string-key / cell-hover code path.
        page.locator('#fldC').scroll_into_view_if_needed()
        box = page.locator('#fldC').bounding_box()
        assert box
        for f in [.3, .5, .7]:
            page.mouse.move(box['x'] + f * box['width'], box['y'] + .5 * box['height'])
            page.wait_for_timeout(120)
        page.locator('#benchRun').click()
        page.wait_for_function("document.getElementById('benchStat').textContent.startsWith('done')", timeout=300000)
        rows = page.locator('#benchBody tr')
        assert rows.count() == 9
        for row in rows.all():
            cells = row.locator('td').all_text_contents()
            assert len(cells) == 6, cells
            counts = [tuple(map(int, re.findall(r'\d+', c))) for c in cells if '/ 25' in c]
            assert len(counts) == 2 and counts[1][0] <= counts[0][0], cells
            assert not any(re.search(r'\b(?:NaN|Infinity)\b', c) for c in cells), cells
            report['benchmark'].append(cells)
        page.locator('#benchRun').scroll_into_view_if_needed()
        page.screenshot(path=str(OUT / 'article-benchmark.png'))
        page.set_viewport_size({'width':390,'height':844})
        page.evaluate('window.scrollTo(0,0)')
        page.screenshot(path=str(OUT / 'article-mobile.png'))
        page.close()

        page = page_for('/frame-registration-slides/')
        deck = json.loads(page.locator('#bento-doc').text_content())
        assert len(deck['slides']) == 20
        for i, slide in enumerate(deck['slides']):
            if i: page.keyboard.press('ArrowRight')
            page.wait_for_timeout(1300)
            assert page.locator('.bento-el:visible').count() > 0, f'Blank slide {i+1}'
            assert page.locator('img:visible').evaluate_all('(xs) => xs.every(x => x.complete && x.naturalWidth > 0)'), f'Broken image on slide {i+1}'
            overflow = page.locator('.bento-el-text:visible').evaluate_all('''(els) => els.flatMap(el => {
                const inner = el.querySelector('.bento-text-inner');
                if (!inner || !inner.textContent.trim()) return [];
                const r = document.createRange(); r.selectNodeContents(inner);
                const box = el.getBoundingClientRect(); const text = r.getBoundingClientRect();
                return text.bottom > box.bottom + 5 || text.right > box.right + 5
                  ? [{id:el.dataset.elId, text:inner.textContent, box:{bottom:box.bottom,right:box.right}, textBox:{bottom:text.bottom,right:text.right}}] : [];
            })''')
            for item in overflow: item['slide'] = i+1
            report['slideTextOverflow'].extend(overflow)
            page.screenshot(path=str(OUT / f'slide-{i+1:02d}.png'))
        page.close()

        page = page_for('/frame-registration-slides/live/')
        for mode in ['ransac','icp','ndt']:
            page.locator('#tab-' + mode).click()
            page.locator('#' + mode + '-step').click()
            page.wait_for_timeout(300)
            assert page.locator('#metric-a').inner_text() != '0', mode
            assert not re.search(r'\b(?:NaN|Infinity)\b', page.locator('#metric-pose').inner_text())
            page.screenshot(path=str(OUT / f'live-{mode}.png'))
            page.locator('#' + mode + '-reset').click()
        page.set_viewport_size({'width':390,'height':844})
        page.screenshot(path=str(OUT / 'live-mobile.png'))
        page.close()
        assert not report['errors'], report['errors']
        assert not report['missingLocalResources'], report['missingLocalResources']
        assert not report['slideTextOverflow'], report['slideTextOverflow']
        report['passed'] = True
    finally:
        (OUT / 'browser-report.json').write_text(json.dumps(report, indent=2))
        browser.close()
        server.shutdown()
print('Article mathematics, demo controls, 25-seed benchmark, 20 slides, and three live tabs passed.')
