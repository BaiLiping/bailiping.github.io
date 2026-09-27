"""HTTP/Bento integration regression. Requires Playwright and Chromium.
Run from a full repository checkout. This is separate from offline preview checks.
"""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
import json
import os
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'test-results' / 'likelihood-vs-density'
OUT.mkdir(parents=True, exist_ok=True)

class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *_args):
        pass

server = ThreadingHTTPServer(('127.0.0.1', 0), partial(QuietHandler, directory=str(ROOT)))
Thread(target=server.serve_forever, daemon=True).start()
base = f'http://127.0.0.1:{server.server_port}/likelihood-vs-density/'
report = {'errors': [], 'slides': [], 'labs': []}
try:
    with sync_playwright() as p:
        options = {'executable_path': os.environ['CHROMIUM']} if os.environ.get('CHROMIUM') else {}
        browser = p.chromium.launch(**options)
        page = browser.new_page(viewport={'width': 1440, 'height': 900}, reduced_motion='reduce')
        page.on('pageerror', lambda error: report['errors'].append(str(error)))
        response = page.goto(base, wait_until='networkidle')
        assert response and response.ok
        page.wait_for_selector('.bento-slide', timeout=30000)
        page.wait_for_function('document.getElementById("loading").hidden')
        doc = json.loads(page.locator('#bento-doc').text_content())
        live = json.loads(page.locator('#bento-inline-live-map').text_content())
        assert len(doc['slides']) == 25 and len(live) == 3
        for entry in live:
            assert doc['slides'][entry['slideIndex']]['id'] == entry['slide']
        for i, slide in enumerate(doc['slides']):
            page.evaluate('(i) => location.hash = "#/" + i', i)
            current = page.locator(f'section.present .bento-slide[data-slide-id="{slide["id"]}"]')
            current.wait_for(state='visible')
            if any('math-tex' in el.get('html', '') for el in slide['elements']):
                current.locator('mjx-container').first.wait_for(timeout=30000)
            page.wait_for_timeout(400)
            assert current.locator('[data-mml-node="merror"]').count() == 0
            if slide['id'].endswith('-lab'):
                iframe = current.locator('iframe')
                iframe.wait_for()
                frame = iframe.content_frame
                frame.locator('#metrics .metric').first.wait_for()
                page.wait_for_function('(id) => document.querySelector(`[data-slide-id="${id}"] iframe`)?.dataset.ready === "true"', arg=slide['id'])
                dims = frame.locator('body').evaluate('(b)=>({w:b.scrollWidth,h:b.scrollHeight,iw:innerWidth,ih:innerHeight})')
                assert dims['w'] <= dims['iw'] + 1 and dims['h'] <= dims['ih'] + 1, dims
            report['slides'].append(slide['id'])
            page.screenshot(path=str(OUT / f'slide-{i+1:02d}.png'))
        # Named route and iframe-to-deck navigation; slider arrows must stay local.
        page.goto(base + '#coin-lab', wait_until='networkidle')
        page.wait_for_selector('section.present [data-slide-id="coin-lab"]')
        frame = page.locator('section.present iframe').content_frame
        frame.locator('#theta').press('ArrowRight')
        assert page.locator('section.present [data-slide-id="coin-lab"]').count() == 1
        frame.locator('body').evaluate('(b)=>{b.tabIndex=-1;b.focus()}')
        frame.locator('body').press('PageDown')
        page.wait_for_selector('section.present [data-slide-id="gaussian"]')
        for mode in ['coin', 'gaussian', 'prior']:
            for width in [1132, 390]:
                page.set_viewport_size({'width': width, 'height': 844})
                page.goto(base + 'live/?demo=' + mode, wait_until='networkidle')
                page.locator('#metrics .metric').first.wait_for()
                for slider in page.locator('input[type="range"]').all():
                    for bound in ['min', 'max']:
                        slider.evaluate('(el,k)=>{el.value=el[k];el.dispatchEvent(new Event("input",{bubbles:true}))}', bound)
                        assert 'NaN' not in page.locator('#plots').inner_html()
                page.locator('#reset').click()
                assert page.evaluate('document.documentElement.scrollWidth <= innerWidth + 1')
                if mode == 'prior':
                    old = page.locator('.plot').nth(1).inner_html()
                    page.locator('#alpha').fill('10')
                    page.locator('#alpha').dispatch_event('input')
                    assert old == page.locator('.plot').nth(1).inner_html()
                report['labs'].append({'mode': mode, 'width': width})
                page.screenshot(path=str(OUT / f'lab-{mode}-{width}.png'), full_page=True)
        assert not report['errors'], report['errors']
        browser.close()
finally:
    server.shutdown()
    server.server_close()
    (OUT / 'browser-report.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
