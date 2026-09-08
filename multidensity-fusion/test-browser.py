"""Full HTTP/Chromium integration check. Run from any working directory.
Requires Playwright and its Chromium installation. No credentials or writes.
"""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
import json
import os
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'test-results' / 'multidensity-fusion'
OUT.mkdir(parents=True, exist_ok=True)
class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *_args):
        pass
server = ThreadingHTTPServer(('127.0.0.1', 0), partial(QuietHandler, directory=str(ROOT)))
Thread(target=server.serve_forever, daemon=True).start()
base = f'http://127.0.0.1:{server.server_port}/multidensity-fusion/'
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
        page.wait_for_function('window.MathJax && typeof MathJax.tex2svgPromise === "function"')
        doc = json.loads(page.locator('#bento-doc').text_content())
        assert doc['docId'] == 'multidensity-fusion-bento'
        assert len(doc['slides']) == 40
        assert doc['meta']['optimalityCompanionVersion'] == 1
        live_map = json.loads(page.locator('#bento-inline-live-map').text_content())
        assert len(live_map) == 6
        for entry in live_map:
            assert doc['slides'][entry['slideIndex']]['id'] == entry['slide']
        for i, slide in enumerate(doc['slides']):
            page.evaluate('(i) => location.hash = "#/" + i', i)
            current = page.locator(f'section.present .bento-slide[data-slide-id="{slide["id"]}"]')
            current.wait_for(state='visible')
            page.wait_for_timeout(300)
            assert current.locator('.bento-el-text').count() > 0
            assert page.locator('[data-mml-node="merror"]').count() == 0
            if slide['id'].endswith('-lab'):
                frame = current.locator('iframe').content_frame
                frame.locator('#metrics .metric').first.wait_for()
                dimensions = frame.locator('body').evaluate('(b)=>({h:b.scrollHeight,w:b.scrollWidth,ih:innerHeight,iw:innerWidth})')
                assert dimensions['w'] <= dimensions['iw'] + 1, dimensions
                assert dimensions['h'] <= dimensions['ih'] + 1, dimensions
                report['labs'].append({'slide': slide['id'], **dimensions})
            report['slides'].append(slide['id'])
            page.screenshot(path=str(OUT / f'slide-{i+1:02d}.png'))
        # Named routes and a real parent/iframe navigation message.
        page.goto(base + '#prior-lab', wait_until='networkidle')
        page.wait_for_selector('section.present [data-slide-id="prior-lab"]')
        frame = page.locator('section.present iframe').content_frame
        frame.locator('#m').fill('6')
        frame.locator('#m').dispatch_event('input')
        frame.locator('#m').press('ArrowRight')
        assert page.locator('section.present [data-slide-id="prior-lab"]').count() == 1
        frame.locator('body').evaluate('(b)=>{b.tabIndex=-1;b.focus()}')
        frame.locator('body').press('PageDown')
        page.wait_for_selector('section.present [data-slide-id="known-correlation"]')
        for mode in ['prior', 'correlation', 'geometry', 'pooling', 'bernoulli', 'rumors']:
            for width in [1132, 390]:
                page.set_viewport_size({'width': width, 'height': 844})
                page.goto(base + 'live/?demo=' + mode, wait_until='networkidle')
                page.locator('#metrics .metric').first.wait_for()
                for slider in page.locator('input[type="range"]').all():
                    for bound in ['min', 'max']:
                        slider.evaluate('(el,k)=>{el.value=el[k];el.dispatchEvent(new Event("input",{bubbles:true}))}', bound)
                page.locator('#reset').click()
                assert page.locator('#plot').evaluate('(el)=>!el.innerHTML.includes("NaN")')
                assert page.evaluate('document.documentElement.scrollWidth <= innerWidth + 1')
                if mode == 'pooling':
                    page.locator('#preset').select_option('disjoint')
                    assert page.locator('#metrics').inner_text().count('undefined') == 2
                    page.locator('#w').fill('0')
                    page.locator('#w').dispatch_event('input')
                    assert 'normalized' in page.locator('#metrics').inner_text()
                    page.locator('#reset').click()
                page.screenshot(path=str(OUT / f'lab-{mode}-{width}.png'), full_page=True)
        assert not report['errors'], report['errors']
        browser.close()
finally:
    server.shutdown()
    (OUT / 'browser-report.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
