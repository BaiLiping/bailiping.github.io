"""Build the deck and serve the repository on port 8765 before running."""
import json
import math
import os
import traceback
from pathlib import Path
from playwright.sync_api import sync_playwright

OUT = Path('viem-validation')
OUT.mkdir(exist_ok=True)
URL = os.environ.get('VIEM_TEST_BASE', 'http://127.0.0.1:8765') + '/variational-inference-em/'
report = {'status': 'running', 'slides': [], 'labs': [], 'errors': []}


def close(a, b, tol=1e-7):
    assert math.isfinite(a) and math.isfinite(b) and abs(a-b) <= tol, (a, b)


def screen(page, name):
    page.screenshot(path=str(OUT / (name + '.png')), full_page=True)


try:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={'width': 1440, 'height': 900}, device_scale_factor=1)
        page.on('pageerror', lambda e: report['errors'].append(str(e)))
        page.goto(URL, wait_until='networkidle', timeout=60000)
        page.wait_for_function('window.bento && window.bento.doc && window.bento.doc.slides.length === 26', timeout=60000)
        page.wait_for_function('window.MathJax && MathJax.startup && MathJax.startup.document', timeout=60000)
        slide_ids = page.evaluate('window.bento.doc.slides.map(s=>s.id)')
        for i, sid in enumerate(slide_ids):
            page.goto(URL + '#/' + str(i), wait_until='domcontentloaded')
            page.wait_for_timeout(700)
            locator = page.locator('.bento-slide[data-slide-id="' + sid + '"]').first
            locator.wait_for(state='attached', timeout=15000)
            if not locator.is_visible():
                page.reload(wait_until='networkidle')
                page.wait_for_timeout(600)
                locator = page.locator('.bento-slide[data-slide-id="' + sid + '"]').first
            try:
                locator.scroll_into_view_if_needed(timeout=4000)
            except Exception:
                pass
            page.wait_for_timeout(350)
            data = {'index': i+1, 'id': sid, 'mathErrors': locator.locator('mjx-merror, [data-mjx-error]').count(), 'visible': locator.is_visible(), 'mathContainers': locator.locator('mjx-container').count()}
            report['slides'].append(data)
            assert data['visible'], 'Slide not visible: ' + sid
            assert data['mathErrors'] == 0, 'Math rendering error: ' + sid
            if locator.locator('.math-tex').count():
                assert data['mathContainers'] > 0, 'Math not typeset: ' + sid
            screen(page, f'slide-{i+1:02d}-{sid}')
        for mode in ['meanfield', 'em']:
            page.goto(URL + 'live/?demo=' + mode + '&embed=region', wait_until='networkidle')
            page.wait_for_function('window.VILab')
            page.set_viewport_size({'width': 1136, 'height': 475})
            if mode == 'meanfield':
                page.locator('#reset').click()
                before = page.evaluate('VILab.getState()')
                page.locator('#step').click()
                after = page.evaluate('VILab.getState()')
                close(after['mean'][0], before['rho'] * before['mean'][1])
                close(after['mean'][1], before['mean'][1])
                assert after['updates'] == 1
                page.locator('#fit').click()
                close(float(page.locator('#mf-best').inner_text()), -.5*math.log(1-.8**2), .001)
                page.locator('#independent').click()
                close(float(page.locator('#mf-kl').inner_text()), 0, .001)
                page.locator('#correlated').click()
                page.locator('#reset').click()
                page.locator('#step').click()
            else:
                before = page.evaluate('VILab.getState()')
                page.locator('#step').click()
                after_e = page.evaluate('VILab.getState()')
                close(before['history'][-1]['likelihood'], after_e['history'][-1]['likelihood'])
                close(after_e['history'][-1]['gap'], 0)
                page.locator('#step').click()
                after_m = page.evaluate('VILab.getState()')
                assert after_m['history'][-1]['likelihood'] >= after_e['history'][-1]['likelihood']-1e-7
                assert after_m['q'] == after_e['q'], 'M-step modified q'
                page.locator('#initialization').select_option('identical')
                page.locator('#step').click()
                page.locator('#step').click()
                sym = page.evaluate('VILab.getState()')
                close(sym['model']['mean'][0], sym['model']['mean'][1])
                page.locator('#initialization').select_option('spread')
                page.locator('#learn').check()
                for _ in range(12):
                    page.locator('#step').click()
                assert min(page.evaluate('VILab.getState()')['model']['variance']) >= .09
            screen(page, 'lab-' + mode + '-desktop')
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth+1'), 'Horizontal overflow: '+mode
            page.set_viewport_size({'width': 390, 'height': 844})
            page.wait_for_timeout(200)
            screen(page, 'lab-' + mode + '-mobile')
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth+1'), 'Mobile horizontal overflow: '+mode
            report['labs'].append({'mode': mode, 'desktop': 'passed', 'mobile': 'passed', 'controls': 'passed'})
        page.set_viewport_size({'width': 1440, 'height': 900})
        for index, mode in [(9, 'meanfield'), (17, 'em')]:
            page.goto(URL + '#/' + str(index), wait_until='networkidle')
            page.wait_for_selector('iframe[data-ready="true"]', timeout=30000)
            frames = [f for f in page.frames if 'demo=' + mode in f.url]
            assert frames, 'Missing embedded lab '+mode
            frames[0].wait_for_function('window.VILab', timeout=10000)
            frames[0].locator('#step').click()
            screen(page, 'embedded-' + mode)
        assert not report['errors'], report['errors']
        report['status'] = 'passed'
        browser.close()
except Exception as e:
    report['status'] = 'failed'
    report['exception'] = str(e)
    report['traceback'] = traceback.format_exc()
finally:
    (OUT / 'report.json').write_text(json.dumps(report, indent=2))
    try:
        from PIL import Image, ImageDraw
        thumbs = []
        for fn in sorted(OUT.glob('slide-*.png')):
            im = Image.open(fn).convert('RGB')
            im.thumbnail((480, 300))
            tile = Image.new('RGB', (500, 330), 'white')
            tile.paste(im, ((500-im.width)//2, 8))
            ImageDraw.Draw(tile).text((10, 311), fn.stem, fill='black')
            thumbs.append(tile)
        for start in range(0, len(thumbs), 8):
            group = thumbs[start:start+8]
            contact = Image.new('RGB', (1000, 330*((len(group)+1)//2)), '#eeeeee')
            for i, im in enumerate(group):
                contact.paste(im, ((i%2)*500, (i//2)*330))
            contact.save(OUT / f'contact-{start//8+1}.jpg', quality=90)
    except Exception as e:
        print('Contact sheet:', e)
    print(json.dumps(report, indent=2))
if report['status'] != 'passed':
    raise SystemExit(1)
