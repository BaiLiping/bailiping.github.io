"""Browser regression test for real URLs, sandboxed iframes, and lesson controls.
Run from the repository root after installing Playwright and its Chromium browser.
"""
import json
import os
import re
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'bp-pmbm-browser-report'
OUT.mkdir(exist_ok=True)
class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *_args):
        pass
server = ThreadingHTTPServer(('127.0.0.1', 0), partial(QuietHandler, directory=str(ROOT)))
threading.Thread(target=server.serve_forever, daemon=True).start()
base = f'http://127.0.0.1:{server.server_port}'
errors = []
report = {'article': False, 'live_modes': [], 'embedded_modes': [], 'slides': []}

def close_matrix(a, b):
    assert len(a) == len(b)
    for ra, rb in zip(a, b):
        assert len(ra) == len(rb)
        assert max(abs(x-y) for x,y in zip(ra,rb)) < 1e-10

def watch(page):
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.set_default_timeout(20000)
    return page

try:
    with sync_playwright() as playwright:
        executable = os.environ.get('CHROMIUM_PATH')
        browser = playwright.chromium.launch(headless=True, **({'executable_path': executable} if executable else {}))
        page = watch(browser.new_page(viewport={'width':1440, 'height':1000}, reduced_motion='reduce'))
        page.goto(base+'/bp-vs-pmbm/')
        page.wait_for_function('window.BPAssociationAudit !== undefined')
        reference = page.evaluate('BPAssociationAudit.snapshot()')
        assert reference['converged'] and reference['iterations'] == 30
        assert reference['graph']['cycles'] == 2
        page.locator('#bpStep').click()
        assert 'Measurement beliefs appear' not in page.locator('#bjheat').inner_text()
        page.locator('#bpEnd').click()
        page.locator('#kSlider').evaluate("e=>e.value='1'")
        page.locator('#kSlider').dispatch_event('input')
        assert page.locator('#topkheat td').count() > 0
        page.locator('#treePreset').click()
        assert page.evaluate('BPAssociationAudit.snapshot().graph.acyclic')
        page.locator('#gateToggle').uncheck()
        assert page.evaluate('BPAssociationAudit.snapshot().graph.edges') == 12
        page.locator('#symmetricPreset').click()
        assert not page.evaluate('BPAssociationAudit.snapshot().graph.acyclic')
        page.screenshot(path=str(OUT/'article.png'), full_page=True)
        report['article'] = True
        page.close()
        for mode in ['assignment', 'bp', 'hypotheses']:
            page = watch(browser.new_page(viewport={'width':1136,'height':435}, reduced_motion='reduce'))
            page.goto(base+f'/bp-vs-pmbm-slides/live/?demo={mode}&embed=region')
            page.wait_for_function('window.BPAssociationAudit !== undefined')
            state = page.evaluate('BPAssociationAudit.snapshot()')
            close_matrix(state['L'], reference['L'])
            close_matrix(state['bp'], reference['bp'])
            close_matrix(state['exact'], reference['exact'])
            if mode == 'bp':
                page.locator('#step').click()
                assert page.evaluate('BPAssociationAudit.snapshot().iteration') == 1
                page.locator('#end').click()
            if mode == 'hypotheses':
                page.locator('#keep').evaluate("e=>e.value='1'")
                page.locator('#keep').dispatch_event('input')
                assert page.evaluate('BPAssociationAudit.snapshot().k') == 1
            dimensions=page.evaluate('({w:document.documentElement.scrollWidth,h:document.documentElement.scrollHeight})')
            assert dimensions['w'] <= 1138 and dimensions['h'] <= 437, (mode, dimensions)
            if mode == 'assignment':
                page.locator('#pd').evaluate("e=>e.value='0.99'")
                page.locator('#pd').dispatch_event('input')
                assert page.evaluate('BPAssociationAudit.snapshot().scene.PD') == .99
                page.locator('#gate').uncheck()
                assert page.evaluate('BPAssociationAudit.snapshot().graph.edges') == 12
                page.locator('#sceneReset').click()
                close_matrix(page.evaluate('BPAssociationAudit.snapshot().L'), reference['L'])
            elif mode == 'bp':
                page.locator('#reset').click()
                page.locator('#play').click()
                page.wait_for_function('BPAssociationAudit.snapshot().iteration > 0')
                page.locator('#play').click()
                assert not page.evaluate('BPAssociationAudit.snapshot().playing')
                page.locator('#end').click()
                assert page.locator('#bpGap').inner_text() == '6.93'
                page.locator('#compare').check()
                assert page.locator('#bpTable small').count() == 15
                page.locator('#edge').select_option('0,1')
                assert page.evaluate('BPAssociationAudit.snapshot().selected') == [0, 1]
            else:
                page.locator('#keepAll').click()
                assert page.locator('#mass').inner_text() == '100.0%'
                assert page.locator('#topGap').inner_text() == '0.00'
                page.locator('#hypReset').click()
                assert page.evaluate('BPAssociationAudit.snapshot().k') == 5
            for preset in ['separated', 'symmetric']:
                page.locator('#preset').select_option(preset)
                assert page.evaluate('BPAssociationAudit.snapshot().graph.acyclic') == (preset == 'separated')
            page.locator('#preset').select_option('default')
            page.screenshot(path=str(OUT/f'live-{mode}.png'))
            report['live_modes'].append(mode)
            page.close()
        html=(ROOT/'bp-vs-pmbm-slides/index.html').read_text()
        doc=json.loads(re.search(r'id="bento-doc">\s*([\s\S]*?)</script>', html).group(1))
        page=watch(browser.new_page(viewport={'width':1440,'height':900}, reduced_motion='reduce'))
        page.goto(base+'/bp-vs-pmbm-slides/')
        for index, slide in enumerate(doc['slides']):
            selector=f'.slides > section.present [data-slide-id="{slide["id"]}"]'
            root=page.locator(selector).first
            root.wait_for(state='visible')
            if slide['id'] in ['s-weights-live','s-bp-messages-live','s-pruning-live']:
                iframe=root.locator('iframe[data-ready="true"]')
                iframe.wait_for(state='visible')
                frame=iframe.element_handle().content_frame()
                frame.wait_for_function('window.BPAssociationAudit !== undefined')
                state=frame.evaluate('BPAssociationAudit.snapshot()')
                close_matrix(state['L'],reference['L'])
                close_matrix(state['bp'],reference['bp'])
                report['embedded_modes'].append(state['mode'])
            page.wait_for_timeout(250)
            page.wait_for_function("!document.querySelector('section.present .math-tex:not([data-math-rendered])')")
            assert not page.locator('section.present [data-mjx-error]').count()
            overflow=root.evaluate('''e=>Array.from(e.querySelectorAll('[data-el-id]')).filter(t=>t.clientHeight>0&&t.scrollHeight>t.clientHeight+4).map(t=>t.dataset.elId)''')
            assert not overflow, (slide['id'], overflow)
            page.screenshot(path=str(OUT/f'slide-{index+1:02}.png'))
            report['slides'].append(slide['id'])
            if index+1<len(doc['slides']):
                page.keyboard.press('ArrowRight')
        assert len(report['slides'])==17
        assert report['slides'][-1]=='s-extensions'
        assert report['embedded_modes']==['assignment','bp','hypotheses']
        assert not errors, errors
        page.close()
        browser.close()
finally:
    server.shutdown()
    report['errors']=errors
    (OUT/'report.json').write_text(json.dumps(report,indent=2))
# Deployment is independent of CI. Record current public bytes without making
# a local regression check depend on CDN propagation or external availability.
from urllib.request import Request, urlopen
import hashlib
report['public_files'] = {}
for relative in ['bp-vs-pmbm/index.html', 'bp-vs-pmbm/association-model.js', 'bp-vs-pmbm-slides/index.html']:
    url='https://bailiping.com/'+relative
    try:
        request=Request(url, headers={'Cache-Control':'no-cache', 'User-Agent':'BP-PMBM-Lesson-Audit'})
        with urlopen(request, timeout=15) as response:
            actual=response.read()
        expected=(ROOT/relative).read_bytes()
        report['public_files'][relative]={'matches_checkout':actual==expected,'sha256':hashlib.sha256(actual).hexdigest()}
    except Exception as error:
        report['public_files'][relative]={'verification_error':str(error)}
(OUT/'report.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
