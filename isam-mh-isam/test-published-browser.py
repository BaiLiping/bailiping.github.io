"""Exercise the new lab, the native deck, and sandbox integration before publishing."""
import json
import re
from pathlib import Path
from playwright.sync_api import sync_playwright

OUT = Path('isam-validation')
OUT.mkdir(exist_ok=True)
BASE = 'http://127.0.0.1:8765/isam-mh-isam/'
html = Path('isam-mh-isam/index.html').read_text()
doc = json.loads(re.search(r'<script type="application/bento\+json" id="bento-doc">(.*?)</script>', html, re.S).group(1))
assert len(doc['slides']) == 25
errors = []
report = {'slide_count': 25, 'views': []}
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={'width': 1136, 'height': 475})
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.goto(BASE + 'live/dp.html?embed=region')
    page.wait_for_function('window.isamDPState')
    assert page.locator('#dp-s').inner_text() == '1.00'
    page.click('#dp-example')
    assert page.locator('#dp-s').inner_text() == '3.00'
    assert page.locator('#dp-u').inner_text() == '2.00'
    for target in [-1, 0, 1, 2.5, 4, 5]:
        page.locator('#dp-z').evaluate('(el, z) => {el.value=z; el.dispatchEvent(new Event("input", {bubbles:true}));}', target)
        state = page.evaluate('window.isamDPState')
        assert abs(state['s'] - (1+2*target)/3) < 1e-10
        assert abs(state['u'] - (state['s']+1)/2) < 1e-10
    page.click('#dp-reset')
    assert page.locator('#dp-s').inner_text() == '1.00'
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    assert page.evaluate('document.documentElement.scrollHeight <= innerHeight + 1')
    page.screenshot(path=str(OUT/'dp-embedded.png'))
    for width,height in [(390,844),(1440,900)]:
        page.set_viewport_size({'width':width,'height':height})
        page.goto(BASE+'live/dp.html')
        page.wait_for_function('window.isamDPState')
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        page.click('#dp-example')
        page.screenshot(path=str(OUT/f'dp-{width}.png'),full_page=True)
        report['views'].append([width,height])
    page.set_viewport_size({'width':1280,'height':800})
    # Native Bento source rendering and MathJax must work on the revised slides.
    for sid in ['objective','linearize','isam']:
        page.goto(BASE+'#'+sid)
        page.wait_for_function('window.bento && document.querySelector(".bento-slide")', timeout=90000)
        page.wait_for_function('document.querySelector("mjx-container")', timeout=90000)
        page.wait_for_timeout(1200)
        assert not page.locator('mjx-merror, [data-mml-node="merror"]').count()
        page.screenshot(path=str(OUT/f'{sid}.png'))
    page.goto(BASE+'#dp-demo')
    page.wait_for_function('window.bento && document.querySelector("iframe")', timeout=90000)
    frame = page.frame_locator('iframe[src*="dp.html"]').first
    frame.locator('#dp-example').click(timeout=30000)
    assert frame.locator('#dp-s').inner_text() == '3.00'
    assert frame.locator('#dp-u').inner_text() == '2.00'
    page.screenshot(path=str(OUT/'dp-in-bento.png'))
    browser.close()
assert not errors, errors
report['javascript_errors']=errors
(OUT/'browser-results.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
