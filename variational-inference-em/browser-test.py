"""Validate the built deck served locally on port 8765."""
import json
import math
import os
import traceback
from pathlib import Path
from playwright.sync_api import sync_playwright
from PIL import Image, ImageDraw

out = Path('viem-validation')
out.mkdir(exist_ok=True)
url = os.environ.get('VIEM_TEST_BASE', 'http://127.0.0.1:8765') + '/variational-inference-em/'
report = {'status': 'running', 'slides': [], 'labs': [], 'errors': []}

def close(a, b, tol=1e-7):
    assert math.isfinite(a) and math.isfinite(b) and abs(a-b) <= tol, (a, b)

def screenshot(page, name):
    page.screenshot(path=str(out / (name+'.png')), full_page=True)

try:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={'width':1440, 'height':900})
        page.on('pageerror', lambda e: report['errors'].append(str(e)))
        page.goto(url, wait_until='networkidle', timeout=60000)
        page.wait_for_function('window.bento && bento.doc.slides.length === 26', timeout=60000)
        page.wait_for_function('window.MathJax && MathJax.startup.document', timeout=60000)
        ids = page.evaluate('bento.doc.slides.map(s=>s.id)')
        for i, sid in enumerate(ids):
            page.goto(url+'#/'+str(i), wait_until='domcontentloaded')
            page.wait_for_timeout(700)
            root = page.locator('.bento-slide[data-slide-id="'+sid+'"]').first
            root.wait_for(state='visible', timeout=15000)
            root.scroll_into_view_if_needed()
            page.wait_for_timeout(350)
            errors = root.locator('mjx-merror, [data-mjx-error]').count()
            maths = root.locator('mjx-container').count()
            assert errors == 0, sid
            if root.locator('.math-tex').count():
                assert maths > 0, 'Math not typeset: '+sid
            if sid in ['comparison', 'summary']:
                assert root.locator('table.bento-table').count() == 1
                assert root.locator('table.bento-table td').count() == 15
            report['slides'].append({'id':sid,'visible':True,'mathErrors':errors,'mathContainers':maths})
            screenshot(page, f'slide-{i+1:02d}-{sid}')
        for mode in ['meanfield','em']:
            page.goto(url+'live/?demo='+mode+'&embed=region', wait_until='networkidle')
            page.wait_for_function('window.VILab')
            page.set_viewport_size({'width':1136,'height':475})
            if mode == 'meanfield':
                page.locator('#reset').click()
                before = page.evaluate('VILab.getState()')
                page.locator('#step').click()
                after = page.evaluate('VILab.getState()')
                close(after['mean'][0],before['rho']*before['mean'][1])
                close(after['mean'][1],before['mean'][1])
                page.locator('#fit').click()
                close(float(page.locator('#mf-best').inner_text()),-.5*math.log(.36),.001)
                page.locator('#independent').click()
                close(float(page.locator('#mf-kl').inner_text()),0,.001)
                page.locator('#correlated').click()
                page.locator('#reset').click()
                page.locator('#step').click()
            else:
                before = page.evaluate('VILab.getState()')
                page.locator('#step').click()
                e = page.evaluate('VILab.getState()')
                close(before['history'][-1]['likelihood'],e['history'][-1]['likelihood'])
                close(e['history'][-1]['gap'],0)
                page.locator('#step').click()
                m = page.evaluate('VILab.getState()')
                assert m['history'][-1]['likelihood'] >= e['history'][-1]['likelihood']-1e-7
                assert m['q'] == e['q']
                page.locator('#initialization').select_option('identical')
                page.locator('#step').click()
                page.locator('#step').click()
                s = page.evaluate('VILab.getState()')
                close(s['model']['mean'][0],s['model']['mean'][1])
                page.locator('#initialization').select_option('spread')
                page.locator('#learn').check()
                for _ in range(12):
                    page.locator('#step').click()
                assert min(page.evaluate('VILab.getState()')['model']['variance']) >= .09
            screenshot(page,'lab-'+mode+'-desktop')
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth+1')
            page.set_viewport_size({'width':390,'height':844})
            page.wait_for_timeout(200)
            screenshot(page,'lab-'+mode+'-mobile')
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth+1')
            report['labs'].append({'mode':mode,'controls':'passed','desktop':'passed','mobile':'passed'})
        page.set_viewport_size({'width':1440,'height':900})
        for i, mode in [(9,'meanfield'),(17,'em')]:
            page.goto(url+'#/'+str(i),wait_until='networkidle')
            selector = 'iframe[data-ready="true"][data-bento-src*="demo='+mode+'&"]'
            page.locator(selector).wait_for(state='visible',timeout=30000)
            page.frame_locator(selector).locator('#step').click()
            screenshot(page,'embedded-'+mode)
        assert not report['errors'],report['errors']
        report['status']='passed'
        browser.close()
except Exception as exc:
    report['status']='failed'
    report['exception']=str(exc)
    report['traceback']=traceback.format_exc()
finally:
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    thumbnails=[]
    for fn in sorted(out.glob('slide-*.png')):
        im=Image.open(fn).convert('RGB')
        im.thumbnail((480,300))
        tile=Image.new('RGB',(500,330),'white')
        tile.paste(im,((500-im.width)//2,8))
        ImageDraw.Draw(tile).text((10,311),fn.stem,fill='black')
        thumbnails.append(tile)
    for start in range(0,len(thumbnails),8):
        group=thumbnails[start:start+8]
        sheet=Image.new('RGB',(1000,330*((len(group)+1)//2)),'#eeeeee')
        for i,im in enumerate(group):
            sheet.paste(im,((i%2)*500,(i//2)*330))
        sheet.save(out/f'contact-{start//8+1}.jpg',quality=90)
if report['status']!='passed':
    raise SystemExit(1)
