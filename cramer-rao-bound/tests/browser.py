#!/usr/bin/env python3
"""Browser regression tests. Requires Python playwright + an installed Chromium.
Set CHROMIUM_EXECUTABLE to use a system browser; otherwise uses Playwright's browser.
Tests load the self-contained document with set_content, so no server is needed.
"""
import os, json
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1]
checks=0

def check(test, message):
    global checks
    checks+=1
    assert test, message

def close(a,b,tol=1e-9):check(abs(a-b)<tol,f'{a} != {b}')

def overflow(page):
    return page.locator('.slide.active').evaluate('''s=>[...s.querySelectorAll('.card,.body')].filter(e=>e.scrollHeight>e.clientHeight+3||e.scrollWidth>e.clientWidth+3).map(e=>({tag:e.className,wh:[e.clientWidth,e.clientHeight,e.scrollWidth,e.scrollHeight],text:e.innerText.slice(-140)}))''')

with sync_playwright() as p:
    options={'headless':True,'args':['--no-sandbox']}
    if os.environ.get('CHROMIUM_EXECUTABLE'):options['executable_path']=os.environ['CHROMIUM_EXECUTABLE']
    browser=p.chromium.launch(**options)
    page=browser.new_page(viewport={'width':1440,'height':920},device_scale_factor=1)
    errors=[];requests=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.on('request',lambda r:requests.append(r.url))
    page.set_content((ROOT/'index.html').read_text());page.wait_for_function('window.CRBDeck')
    state=lambda:page.evaluate('CRBDeck.state()')
    go=lambda slug:page.evaluate('(s)=>CRBDeck.go(s)',slug)
    def slide_fit(label):
        bad=overflow(page);check(not bad, f'{label}: overflow {bad}')
    def range_set(selector,value):
        page.locator(selector).fill(str(value));page.locator(selector).dispatch_event('input')
    for i in range(23):
        go(i);slide_fit(f'slide {i+1}')
    go('gaussian-lab');s=state();close(s['mc']['variance'],.256984597853496)
    initial=s['mc'].copy()
    page.locator('#mc-one').click();check(state()['mc']['count']==2001,'+1 trial')
    page.locator('#mc-more').click();check(state()['mc']['count']==3001,'+1000 trials')
    range_set('#mc-n',64);check(state()['mc']['count']==2000,'n change replays')
    check(page.locator('#mc-crb').inner_text()=='0.0625','n64 CRB')
    range_set('#mc-sigma',4);check(page.locator('#mc-crb').inner_text()=='0.2500','sigma4 CRB')
    page.locator('#mc-estimator').select_option('first')
    check(page.locator('#mc-exact').inner_text()=='16.0000','first observation variance')
    slide_fit('Lab1 first observation')
    page.locator('#mc-seed').fill('2026');page.locator('#mc-seed').dispatch_event('change')
    check(state()['mc']['seed']==2026,'seed')
    page.locator('#mc-reset').click();check(state()['mc']==initial,'seeded reset exact')
    for _ in range(18):page.locator('#mc-more').click()
    check(state()['mc']['count']==20000,'trial cap');check(page.locator('#mc-one').is_disabled(),'trial cap disabled')
    page.locator('#mc-reset').click()
    go('bias-lab');close(state()['bias']['mse'],.10043125)
    for preset in ['far','unbiased','constant','near']:
        page.locator(f'[data-bias-preset="{preset}"]').click();slide_fit('Bias '+preset)
        if preset=='far':close(state()['bias']['mse'],.885625)
        if preset=='unbiased':close(state()['bias']['mse'],.25)
        if preset=='constant':
            close(state()['bias']['variance'],0);check('Point mass' in page.locator('#bias-chart').inner_text(),'alpha0 spike')
    for alpha in [0,.01,.99,1]:
        range_set('#bias-alpha',alpha)
        for mu in [-3,0,3]:range_set('#bias-mu',mu);slide_fit(f'bias extremes {alpha}, {mu}')
    page.locator('#bias-reset').click()
    go('geometry-lab');close(state()['geometry']['peb'],.625)
    page.locator('#geom-bias').check();close(state()['geometry']['peb'],.625)
    page.locator('[data-geom-preset="cluster"]').click();check(state()['geometry']['peb']>1,'cluster nuisance loss')
    slide_fit('geometry cluster+offset')
    page.locator('[data-geom-preset="collinear"]').click()
    check(state()['geometry']['rank']==1,'collinear singular');check(state()['geometry']['covariance'] is None,'no misleading inverse')
    slide_fit('geometry collinear')
    page.locator('#geom-reset').click()
    range_set('#geom-sigma',1.2);close(state()['geometry']['peb'],1.25)
    # Coordinates through controls, target coincidence with A1: explicit invalid guard.
    range_set('#geom-x',-4);range_set('#geom-y',-3)
    check(state()['geometry']['invalid'],'coincidence flag');slide_fit('geometry invalid')
    page.locator('#geom-reset').click()
    # Drag in actual SVG coordinates. Native inverse screen CTM must account for letterboxing.
    get_screen=lambda x,y:page.locator('#geometry-chart svg').evaluate('(e,p)=>{const q=new DOMPoint(...p).matrixTransform(e.getScreenCTM());return {x:q.x,y:q.y};}',[x,y])
    origin=get_screen(380,180);dest=get_screen(420,140)
    page.mouse.move(**origin);page.mouse.down();page.mouse.move(**dest,steps=6);page.mouse.up()
    close(state()['geometry']['target'][0],1,.03);close(state()['geometry']['target'][1],1,.03)
    # Arrow keys edit selected SVG point, rather than navigating the deck.
    page.locator('#geometry-chart [data-point="4"]').focus();page.keyboard.press('ArrowRight')
    close(state()['geometry']['target'][0],1.1,.03);check(state()['current']==16,'arrow does not change slide when moving a point')
    page.locator('#geom-reset').click()
    # Navigation, overview, notes, hashes, and preserved experiment state.
    page.locator('#next').click();check(state()['current']==17,'next')
    check(page.evaluate('location.hash')=='#nuisance','semantic hash')
    page.locator('#notes-button').click();check(page.locator('#notes').is_visible(),'notes show')
    page.locator('#notes-close').click()
    page.locator('#outline').click();check(page.locator('[data-toc]').count()==23,'overview count')
    page.locator('[data-toc="8"]').click();check(state()['current']==8,'overview navigation')
    check(state()['mc']==initial,'MC state persists')
    # Overview and normal-body keyboard navigation.
    page.locator('#mc-reset').click();page.locator('body').click(position={'x':15,'y':450})
    page.keyboard.press('End');check(state()['current']==22,'End key')
    page.keyboard.press('Home');check(state()['current']==0,'Home key')
    page.locator('#read-mode').click();check(page.locator('body').evaluate("e=>e.classList.contains('reading')"),'read view')
    page.locator('#read-mode').click()
    # Responsive reading layout on a narrow phone.
    phone=browser.new_page(viewport={'width':390,'height':844},device_scale_factor=1)
    phone.on('pageerror',lambda e:errors.append(str(e)))
    phone.set_content((ROOT/'index.html').read_text());phone.wait_for_function('window.CRBDeck')
    check(phone.locator('body').evaluate("e=>e.classList.contains('reading')"),'mobile starts reading view')
    width=phone.evaluate('({view:innerWidth,scroll:document.documentElement.scrollWidth})')
    check(width['scroll']<=width['view']+2,f'mobile horizontal overflow: {width}')
    phone.locator('#bias-alpha').fill('0.7');phone.locator('#bias-alpha').dispatch_event('input')
    close(phone.evaluate('CRBDeck.state().bias.alpha'),.7)
    check(not errors,f'JavaScript errors: {errors}')
    check(not requests,f'Unexpected network requests: {requests}')
    print(f'{checks} browser assertions passed; zero JavaScript errors and zero external requests.')
    browser.close()
