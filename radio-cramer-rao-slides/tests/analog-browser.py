"""Browser regression for the analog revision. Run from the repository root."""
import json, os, re, base64, zlib
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'radio-crb-validation';OUT.mkdir(exist_ok=True)
BASE=os.environ.get('RADIO_TEST_URL','http://127.0.0.1:8765')
INLINE=os.environ.get('RADIO_INLINE_TEST')=='1'
DECK=json.loads((ROOT/'radio-cramer-rao-slides/deck.json').read_text())
CHANGED=['overview','setup','project-setup','sweep-time','project-fim','beam-identifiability','sweep-coherence','project-codes','unknown-channel','unknown-clock','extensions','formula-sheet','references']
def inlined(path):
    s=path.read_text()
    def script(m):
        p=(path.parent/m.group(1)).resolve()
        return '<script>'+p.read_text().replace('</script','<\\/script')+'</script>' if p.exists() else ''
    s=re.sub(r'<script[^>]*?src="([^"]+)"[^>]*>\s*</script>',script,s)
    def css(m):
        p=(path.parent/m.group(1)).resolve()
        return '<style>'+p.read_text()+'</style>' if p.exists() else ''
    s=re.sub(r'<link[^>]*rel="stylesheet"[^>]*href="([^"]+)"[^>]*>',css,s)
    if 'id="bento-rt"' in s:
        css=zlib.decompress(base64.b64decode(re.search(r'<script id="bento-rt-css"[^>]*>(.*?)</script>',s,re.S)[1]),-15).decode()
        js=zlib.decompress(base64.b64decode(re.search(r'<script id="bento-rt"[^>]*>(.*?)</script>',s,re.S)[1]),-15).decode()
        s=re.sub(r'<script>\s*\(async \(\) => \{[\s\S]*?</script>',lambda m:'<style>'+css+'</style><script>'+js+'</script>',s)
        s=re.sub(r'(<script type="application/json" id="bento-inline-live-map">)[\s\S]*?(</script>)',r'\1[]\2',s)
    # In-memory storage supplies the opaque about:blank origin used only for offline screenshots.
    storage='<script>for(const name of ["localStorage","sessionStorage"]){const data={};Object.defineProperty(window,name,{value:{getItem:k=>data[k]??null,setItem:(k,v)=>{data[k]=v},removeItem:k=>delete data[k],clear:()=>{}},configurable:true});}</script>'
    return storage+s
with sync_playwright() as p:
    opts={'headless':True}
    if os.environ.get('RADIO_BROWSER'):opts['executable_path']=os.environ['RADIO_BROWSER']
    b=p.chromium.launch(**opts)
    page=b.new_page(viewport={'width':1440,'height':900});errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    if INLINE:page.set_content(inlined(ROOT/'radio-cramer-rao-slides/index.html'))
    else:page.goto(BASE+'/radio-cramer-rao-slides/')
    page.wait_for_selector('.bento-slide');page.wait_for_timeout(1800)
    assert page.locator('.bento-slide').count()==40
    overflow=[]
    for sid in CHANGED:
        i=next(i for i,s in enumerate(DECK['slides']) if s['id']==sid)
        page.evaluate('(i)=>location.hash="#/"+i',i);page.wait_for_timeout(180)
        root=page.locator('.bento-slide[data-slide-id="'+sid+'"]')
        assert root.is_visible(),sid
        assert root.locator('mjx-merror,[data-mml-node="merror"]').count()==0,sid
        bad=root.locator('.bento-el-text').evaluate_all('els=>els.map(e=>({id:e.dataset.elId,extra:e.firstElementChild.scrollHeight-e.clientHeight,wide:e.firstElementChild.scrollWidth-e.clientWidth})).filter(e=>e.extra>4||e.wide>5)')
        overflow.extend({'slide':sid,**x} for x in bad)
        page.screenshot(path=str(OUT/(sid+'.png')))
    refs=page.locator('.bento-slide[data-slide-id="references"] [data-link]').evaluate_all('es=>es.map(e=>e.dataset.link)')
    for ref in ['1711.08408','2301.10689','1702.01605','2002.04481']:assert any(ref in url for url in refs)
    # Check the actual sandboxed frame and keyboard navigation in normal CI.
    if not INLINE:
        page.goto(BASE+'/radio-cramer-rao-slides/#project-live')
        frame=page.frame_locator('iframe').first
        frame.locator('#bounds tr').first.wait_for()
        assert frame.locator('#bounds tr').count()==7
        frame.locator('#schedule').select_option('fixed')
        assert '3 / 7' in frame.locator('#rank').inner_text()
        page.screenshot(path=str(OUT/'embedded-lab.png'))
        frame.locator('#schedule').press('PageDown')
        page.wait_for_timeout(250)
        assert page.locator('.bento-slide[data-slide-id="unknown-channel"]').is_visible()
    # Standalone and narrow-screen laboratory controls.
    for width,height in [(1200,900),(390,844)]:
        page.close();page=b.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
        page.set_viewport_size({'width':width,'height':height})
        if INLINE:page.set_content(inlined(ROOT/'radio-cramer-rao-slides/analog/index.html'))
        else:page.goto(BASE+'/radio-cramer-rao-slides/analog/')
        page.locator('#reset').click();assert page.locator('#bounds tr').count()==7
        assert page.evaluate('window.latestAnalogResult.symbols')==3200
        page.locator('#schedule').select_option('fixed')
        assert page.evaluate('window.latestAnalogResult.rank')==3
        page.locator('#reset').click()
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
        page.screenshot(path=str(OUT/('lab-'+str(width)+'.png')),full_page=True)
    report={'slides':40,'changedSlidesChecked':len(CHANGED),'overflow':overflow,'pageErrors':errors,'offline':INLINE}
    (OUT/'browser-report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
    assert not errors,errors
    assert not overflow,overflow
    b.close()
