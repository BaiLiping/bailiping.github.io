'use strict';
// Optional browser checks / PDF export. Uses Playwright or PLAYWRIGHT_MODULE.
const fs=require('node:fs');
const path=require('node:path');
const os=require('node:os');
const assert=require('node:assert/strict');
const {pathToFileURL}=require('node:url');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const out=process.env.EO_QA_DIR || fs.mkdtempSync(path.join(os.tmpdir(),'eo-four-scenarios-'));
const manifest=require('./build.json');
const content=require('./content.cjs');
fs.mkdirSync(out,{recursive:true});

(async()=>{
  const browser=await chromium.launch({
    headless:true,executablePath:process.env.CHROME_BIN || '/usr/bin/google-chrome',
    args:['--no-sandbox']
  });
  try {
    const page=await browser.newPage({viewport:{width:1440,height:900}});
    const errors=[],requests=[],issues=[];
    page.on('pageerror',e=>errors.push(e.message));
    page.on('request',r=>{if(/^https?:/.test(r.url()))requests.push(r.url());});
    await page.goto(pathToFileURL(path.join(__dirname,'index.html')).href);
    await page.evaluate(()=>document.fonts.ready);
    assert.equal(await page.evaluate(()=>EOSlides.count),content.slides.length);
    assert.equal(await page.locator('meta[name="eo-source-sha256"]').getAttribute('content'),manifest.articleSHA256);
    assert.equal(await page.locator('.source-figure img').count(),manifest.embeddedFigures.length);
    assert.equal(await page.locator('mjx-merror,[data-mjx-error]').count(),0);
    assert(await page.evaluate(()=>[...document.images].every(i=>i.complete&&i.naturalWidth>0)));
    for(let i=0;i<content.slides.length;i++) {
      await page.evaluate(i=>EOSlides.show(i),i);
      const problems=await page.evaluate(()=>{
        const s=document.querySelector('.slide.active'),scale=s.getBoundingClientRect().width/1280;
        const nodes=[s.querySelector('h1'),...s.querySelectorAll('.panel')];
        const found=[];
        for(const n of nodes){
          const r=n.getBoundingClientRect();
          if(n.scrollHeight>n.clientHeight+3 || n.scrollWidth>n.clientWidth+3)found.push(n.className||n.tagName);
          if(n.matches('.panel'))for(const child of n.children){
            const b=child.getBoundingClientRect();
            if(b.bottom>r.bottom+3*scale || b.top<r.top-3*scale || b.right>r.right+3*scale || b.left<r.left-3*scale)found.push(child.className||child.tagName);
          }
        }
        const title=s.querySelector('h1').getBoundingClientRect();
        const body=s.querySelector('.slide-content').getBoundingClientRect();
        const footer=s.querySelector('.slide-footer').getBoundingClientRect();
        if(title.bottom>body.top+scale)found.push('title-overlaps-content');
        if(body.bottom>footer.top-3*scale)found.push('content-overlaps-footer');
        return [...new Set(found)];
      });
      if(problems.length)issues.push({slide:i+1,id:content.slides[i].id,problems});
      if(process.env.EO_QA_SCREENSHOTS==='all' || ['cover','s1-factors','s1-graph','s2-existence','s3-graph','s4-prior','s4-unaries','s4-result','s4-graph','grbp-graph'].includes(content.slides[i].id)){
        await page.screenshot({path:path.join(out,String(i+1).padStart(2,'0')+'-'+content.slides[i].id+'.png')});
      }
    }
    await page.evaluate(()=>EOSlides.show(0));
    await page.keyboard.press('ArrowRight');assert.equal(await page.evaluate(()=>EOSlides.current),1);
    await page.locator('.slide.active a[href="#grbp-preclustering"]').click();
    assert.equal(await page.locator('.slide.active').getAttribute('id'),'grbp-preclustering');
    await page.keyboard.press('PageUp');
    assert.equal(await page.locator('.slide.active').getAttribute('id'),'s4-graph');
    await page.keyboard.press('PageDown');
    assert.equal(await page.locator('.slide.active').getAttribute('id'),'grbp-preclustering');
    await page.evaluate(()=>EOSlides.show(1));
    await page.locator('.slide.active a[href="#s4-anchor"]').click();
    assert.equal(await page.locator('.slide.active').getAttribute('id'),'s4-anchor');
    await page.locator('[data-notes]').click();assert(await page.locator('.notes-dialog').isVisible());
    await page.keyboard.press('Escape');assert(!(await page.locator('.notes-dialog').isVisible()));
    assert(await page.locator('[data-notes]').evaluate(e=>e===document.activeElement));
    await page.locator('[data-overview]').click();assert(await page.locator('.overview-dialog').isVisible());
    await page.keyboard.press('ArrowRight');
    assert.equal(await page.locator('.slide.active').getAttribute('id'),'s4-anchor');
    await page.locator('.overview-grid button').nth(0).click();assert.equal(await page.evaluate(()=>EOSlides.current),0);
    assert(await page.locator('[data-overview]').evaluate(e=>e===document.activeElement));
    await page.keyboard.press('End');assert.equal(await page.evaluate(()=>EOSlides.current),content.slides.length-1);
    await page.keyboard.press('Home');assert.equal(await page.evaluate(()=>EOSlides.current),0);
    await page.setViewportSize({width:390,height:844});
    await page.screenshot({path:path.join(out,'mobile.png')});
    assert(await page.evaluate(()=>document.body.scrollWidth<=innerWidth+1));
    await page.locator('[data-overview]').click();
    assert(await page.evaluate(()=>{const d=document.querySelector('.overview-dialog');return d.scrollWidth<=d.clientWidth+2;}));
    await page.keyboard.press('Escape');
    await page.setViewportSize({width:1440,height:900});
    if(process.argv.includes('--pdf')) {
      // PDF links must resolve on the public site, not on the build machine.
      await page.evaluate(() => document.querySelectorAll('.slide a[href]').forEach(a => {
        a.href = new URL(a.getAttribute('href'), 'https://bailiping.com/eo-derivation-slides/').href;
      }));
      await page.pdf({path:path.join(__dirname,'slides.pdf'),printBackground:true,preferCSSPageSize:true,displayHeaderFooter:false});
    }
    const report={slidesChecked:content.slides.length,articleSHA256:manifest.articleSHA256,embeddedGraphs:manifest.embeddedFigures.length,
      slideLayoutIssues:issues,javascriptErrors:errors,externalRuntimeRequests:requests,
      navigation:true,sectionLinks:true,notes:true,overview:true,focusReturn:true,mobileWidth:true,
      pdfExported:process.argv.includes('--pdf'),success:!issues.length&&!errors.length&&!requests.length};
    fs.writeFileSync(path.join(__dirname,'qa-results.json'),JSON.stringify(report,null,2)+'\n');
    console.log(JSON.stringify(report,null,2));console.log('Screenshots: '+out);
    assert.deepEqual(issues,[]);assert.deepEqual(errors,[]);assert.deepEqual(requests,[]);
  } finally {
    await browser.close();
  }
})().catch(e=>{console.error(e);process.exitCode=1;});
