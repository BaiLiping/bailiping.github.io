// Presentation integration, inline-lab lifecycle, and numerical UI regressions.
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const {pathToFileURL}=require('node:url');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const root=path.resolve(__dirname,'..');
const base=process.env.CRB_BASE_URL||'http://127.0.0.1:8772/cramer-rao-bound/';
const out=process.env.CRB_QA_DIR||'/tmp/crb-bento-qa';
let checks=0;
function check(value,message){checks++;assert.ok(value,message);}
function close(a,b,tol=1e-9){check(Math.abs(a-b)<tol,`${a} != ${b}`);}
async function go(page,index){
 await page.evaluate(i=>{location.hash='#/'+i;},index);
 await page.waitForFunction(i=>document.querySelector('section.present .bento-slide')?.dataset.slideId===window.bento.doc.slides[i].id,index);
}
async function labFrame(page){
 await page.waitForSelector('.crb-lab-frame:visible');
 return page.locator('.crb-lab-frame').elementHandle().then(h=>h.contentFrame());
}
async function range(frame,id,value){await frame.locator('#'+id).fill(String(value));await frame.locator('#'+id).dispatchEvent('input');}
(async()=>{
 fs.mkdirSync(out,{recursive:true});
 const browser=await chromium.launch({headless:true,...(process.env.CHROMIUM_EXECUTABLE?{executablePath:process.env.CHROMIUM_EXECUTABLE}:{})});
 const errors=[];
 try {
  const page=await browser.newPage({viewport:{width:1440,height:900}});
  page.on('pageerror',e=>errors.push(String(e)));
  page.on('response',r=>{if(r.status()>=400)errors.push(`${r.status()} ${r.url()}`);});
  await page.goto(base,{waitUntil:'networkidle'});
  await page.waitForFunction(()=>window.bento?.doc);
  check(await page.evaluate(()=>window.bento.doc.slides.length)===23,'23 native Bento slides');
  check(await page.locator('iframe').count()===0,'No lab loaded on a normal slide');
  check(await page.locator('.try-live,#crb-dialog').count()===0,'No launch button or modal remains');
  for(const [width,height] of [[1440,900],[390,844]]){
   await page.setViewportSize({width,height});
   for(let i=0;i<23;i++){
    await go(page,i);
    const slide=page.locator('section.present .bento-slide');
    await slide.locator('img').evaluateAll(images=>Promise.all(images.map(im=>im.decode())));
    const overflow=await slide.evaluate(s=>[...s.querySelectorAll('.bento-text-inner')].filter(e=>{const p=e.closest('[data-el-id]');return e.scrollHeight>p.clientHeight+4||e.scrollWidth>p.clientWidth+4;}).map(e=>e.closest('[data-el-id]').dataset.elId));
    check(!overflow.length,`Slide ${i+1} at ${width}px has text overflow: ${overflow}`);
    check(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+2),`No page overflow at ${width}`);
    if([8,11,16].includes(i))await labFrame(page);
    else check(await page.locator('iframe').count()===0,'Normal slide unloads the experiment');
    if(width===1440||[0,8,11,16,22].includes(i))await page.screenshot({path:path.join(out,`slide-${i+1}-${width}.png`)});
   }
   for(const [index,lab] of [[8,'gaussian-lab'],[11,'bias-lab'],[16,'geometry-lab']]){
    await go(page,index);
    const frame=await labFrame(page);
    const state=()=>frame.evaluate(()=>CRBDeck.state());
    check(await page.locator('iframe').count()===1,'Only the active experiment is mounted');
    check((await state()).current===lab,'Correct compact lab opens automatically');
    check(await frame.locator('[data-lab-panel]:visible').count()===1,'Only the selected experiment is visible');
    check(await frame.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+2),'Lab has no horizontal overflow');
    if(width===1440)check(await frame.evaluate(()=>document.documentElement.scrollHeight<=innerHeight+2),'Desktop lab fits the slide');
    else check(await page.locator('.crb-mobile-lab').isVisible(),'Mobile slide uses a readable layout');
    if(lab==='gaussian-lab'){
     const initial=(await state()).mc;
     close(initial.variance,.256984597853496);
     await range(frame,'mc-n',64);check(await frame.locator('#mc-crb').innerText()==='0.0625','n64 CRB');
     await range(frame,'mc-sigma',4);await frame.locator('#mc-estimator').selectOption('first');
     check(await frame.locator('#mc-exact').innerText()==='16.0000','Discarding data changes estimator variance');
     await frame.locator('#mc-n').press('ArrowRight');
     check((await state()).mc.n===65,'Slider arrow keys adjust the experiment');
     check(new URL(page.url()).hash==='#/8','Slider keys do not navigate the deck');
     await frame.locator('#mc-reset').click();assert.deepEqual((await state()).mc,initial);
    }else if(lab==='bias-lab'){
     close((await state()).bias.mse,.10043125);
     await frame.locator('[data-bias-preset="far"]').click();close((await state()).bias.mse,.885625);
     await frame.locator('[data-bias-preset="constant"]').click();close((await state()).bias.variance,0);
     check((await frame.locator('#bias-chart').innerText()).includes('Point mass'),'Zero factor is a point mass');
     await frame.locator('#bias-reset').click();
    }else{
     close((await state()).geometry.peb,.625);
     await frame.locator('[data-geom-preset="cluster"]').click();const known=(await state()).geometry.peb;
     await frame.locator('#geom-bias').check();check((await state()).geometry.peb>=known,'Nuisance offset loses information');
     await frame.locator('[data-geom-preset="collinear"]').click();check((await state()).geometry.rank===1,'Singular geometry flagged');
     check((await state()).geometry.covariance===null,'No finite inverse at rank deficiency');
     await frame.locator('#geom-reset').click();await range(frame,'geom-x',-4);await range(frame,'geom-y',-3);
     check((await state()).geometry.invalid,'Coincidence flagged');await frame.locator('#geom-reset').click();
     await frame.locator('#geometry-chart').scrollIntoViewIfNeeded();
     const local=await frame.locator('#geometry-chart svg').evaluate(e=>{const matrix=e.getScreenCTM();return [[380,180],[420,140]].map(p=>{const q=new DOMPoint(...p).matrixTransform(matrix);return {x:q.x,y:q.y};});});
     const box=await page.locator('.crb-lab-frame').boundingBox();const scale=box.width/await frame.evaluate(()=>innerWidth);
     await page.mouse.move(box.x+local[0].x*scale,box.y+local[0].y*scale);await page.mouse.down();await page.mouse.move(box.x+local[1].x*scale,box.y+local[1].y*scale,{steps:6});await page.mouse.up();
     close((await state()).geometry.target[0],1,.04);close((await state()).geometry.target[1],1,.04);
     await frame.locator('[data-point="4"]').focus();await frame.locator('[data-point="4"]').press('ArrowRight');
     close((await state()).geometry.target[0],1.1,.04);
     check(new URL(page.url()).hash==='#/16','Editing geometry does not navigate parent');
    }
    await page.screenshot({path:path.join(out,`${lab}-${width}.png`)});
    // Escape from an input opens the ordinary deck overview and unloads the lab.
    await frame.locator(lab==='gaussian-lab'?'#mc-n':lab==='bias-lab'?'#bias-alpha':'#geom-x').press('Escape');
    await page.waitForSelector('.reveal.overview');
    check(await page.locator('iframe').count()===0,'Overview removes the iframe');
    check(await page.locator('.reveal').evaluate(e=>document.activeElement===e),'Overview restores focus to deck');
    await page.keyboard.press('Escape');await labFrame(page);
    if(width===390){
     await page.getByRole('button',{name:'Next →',exact:true}).click();
    }else{
     const resumed=await labFrame(page);await resumed.locator('.plot-card:visible').click({position:{x:5,y:5}});await page.keyboard.press('ArrowRight');
    }
    await page.waitForFunction(i=>document.querySelector('section.present .bento-slide')?.dataset.slideId===window.bento.doc.slides[i+1].id,index);
    check(await page.locator('iframe').count()===0,'Navigation removes the iframe');
   }
  }
  // Exit while a lab response is delayed; a late response must not remount it.
  await page.setViewportSize({width:1440,height:900});await go(page,0);
  let release;const gate=new Promise(resolve=>release=resolve);
  const delayed=async route=>{await gate;await route.continue().catch(()=>{});};
  await page.route('**/live/**',delayed);await go(page,8);
  check(await page.locator('.crb-lab-loading').isVisible(),'Explicit loading state');
  check(await page.locator('.crb-lab-frame').evaluate(e=>e.hidden),'No iframe flash');
  await go(page,9);release();await page.unroute('**/live/**',delayed);
  check(await page.locator('iframe').count()===0,'Delayed load is torn down');
  await go(page,17);await page.locator('section.present a[href="#/16"]').click();await labFrame(page);
  check(new URL(page.url()).pathname==='/cramer-rao-bound/','Geometry shortcut stays in this deck');
  await page.goto(base+'#geometry-lab');await labFrame(page);
  check(await page.locator('section.present .bento-slide').getAttribute('data-slide-id')==='s-geometry-lab','Legacy semantic links preserved');
  await page.goto(base+'?static=1#/8');await page.waitForFunction(()=>window.bento?.doc);
  check(await page.locator('iframe').count()===0,'Static export retains the figures without live controls');
  // The full guide and standalone labs remain usable independently.
  await page.setViewportSize({width:390,height:844});await page.goto(base+'guide/');
  await page.waitForFunction(()=>window.CRBDeck);
  check(await page.locator('body').evaluate(e=>e.classList.contains('reading')),'Full guide opens in reading view');
  check(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+2),'Full guide mobile width');
  await page.goto(base+'live/?lab=bias-lab');await page.waitForFunction(()=>window.CRBDeck);
  check(await page.locator('.lab-nav').isVisible(),'Standalone lab navigation preserved');
  // Offline normal slides and local-file inline labs require no network.
  const offline=await browser.newContext({offline:true});const filePage=await offline.newPage();
  filePage.on('pageerror',e=>errors.push(String(e)));
  await filePage.goto(pathToFileURL(path.join(root,'index.html')).href);
  await filePage.waitForFunction(()=>window.bento?.doc);
  await filePage.locator('img').evaluateAll(images=>Promise.all(images.map(im=>im.decode())));
  await go(filePage,8);await labFrame(filePage);
  check(true,'Offline deck automatically loads its inline lab');
  await go(filePage,9);check(await filePage.locator('iframe').count()===0,'Offline navigation unloads lab');await offline.close();
  check(errors.length===0,`Browser errors: ${errors}`);
  fs.writeFileSync(path.join(out,'report.json'),JSON.stringify({checks,slides:23,labs:3,viewports:[1440,390],errors},null,2));
  console.log(`${checks} presentation and lab checks passed; zero browser errors.`);
 }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exit(1)});
