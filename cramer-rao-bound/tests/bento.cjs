// Presentation integration, compact-lab lifecycle, and numerical UI regressions.
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
async function range(frame,id,value){await frame.locator('#'+id).fill(String(value));await frame.locator('#'+id).dispatchEvent('input');}
(async()=>{
 fs.mkdirSync(out,{recursive:true});
 const browser=await chromium.launch({headless:true,...(process.env.CHROMIUM_EXECUTABLE?{executablePath:process.env.CHROMIUM_EXECUTABLE}:{})});
 const errors=[];
 try {
  const page=await browser.newPage({viewport:{width:1440,height:900}});
  page.on('pageerror',e=>errors.push(String(e)));
  await page.goto(base,{waitUntil:'networkidle'});
  await page.waitForFunction(()=>window.bento?.doc);
  check(await page.evaluate(()=>window.bento.doc.slides.length)===23,'23 native Bento slides');
  check(await page.locator('iframe').count()===0,'No lab loaded before a request');
  for(const [width,height] of [[1440,900],[390,844]]){
   await page.setViewportSize({width,height});
   for(let i=0;i<23;i++){
    await go(page,i);
    const slide=page.locator('section.present .bento-slide');
    await slide.locator('img').evaluateAll(images=>Promise.all(images.map(im=>im.decode())));
    const overflow=await slide.evaluate(s=>[...s.querySelectorAll('.bento-text-inner')].filter(e=>{const p=e.closest('[data-el-id]');return e.scrollHeight>p.clientHeight+4||e.scrollWidth>p.clientWidth+4;}).map(e=>e.closest('[data-el-id]').dataset.elId));
    check(!overflow.length,`Slide ${i+1} at ${width}px has text overflow: ${overflow}`);
    check(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+2),`No page overflow at ${width}`);
    if(width===1440||[0,8,11,16,22].includes(i))await page.screenshot({path:path.join(out,`slide-${i+1}-${width}.png`)});
   }
   for(const [index,lab] of [[8,'gaussian-lab'],[11,'bias-lab'],[16,'geometry-lab']]){
    await go(page,index);
    const trigger=page.locator('section.present .try-live');
    await trigger.focus();await trigger.press('Enter');
    await page.waitForFunction(()=>document.querySelector('#crb-dialog').open&&getComputedStyle(document.querySelector('#demo-frame iframe')).visibility==='visible');
    const frame=await page.locator('#demo-frame iframe').elementHandle().then(h=>h.contentFrame());
    const state=()=>frame.evaluate(()=>CRBDeck.state());
    check((await state()).current===lab,'Correct compact lab');
    check(await frame.locator('[data-lab-panel]').count()===3,'Only compact experiment panels');
    check(await frame.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+2),'Lab has no horizontal overflow');
    if(lab==='gaussian-lab'){
     const initial=(await state()).mc;
     close(initial.variance,.256984597853496);
     await range(frame,'mc-n',64);check(await frame.locator('#mc-crb').innerText()==='0.0625','n64 CRB');
     await range(frame,'mc-sigma',4);await frame.locator('#mc-estimator').selectOption('first');
     check(await frame.locator('#mc-exact').innerText()==='16.0000','Discarding data changes estimator variance');
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
     const box=await page.locator('#demo-frame iframe').boundingBox();
     await page.mouse.move(box.x+local[0].x,box.y+local[0].y);await page.mouse.down();await page.mouse.move(box.x+local[1].x,box.y+local[1].y,{steps:6});await page.mouse.up();
     close((await state()).geometry.target[0],1,.04);close((await state()).geometry.target[1],1,.04);
     await frame.locator('[data-point="4"]').focus();await frame.locator('[data-point="4"]').press('ArrowRight');
     close((await state()).geometry.target[0],1.1,.04);
     check(await page.locator('section.present .bento-slide').getAttribute('data-slide-id')==='s-geometry-lab','Editing does not navigate parent');
    }
    await page.screenshot({path:path.join(out,`${lab}-${width}.png`)});
    if(lab==='bias-lab')await frame.locator('#bias-alpha').press('Escape');else await page.locator('#demo-back').click();
    await page.waitForFunction(()=>!document.querySelector('#crb-dialog').open);
    check(await page.locator('iframe').count()===0,'Close removes the iframe');
    check(await trigger.evaluate(e=>document.activeElement===e),'Close restores trigger focus');
   }
  }
  // Exit while the lab response is delayed. A late response must not remount it.
  await page.setViewportSize({width:1440,height:900});await go(page,8);
  let release;const gate=new Promise(resolve=>release=resolve);
  const delayed=async route=>{await gate;await route.continue().catch(()=>{});};
  await page.route('**/live/**',delayed);
  await page.locator('section.present .try-live').click();
  check(await page.locator('#demo-loading').isVisible(),'Explicit loading state');
  check(await page.locator('#demo-frame iframe').evaluate(e=>getComputedStyle(e).visibility==='hidden'),'No iframe flash');
  await page.locator('#demo-back').click();release();await page.unroute('**/live/**',delayed);
  check(await page.locator('iframe').count()===0,'Delayed load is torn down');
  await go(page,0);await page.locator('body').click({position:{x:10,y:400}});await page.keyboard.press('ArrowRight');
  await page.waitForFunction(()=>document.querySelector('section.present .bento-slide')?.dataset.slideId==='s-repeat');
  check(true,'Bento arrow navigation');
  await page.goto(base+'#geometry-lab');
  await page.waitForFunction(()=>document.querySelector('section.present .bento-slide')?.dataset.slideId==='s-geometry-lab');
  check(true,'Legacy semantic links preserved');
  // The guide remains readable on a phone, independent of slide scaling.
  await page.setViewportSize({width:390,height:844});await page.goto(base+'guide/');
  await page.waitForFunction(()=>window.CRBDeck);
  check(await page.locator('body').evaluate(e=>e.classList.contains('reading')),'Full guide opens in reading view');
  check(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+2),'Full guide mobile width');
  // Offline normal slides and the local-file lab handshake require no network.
  const offline=await browser.newContext({offline:true});const filePage=await offline.newPage();
  filePage.on('pageerror',e=>errors.push(String(e)));
  await filePage.goto(pathToFileURL(path.join(root,'index.html')).href);
  await filePage.waitForFunction(()=>window.bento?.doc);
  await filePage.locator('img').evaluateAll(images=>Promise.all(images.map(im=>im.decode())));
  await go(filePage,8);await filePage.locator('section.present .try-live').click();
  await filePage.waitForFunction(()=>getComputedStyle(document.querySelector('#demo-frame iframe')).visibility==='visible');
  check(true,'Offline deck and lab load from the source package');
  await filePage.locator('#demo-back').click();await offline.close();
  check(errors.length===0,`Browser errors: ${errors}`);
  fs.writeFileSync(path.join(out,'report.json'),JSON.stringify({checks,slides:23,labs:3,viewports:[1440,390],errors},null,2));
  console.log(`${checks} presentation and lab checks passed; zero browser errors.`);
 }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exit(1)});
