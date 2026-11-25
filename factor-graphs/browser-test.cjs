// Run with Playwright available through NODE_PATH; output stays outside the site.
const {chromium}=require('playwright'),assert=require('node:assert/strict'),fs=require('node:fs');
const BP=require('./model.js');
const base=process.env.BP_BASE_URL||'http://127.0.0.1:8793/factor-graphs/';
const out=process.env.BP_QA_DIR||'/tmp/factor-graphs-qa';fs.mkdirSync(out,{recursive:true});
(async()=>{
 const browser=await chromium.launch({headless:true,...(process.env.BP_CHROMIUM?{executablePath:process.env.BP_CHROMIUM}:{})});
 const page=await browser.newPage({viewport:{width:1280,height:720}}),errors=[],overflow=[];
 page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});
 await page.goto(base);await page.waitForSelector('.reveal section.present .bento-slide');
 const ids=await page.evaluate(()=>JSON.parse(document.querySelector('#bento-doc').textContent).slides.map(s=>s.id));assert.equal(ids.length,14);
 async function navigate(i){await page.evaluate(n=>{location.hash='#/'+n;},i);await page.waitForFunction(id=>document.querySelector('section.present .bento-slide')?.dataset.slideId===id,ids[i]);await page.waitForTimeout(180);await page.waitForFunction(()=>[...document.querySelectorAll('section.present .math-tex')].every(e=>e.querySelector('mjx-container')));}
 for(let i=0;i<ids.length;i++){
  await navigate(i);
  if(ids[i].endsWith('-live')){const f=page.frames().find(f=>f.url().includes('embed=region'));await f.waitForSelector('.lab');await f.waitForFunction(()=>[...document.querySelectorAll('.math-tex')].every(e=>e.querySelector('mjx-container')));}
  const issues=await page.evaluate(()=>[...document.querySelectorAll('section.present .bento-el-text')].flatMap(e=>{const r=e.getBoundingClientRect(),child=e.firstElementChild,ir=child.getBoundingClientRect();return ir.bottom>r.bottom+3||ir.right>r.right+3?[{id:e.dataset.elId,extraBottom:ir.bottom-r.bottom,extraRight:ir.right-r.right}]:[]}));
  if(issues.length)overflow.push({slide:ids[i],issues});
  assert.equal(await page.locator('section.present [data-mml-node="merror"]').count(),0);
  await page.screenshot({path:`${out}/${String(i+1).padStart(2,'0')}-${ids[i]}.png`});
 }
 await navigate(6);assert.ok(!page.frames().some(f=>f.url().includes('embed=region')),'Intro must not mount a live frame');
 await page.keyboard.press('ArrowRight');await page.waitForSelector('iframe[src*="demo=schedule"]');
 let frame=page.frames().find(f=>f.url().includes('demo=schedule'));await frame.waitForSelector('#next');
 assert.equal(await frame.locator('#message').inputValue(),'C>x3');
 assert.match(await frame.locator('#phase-title').textContent(),/3\/5/);
 assert.equal(await frame.locator('.message-calculation tbody tr').count(),4);
 for(let phase=1;phase<=5;phase++){
  await frame.locator(`[data-phase="${phase}"]`).click();
  assert.equal(await frame.locator('.beliefs .pending').count(),[5,5,4,2,0][phase-1]);
  for(const m of BP.run().phases[phase-1]){
   await frame.locator('#message').selectOption(BP.key(m.from,m.to));const values=await frame.locator('#vector strong').allTextContents();
   values.forEach((v,i)=>assert.ok(Math.abs(Number(v)-m.values[i])<.00006));
   for(const output of [0,1]){
    await frame.locator(`[data-retained="${output}"]`).click();
    const terms=await frame.locator('.message-calculation tbody .term').allTextContents();
    assert.equal(terms.length,m.terms[output].length);
    terms.forEach((v,i)=>assert.ok(Math.abs(Number(v)-m.terms[output][i].weight)<.00006));
    assert.ok(Math.abs(Number(await frame.locator('.message-calculation tfoot .term').textContent())-m.values[output])<.00006);
   }
  }
 }
 await frame.waitForFunction(()=>[...document.querySelectorAll('.math-tex')].every(e=>e.querySelector('mjx-container')));await page.screenshot({path:`${out}/schedule-step5.png`});
 for(const key of ['a','b','q'])for(const value of key==='q'?['0.5','0.99']:['0.05','0.95']){await frame.locator('#'+key).fill(value);assert.equal(Number(await frame.locator('#'+key+'-value').textContent()),Number(value));}
 assert.match(await frame.locator('#verification').textContent(),/32 configurations/);
 await frame.locator('#reset').click();await frame.locator('[data-phase="2"]').click();
 await frame.locator('[data-variable="x3"]').click();
 assert.match(await frame.locator('.belief-calculation caption').textContent(),/2 of 3/);
 assert.match(await frame.locator('#message-note').textContent(),/Partial belief/);
 assert.ok(Math.abs(Number((await frame.locator('#vector strong').allTextContents())[1])-15/27)<.00006);
 await frame.locator('[data-phase="3"]').click();
 assert.equal(await frame.locator('.belief-calculation tbody tr').count(),3);
 assert.match(await frame.locator('#message-note').textContent(),/Exact marginal/);
 assert.ok(Math.abs(Number((await frame.locator('#vector strong').allTextContents())[1])-7.92/13.584)<.00006);
 await frame.waitForFunction(()=>[...document.querySelectorAll('.math-tex')].every(e=>e.querySelector('mjx-container')));await page.screenshot({path:`${out}/schedule-belief.png`});
 await frame.locator('[data-view="message"]').click();await frame.locator('#message').selectOption('x3>C');
 assert.deepEqual(await frame.locator('#vector strong').allTextContents(),['12','15']);
 await frame.locator('#a').fill('0.95');await frame.locator('#q').fill('0.5');
 assert.deepEqual(await frame.locator('#vector strong').allTextContents(),['12','15']);
 await frame.locator('#reset').click();assert.equal(await frame.locator('#a').inputValue(),'0.3');assert.match(await frame.locator('#phase-title').textContent(),/1\/5/);
 await frame.locator('#play').click();await frame.waitForFunction(()=>document.querySelector('#phase-title').textContent.startsWith('2/5'));await frame.locator('#play').click();
 const paused=await frame.locator('#phase-title').textContent();await page.waitForTimeout(2000);assert.equal(await frame.locator('#phase-title').textContent(),paused);
 await frame.locator('#a').focus();const slideURL=page.url();await page.keyboard.press('ArrowRight');assert.equal(page.url(),slideURL);assert.equal(await frame.locator('#a').inputValue(),'0.31');
 await page.keyboard.press('Escape');await page.waitForFunction(()=>document.activeElement.classList.contains('reveal'));
 await frame.locator('#b').focus();await page.keyboard.press('PageUp');await page.waitForFunction(()=>document.querySelector('section.present .bento-slide')?.dataset.slideId==='schedule');
 await page.keyboard.press('PageDown');await page.waitForFunction(()=>document.querySelector('section.present .bento-slide')?.dataset.slideId==='schedule-live');
 frame=page.frames().find(f=>f.url().includes('demo=schedule'));await frame.waitForSelector('#a');await frame.locator('#a').focus();await page.keyboard.press('PageDown');await page.waitForFunction(()=>document.querySelector('section.present .bento-slide')?.dataset.slideId==='factor-message');
 await navigate(9);frame=page.frames().find(f=>f.url().includes('demo=factor'));await frame.waitForSelector('#terms');
 assert.equal(await frame.locator('#terms tr').count(),4);
 for(const output of [0,1]){await frame.locator(`[data-output="${output}"]`).click();const sum=Number(await frame.locator('#term-sum .term').textContent());assert.ok(Math.abs(sum-BP.run().messages['C>x3'].values[output])<1e-10);}
 await frame.locator('#q').fill('0.5');assert.equal(await frame.locator('#term-sum .term').textContent(),'0.5');
 await frame.locator('#reset').click();await frame.waitForFunction(()=>[...document.querySelectorAll('.math-tex')].every(e=>e.querySelector('mjx-container')));await page.screenshot({path:`${out}/factor-message.png`});
 // The shared print style reveals static fallbacks and hides the live frame.
 await page.waitForFunction(()=>[...document.querySelectorAll('#bento-print .math-tex')].every(e=>e.querySelector('mjx-container')));
 await page.emulateMedia({media:'print'});assert.equal(await page.locator('section.present iframe').isVisible(),false);assert.equal(await page.locator('#bento-print .bp-page').count(),14);await page.locator('#bento-print .bp-page').nth(7).screenshot({path:`${out}/print-schedule.png`});await page.locator('#bento-print .bp-page').nth(9).screenshot({path:`${out}/print-fallback.png`});await page.emulateMedia({media:'screen'});
 await page.locator('.reveal').focus();await page.keyboard.press('o');await page.waitForTimeout(400);assert.ok(await page.locator('.reveal.overview').count());await page.screenshot({path:`${out}/overview.png`});await page.keyboard.press('o');
 const narrow=await browser.newPage({viewport:{width:1024,height:768}});await narrow.goto(base+'#/7');await narrow.waitForSelector('iframe[src*="demo=schedule"]');await narrow.waitForTimeout(500);await narrow.screenshot({path:`${out}/narrow-deck.png`});await narrow.close();
 await page.goto(base+'study.html');await page.waitForSelector('article .math-tex mjx-container');assert.equal(await page.locator('article').count(),14);assert.equal(await page.locator('[data-mml-node="merror"]').count(),0);
 for(const demo of ['schedule','factor']){
  const mobile=await browser.newPage({viewport:{width:390,height:844}});await mobile.goto(base+'live/?demo='+demo);await mobile.waitForSelector('#a');await mobile.waitForFunction(()=>[...document.querySelectorAll('.math-tex')].every(e=>e.querySelector('mjx-container')));
  assert.ok(await mobile.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'Mobile route must not overflow');
  await mobile.locator('#q').fill('0.99');await mobile.locator('#reset').click();await mobile.waitForFunction(()=>[...document.querySelectorAll('.math-tex')].every(e=>e.querySelector('mjx-container')));await mobile.screenshot({path:`${out}/mobile-${demo}.png`,fullPage:true});await mobile.close();
 }
 console.log(JSON.stringify({slides:ids.length,liveDemos:2,overflow,errors,output:out},null,2));
 await browser.close();assert.deepEqual(errors,[]);assert.deepEqual(overflow,[]);
})().catch(e=>{console.error(e);process.exit(1);});
