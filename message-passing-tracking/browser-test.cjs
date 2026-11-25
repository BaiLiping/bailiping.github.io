// Requires Playwright through NODE_PATH. Screenshots stay outside the site.
const {chromium}=require('playwright'),assert=require('node:assert/strict'),fs=require('node:fs'),M=require('./model.js'),G=require('./graph.js');
const base=process.env.TRACKING_BASE_URL||'http://127.0.0.1:8793/message-passing-tracking/',out=process.env.TRACKING_QA_DIR||'/tmp/tracking-qa';fs.mkdirSync(out,{recursive:true});
const close=(a,b,tolerance=.00006)=>assert.ok(Math.abs(a-b)<tolerance,`${a} != ${b}`);
(async()=>{
 const browser=await chromium.launch({headless:true,...(process.env.BP_CHROMIUM?{executablePath:process.env.BP_CHROMIUM}:{})});
 try{
  const page=await browser.newPage({viewport:{width:1280,height:720}}),errors=[],overflow=[];
  page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});
  await page.goto(base);await page.waitForSelector('.reveal section.present .bento-slide');
  const ids=await page.evaluate(()=>JSON.parse(document.querySelector('#bento-doc').textContent).slides.map(s=>s.id));assert.equal(ids.length,12);
  const math=async frame=>{await frame.waitForFunction(()=>[...document.querySelectorAll('.math-tex')].every(e=>e.querySelector('mjx-container')));assert.equal(await frame.locator('[data-mml-node="merror"]').count(),0);};
  async function navigate(i){await page.evaluate(n=>{location.hash='#/'+n;},i);await page.waitForFunction(id=>document.querySelector('section.present .bento-slide')?.dataset.slideId===id,ids[i]);await page.waitForTimeout(160);await math(page);}
  for(let i=0;i<ids.length;i++){
   await navigate(i);if(ids[i].endsWith('-live')){const f=page.frames().find(f=>f.url().includes('embed=region'));await f.waitForSelector('#next');await math(f);}
   const issues=await page.evaluate(()=>[...document.querySelectorAll('section.present .bento-el-text')].flatMap(e=>{const r=e.getBoundingClientRect(),inner=e.firstElementChild.getBoundingClientRect();return inner.bottom>r.bottom+3||inner.right>r.right+3?[{id:e.dataset.elId,bottom:inner.bottom-r.bottom,right:inner.right-r.right}]:[]}));
   if(issues.length)overflow.push({slide:ids[i],issues});await page.screenshot({path:`${out}/${String(i+1).padStart(2,'0')}-${ids[i]}.png`});
  }
  await navigate(5);assert.ok(!page.frames().some(f=>f.url().includes('embed=region')));await page.keyboard.press('ArrowRight');await page.waitForSelector('iframe[src*="embed=region"]');
  let frame=page.frames().find(f=>f.url().includes('embed=region'));await frame.waitForSelector('#step');assert.equal(await frame.locator('#step option').count(),17);
  const model=M.run(),steps=M.steps();
  const expected=(s,j,m,kind='legacy')=>s.id==='prior'?M.priors[j]:['predict','copy'].includes(s.id)?model.alpha[j]:s.id==='beta'?model.beta[j]:s.id==='xi'?model.xi[m]:s.id==='initial'?model.da.initial[j][m].values:s.id==='nu'?model.da.rounds[s.iteration-1].nu[j][m]:s.id==='phi'?model.da.rounds[s.iteration-1].phi[j][m]:s.id==='kappa'?model.da.kappa[j]:s.id==='iota'?model.da.iota[m]:s.id==='gamma'?model.gamma[j]:s.id==='zeta'?model.zeta[m]:kind==='legacy'?model.legacy[j]:model.newTargets[m];
  for(let index=0;index<steps.length;index++){
   const s=steps[index];await frame.locator('#step').selectOption(String(index));
   for(const [j,m] of [[0,0],[1,1],[0,1],[1,0]]){
    await frame.locator('#j').selectOption({value:String(j)});await frame.locator('#m').selectOption({value:String(m)});
    const values=await frame.locator('#vector strong').allTextContents();values.forEach((v,y)=>close(Number(v),expected(s,j,m)[y]));
    assert.deepEqual(await frame.locator('.active-path').evaluateAll(es=>es.map(e=>[e.dataset.from,e.dataset.to])),G.paths(s,j,m));
    if(['predict','beta','xi','initial','nu','phi','gamma','zeta'].includes(s.id))for(const entry of [0,1,2]){
     await frame.locator(`[data-entry="${entry}"]`).click();
     const terms=await frame.locator('.arithmetic tbody .term').allTextContents();const raw=Number(await frame.locator('.arithmetic tfoot .term').textContent());
     close(terms.reduce((sum,t)=>sum+Number(t),0),raw,.0003);
     if(!['initial','nu','phi'].includes(s.id))close(raw,expected(s,j,m)[entry]);
    }
   }
   await math(frame);
   const clipping=await frame.evaluate(()=>{const top=document.querySelector('.parameters').getBoundingClientRect().top;return ['#note','#vector','#beliefs'].filter(s=>!document.querySelector(s).hidden).map(s=>({selector:s,bottom:document.querySelector(s).getBoundingClientRect().bottom,limit:top})).filter(r=>r.bottom>r.limit+1);});
   if(clipping.length)overflow.push({step:index,clipping});
   await page.screenshot({path:`${out}/step-${String(index+1).padStart(2,'0')}-${s.id}.png`});
  }
  await frame.locator('[data-kind="new"]').click();(await frame.locator('#vector strong').allTextContents()).forEach((v,y)=>close(Number(v),model.newTargets[0][y]));
  await frame.locator('[data-candidate="3"]').click();assert.equal(await frame.locator('#m').inputValue(),'1');
  await frame.locator('[data-group="2"]').click();assert.equal(await frame.locator('#app').getAttribute('data-step'),'initial');
  await frame.locator('[data-node="psi01"]').click();assert.equal(await frame.locator('#j').inputValue(),'0');assert.equal(await frame.locator('#m').inputValue(),'1');
  await frame.locator('[data-node="psi10"]').focus();await page.keyboard.press('Enter');assert.equal(await frame.locator('#j').inputValue(),'1');assert.equal(await frame.locator('#m').inputValue(),'0');
  for(const key of ['z2','pD','birth'])for(const val of key==='z2'?['-0.2','1.2']:key==='pD'?['0.05','0.99']:['0','1']){await frame.locator('#'+key).fill(val);assert.equal(Number(await frame.locator('#'+key+'-value').textContent()),Number(val));}
  for(const iterations of [1,2,5,8,12]){await frame.locator('#iterations').selectOption(String(iterations));assert.equal(await frame.locator('#step option').count(),11+2*iterations);}
  await frame.locator('[data-group="4"]').click();assert.match(await frame.locator('#check').textContent(),/largest state error/);await math(frame);await page.screenshot({path:`${out}/extreme-beliefs.png`});
  await frame.locator('#reset').click();assert.equal(await frame.locator('#z2').inputValue(),'0.62');assert.equal(await frame.locator('#step').inputValue(),'0');assert.equal(await frame.locator('#iterations').inputValue(),'3');
  await frame.locator('#next').click();assert.equal(await frame.locator('#app').getAttribute('data-step'),'predict');await frame.locator('#prev').click();assert.equal(await frame.locator('#step').inputValue(),'0');
  await frame.locator('#play').click();await frame.waitForFunction(()=>document.querySelector('#step').value==='1');await frame.locator('#play').click();await page.waitForTimeout(2800);assert.equal(await frame.locator('#step').inputValue(),'1');
  await frame.locator('#z2').focus();const url=page.url();await page.keyboard.press('ArrowRight');assert.equal(page.url(),url);assert.equal(await frame.locator('#z2').inputValue(),'0.63');
  await page.keyboard.press('Escape');await page.waitForFunction(()=>document.activeElement.classList.contains('reveal'));
  await frame.locator('#pD').focus();await page.keyboard.press('PageUp');await page.waitForFunction(()=>document.querySelector('section.present .bento-slide')?.dataset.slideId==='schedule');await page.keyboard.press('PageDown');await page.waitForSelector('iframe[src*="embed=region"]');
  frame=page.frames().find(f=>f.url().includes('embed=region'));await frame.waitForSelector('#pD');await frame.locator('#pD').focus();await page.keyboard.press('PageDown');await page.waitForFunction(()=>document.querySelector('section.present .bento-slide')?.dataset.slideId==='association');
  await navigate(6);await math(page);await page.emulateMedia({media:'print'});await page.waitForFunction(()=>[...document.querySelectorAll('#bento-print .math-tex')].every(e=>e.querySelector('mjx-container')));assert.equal(await page.locator('#bento-print .bp-page').count(),12);assert.equal(await page.locator('section.present iframe').isVisible(),false);await page.locator('#bento-print .bp-page').nth(6).screenshot({path:`${out}/print-fallback.png`});await page.emulateMedia({media:'screen'});
  await page.locator('.reveal').focus();await page.keyboard.press('o');await page.waitForTimeout(350);assert.ok(await page.locator('.reveal.overview').count());await page.screenshot({path:`${out}/overview.png`});await page.keyboard.press('o');
  await page.goto(base+'study.html');await page.waitForSelector('article .math-tex mjx-container');await math(page);assert.equal(await page.locator('article').count(),12);assert.equal(await page.locator('article#graph').evaluate(e=>getComputedStyle(e).display),'block');assert.ok(await page.locator('article#graph .diagram').evaluate(e=>e.getBoundingClientRect().width>600));
  for(const viewport of [{width:1024,height:768},{width:390,height:844}]){
   const p=await browser.newPage({viewport});await p.goto(base+'live/');await p.waitForSelector('#next');await math(p);assert.ok(await p.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));await p.locator('#step').selectOption('7');await math(p);await p.screenshot({path:`${out}/direct-${viewport.width}.png`,fullPage:true});await p.locator('[data-group="4"]').click();await p.locator('[data-kind="new"]').click();await math(p);assert.ok(await p.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));await p.screenshot({path:`${out}/belief-${viewport.width}.png`,fullPage:true});await p.close();
  }
  const narrow=await browser.newPage({viewport:{width:1024,height:768}});await narrow.goto(base+'#/6');await narrow.waitForSelector('iframe');await narrow.waitForTimeout(600);await narrow.screenshot({path:`${out}/narrow-deck.png`});await narrow.close();
  console.log(JSON.stringify({slides:ids.length,steps:steps.length,errors,overflow,output:out},null,2));assert.deepEqual(errors,[]);assert.deepEqual(overflow,[]);
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1);});
