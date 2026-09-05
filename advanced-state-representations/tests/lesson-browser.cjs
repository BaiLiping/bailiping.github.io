'use strict';
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const http=require('node:http');
const {chromium}=require('playwright');
const root=path.resolve(__dirname,'../..');
const out=path.join(root,'asr-qa');fs.mkdirSync(out,{recursive:true});
const html=fs.readFileSync(path.join(root,'advanced-state-representations/index.html'),'utf8');
const doc=JSON.parse(html.match(/<script type="application\/bento\+json" id="bento-doc">([\s\S]*?)<\/script>/)[1]);
const maps=JSON.parse(html.match(/<script type="application\/json" id="bento-inline-live-map">([\s\S]*?)<\/script>/)[1]);
const errors=[],warnings=[],checks=[];
const server=http.createServer((req,res)=>{let file=path.resolve(root,'.'+decodeURIComponent(req.url.split('?')[0]));if(!file.startsWith(root+path.sep)&&file!==root){res.writeHead(403);return res.end();}try{if(fs.statSync(file).isDirectory())file=path.join(file,'index.html');const mime={'.html':'text/html','.js':'text/javascript','.mjs':'text/javascript','.css':'text/css','.svg':'image/svg+xml','.png':'image/png','.woff2':'font/woff2','.json':'application/json','.md':'text/plain'}[path.extname(file)]||'application/octet-stream';res.writeHead(200,{'Content-Type':mime});fs.createReadStream(file).pipe(res);}catch{res.writeHead(404);res.end('Not found');}});
(async()=>{await new Promise(r=>server.listen(0,'127.0.0.1',r));const origin=`http://127.0.0.1:${server.address().port}`,base=origin+'/advanced-state-representations/';const browser=await chromium.launch({headless:true,args:['--no-sandbox'],...(process.env.CHROMIUM_PATH?{executablePath:process.env.CHROMIUM_PATH}:{})});const context=await browser.newContext({viewport:{width:1280,height:720},reducedMotion:'reduce'});
 const bundle=process.env.MATHJAX_BUNDLE;if(bundle)await context.route('**/tex-svg-full.js',r=>r.fulfill({path:bundle,contentType:'text/javascript'}));
 const page=await context.newPage();page.on('pageerror',e=>errors.push(String(e)));page.on('console',m=>{if(m.type()==='error'&&!m.text().includes('favicon'))errors.push(m.text());});
 async function active(id){await page.waitForSelector(`section.present .bento-slide[data-slide-id="${id}"]`,{timeout:20000});return page.locator(`section.present .bento-slide[data-slide-id="${id}"]`).last();}
 async function go(i){await page.goto(base+'#/'+i,{waitUntil:'domcontentloaded'});await page.waitForSelector('.bento-slide',{state:'attached',timeout:30000});if(!await page.locator('section.present .bento-slide').count()){const b=page.getByRole('button',{name:/present/i}).first();if(await b.count())await b.click();}const r=await active(doc.slides[i].id);await page.waitForFunction(()=>typeof window.MathJax?.tex2svgPromise==='function',{timeout:30000});await page.waitForTimeout(550);await page.waitForFunction(()=>[...document.querySelectorAll('section.present .math-tex')].every(n=>n.querySelector('mjx-container')),{timeout:30000});return r;}
 try{
  for(let i=0;i<doc.slides.length;i++){
   const s=doc.slides[i],r=await go(i);assert.equal(await r.locator('[data-mml-node="merror"],mjx-merror').count(),0,'MathJax error on '+s.id);
   const bad=await r.evaluate(el=>[...el.querySelectorAll('.bento-text-inner')].flatMap(n=>{const box=n.getBoundingClientRect(),range=document.createRange();range.selectNodeContents(n);const content=range.getBoundingClientRect();return (content.bottom>box.bottom+4||content.right>box.right+4)?[{text:n.textContent.slice(0,160),box:{x:box.x,y:box.y,w:box.width,h:box.height},content:{x:content.x,y:content.y,w:content.width,h:content.height}}]:[];}));if(bad.length)warnings.push({slide:s.id,overflow:bad});
   await page.screenshot({path:path.join(out,String(i+1).padStart(2,'0')+'-'+s.id+'.png')});checks.push('slide '+s.id);
  }
  for(const m of maps){
   const demo=new URL(m.src,base).searchParams.get('demo'),r=await go(m.slideIndex),iframe=r.locator('iframe');await iframe.waitFor();const frame=await (await iframe.elementHandle()).contentFrame();await frame.waitForFunction(()=>window.ASRLab);
   assert.equal(await frame.evaluate(()=>window.ASRLab.demo),demo);
   assert.equal(await page.locator('iframe[src*="demo="]').count(),1,'inactive labs should unload');
   const input=frame.locator('input[type="range"]').first();await input.focus();const before=await input.inputValue(),url=page.url();await input.press('ArrowRight');await page.waitForTimeout(70);assert.equal(page.url(),url,'slider arrow navigated parent');assert.notEqual(await input.inputValue(),before,'slider arrow did not change control');
   await input.press('Escape');await page.waitForTimeout(80);assert.ok(await page.evaluate(()=>document.activeElement?.classList.contains('reveal')),'Escape did not restore Bento focus');
   await input.focus();await input.press('PageDown');await active(doc.slides[m.slideIndex+1].id);checks.push('embedded focus/navigation '+demo);
   await go(m.slideIndex);await page.emulateMedia({media:'print'});const printed=page.locator(`.bento-slide[data-slide-id="${m.slide}"]`).last();assert.equal(await printed.locator('iframe').evaluate(e=>getComputedStyle(e).display),'none');assert.ok(await printed.locator('img').count()>0,'static figure missing');await page.emulateMedia({media:'screen'});checks.push('print fallback '+demo);
  }
  for(const demo of maps.map(m=>new URL(m.src,base).searchParams.get('demo'))){
   await page.setViewportSize({width:1136,height:475});await page.goto(base+'live/?demo='+demo+'&embed=region');await page.waitForFunction(()=>window.ASRLab);const defaultState=await page.evaluate(()=>window.ASRLab.state);
   for(const control of await page.locator('input[type="range"],select').all()){
    const data=await control.evaluate(e=>({id:e.id,range:e.type==='range',values:e.type==='range'?[e.min,e.max]:[...e.options].map(o=>o.value)}));
    for(const value of data.values){await control.evaluate((e,v)=>{e.value=v;e.dispatchEvent(new Event('input',{bubbles:true}));},value);await page.waitForFunction(({key,v})=>String(window.ASRLab.state[key])===v,{key:data.id,v:value});assert.ok(!await page.locator('#figure').innerHTML().then(s=>/NaN|Infinity|undefined/.test(s)),demo+' nonfinite figure');}
   }
   await page.locator('#reset').click();assert.deepEqual(await page.evaluate(()=>window.ASRLab.state),defaultState,'reset '+demo);
   if(demo==='optimize'){await page.locator('#step').click();assert.equal(await page.evaluate(()=>window.ASRLab.state.steps),1);}
   if(demo==='pose'){await page.locator('#collapse').click();assert.equal(await page.evaluate(()=>window.ASRLab.state.angle),180);}
   const sizes=await page.evaluate(()=>({h:document.documentElement.scrollHeight,w:document.documentElement.scrollWidth}));if(sizes.h>479||sizes.w>1140)warnings.push({demo,embeddedOverflow:sizes});
   await page.screenshot({path:path.join(out,'lab-'+demo+'.png')});checks.push('all controls '+demo);
   await page.setViewportSize({width:390,height:844});await page.goto(base+'live/?demo='+demo);await page.waitForFunction(()=>window.ASRLab);assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+2),'mobile horizontal page overflow '+demo);await page.screenshot({path:path.join(out,'mobile-'+demo+'.png'),fullPage:true});checks.push('mobile '+demo);
  }
  await page.setViewportSize({width:1100,height:850});await page.goto(base+'study.html');await page.waitForFunction(()=>typeof MathJax?.tex2svgPromise==='function');await page.waitForFunction(()=>[...document.querySelectorAll('.math-tex')].every(n=>n.querySelector('mjx-container')),{timeout:60000});assert.equal(await page.locator('[data-mml-node="merror"]').count(),0);assert.equal(await page.locator('main>section').count(),doc.slides.length);checks.push('complete study notes and math');
  await page.screenshot({path:path.join(out,'study.png')});
  assert.deepEqual(errors,[],'browser console errors');
  fs.writeFileSync(path.join(out,'report.json'),JSON.stringify({slides:doc.slides.length,labs:maps.length,checks,errors,warnings},null,2));
  console.log(JSON.stringify({slides:doc.slides.length,labs:maps.length,checks:checks.length,errors,warnings},null,2));
  if(warnings.length)throw Error('Layout warnings require visual review: see report.json');
 }catch(e){fs.writeFileSync(path.join(out,'report.json'),JSON.stringify({checks,errors,warnings,failure:String(e),url:page.url()},null,2));fs.writeFileSync(path.join(out,'failure-dom.html'),await page.content());await page.screenshot({path:path.join(out,'failure.png'),fullPage:true});throw e;}finally{await browser.close();server.close();}
})().catch(e=>{console.error(e);process.exitCode=1;server.close();});
