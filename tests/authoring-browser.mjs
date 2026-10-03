// Run with BAILIPING_PLAYWRIGHT_MODULE=/path/to/playwright/index.mjs node tests/authoring-browser.mjs
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import assert from 'node:assert/strict';
import {fileURLToPath} from 'node:url';
import {execFileSync} from 'node:child_process';
import {createAuthoringServer} from '../scripts/authoring-server.mjs';
import {readDocument} from '../scripts/authoring-build.mjs';
const {chromium}=await import(process.env.BAILIPING_PLAYWRIGHT_MODULE||'playwright');
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const temp=fs.mkdtempSync(path.join(os.tmpdir(),'bailiping-editor-browser-'));
for(const directory of ['assets','authoring','scripts','et-handover','radar-slam','kalman-filter-derivations','likelihood-vs-density','multidensity-fusion'])fs.cpSync(path.join(root,directory),path.join(temp,directory),{recursive:true});
const server=createAuthoringServer({root:temp});await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
const origin='http://127.0.0.1:'+server.address().port;
const browser=await chromium.launch({channel:'chrome',headless:true});
const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[];
page.on('pageerror',error=>errors.push(error.message));
const ready=async slug=>{await page.goto(origin+'/'+slug+'/?edit=1');await page.waitForFunction(()=>!!window.bento?.loadDoc&&!document.querySelector('#authoring-save').disabled)};
const file=path.join(temp,'et-handover/index.html');
try{
  await page.goto(origin+'/__authoring/');await page.locator('.card').first().waitFor();
  assert.equal(await page.locator('.card').count(),5);
  await ready('et-handover');
  const original=readDocument(fs.readFileSync(file,'utf8'));
  const title=original.slides[0].elements.find(e=>e.type==='text'&&e.fontSize>=32);
  assert.ok(title);
  const text=page.locator(`.ed-canvas-wrap [data-el-id="${title.id}"] .bento-text-inner`);
  await text.dblclick({delay:100});await page.keyboard.insertText('A visually edited handover presentation');
  await page.locator('#authoring-save').click();await page.waitForFunction(()=>document.querySelector('#authoring-status').textContent.startsWith('Saved locally'));
  assert.equal(readDocument(fs.readFileSync(file,'utf8')).slides[0].elements.find(e=>e.id===title.id).html,'A visually edited handover presentation');
  console.log('Text edited and saved through the UI.');

  // Move an ordinary text object through native pointer interaction.
  const movable=original.slides[0].elements.find(e=>e.type==='text'&&e.id!==title.id&&e.w<700&&e.y>500);
  assert.ok(movable);
  const element=page.locator(`.ed-canvas-wrap [data-el-id="${movable.id}"]`);
  await element.click({delay:100});await page.waitForFunction(id=>window.bento.selection.includes(id),movable.id);
  const box=await element.boundingBox();
  await page.mouse.move(box.x+20,box.y+box.height/2);await page.mouse.down();
  // Moveable selects asynchronously on pointer-down; allow its next frame.
  await page.waitForTimeout(100);
  await page.mouse.move(box.x+55,box.y+box.height/2-20,{steps:12});await page.mouse.up();
  await page.waitForFunction(({id,x,y})=>{const e=window.bento.doc.slides[0].elements.find(e=>e.id===id);return e.x!==x||e.y!==y},{id:movable.id,x:movable.x,y:movable.y},{timeout:3000}).catch(()=>{});
  const moved=await page.evaluate(id=>window.bento.doc.slides[0].elements.find(e=>e.id===id),movable.id);
  assert.ok(moved.x!==movable.x||moved.y!==movable.y,'drag changed the object position');
  await page.keyboard.press('Meta+s');await page.waitForFunction(()=>document.querySelector('#authoring-status').textContent.startsWith('Saved locally'));
  await ready('et-handover');
  assert.equal(await page.locator(`.ed-canvas-wrap [data-el-id="${title.id}"] .bento-text-inner`).innerText(),'A visually edited handover presentation');
  execFileSync(process.execPath,['et-handover/build.mjs'],{cwd:temp});
  const rebuilt=readDocument(fs.readFileSync(file,'utf8'));
  assert.equal(rebuilt.slides[0].elements.find(e=>e.id===title.id).html,'A visually edited handover presentation');
  assert.equal(rebuilt.slides[0].elements.find(e=>e.id===movable.id).x,moved.x);
  console.log('Position, save/reopen, and actual generator rebuild preserve edits.');

  await ready('et-handover');
  const mathIndex=original.slides.findIndex(s=>s.elements.some(e=>e.type==='text'&&/\\\[/.test(e.html)));
  const mathElement=original.slides[mathIndex].elements.find(e=>e.type==='text'&&/\\\[/.test(e.html));
  await page.locator(`.ed-thumb[data-index="${mathIndex}"]`).click();
  const equation=page.locator(`.ed-canvas-wrap [data-el-id="${mathElement.id}"] .bento-text-inner`);
  await equation.locator('mjx-container').first().waitFor({timeout:30000});
  await equation.dblclick({delay:100});
  assert.match(await equation.innerText(),/\\\[/);assert.equal(await equation.locator('mjx-container').count(),0);
  await page.keyboard.press('Escape');
  await page.locator('#authoring-save').click();await page.waitForFunction(()=>document.querySelector('#authoring-status').textContent.startsWith('Saved locally'));
  const afterMath=readDocument(fs.readFileSync(file,'utf8'));
  assert.match(afterMath.slides[mathIndex].elements.find(e=>e.id===mathElement.id).html,/\\\[/);
  assert.doesNotMatch(afterMath.slides[mathIndex].elements.find(e=>e.id===mathElement.id).html,/<mjx-container/);
  console.log('Math renders, reveals LaTeX for editing, and saves LaTeX.');

  const liveIndex=afterMath.slides.findIndex(s=>s.id==='s-score-live');
  await page.locator(`.ed-thumb[data-index="${liveIndex}"]`).click();
  await page.getByRole('button',{name:'Slideshow',exact:true}).click();
  await page.locator('.bento-present-overlay .companion-demo-frame[src]').first().waitFor({timeout:20000});
  const frame=page.frameLocator('.bento-present-overlay .companion-demo-frame[src]').first();
  // This older lab deliberately hides body while revealing its embedded region.
  await frame.locator('input:visible,button:visible,select:visible').first().waitFor();
  const slider=frame.locator('input[type="range"]:visible').first();
  const oldValue=await slider.inputValue();await slider.focus();await slider.press('ArrowRight');
  if(await slider.inputValue()===oldValue)await slider.press('ArrowLeft');
  assert.notEqual(await slider.inputValue(),oldValue);
  await page.screenshot({path:'/tmp/bailiping-editor-live-preview.png'});
  await page.keyboard.press('Escape');
  console.log('The editor’s Slideshow mounts the interactive lab.');

  for(const slug of ['radar-slam','likelihood-vs-density','multidensity-fusion']){
    await ready(slug);
    assert.ok(await page.locator('.ed-canvas-wrap .bento-slide').count());
    assert.ok(await page.locator('[oai-annotation-metadata]').count());
    await page.screenshot({path:'/tmp/bailiping-editor-'+slug+'.png'});
    console.log(slug+': visual editor and annotation targets loaded.');
  }
  assert.deepEqual(errors,[]);
  console.log('Browser checks passed; all edits were in the temporary checkout.');
}catch(error){
  await page.screenshot({path:'/tmp/bailiping-editor-failure.png'}).catch(()=>{});
  console.log(await page.evaluate(()=>({url:location.href,overlays:document.querySelectorAll('.bento-present-overlay').length,frames:[...document.querySelectorAll('iframe')].map(f=>({src:f.getAttribute('src'),root:f.closest('[data-slide-id]')?.dataset.slideId})),present:[...document.querySelectorAll('section.present')].map(s=>s.textContent.slice(0,100)),scripts:[...document.scripts].filter(s=>s.src.includes('inline')).map(s=>s.src)})).catch(()=>({})));
  throw error;
}finally{
  await browser.close();await new Promise(resolve=>server.close(resolve));
  if(process.env.KEEP_AUTHORING_QA)console.log('QA checkout: '+temp);else fs.rmSync(temp,{recursive:true,force:true});
}
