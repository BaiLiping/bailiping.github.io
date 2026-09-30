// Export the actual native Bento slides and their independent static print copy.
const fs=require('node:fs');
const path=require('node:path');
const {pathToFileURL}=require('node:url');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const root=path.resolve(__dirname,'..');
const args=process.argv.slice(2);
const value=(flag,fallback)=>args.includes(flag)?path.resolve(args[args.indexOf(flag)+1]):fallback;
const output=value('--output',path.join(root,'exports'));
const screenshots=value('--screenshots',path.join(output,'slides'));
(async()=>{
 fs.mkdirSync(output,{recursive:true});fs.mkdirSync(screenshots,{recursive:true});
 const browser=await chromium.launch({headless:true,...(process.env.CHROMIUM_EXECUTABLE?{executablePath:process.env.CHROMIUM_EXECUTABLE}:{})});
 try {
  const page=await browser.newPage({viewport:{width:1280,height:720},deviceScaleFactor:2});
  await page.goto(pathToFileURL(path.join(root,'index.html')).href);
  await page.waitForFunction(()=>window.bento?.doc);
  await page.locator('img').evaluateAll(images=>Promise.all(images.map(im=>im.decode())));
  await page.pdf({path:path.join(output,'Cramer-Rao-Bound.pdf'),printBackground:true,preferCSSPageSize:true});
  const slides=await page.evaluate(()=>window.bento.doc.slides);
  for(let i=0;i<slides.length;i++){
   await page.evaluate(i=>{location.hash='#/'+i;},i);
   await page.waitForFunction(id=>document.querySelector('section.present .bento-slide')?.dataset.slideId===id,slides[i].id);
   await page.locator('section.present .bento-slide').screenshot({path:path.join(screenshots,`slide-${String(i+1).padStart(2,'0')}.png`)});
  }
  fs.writeFileSync(path.join(output,'bento-slides.json'),JSON.stringify(slides,null,2));
  console.log(`Exported ${slides.length} Bento slide images and a static PDF.`);
 }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exit(1)});
