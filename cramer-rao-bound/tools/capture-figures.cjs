// Generate static vector figures from the same calculations used by the guide.
const fs=require('node:fs');
const path=require('node:path');
const {pathToFileURL}=require('node:url');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const root=path.resolve(__dirname,'..');
(async()=>{
 const browser=await chromium.launch({headless:true,...(process.env.CHROMIUM_EXECUTABLE?{executablePath:process.env.CHROMIUM_EXECUTABLE}:{})});
 try {
  const page=await browser.newPage({viewport:{width:1440,height:900}});
  await page.goto(pathToFileURL(path.join(root,'guide/index.html')).href);
  await page.waitForFunction(()=>window.CRBDeck);
  const figures=await page.evaluate(()=>[
   ...[...document.querySelectorAll('[data-chart]')].map(e=>({name:e.dataset.chart,svg:e.querySelector('svg').outerHTML})),
   ...['mc-chart','bias-chart','bias-bars','geometry-chart'].map(name=>({name,svg:document.querySelector('#'+name+' svg').outerHTML}))
  ]);
  fs.mkdirSync(path.join(root,'assets'),{recursive:true});
  for(const {name,svg} of figures)fs.writeFileSync(path.join(root,'assets',name+'.svg'),svg+'\n');
  fs.writeFileSync(path.join(root,'assets/lab-states.json'),JSON.stringify(await page.evaluate(()=>CRBDeck.state()),null,2)+'\n');
  console.log(`Captured ${figures.length} vector figures from the numerical implementation.`);
 } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exit(1)});
