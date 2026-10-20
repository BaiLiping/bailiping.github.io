const { chromium } = require('playwright');
const assert = require('node:assert/strict');
(async () => {
  const browser = await chromium.launch({headless:true});
  try {
    for (const device of [{width:1440,height:900},{width:390,height:844}]) {
      const context = await browser.newContext({viewport:device,acceptDownloads:true});
      const page = await context.newPage();
      const errors=[];
      page.on('pageerror',e=>errors.push(e.message));
      await page.goto('http://127.0.0.1:8765/slam-handbook/');
      for(let n=1;n<=18;n++) {
        await page.evaluate(n=>location.hash='#chapter-'+n,n);
        await page.waitForFunction(n=>document.querySelector('h1')?.textContent===window.SLAM_CHAPTERS.find(c=>c.n===n).title&&document.querySelectorAll('.plot-surface svg').length===3,n,{timeout:20000});
        const width=await page.evaluate(()=>[document.documentElement.scrollWidth,innerWidth]);
        assert(width[0] <= width[1]+3,'horizontal overflow in chapter '+n);
        assert.equal(await page.locator('#lesson [role=alert]').count(),0);
      }
      assert.deepEqual(errors,[]);
      await context.close();
    }
    console.log('PASS browser smoke: 18 chapters, 54 rendered demos at desktop and mobile widths');
  } finally { await browser.close(); }
})().catch(e=>{console.error(e);process.exitCode=1});
