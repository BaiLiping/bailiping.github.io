// Browser controls and the ACTUAL canonical Bento deck, not a mock renderer.
const {chromium}=require('playwright'),http=require('node:http'),fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'../..');
const server=http.createServer((req,res)=>{let f=path.resolve(root,'.'+decodeURIComponent(new URL(req.url,'http://local').pathname));if(!f.startsWith(root+path.sep)){res.writeHead(403);return res.end();}if(fs.existsSync(f)&&fs.statSync(f).isDirectory())f=path.join(f,'index.html');if(!fs.existsSync(f)){res.writeHead(404);return res.end();}res.writeHead(200,{'Content-Type':{'.html':'text/html','.js':'text/javascript','.mjs':'text/javascript','.css':'text/css','.json':'application/json','.svg':'image/svg+xml'}[path.extname(f)]||'application/octet-stream'});fs.createReadStream(f).pipe(res);});
(async()=>{await new Promise(resolve=>server.listen(8092,'127.0.0.1',resolve));const browser=await chromium.launch({headless:true});fs.mkdirSync('kalman-qa',{recursive:true});try{
 for(const mode of['bayes','mse','wls','kl','graphs']){
  const p=await browser.newPage({viewport:{width:1132,height:492}}),errors=[];p.on('pageerror',e=>errors.push(e.message));
  await p.goto(`http://127.0.0.1:8092/kalman-filter-derivations/live/families.html?demo=${mode}&embed=region`);await p.waitForFunction(()=>window.KalmanFamilyLab);
  if(mode==='mse'){await p.selectOption('#shape','two-point');await p.click('[data-action="optimum"]');assert.ok(Math.abs(await p.evaluate(()=>KalmanFamilyLab.getResult().cross))<1e-10);}
  if(mode==='wls'){await p.click('[data-action="newton"]');assert.ok(Math.abs(await p.evaluate(()=>KalmanFamilyLab.getResult().gradient))<1e-10);}
  if(mode==='kl'){await p.click('[data-action="mean"]');assert.ok(await p.evaluate(()=>KalmanFamilyLab.getResult().gap)>.1);await p.click('[data-action="posterior"]');assert.ok(Math.abs(await p.evaluate(()=>KalmanFamilyLab.getResult().gap))<1e-10);}
  if(mode==='graphs'){await p.click('[data-action="all"]');assert.equal(await p.evaluate(()=>KalmanFamilyLab.getResult().active.length),1);}
  assert.deepEqual(errors,[],mode);const sizes=await p.evaluate(()=>({width:document.documentElement.scrollWidth,viewport:innerWidth,panels:[...document.querySelectorAll('aside,.stage')].map(e=>({height:e.clientHeight,scroll:e.scrollHeight}))}));assert.ok(sizes.width<=sizes.viewport);console.log('LAB',mode,JSON.stringify(sizes));
  await p.screenshot({path:`kalman-qa/${mode}.png`});await p.click('[data-action="reset"]');await p.close();
 }
 const p=await browser.newPage({viewport:{width:1440,height:900}}),errors=[];p.on('pageerror',e=>errors.push(e.message));
 await p.goto('http://127.0.0.1:8092/kalman-filter-derivations/#family-comparison');await p.waitForFunction(()=>location.hash==='#/1');
 assert.equal(await p.evaluate(()=>JSON.parse(document.getElementById('bento-doc').textContent).slides.length),33);
 await p.waitForFunction(()=>window.MathJax&&MathJax.startup&&MathJax.startup.promise);await p.evaluate(()=>MathJax.startup.promise);
 // Compile every authored equation, including hidden sheets, with the actual MathJax version.
 const math=await p.evaluate(async()=>{const doc=JSON.parse(document.getElementById('bento-doc').textContent),errors=[];let count=0;for(const slide of doc.slides)for(const el of slide.elements){if(!el.html)continue;const node=document.createElement('div');node.innerHTML=el.html;for(const span of node.querySelectorAll('.math-tex')){const text=span.textContent.trim(),source=text.slice(2,-2);const svg=await MathJax.tex2svgPromise(source,{display:span.classList.contains('math-display')});count++;if(svg.querySelector('[data-mml-node="merror"]'))errors.push({slide:slide.id,element:el.id,source,error:svg.textContent});}}return{count,errors};});
 assert.deepEqual(math.errors,[]);console.log('MATHJAX',JSON.stringify(math));
 for(const id of['overview','family-comparison','group-bayes','group-mse','group-wls','group-kl','group-graphs','group-numerical']){
  await p.evaluate(id=>{location.hash='#'+id;},id);await p.waitForTimeout(450);await p.screenshot({path:`kalman-qa/${id}.png`});
 }
 await p.goto('http://127.0.0.1:8092/kalman-filter-derivations/#bayes-lab');await p.waitForFunction(()=>location.hash==='#/6');
 await p.waitForFunction(()=>[...document.querySelectorAll('iframe')].some(f=>f.src.includes('families.html?demo=bayes')));const frame=p.frames().find(f=>f.url().includes('families.html?demo=bayes'));assert.ok(frame,'inline Bayes experiment mounted');await frame.waitForFunction(()=>window.KalmanFamilyLab);assert.deepEqual(errors,[]);await p.screenshot({path:'kalman-qa/embedded.png'});await p.close();
 console.log('Canonical deck, 33 slides, named routes, MathJax and inline mounting passed.');
}finally{await browser.close();server.close();}})().catch(e=>{console.error(e);server.close();process.exitCode=1;});
