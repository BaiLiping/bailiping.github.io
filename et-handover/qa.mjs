// EO-specific integration checks and deterministic print fallbacks.
import fs from 'node:fs/promises';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
const here = path.dirname(fileURLToPath(import.meta.url));
const base = process.argv[2] || 'http://127.0.0.1:8767';
const outDir = process.env.BENTO_QA_DIR || '/tmp/eo-grbp-qa';
const chrome = process.env.CHROME_BIN || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const profile = await fs.mkdtemp(path.join(os.tmpdir(), 'eo-grbp-qa-'));
function delay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function availablePort() {
  return await new Promise((resolve, reject) => {
    const server = net.createServer();
    server.once('error', reject);
    server.listen(0, '127.0.0.1', () => {
      const address = server.address();
      server.close(error => error ? reject(error) : resolve(address.port));
    });
  });
}

class Cdp {
  constructor(socket) {
    this.socket = socket;
    this.nextId = 1;
    this.pending = new Map();
    this.events = [];
    socket.addEventListener('message', event => {
      const message = JSON.parse(event.data);
      if (message.id) {
        const pending = this.pending.get(message.id);
        if (!pending) return;
        this.pending.delete(message.id);
        if (message.error) pending.reject(new Error(message.error.message));
        else pending.resolve(message.result);
      } else {
        this.events.push(message);
      }
    });
  }

  static async connect(url) {
    const socket = new WebSocket(url);
    await new Promise((resolve, reject) => {
      socket.addEventListener('open', resolve, { once: true });
      socket.addEventListener('error', reject, { once: true });
    });
    return new Cdp(socket);
  }

  send(method, params = {}) {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.socket.send(JSON.stringify({ id, method, params }));
    });
  }

  close() {
    this.socket.close();
  }
}

async function waitForDebug(port) {
  const endpoint = 'http://127.0.0.1:' + port + '/json/version';
  for (let attempt = 0; attempt < 100; attempt += 1) {
    try {
      const response = await fetch(endpoint);
      if (response.ok) return;
    } catch {
      // Chrome is still starting.
    }
    await delay(100);
  }
  throw new Error('Chrome debugging endpoint did not start');
}

async function evaluate(cdp, expression) {
  const result = await cdp.send('Runtime.evaluate', {
    expression,
    awaitPromise: true,
    returnByValue: true
  });
  if (result.exceptionDetails) {
    throw new Error(result.exceptionDetails.text || 'Runtime evaluation failed');
  }
  return result.result.value;
}

async function waitFor(cdp, expression, label, timeoutMs = 15000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (await evaluate(cdp, expression)) return;
    await delay(100);
  }
  const state = await evaluate(
    cdp,
    '({ href: location.href, ready: document.readyState, bento: typeof window.bento, validate: typeof window.bento?.validate, slides: document.querySelectorAll(".bento-slide").length, title: document.title })'
  );
  throw new Error('Timed out waiting for ' + label + ': ' + JSON.stringify(state));
}


const port = await availablePort();
const browser = spawn(chrome, ['--headless=new','--disable-gpu','--hide-scrollbars','--remote-debugging-port='+port,'--user-data-dir='+profile,'about:blank'], {stdio:'ignore'});
let cdp;
const report = [];
function assert(value, message) { if (!value) throw new Error(message); }
try {
  await waitForDebug(port);
  const target = await (await fetch('http://127.0.0.1:'+port+'/json/new?about:blank',{method:'PUT'})).json();
  cdp=await Cdp.connect(target.webSocketDebuggerUrl);
  await cdp.send('Page.enable'); await cdp.send('Runtime.enable'); await cdp.send('Log.enable');
  await fs.mkdir(outDir,{recursive:true}); await fs.mkdir(path.join(here,'assets'),{recursive:true});
  const viewport = async(w,h)=>cdp.send('Emulation.setDeviceMetricsOverride',{width:w,height:h,deviceScaleFactor:1,mobile:false});
  const mathReady = async()=> {
    await waitFor(cdp,'Boolean(window.MathJax?.startup?.document)', 'MathJax startup');
    await delay(500);
    await waitFor(cdp,'[...document.querySelectorAll(".math-tex")].every(n=>n.querySelector("mjx-container svg"))','all math rendered');
    const bad=await evaluate(cdp,'document.querySelectorAll("[data-mjx-error], [data-mml-node=merror]").length');
    assert(bad===0,'MathJax errors: '+bad);
  };
  const go=async(url)=>{await cdp.send('Page.navigate',{url:base+url});await waitFor(cdp,'document.readyState === "complete"','page load');await mathReady();};
  const shot=async(file)=>{const r=await cdp.send('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});await fs.writeFile(file,Buffer.from(r.data,'base64'));};
  for (const [id,file] of [['handover-score-demo','score-fallback'],['toy','timeline-fallback']]) {
    await viewport(1200,486);
    await go('/et-handover/live/?slide-embed=%23'+id+'&v=grbp-2026-09');
    await shot(path.join(here,'assets',file+'.png'));
    const overflow=await evaluate(cdp,`(()=>{const t=document.getElementById(${JSON.stringify(id)});return {x:t.scrollWidth>t.clientWidth+2,y:t.scrollHeight>t.clientHeight+2};})()`);
    assert(!overflow.x&&!overflow.y,'Embedded overflow '+id+': '+JSON.stringify(overflow));
    report.push({embed:id,overflow});
    if (id === 'toy') {
      const highlight = await evaluate(cdp,`(()=>{
        const checks=[];
        for(const k of [0,T.events.e5-1,T.events.e5,T.events.e5+16,T.events.ack-1,T.events.ack,T.events.e6,N_FRAMES-1,T.events.e5-1]) {
          seek(k);
          checks.push({frame:k,visible:getComputedStyle(transferArrow).display!=='none',expected:k>=T.events.e5&&k<T.events.ack});
        }
        seek(0);
        return checks;
      })()`);
      assert(highlight.every(c=>c.visible===c.expected),'Transfer arrow visible outside handshake: '+JSON.stringify(highlight));
      report.push({transferHighlight:highlight});
      const sidebar = await evaluate(cdp,`(()=>({
        removed:!document.getElementById('transfer-focus'),
        blocks:document.querySelectorAll('.toy-side .side-block').length,
        scoresVisible:getComputedStyle(document.getElementById('lambda-meters').closest('.side-block')).display!=='none',
        fits:document.querySelector('.toy-side').getBoundingClientRect().bottom<=innerHeight
      }))()`);
      assert(sidebar.removed && sidebar.blocks===6 && sidebar.scoresVisible && sidebar.fits,'Sidebar restoration failed: '+JSON.stringify(sidebar));
      report.push({sidebar});
    }
  }
  await viewport(1280,720); await go('/et-handover/live/');
  const count=await evaluate(cdp,'document.querySelectorAll(".math-tex mjx-container svg").length');
  assert(count===10,'Expected all 10 live-page expressions typeset; got '+count);
  const variants=await evaluate(cdp,`(()=>{seek(200);const r={};for(const v of ['H','HM','HL','HP']){const input=document.querySelector('input[value="'+v+'"]');input.click();r[v]=document.getElementById('stat-total').textContent;}return r;})()`);
  assert(variants.H!==variants.HM && variants.HM!==variants.HL && variants.HL===variants.HP,'Variant payload accounting '+JSON.stringify(variants));
  const controlResult=await evaluate(cdp,`(()=>{seek(0);document.getElementById('btn-next').click();const event=frame;document.getElementById('btn-fwd1').click();const next=frame;document.getElementById('btn-back1').click();const back=frame;document.getElementById('btn-play').click();const active=playing;setPlaying(false);return {event,next,back,active,stopped:!playing};})()`);
  assert(controlResult.next===controlResult.event+1&&controlResult.back===controlResult.event&&controlResult.active&&controlResult.stopped,'Timeline buttons failed');
  const transfer=await evaluate(cdp,`({before:ownerAt(T.events.e5+31),after:ownerAt(T.events.ack),shadowBefore:shadowsAt(T.events.ack-1),shadowAfter:shadowsAt(T.events.ack)})`);
  assert(transfer.before==='A'&&transfer.after==='C'&&transfer.shadowBefore.includes('C')&&!transfer.shadowAfter.includes('C'),'Ownership must wait for acknowledgment');
  report.push({transfer});
  const extent=await evaluate(cdp,`(()=>{const s=document.getElementById('sr-extent');const r=[];for(const v of ['.55','1.7']){s.value=v;s.dispatchEvent(new Event('input',{bubbles:true}));r.push(document.getElementById('sr-area').textContent);}return r;})()`);
  assert(extent[0]!==extent[1],'Extent slider did not update model');
  report.push({pageMathCount:count,variants,controlResult,extent});
  for(const [w,h] of [[1280,720],[1024,768],[390,844]]) {
    await viewport(w,h);
    await evaluate(cdp,'document.getElementById("score-title").scrollIntoView({behavior:"instant",block:"start"});true'); await delay(100);
    const overflow=await evaluate(cdp,'document.documentElement.scrollWidth > innerWidth+2');
    assert(!overflow,'Page horizontal overflow at '+w);
    await shot(path.join(outDir,'page-score-'+w+'.png'));
  }
  await viewport(1280,720); await go('/et-handover/');
  await waitFor(cdp,'Boolean(window.bento?.doc)','Bento boot');
  const slides=await evaluate(cdp,'window.bento.doc.slides.map(s=>s.id)');
  for(let i=0;i<slides.length;i++) {
    await evaluate(cdp,`location.hash='#/${i}';true`);
    await waitFor(cdp,`document.querySelector('section.present .bento-slide')?.dataset.slideId===${JSON.stringify(slides[i])}`,'slide '+i);
    await mathReady();
    const live=await evaluate(cdp,'Boolean(document.querySelector("section.present iframe"))');
    if(live) await waitFor(cdp,'document.querySelector("section.present iframe")?.dataset.ready==="true"','live ready');
    const errors=await evaluate(cdp,`(()=>{const root=document.querySelector('section.present .bento-slide');return [...root.querySelectorAll('img')].filter(n=>!n.complete||!n.naturalWidth).map(n=>n.src);})()`);
    assert(!errors.length,'Broken slide images '+errors);
    await shot(path.join(outDir,String(i+1).padStart(2,'0')+'-'+slides[i]+'.png'));
  }
  // Regression: normal Bento arrow navigation can leave the hash unchanged.
  await evaluate(cdp,"location.hash='#/"+slides.indexOf('s-score')+"';true");
  await waitFor(cdp,"document.querySelector('section.present .bento-slide')?.dataset.slideId==='s-score'",'trigger introduction');
  await cdp.send('Input.dispatchKeyEvent',{type:'keyDown',key:'ArrowRight',code:'ArrowRight',windowsVirtualKeyCode:39});
  await cdp.send('Input.dispatchKeyEvent',{type:'keyUp',key:'ArrowRight',code:'ArrowRight',windowsVirtualKeyCode:39});
  await waitFor(cdp,"document.querySelector('section.present iframe')?.dataset.ready==='true'",'trigger live for keyboard test');
  await waitFor(cdp,`(()=>{const f=document.querySelector('section.present iframe');return f?.contentDocument?.readyState==='complete' && Boolean(f.contentDocument.getElementById('sr-extent'));})()`,'embedded controls loaded');
  await evaluate(cdp,`(()=>{const f=document.querySelector('section.present iframe');f.contentWindow.dispatchEvent(new f.contentWindow.KeyboardEvent('keydown',{key:'PageUp',bubbles:true}));return true;})()`);
  await waitFor(cdp,"document.querySelector('section.present .bento-slide')?.dataset.slideId==='s-score'",'Page Up returns to intro');
  report.push({keyboardReturnToIntro:true});
  await viewport(1024,768);await delay(150);await mathReady();await shot(path.join(outDir,'deck-1024.png'));
  // Verify every configured static fallback survives print; no live frame may print.
  await cdp.send('Emulation.setEmulatedMedia',{media:'print'});
  const print=await evaluate(cdp,`(()=>({visibleFrames:[...document.querySelectorAll('.companion-demo-stage')].filter(n=>getComputedStyle(n).display!=='none').length,brokenImages:[...document.images].filter(n=>!n.complete||!n.naturalWidth).length}))()`);
  assert(print.visibleFrames===0 && print.brokenImages===0,'Print fallback check failed');
  const errors=cdp.events.filter(e=>e.method==='Runtime.exceptionThrown'||(e.method==='Log.entryAdded'&&e.params.entry.level==='error'));
  assert(!errors.length,'Browser errors: '+JSON.stringify(errors));
  report.push({slides:slides.length,print,browserErrors:errors.length});
  await fs.writeFile(path.join(outDir,'report.json'),JSON.stringify(report,null,2));
  console.log(JSON.stringify(report,null,2));
} finally {cdp?.close();browser.kill('SIGTERM');await delay(150);await fs.rm(profile,{recursive:true,force:true});}
