import { deck, inlineLiveMap } from './bento-deck.mjs';

/** Reuse the exact engine shipped with the reference deck. Only the two
 * compressed runtime blocks are extracted; no reference slide, script,
 * stylesheet link, or live-demo configuration is copied or executed.
 * Keeping this adapter small avoids duplicating/forking the Bento backend.
 */
const safeJSON = value => JSON.stringify(value).replaceAll('<', '\\u003c');
document.getElementById('bento-doc').textContent = safeJSON(deck);
document.getElementById('bento-inline-live-map').textContent = safeJSON(inlineLiveMap);
const indexes = Object.fromEntries(deck.slides.map((s,i)=>[s.id,i]));
function route() {
  let name;
  try { name = decodeURIComponent(location.hash.replace(/^#\/?/,'')); } catch { return; }
  if (Object.hasOwn(indexes,name)) history.replaceState(null,'',location.pathname+location.search+'#/'+indexes[name]);
}
route(); addEventListener('hashchange',route);
async function inflate(node) {
  if (!node?.textContent.trim()) throw new Error('The shared Bento runtime block is missing.');
  const bytes=Uint8Array.from(atob(node.textContent.trim()),c=>c.charCodeAt(0));
  const stream=new Blob([bytes]).stream().pipeThrough(new DecompressionStream('deflate-raw'));
  return new Response(stream).text();
}
async function start() {
  if (typeof DecompressionStream==='undefined') throw new Error('This deck needs a current browser with DecompressionStream support.');
  const response=await fetch(new URL('../kalman-filter-derivations/index.html',import.meta.url),{signal:AbortSignal.timeout(25000)});
  if (!response.ok) throw new Error(`The shared Bento engine could not be loaded (HTTP ${response.status}).`);
  const template=new DOMParser().parseFromString(await response.text(),'text/html');
  const blocks=['bento-rt-css','bento-rt'].map(id=>template.getElementById(id));
  if(blocks.some(node=>!node?.textContent.trim())) throw new Error('The shared Bento runtime blocks are missing.');
  // Keep the inert blocks in this document too: Bento's native Save a copy
  // export reads these IDs when packaging its self-contained engine.
  document.body.append(...blocks.map(node=>node.cloneNode(true)));
  const [css,js]=await Promise.all(blocks.map(inflate));
  const style=document.createElement('style');style.textContent=css;document.head.append(style);
  // The live host reads its map at startup, after the map above has been filled.
  const host=document.createElement('script');host.src=new URL('../assets/bento-inline-live.js',import.meta.url).href;
  document.body.append(host);
  host.onerror=()=>console.error('The shared inline-live host could not load; standalone labs remain available.');
  const loading=document.getElementById('loading');
  const observer=new MutationObserver(()=>{
    if(document.querySelector('.bento-slide')){loading.hidden=true;observer.disconnect();}
  });
  observer.observe(document.body,{childList:true,subtree:true});
  const url=URL.createObjectURL(new Blob([js],{type:'text/javascript'}));
  try { await import(url); } finally { URL.revokeObjectURL(url); }
  if(document.querySelector('.bento-slide')){loading.hidden=true;observer.disconnect();}
}
start().catch(error=>{
  console.error('Bento startup failed:',error);
  document.getElementById('load-status').textContent=error.message+' The direct lab link below does not depend on the slide engine.';
  const retry=document.createElement('button');retry.textContent='Retry loading';retry.onclick=()=>location.reload();
  document.getElementById('loading').append(retry);
});
