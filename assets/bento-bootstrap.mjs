// Bootstrap the unchanged, embedded Bento engine for module-authored decks.
const doc=JSON.parse(document.getElementById('bento-doc').textContent);
const routes=Object.fromEntries(doc.slides.map((slide,index)=>[slide.id,index]));
function route(){let id;try{id=decodeURIComponent(location.hash.replace(/^#\/?/,''))}catch{return}if(Object.hasOwn(routes,id))history.replaceState(null,'',location.pathname+location.search+'#/'+routes[id])}
route();addEventListener('hashchange',route);
async function inflate(id){const node=document.getElementById(id);const bytes=Uint8Array.from(atob(node.textContent.trim()),c=>c.charCodeAt(0));return new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('deflate-raw'))).text()}
try{
  const [css,js]=await Promise.all(['bento-rt-css','bento-rt'].map(inflate));
  const style=document.createElement('style');style.textContent=css;document.head.append(style);
  const url=URL.createObjectURL(new Blob([js],{type:'text/javascript'}));
  try{await import(url)}finally{URL.revokeObjectURL(url)}
  document.getElementById('loading').hidden=true;
}catch(error){document.getElementById('load-status').textContent='Could not load the presentation: '+error.message;console.error(error)}
