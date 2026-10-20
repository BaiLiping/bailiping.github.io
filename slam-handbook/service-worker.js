/* Offline-first cache of the handbook reader and numerical models. */
const CACHE = "slam-handbook-20261020-v1";
const ASSETS = ["./","./index.html","./style.css","./style-base.css","./style-experiments.css","./style-responsive.css","./chapters-1.js","./chapters-2.js","./chapters-3.js","./models-1.js","./models-2.js","./models-3.js","./app.js"];
for (let n=1; n<=18; n++) ASSETS.push("./model-"+String(n).padStart(2,"0")+".js");
self.addEventListener("install", event=>event.waitUntil(caches.open(CACHE).then(cache=>cache.addAll(ASSETS)).then(()=>self.skipWaiting())));
self.addEventListener("activate", event=>event.waitUntil(caches.keys().then(keys=>Promise.all(keys.filter(k=>k.startsWith("slam-handbook-")&&k!==CACHE).map(k=>caches.delete(k)))).then(()=>self.clients.claim())));
self.addEventListener("fetch", event=>{
  const url=new URL(event.request.url);
  if(event.request.method!=="GET" || url.origin!==location.origin || !url.pathname.startsWith("/slam-handbook/"))return;
  event.respondWith(caches.match(event.request).then(found=>found||fetch(event.request).then(response=>{
    if(response.ok){const cloned=response.clone();caches.open(CACHE).then(cache=>cache.put(event.request,cloned));}
    return response;
  })));
});