(() => {
 const observer=new IntersectionObserver(entries=>{for(const e of entries)if(!e.isIntersecting)e.target.contentWindow?.postMessage({type:'grassmann-pause'},location.origin);},{threshold:.1});
 document.querySelectorAll('iframe').forEach(frame=>observer.observe(frame));
})();
