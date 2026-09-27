// Keep copied links aligned with Bento's current slide after relative navigation.
// The shared inline adapter correctly navigates through Reveal's controls, which
// do not always update the URL in this read-only presentation shell.
(() => {
  let queued=false;
  function sync(){
    queued=false;
    const sections=[...document.querySelectorAll('.reveal .slides > section')];
    const index=sections.findIndex(s=>s.classList.contains('present'));
    if(index>=0&&location.hash!==`#/${index}`)history.replaceState(null,'',location.pathname+location.search+`#/${index}`);
  }
  new MutationObserver(()=>{if(!queued){queued=true;requestAnimationFrame(sync);}}).observe(document.documentElement,{subtree:true,childList:true,attributes:true,attributeFilter:['class']});
})();
