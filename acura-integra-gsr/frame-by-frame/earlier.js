document.addEventListener('change',event=>{
  const toggle=event.target.closest('.earlier input');
  if (!toggle || !toggle.checked) return;
  const trace=toggle.closest('article').querySelector('.early-trace');
  if (trace.dataset.early) { trace.src=trace.dataset.early;delete trace.dataset.early; }
});
const jump=document.getElementById('jump');
jump.addEventListener('change',()=>{if (jump.value) location.hash=jump.value;});
