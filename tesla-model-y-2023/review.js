const frames = await (await fetch('comparisons.json')).json();
const $ = id => document.getElementById(id);
const selector = $('frame');
let current = 0, mode = 'pair';
for(const [i,f] of frames.entries()) selector.add(new Option(f.label,i));
function display(){
  const f=frames[current];
  selector.value=current;
  $('ref').src=f.source;
  $('ref').alt=f.label;
  $('credit').textContent=f.credit;
  $('match').href='./?frame='+encodeURIComponent(f.id)+'#model';
  $('full').href=f.source;
  const file='./comparisons/'+f.id;
  $('native').href=file+'.webp';
  $('render').src=file+(mode==='outline'?'-outline.webp':'.webp');
  const [w,h]=f.imageSize;
  for(const el of [$('render'),$('ref')]){el.width=w;el.height=h;}
  $('stage').className=mode==='pair'?'pair':'overlay';
  $('second').style.opacity=mode==='blend'?$('opacity').value/100:1;
  $('opacity').hidden=mode!=='blend';
  $('observed').textContent=f.status+' · '+f.note;
  $('metric').textContent=f.metric?
    w+' × '+h+' px. Median edge distance '+f.metric.median+' px. The 95th percentile is '+f.metric.p95+' px, or '+(100*f.metric.p95/f.metric.width).toFixed(1)+'% of the visible vehicle width in this frame.':w+' × '+h+' px. Detail camera estimated visually.';
  document.querySelectorAll('[data-mode]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.mode===mode)));
}
function select(i){
  current=(i+frames.length)%frames.length;display();
  const u=new URL(location.href);u.searchParams.set('frame',frames[current].id);history.replaceState(null,'',u);
}
selector.onchange=()=>select(Number(selector.value));
$('prev').onclick=()=>select(current-1);$('next').onclick=()=>select(current+1);
$('opacity').oninput=()=>{$('second').style.opacity=$('opacity').value/100;};
for(const b of document.querySelectorAll('[data-mode]'))b.onclick=()=>{mode=b.dataset.mode;display();};
current=Math.max(0,frames.findIndex(f=>f.id===new URLSearchParams(location.search).get('frame')));
display();
