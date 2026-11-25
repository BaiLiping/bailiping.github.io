import {changesBetween} from '../assets/authoring-model.mjs';
const config = JSON.parse(document.getElementById('authoring-config').textContent);
const bar = document.createElement('nav');bar.id='authoring-bar';bar.setAttribute('aria-label','Website authoring');
bar.innerHTML='<a href="/__authoring/">← Presentations</a><span id="authoring-status" role="status">Opening the editor…</span><span class="spacer"></span><button id="authoring-help-button" title="Editing guide">Help</button><button id="authoring-export" class="secondary">Export draft</button><button id="authoring-import" class="secondary">Import draft</button><button id="authoring-feedback">Copy feedback</button><button id="authoring-preview">Preview saved</button><button id="authoring-save" class="primary" disabled>Save changes</button>';
document.body.prepend(bar);
const status=document.getElementById('authoring-status'),saveButton=document.getElementById('authoring-save');
const help=document.createElement('aside');help.id='authoring-help';help.hidden=true;
help.innerHTML='<p><b>Edit directly.</b> Double-click text to type. Drag objects or their corner handles. Use the right panel for fonts, colours, and sizes. Drag slide thumbnails to reorder them.</p><p><b>Save changes · ⌘S.</b> Saves this website draft on your Mac. Preview saved opens the latest saved presentation. Slideshow previews your current edits and live demonstrations.</p><p><b>Work with Codex.</b> Add comments with Bento’s Comment tool (C), then Copy feedback into our chat. Ask Codex to check and publish your saved changes.</p><p><b>Equations and labs.</b> Equations reveal their LaTeX while typing. Keep each live slide beside its introduction. Ask Codex to change a demonstration’s calculations.</p><p>Export draft keeps a portable backup. Import restores it. Bento’s own Save-as menu exports an HTML copy; use Save changes here for the website.</p>';
document.body.append(help);
document.getElementById('authoring-help-button').onclick=()=>{help.hidden=!help.hidden};
let baseline, saved, saving=false, ready=false, revision=config.revision;
const copy=value=>structuredClone(value);
const show=(text,error=false)=>{status.textContent=text;status.title=text;status.toggleAttribute('data-error',error)};
const documentNow=()=>copy(window.bento.doc);
const changed=()=>ready&&changesBetween(saved,window.bento.doc).length>0;
function commitText(){document.activeElement?.blur()}

async function save(){
  if(!ready||saving)return;
  commitText();saving=true;saveButton.disabled=true;show('Saving…');
  const edited=documentNow();
  try{
    const response=await fetch('/__authoring/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session:config.session,revision,baseline,edited})});
    const result=await response.json();if(!response.ok)throw Error(result.error||'Save failed.');
    revision=result.revision;baseline=copy(edited);saved=copy(edited);show('Saved locally · ready to publish');
  }catch(error){show(error.message,true);help.hidden=false}
  finally{saving=false;saveButton.disabled=false}
}
saveButton.onclick=save;
document.addEventListener('click',event=>{
  const button=event.target.closest?.('#app button');
  if(button?.textContent.trim()==='Save'&&!button.closest('.ed-save-menu')){event.preventDefault();event.stopImmediatePropagation();save()}
},true);
// Capture before Bento's own file-download shortcut, including while typing.
window.addEventListener('keydown',event=>{if((event.metaKey||event.ctrlKey)&&event.key.toLowerCase()==='s'&&!event.shiftKey){event.preventDefault();event.stopImmediatePropagation();save()}},true);
document.getElementById('authoring-preview').onclick=()=>window.open('/'+config.deck+'/?preview='+Date.now(),'_blank','noopener');

function download(value,name){const url=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}
document.getElementById('authoring-export').onclick=()=>{if(!ready)return;commitText();download({format:'bailiping/draft',version:1,deck:config.deck,baseline,edited:documentNow()},config.deck+'.bailiping.json');show('Draft exported')};
const input=document.createElement('input');input.type='file';input.accept='.json';input.hidden=true;document.body.append(input);
document.getElementById('authoring-import').onclick=()=>input.click();
input.onchange=async()=>{try{const file=input.files[0];if(!file)return;const draft=JSON.parse(await file.text());if(draft.format!=='bailiping/draft'||draft.deck!==config.deck)throw Error('Choose a draft for this presentation.');const {applyChanges}=await import('../assets/authoring-model.mjs');const merged=applyChanges(documentNow(),changesBetween(draft.baseline,draft.edited));if(!window.bento.loadDoc(merged))throw Error('Bento could not read the draft.');show('Draft imported · save to keep these changes')}catch(error){show(error.message,true)}finally{input.value=''}};
document.getElementById('authoring-feedback').onclick=async()=>{
  if(!ready)return;commitText();const comments=window.bento.comments?.().filter(c=>!c.resolved)||[];
  const selected=window.bento.selection||[];
  const lines=[`Please review my ${changed()?'current draft':'saved edits'} for https://bailiping.com/${config.deck}/.`, `Local website: ${config.root}`, `Saved visual edits: ${config.deck}/authoring.json`, ...(selected.length?[`Selected objects: ${selected.join(', ')}`]:[]), ...comments.map(c=>`Slide ${c.slideIndex+1} (${c.slideId})${c.anchor?.elementId?' / '+c.anchor.elementId:''}: ${c.text}`)];
  if(changed())lines.push('I still have unsaved changes in the editor.');
  if(!comments.length)lines.push('Check the layout and interactive demonstrations before publishing.');
  const text=lines.join('\n');
  try{await navigator.clipboard.writeText(text);show('Feedback copied · paste it into our chat')}catch{download({feedback:text},config.deck+'.feedback.json');show('Feedback downloaded')}
};

// Restore authored markup on focus, before Bento selects the editable text.
// This prevents MathJax SVG output from being committed as the text source.
document.addEventListener('focusin',event=>{
  const inner=event.target.closest?.('.bento-text-inner[contenteditable="true"]');if(!inner||!window.bento)return;
  const slide=inner.closest('[data-slide-id]'),element=inner.closest('[data-el-id]');
  const source=window.bento.doc.slides.find(s=>s.id===slide?.dataset.slideId)?.elements.find(e=>e.id===element?.dataset.elId);
  if(source?.type==='text'&&/\\[([]|math-tex/.test(source.html))inner.innerHTML=source.html;
},true);

let mathQueued=false,mathRunning=false;
function queueMath(){if(mathQueued)return;mathQueued=true;setTimeout(renderMath,100)}
async function renderMath(){
  mathQueued=false;if(mathRunning||!window.MathJax?.tex2svgPromise)return;mathRunning=true;
  try{
    for(const span of document.querySelectorAll('.bento-text-inner span')){
      if(span.closest('.bento-editing,[contenteditable="true"]')||span.querySelector('mjx-container')||span.dataset.authoringMath==='pending')continue;
      const text=span.textContent.trim(),display=text.startsWith('\\[')&&text.endsWith('\\]'),inline=text.startsWith('\\(')&&text.endsWith('\\)');
      if(!display&&!inline)continue;
      span.dataset.authoringMath='pending';
      try{const rendered=await MathJax.tex2svgPromise(text.slice(2,-2),{display});if(span.isConnected&&!span.closest('.bento-editing,[contenteditable="true"]')&&span.textContent.trim()===text)span.replaceChildren(rendered)}finally{delete span.dataset.authoringMath}
    }
  }finally{mathRunning=false}
}
new MutationObserver(queueMath).observe(document.body,{childList:true,subtree:true,attributes:true,attributeFilter:['contenteditable']});
window.addEventListener('mathjax-ready',queueMath);

const start=Date.now();
const timer=setInterval(()=>{
  if(!ready){
    if(!window.bento?.loadDoc){if(Date.now()-start>25000){show('The editor is still loading. Check the connection and reload.',true);clearInterval(timer)}return}
    baseline=documentNow();saved=copy(baseline);ready=true;saveButton.disabled=false;show('Ready · edits save on this Mac');queueMath();
    window.MathJax?.startup?.promise?.then(queueMath).catch(()=>{});
  }
  if(!saving&&changed()&&!status.hasAttribute('data-error'))show('Unsaved changes · ⌘S to save');
},800);
window.addEventListener('beforeunload',event=>{if(changed()){event.preventDefault();event.returnValue=''}else event.stopImmediatePropagation()},true);
