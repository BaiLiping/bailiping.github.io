import {cases,initialState,transition,currentView} from './model.mjs';
let state=initialState();
if(new URLSearchParams(location.search).get('embed')==='region')document.documentElement.dataset.embedded='true';
const scene=document.querySelector('#scene'),picture=document.querySelector('#picture'),play=document.querySelector('#play');
scene.innerHTML=cases.map((c,i)=>`<option value="${i}">${c.title}</option>`).join('');
function render(){const v=currentView(state);if(picture.getAttribute('src')!==v.src)picture.src=v.src;picture.alt=`Original nuScenes visualization: ${v.title}. ${v.observation}`;document.querySelector('#observation').textContent=v.observation;document.querySelector('#watch').textContent=v.watch;play.textContent=v.playing?'Stop animation':'Play animation';play.setAttribute('aria-pressed',String(v.playing));document.querySelector('#source').href='https://www.notion.so/2270d664735d8183bd8bc130601992e2#'+v.source;document.querySelector('#full-image').href=v.src;scene.value=state.selected;document.querySelector('#status').textContent=v.playing?'Playing at 2× · loops continuously':'Still image';}
scene.addEventListener('change',()=>{state=transition(state,{type:'select',index:scene.value});render()});
play.addEventListener('click',()=>{state=transition(state,{type:'play'});render()});
window.addEventListener('message',event=>{if(event.source!==parent)return;if(event.data?.type==='bento-live-pause'||event.data?.type==='bento-live-resume'){state=transition(state,{type:'visibility',visible:event.data.type==='bento-live-resume'});render()}});
document.addEventListener('visibilitychange',()=>{if(document.hidden){state=transition(state,{type:'visibility',visible:false});render()}else{state={...state,visible:true};render()}});
picture.addEventListener('error',()=>{document.querySelector('#status').textContent='Image unavailable. Open the source notes using the link below.'});
render();
