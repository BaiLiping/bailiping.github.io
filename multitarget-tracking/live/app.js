(()=>{
const cases = [
  {id:'occlusion',title:'Occlusion',still:'figure-02.png',animation:'figure-01.gif',observation:'The original note identifies 13 frames of missed pedestrian detection.',watch:'Watch the gap between detections and the continuity of the estimated track.',source:'2270d664735d81f9ac14d58c1520ce94'},
  {id:'crossing',title:'Crossing tracks',still:'figure-05.png',animation:'figure-06.gif',observation:'The source attributes an identity switch to intersecting tracks.',watch:'Follow the trajectories through the crossing, where association becomes ambiguous.',source:'2270d664735d819abb00c2ec8fc3a1ba'},
  {id:'bus',title:'Bus-track intersection',still:'figure-03.png',animation:'figure-04.gif',observation:'Two bus tracks intersect; the notes record termination and restart.',watch:'Look for the point at which the original track stops and another begins.',source:'2270d664735d81378717c61dac5cfc19'},
  {id:'classification',title:'Alternating classification',still:'figure-22.png',animation:'figure-21.gif',observation:'The notes collect examples of unstable class labels and early track termination.',watch:'Inspect the changing class assignments across frames. The still is a related example from the same section.',source:'2270d664735d81cebc49e2f9fcd1e55f'}
];
function initialState(){return {selected:0,playing:true,visible:true}}
function transition(state,action){
  if(action.type==='select')return {...state,selected:Math.max(0,Math.min(cases.length-1,Number(action.index)||0)),playing:true};
  if(action.type==='play')return {...state,playing:!state.playing};
  if(action.type==='visibility')return {...state,visible:!!action.visible};
  return initialState();
}
function currentView(state){const item=cases[state.selected];return {...item,src:'../assets/'+(state.playing&&state.visible?'playback/'+item.animation.replace('.gif','-2x.gif'):item.still),playing:state.playing&&state.visible}}

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

})();
