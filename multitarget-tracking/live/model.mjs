export const cases = [
  {id:'occlusion',title:'Occlusion',still:'figure-02.png',animation:'figure-01.gif',observation:'The original note identifies 13 frames of missed pedestrian detection.',watch:'Watch the gap between detections and the continuity of the estimated track.',source:'2270d664735d81f9ac14d58c1520ce94'},
  {id:'crossing',title:'Crossing tracks',still:'figure-05.png',animation:'figure-06.gif',observation:'The source attributes an identity switch to intersecting tracks.',watch:'Follow the trajectories through the crossing, where association becomes ambiguous.',source:'2270d664735d819abb00c2ec8fc3a1ba'},
  {id:'bus',title:'Bus-track intersection',still:'figure-03.png',animation:'figure-04.gif',observation:'Two bus tracks intersect; the notes record termination and restart.',watch:'Look for the point at which the original track stops and another begins.',source:'2270d664735d81378717c61dac5cfc19'},
  {id:'classification',title:'Alternating classification',still:'figure-22.png',animation:'figure-21.gif',observation:'The notes collect examples of unstable class labels and early track termination.',watch:'Inspect the changing class assignments across frames. The still is a related example from the same section.',source:'2270d664735d81cebc49e2f9fcd1e55f'}
];
export function initialState(){return {selected:0,playing:false,visible:true}}
export function transition(state,action){
  if(action.type==='select')return {...state,selected:Math.max(0,Math.min(cases.length-1,Number(action.index)||0)),playing:false};
  if(action.type==='play')return {...state,playing:!state.playing};
  if(action.type==='visibility')return {...state,visible:!!action.visible,playing:action.visible?state.playing:false};
  return initialState();
}
export function currentView(state){const item=cases[state.selected];return {...item,src:'../assets/'+(state.playing&&state.visible?item.animation:item.still),playing:state.playing&&state.visible}}
