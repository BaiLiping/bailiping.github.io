/* Load Part I numerical models without third-party libraries. */
window.SLAM_MODELS=window.SLAM_MODELS||{};
window.SLAM_UTIL=window.SLAM_UTIL||{
 clamp:(x,a,b)=>Math.min(b,Math.max(a,x)),
 range:(a,b,n=80)=>Array.from({length:n+1},(_,i)=>a+(b-a)*i/n),
 line:(name,points)=>({name,points}),
 out:(metrics,insight,plot)=>({metrics,insight,plot}),
 curve:(xLabel,yLabel,...series)=>({xLabel,yLabel,series})
};
window.SLAM_LOADS=window.SLAM_LOADS||[];
for(let n=1;n<=6;n++){
 window.SLAM_LOADS.push(new Promise((resolve,reject)=>{
 const s=document.createElement('script');s.src='./model-'+String(n).padStart(2,'0')+'.js';s.onload=resolve;s.onerror=()=>reject(Error('Could not load experiments for chapter '+n));document.head.appendChild(s);
 }));
}
