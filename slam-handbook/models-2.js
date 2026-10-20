/* First-party Part II computational models. */
window.SLAM_LOADS=window.SLAM_LOADS||[];
for(let n=7;n<=12;n++){
 window.SLAM_LOADS.push(new Promise((resolve,reject)=>{
 const el=document.createElement("script");el.src="./model-"+String(n).padStart(2,"0")+".js";el.onload=resolve;el.onerror=()=>reject(Error("Missing model for chapter "+n));document.head.appendChild(el);
 }));
}
