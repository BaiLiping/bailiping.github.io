/* Part III model modules. */
window.SLAM_LOADS=window.SLAM_LOADS||[];
for(let n=13;n<=18;n++){window.SLAM_LOADS.push(new Promise((resolve,reject)=>{const s=document.createElement('script');s.src='./model-'+n+'.js';s.onload=resolve;s.onerror=reject;document.head.appendChild(s)}))}
