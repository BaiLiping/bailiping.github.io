/* Exact linear-quadratic teaching model, shared by browser and Node tests. */
(function(root){
'use strict';
const child=(u,s)=>0.5*(u-1)**2+0.5*(s-u)**2;
const summary=s=>0.25*(s-1)**2;
const recover=s=>(s+1)/2;
const external=(s,z)=>0.5*(s-z)**2;
function solve(z){
  if(!Number.isFinite(z))throw new TypeError('The external target must be finite');
  const s=(1+2*z)/3,u=recover(s);
  return {z,s,u,cost:summary(s)+external(s,z)};
}
const api={child,summary,recover,external,solve};
if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.ISAMSummary=api;
})(typeof globalThis!=='undefined'?globalThis:this);
