(function(){
const M=window.RadarModel;
const C={ink:'#203129',muted:'#66756e',green:'#2f6b4f',blue:'#496e87',orange:'#a94f2a',purple:'#85679d',grid:'#e2e7dd',gray:'#9ca79b',red:'#b5594b'};
const finite=(v,d=2)=>Number.isFinite(v)?v.toFixed(d):'—';
function text(c,s,x,y,color=C.muted,size=11,align='left'){c.save();c.fillStyle=color;c.font=`${size}px system-ui, sans-serif`;c.textAlign=align;c.fillText(s,x,y);c.restore();}
function axes(c,box,bounds,xlabel,ylabel,equal=false){
 let [xmin,xmax,ymin,ymax]=bounds;const {x,y,w,h}=box;
 if(equal){const r=w/h,spanX=xmax-xmin,spanY=ymax-ymin;if(spanX/spanY<r){const mid=(xmax+xmin)/2,d=spanY*r/2;xmin=mid-d;xmax=mid+d;}else{const mid=(ymax+ymin)/2,d=spanX/r/2;ymin=mid-d;ymax=mid+d;}}
 const X=v=>x+(v-xmin)/(xmax-xmin)*w,Y=v=>y+h-(v-ymin)/(ymax-ymin)*h;
 c.save();c.strokeStyle=C.grid;c.lineWidth=1;
 for(let i=0;i<=4;i++){const xx=x+w*i/4,yy=y+h*i/4;c.beginPath();c.moveTo(xx,y);c.lineTo(xx,y+h);c.stroke();c.beginPath();c.moveTo(x,yy);c.lineTo(x+w,yy);c.stroke();text(c,finite(xmin+(xmax-xmin)*i/4,(xmax-xmin)<5?1:0),xx,y+h+17,C.muted,9,'center');text(c,finite(ymax-(ymax-ymin)*i/4,(ymax-ymin)<5?1:0),x-8,yy+3,C.muted,9,'right');}
 c.strokeStyle='#b8c5b8';c.strokeRect(x,y,w,h);text(c,xlabel,x+w/2,y+h+34,C.muted,10,'center');c.save();c.translate(x-32,y+h/2);c.rotate(-Math.PI/2);text(c,ylabel,0,0,C.muted,10,'center');c.restore();c.restore();
 function clip(fn){c.save();c.beginPath();c.rect(x,y,w,h);c.clip();fn();c.restore();}
 function line(points,color=C.green,width=1.8,dash=[]){clip(()=>{c.strokeStyle=color;c.lineWidth=width;c.setLineDash(dash);c.beginPath();let started=false;for(const p of points){if(p[1]===null||!Number.isFinite(p[1])){started=false;continue;}if(!started){c.moveTo(X(p[0]),Y(p[1]));started=true;}else c.lineTo(X(p[0]),Y(p[1]));}c.stroke();});}
 function dot(p,color=C.green,r=2.5,hollow=false,cross=false){clip(()=>{c.fillStyle=color;c.strokeStyle=color;c.lineWidth=1.2;if(cross){c.beginPath();c.moveTo(X(p[0])-r,Y(p[1])-r);c.lineTo(X(p[0])+r,Y(p[1])+r);c.moveTo(X(p[0])+r,Y(p[1])-r);c.lineTo(X(p[0])-r,Y(p[1])+r);c.stroke();}else{c.beginPath();c.arc(X(p[0]),Y(p[1]),r,0,2*Math.PI);hollow?c.stroke():c.fill();}});}
 function pose(t,color=C.green){clip(()=>{const xx=X(t[0]),yy=Y(t[1]),a=-t[2];c.save();c.translate(xx,yy);c.rotate(a);c.fillStyle=color;c.beginPath();c.moveTo(10,0);c.lineTo(-6,-5);c.lineTo(-3,0);c.lineTo(-6,5);c.closePath();c.fill();c.restore();});}
 return {X,Y,line,dot,pose,clip,bounds:[xmin,xmax,ymin,ymax]};
}
function pairBoxes(w,h){return w<500?[{x:43,y:28,w:w-61,h:190},{x:43,y:290,w:w-61,h:225}]:[{x:42,y:31,w:w/2-61,h:h-82},{x:w/2+34,y:31,w:w/2-51,h:h-82}];}
function drawMapped(plot,mission,poses,last,step=1){for(let k=0;k<=last;k+=step){mission.scans[k].forEach((q,j)=>{if(mission.accepted[k].has(j))plot.dot(M.transform(poses[k],q),'#78a286',1.15);});}}
window.RadarRender={C,text,axes,pairBoxes,drawMapped};
})();
