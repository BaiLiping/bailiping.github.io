import * as M from './model.mjs';
export const colors={ink:'#213D36',muted:'#64746D',teal:'#137B69',gold:'#BC702D',purple:'#8063AB',line:'#D4DDD5',paper:'#F8F6F0'};
export function setupCanvas(canvas) {
 const {width,height}=canvas.getBoundingClientRect(),dpr=Math.min(devicePixelRatio||1,2);
 canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr);
 const c=canvas.getContext('2d');c.scale(dpr,dpr);c.clearRect(0,0,width,height);
 return {c,w:width,h:height};
}
export function camera(c,w,h,{yaw=.55,pitch=.45,scale=1,cx=w/2,cy=h*.54}={}) {
 const unit=Math.min(w,h)*.29*scale;
 function project(p) { const x=Math.cos(yaw)*p[0]-Math.sin(yaw)*p[1],d=Math.sin(yaw)*p[0]+Math.cos(yaw)*p[1];return [cx+unit*x,cy-unit*(Math.cos(pitch)*p[2]-Math.sin(pitch)*d)]; }
 function path(points,color=colors.line,width=1,fill=null,dash=[]) {
  c.beginPath();points.forEach((p,i)=>{const [x,y]=project(p);i?c.lineTo(x,y):c.moveTo(x,y);});
  if(fill){c.closePath();c.fillStyle=fill;c.fill();}c.strokeStyle=color;c.lineWidth=width;c.setLineDash(dash);c.stroke();c.setLineDash([]);
 }
 function text(p,t,color=colors.ink,dx=7,dy=-7){const [x,y]=project(p);label(c,t,x+dx,y+dy,color);}
 function vector(v,color,labelText,start=[0,0,0]) {
  path([start,v],color,2.5);const a=project(start),b=project(v),ang=Math.atan2(b[1]-a[1],b[0]-a[0]);
  c.beginPath();c.moveTo(...b);c.lineTo(b[0]-8*Math.cos(ang-.42),b[1]-8*Math.sin(ang-.42));c.lineTo(b[0]-8*Math.cos(ang+.42),b[1]-8*Math.sin(ang+.42));c.closePath();c.fillStyle=color;c.fill();
  if(labelText)text(v,labelText,color);
 }
 function dot(p,color=colors.teal,r=3){const v=project(p);c.beginPath();c.arc(...v,r,0,2*Math.PI);c.fillStyle=color;c.fill();}
 function plane(Q,{center=[0,0,0],radius=1.4,color=colors.teal,fill='#137B691C',grid=true}={}) {
  const at=(a,b)=>Q.map((r,i)=>r[0]*a+r[1]*b+center[i]);
  path([at(-radius,-radius),at(radius,-radius),at(radius,radius),at(-radius,radius)],color,1.2,fill);
  if(grid) for(let t=-1;t<=1;t+=.5){path([at(-radius,t*radius),at(radius,t*radius)],color+'38');path([at(t*radius,-radius),at(t*radius,radius)],color+'38');}
 }
 function axes(length=1.6){[[length,0,0],[0,length,0],[0,0,length]].forEach((p,i)=>{path([p.map(x=>-x*.7),p],colors.line,1);text(p,['x','y','z'][i],colors.muted);});}
 return {project,path,text,vector,dot,plane,axes};
}
export function label(c,t,x,y,color=colors.ink,size=12){c.fillStyle=color;c.font=`${size}px system-ui, sans-serif`;c.fillText(t,x,y);}
export function basisPicture(ctx,Q,Y,view={}) {
 const {c,w,h}=ctx,v=camera(c,w,h,view);v.axes();v.plane(Q);
 const point=[.8,.55,1.25],p=M.projector(Q).map(r=>M.dot(r,point));v.path([point,p],colors.purple,1.5,null,[4,4]);v.dot(point,colors.purple);v.dot(p,colors.purple);v.text(point,'x',colors.purple);v.text(p,'Px',colors.purple);
 M.transpose(Q).forEach((u,i)=>v.vector(u.map(x=>x*1.18),colors.gold,'q'+(i+1)));
 M.transpose(Y).forEach((u,i)=>v.vector(u.map(x=>x*1.18),colors.teal,'y'+(i+1)));
 label(c,'Gold: Q     Teal: QO     Violet: a vector and its projection',16,h-15,colors.muted,11);
}
export function anglesPicture(ctx,a,b) {
 const {c,w,h}=ctx,r=Math.min(w*.18,h*.28),cy=h*.50;
 [a,b].forEach((angle,i)=>{
  const cx=w*(i?.75:.25),xy=t=>[cx+r*Math.cos(t),cy-r*Math.sin(t)];
  c.strokeStyle=colors.line;c.lineWidth=1;c.beginPath();c.arc(cx,cy,r,0,2*Math.PI);c.stroke();
  c.beginPath();c.moveTo(cx-r-15,cy);c.lineTo(cx+r+15,cy);c.moveTo(cx,cy+r+15);c.lineTo(cx,cy-r-15);c.stroke();
  c.beginPath();c.moveTo(cx,cy);c.arc(cx,cy,r*.5,0,-angle,true);c.closePath();c.fillStyle='#BC702D20';c.fill();
  [[0,colors.gold],[angle,colors.teal]].forEach(([t,col])=>{c.beginPath();c.moveTo(cx,cy);c.lineTo(...xy(t));c.lineWidth=3;c.strokeStyle=col;c.stroke();});
  label(c,'e'+(i+1),cx+r+9,cy+19,colors.muted);label(c,'e'+(i+3),cx+8,cy-r-9,colors.muted);
  label(c,M.deg(angle).toFixed(0)+'°',cx-15,cy+r+40,colors.teal,22);
  label(c,`Principal-coordinate slice ${i+1}`,cx-r,25,colors.muted,12);
 });
 label(c,'Two independent 2-D slices of R⁴; not a 3-D projection of the whole space.',16,h-12,colors.muted,11);
}
export function geodesicPicture(ctx,u,v,t,view={}) {
 const {c,w,h}=ctx,cam=camera(c,w,h,{...view,scale:1.12});
 for(const z of [-.5,0,.5]) {const r=Math.sqrt(1-z*z);cam.path(Array.from({length:65},(_,i)=>[r*Math.cos(i*Math.PI/32),r*Math.sin(i*Math.PI/32),z]),colors.line,1);}
 for(let a=0;a<Math.PI;a+=Math.PI/4)cam.path(Array.from({length:65},(_,i)=>[Math.cos(a)*Math.cos(i*Math.PI/32),Math.sin(a)*Math.cos(i*Math.PI/32),Math.sin(i*Math.PI/32)]),colors.line,1);
 const result=M.lineGeodesic(u,v,t),path=Array.from({length:65},(_,i)=>M.lineGeodesic(u,v,i/64).point);
 cam.path(path,colors.teal,3);cam.path(path.map(p=>p.map(x=>-x)),colors.teal,1,null,[4,4]);
 cam.vector(u,colors.gold,'[u]');cam.vector(result.aligned,colors.purple,'[v]');cam.vector(result.point,colors.teal,'q(t)');cam.path([result.point,result.point.map(x=>-x)],colors.teal,1.2);cam.dot(result.point.map(x=>-x),colors.teal);
 label(c,'Antipodal points represent the same line. The sphere is a double cover.',16,h-12,colors.muted,11);
}
export function pcaPicture(ctx,data,Q,history,view={}) {
 const {c,w,h}=ctx,v=camera(c,w,h,{...view,scale:.27,cy:h*.47});v.plane(Q,{radius:4,color:colors.teal});
 data.points.forEach(p=>{v.dot(p,colors.purple+'B0',2.2);});
 const y=h-55,left=35,right=w-20,lo=history[0],span=Math.max(.1,data.optimum-lo);
 c.strokeStyle=colors.line;c.beginPath();c.moveTo(left,y);c.lineTo(right,y);c.stroke();
 c.strokeStyle=colors.teal;c.lineWidth=2;c.beginPath();history.forEach((s,i)=>{const x=left+(right-left)*i/Math.max(35,history.length-1),yy=y-35*(s-lo)/span;i?c.lineTo(x,yy):c.moveTo(x,yy);});c.stroke();
 label(c,'Captured variance per iteration →',left,h-17,colors.muted,11);
}
export function associationPicture(ctx,scene,result,view={}) {
 const {c,w,h}=ctx,centers=[w*.25,w*.75],cams=centers.map(cx=>camera(c,w/2,h,{...view,scale:.31,cx,cy:h*.53}));
 label(c,'SCAN A · 6 objects',16,24,colors.ink,13);label(c,`SCAN B · ${scene.target.length} objects`,w/2+16,24,colors.ink,13);
 label(c,'Each scan has its own sensor frame. Object sizes are display windows.',16,h-10,colors.muted,10);
 const selected=result.selected.map(u=>result.candidates[u]),occupied=[[],[]];
 [scene.source,scene.target].forEach((objects,s)=>objects.forEach((o,i)=>{
  const active=selected.some(m=>(s?m.j:m.i)===i),col=active?colors.teal:colors.muted;
  // A display offset centers scan B without changing any descriptor calculation.
  const b=o.b.map((x,k)=>x-(s?scene.t[k]:0));
  if(o.type==='line'){const d=M.transpose(o.A)[0];cams[s].path([b.map((x,k)=>x-d[k]*1.1),b.map((x,k)=>x+d[k]*1.1)],col,active?4:2);}
  else cams[s].plane(o.A,{center:b,radius:.65,color:col,fill:active?'#137B6930':'#64746D10',grid:false});
  const [px,py]=cams[s].project(b);let dy=-9;
  for(const candidate of [-9,-23,17,31]){dy=candidate;if(!occupied[s].some(([x,y])=>Math.abs(x-px)<26&&Math.abs(y-(py+dy))<13))break;}
  occupied[s].push([px,py+dy]);cams[s].text(b,(s?'B':'A')+(i+1),col,5,dy);
 }));
 selected.forEach(({i,j})=>{
  const a=cams[0].project(scene.source[i].b),b=cams[1].project(scene.target[j].b.map((x,k)=>x-scene.t[k]));
  c.beginPath();c.moveTo(...a);c.bezierCurveTo(w*.46,a[1],w*.54,b[1],...b);c.lineWidth=1.4;c.strokeStyle='#137B6970';c.stroke();
 });
}
export function graphPicture(ctx,result) {
 const {c,w,h}=ctx,n=result.weights.length,side=Math.min(h-34,w*.55),cell=side/n,x=18,y=18,selected=new Set(result.selected);
 result.weights.forEach((row,i)=>row.forEach((v,j)=>{c.fillStyle=v>0?`rgba(19,123,105,${.12+.88*v})`:'#EBEAE4';c.fillRect(x+j*cell,y+i*cell,cell-.5,cell-.5);}));
 c.strokeStyle=colors.gold;c.lineWidth=1.5;result.selected.forEach(i=>c.strokeRect(x-.5,y+i*cell-.5,side+1,cell+1));
 label(c,'CONSISTENCY GRAPH',side+40,30,colors.ink,11);
 label(c,w<550?`${n} candidate vertices`:`${n} candidate matches = graph vertices`,side+40,54,colors.muted,11);
 label(c,w<550?'Teal: weight · pale: forbidden':'Teal: Eq. (8) weight; pale: forbidden',side+40,76,colors.muted,11);
 label(c,'Gold rows: selected clique',side+40,98,colors.gold,11);
 label(c,`${result.visited} feasible cliques evaluated`,side+40,125,colors.muted,11);
}
