(function(){
"use strict";
/* ---------- shared helpers ---------- */
const M_PER_PX=0.1, C_M_PER_NS=0.2998;
const dist=(a,b)=>Math.hypot(a[0]-b[0],a[1]-b[1]);
const sub=(a,b)=>[a[0]-b[0],a[1]-b[1]];
const unit=v=>{const n=Math.hypot(v[0],v[1])||1;return [v[0]/n,v[1]/n];};
/* mirror point p across the line through Q with unit normal n */
const mirrorLine=(p,Q,n)=>{const d=(p[0]-Q[0])*n[0]+(p[1]-Q[1])*n[1];return [p[0]-2*d*n[0],p[1]-2*d*n[1]];};
/* intersection of line A->B with the line through Q of unit normal n */
const hitLine=(A,B,Q,n)=>{const d=sub(B,A);const t=((Q[0]-A[0])*n[0]+(Q[1]-A[1])*n[1])/(d[0]*n[0]+d[1]*n[1]);return [A[0]+t*d[0],A[1]+t*d[1]];};
/* ordered ray/line intersection: unlike hitLine, retain the signed distance so
   a construction can reject a bounce behind the ray origin */
const hitRayLine=(O,d,Q,n)=>{
  const den=d[0]*n[0]+d[1]*n[1];
  if(!Number.isFinite(den)||Math.abs(den)<1e-9)return null;
  const t=((Q[0]-O[0])*n[0]+(Q[1]-O[1])*n[1])/den;
  if(!Number.isFinite(t))return null;
  return {p:[O[0]+t*d[0],O[1]+t*d[1]],t};
};
const txt=(x,y,s,c,sz,anchor)=>`<text x="${x}" y="${y}" fill="${c??'#51606e'}" font-size="${sz??11}" font-family="ui-monospace,Menlo,monospace" text-anchor="${anchor??'start'}">${s}</text>`;
const seg=(a,b,c,w,dash,o)=>`<line x1="${a[0]}" y1="${a[1]}" x2="${b[0]}" y2="${b[1]}" stroke="${c}" stroke-width="${w??1.6}" ${dash?`stroke-dasharray="${dash}"`:""} opacity="${o??1}"/>`;
function arrow(a,b,c,w,dash,o){
  const ang=Math.atan2(b[1]-a[1],b[0]-a[0]),h=7;
  const p1=[b[0]-h*Math.cos(ang-0.42),b[1]-h*Math.sin(ang-0.42)],p2=[b[0]-h*Math.cos(ang+0.42),b[1]-h*Math.sin(ang+0.42)];
  return `<g opacity="${o??1}">`+seg(a,b,c,w,dash)+`<path d="M${b[0]},${b[1]} L${p1[0]},${p1[1]} L${p2[0]},${p2[1]} Z" fill="${c}"/></g>`;
}
const headingMark=(p,theta)=>{
  const h=dirOf(theta),q=[p[0]+32*h[0],p[1]+32*h[1]];
  return arrow(p,q,"#2ca02c",1.7,null,0.9)+txt(q[0]+5,q[1]-5,"body x · θ","#1d7a1d",9.5);
};
function ellipse(F1,F2,L,stroke,w,dash,o){
  const c2=dist(F1,F2)/2; if(L/2<=c2+1) return "";
  const a=L/2,b=Math.sqrt(a*a-c2*c2);
  const cx=(F1[0]+F2[0])/2, cy=(F1[1]+F2[1])/2;
  const ang=Math.atan2(F2[1]-F1[1],F2[0]-F1[0])*180/Math.PI;
  return `<ellipse cx="${cx}" cy="${cy}" rx="${a}" ry="${b}" transform="rotate(${ang.toFixed(2)} ${cx} ${cy})" fill="none" stroke="${stroke}" stroke-width="${w??1.6}" ${dash?`stroke-dasharray="${dash}"`:""} opacity="${o??1}"/>`;
}
/* parametric ellipse from two foci + string length (for the pin-and-string trace) */
function ellParams(F1,F2,L){
  const c2=dist(F1,F2)/2; if(L/2<=c2+1) return null;
  const a=L/2,b=Math.sqrt(a*a-c2*c2);
  return {cx:(F1[0]+F2[0])/2,cy:(F1[1]+F2[1])/2,a,b,ang:Math.atan2(F2[1]-F1[1],F2[0]-F1[0])};
}
function ellPoint(p,th){
  const x=p.a*Math.cos(th),y=p.b*Math.sin(th),ca=Math.cos(p.ang),sa=Math.sin(p.ang);
  return [p.cx+x*ca-y*sa,p.cy+x*sa+y*ca];
}
function ellPartial(p,th1,stroke,w,dash,o){
  let d="";const n=Math.max(2,Math.ceil(th1/0.06));
  for(let i=0;i<=n;i++){const q=ellPoint(p,th1*i/n);d+=(i?"L":"M")+q[0].toFixed(1)+","+q[1].toFixed(1);}
  return `<path d="${d}" fill="none" stroke="${stroke}" stroke-width="${w}" ${dash?`stroke-dasharray="${dash}"`:""} opacity="${o??1}"/>`;
}
/* the string: two taut segments from the pins (foci) to the pencil at E */
function stringViz(F1,F2,E,color,eq,labelPos){
  const d2=dist(E,F1)*M_PER_PX,d1=dist(E,F2)*M_PER_PX;
  return seg(F1,E,color,1.3,"2 3",0.9)+seg(F2,E,color,1.3,"2 3",0.9)
    +`<circle cx="${E[0]}" cy="${E[1]}" r="4.2" fill="${color}"/>`
    +txt((F2[0]+E[0])/2+6,(F2[1]+E[1])/2-6,d1.toFixed(1),color,10)
    +txt((F1[0]+E[0])/2+6,(F1[1]+E[1])/2-6,d2.toFixed(1),color,10)
    +txt(labelPos[0],labelPos[1],`${eq} = ${d1.toFixed(1)} + ${d2.toFixed(1)} = ${(d1+d2).toFixed(1)} m — constant`,color,11.5);
}
/* dashed ring marking a focus (pin) of an ellipse, in that ellipse's color */
const pin=(p,c,r)=>`<circle cx="${p[0]}" cy="${p[1]}" r="${r}" fill="none" stroke="${c}" stroke-width="1.6" stroke-dasharray="3 3" opacity="0.9"/>`;
const reduced=matchMedia("(prefers-reduced-motion: reduce)").matches;
/* intersection of ray (origin O = one focus, direction u) with ellipse foci (F,O), sum L */
function rayEllipse(F,O,u,L){
  const d=sub(F,O), t=(L*L-(d[0]*d[0]+d[1]*d[1]))/(2*(L-(u[0]*d[0]+u[1]*d[1])));
  return [O[0]+t*u[0],O[1]+t*u[1]];
}
const bsMark=(p,label,lx,ly)=>`<rect x="${p[0]-6}" y="${p[1]-6}" width="12" height="12" fill="#16222e"/>`+txt(p[0]+(lx??10),p[1]+(ly??4),label??"BS","#16222e",12);
const ueMark=(p,label)=>`<circle cx="${p[0]}" cy="${p[1]}" r="7" fill="#2ca02c" stroke="#fff" stroke-width="2"/><circle cx="${p[0]}" cy="${p[1]}" r="12" fill="none" stroke="#2ca02c" stroke-dasharray="2 3" opacity="0.7"/>`+txt(p[0]+15,p[1]+4,label??"UE","#1d7a1d",12);
const vaMark=(p,label,o)=>`<g opacity="${o??1}"><rect x="${p[0]-5.5}" y="${p[1]-5.5}" width="11" height="11" transform="rotate(45 ${p[0]} ${p[1]})" fill="none" stroke="#7c4dbe" stroke-width="2.4"/></g>`+txt(p[0]+14,p[1]-10,label,"#5d3691",12);
const vaDot=p=>`<rect x="${p[0]-4}" y="${p[1]-4}" width="8" height="8" transform="rotate(45 ${p[0]} ${p[1]})" fill="#7c4dbe"/>`;
const ipMark=(p,label,dx,dy)=>`<circle cx="${p[0]}" cy="${p[1]}" r="5" fill="#0e8f7e"/>`+txt(p[0]+(dx??9),p[1]+(dy??-8),label,"#0a6b5e",12);
function fmt(Lpx){const m=Lpx*M_PER_PX;return `${m.toFixed(1)} m · τ ${(m/C_M_PER_NS).toFixed(0)} ns`;}
const aoaOf=u=>((-Math.atan2(u[1],u[0])*180/Math.PI)+360)%360;
const degFmt=a=>{const d=((a%360)+540)%360-180;return d.toFixed(1)+"°";};   // display in (−180°, 180°]
const dirOf=phiDeg=>{const a=phiDeg*Math.PI/180;return [Math.cos(a),-Math.sin(a)];};
/* seeded gaussians (deterministic across loads) */
function gaussians(seed,n){
  let s=seed>>>0; const out=[];
  const rnd=()=>{s|=0;s=(s+0x6D2B79F5)|0;let t=Math.imul(s^(s>>>15),1|s);t=(t+Math.imul(t^(t>>>7),61|t))^t;return((t^(t>>>14))>>>0)/4294967296;};
  for(let i=0;i<n;i++){const u1=Math.max(rnd(),1e-9),u2=rnd();out.push(Math.sqrt(-2*Math.log(u1))*Math.cos(2*Math.PI*u2));}
  return out;
}
const G1=gaussians(11,34), G2=gaussians(77,34);
/* dragging */
function svgPt(svg,ev){
  const r=svg.getBoundingClientRect(), vb=svg.viewBox.baseVal;
  return [vb.x+(ev.clientX-r.left)/r.width*vb.width, vb.y+(ev.clientY-r.top)/r.height*vb.height];
}
function makeDraggable(svg,getP,setP,box,render){
  let drag=false;
  svg.setAttribute("tabindex","0");
  svg.setAttribute("aria-keyshortcuts","ArrowLeft ArrowRight ArrowUp ArrowDown");
  svg.addEventListener("pointerdown",ev=>{
    const p=svgPt(svg,ev);
    if(dist(p,getP())<26){drag=true;svg.setPointerCapture(ev.pointerId);ev.preventDefault();}
  });
  svg.addEventListener("pointermove",ev=>{
    if(!drag)return;
    const p=svgPt(svg,ev);
    setP([Math.max(box[0],Math.min(box[2],p[0])),Math.max(box[1],Math.min(box[3],p[1]))]);
    render();
  });
  const up=()=>{drag=false;};
  svg.addEventListener("pointerup",up); svg.addEventListener("pointercancel",up);
  svg.addEventListener("keydown",ev=>{
    const delta={ArrowLeft:[-1,0],ArrowRight:[1,0],ArrowUp:[0,-1],ArrowDown:[0,1]}[ev.key];
    if(!delta)return;
    const p=getP(),step=ev.shiftKey?15:5;
    setP([Math.max(box[0],Math.min(box[2],p[0]+step*delta[0])),Math.max(box[1],Math.min(box[3],p[1]+step*delta[1]))]);
    render();ev.preventDefault();
  });
}
const $=id=>document.getElementById("u"+id);
document.querySelectorAll('#unknown-map input[type="range"],#unknown-pose-map input[type="range"]').forEach(input=>{
  if(input.getAttribute("aria-label")||input.getAttribute("aria-labelledby"))return;
  let label=input.previousElementSibling;
  while(label&&label.tagName!=="LABEL")label=label.previousElementSibling;
  const name=label?.querySelector("span")?.textContent||label?.textContent||input.id;
  input.setAttribute("aria-label",name.trim());
});
document.querySelectorAll('#unknown-map .statline,#unknown-map .hint[id^="ucap"],#unknown-pose-map .statline,#unknown-pose-map .hint[id^="ucap"]').forEach(el=>el.setAttribute("aria-live","polite"));

/* =================== DEMO 1: single bounce, data-driven =================== */
const svg1=$("svg1");
const T1={on:false,th:0}; let raf1=null;
/* the wall is tilted 18° off vertical: an axis-aligned wall would make the global
   AoA and AoD come out as mirror images (equal magnitudes) — the reflection law only
   ties the two legs to equal angles about the wall NORMAL, not to each other */
const W1TILT=18*Math.PI/180;
const D1={BS:[110,280],UE:[170,380],WC:[300,150],WD:[Math.sin(W1TILT),Math.cos(W1TILT)],WLEN:330};
D1.WN=[D1.WD[1],-D1.WD[0]];
D1.WEND=[D1.WC[0]+D1.WD[0]*D1.WLEN,D1.WC[1]+D1.WD[1]*D1.WLEN];
const mirrorW=p=>{const d=(p[0]-D1.WC[0])*D1.WN[0]+(p[1]-D1.WC[1])*D1.WN[1];
  return [p[0]-2*d*D1.WN[0],p[1]-2*d*D1.WN[1]];};
D1.VAtrue=mirrorW(D1.BS);
const ck1=id=>$(id).checked;
function specular1(){ // the data a clean estimator would return at the current UE
  const dV=sub(D1.VAtrue,D1.UE);
  const t=((D1.WC[0]-D1.UE[0])*D1.WN[0]+(D1.WC[1]-D1.UE[1])*D1.WN[1])/(dV[0]*D1.WN[0]+dV[1]*D1.WN[1]);
  const Pt=[D1.UE[0]+t*dV[0],D1.UE[1]+t*dV[1]];
  return {L:dist(D1.UE,D1.VAtrue), phi:aoaOf(unit(dV)), psi:aoaOf(unit(sub(Pt,D1.BS))), Pt};
}
function render1(){
  const {BS,UE,WC,WN,WEND}=D1;
  const sp=specular1();
  const sig=+$("sSig1").value;
  $("oSig1").textContent=(sig*M_PER_PX).toFixed(1)+" m";
  const L=sp.L + (+$("s1L").value)/M_PER_PX;   // measured length (data)
  const phi=sp.phi + (+$("s1P").value);        // measured AoA (data)
  const u=dirOf(phi);
  const psi=sp.psi + (+$("s1D").value);        // measured AoD (data)
  const ud=dirOf(psi);
  const P=rayEllipse(BS,UE,u,L);               // incidence point from AoA side
  const Pd=rayEllipse(UE,BS,ud,L);             // incidence point from AoD side
  const VA=[UE[0]+L*u[0],UE[1]+L*u[1]];        // virtual anchor from data
  $("o1L").textContent=(L*M_PER_PX).toFixed(1)+" m";
  $("o1P").textContent=degFmt(phi);
  $("o1D").textContent=degFmt(psi);
  let s="";
  /* ghost true wall (unknown to the receiver) */
  s+=seg(WC,WEND,"#8a97a3",3,null,0.35)+txt(WC[0]+16,WC[1]+14,"reference wall · not input","#8a97a3",10.5);
  if(ck1("c1_ell")){
    const ep=ellParams(BS,UE,L);
    if(T1.on&&ep){
      s+=ellPartial(ep,T1.th,"#e8720c",2);
      s+=`<circle cx="${BS[0]}" cy="${BS[1]}" r="3" fill="#b45607"/><circle cx="${UE[0]}" cy="${UE[1]}" r="3" fill="#b45607"/>`;
      s+=stringViz(BS,UE,ellPoint(ep,T1.th),"#b45607","‖x−UE‖ + ‖x−BS‖",[-150,86]);
    } else {
      s+=ellipse(BS,UE,L,"#e8720c",1.8,null,0.9)
        +pin(BS,"#e8720c",11)+pin(UE,"#e8720c",16)
        +txt(UE[0]-165,UE[1]+128,"‖x−BS‖+‖x−UE‖ = L","#b45607",11);
    }
  }
  /* implied physical path from the data */
  s+=arrow(BS,P,"#51606e",2)+arrow(P,UE,"#51606e",2);
  if(ck1("c1_ray")){
    s+=arrow(UE,[UE[0]+u[0]*(L+34),UE[1]+u[1]*(L+34)],"#16222e",1.4,"6 4",0.55);
    s+=txt(UE[0]+u[0]*40+8,UE[1]+u[1]*40-6,"AoA φ","#16222e",11);
    s+=arrow(BS,[BS[0]+ud[0]*(dist(BS,Pd)+40),BS[1]+ud[1]*(dist(BS,Pd)+40)],"#16222e",1.4,"6 4",0.55);
    s+=txt(BS[0]+ud[0]*58+6,BS[1]+ud[1]*58-8,"AoD ψ","#16222e",11);
  }
  if(ck1("c1_va")){
    s+=seg(P,VA,"#7c4dbe",1.8,"5 4",0.9);
    s+=vaMark(VA,"VA = UE + L·û");
    s+=txt((P[0]+VA[0])/2-6,(P[1]+VA[1])/2-8,"d₂ (mirrored)","#5d3691",10.5);
  }
  if(ck1("c1_ip")){
    /* inferred wall: perpendicular bisector of BS–VA, through P */
    const n=unit(sub(VA,BS)), w=[-n[1],n[0]];
    const a=[P[0]-130*w[0],P[1]-130*w[1]], b=[P[0]+130*w[0],P[1]+130*w[1]];
    s+=seg(a,b,"#16222e",4.5);
    for(let i=-120;i<=120;i+=18){
      const q=[P[0]+i*w[0],P[1]+i*w[1]];
      s+=seg(q,[q[0]+8*(n[0]+w[0])*0.8,q[1]+8*(n[1]+w[1])*0.8],"#8a97a3",1.2);
    }
    s+=txt(b[0]+6,b[1]+4,"wall inferred from (τ, φ)","#16222e",10.5);
    s+=ipMark(P,"P (incidence)",10,-10);
    if(dist(P,Pd)>4){
      s+=`<circle cx="${Pd[0]}" cy="${Pd[1]}" r="5" fill="none" stroke="#0a6b5e" stroke-width="2"/>`
        +txt(Pd[0]+9,Pd[1]+13,"P from ψ","#0a6b5e",10.5);
    }
    s+=txt((UE[0]+P[0])/2-26,(UE[1]+P[1])/2+16,"d₁","#0a6b5e",11)
      +txt((BS[0]+P[0])/2-10,(BS[1]+P[1])/2-8,"d₂","#51606e",11);
  }
  if(ck1("c1_mir")){
    const mid=[(BS[0]+VA[0])/2,(BS[1]+VA[1])/2];
    s+=seg(BS,VA,"#8a97a3",1.2,"5 4",0.85)
      +`<circle cx="${mid[0]}" cy="${mid[1]}" r="2.4" fill="#8a97a3"/>`
      +txt(mid[0]-8,mid[1]-10,"mirror plane bisects BS↔VA","#8a97a3",10.5,"end");
  }
  /* dispersive wall: the true specular point diffuses into a weighted cloud, and so does the VA */
  if(sig>0){
    const sSpec=(sp.Pt[0]-WC[0])*D1.WD[0]+(sp.Pt[1]-WC[1])*D1.WD[1];
    for(let i=0;i<G1.length;i++){
      const sv=Math.max(8,Math.min(D1.WLEN-8,sSpec+sig*G1[i]));
      const q=[WC[0]+D1.WD[0]*sv,WC[1]+D1.WD[1]*sv];
      const w=Math.exp(-(G1[i]*G1[i])/2);
      if(i<7) s+=`<path d="M${BS[0]},${BS[1]} L${q[0]},${q[1]} L${UE[0]},${UE[1]}" fill="none" stroke="#0e8f7e" stroke-width="1" opacity="${(0.07+0.2*w).toFixed(2)}"/>`;
      s+=`<circle cx="${q[0]-3*WN[0]}" cy="${q[1]-3*WN[1]}" r="${(2.2+2.8*w).toFixed(1)}" fill="#0e8f7e" opacity="${(0.16+0.62*w).toFixed(2)}"/>`;
      const Lq=dist(BS,q)+dist(q,UE);
      const uq=unit(sub(q,UE));
      const vq=[UE[0]+Lq*uq[0],UE[1]+Lq*uq[1]];
      s+=`<circle cx="${vq[0]}" cy="${vq[1]}" r="${(1.6+2.2*w).toFixed(1)}" fill="#7c4dbe" opacity="${(0.12+0.55*w).toFixed(2)}"/>`;
    }
  }
  s+=bsMark(BS)+ueMark(UE);
  svg1.innerHTML=s;
  /* wall consistency check */
  const n=unit(sub(VA,BS));
  const tilt=Math.acos(Math.min(1,Math.abs(n[0]*WN[0]+n[1]*WN[1])))*180/Math.PI;
  const offm=Math.abs((P[0]-WC[0])*WN[0]+(P[1]-WC[1])*WN[1])*M_PER_PX;
  const mism=dist(P,Pd)*M_PER_PX;
  const consistent=tilt<0.3&&offm<0.15&&mism<0.15;
  $("stat1").innerHTML=
    `<b>L</b> = ${fmt(L)}<br><b>φ</b> = ${degFmt(phi)} · <b>ψ</b> = ${degFmt(psi)}<br>`+
    `bistatic ‖P(φ)−P(ψ)‖ ${mism.toFixed(2)} m · wall tilt ${tilt.toFixed(1)}° · P off wall ${offm.toFixed(2)} m<br>`+
    (consistent?`<span class="ok">✓ data specular-consistent — both angles hit one wall point</span>`
               :`<span class="off">✗ inconsistent — a bistatic gate would reject this triple</span>`)+
    (sig>0?`<br>${G1.length} perturbed sub-path samples — incidence-point smear induces VA spread`:"");
}
function stop1(){T1.on=false;if(raf1){cancelAnimationFrame(raf1);raf1=null;}}
function trace1(){
  stop1();
  if(reduced||!ck1("c1_ell")){render1();return;}
  T1.on=true;
  const t0=performance.now(),DUR=3400;
  const tick=now=>{
    const f=Math.min(1,(now-t0)/DUR);
    T1.th=2*Math.PI*f;
    render1();
    if(f<1){raf1=requestAnimationFrame(tick);}else{stop1();render1();}
  };
  raf1=requestAnimationFrame(tick);
}
["c1_ell","c1_ray","c1_va","c1_ip","c1_mir"].forEach(id=>$(id).addEventListener("change",()=>{stop1();render1();}));
["s1L","s1P","s1D","sSig1"].forEach(id=>$(id).addEventListener("input",()=>{stop1();render1();}));
$("b1R").addEventListener("click",()=>{$("s1L").value=0;$("s1P").value=0;$("s1D").value=0;stop1();render1();});
$("b1T").addEventListener("click",trace1);
makeDraggable(svg1,()=>D1.UE,p=>{D1.UE=p;$("s1L").value=0;$("s1P").value=0;$("s1D").value=0;stop1();},[120,300,240,440],render1);
render1();

/* =================== DEMO 2: double bounce (tilted corner) — stepped construction =================== */
const svg2=$("svg2");
/* wall B is tilted 20° off perpendicular — exactly perpendicular walls would form a
   corner retroreflector and force AoD ∥ AoA for every UE position */
const TILT=20*Math.PI/180;
const D2={BS:[90,300],AY:225,AX0:40,AX1:330,UE:[250,470],
  BC:[330,225], BD:[Math.sin(TILT),Math.cos(TILT)], BLEN:320};
D2.BN=[D2.BD[1],-D2.BD[0]];                                    // outward wall-B normal
D2.BEND=[D2.BC[0]+D2.BD[0]*D2.BLEN,D2.BC[1]+D2.BD[1]*D2.BLEN];
const mirrorB=p=>{const d=(p[0]-D2.BC[0])*D2.BN[0]+(p[1]-D2.BC[1])*D2.BN[1];
  return [p[0]-2*d*D2.BN[0],p[1]-2*d*D2.BN[1]];};
D2.VA1=[D2.BS[0],2*D2.AY-D2.BS[1]];          // mirror of BS in wall A (1st bounce)
D2.VA2=mirrorB(D2.VA1);                      // mirror of VA1 in wall B (2nd bounce)
function specular2(){
  const u=unit(sub(D2.VA2,D2.UE));
  return {L:dist(D2.UE,D2.VA2), phi:aoaOf(u)};
}
const STEP2CAP=["",
 "① <b>Hypothesis: one bounce.</b> Path 2’s delay L₂ pins a string at BS &amp; UE (red). If the path bounced once, its point would sit where the AoA ray crosses this ellipse: P(φ).",
 "② <b>Test it with the AoD.</b> A single bounce must satisfy both measured angles at the same point — but the AoD ray from the BS crosses the ellipse at a different point P(ψ). The gap between P(φ) and P(ψ) → hypothesis rejected.",
 "③ <b>Bootstrap VA¹ from path 1.</b> Path 1’s delay L⁽¹⁾ draws the orange ellipse (foci BS &amp; UE) — and the test that failed for path 2 passes here: path 1’s AoA and AoD rays cross its ellipse at the <em>same</em> point ✓, so a single bounce on wall A is not rejected. Walking the full L⁽¹⁾ lands on VA¹.",
 "④ <b>Bounce-2 ellipse.</b> With VA¹ fixed, re-pin the same L₂ string at VA¹ &amp; UE (teal). The AoA ray now crosses it exactly on wall B → P₂.",
 "⑤ <b>Bounce-1 segment.</b> Peel off ‖UE→P₂‖; the leftover string pinned at BS &amp; P₂ (green) meets the AoD ray on wall A → P₁. The two-bounce interpretation is complete and both angles are satisfied.",
 "⑥ <b>The full picture:</b> the hollow reference VA², the measurement-derived mirror construction, path 2’s VA¹ candidate curve, and illustrative tangential perturbation clouds on every bounce. The reference overlay and clouds are not estimator inputs."];
let step2=1;
const T2={on:false,th:0,key:null}; let raf2=null;
function stop2(){T2.on=false;T2.key=null;if(raf2){cancelAnimationFrame(raf2);raf2=null;}}
const STEP2KEY={1:"bad",3:"boot",4:"good",5:"e1"};
function goStep2(sN,animate){
  stop2();
  step2=Math.max(1,Math.min(6,sN));
  $("cap2").innerHTML=STEP2CAP[step2];
  $("b2Prev").disabled=step2===1; $("b2Next").disabled=step2===6;
  $("o2Step").textContent=step2+" / 6";
  const key=STEP2KEY[step2];
  if(animate&&key&&!reduced){
    T2.on=true;T2.key=key;
    const t0=performance.now(),DUR=2600;
    const tick=now=>{
      const f=Math.min(1,(now-t0)/DUR);
      T2.th=2*Math.PI*f;
      render2();
      if(f<1){raf2=requestAnimationFrame(tick);}else{stop2();render2();}
    };
    raf2=requestAnimationFrame(tick);
  } else render2();
}
function render2(){
  const {BS,UE,VA1,VA2,AY,AX0,AX1,BC,BD,BN,BLEN,BEND}=D2;
  const sig=+$("sSig").value;
  $("oSig").textContent=(sig*M_PER_PX).toFixed(1)+" m";
  const sp=specular2();
  const L2=sp.L + (+$("s2L").value)/M_PER_PX;
  const phi=sp.phi + (+$("s2P").value);
  const u=dirOf(phi);
  $("o2L").textContent=(L2*M_PER_PX).toFixed(1)+" m";
  $("o2P").textContent=degFmt(phi);
  /* true specular geometry (ground truth + AoD specular value) */
  const dV=sub(VA2,UE);
  const tB=((BC[0]-UE[0])*BN[0]+(BC[1]-UE[1])*BN[1])/(dV[0]*BN[0]+dV[1]*BN[1]);
  const P2t=[UE[0]+tB*dV[0],UE[1]+tB*dV[1]];
  const tA=(AY-P2t[1])/(VA1[1]-P2t[1]); const P1t=[P2t[0]+tA*(VA1[0]-P2t[0]),AY];
  const psi=aoaOf(unit(sub(P1t,BS))) + (+$("s2D").value);
  const ud=dirOf(psi);
  $("o2D").textContent=degFmt(psi);
  /* bootstrap VA1 from path 1 (single bounce off wall A) */
  const Ls=dist(UE,VA1) + (+$("s2Lb").value)/M_PER_PX;
  const phib=aoaOf(unit(sub(VA1,UE))) + (+$("s2Pb").value);
  const us=dirOf(phib);
  $("o2Lb").textContent=(Ls*M_PER_PX).toFixed(1)+" m";
  $("o2Pb").textContent=degFmt(phib);
  const Ps=rayEllipse(BS,UE,us,Ls);
  const VA1e=[UE[0]+Ls*us[0],UE[1]+Ls*us[1]];
  /* path 1's AoD — used to CONFIRM the single-bounce hypothesis for path 1 */
  const tps=(AY-UE[1])/(VA1[1]-UE[1]);
  const Pst=[UE[0]+tps*(VA1[0]-UE[0]),AY];
  const psib=aoaOf(unit(sub(Pst,BS))) + (+$("s2Db").value);
  const udb=dirOf(psib);
  $("o2Db").textContent=degFmt(psib);
  const Psd=rayEllipse(UE,BS,udb,Ls);
  /* estimates from the data */
  const P2=rayEllipse(VA1e,UE,u,L2);
  const Lseg=L2-dist(UE,P2);
  const P1=rayEllipse(P2,BS,ud,Lseg);
  const VA2d=[UE[0]+L2*u[0],UE[1]+L2*u[1]];
  $("o2L1").textContent=(Lseg*M_PER_PX).toFixed(1)+" m";
  const ph=rayEllipse(BS,UE,u,L2);           // 1-bounce candidate from the AoA
  const phD=rayEllipse(UE,BS,ud,L2);         // 1-bounce candidate from the AoD
  let s="";
  /* walls */
  s+=seg([AX0,AY],[AX1,AY],"#16222e",5);
  for(let x=AX0;x<AX1;x+=18)s+=seg([x,AY],[x-9,AY-9],"#8a97a3",1.2);
  s+=txt(AX0+4,AY-14,"ref wall A · not input","#51606e",11);
  s+=seg(BC,BEND,"#16222e",5);
  for(let k=10;k<BLEN;k+=18){const q=[BC[0]+BD[0]*k,BC[1]+BD[1]*k];
    s+=seg(q,[q[0]+7*(BN[0]+BD[0]),q[1]+7*(BN[1]+BD[1])],"#8a97a3",1.2);}
  s+=txt(BC[0]+BD[0]*128+BN[0]*16,BC[1]+BD[1]*128+BN[1]*16,"ref wall B","#51606e",11)
    +txt(BC[0]+BD[0]*142+BN[0]*16,BC[1]+BD[1]*142+BN[1]*16,"not input","#51606e",11);
  /* true physical path — ground truth, faint */
  s+=arrow(BS,P1t,"#8a97a3",1.6,null,0.5)+arrow(P1t,P2t,"#8a97a3",1.6,null,0.5)+arrow(P2t,UE,"#8a97a3",1.6,null,0.5);
  const tracing=k=>T2.on&&T2.key===k;
  /* STEP 1+: the 1-bounce hypothesis */
  {
    const ep=ellParams(BS,UE,L2);
    if(tracing("bad")&&ep){
      s+=ellPartial(ep,T2.th,"#c22f2f",1.8,"7 5");
      s+=stringViz(BS,UE,ellPoint(ep,T2.th),"#c22f2f","1-bounce hyp. — string at BS &amp; UE: ‖x−UE‖ + ‖x−BS‖",[-140,4]);
    } else {
      s+=ellipse(BS,UE,L2,"#c22f2f",1.5,"7 5",step2<=2?0.8:0.2);
      if(step2<=2) s+=pin(BS,"#c22f2f",11)+pin(UE,"#c22f2f",16);
      s+=arrow(UE,[UE[0]+u[0]*(dist(UE,ph)+40),UE[1]+u[1]*(dist(UE,ph)+40)],"#16222e",1.4,"6 4",step2<=2?0.6:0.3);
      if(step2<=2) s+=txt(UE[0]+u[0]*46+6,UE[1]+u[1]*46+14,"AoA φ","#16222e",11);
      s+=`<circle cx="${ph[0]}" cy="${ph[1]}" r="4.5" fill="none" stroke="#c22f2f" stroke-width="2" opacity="${step2<=2?1:0.3}"/>`;
      if(step2<=2){
        s+=seg(BS,ph,"#c22f2f",1.1,"2 3",0.6)+seg(UE,ph,"#c22f2f",1.1,"2 3",0.6);
        const dbs=dist(ph,BS)*M_PER_PX,due=dist(ph,UE)*M_PER_PX;
        s+=txt(ph[0]+10,ph[1]-9,"P(φ) if 1 bounce","#c22f2f",10.5)
          +txt(ph[0]+10,ph[1]+5,`${dbs.toFixed(1)}+${due.toFixed(1)} = ${(dbs+due).toFixed(1)} m`,"#c22f2f",10);
      }
      if(step2>=3) s+=txt(ph[0]+9,ph[1]-8,"rejected ✗","#c22f2f",10);
    }
  }
  /* STEP 2+: the AoD test */
  if(step2>=2&&!tracing("bad")){
    s+=arrow(BS,[BS[0]+ud[0]*(dist(BS,phD)+40),BS[1]+ud[1]*(dist(BS,phD)+40)],"#16222e",1.4,"6 4",step2===2?0.6:0.3);
    if(step2===2){
      s+=txt(BS[0]+ud[0]*58+6,BS[1]+ud[1]*58-8,"AoD ψ","#16222e",11);
      s+=`<circle cx="${phD[0]}" cy="${phD[1]}" r="4.5" fill="none" stroke="#c22f2f" stroke-width="2"/>`
        +txt(phD[0]-9,phD[1]-9,"P(ψ)","#c22f2f",10.5,"end");
      s+=seg(ph,phD,"#c22f2f",2,"3 3",0.9);
      const mm=dist(ph,phD)*M_PER_PX;
      s+=txt((ph[0]+phD[0])/2+10,(ph[1]+phD[1])/2,`${mm.toFixed(1)} m apart → reject`,"#c22f2f",11);
    }
  }
  /* STEP 3+: bootstrap VA1 from path 1 */
  if(step2>=3){
    const ep=ellParams(BS,UE,Ls);
    if(tracing("boot")&&ep){
      s+=ellPartial(ep,T2.th,"#e8720c",2);
      s+=stringViz(BS,UE,ellPoint(ep,T2.th),"#b45607","path 1 — string of L⁽¹⁾ at BS &amp; UE: ‖x−UE‖ + ‖x−BS‖",[-140,4]);
    } else {
      s+=ellipse(BS,UE,Ls,"#e8720c",1.8,null,step2===3?0.9:0.35);
      if(step2===3) s+=pin(BS,"#e8720c",11)+pin(UE,"#e8720c",16);
      s+=arrow(BS,Ps,"#e8720c",1.8,null,0.85)+arrow(Ps,UE,"#e8720c",1.8,null,0.85);
      s+=arrow(UE,[UE[0]+us[0]*(Ls+24),UE[1]+us[1]*(Ls+24)],"#b45607",1.3,"6 4",0.7);
      s+=`<circle cx="${Ps[0]}" cy="${Ps[1]}" r="4" fill="#e8720c"/>`;
      if(step2===3){
        /* the confirming AoD test: both of path 1's angles hit the same ellipse point */
        s+=arrow(BS,[BS[0]+udb[0]*(dist(BS,Psd)+36),BS[1]+udb[1]*(dist(BS,Psd)+36)],"#b45607",1.2,"6 4",0.65);
        const mmb=dist(Ps,Psd)*M_PER_PX;
        if(mmb<0.5){
          s+=txt(Ps[0]-10,Ps[1]+20,"AoA &amp; AoD agree ✓ — 1 bounce not rejected","#1d7a1d",10.5,"end");
        } else {
          s+=`<circle cx="${Psd[0]}" cy="${Psd[1]}" r="4.5" fill="none" stroke="#c22f2f" stroke-width="2"/>`
            +seg(Ps,Psd,"#c22f2f",2,"3 3",0.9)
            +txt(Ps[0]-10,Ps[1]+20,`P(φ⁽¹⁾) ↔ P(ψ⁽¹⁾) ${mmb.toFixed(1)} m apart — not a clean single bounce`,"#c22f2f",10.5,"end");
        }
      }
      s+=vaMark(VA1e,"VA¹ (bootstrapped)");
      if(step2===3) s+=txt(VA1e[0]+12,VA1e[1]+22,"walk the full L⁽¹⁾ → VA¹","#b45607",10.5);
    }
  }
  /* STEP 4+: bounce-2 ellipse */
  if(step2>=4&&!tracing("boot")){
    const ep=ellParams(VA1e,UE,L2);
    if(tracing("good")&&ep){
      s+=ellPartial(ep,T2.th,"#0e8f7e",2);
      s+=stringViz(VA1e,UE,ellPoint(ep,T2.th),"#0a6b5e","bounce-2 — same L₂, string at VA¹ &amp; UE: ‖x−UE‖ + ‖x−VA¹‖",[-140,4]);
    } else {
      s+=ellipse(VA1e,UE,L2,"#0e8f7e",1.8,null,step2===4?0.9:0.45);
      if(step2===4) s+=pin(VA1e,"#0e8f7e",11)+pin(UE,"#0e8f7e",20);
      s+=arrow(UE,[UE[0]+u[0]*(dist(UE,P2)+34),UE[1]+u[1]*(dist(UE,P2)+34)],"#16222e",1.4,"6 4",0.5);
      s+=ipMark(P2,"P₂",10,4);
    }
  }
  /* STEP 5+: bounce-1 segment ellipse */
  if(step2>=5&&!tracing("good")){
    const ep=ellParams(BS,P2,Lseg);
    if(tracing("e1")&&ep){
      s+=ellPartial(ep,T2.th,"#2ca02c",2);
      s+=stringViz(BS,P2,ellPoint(ep,T2.th),"#1d7a1d","bounce-1 — leftover L₂ − ‖UE→P₂‖, string at BS &amp; P₂: ‖x−P₂‖ + ‖x−BS‖",[-140,4]);
    } else if(ep){
      s+=ellipse(BS,P2,Lseg,"#2ca02c",1.7,null,step2===5?0.9:0.45);
      if(step2===5) s+=pin(BS,"#2ca02c",15)+pin(P2,"#2ca02c",10);
      s+=arrow(BS,[BS[0]+ud[0]*(dist(BS,P1)+42),BS[1]+ud[1]*(dist(BS,P1)+42)],"#16222e",1.4,"6 4",0.5);
      s+=ipMark(P1,"P₁",-6,-12);
      s+=seg(BS,P1,"#0e8f7e",1.6,null,0.8)+seg(P1,P2,"#0e8f7e",1.6,null,0.8)+seg(P2,UE,"#0e8f7e",1.6,null,0.8);
    }
  }
  /* STEP 6: the full picture */
  if(step2===6){
    s+=seg([AX1,AY],[640,AY],"#8a97a3",1,"3 4",0.55)
      +seg([BC[0]-BD[0]*190,BC[1]-BD[1]*190],BC,"#8a97a3",1,"3 4",0.55);
    s+=seg(BS,VA1,"#8a97a3",1.2,"5 4",0.85)+seg(VA1,VA2,"#8a97a3",1.2,"5 4",0.85);
    s+=txt(BS[0]-12,262,"mirror in A","#8a97a3",10.5,"end")
      +txt((VA1[0]+VA2[0])/2-36,(VA1[1]+VA2[1])/2-12,"mirror in B","#8a97a3",10.5);
    const fam=family({BS,UE,L:L2,u,ud});
    if(fam.length>1){
      s+=`<path d="${poly4(fam)}" fill="none" stroke="#7c4dbe" stroke-width="1.4" opacity="0.5"/>`;
      const e=fam[Math.min(10,fam.length-1)];
      s+=txt(e.cand[0]-8,e.cand[1]-12,"VA¹ family (path 2 alone)","#5d3691",10,"end");
    }
    if(sig>0){
      for(let i=0;i<G1.length;i++){
        const q1=[Math.max(AX0+6,Math.min(AX1-6,P1t[0]+sig*G1[i])),AY];
        const sP2=(P2t[0]-BC[0])*BD[0]+(P2t[1]-BC[1])*BD[1];
        const s2v=Math.max(8,Math.min(BLEN-8,sP2+sig*G2[i]));
        const q2=[BC[0]+BD[0]*s2v,BC[1]+BD[1]*s2v];
        const w=Math.exp(-(G1[i]*G1[i]+G2[i]*G2[i])/2);
        if(i<7) s+=`<path d="M${BS[0]},${BS[1]} L${q1[0]},${q1[1]} L${q2[0]},${q2[1]} L${UE[0]},${UE[1]}" fill="none" stroke="#0e8f7e" stroke-width="1" opacity="${(0.08+0.22*w).toFixed(2)}"/>`;
        s+=`<circle cx="${q1[0]}" cy="${q1[1]-3}" r="${(2.2+2.8*w).toFixed(1)}" fill="#0e8f7e" opacity="${(0.16+0.62*w).toFixed(2)}"/>`;
        s+=`<circle cx="${q2[0]+3*BN[0]}" cy="${q2[1]+3*BN[1]}" r="${(2.2+2.8*w).toFixed(1)}" fill="#0e8f7e" opacity="${(0.16+0.62*w).toFixed(2)}"/>`;
        const Lq=dist(BS,q1)+dist(q1,q2)+dist(q2,UE);
        const uq=unit(sub(q2,UE));
        const vq=[UE[0]+Lq*uq[0],UE[1]+Lq*uq[1]];
        s+=`<circle cx="${vq[0]}" cy="${vq[1]}" r="${(1.6+2.2*w).toFixed(1)}" fill="#7c4dbe" opacity="${(0.12+0.55*w).toFixed(2)}"/>`;
        /* path 1's bounce is dispersive too — its scatter smears the bootstrapped VA1 */
        const q3=[Math.max(AX0+6,Math.min(AX1-6,Pst[0]+sig*G2[i])),AY];
        const w3=Math.exp(-(G2[i]*G2[i])/2);
        if(i<5) s+=`<path d="M${BS[0]},${BS[1]} L${q3[0]},${q3[1]} L${UE[0]},${UE[1]}" fill="none" stroke="#e8720c" stroke-width="1" opacity="${(0.06+0.2*w3).toFixed(2)}"/>`;
        s+=`<circle cx="${q3[0]}" cy="${q3[1]-3}" r="${(2+2.6*w3).toFixed(1)}" fill="#e8720c" opacity="${(0.15+0.6*w3).toFixed(2)}"/>`;
        const Lq3=dist(BS,q3)+dist(q3,UE);
        const uq3=unit(sub(q3,UE));
        const vq3=[UE[0]+Lq3*uq3[0],UE[1]+Lq3*uq3[1]];
        s+=`<circle cx="${vq3[0]}" cy="${vq3[1]}" r="${(1.5+2*w3).toFixed(1)}" fill="#7c4dbe" opacity="${(0.12+0.5*w3).toFixed(2)}"/>`;
      }
    }
    s+=seg(P2,VA2d,"#7c4dbe",1.8,"5 4",0.9);
    s+=vaMark(VA2,"VA² (bounce 2 · ref)");
    s+=vaDot(VA2d);
  }
  s+=bsMark(BS,"BS",-30,4)+ueMark(UE);
  svg2.innerHTML=s;
  const offB=Math.abs((P2[0]-BC[0])*BN[0]+(P2[1]-BC[1])*BN[1])*M_PER_PX, offA=Math.abs(P1[1]-AY)*M_PER_PX;
  const va1err=dist(VA1e,VA1)*M_PER_PX;
  const mm=dist(ph,phD)*M_PER_PX;
  const consistent=offB<0.15&&offA<0.15&&va1err<0.15;
  $("stat2").innerHTML=
    `<b>L₂</b> = ${fmt(L2)} · <b>φ</b> = ${degFmt(phi)} · <b>ψ</b> = ${degFmt(psi)}<br>`+
    `path-2 1-bounce test: ${mm.toFixed(1)} m apart → ${mm>0.5?"reject":"cannot reject"}<br>`+
    `path-1 1-bounce test: ${(dist(Ps,Psd)*M_PER_PX).toFixed(2)} m apart → ${dist(Ps,Psd)*M_PER_PX<0.5?"confirmed ✓":"suspect"}<br>`+
    `bootstrapped VA¹ error: ${va1err.toFixed(2)} m<br>`+
    (consistent?`<span class="ok">✓ 2-bounce interpretation consistent — P₁ on wall A, P₂ on wall B</span>`
               :`<span class="off">P₂ off wall B by ${offB.toFixed(2)} m · P₁ off wall A by ${offA.toFixed(2)} m</span>`)+
    (step2===6?`<br>${sig>0?`${G1.length} perturbed incidence-point samples / wall`:"illustrative smear removed (σ = 0)"}`:"");
}
["sSig","s2L","s2P","s2D","s2Lb","s2Pb","s2Db"].forEach(id=>$(id).addEventListener("input",()=>{stop2();render2();}));
$("b2R").addEventListener("click",()=>{["s2L","s2P","s2D","s2Lb","s2Pb","s2Db"].forEach(id=>$(id).value=0);stop2();render2();});
$("b2Prev").addEventListener("click",()=>goStep2(step2-1,false));
$("b2Next").addEventListener("click",()=>goStep2(step2+1,true));
makeDraggable(svg2,()=>D2.UE,p=>{D2.UE=p;["s2L","s2P","s2D","s2Lb","s2Pb","s2Db"].forEach(id=>$(id).value=0);stop2();},[230,430,300,510],render2);
goStep2(1,false);

/* =================== DEMO 3: double bounce between parallel walls — stepped construction =================== */
const svg3=$("svg3");
const D3={BS:[150,480],UE:[250,230],LX:100,RX:320,WY0:150,WY1:600};
D3.VA1=[2*D3.RX-D3.BS[0],D3.BS[1]];   // mirror of BS in wall R (1st bounce)
D3.VA2=[2*D3.LX-D3.VA1[0],D3.VA1[1]]; // mirror of VA1 in wall L (2nd bounce)
function specular3(){
  const dV=sub(D3.VA2,D3.UE);
  const t=(D3.LX-D3.UE[0])/dV[0];
  const P2t=[D3.LX,D3.UE[1]+t*dV[1]];
  const t1=(D3.RX-P2t[0])/(D3.VA1[0]-P2t[0]);
  const P1t=[D3.RX,P2t[1]+t1*(D3.VA1[1]-P2t[1])];
  return {L:dist(D3.UE,D3.VA2),phi:aoaOf(unit(dV)),psi:aoaOf(unit(sub(P1t,D3.BS))),P1t,P2t};
}
const STEP3CAP=["",
 "① <b>Hypothesis: one bounce.</b> Path 2’s delay L₂ pins a string at BS &amp; UE (red); the AoA ray crosses it at the hypothesized bounce point P(φ).",
 "② <b>Test it with the AoD.</b> The AoD ray from the BS crosses the same ellipse at P(ψ) — on the other side of the corridor. The two points are far apart → hypothesis rejected.",
 "③ <b>Bootstrap VA¹ from path 1.</b> Path 1’s delay L⁽¹⁾ draws the orange ellipse (foci BS &amp; UE) — and the test that failed for path 2 passes here: path 1’s AoA and AoD rays cross its ellipse at the <em>same</em> point ✓, so a single bounce on wall R is not rejected. Walking the full L⁽¹⁾ lands on VA¹.",
 "④ <b>Bounce-2 ellipse.</b> With VA¹ fixed, re-pin the same L₂ string at VA¹ &amp; UE (teal). The AoA ray now crosses it exactly on wall L → P₂.",
 "⑤ <b>Bounce-1 segment.</b> Peel off ‖UE→P₂‖; the leftover string pinned at BS &amp; P₂ (green) meets the AoD ray on wall R → P₁. Both measured angles are satisfied.",
 "⑥ <b>The full picture:</b> the reference VA², the mirror construction (the image-source ladder), path 2’s VA¹ candidate curve, and illustrative tangential perturbation clouds on every bounce. The clouds are not a diffuse-propagation model."];
let step3=1;
const T3={on:false,th:0,key:null}; let raf3=null;
function stop3(){T3.on=false;T3.key=null;if(raf3){cancelAnimationFrame(raf3);raf3=null;}}
const STEP3KEY={1:"bad",3:"boot",4:"good",5:"e1"};
function goStep3(sN,animate){
  stop3();
  step3=Math.max(1,Math.min(6,sN));
  $("cap3").innerHTML=STEP3CAP[step3];
  $("b3Prev").disabled=step3===1; $("b3Next").disabled=step3===6;
  $("o3Step").textContent=step3+" / 6";
  const key=STEP3KEY[step3];
  if(animate&&key&&!reduced){
    T3.on=true;T3.key=key;
    const t0=performance.now(),DUR=2600;
    const tick=now=>{
      const f=Math.min(1,(now-t0)/DUR);
      T3.th=2*Math.PI*f;
      render3();
      if(f<1){raf3=requestAnimationFrame(tick);}else{stop3();render3();}
    };
    raf3=requestAnimationFrame(tick);
  } else render3();
}
function render3(){
  const {BS,UE,VA1,VA2,LX,RX,WY0,WY1}=D3;
  const sig=+$("sSig3").value;
  $("oSig3").textContent=(sig*M_PER_PX).toFixed(1)+" m";
  const sp=specular3();
  const L2=sp.L + (+$("s3L").value)/M_PER_PX;
  const phi=sp.phi + (+$("s3P").value);
  const u=dirOf(phi);
  const psi=sp.psi + (+$("s3D").value);
  const ud=dirOf(psi);
  $("o3L").textContent=(L2*M_PER_PX).toFixed(1)+" m";
  $("o3P").textContent=degFmt(phi);
  $("o3D").textContent=degFmt(psi);
  const {P1t,P2t}=sp;
  /* bootstrap VA1 from path 1 (single bounce off wall R) */
  const Ls=dist(UE,VA1) + (+$("s3Lb").value)/M_PER_PX;
  const phib=aoaOf(unit(sub(VA1,UE))) + (+$("s3Pb").value);
  const us=dirOf(phib);
  $("o3Lb").textContent=(Ls*M_PER_PX).toFixed(1)+" m";
  $("o3Pb").textContent=degFmt(phib);
  const Ps=rayEllipse(BS,UE,us,Ls);
  const VA1e=[UE[0]+Ls*us[0],UE[1]+Ls*us[1]];
  /* path 1's AoD — used to CONFIRM the single-bounce hypothesis for path 1 */
  const tps=(RX-UE[0])/(VA1[0]-UE[0]);
  const Pst=[RX,UE[1]+tps*(VA1[1]-UE[1])];
  const psib=aoaOf(unit(sub(Pst,BS))) + (+$("s3Db").value);
  const udb=dirOf(psib);
  $("o3Db").textContent=degFmt(psib);
  const Psd=rayEllipse(UE,BS,udb,Ls);
  /* estimates from the data */
  const P2=rayEllipse(VA1e,UE,u,L2);
  const Lseg=L2-dist(UE,P2);
  const P1=rayEllipse(P2,BS,ud,Lseg);
  const VA2d=[UE[0]+L2*u[0],UE[1]+L2*u[1]];
  $("o3L1").textContent=(Lseg*M_PER_PX).toFixed(1)+" m";
  const ph=rayEllipse(BS,UE,u,L2);
  const phD=rayEllipse(UE,BS,ud,L2);
  let s="";
  /* walls */
  s+=seg([LX,WY0],[LX,WY1],"#16222e",5);
  for(let y=WY0;y<WY1;y+=18)s+=seg([LX,y],[LX-9,y+9],"#8a97a3",1.2);
  s+=txt(LX-14,WY0+16,"ref L · not input","#51606e",11,"end");
  s+=seg([RX,WY0],[RX,WY1],"#16222e",5);
  for(let y=WY0;y<WY1;y+=18)s+=seg([RX,y],[RX+9,y+9],"#8a97a3",1.2);
  s+=txt(RX+14,WY0+16,"ref R · not input","#51606e",11);
  /* true physical path — ground truth, faint */
  s+=arrow(BS,P1t,"#8a97a3",1.6,null,0.5)+arrow(P1t,P2t,"#8a97a3",1.6,null,0.5)+arrow(P2t,UE,"#8a97a3",1.6,null,0.5);
  const tracing=k=>T3.on&&T3.key===k;
  /* STEP 1+: the 1-bounce hypothesis */
  {
    const ep=ellParams(BS,UE,L2);
    if(tracing("bad")&&ep){
      s+=ellPartial(ep,T3.th,"#c22f2f",1.8,"7 5");
      s+=stringViz(BS,UE,ellPoint(ep,T3.th),"#c22f2f","1-bounce hyp. — string at BS &amp; UE: ‖x−UE‖ + ‖x−BS‖",[-390,56]);
    } else {
      s+=ellipse(BS,UE,L2,"#c22f2f",1.5,"7 5",step3<=2?0.8:0.2);
      if(step3<=2) s+=pin(BS,"#c22f2f",11)+pin(UE,"#c22f2f",16);
      s+=arrow(UE,[UE[0]+u[0]*(dist(UE,ph)+40),UE[1]+u[1]*(dist(UE,ph)+40)],"#16222e",1.4,"6 4",step3<=2?0.6:0.3);
      if(step3<=2) s+=txt(UE[0]+u[0]*60+6,UE[1]+u[1]*60-8,"AoA φ","#16222e",11);
      s+=`<circle cx="${ph[0]}" cy="${ph[1]}" r="4.5" fill="none" stroke="#c22f2f" stroke-width="2" opacity="${step3<=2?1:0.3}"/>`;
      if(step3<=2){
        s+=seg(BS,ph,"#c22f2f",1.1,"2 3",0.6)+seg(UE,ph,"#c22f2f",1.1,"2 3",0.6);
        const dbs=dist(ph,BS)*M_PER_PX,due=dist(ph,UE)*M_PER_PX;
        s+=txt(ph[0]+10,ph[1]-9,"P(φ) if 1 bounce","#c22f2f",10.5)
          +txt(ph[0]+10,ph[1]+5,`${dbs.toFixed(1)}+${due.toFixed(1)} = ${(dbs+due).toFixed(1)} m`,"#c22f2f",10);
      }
      if(step3>=3) s+=txt(ph[0]+9,ph[1]-8,"rejected ✗","#c22f2f",10);
    }
  }
  /* STEP 2+: the AoD test */
  if(step3>=2&&!tracing("bad")){
    s+=arrow(BS,[BS[0]+ud[0]*(dist(BS,phD)+40),BS[1]+ud[1]*(dist(BS,phD)+40)],"#16222e",1.4,"6 4",step3===2?0.6:0.3);
    if(step3===2){
      s+=txt(BS[0]+ud[0]*58+6,BS[1]+ud[1]*58+16,"AoD ψ","#16222e",11);
      s+=`<circle cx="${phD[0]}" cy="${phD[1]}" r="4.5" fill="none" stroke="#c22f2f" stroke-width="2"/>`
        +txt(phD[0]+10,phD[1]-9,"P(ψ)","#c22f2f",10.5);
      s+=seg(ph,phD,"#c22f2f",2,"3 3",0.9);
      const mm=dist(ph,phD)*M_PER_PX;
      s+=txt((ph[0]+phD[0])/2-30,(ph[1]+phD[1])/2-10,`${mm.toFixed(1)} m apart → reject`,"#c22f2f",11);
    }
  }
  /* STEP 3+: bootstrap VA1 from path 1 */
  if(step3>=3){
    const ep=ellParams(BS,UE,Ls);
    if(tracing("boot")&&ep){
      s+=ellPartial(ep,T3.th,"#e8720c",2);
      s+=stringViz(BS,UE,ellPoint(ep,T3.th),"#b45607","path 1 — string of L⁽¹⁾ at BS &amp; UE: ‖x−UE‖ + ‖x−BS‖",[-390,56]);
    } else {
      s+=ellipse(BS,UE,Ls,"#e8720c",1.8,null,step3===3?0.9:0.35);
      if(step3===3) s+=pin(BS,"#e8720c",11)+pin(UE,"#e8720c",16);
      s+=arrow(BS,Ps,"#e8720c",1.8,null,0.85)+arrow(Ps,UE,"#e8720c",1.8,null,0.85);
      s+=arrow(UE,[UE[0]+us[0]*(Ls+24),UE[1]+us[1]*(Ls+24)],"#b45607",1.3,"6 4",0.7);
      s+=`<circle cx="${Ps[0]}" cy="${Ps[1]}" r="4" fill="#e8720c"/>`;
      if(step3===3){
        s+=txt(Ps[0]+10,Ps[1]+14,"path-1 bounce","#b45607",10.5);
        s+=arrow(BS,[BS[0]+udb[0]*(dist(BS,Psd)+36),BS[1]+udb[1]*(dist(BS,Psd)+36)],"#b45607",1.2,"6 4",0.65);
        const mmb=dist(Ps,Psd)*M_PER_PX;
        if(mmb<0.5){
          s+=txt(Ps[0]-10,Ps[1]-14,"AoA &amp; AoD agree ✓ — 1 bounce not rejected","#1d7a1d",10.5,"end");
        } else {
          s+=`<circle cx="${Psd[0]}" cy="${Psd[1]}" r="4.5" fill="none" stroke="#c22f2f" stroke-width="2"/>`
            +seg(Ps,Psd,"#c22f2f",2,"3 3",0.9)
            +txt(Ps[0]-10,Ps[1]-14,`P(φ⁽¹⁾) ↔ P(ψ⁽¹⁾) ${mmb.toFixed(1)} m apart — not a clean single bounce`,"#c22f2f",10.5,"end");
        }
      }
      s+=vaMark(VA1e,"VA¹ (bootstrapped)");
      if(step3===3) s+=txt(VA1e[0]+12,VA1e[1]+22,"walk the full L⁽¹⁾ → VA¹","#b45607",10.5);
    }
  }
  /* STEP 4+: bounce-2 ellipse */
  if(step3>=4&&!tracing("boot")){
    const ep=ellParams(VA1e,UE,L2);
    if(tracing("good")&&ep){
      s+=ellPartial(ep,T3.th,"#0e8f7e",2);
      s+=stringViz(VA1e,UE,ellPoint(ep,T3.th),"#0a6b5e","bounce-2 — same L₂, string at VA¹ &amp; UE: ‖x−UE‖ + ‖x−VA¹‖",[-390,56]);
    } else {
      s+=ellipse(VA1e,UE,L2,"#0e8f7e",1.8,null,step3===4?0.9:0.45);
      if(step3===4) s+=pin(VA1e,"#0e8f7e",11)+pin(UE,"#0e8f7e",20);
      s+=arrow(UE,[UE[0]+u[0]*(dist(UE,P2)+34),UE[1]+u[1]*(dist(UE,P2)+34)],"#16222e",1.4,"6 4",0.5);
      s+=ipMark(P2,"P₂",-30,-10);
    }
  }
  /* STEP 5+: bounce-1 segment ellipse */
  if(step3>=5&&!tracing("good")){
    const ep=ellParams(BS,P2,Lseg);
    if(tracing("e1")&&ep){
      s+=ellPartial(ep,T3.th,"#2ca02c",2);
      s+=stringViz(BS,P2,ellPoint(ep,T3.th),"#1d7a1d","bounce-1 — leftover L₂ − ‖UE→P₂‖, string at BS &amp; P₂: ‖x−P₂‖ + ‖x−BS‖",[-390,56]);
    } else if(ep){
      s+=ellipse(BS,P2,Lseg,"#2ca02c",1.7,null,step3===5?0.9:0.45);
      if(step3===5) s+=pin(BS,"#2ca02c",15)+pin(P2,"#2ca02c",10);
      s+=arrow(BS,[BS[0]+ud[0]*(dist(BS,P1)+42),BS[1]+ud[1]*(dist(BS,P1)+42)],"#16222e",1.4,"6 4",0.5);
      s+=ipMark(P1,"P₁",12,-10);
      s+=seg(BS,P1,"#0e8f7e",1.6,null,0.8)+seg(P1,P2,"#0e8f7e",1.6,null,0.8)+seg(P2,UE,"#0e8f7e",1.6,null,0.8);
    }
  }
  /* STEP 6: the full picture */
  if(step3===6){
    s+=seg(BS,VA1,"#8a97a3",1.2,"5 4",0.85)+seg(VA1,VA2,"#8a97a3",1.2,"5 4",0.85);
    s+=txt((BS[0]+VA1[0])/2+40,BS[1]-8,"mirror in R","#8a97a3",10.5)
      +txt((VA1[0]+VA2[0])/2-120,VA1[1]-8,"mirror in L","#8a97a3",10.5);
    const fam=family({BS,UE,L:L2,u,ud});
    if(fam.length>1){
      s+=`<path d="${poly4(fam)}" fill="none" stroke="#7c4dbe" stroke-width="1.4" opacity="0.5"/>`;
      const e=fam[Math.min(10,fam.length-1)];
      s+=txt(e.cand[0]-8,e.cand[1]-12,"VA¹ family (path 2 alone)","#5d3691",10,"end");
    }
    if(sig>0){
      for(let i=0;i<G1.length;i++){
        const q1=[RX,Math.max(WY0+6,Math.min(WY1-6,P1t[1]+sig*G1[i]))];
        const q2=[LX,Math.max(WY0+6,Math.min(WY1-6,P2t[1]+sig*G2[i]))];
        const w=Math.exp(-(G1[i]*G1[i]+G2[i]*G2[i])/2);
        if(i<7) s+=`<path d="M${BS[0]},${BS[1]} L${q1[0]},${q1[1]} L${q2[0]},${q2[1]} L${UE[0]},${UE[1]}" fill="none" stroke="#0e8f7e" stroke-width="1" opacity="${(0.08+0.22*w).toFixed(2)}"/>`;
        s+=`<circle cx="${q1[0]+3}" cy="${q1[1]}" r="${(2.2+2.8*w).toFixed(1)}" fill="#0e8f7e" opacity="${(0.16+0.62*w).toFixed(2)}"/>`;
        s+=`<circle cx="${q2[0]-3}" cy="${q2[1]}" r="${(2.2+2.8*w).toFixed(1)}" fill="#0e8f7e" opacity="${(0.16+0.62*w).toFixed(2)}"/>`;
        const Lq=dist(BS,q1)+dist(q1,q2)+dist(q2,UE);
        const uq=unit(sub(q2,UE));
        const vq=[UE[0]+Lq*uq[0],UE[1]+Lq*uq[1]];
        s+=`<circle cx="${vq[0]}" cy="${vq[1]}" r="${(1.6+2.2*w).toFixed(1)}" fill="#7c4dbe" opacity="${(0.12+0.55*w).toFixed(2)}"/>`;
        /* path 1's bounce is dispersive too — its scatter smears the bootstrapped VA1 */
        const q3=[RX,Math.max(WY0+6,Math.min(WY1-6,Pst[1]+sig*G2[i]))];
        const w3=Math.exp(-(G2[i]*G2[i])/2);
        if(i<5) s+=`<path d="M${BS[0]},${BS[1]} L${q3[0]},${q3[1]} L${UE[0]},${UE[1]}" fill="none" stroke="#e8720c" stroke-width="1" opacity="${(0.06+0.2*w3).toFixed(2)}"/>`;
        s+=`<circle cx="${q3[0]+3}" cy="${q3[1]}" r="${(2+2.6*w3).toFixed(1)}" fill="#e8720c" opacity="${(0.15+0.6*w3).toFixed(2)}"/>`;
        const Lq3=dist(BS,q3)+dist(q3,UE);
        const uq3=unit(sub(q3,UE));
        const vq3=[UE[0]+Lq3*uq3[0],UE[1]+Lq3*uq3[1]];
        s+=`<circle cx="${vq3[0]}" cy="${vq3[1]}" r="${(1.5+2*w3).toFixed(1)}" fill="#7c4dbe" opacity="${(0.12+0.5*w3).toFixed(2)}"/>`;
      }
    }
    s+=seg(P2,VA2d,"#7c4dbe",1.8,"5 4",0.9);
    s+=vaMark(VA2,"VA² (bounce 2 · ref)");
    s+=vaDot(VA2d);
  }
  s+=bsMark(BS,"BS",12,20)+ueMark(UE);
  svg3.innerHTML=s;
  const offR=Math.abs(P1[0]-RX)*M_PER_PX, offL=Math.abs(P2[0]-LX)*M_PER_PX;
  const va1err=dist(VA1e,VA1)*M_PER_PX;
  const mm=dist(ph,phD)*M_PER_PX;
  const consistent=offR<0.15&&offL<0.15&&va1err<0.15;
  $("stat3").innerHTML=
    `<b>L₂</b> = ${fmt(L2)} · <b>φ</b> = ${degFmt(phi)} · <b>ψ</b> = ${degFmt(psi)}<br>`+
    `corridor lock: ψ = φ ± 180° — first &amp; last legs parallel<br>`+
    `path-2 1-bounce test: ${mm.toFixed(1)} m apart → ${mm>0.5?"reject":"cannot reject"}<br>`+
    `path-1 1-bounce test: ${(dist(Ps,Psd)*M_PER_PX).toFixed(2)} m apart → ${dist(Ps,Psd)*M_PER_PX<0.5?"confirmed ✓":"suspect"}<br>`+
    `bootstrapped VA¹ error: ${va1err.toFixed(2)} m<br>`+
    (consistent?`<span class="ok">✓ 2-bounce interpretation consistent — P₁ on wall R, P₂ on wall L</span>`
               :`<span class="off">P₁ off wall R by ${offR.toFixed(2)} m · P₂ off wall L by ${offL.toFixed(2)} m</span>`)+
    (step3===6?`<br>${sig>0?`${G1.length} perturbed incidence-point samples / wall`:"illustrative smear removed (σ = 0)"}`:"");
}
["sSig3","s3L","s3P","s3D","s3Lb","s3Pb","s3Db"].forEach(id=>$(id).addEventListener("input",()=>{stop3();render3();}));
$("b3R").addEventListener("click",()=>{["s3L","s3P","s3D","s3Lb","s3Pb","s3Db"].forEach(id=>$(id).value=0);stop3();render3();});
$("b3Prev").addEventListener("click",()=>goStep3(step3-1,false));
$("b3Next").addEventListener("click",()=>goStep3(step3+1,true));
makeDraggable(svg3,()=>D3.UE,p=>{D3.UE=p;["s3L","s3P","s3D","s3Lb","s3Pb","s3Db"].forEach(id=>$(id).value=0);stop3();},[160,200,300,400],render3);
goStep3(1,false);

/* =================== DEMO 6: triple bounce (corner + floor) — stepped construction =================== */
const svg6=$("svg6");
const G3=gaussians(123,34);
/* wall C (the floor) is tilted 8° off wall A: a parallel pair would lock the two measured
   angles corridor-style for every UE — the tilt keeps the corner case generic */
const FTILT=8*Math.PI/180;
const D6={BS:[90,300],UE:[250,470],AY:225,AX0:40,AX1:330,
  BC:[330,225],BD:[Math.sin(TILT),Math.cos(TILT)],BLEN:320,
  CC:[60,585],CD:[Math.cos(FTILT),-Math.sin(FTILT)],CLEN:370};
D6.BN=[D6.BD[1],-D6.BD[0]];
D6.BEND=[D6.BC[0]+D6.BD[0]*D6.BLEN,D6.BC[1]+D6.BD[1]*D6.BLEN];
D6.CN=[D6.CD[1],-D6.CD[0]];                    // floor normal, pointing up into the room
D6.CEND=[D6.CC[0]+D6.CD[0]*D6.CLEN,D6.CC[1]+D6.CD[1]*D6.CLEN];
D6.VA1=[D6.BS[0],2*D6.AY-D6.BS[1]];            // mirror of BS in wall A
D6.VA2=mirrorLine(D6.VA1,D6.BC,D6.BN);         // mirror of VA1 in wall B
D6.VA3=mirrorLine(D6.VA2,D6.CC,D6.CN);         // mirror of VA2 in wall C
/* ground-truth specular geometry of all three input paths at the current UE */
function specular6(){
  const {BS,UE,VA1,VA2,VA3,AY,BC,BN,CC,CN}=D6;
  const P3t=hitLine(UE,VA3,CC,CN);
  const P2t=hitLine(P3t,VA2,BC,BN);
  const tA=(AY-P2t[1])/(VA1[1]-P2t[1]);
  const P1t=[P2t[0]+tA*(VA1[0]-P2t[0]),AY];    // path 3: A -> B -> C
  const Q2t=hitLine(UE,VA2,BC,BN);
  const tA2=(AY-Q2t[1])/(VA1[1]-Q2t[1]);
  const Q1t=[Q2t[0]+tA2*(VA1[0]-Q2t[0]),AY];   // path 2: A -> B
  return {L3:dist(UE,VA3),phi:aoaOf(unit(sub(VA3,UE))),psi:aoaOf(unit(sub(P1t,BS))),
          L2:dist(UE,VA2),phi2:aoaOf(unit(sub(VA2,UE))),
          L1:dist(UE,VA1),phi1:aoaOf(unit(sub(VA1,UE))),
          P1t,P2t,P3t,Q1t,Q2t};
}
const STEP6CAP=["",
 "① <b>Hypothesis: one bounce.</b> Path 3’s delay L₃ pins a string at BS &amp; UE (red); if the path bounced once, its point sits where the AoA ray crosses: P(φ).",
 "② <b>Test it with the AoD.</b> A single bounce must satisfy both measured angles at one point — but the AoD ray crosses the ellipse far from P(φ). Rejected.",
 "③ <b>Bootstrap VA¹ from path 1.</b> The single bounce off wall A is map-free: walk its full L⁽¹⁾ along φ⁽¹⁾ → VA¹.",
 "④ <b>Hypothesis: two bounces.</b> Anchor on VA¹ and run the double-bounce recipe on path 3’s data: string at VA¹ &amp; UE → P₂ʰ, leftover + AoD → P₁ʰ. Coherent — but the wall implied at P₁ʰ must mirror BS back onto its own anchor VA¹, and it misses. Self-contradiction → rejected.",
 "⑤ <b>Bootstrap VA² from path 2.</b> The full-length walk works at every order: L⁽²⁾ along φ⁽²⁾ → VA². (Path 2 itself was vetted by the double-bounce section’s machinery.)",
 "⑥ <b>Bounce-3 ellipse.</b> Pin L₃ at VA² &amp; UE (teal); the measured AoA picks P₃ on wall C.",
 "⑦ <b>Bounce-2 ellipse.</b> Peel off ‖UE→P₃‖; leftover string at VA¹ &amp; P₃ (green). No array saw this bounce — <em>aim from P₃ at the bootstrapped VA²</em> to pick P₂. This is a prefix-VA constraint, not a prior wall map or a new array angle.",
 "⑧ <b>Bounce-1 segment.</b> Peel again; leftover at BS &amp; P₂ (blue); the measured AoD picks P₁ on wall A. Both measured angles used — recursion complete.",
 "⑨ <b>The full picture:</b> the mirror ladder BS→VA¹→VA²→VA³, the data walk UE + L₃·û → VA³, and illustrative tangential perturbation clouds on all three bounces. This shows footprint accumulation only; it is not a calibrated rough-surface scattering law."];
let step6=1;
const T6={on:false,th:0,key:null}; let raf6=null;
function stop6(){T6.on=false;T6.key=null;if(raf6){cancelAnimationFrame(raf6);raf6=null;}}
const STEP6KEY={1:"bad",4:"hyp2",6:"e3",7:"e2",8:"e1"};
function goStep6(sN,animate){
  stop6();
  step6=Math.max(1,Math.min(9,sN));
  $("cap6").innerHTML=STEP6CAP[step6];
  $("b6Prev").disabled=step6===1; $("b6Next").disabled=step6===9;
  $("o6Step").textContent=step6+" / 9";
  const key=STEP6KEY[step6];
  if(animate&&key&&!reduced){
    T6.on=true;T6.key=key;
    const t0=performance.now(),DUR=2600;
    const tick=now=>{
      const f=Math.min(1,(now-t0)/DUR);
      T6.th=2*Math.PI*f;
      render6();
      if(f<1){raf6=requestAnimationFrame(tick);}else{stop6();render6();}
    };
    raf6=requestAnimationFrame(tick);
  } else render6();
}
function render6(){
  const {BS,UE,VA1,VA2,VA3,AY,AX0,AX1,BC,BD,BN,BLEN,BEND,CC,CD,CN,CLEN,CEND}=D6;
  const sig=+$("sSig6").value;
  $("oSig6").textContent=(sig*M_PER_PX).toFixed(1)+" m";
  const sp=specular6();
  const L3=sp.L3+(+$("s6L3").value)/M_PER_PX;
  const phi=sp.phi+(+$("s6P3").value); const u=dirOf(phi);
  const psi=sp.psi+(+$("s6D3").value); const ud=dirOf(psi);
  const L1s=sp.L1+(+$("s6L1").value)/M_PER_PX;
  const phi1=sp.phi1+(+$("s6P1").value); const u1=dirOf(phi1);
  const L2s=sp.L2+(+$("s6L2").value)/M_PER_PX;
  const phi2=sp.phi2+(+$("s6P2").value); const u2=dirOf(phi2);
  $("o6L3").textContent=(L3*M_PER_PX).toFixed(1)+" m"; $("o6P3").textContent=degFmt(phi); $("o6D3").textContent=degFmt(psi);
  $("o6L1").textContent=(L1s*M_PER_PX).toFixed(1)+" m"; $("o6P1").textContent=degFmt(phi1);
  $("o6L2").textContent=(L2s*M_PER_PX).toFixed(1)+" m"; $("o6P2").textContent=degFmt(phi2);
  /* bootstrapped anchors — full-length walks of paths 1 and 2 */
  const Ps1=rayEllipse(BS,UE,u1,L1s);
  const VA1e=[UE[0]+L1s*u1[0],UE[1]+L1s*u1[1]];
  const VA2e=[UE[0]+L2s*u2[0],UE[1]+L2s*u2[1]];
  /* 1-bounce hypothesis */
  const ph=rayEllipse(BS,UE,u,L3), phD=rayEllipse(UE,BS,ud,L3);
  /* 2-bounce hypothesis anchored on VA1: coherent, but must return its own anchor */
  const h2P2=rayEllipse(VA1e,UE,u,L3);
  const h2rest=L3-dist(UE,h2P2);
  const h2P1=rayEllipse(h2P2,BS,ud,h2rest);
  const h2a=unit(sub(BS,h2P1)), h2b=unit(sub(h2P2,h2P1));
  const h2n=unit([h2a[0]+h2b[0],h2a[1]+h2b[1]]);
  const h2cand=mirrorLine(BS,h2P1,h2n);
  const h2miss=dist(h2cand,VA1e)*M_PER_PX;
  /* the recursion: peel legs from the UE end */
  const P3=rayEllipse(VA2e,UE,u,L3);
  const L32=L3-dist(UE,P3);
  const um=unit(sub(VA2e,P3));                 // aim at the data-derived prefix VA2
  const P2=rayEllipse(VA1e,P3,um,L32);
  const L21=L32-dist(P3,P2);
  const P1=rayEllipse(P2,BS,ud,L21);
  const feas6=!!ellParams(BS,P2,L21);   // the twice-peeled string can collapse below ‖BS–P₂‖
  const VA3d=[UE[0]+L3*u[0],UE[1]+L3*u[1]];
  $("o6R32").textContent=(L32*M_PER_PX).toFixed(1)+" m";
  $("o6R21").textContent=(L21*M_PER_PX).toFixed(1)+" m";
  let s="", vout=0;
  /* walls */
  s+=seg([AX0,AY],[AX1,AY],"#16222e",5);
  for(let x=AX0;x<AX1;x+=18)s+=seg([x,AY],[x-9,AY-9],"#8a97a3",1.2);
  s+=txt(AX0+4,AY-14,"ref A · not input","#51606e",11);
  s+=seg(BC,BEND,"#16222e",5);
  for(let k=10;k<BLEN;k+=18){const q=[BC[0]+BD[0]*k,BC[1]+BD[1]*k];
    s+=seg(q,[q[0]+7*(BN[0]+BD[0]),q[1]+7*(BN[1]+BD[1])],"#8a97a3",1.2);}
  s+=txt(BC[0]+BD[0]*128+BN[0]*16,BC[1]+BD[1]*128+BN[1]*16,"ref B","#51606e",11)
    +txt(BC[0]+BD[0]*142+BN[0]*16,BC[1]+BD[1]*142+BN[1]*16,"not input","#51606e",11);
  s+=seg(CC,CEND,"#16222e",5);
  for(let k=10;k<CLEN;k+=18){const q=[CC[0]+CD[0]*k,CC[1]+CD[1]*k];
    s+=seg(q,[q[0]+7*(CD[0]-CN[0]),q[1]+7*(CD[1]-CN[1])],"#8a97a3",1.2);}
  s+=txt(CC[0]+CD[0]*150-CN[0]*16,CC[1]+CD[1]*150-CN[1]*16,"ref C · not input","#51606e",11);
  /* true physical path — ground truth, faint */
  s+=arrow(BS,sp.P1t,"#8a97a3",1.6,null,0.5)+arrow(sp.P1t,sp.P2t,"#8a97a3",1.6,null,0.5)
    +arrow(sp.P2t,sp.P3t,"#8a97a3",1.6,null,0.5)+arrow(sp.P3t,UE,"#8a97a3",1.6,null,0.5);
  const tracing=k=>T6.on&&T6.key===k;
  /* STEP 1+: the 1-bounce hypothesis */
  {
    const ep=ellParams(BS,UE,L3);
    if(tracing("bad")&&ep){
      s+=ellPartial(ep,T6.th,"#c22f2f",1.8,"7 5");
      s+=stringViz(BS,UE,ellPoint(ep,T6.th),"#c22f2f","1-bounce hyp. — string at BS &amp; UE: ‖x−UE‖ + ‖x−BS‖",[-160,-70]);
    } else {
      s+=ellipse(BS,UE,L3,"#c22f2f",1.5,"7 5",step6<=2?0.8:0.2);
      if(step6<=2) s+=pin(BS,"#c22f2f",11)+pin(UE,"#c22f2f",16);
      s+=arrow(UE,[UE[0]+u[0]*(dist(UE,ph)+40),UE[1]+u[1]*(dist(UE,ph)+40)],"#16222e",1.4,"6 4",step6<=2?0.6:0.3);
      if(step6<=2) s+=txt(UE[0]+u[0]*46+8,UE[1]+u[1]*46+4,"AoA φ","#16222e",11);
      s+=`<circle cx="${ph[0]}" cy="${ph[1]}" r="4.5" fill="none" stroke="#c22f2f" stroke-width="2" opacity="${step6<=2?1:0.3}"/>`;
      if(step6<=2){
        s+=seg(BS,ph,"#c22f2f",1.1,"2 3",0.6)+seg(UE,ph,"#c22f2f",1.1,"2 3",0.6);
        s+=txt(ph[0]+10,ph[1]+4,"P(φ) if 1 bounce","#c22f2f",10.5);
      }
      if(step6>=3) s+=txt(ph[0]+9,ph[1]-8,"1-bounce ✗","#c22f2f",10);
    }
  }
  /* STEP 2+: the AoD test */
  if(step6>=2&&!tracing("bad")){
    s+=arrow(BS,[BS[0]+ud[0]*(dist(BS,phD)+40),BS[1]+ud[1]*(dist(BS,phD)+40)],"#16222e",1.4,"6 4",step6===2?0.6:0.3);
    if(step6===2){
      s+=txt(BS[0]+ud[0]*58+6,BS[1]+ud[1]*58-8,"AoD ψ","#16222e",11);
      s+=`<circle cx="${phD[0]}" cy="${phD[1]}" r="4.5" fill="none" stroke="#c22f2f" stroke-width="2"/>`
        +txt(phD[0]+9,phD[1]-8,"P(ψ)","#c22f2f",10.5);
      s+=seg(ph,phD,"#c22f2f",2,"3 3",0.9);
      const mm1=dist(ph,phD)*M_PER_PX;
      s+=txt((ph[0]+phD[0])/2-14,(ph[1]+phD[1])/2,`${mm1.toFixed(1)} m apart → reject`,"#c22f2f",11,"end");
    }
  }
  /* STEP 3+: bootstrap VA1 from path 1 */
  if(step6>=3){
    s+=ellipse(BS,UE,L1s,"#e8720c",1.8,null,step6===3?0.9:0.3);
    if(step6===3) s+=pin(BS,"#e8720c",11)+pin(UE,"#e8720c",16);
    s+=arrow(BS,Ps1,"#e8720c",1.8,null,0.85)+arrow(Ps1,UE,"#e8720c",1.8,null,0.85);
    s+=arrow(UE,[UE[0]+u1[0]*(L1s+24),UE[1]+u1[1]*(L1s+24)],"#b45607",1.3,"6 4",0.7);
    s+=`<circle cx="${Ps1[0]}" cy="${Ps1[1]}" r="4" fill="#e8720c"/>`;
    s+=vaMark(VA1e,"VA¹ (path 1)");
    if(step6===3) s+=txt(VA1e[0]+12,VA1e[1]+22,"walk the full L⁽¹⁾ → VA¹","#b45607",10.5);
  }
  /* STEP 4: the 2-bounce hypothesis, anchored on VA1 — self-contradicts */
  if(step6>=4){
    const ep=ellParams(VA1e,UE,L3);
    if(tracing("hyp2")&&ep){
      s+=ellPartial(ep,T6.th,"#c22f2f",1.8,"3 4");
      s+=stringViz(VA1e,UE,ellPoint(ep,T6.th),"#c22f2f","2-bounce hyp. — string at VA¹ &amp; UE: ‖x−UE‖ + ‖x−VA¹‖",[-160,-70]);
    } else if(step6===4){
      s+=ellipse(VA1e,UE,L3,"#c22f2f",1.5,"3 4",0.75);
      s+=pin(VA1e,"#c22f2f",11)+pin(UE,"#c22f2f",16);
      s+=`<circle cx="${h2P2[0]}" cy="${h2P2[1]}" r="4.5" fill="none" stroke="#c22f2f" stroke-width="2"/>`+txt(h2P2[0]+9,h2P2[1]+4,"P₂ʰ","#c22f2f",10.5);
      s+=`<circle cx="${h2P1[0]}" cy="${h2P1[1]}" r="4.5" fill="none" stroke="#c22f2f" stroke-width="2"/>`+txt(h2P1[0]-8,h2P1[1]-10,"P₁ʰ","#c22f2f",10.5,"end");
      s+=seg(BS,h2P1,"#c22f2f",1.2,"2 3",0.7)+seg(h2P1,h2P2,"#c22f2f",1.2,"2 3",0.7)+seg(h2P2,UE,"#c22f2f",1.2,"2 3",0.7);
      const w=[-h2n[1],h2n[0]];
      s+=seg([h2P1[0]-70*w[0],h2P1[1]-70*w[1]],[h2P1[0]+70*w[0],h2P1[1]+70*w[1]],"#16222e",2.5,"5 3",0.75);
      s+=`<circle cx="${h2cand[0]}" cy="${h2cand[1]}" r="4.5" fill="none" stroke="#7c4dbe" stroke-width="2" stroke-dasharray="2 2"/>`;
      s+=seg(h2cand,VA1e,"#c22f2f",1.6,"3 3",0.9);
      s+=txt(h2cand[0]+10,h2cand[1]-8,`implied wall mirrors BS to here — misses VA¹ by ${h2miss.toFixed(1)} m → reject`,"#c22f2f",10.5);
    } else {
      s+=txt(h2P2[0]+8,h2P2[1]+14,"2-bounce ✗","#c22f2f",10);
    }
  }
  /* STEP 5+: bootstrap VA2 from path 2 — the walk works at every order */
  if(step6>=5&&!tracing("hyp2")){
    s+=arrow(BS,sp.Q1t,"#b8860b",1.6,null,step6===5?0.85:0.4)+arrow(sp.Q1t,sp.Q2t,"#b8860b",1.6,null,step6===5?0.85:0.4)+arrow(sp.Q2t,UE,"#b8860b",1.6,null,step6===5?0.85:0.4);
    s+=arrow(UE,[UE[0]+u2[0]*(L2s+24),UE[1]+u2[1]*(L2s+24)],"#96700a",1.3,"6 4",0.7);
    s+=vaMark(VA2e,"VA² (path 2)");
    if(step6===5) s+=txt(VA2e[0]-14,VA2e[1]+24,"walk the full L⁽²⁾ → VA²","#96700a",10.5,"end");
  }
  /* STEP 6+: bounce-3 ellipse */
  if(step6>=6){
    const ep=ellParams(VA2e,UE,L3);
    if(tracing("e3")&&ep){
      s+=ellPartial(ep,T6.th,"#0e8f7e",2);
      s+=stringViz(VA2e,UE,ellPoint(ep,T6.th),"#0a6b5e","bounce-3 — same L₃, string at VA² &amp; UE: ‖x−UE‖ + ‖x−VA²‖",[-160,-70]);
    } else {
      s+=ellipse(VA2e,UE,L3,"#0e8f7e",1.8,null,step6===6?0.9:0.45);
      if(step6===6) s+=pin(VA2e,"#0e8f7e",11)+pin(UE,"#0e8f7e",20);
      s+=arrow(UE,[UE[0]+u[0]*(dist(UE,P3)+34),UE[1]+u[1]*(dist(UE,P3)+34)],"#16222e",1.4,"6 4",0.5);
      s+=ipMark(P3,"P₃",10,-8);
    }
  }
  /* STEP 7+: bounce-2 ellipse — the middle bounce, picked by the prefix VA */
  if(step6>=7&&!tracing("e3")){
    const ep=ellParams(VA1e,P3,L32);
    if(tracing("e2")&&ep){
      s+=ellPartial(ep,T6.th,"#2ca02c",2);
      s+=stringViz(VA1e,P3,ellPoint(ep,T6.th),"#1d7a1d","bounce-2 — leftover L₃ − ‖UE→P₃‖, string at VA¹ &amp; P₃",[-160,-70]);
    } else if(ep){
      s+=ellipse(VA1e,P3,L32,"#2ca02c",1.7,null,step6===7?0.9:0.45);
      if(step6===7) s+=pin(VA1e,"#2ca02c",15)+pin(P3,"#2ca02c",10);
      s+=arrow(P3,[P3[0]+um[0]*(dist(P3,P2)+42),P3[1]+um[1]*(dist(P3,P2)+42)],"#5d3691",1.5,"6 4",0.75);
      if(step6===7) s+=txt(P3[0]+um[0]*70+8,P3[1]+um[1]*70,"aim at bootstrapped VA²","#5d3691",10.5);
      s+=ipMark(P2,"P₂",10,-8);
    }
  }
  /* STEP 8+: bounce-1 segment ellipse */
  if(step6>=8&&!tracing("e2")){
    const ep=ellParams(BS,P2,L21);
    if(tracing("e1")&&ep){
      s+=ellPartial(ep,T6.th,"#3a6ea5",2);
      s+=stringViz(BS,P2,ellPoint(ep,T6.th),"#2b527c","bounce-1 — leftover, string at BS &amp; P₂: ‖x−P₂‖ + ‖x−BS‖",[-160,-70]);
    } else if(ep){
      s+=ellipse(BS,P2,L21,"#3a6ea5",1.7,null,step6===8?0.9:0.45);
      if(step6===8) s+=pin(BS,"#3a6ea5",15)+pin(P2,"#3a6ea5",10);
      s+=arrow(BS,[BS[0]+ud[0]*(dist(BS,P1)+42),BS[1]+ud[1]*(dist(BS,P1)+42)],"#16222e",1.4,"6 4",0.5);
      s+=ipMark(P1,"P₁",-6,-12);
      s+=seg(BS,P1,"#0e8f7e",1.6,null,0.8)+seg(P1,P2,"#0e8f7e",1.6,null,0.8)+seg(P2,P3,"#0e8f7e",1.6,null,0.8)+seg(P3,UE,"#0e8f7e",1.6,null,0.8);
    } else if(step6===8){
      s+=txt(BS[0]-24,BS[1]+38,"leftover string &lt; ‖BS–P₂‖ — no bounce-1 ellipse at these settings","#c22f2f",10.5);
    }
  }
  /* STEP 9: the full picture */
  if(step6===9){
    s+=seg(BS,VA1,"#8a97a3",1.2,"5 4",0.85)+seg(VA1,VA2,"#8a97a3",1.2,"5 4",0.85)+seg(VA2,VA3,"#8a97a3",1.2,"5 4",0.85);
    s+=txt(BS[0]-12,262,"mirror in A","#8a97a3",10.5,"end")
      +txt((VA1[0]+VA2[0])/2-36,(VA1[1]+VA2[1])/2-12,"mirror in B","#8a97a3",10.5)
      +txt((VA2[0]+VA3[0])/2+10,(VA2[1]+VA3[1])/2,"mirror in C","#8a97a3",10.5);
    if(sig>0){
      for(let i=0;i<G1.length;i++){
        const sA=Math.max(AX0+6,Math.min(AX1-6,sp.P1t[0]+sig*G1[i]));
        const q1=[sA,AY];
        const sB0=(sp.P2t[0]-BC[0])*BD[0]+(sp.P2t[1]-BC[1])*BD[1];
        const sB=Math.max(8,Math.min(BLEN-8,sB0+sig*G2[i]));
        const q2=[BC[0]+BD[0]*sB,BC[1]+BD[1]*sB];
        const sC0=(sp.P3t[0]-CC[0])*CD[0]+(sp.P3t[1]-CC[1])*CD[1];
        const sC=Math.max(8,Math.min(CLEN-8,sC0+sig*G3[i]));
        const q3=[CC[0]+CD[0]*sC,CC[1]+CD[1]*sC];
        const w=Math.exp(-(G1[i]*G1[i]+G2[i]*G2[i]+G3[i]*G3[i])/2);
        if(i<7) s+=`<path d="M${BS[0]},${BS[1]} L${q1[0]},${q1[1]} L${q2[0]},${q2[1]} L${q3[0]},${q3[1]} L${UE[0]},${UE[1]}" fill="none" stroke="#0e8f7e" stroke-width="1" opacity="${(0.08+0.22*w).toFixed(2)}"/>`;
        s+=`<circle cx="${q1[0]}" cy="${q1[1]-3}" r="${(2.2+2.8*w).toFixed(1)}" fill="#0e8f7e" opacity="${(0.16+0.62*w).toFixed(2)}"/>`;
        s+=`<circle cx="${q2[0]+3*BN[0]}" cy="${q2[1]+3*BN[1]}" r="${(2.2+2.8*w).toFixed(1)}" fill="#0e8f7e" opacity="${(0.16+0.62*w).toFixed(2)}"/>`;
        s+=`<circle cx="${q3[0]+3*CN[0]}" cy="${q3[1]+3*CN[1]}" r="${(2.2+2.8*w).toFixed(1)}" fill="#0e8f7e" opacity="${(0.16+0.62*w).toFixed(2)}"/>`;
        const Lq=dist(BS,q1)+dist(q1,q2)+dist(q2,q3)+dist(q3,UE);
        const uq=unit(sub(q3,UE));
        const vq=[UE[0]+Lq*uq[0],UE[1]+Lq*uq[1]];
        if(vq[0]>-206&&vq[0]<776&&vq[1]>-156&&vq[1]<1166)
          s+=`<circle cx="${vq[0]}" cy="${vq[1]}" r="${(1.6+2.2*w).toFixed(1)}" fill="#7c4dbe" opacity="${(0.12+0.55*w).toFixed(2)}"/>`;
        else vout++;
      }
    }
    s+=seg(P3,VA3d,"#7c4dbe",1.8,"5 4",0.9);
    s+=vaMark(VA3,"VA³ (ref)",0.9);
    s+=vaDot(VA3d);
  }
  s+=bsMark(BS,"BS",-30,4)+ueMark(UE);
  svg6.innerHTML=s;
  const offA=Math.abs(P1[1]-AY)*M_PER_PX;
  const offB=Math.abs((P2[0]-BC[0])*BN[0]+(P2[1]-BC[1])*BN[1])*M_PER_PX;
  const offC=Math.abs((P3[0]-CC[0])*CN[0]+(P3[1]-CC[1])*CN[1])*M_PER_PX;
  const va1err=dist(VA1e,VA1)*M_PER_PX, va2err=dist(VA2e,VA2)*M_PER_PX, va3err=dist(VA3d,VA3)*M_PER_PX;
  const mm=dist(ph,phD)*M_PER_PX;
  const consistent=offA<0.15&&offB<0.15&&offC<0.15&&va1err<0.15&&va2err<0.15;
  $("stat6").innerHTML=
    `<b>L₃</b> = ${fmt(L3)} · <b>φ</b> = ${degFmt(phi)} · <b>ψ</b> = ${degFmt(psi)}<br>`+
    `1-bounce test: P(φ) ↔ P(ψ) ${mm.toFixed(1)} m apart → ${mm>0.5?"reject":"cannot reject"}<br>`+
    `2-bounce test: mirror(BS) misses VA¹ by ${h2miss.toFixed(1)} m → ${h2miss>0.5?"reject":"cannot reject"}<br>`+
    `anchors: VA¹ err ${va1err.toFixed(2)} m · VA² err ${va2err.toFixed(2)} m · walk → VA³ err ${va3err.toFixed(2)} m<br>`+
    (!feas6?`<span class="off">bounce-1 peel infeasible — leftover ${(L21*M_PER_PX).toFixed(1)} m &lt; ‖BS→P₂‖ ${(dist(BS,P2)*M_PER_PX).toFixed(1)} m</span>`
     :consistent?`<span class="ok">✓ 3-bounce interpretation consistent — P₁ on A, P₂ on B, P₃ on C</span>`
               :`<span class="off">P₁ off A ${offA.toFixed(2)} m · P₂ off B ${offB.toFixed(2)} m · P₃ off C ${offC.toFixed(2)} m</span>`)+
    (step6===9?`<br>${sig>0?`${G1.length} perturbed incidence-point samples / wall${vout?` · ${vout} phantom VAs beyond the frame`:""}`:"illustrative smear removed (σ = 0)"}`:"");
}
["sSig6","s6L1","s6P1","s6L2","s6P2","s6L3","s6P3","s6D3"].forEach(id=>$(id).addEventListener("input",()=>{stop6();render6();}));
$("b6R").addEventListener("click",()=>{["s6L1","s6P1","s6L2","s6P2","s6L3","s6P3","s6D3"].forEach(id=>$(id).value=0);stop6();render6();});
$("b6Prev").addEventListener("click",()=>goStep6(step6-1,false));
$("b6Next").addEventListener("click",()=>goStep6(step6+1,true));
makeDraggable(svg6,()=>D6.UE,p=>{D6.UE=p;["s6L1","s6P1","s6L2","s6P2","s6L3","s6P3","s6D3"].forEach(id=>$(id).value=0);stop6();},[230,430,300,510],render6);
goStep6(1,false);

/* =================== DEMO 7: triple bounce between parallel walls — stepped construction =================== */
const svg7=$("svg7");
const D7={BS:[150,480],UE:[250,230],LX:100,RX:320,WY0:150,WY1:600};
D7.VA1=[2*D7.RX-D7.BS[0],D7.BS[1]];    // mirror of BS in wall R
D7.VA2=[2*D7.LX-D7.VA1[0],D7.VA1[1]];  // mirror of VA1 in wall L
D7.VA3=[2*D7.RX-D7.VA2[0],D7.VA2[1]];  // mirror of VA2 in wall R — the third rung
function specular7(){
  const {BS,UE,VA1,VA2,VA3,LX,RX}=D7;
  const t3=(RX-UE[0])/(VA3[0]-UE[0]); const P3t=[RX,UE[1]+t3*(VA3[1]-UE[1])];
  const t2=(LX-P3t[0])/(VA2[0]-P3t[0]); const P2t=[LX,P3t[1]+t2*(VA2[1]-P3t[1])];
  const t1=(RX-P2t[0])/(VA1[0]-P2t[0]); const P1t=[RX,P2t[1]+t1*(VA1[1]-P2t[1])];
  const q2=(LX-UE[0])/(VA2[0]-UE[0]); const Q2t=[LX,UE[1]+q2*(VA2[1]-UE[1])];
  const q1=(RX-Q2t[0])/(VA1[0]-Q2t[0]); const Q1t=[RX,Q2t[1]+q1*(VA1[1]-Q2t[1])];
  return {L3:dist(UE,VA3),phi:aoaOf(unit(sub(VA3,UE))),psi:aoaOf(unit(sub(P1t,BS))),
          L2:dist(UE,VA2),phi2:aoaOf(unit(sub(VA2,UE))),
          L1:dist(UE,VA1),phi1:aoaOf(unit(sub(VA1,UE))),
          P1t,P2t,P3t,Q1t,Q2t};
}
const STEP7CAP=["",
 "① <b>Hypothesis: one bounce.</b> Path 3’s delay L₃ pins a string at BS &amp; UE (red); the AoA ray crosses it at P(φ).",
 "② <b>Test it with the AoD — and it passes.</b> Both measured rays cross the ellipse at the <em>same</em> point: odd bounce counts share the mirror lock ψ = −φ, so angles alone cannot tell 1 bounce from 3. After the associated prefix paths infer the corridor walls, forward visibility reveals that the BS ray crosses inferred wall R first. Occlusion then rejects the one-bounce reading.",
 "③ <b>Bootstrap VA¹ from path 1.</b> The single bounce off wall R is map-free: walk its full L⁽¹⁾ along φ⁽¹⁾ → VA¹.",
 "④ <b>Hypothesis: two bounces — dead on arrival.</b> Any double bounce in parallel walls locks ψ = φ ± 180° (legs anti-parallel); the measured pair is mirror-locked instead. Parity alone rejects. The VA¹-anchored construction (dashed) only confirms: its P₂ʰ lies beyond the corridor, reached through wall R.",
 "⑤ <b>Bootstrap VA² from path 2.</b> Walk the double bounce’s full L⁽²⁾ along φ⁽²⁾ → VA², the second rung of the ladder.",
 "⑥ <b>Bounce-3 ellipse.</b> Pin L₃ at VA² &amp; UE (teal); the measured AoA picks P₃ on wall R.",
 "⑦ <b>Bounce-2 ellipse.</b> Peel off ‖UE→P₃‖; leftover string at VA¹ &amp; P₃ (green). The middle bounce is invisible to both arrays — <em>aim from P₃ at the bootstrapped VA²</em> to pick P₂. This uses the associated prefix, not a supplied wall map.",
 "⑧ <b>Bounce-1 segment.</b> Peel again; leftover at BS &amp; P₂ (blue); the measured AoD picks P₁ on wall R. Recursion complete.",
 "⑨ <b>The full picture:</b> the image-source ladder marches VA¹ right, VA² left, VA³ right — same-parity rungs exactly two corridor-widths apart, since two mirrors compose to one translation — the data walk lands on VA³, and independent illustrative per-bounce smear accumulates into the broadest VA³ cluster. This is VA uncertainty, not physical diffuse scattering γ<sup>sc</sup>."];
let step7=1;
const T7={on:false,th:0,key:null}; let raf7=null;
function stop7(){T7.on=false;T7.key=null;if(raf7){cancelAnimationFrame(raf7);raf7=null;}}
const STEP7KEY={1:"bad",4:"hyp2",6:"e3",7:"e2",8:"e1"};
function goStep7(sN,animate){
  stop7();
  step7=Math.max(1,Math.min(9,sN));
  $("cap7").innerHTML=STEP7CAP[step7];
  $("b7Prev").disabled=step7===1; $("b7Next").disabled=step7===9;
  $("o7Step").textContent=step7+" / 9";
  const key=STEP7KEY[step7];
  if(animate&&key&&!reduced){
    T7.on=true;T7.key=key;
    const t0=performance.now(),DUR=2600;
    const tick=now=>{
      const f=Math.min(1,(now-t0)/DUR);
      T7.th=2*Math.PI*f;
      render7();
      if(f<1){raf7=requestAnimationFrame(tick);}else{stop7();render7();}
    };
    raf7=requestAnimationFrame(tick);
  } else render7();
}
function render7(){
  const {BS,UE,VA1,VA2,VA3,LX,RX,WY0,WY1}=D7;
  const sig=+$("sSig7").value;
  $("oSig7").textContent=(sig*M_PER_PX).toFixed(1)+" m";
  const sp=specular7();
  const L3=sp.L3+(+$("s7L3").value)/M_PER_PX;
  const phi=sp.phi+(+$("s7P3").value); const u=dirOf(phi);
  const psi=sp.psi+(+$("s7D3").value); const ud=dirOf(psi);
  const L1s=sp.L1+(+$("s7L1").value)/M_PER_PX;
  const phi1=sp.phi1+(+$("s7P1").value); const u1=dirOf(phi1);
  const L2s=sp.L2+(+$("s7L2").value)/M_PER_PX;
  const phi2=sp.phi2+(+$("s7P2").value); const u2=dirOf(phi2);
  $("o7L3").textContent=(L3*M_PER_PX).toFixed(1)+" m"; $("o7P3").textContent=degFmt(phi); $("o7D3").textContent=degFmt(psi);
  $("o7L1").textContent=(L1s*M_PER_PX).toFixed(1)+" m"; $("o7P1").textContent=degFmt(phi1);
  $("o7L2").textContent=(L2s*M_PER_PX).toFixed(1)+" m"; $("o7P2").textContent=degFmt(phi2);
  const Ps1=rayEllipse(BS,UE,u1,L1s);
  const VA1e=[UE[0]+L1s*u1[0],UE[1]+L1s*u1[1]];
  const VA2e=[UE[0]+L2s*u2[0],UE[1]+L2s*u2[1]];
  const ph=rayEllipse(BS,UE,u,L3), phD=rayEllipse(UE,BS,ud,L3);
  const h2P2=rayEllipse(VA1e,UE,u,L3);
  /* the recursion */
  const P3=rayEllipse(VA2e,UE,u,L3);
  const L32=L3-dist(UE,P3);
  const um=unit(sub(VA2e,P3));                 // aim at the data-derived prefix VA2
  const P2=rayEllipse(VA1e,P3,um,L32);
  const L21=L32-dist(P3,P2);
  const P1=rayEllipse(P2,BS,ud,L21);
  const feas7=!!ellParams(BS,P2,L21);   // the twice-peeled string can collapse below ‖BS–P₂‖
  const VA3d=[UE[0]+L3*u[0],UE[1]+L3*u[1]];
  $("o7R32").textContent=(L32*M_PER_PX).toFixed(1)+" m";
  $("o7R21").textContent=(L21*M_PER_PX).toFixed(1)+" m";
  /* parity: odd orders lock ψ = −φ, even orders lock ψ = φ ± 180° */
  const wr=a=>((a%360)+540)%360-180;
  const dOdd=Math.abs(wr(psi+phi)), dEven=Math.abs(wr(psi-phi-180));
  let s="", vout=0;
  /* walls */
  s+=seg([LX,WY0],[LX,WY1],"#16222e",5);
  for(let y=WY0;y<WY1;y+=18)s+=seg([LX,y],[LX-9,y+9],"#8a97a3",1.2);
  s+=txt(LX-14,WY0+16,"ref L · not input","#51606e",11,"end");
  s+=seg([RX,WY0],[RX,WY1],"#16222e",5);
  for(let y=WY0;y<WY1;y+=18)s+=seg([RX,y],[RX+9,y+9],"#8a97a3",1.2);
  s+=txt(RX+14,WY0+16,"ref R · not input","#51606e",11);
  /* true physical path — ground truth, faint */
  s+=arrow(BS,sp.P1t,"#8a97a3",1.6,null,0.5)+arrow(sp.P1t,sp.P2t,"#8a97a3",1.6,null,0.5)
    +arrow(sp.P2t,sp.P3t,"#8a97a3",1.6,null,0.5)+arrow(sp.P3t,UE,"#8a97a3",1.6,null,0.5);
  const tracing=k=>T7.on&&T7.key===k;
  /* STEP 1+: the 1-bounce hypothesis */
  {
    const ep=ellParams(BS,UE,L3);
    if(tracing("bad")&&ep){
      s+=ellPartial(ep,T7.th,"#c22f2f",1.8,"7 5");
      s+=stringViz(BS,UE,ellPoint(ep,T7.th),"#c22f2f","1-bounce hyp. — string at BS &amp; UE: ‖x−UE‖ + ‖x−BS‖",[-390,30]);
    } else {
      s+=ellipse(BS,UE,L3,"#c22f2f",1.5,"7 5",step7<=2?0.8:0.2);
      if(step7<=2) s+=pin(BS,"#c22f2f",11)+pin(UE,"#c22f2f",16);
      s+=arrow(UE,[UE[0]+u[0]*(dist(UE,ph)+40),UE[1]+u[1]*(dist(UE,ph)+40)],"#16222e",1.4,"6 4",step7<=2?0.6:0.3);
      if(step7<=2) s+=txt(UE[0]+u[0]*60+6,UE[1]+u[1]*60-8,"AoA φ","#16222e",11);
      s+=`<circle cx="${ph[0]}" cy="${ph[1]}" r="4.5" fill="none" stroke="#c22f2f" stroke-width="2" opacity="${step7<=2?1:0.3}"/>`;
      if(step7<=2){
        s+=seg(BS,ph,"#c22f2f",1.1,"2 3",0.6)+seg(UE,ph,"#c22f2f",1.1,"2 3",0.6);
        s+=txt(ph[0]+10,ph[1]-9,"P(φ) if 1 bounce","#c22f2f",10.5);
      }
      if(step7>=3) s+=txt(ph[0]+9,ph[1]-8,"1-bounce ✗ (occluded)","#c22f2f",10);
    }
  }
  /* STEP 2+: the AoD test — which PASSES; occlusion rejects instead */
  if(step7>=2&&!tracing("bad")){
    const IX=2*RX-LX;   // wall L as seen in mirror R
    s+=arrow(BS,[BS[0]+ud[0]*(dist(BS,phD)+40),BS[1]+ud[1]*(dist(BS,phD)+40)],"#16222e",1.4,"6 4",step7===2?0.6:0.3);
    if(step7===2){
      s+=seg([IX,WY0],[IX,WY1],"#8a97a3",2,"7 5",0.55)
        +txt(IX+12,WY0+34,"wall L, as seen in mirror R","#8a97a3",10.5);
      s+=txt(BS[0]+ud[0]*58+6,BS[1]+ud[1]*58+16,"AoD ψ","#16222e",11);
      const mm7=dist(ph,phD)*M_PER_PX;
      if(mm7<0.5){
        s+=`<circle cx="${phD[0]}" cy="${phD[1]}" r="8" fill="none" stroke="#c22f2f" stroke-width="1.6" stroke-dasharray="3 3"/>`
          +txt(ph[0]+12,ph[1]+16,"P(ψ) = P(φ) — angles agree!","#c22f2f",10.5);
      } else {
        s+=`<circle cx="${phD[0]}" cy="${phD[1]}" r="4.5" fill="none" stroke="#c22f2f" stroke-width="2"/>`
          +seg(ph,phD,"#c22f2f",2,"3 3",0.9)
          +txt((ph[0]+phD[0])/2+10,(ph[1]+phD[1])/2,`${mm7.toFixed(1)} m apart — perturbed off the lock`,"#c22f2f",11);
      }
      s+=seg(sp.P2t,ph,"#8a97a3",1.1,"3 4",0.4)
        +txt(sp.P2t[0]+(ph[0]-sp.P2t[0])*0.72+10,sp.P2t[1]+(ph[1]-sp.P2t[1])*0.72-8,"= P₂ in the mirror","#8a97a3",10.5);
      /* the occlusion: the BS ray pierces wall R exactly at the true P1 */
      const tocc=(RX-BS[0])/(ph[0]-BS[0]);
      if(tocc>0&&tocc<1){
        const XP=[RX,BS[1]+tocc*(ph[1]-BS[1])];
        s+=seg([XP[0]-6,XP[1]-6],[XP[0]+6,XP[1]+6],"#c22f2f",2.6)+seg([XP[0]-6,XP[1]+6],[XP[0]+6,XP[1]-6],"#c22f2f",2.6)
          +txt(XP[0]+11,XP[1]+4,"through wall R → reject (occlusion)","#c22f2f",10.5);
      }
    }
  }
  /* STEP 3+: bootstrap VA1 from path 1 */
  if(step7>=3){
    s+=ellipse(BS,UE,L1s,"#e8720c",1.8,null,step7===3?0.9:0.3);
    if(step7===3) s+=pin(BS,"#e8720c",11)+pin(UE,"#e8720c",16);
    s+=arrow(BS,Ps1,"#e8720c",1.8,null,0.85)+arrow(Ps1,UE,"#e8720c",1.8,null,0.85);
    s+=arrow(UE,[UE[0]+u1[0]*(L1s+24),UE[1]+u1[1]*(L1s+24)],"#b45607",1.3,"6 4",0.7);
    s+=`<circle cx="${Ps1[0]}" cy="${Ps1[1]}" r="4" fill="#e8720c"/>`;
    s+=vaMark(VA1e,"VA¹ (path 1)");
    if(step7===3) s+=txt(VA1e[0]+12,VA1e[1]+22,"walk the full L⁽¹⁾ → VA¹","#b45607",10.5);
  }
  /* STEP 4: the 2-bounce hypothesis — killed by parity */
  if(step7>=4){
    if(tracing("hyp2")){
      const ep=ellParams(VA1e,UE,L3);
      if(ep){
        s+=ellPartial(ep,T7.th,"#c22f2f",1.8,"3 4");
        s+=stringViz(VA1e,UE,ellPoint(ep,T7.th),"#c22f2f","2-bounce hyp. — string at VA¹ &amp; UE: ‖x−UE‖ + ‖x−VA¹‖",[-390,30]);
      }
    } else if(step7===4){
      s+=ellipse(VA1e,UE,L3,"#c22f2f",1.5,"3 4",0.75);
      s+=pin(VA1e,"#c22f2f",11)+pin(UE,"#c22f2f",16);
      s+=`<circle cx="${h2P2[0]}" cy="${h2P2[1]}" r="4.5" fill="none" stroke="#c22f2f" stroke-width="2"/>`+txt(h2P2[0]+9,h2P2[1]+4,"P₂ʰ — beyond the corridor","#c22f2f",10.5);
      s+=seg(UE,h2P2,"#c22f2f",1.2,"2 3",0.7);
      const tocc2=(RX-UE[0])/(h2P2[0]-UE[0]);
      if(tocc2>0&&tocc2<1){
        const XP2=[RX,UE[1]+tocc2*(h2P2[1]-UE[1])];
        s+=seg([XP2[0]-6,XP2[1]-6],[XP2[0]+6,XP2[1]+6],"#c22f2f",2.6)+seg([XP2[0]-6,XP2[1]+6],[XP2[0]+6,XP2[1]-6],"#c22f2f",2.6);
      }
      s+=txt(BS[0]-24,BS[1]-44,`parity: measured ψ = −φ (odd lock) — a double needs ψ = φ ± 180°`,"#c22f2f",10.5)
        +txt(BS[0]-24,BS[1]-30,`|ψ − φ ∓ 180°| = ${dEven.toFixed(1)}° → reject by parity alone`,"#c22f2f",10.5);
    } else {
      s+=txt(h2P2[0]+8,h2P2[1]+14,"2-bounce ✗ (parity)","#c22f2f",10);
    }
  }
  /* STEP 5+: bootstrap VA2 from path 2 */
  if(step7>=5&&!tracing("hyp2")){
    s+=arrow(BS,sp.Q1t,"#b8860b",1.6,null,step7===5?0.85:0.4)+arrow(sp.Q1t,sp.Q2t,"#b8860b",1.6,null,step7===5?0.85:0.4)+arrow(sp.Q2t,UE,"#b8860b",1.6,null,step7===5?0.85:0.4);
    s+=arrow(UE,[UE[0]+u2[0]*(L2s+24),UE[1]+u2[1]*(L2s+24)],"#96700a",1.3,"6 4",0.7);
    s+=vaMark(VA2e,"VA² (path 2)");
    if(step7===5) s+=txt(VA2e[0]+12,VA2e[1]+24,"walk the full L⁽²⁾ → VA²","#96700a",10.5);
  }
  /* STEP 6+: bounce-3 ellipse */
  if(step7>=6){
    const ep=ellParams(VA2e,UE,L3);
    if(tracing("e3")&&ep){
      s+=ellPartial(ep,T7.th,"#0e8f7e",2);
      s+=stringViz(VA2e,UE,ellPoint(ep,T7.th),"#0a6b5e","bounce-3 — same L₃, string at VA² &amp; UE: ‖x−UE‖ + ‖x−VA²‖",[-390,30]);
    } else {
      s+=ellipse(VA2e,UE,L3,"#0e8f7e",1.8,null,step7===6?0.9:0.45);
      if(step7===6) s+=pin(VA2e,"#0e8f7e",11)+pin(UE,"#0e8f7e",20);
      s+=arrow(UE,[UE[0]+u[0]*(dist(UE,P3)+34),UE[1]+u[1]*(dist(UE,P3)+34)],"#16222e",1.4,"6 4",0.5);
      s+=ipMark(P3,"P₃",12,-10);
    }
  }
  /* STEP 7+: bounce-2 ellipse — the middle bounce, picked by the prefix VA */
  if(step7>=7&&!tracing("e3")){
    const ep=ellParams(VA1e,P3,L32);
    if(tracing("e2")&&ep){
      s+=ellPartial(ep,T7.th,"#2ca02c",2);
      s+=stringViz(VA1e,P3,ellPoint(ep,T7.th),"#1d7a1d","bounce-2 — leftover L₃ − ‖UE→P₃‖, string at VA¹ &amp; P₃",[-390,30]);
    } else if(ep){
      s+=ellipse(VA1e,P3,L32,"#2ca02c",1.7,null,step7===7?0.9:0.45);
      if(step7===7) s+=pin(VA1e,"#2ca02c",15)+pin(P3,"#2ca02c",10);
      s+=arrow(P3,[P3[0]+um[0]*(dist(P3,P2)+42),P3[1]+um[1]*(dist(P3,P2)+42)],"#5d3691",1.5,"6 4",0.75);
      if(step7===7) s+=txt(P3[0]+um[0]*90+10,P3[1]+um[1]*90-8,"aim at bootstrapped VA²","#5d3691",10.5);
      s+=ipMark(P2,"P₂",-30,-10);
    }
  }
  /* STEP 8+: bounce-1 segment ellipse */
  if(step7>=8&&!tracing("e2")){
    const ep=ellParams(BS,P2,L21);
    if(tracing("e1")&&ep){
      s+=ellPartial(ep,T7.th,"#3a6ea5",2);
      s+=stringViz(BS,P2,ellPoint(ep,T7.th),"#2b527c","bounce-1 — leftover, string at BS &amp; P₂: ‖x−P₂‖ + ‖x−BS‖",[-390,30]);
    } else if(ep){
      s+=ellipse(BS,P2,L21,"#3a6ea5",1.7,null,step7===8?0.9:0.45);
      if(step7===8) s+=pin(BS,"#3a6ea5",15)+pin(P2,"#3a6ea5",10);
      s+=arrow(BS,[BS[0]+ud[0]*(dist(BS,P1)+42),BS[1]+ud[1]*(dist(BS,P1)+42)],"#16222e",1.4,"6 4",0.5);
      s+=ipMark(P1,"P₁",12,-10);
      s+=seg(BS,P1,"#0e8f7e",1.6,null,0.8)+seg(P1,P2,"#0e8f7e",1.6,null,0.8)+seg(P2,P3,"#0e8f7e",1.6,null,0.8)+seg(P3,UE,"#0e8f7e",1.6,null,0.8);
    } else if(step7===8){
      s+=txt(BS[0]-24,BS[1]+38,"leftover string &lt; ‖BS–P₂‖ — no bounce-1 ellipse at these settings","#c22f2f",10.5);
    }
  }
  /* STEP 9: the full picture — the ladder */
  if(step7===9){
    s+=seg(BS,VA1,"#8a97a3",1.2,"5 4",0.85);
    s+=`<path d="M${VA1[0]},${VA1[1]} Q${(VA1[0]+VA2[0])/2},${VA1[1]-46} ${VA2[0]},${VA2[1]}" fill="none" stroke="#8a97a3" stroke-width="1.2" stroke-dasharray="5 4" opacity="0.85"/>`;
    s+=`<path d="M${VA2[0]},${VA2[1]} Q${(VA2[0]+VA3[0])/2},${VA2[1]+52} ${VA3[0]},${VA3[1]}" fill="none" stroke="#8a97a3" stroke-width="1.2" stroke-dasharray="5 4" opacity="0.85"/>`;
    s+=txt((BS[0]+VA1[0])/2+30,BS[1]-8,"mirror in R","#8a97a3",10.5)
      +txt((VA1[0]+VA2[0])/2-30,VA1[1]-32,"mirror in L","#8a97a3",10.5)
      +txt((VA2[0]+VA3[0])/2-30,VA2[1]+44,"mirror in R","#8a97a3",10.5);
    /* same-parity rungs repeat at a fixed 2-width translation — measure VA1 -> VA3 */
    s+=seg([VA1[0],VA1[1]+64],[VA3[0],VA3[1]+64],"#8a97a3",1.1,"2 3",0.7)
      +seg([VA1[0],VA1[1]+56],[VA1[0],VA1[1]+72],"#8a97a3",1.1,null,0.7)
      +seg([VA3[0],VA3[1]+56],[VA3[0],VA3[1]+72],"#8a97a3",1.1,null,0.7)
      +txt((VA1[0]+VA3[0])/2-150,VA1[1]+86,"VA³ − VA¹ = 2 × width — two mirrors = one translation","#8a97a3",10.5);
    if(sig>0){
      for(let i=0;i<G1.length;i++){
        const q1=[RX,Math.max(WY0+6,Math.min(WY1-6,sp.P1t[1]+sig*G1[i]))];
        const q2=[LX,Math.max(WY0+6,Math.min(WY1-6,sp.P2t[1]+sig*G2[i]))];
        const q3=[RX,Math.max(WY0+6,Math.min(WY1-6,sp.P3t[1]+sig*G3[i]))];
        const w=Math.exp(-(G1[i]*G1[i]+G2[i]*G2[i]+G3[i]*G3[i])/2);
        if(i<7) s+=`<path d="M${BS[0]},${BS[1]} L${q1[0]},${q1[1]} L${q2[0]},${q2[1]} L${q3[0]},${q3[1]} L${UE[0]},${UE[1]}" fill="none" stroke="#0e8f7e" stroke-width="1" opacity="${(0.08+0.22*w).toFixed(2)}"/>`;
        s+=`<circle cx="${q1[0]+3}" cy="${q1[1]}" r="${(2.2+2.8*w).toFixed(1)}" fill="#0e8f7e" opacity="${(0.16+0.62*w).toFixed(2)}"/>`;
        s+=`<circle cx="${q2[0]-3}" cy="${q2[1]}" r="${(2.2+2.8*w).toFixed(1)}" fill="#0e8f7e" opacity="${(0.16+0.62*w).toFixed(2)}"/>`;
        s+=`<circle cx="${q3[0]+3}" cy="${q3[1]}" r="${(2.2+2.8*w).toFixed(1)}" fill="#0e8f7e" opacity="${(0.16+0.62*w).toFixed(2)}"/>`;
        const Lq=dist(BS,q1)+dist(q1,q2)+dist(q2,q3)+dist(q3,UE);
        const uq=unit(sub(q3,UE));
        const vq=[UE[0]+Lq*uq[0],UE[1]+Lq*uq[1]];
        if(vq[0]>-466&&vq[0]<1106&&vq[1]>-76&&vq[1]<866)
          s+=`<circle cx="${vq[0]}" cy="${vq[1]}" r="${(1.6+2.2*w).toFixed(1)}" fill="#7c4dbe" opacity="${(0.12+0.55*w).toFixed(2)}"/>`;
        else vout++;
      }
    }
    s+=seg(P3,VA3d,"#7c4dbe",1.8,"5 4",0.9);
    s+=vaMark(VA3,"VA³ (ref)",0.9);
    s+=vaDot(VA3d);
  }
  s+=bsMark(BS,"BS",12,20)+ueMark(UE);
  svg7.innerHTML=s;
  const offR1=Math.abs(P1[0]-RX)*M_PER_PX, offL2=Math.abs(P2[0]-LX)*M_PER_PX, offR3=Math.abs(P3[0]-RX)*M_PER_PX;
  const va1err=dist(VA1e,VA1)*M_PER_PX, va2err=dist(VA2e,VA2)*M_PER_PX, va3err=dist(VA3d,VA3)*M_PER_PX;
  const mm=dist(ph,phD)*M_PER_PX;
  const consistent=offR1<0.15&&offL2<0.15&&offR3<0.15&&va1err<0.15&&va2err<0.15;
  $("stat7").innerHTML=
    `<b>L₃</b> = ${fmt(L3)} · <b>φ</b> = ${degFmt(phi)} · <b>ψ</b> = ${degFmt(psi)}<br>`+
    `parity: |ψ + φ| = ${dOdd.toFixed(1)}° — odd lock ${dOdd<0.5?"✓":"✗"} · |ψ − φ ∓ 180°| = ${dEven.toFixed(1)}° — even lock ${dEven<0.5?"✓":"✗"}<br>`+
    `1-bounce angle test: P(φ) ↔ P(ψ) ${mm.toFixed(2)} m apart → ${mm<0.5?"cannot reject — occlusion must decide":"reject"}<br>`+
    `anchors: VA¹ err ${va1err.toFixed(2)} m · VA² err ${va2err.toFixed(2)} m · walk → VA³ err ${va3err.toFixed(2)} m<br>`+
    (!feas7?`<span class="off">bounce-1 peel infeasible — leftover ${(L21*M_PER_PX).toFixed(1)} m &lt; ‖BS→P₂‖ ${(dist(BS,P2)*M_PER_PX).toFixed(1)} m</span>`
     :consistent?`<span class="ok">✓ 3-bounce interpretation consistent — P₁, P₃ on wall R, P₂ on wall L</span>`
               :`<span class="off">P₁ off R ${offR1.toFixed(2)} m · P₂ off L ${offL2.toFixed(2)} m · P₃ off R ${offR3.toFixed(2)} m</span>`)+
    (step7===9?`<br>${sig>0?`${G1.length} perturbed incidence-point samples / wall${vout?` · ${vout} phantom VAs beyond the frame`:""}`:"illustrative smear removed (σ = 0)"}`:"");
}
["sSig7","s7L1","s7P1","s7L2","s7P2","s7L3","s7P3","s7D3"].forEach(id=>$(id).addEventListener("input",()=>{stop7();render7();}));
$("b7R").addEventListener("click",()=>{["s7L1","s7P1","s7L2","s7P2","s7L3","s7P3","s7D3"].forEach(id=>$(id).value=0);stop7();render7();});
$("b7Prev").addEventListener("click",()=>goStep7(step7-1,false));
$("b7Next").addEventListener("click",()=>goStep7(step7+1,true));
makeDraggable(svg7,()=>D7.UE,p=>{D7.UE=p;["s7L1","s7P1","s7L2","s7P2","s7L3","s7P3","s7D3"].forEach(id=>$(id).value=0);stop7();},[160,200,300,400],render7);
goStep7(1,false);

/* Shared 2-bounce algebra used by the stepped construction and §3 demos.
   A member chooses a last-leg split; the returned VA1 is only a candidate. */
function specAt(UE){
  const dV=sub(D2.VA2,UE);
  const tB=((D2.BC[0]-UE[0])*D2.BN[0]+(D2.BC[1]-UE[1])*D2.BN[1])/(dV[0]*D2.BN[0]+dV[1]*D2.BN[1]);
  const P2t=[UE[0]+tB*dV[0],UE[1]+tB*dV[1]];
  const tA=(D2.AY-P2t[1])/(D2.VA1[1]-P2t[1]);
  const P1t=[P2t[0]+tA*(D2.VA1[0]-P2t[0]),D2.AY];
  return {BS:D2.BS,UE,L:dist(UE,D2.VA2),u:unit(dV),ud:unit(sub(P1t,D2.BS))};
}
function member(sp,t2){
  const BS=sp.BS,P2=[sp.UE[0]+t2*sp.u[0],sp.UE[1]+t2*sp.u[1]],R=sp.L-t2,d=sub(P2,BS);
  const den=R-(sp.ud[0]*d[0]+sp.ud[1]*d[1]);if(den<=1e-6)return null;
  const t1=(R*R-(d[0]*d[0]+d[1]*d[1]))/(2*den);if(t1<=4||t1>=R-4)return null;
  const P1=[BS[0]+t1*sp.ud[0],BS[1]+t1*sp.ud[1]],a=unit(sub(BS,P1)),b=unit(sub(P2,P1)),n=unit([a[0]+b[0],a[1]+b[1]]);
  const dd=(BS[0]-P1[0])*n[0]+(BS[1]-P1[1])*n[1];
  return {P1,P2,t1,t2,cand:[BS[0]-2*dd*n[0],BS[1]-2*dd*n[1]],n};
}
function family(sp){
  const out=[];for(let k=0;k<=120;k++){const t2=sp.L*(0.06+0.80*k/120),m=member(sp,t2);if(m)out.push(m);}return out;
}
function poly4(fam){return "M"+fam.map(m=>m.cand[0].toFixed(1)+","+m.cand[1].toFixed(1)).join(" L");}

/* =================== DEMO 4: common-rotation ambiguity =================== */
const svg4=$("svg4"), C4=D2.BC, U4=[[240,440],[295,505]];
const add4=(a,b)=>[a[0]+b[0],a[1]+b[1]], scl4=(a,k)=>[a[0]*k,a[1]*k], dot4=(a,b)=>a[0]*b[0]+a[1]*b[1];
const rot4=(v,a)=>[v[0]*Math.cos(a)-v[1]*Math.sin(a),v[0]*Math.sin(a)+v[1]*Math.cos(a)];
const line4=(c,d,h,color,w,dash,o)=>seg(add4(c,scl4(d,-h)),add4(c,scl4(d,h)),color,w,dash,o);
const angle4=(a,b)=>Math.acos(Math.max(-1,Math.min(1,dot4(unit(a),unit(b)))))*180/Math.PI;
function state4(phi,UE){
  const dA=rot4([1,0],phi),dB=rot4(D2.BD,phi),nA=[-dA[1],dA[0]],nB=[-dB[1],dB[0]];
  const va1=mirrorLine(D2.BS,C4,nA),va2=mirrorLine(va1,C4,nB);
  const reverse=mirrorLine(mirrorLine(UE,C4,nB),C4,nA);
  const q2=hitLine(UE,va2,C4,nB),q1=hitLine(q2,va1,C4,nA);
  const legs=[sub(q1,D2.BS),sub(q2,q1),sub(UE,q2)];
  const incidence=[Math.acos(Math.min(1,Math.abs(dot4(unit(legs[0]),nA))))*180/Math.PI,
                   Math.acos(Math.min(1,Math.abs(dot4(unit(legs[1]),nB))))*180/Math.PI];
  const L=dist(D2.BS,q1)+dist(q1,q2)+dist(q2,UE);
  return {dA,dB,nA,nB,va1,va2,reverse,q1,q2,incidence,L,loss:7+0.035*incidence[0]+0.045*incidence[1]};
}
function render4(){
  const deg=+$("s4t").value,phi=deg*Math.PI/180,S=U4.map(u=>state4(phi,u)),S0=U4.map(u=>state4(0,u));
  $("o4t").textContent=(deg>=0?"+":"")+deg.toFixed(1)+"°";
  let s=`<rect x="12" y="12" width="510" height="510" rx="5" fill="#fbfcfd" stroke="#d7dee5"/>`
    +`<rect x="540" y="12" width="238" height="510" rx="5" fill="#fff" stroke="#d7dee5"/>`
    +txt(28,38,"same two-bounce measurements · different walls","#8a97a3",10)
    +txt(558,38,"what remains invariant?","#8a97a3",10);
  s+=`<g transform="translate(20 35) scale(.88)">`;
  /* zero-rotation reference */
  s+=line4(C4,[1,0],285,"#8a97a3",2,"4 4",0.25)+line4(C4,D2.BD,285,"#8a97a3",2,"4 4",0.25);
  if($("c4_famB").checked){
    s+=line4(C4,S[0].dA,285,"#16222e",4,null,.88)+line4(C4,S[0].dB,285,"#16222e",4,null,.88)
      +txt(add4(C4,scl4(S[0].dA,-205))[0],add4(C4,scl4(S[0].dA,-205))[1]-10,"wall A′","#16222e",10)
      +txt(add4(C4,scl4(S[0].dB,205))[0]+7,add4(C4,scl4(S[0].dB,205))[1],"wall B′","#16222e",10)
      +`<circle cx="${C4[0]}" cy="${C4[1]}" r="5" fill="#16222e"/>`+txt(C4[0]+9,C4[1]-9,"shared pivot c","#16222e",10);
  }
  if($("c4_famA").checked){
    const v=S[0].va2;
    s+=`<rect x="${v[0]-5.5}" y="${v[1]-5.5}" width="11" height="11" transform="rotate(45 ${v[0]} ${v[1]})" fill="none" stroke="#7c4dbe" stroke-width="2.4"/>`
      +txt(v[0]-12,v[1]+24,"composite VA² (fixed)","#5d3691",10,"end");
    U4.forEach((u,i)=>{
      const aoa=unit(sub(S[i].va2,u)),aod=unit(sub(S[i].reverse,D2.BS));
      s+=seg(u,add4(u,scl4(aoa,310)),"#7c4dbe",1.2,"6 4",.55)
        +seg(D2.BS,add4(D2.BS,scl4(aod,390)),"#7c4dbe",1.2,"6 4",.45);
    });
  }
  if($("c4_mem").checked){
    S.forEach((g,i)=>{
      const c=i?"#0e8f7e":"#e8720c";
      s+=`<path d="M${D2.BS.join(',')} L${g.q1.join(',')} L${g.q2.join(',')} L${U4[i].join(',')}" fill="none" stroke="${c}" stroke-width="${i?2.2:2.7}" ${i?'stroke-dasharray="5 3"':''} opacity=".9"/>`
        +`<circle cx="${g.q1[0]}" cy="${g.q1[1]}" r="4" fill="${c}"/><circle cx="${g.q2[0]}" cy="${g.q2[1]}" r="4" fill="${c}"/>`
        +txt(g.q1[0]+7,g.q1[1]-8,"q₁"+(i+1),c,9)+txt(g.q2[0]+7,g.q2[1]-8,"q₂"+(i+1),c,9);
    });
  }
  s+=bsMark(D2.BS,"BS",-31,-12)+arrow(U4[0],U4[1],"#2ca02c",1.4,"4 4",.65);
  U4.forEach((u,i)=>{s+=ueMark(u,"pose "+(i?"b":"a"));});
  s+=`</g>`;
  /* compact invariant / diagnostic panel */
  const vaDrift=Math.max(...S.map((g,i)=>dist(g.va2,S0[i].va2)))*M_PER_PX;
  const revDrift=Math.max(...S.map((g,i)=>dist(g.reverse,S0[i].reverse)))*M_PER_PX;
  const rangeDrift=Math.max(...S.map((g,i)=>Math.abs(g.L-S0[i].L)))*M_PER_PX;
  const bearingDrift=Math.max(...S.map((g,i)=>Math.max(angle4(sub(g.va2,U4[i]),sub(S0[i].va2,U4[i])),angle4(sub(g.reverse,D2.BS),sub(S0[i].reverse,D2.BS)))));
  const qMove=Math.max(...S.flatMap((g,i)=>[dist(g.q1,S0[i].q1),dist(g.q2,S0[i].q2)]))*M_PER_PX;
  const lossDelta=S.map((g,i)=>g.loss-S0[i].loss);
  s+=txt(558,72,"delay L","#16222e",10)+txt(752,72,rangeDrift.toExponential(1)+" m","#1d7a1d",10,"end")
    +txt(558,102,"AoA / AoD","#16222e",10)+txt(752,102,bearingDrift.toExponential(1)+"°","#1d7a1d",10,"end")
    +txt(558,132,"composite images","#16222e",10)+txt(752,132,Math.max(vaDrift,revDrift).toExponential(1)+" m","#1d7a1d",10,"end")
    +txt(558,174,"but incidence points move","#0a6b5e",10)+txt(752,174,qMove.toFixed(2)+" m","#0a6b5e",10,"end")
    +seg([558,193],[752,193],"#d7dee5",1);
  if($("c4_boot").checked){
    s+=txt(558,222,"illustrative bounce-loss change","#b45607",10);
    lossDelta.forEach((v,i)=>{const y=258+i*64,x0=655,w=Math.max(-78,Math.min(78,v*28));
      s+=txt(558,y+4,"pose "+(i?"b":"a"),"#51606e",9.5)+seg([x0-80,y],[x0+80,y],"#d7dee5",5)
        +`<rect x="${Math.min(x0,x0+w)}" y="${y-6}" width="${Math.abs(w)}" height="12" fill="#e8720c" opacity=".78"/>`
        +txt(752,y+4,(v>=0?"+":"")+v.toFixed(2)+" dB","#b45607",9.5,"end");
    });
    s+=txt(558,370,"geometry cannot see Δα;","#51606e",9.5)+txt(558,389,"calibrated radiometry might.","#51606e",9.5);
  }
  s+=txt(558,448,"common rotation is a true null mode","#c22f2f",9.5)+txt(558,470,"until another independent factor","#51606e",9.5)+txt(558,488,"breaks it.","#51606e",9.5);
  svg4.innerHTML=s;
  $("stat4").innerHTML=`delay drift: <b class="ok">${rangeDrift.toExponential(1)} m</b> · bearing drift: <b class="ok">${bearingDrift.toExponential(1)}°</b><br>max incidence-point motion: <b>${qMove.toFixed(2)} m</b><br>illustrative loss change: <b>${Math.max(...lossDelta.map(Math.abs)).toFixed(2)} dB</b>`;
}
["c4_famA","c4_famB","c4_mem","c4_boot"].forEach(id=>$(id).addEventListener("change",render4));
$("s4t").addEventListener("input",render4);render4();

/* =================== SECTION 4: KNOWN BS, UNKNOWN UE POSE AND MAP =================== */
/* ---- section 4.1: unknown UE, single bounce — changed foci (P, E), stepped ---- */
(function(){
const svg=$("svg31");
if(!svg)return;
const GD31=gaussians(101,60);
const CAP=["",
 "① <b>The unknowns.</b> Only the data exists — (τ, φ, ψ) measured by the two arrays. The wall (faint reference) and the UE are what we must find — the UE is not drawn at all: it is the thing being estimated; the old string, pinned at BS &amp; UE, cannot even be drawn: one focus is missing.",
 "② <b>Walk the AoD.</b> Eject the departure ray from the BS and walk the <em>full</em> measured length L along it. Its endpoint E is the mirror image of the UE across the unknown wall — the dual of §3.1's VA walk, and the replacement focus.",
 "③ <b>Hypothesize the bounce.</b> Choose P on the AoD ray (slider below). The walk to P spends ‖BS→P‖ of the string; ‖P→E‖ = L − ‖BS→P‖ is exactly what remains for the last leg.",
 "④ <b>Re-pin the string.</b> The leftover string, pinned at P, sweeps a circle through E — every point of it lies at the correct remaining distance from the bounce.",
 "⑤ <b>Cut with the reversed AoA.</b> φ<sub>body</sub> is the arrival direction measured in the UE's <em>own</em> frame. Turning it into a global direction requires the unknown θ. The last leg toward the UE runs opposite it: absolute direction φ<sub>body</sub> + θ + 180° — equivalently, the AoD ray rotated about P by π − (ψ − φ<sub>body</sub> − θ) (arc). That ray cuts the circle at the candidate UE(P). Note ‖P→UE(P)‖ = ‖P→E‖ <em>for every P</em> — the equality is built into the construction, so it cannot select the bounce.",
 "⑥ <b>The implied wall — and the family.</b> The wall runs along the middle between P→UE(P) and P→E: the bisector through P. <b>Finding P is finding the wall.</b> Slide P — every position stays coherent, and the candidates sweep the teal line for the selected θ. Now sweep θ through its full circle — that line pivots and every position <em>still</em> remains coherent: one path = a <b>two</b>-parameter family (P, θ), and the union of candidates fills the delay disk ‖x − BS‖ ≤ L (faint circle), not a line."];
let step=1;
const T={on:false,th:0}; let raf=null;
function stop(){T.on=false;if(raf){cancelAnimationFrame(raf);raf=null;}}
function goStep(n,animate){
  stop();
  step=Math.max(1,Math.min(6,n));
  $("cap31").innerHTML=CAP[step];
  $("b31Prev").disabled=step===1; $("b31Next").disabled=step===6;
  $("o31Step").textContent=step+" / 6";
  if(animate&&step===4&&!reduced){
    T.on=true;
    const t0=performance.now(),DUR=1900;
    const tick=now=>{
      const f=Math.min(1,(now-t0)/DUR);
      T.th=2*Math.PI*f;
      render();
      if(f<1){raf=requestAnimationFrame(tick);}else{stop();render();}
    };
    raf=requestAnimationFrame(tick);
  } else render();
}
function render(){
  const {BS,WC,WEND}=D1;
  const sp=specular1();
  const L=sp.L+(+$("s31L").value)/M_PER_PX;
  const phiMeas=sp.phi+(+$("s31A").value);
  const dH=+$("s31H").value;
  const phi=phiMeas+dH;   /* global arrival direction = measured local AoA + hypothesized heading */
  const psi=sp.psi+(+$("s31D").value);
  const u=dirOf(phi), d=dirOf(psi);
  $("o31L").textContent=(L*M_PER_PX).toFixed(1)+" m";
  $("o31A").textContent=degFmt(phiMeas);
  $("o31H").textContent=(dH>0?"+":"")+dH.toFixed(1)+"°";
  $("o31D").textContent=degFmt(psi);
  const E=[BS[0]+L*d[0],BS[1]+L*d[1]];
  const t=(+$("s31P").value)*L;
  const P=[BS[0]+t*d[0],BS[1]+t*d[1]];
  const r=dist(P,E);
  const U=[P[0]-r*u[0],P[1]-r*u[1]];
  $("o31P").textContent=(t*M_PER_PX).toFixed(1)+" m";
  let s="";
  s+=seg(WC,WEND,"#8a97a3",3,null,0.35)+txt(WC[0]+16,WC[1]+14,"true wall (unknown)","#8a97a3",10.5);
  if(step>=2){
    s+=arrow(BS,E,"#5d3691",1.5,"6 4",0.75);
    s+=txt(BS[0]+d[0]*70+8,BS[1]+d[1]*70-8,"AoD ψ (at the BS) — walked the full L","#5d3691",10.5);
    s+=`<rect x="${E[0]-5}" y="${E[1]-5}" width="10" height="10" transform="rotate(45 ${E[0]} ${E[1]})" fill="#7c4dbe"/>`
      +txt(E[0]-9,E[1]-11,"E = BS + L·d̂ — the mirrored UE","#5d3691",10.5,"end");
  }
  if(step>=3){
    s+=arrow(BS,P,"#51606e",2);
    s+=`<circle cx="${P[0]}" cy="${P[1]}" r="4.5" fill="#e8720c"/>`
      +txt(P[0]+9,P[1]-9,"P (hypothesis)","#b45607",10.5);
    s+=txt((BS[0]+P[0])/2-6,(BS[1]+P[1])/2+18,"‖BS→P‖","#51606e",10);
  }
  if(step>=4){
    if(T.on){
      const a0=Math.atan2(E[1]-P[1],E[0]-P[0]);
      let path="M"+(P[0]+r*Math.cos(a0))+" "+(P[1]+r*Math.sin(a0));
      const N=90;
      for(let k=1;k<=N*T.th/(2*Math.PI);k++){
        const a=a0+k/N*2*Math.PI;
        path+="L"+(P[0]+r*Math.cos(a))+" "+(P[1]+r*Math.sin(a));
      }
      s+=`<path d="${path}" fill="none" stroke="#e8720c" stroke-width="1.8" opacity="0.9"/>`;
      const ae=a0+T.th;
      s+=seg(P,[P[0]+r*Math.cos(ae),P[1]+r*Math.sin(ae)],"#b45607",1.2,"3 3",0.8);
    } else {
      s+=`<circle cx="${P[0]}" cy="${P[1]}" r="${r}" fill="none" stroke="#e8720c" stroke-width="1.8" opacity="0.85"/>`;
    }
    s+=txt(P[0]+12,P[1]+22,"‖x−P‖ = L − ‖BS→P‖","#b45607",10.5);
  }
  if(step>=5){
    s+=arrow(P,[P[0]-(r+38)*u[0],P[1]-(r+38)*u[1]],"#16222e",1.2,"6 4",0.5);
    s+=txt(P[0]-(r+44)*u[0],P[1]-(r+44)*u[1],"last leg ∥ −AoA: φ + 180°","#16222e",10.5);
    /* the rotation from the AoD ray to the last leg: π − (ψ − φ) */
    {
      const rel=((phi+180-psi)%360+360)%360;
      let arc="";
      const N=28;
      for(let k=0;k<=N;k++){
        const a=psi+rel*k/N, q=dirOf(a);
        arc+=(k?"L":"M")+(P[0]+26*q[0])+" "+(P[1]+26*q[1]);
      }
      s+=`<path d="${arc}" fill="none" stroke="#b45607" stroke-width="1.4" opacity="0.85"/>`;
      const mid=dirOf(psi+rel/2);
      s+=txt(P[0]+60*mid[0],P[1]+60*mid[1]+4,`π − (ψ − φ_body − θ) = ${rel.toFixed(0)}°`,"#b45607",10,mid[0]<0?"end":"start");
    }
    s+=arrow(P,U,"#51606e",2);
    s+=`<circle cx="${U[0]}" cy="${U[1]}" r="6" fill="#2ca02c" stroke="#fff" stroke-width="1.6"/>`
      +txt(U[0]+10,U[1]+4,"UE(P)","#1d7a1d",11);
  }
  if(step>=6){
    const n=unit(sub(U,E)), w=[-n[1],n[0]];
    s+=seg([P[0]-130*w[0],P[1]-130*w[1]],[P[0]+130*w[0],P[1]+130*w[1]],"#16222e",4.2,null,0.9);
    s+=txt(P[0]+130*w[0]+6,P[1]+130*w[1]+4,"implied wall — the middle between P→UE and P→E","#16222e",10.5);
    const o=[BS[0]-L*u[0],BS[1]-L*u[1]], v=unit([d[0]+u[0],d[1]+u[1]]);
    s+=seg([o[0]-900*v[0],o[1]-900*v[1]],[o[0]+900*v[0],o[1]+900*v[1]],"#0e8f7e",1.6,"7 5",0.6);
    /* the heading dimension: with θ free the candidate line pivots and sweeps the whole delay disk */
    s+=`<circle cx="${BS[0]}" cy="${BS[1]}" r="${L}" fill="none" stroke="#0e8f7e" stroke-width="1.1" stroke-dasharray="2 5" opacity="0.4"/>`;
    s+=txt(BS[0]+L*Math.cos(0.42)-10,BS[1]-L*Math.sin(0.42)-8,"θ unknown ⇒ union fills the delay disk ‖x−BS‖ ≤ L","#0a6b5e",10,"end");
  }
  const sig=+$("sS31").value;
  $("oS31").textContent=(sig*M_PER_PX).toFixed(1)+" m";
  if(sig>0){
    for(let k=0;k<24;k++){
      const g=GD31[k];
      const w=Math.exp(-0.5*g*g);
      const Sk=[sp.Pt[0]+g*sig*D1.WD[0],sp.Pt[1]+g*sig*D1.WD[1]];
      const Ls=dist(BS,Sk)+dist(Sk,D1.UE);
      const us=dirOf(aoaOf(unit(sub(Sk,D1.UE)))+dH), ds=unit(sub(Sk,BS));
      const Es=[BS[0]+Ls*ds[0],BS[1]+Ls*ds[1]];
      const fr=(+$("s31P").value);
      const ts=fr*Ls, Ps=[BS[0]+ts*ds[0],BS[1]+ts*ds[1]];
      const rs=dist(Ps,Es);
      const Us=[Ps[0]-rs*us[0],Ps[1]-rs*us[1]];
      s+=`<circle cx="${Sk[0]}" cy="${Sk[1]}" r="${1.4+1.4*w}" fill="#0e8f7e" opacity="${0.14+0.4*w}"/>`;
      if(step>=2)s+=`<circle cx="${Es[0]}" cy="${Es[1]}" r="${1.6+1.6*w}" fill="#7c4dbe" opacity="${0.12+0.4*w}"/>`;
      if(step>=5)s+=`<circle cx="${Us[0]}" cy="${Us[1]}" r="${1.6+1.6*w}" fill="#2ca02c" opacity="${0.12+0.4*w}"/>`;
    }
  }
  s+=bsMark(BS,"BS",-30,4);
  svg.innerHTML=s;
  const n2=unit(sub(U,E));
  const tilt=Math.asin(Math.min(1,Math.abs(n2[0]*D1.WN[1]-n2[1]*D1.WN[0])))*180/Math.PI;
  $("stat31").innerHTML= step<5 ? "" :
    `‖P→UE(P)‖ = ‖P→E‖ — <b>equal for every P and every θ</b>: the construction builds the equality in, so it can choose neither the bounce nor the heading. Two more constraints are needed.`+
    (step>=6?`<br>implied wall tilt off the reference wall: ${tilt.toFixed(1)}° — the selected θ reads as a wall tilt of |θ|/2: the two are indistinguishable from one path`:``);
}
["s31P","s31H","s31L","s31A","s31D","sS31"].forEach(id=>$(id).addEventListener("input",()=>{stop();render();}));
$("b31R").addEventListener("click",()=>{["s31L","s31A","s31D"].forEach(id=>$(id).value=0);stop();render();});
$("b31Prev").addEventListener("click",()=>goStep(step-1,false));
$("b31Next").addEventListener("click",()=>goStep(step+1,true));
goStep(1,false);
})();

/* ---- section 4.2: unknown UE, double bounce — the focus climbs the ladder ---- */
(function(){
const svg=$("svg32");
if(!svg)return;
const GD32=gaussians(202,120);
const CAP=["",
 "① <b>The unknowns.</b> Two measured paths: path 1 (τ⁽¹⁾, φ⁽¹⁾, ψ⁽¹⁾), a single bounce off wall A, and path 2 (τ, φ, ψ), a double bounce off A then B. Both walls and the UE are unknown — no string is pinnable anywhere.",
 "② <b>Path 1 = §4.1 verbatim.</b> Walk ψ⁽¹⁾ the full L⁽¹⁾ → E⁽¹⁾, the mirrored UE. Hypothesize P⁽¹⁾ on the ray (slider); UE₁ sits the leftover distance down the reverse-AoA ray, and the candidates sweep line 1 ⊥ wall A.",
 "③ <b>The hypothesis is a wall.</b> P⁽¹⁾ fixes wall A — the middle between P⁽¹⁾→UE₁ and P⁽¹⁾→E⁽¹⁾ — and it stays parallel to the true wall for every P⁽¹⁾. Finding P⁽¹⁾ is finding wall A.",
 "④ <b>Strip path 2 at wall A.</b> Path 2's ψ-ray crosses the hypothesized wall A at P₁ and reflects. The focus climbs the ladder: the string re-pins at P₁, and the full leftover walk L₂ − ‖BS→P₁‖ lands on E₂ — the sub-problem's mirrored UE.",
 "⑤ <b>Hypothesize P₂.</b> On the reflected ray (slider), UE₂ sits the remaining length down path 2's reverse-AoA ray, with ‖P₂→UE₂‖ = ‖P₂→E₂‖ built in as always. The candidates sweep line 2 ⊥ wall B: a second family.",
 "⑥ <b>Cross — and test ray order.</b> The UE must lie on both candidate lines. When their crossing also lies on the <em>forward</em> reflected ray with non-negative segment lengths, it solves P₂ and wall B follows. Sweep P⁽¹⁾ and θ: feasible members retain a two-parameter ambiguity, but combinations that put a bounce behind its ray origin or exhaust L₂ too early are rejected in red. The double bounce changes and prunes the family; it does not supply the heading."];
let step=1;
function goStep(n){
  step=Math.max(1,Math.min(6,n));
  $("cap32").innerHTML=CAP[step];
  $("b32Prev").disabled=step===1; $("b32Next").disabled=step===6;
  $("o32Step").textContent=step+" / 6";
  $("s32P2").disabled=step>=6;
  render();
}
function pathData(){
  const UE=D2.UE;
  const L1s=dist(UE,D2.VA1), u1s=unit(sub(D2.VA1,UE));
  const sA=(D2.AY-UE[1])/(D2.VA1[1]-UE[1]);
  const P1a=[UE[0]+sA*(D2.VA1[0]-UE[0]),D2.AY];
  const sp=specAt(UE);
  return {
    L1:L1s+(+$("s32L1").value)/M_PER_PX,
    phi1:aoaOf(u1s)+(+$("s32A1").value),
    psi1:aoaOf(unit(sub(P1a,D2.BS)))+(+$("s32D1").value),
    L2:sp.L+(+$("s32L2").value)/M_PER_PX,
    phi2:aoaOf(sp.u)+(+$("s32A2").value),
    psi2:aoaOf(sp.ud)+(+$("s32D2").value)};
}
function crossing(l1,l2){
  const den=l1.v[0]*l2.v[1]-l1.v[1]*l2.v[0];
  if(Math.abs(den)<1e-12)return null;
  const w=[l2.o[0]-l1.o[0],l2.o[1]-l1.o[1]];
  const t=(w[0]*l2.v[1]-w[1]*l2.v[0])/den;
  return [l1.o[0]+t*l1.v[0],l1.o[1]+t*l1.v[1]];
}
const fullLine=(l,c,dash,o)=>seg([l.o[0]-900*l.v[0],l.o[1]-900*l.v[1]],
                                 [l.o[0]+900*l.v[0],l.o[1]+900*l.v[1]],c,1.6,dash,o);
function render(){
  const {BS,AY,AX0,AX1,BC,BEND}=D2;
  const dt=pathData();
  const dH=+$("s32H").value;
  $("o32H").textContent=(dH>0?"+":"")+dH.toFixed(1)+"°";
  /* one UE, one unknown heading state: θ rotates both measured body-frame AoAs into global directions */
  const u1=dirOf(dt.phi1+dH), d1=dirOf(dt.psi1);
  const u2=dirOf(dt.phi2+dH), d2=dirOf(dt.psi2);
  $("o32L1").textContent=(dt.L1*M_PER_PX).toFixed(1)+" m";
  $("o32A1").textContent=degFmt(dt.phi1);
  $("o32D1").textContent=degFmt(dt.psi1);
  $("o32L2").textContent=(dt.L2*M_PER_PX).toFixed(1)+" m";
  $("o32A2").textContent=degFmt(dt.phi2);
  $("o32D2").textContent=degFmt(dt.psi2);
  /* path-1 construction (3.1) */
  const E1=[BS[0]+dt.L1*d1[0],BS[1]+dt.L1*d1[1]];
  const t1=(+$("s32P1").value)*dt.L1;
  const Pw=[BS[0]+t1*d1[0],BS[1]+t1*d1[1]];
  const r1=dist(Pw,E1);
  const U1=[Pw[0]-r1*u1[0],Pw[1]-r1*u1[1]];
  const line1={o:[BS[0]-dt.L1*u1[0],BS[1]-dt.L1*u1[1]],v:unit([d1[0]+u1[0],d1[1]+u1[1]])};
  const nA=unit(sub(U1,E1)), wA=[-nA[1],nA[0]];
  /* strip path 2 at hypothesized wall A; signed lengths enforce ray order */
  const H1=hitRayLine(BS,d2,Pw,nA);
  const tp=H1?.t??NaN, P1=H1?.p??BS;
  const denA=nA[0]*d2[0]+nA[1]*d2[1];
  const e=[d2[0]-2*denA*nA[0],d2[1]-2*denA*nA[1]];
  const rem=dt.L2-tp;
  const entryFeasible=!!H1&&tp>0&&rem>0;
  const E2=entryFeasible?[P1[0]+rem*e[0],P1[1]+rem*e[1]]:null;
  const line2=entryFeasible?{o:[P1[0]-rem*u2[0],P1[1]-rem*u2[1]],v:unit([e[0]+u2[0],e[1]+u2[1]])}:null;
  const Xraw=line2?crossing(line1,line2):null;
  const sc=Math.hypot(e[0]+u2[0],e[1]+u2[1]);
  const m2Solved=Xraw&&sc>1e-9?((Xraw[0]-line2.o[0])*line2.v[0]+(Xraw[1]-line2.o[1])*line2.v[1])/sc:NaN;
  const solutionFeasible=entryFeasible&&!!Xraw&&Number.isFinite(m2Solved)&&m2Solved>0&&m2Solved<rem;
  const m2=(step>=6&&solutionFeasible)?m2Solved:(entryFeasible?(+$("s32P2").value)*rem:NaN);
  const P2=entryFeasible?[P1[0]+m2*e[0],P1[1]+m2*e[1]]:null;
  const r2=P2&&E2?dist(P2,E2):NaN;
  const U2=P2?[P2[0]-r2*u2[0],P2[1]-r2*u2[1]]:null;
  $("o32P1").textContent=(t1*M_PER_PX).toFixed(1)+" m";
  $("o32P2").textContent=step>=6?(solutionFeasible?"solved":"infeasible"):(entryFeasible?(m2*M_PER_PX).toFixed(1)+" m":"—");
  let s="";
  /* reference walls */
  s+=seg([AX0,AY],[AX1,AY],"#8a97a3",3,null,0.32)+seg(BC,BEND,"#8a97a3",3,null,0.32);
  s+=txt(AX0+4,AY-8,"wall A (reference)","#8a97a3",10)+txt(BEND[0]-8,BEND[1]-12,"wall B","#8a97a3",10,"end");
  if(step>=2){
    s+=arrow(BS,E1,"#5d3691",1.4,"6 4",0.7);
    s+=`<rect x="${E1[0]-5}" y="${E1[1]-5}" width="10" height="10" transform="rotate(45 ${E1[0]} ${E1[1]})" fill="#7c4dbe"/>`
      +txt(E1[0]+10,E1[1]+4,"E⁽¹⁾","#5d3691",10.5);
    s+=fullLine(line1,"#0e8f7e","7 5",0.5);
    s+=arrow(BS,Pw,"#51606e",1.8)+arrow(Pw,U1,"#51606e",1.8);
    s+=`<circle cx="${Pw[0]}" cy="${Pw[1]}" r="4.5" fill="#e8720c"/>`+txt(Pw[0]+8,Pw[1]-8,"P⁽¹⁾","#b45607",10.5);
    s+=`<circle cx="${U1[0]}" cy="${U1[1]}" r="6" fill="#2ca02c" stroke="#fff" stroke-width="1.6"/>`
      +txt(U1[0]+10,U1[1]+4,"UE₁","#1d7a1d",11)+headingMark(U1,dH);
    s+=txt(line1.o[0]+250*line1.v[0]+8,line1.o[1]+250*line1.v[1]+16,"line 1 ⊥ wall A","#0a6b5e",10);
  }
  if(step>=3){
    s+=seg([Pw[0]-150*wA[0],Pw[1]-150*wA[1]],[Pw[0]+150*wA[0],Pw[1]+150*wA[1]],"#16222e",4,null,0.85);
    s+=txt(Pw[0]+150*wA[0]+6,Pw[1]+150*wA[1]+4,"wall A hypothesis","#16222e",10.5);
  }
  if(step>=4&&entryFeasible){
    s+=arrow(BS,P1,"#7c4dbe",1.4,"5 4",0.7);
    s+=`<circle cx="${P1[0]}" cy="${P1[1]}" r="4.5" fill="#7c4dbe"/>`+txt(P1[0]+8,P1[1]-8,"P₁ = ψ-ray × wall A","#5d3691",10);
    s+=arrow(P1,E2,"#7c4dbe",1.2,"5 4",0.55);
    s+=`<rect x="${E2[0]-5}" y="${E2[1]-5}" width="10" height="10" transform="rotate(45 ${E2[0]} ${E2[1]})" fill="#7c4dbe" opacity="0.8"/>`
      +txt(E2[0]-9,E2[1]-11,"E₂ (sub-problem)","#5d3691",10,"end");
  }else if(step>=4){
    const q=[BS[0]+150*d2[0],BS[1]+150*d2[1]];
    s+=arrow(BS,q,"#c22f2f",1.8,"5 4",0.8)+txt(q[0]+8,q[1]-8,"infeasible: no forward path-2 hit within L₂","#c22f2f",10.5);
  }
  if(step>=5&&entryFeasible){
    s+=fullLine(line2,"#0e8f7e","3 5",0.5);
    s+=arrow(P1,P2,"#51606e",1.8)+arrow(P2,U2,"#51606e",1.8);
    s+=`<circle cx="${P2[0]}" cy="${P2[1]}" r="4.5" fill="#e8720c"/>`+txt(P2[0]+8,P2[1]-8,"P₂","#b45607",10.5);
    s+=`<circle cx="${U2[0]}" cy="${U2[1]}" r="7" fill="none" stroke="#2ca02c" stroke-width="2.2"/>`
      +txt(U2[0]+11,U2[1]+13,"UE₂","#1d7a1d",11);
    s+=txt(line2.o[0]+320*line2.v[0]+8,line2.o[1]+320*line2.v[1]-8,"line 2 ⊥ wall B","#0a6b5e",10);
  }
  if(step>=6&&solutionFeasible){
    const X=Xraw;
    s+=`<path d="M${X[0]-6} ${X[1]-6}L${X[0]+6} ${X[1]+6}M${X[0]-6} ${X[1]+6}L${X[0]+6} ${X[1]-6}" stroke="#0e8f7e" stroke-width="2.2"/>`
      +`<circle cx="${X[0]}" cy="${X[1]}" r="11" fill="none" stroke="#0e8f7e" stroke-width="2"/>`;
    const oB=unit(sub(X,P2));
    const nB=unit([oB[0]-e[0],oB[1]-e[1]]);
    const wB=[-nB[1],nB[0]];
    s+=seg([P2[0]-120*wB[0],P2[1]-120*wB[1]],[P2[0]+120*wB[0],P2[1]+120*wB[1]],"#16222e",4,null,0.85);
    s+=txt(P2[0]-120*wB[0]-6,P2[1]-120*wB[1]+14,"wall B hypothesis","#16222e",10.5,"end");
  }
  const sig=+$("sS32").value;
  $("oS32").textContent=(sig*M_PER_PX).toFixed(1)+" m";
  if(sig>0&&entryFeasible){
    const UEt=D2.UE;
    const sAt=(AY-UEt[1])/(D2.VA1[1]-UEt[1]);
    const B1t=[UEt[0]+sAt*(D2.VA1[0]-UEt[0]),AY];
    const dq=sub(D2.VA2,UEt);
    const tb=((BC[0]-UEt[0])*D2.BN[0]+(BC[1]-UEt[1])*D2.BN[1])/((dq[0]*D2.BN[0]+dq[1]*D2.BN[1])||1e-12);
    const Q2t=[UEt[0]+tb*dq[0],UEt[1]+tb*dq[1]];
    const tA2=(AY-Q2t[1])/(D2.VA1[1]-Q2t[1]);
    const Q1t=[Q2t[0]+tA2*(D2.VA1[0]-Q2t[0]),AY];
    const fr=(+$("s32P1").value);
    for(let k=0;k<24;k++){
      const g0=GD32[3*k],g1=GD32[3*k+1],g2=GD32[3*k+2];
      const w=Math.exp(-0.5*(g0*g0+g1*g1+g2*g2)/3);
      const S0=[B1t[0]+g0*sig,B1t[1]];
      const L1s=dist(BS,S0)+dist(S0,UEt);
      const u1s=dirOf(aoaOf(unit(sub(S0,UEt)))+dH), d1s=unit(sub(S0,BS));
      const E1k=[BS[0]+L1s*d1s[0],BS[1]+L1s*d1s[1]];
      const t1k=fr*L1s, Pwk=[BS[0]+t1k*d1s[0],BS[1]+t1k*d1s[1]];
      const U1k=[Pwk[0]-(L1s-t1k)*u1s[0],Pwk[1]-(L1s-t1k)*u1s[1]];
      const nAk=unit(sub(U1k,E1k));
      const S1k=[Q1t[0]+g1*sig,Q1t[1]];
      const S2k=[Q2t[0]+g2*sig*D2.BD[0],Q2t[1]+g2*sig*D2.BD[1]];
      const L2s=dist(BS,S1k)+dist(S1k,S2k)+dist(S2k,UEt);
      const u2s=dirOf(aoaOf(unit(sub(S2k,UEt)))+dH), d2s=unit(sub(S1k,BS));
      const kk=(d2s[0]*nAk[0]+d2s[1]*nAk[1]);
      const tpk=((Pwk[0]-BS[0])*nAk[0]+(Pwk[1]-BS[1])*nAk[1])/(kk||1e-12);
      const P12k=[BS[0]+tpk*d2s[0],BS[1]+tpk*d2s[1]];
      const e2k=[d2s[0]-2*kk*nAk[0],d2s[1]-2*kk*nAk[1]];
      const rem2k=L2s-tpk;
      const l1k={o:[BS[0]-L1s*u1s[0],BS[1]-L1s*u1s[1]],v:unit([d1s[0]+u1s[0],d1s[1]+u1s[1]])};
      const l2k={o:[P12k[0]-rem2k*u2s[0],P12k[1]-rem2k*u2s[1]],v:unit([e2k[0]+u2s[0],e2k[1]+u2s[1]])};
      const Xk=crossing(l1k,l2k);
      s+=`<circle cx="${S0[0]}" cy="${S0[1]}" r="${1.4+1.4*w}" fill="#0e8f7e" opacity="${0.14+0.4*w}"/>`
        +`<circle cx="${S1k[0]}" cy="${S1k[1]}" r="${1.4+1.4*w}" fill="#0e8f7e" opacity="${0.14+0.4*w}"/>`
        +`<circle cx="${S2k[0]}" cy="${S2k[1]}" r="${1.4+1.4*w}" fill="#0e8f7e" opacity="${0.14+0.4*w}"/>`;
      if(step>=2)s+=`<circle cx="${E1k[0]}" cy="${E1k[1]}" r="${1.6+1.6*w}" fill="#7c4dbe" opacity="${0.12+0.4*w}"/>`
        +`<circle cx="${U1k[0]}" cy="${U1k[1]}" r="${1.6+1.6*w}" fill="#2ca02c" opacity="${0.12+0.4*w}"/>`;
      if(step>=6&&Xk)s+=`<circle cx="${Xk[0]}" cy="${Xk[1]}" r="${2+1.6*w}" fill="none" stroke="#0e8f7e" stroke-width="1.2" opacity="${0.15+0.4*w}"/>`;
    }
  }
  s+=bsMark(BS,"BS",-30,4);
  svg.innerHTML=s;
  const gap=solutionFeasible?dist(U1,Xraw)*M_PER_PX:NaN;
  const tiltA=Math.asin(Math.min(1,Math.abs(nA[0])))*180/Math.PI;
  $("stat32").innerHTML= step<5 ? "" :
    (!entryFeasible?`<span class="off">✗ rejected (P⁽¹⁾, θ): path 2 reaches wall A behind the BS or after exhausting L₂.</span>`:
     step===5?`‖P₂→UE₂‖ = ‖P₂→E₂‖ for each forward P₂; this sub-family still cannot select its own bounce.`:
     !solutionFeasible?`<span class="off">✗ rejected (P⁽¹⁾, θ): the infinite candidate lines cross outside the forward reflected segment. A line crossing is not a physical bounce.</span>`:
    `forward-ray crossing ↔ UE₁: <b>${gap.toFixed(2)} m</b> · all segment lengths positive ✓<br>wall-A hypothesis tilt off reference: ${tiltA.toFixed(1)}° — θ changes the recovered geometry while remaining coupled to wall tilt`);
}
["s32P1","s32P2","s32H","s32L1","s32A1","s32D1","s32L2","s32A2","s32D2","sS32"].forEach(id=>$(id).addEventListener("input",render));
$("b32R").addEventListener("click",()=>{["s32L1","s32A1","s32D1","s32L2","s32A2","s32D2"].forEach(id=>$(id).value=0);render();});
$("b32Prev").addEventListener("click",()=>goStep(step-1));
$("b32Next").addEventListener("click",()=>goStep(step+1));
goStep(1);
})();

/* ---- section 4.4: unknown UE, corridor double — the crossing degenerates ---- */
(function(){
const svg=$("svg34");
if(!svg)return;
const GD=gaussians(233,80);
const CAP=["",
 "① <b>The unknowns.</b> Two measured paths in a corridor: path 1 (single, wall R) and path 2 (double, R→L). Both walls and the UE are unknown.",
 "② <b>Path 1 = §4.1.</b> E⁽¹⁾ = the full AoD walk; P⁽¹⁾ (slider) hypothesizes the bounce; UE₁ sits down the reversed AoA; candidates sweep line 1 ⊥ wall R — in a corridor, straight <em>across</em> it. P⁽¹⁾ fixes the wall-R hypothesis.",
 "③ <b>Strip path 2 at wall R.</b> Its forward AoD ray must cross the hypothesized wall before L₂ is exhausted. A valid hit reflects and the positive leftover walk lands on E₂; a backward or over-budget hit is rejected in red.",
 "④ <b>Line 2 — and the degeneracy.</b> For a valid prefix the candidates sweep a line ⊥ wall L. At the synthetic reference heading (θ = 0) line 2 is parallel to line 1 and, with clean data, <em>coincides</em> with it exactly. Other heading slices survive only when they pass the same ray-order tests.",
 "⑤ <b>Wall L, conditionally.</b> On a surviving slice declare UE₂ = UE₁ and verify that the implied P₂ lies strictly inside the remaining forward segment. Only then does the wall-L hypothesis exist. Bought the wall, not the fix.",
 "⑥ <b>The feasible slide — and illustrative smear.</b> Every surviving (P⁽¹⁾, θ) explanation moves the walls and UE together. A nonzero heading can wedge the lines while they still meet at UE₁; invalid slices are pruned, not extended backward. The illustrative per-bounce incidence-point smear broadens only the feasible candidate family; it is not surface scattering γ<sup>sc</sup>."];
let step=1;
function goStep(n){
  step=Math.max(1,Math.min(6,n));
  $("cap34").innerHTML=CAP[step];
  $("b34Prev").disabled=step===1; $("b34Next").disabled=step===6;
  $("o34Step").textContent=step+" / 6";
  render();
}
function render(){
  const {BS,UE,LX,RX,WY0,WY1,VA1,VA2}=D3;
  const sR=(RX-UE[0])/(VA1[0]-UE[0]); const B1=[RX,UE[1]+sR*(VA1[1]-UE[1])];
  const q2=(LX-UE[0])/(VA2[0]-UE[0]); const Q2t=[LX,UE[1]+q2*(VA2[1]-UE[1])];
  const q1=(RX-Q2t[0])/(VA1[0]-Q2t[0]); const Q1t=[RX,Q2t[1]+q1*(VA1[1]-Q2t[1])];
  const dH=+$("s34H").value;
  $("o34H").textContent=(dH>0?"+":"")+dH.toFixed(1)+"°";
  const p1={L:dist(UE,VA1),u:dirOf(aoaOf(unit(sub(VA1,UE)))+dH),d:unit(sub(B1,BS))};
  const L2=dist(UE,VA2)+(+$("s34L").value)/M_PER_PX;
  const phi2=aoaOf(unit(sub(VA2,UE)))+(+$("s34A").value);
  const psi2=aoaOf(unit(sub(Q1t,BS)))+(+$("s34D").value);
  const p2={L:L2,u:dirOf(phi2+dH),d:dirOf(psi2)};
  const sig=+$("sS34").value;
  $("oS34").textContent=(sig*M_PER_PX).toFixed(1)+" m";
  $("o34L").textContent=(L2*M_PER_PX).toFixed(1)+" m";
  $("o34A").textContent=degFmt(phi2);
  $("o34D").textContent=degFmt(psi2);
  const E1=[BS[0]+p1.L*p1.d[0],BS[1]+p1.L*p1.d[1]];
  const line1={o:[BS[0]-p1.L*p1.u[0],BS[1]-p1.L*p1.u[1]],v:unit([p1.d[0]+p1.u[0],p1.d[1]+p1.u[1]])};
  const t1=(+$("s34P").value)*p1.L;
  const Pw=[BS[0]+t1*p1.d[0],BS[1]+t1*p1.d[1]];
  const U1=[Pw[0]-(p1.L-t1)*p1.u[0],Pw[1]-(p1.L-t1)*p1.u[1]];
  const nA=unit(sub(U1,E1)), wA=[-nA[1],nA[0]];
  $("o34P").textContent=(t1*M_PER_PX).toFixed(1)+" m";
  function strip2(pp){
    const H=hitRayLine(BS,pp.d,Pw,nA);
    if(!H||H.t<=0)return {feasible:false,reason:"path 2 hits wall R behind the BS"};
    const P12=H.p, k=pp.d[0]*nA[0]+pp.d[1]*nA[1];
    const e2=[pp.d[0]-2*k*nA[0],pp.d[1]-2*k*nA[1]];
    const rem2=pp.L-H.t;
    if(!(rem2>0))return {feasible:false,reason:"the first leg exhausts L₂"};
    const E2=[P12[0]+rem2*e2[0],P12[1]+rem2*e2[1]];
    const scale=Math.hypot(e2[0]+pp.u[0],e2[1]+pp.u[1]);
    if(!(scale>1e-9))return {feasible:false,reason:"the final-ray family is degenerate"};
    const line2={o:[P12[0]-rem2*pp.u[0],P12[1]-rem2*pp.u[1]],v:unit([e2[0]+pp.u[0],e2[1]+pp.u[1]])};
    return {P12,e2,rem2,E2,line2,scale,feasible:true,reason:""};
  }
  const S=strip2(p2);
  const nrm=S.feasible?[-S.line2.v[1],S.line2.v[0]]:null;
  const off=S.feasible?Math.abs(nrm[0]*(U1[0]-S.line2.o[0])+nrm[1]*(U1[1]-S.line2.o[1])):NaN;
  const m2=S.feasible?((U1[0]-S.line2.o[0])*S.line2.v[0]+(U1[1]-S.line2.o[1])*S.line2.v[1])/S.scale:NaN;
  const solutionFeasible=S.feasible&&Number.isFinite(m2)&&m2>0&&m2<S.rem2&&off<1e-5;
  const P2s=solutionFeasible?[S.P12[0]+m2*S.e2[0],S.P12[1]+m2*S.e2[1]]:null;
  const nB=solutionFeasible?unit(sub(U1,S.E2)):null, wB=nB?[-nB[1],nB[0]]:null;
  const sinA=S.feasible?Math.abs(line1.v[0]*S.line2.v[1]-line1.v[1]*S.line2.v[0]):NaN;
  let out="";
  out+=seg([LX,WY0],[LX,WY1],"#8a97a3",3,null,0.32)+seg([RX,WY0],[RX,WY1],"#8a97a3",3,null,0.32);
  out+=txt(LX-8,WY0+14,"wall L (reference)","#8a97a3",10,"end")+txt(RX+8,WY0+14,"wall R","#8a97a3",10);
  if(step>=2){
    out+=arrow(BS,E1,"#5d3691",1.3,"6 4",0.65);
    out+=`<rect x="${E1[0]-5}" y="${E1[1]-5}" width="10" height="10" transform="rotate(45 ${E1[0]} ${E1[1]})" fill="#7c4dbe"/>`+txt(E1[0]+9,E1[1]+4,"E⁽¹⁾ (mirrored UE)","#5d3691",10);
    out+=seg([line1.o[0]-900*line1.v[0],line1.o[1]-900*line1.v[1]],[line1.o[0]+900*line1.v[0],line1.o[1]+900*line1.v[1]],"#0e8f7e",1.6,"7 5",0.55);
    out+=arrow(BS,Pw,"#51606e",1.7)+arrow(Pw,U1,"#51606e",1.7);
    out+=`<circle cx="${Pw[0]}" cy="${Pw[1]}" r="4.5" fill="#e8720c"/>`+txt(Pw[0]+8,Pw[1]-8,"P⁽¹⁾","#b45607",10.5);
    out+=`<circle cx="${U1[0]}" cy="${U1[1]}" r="6" fill="#2ca02c" stroke="#fff" stroke-width="1.6"/>`+txt(U1[0]+10,U1[1]+4,"UE₁","#1d7a1d",11)+headingMark(U1,dH);
    out+=seg([Pw[0]-160*wA[0],Pw[1]-160*wA[1]],[Pw[0]+160*wA[0],Pw[1]+160*wA[1]],"#16222e",3.6,null,0.85);
    out+=txt(Pw[0]+160*wA[0]+6,Pw[1]+160*wA[1]+4,"wall R hyp.","#16222e",10);
    out+=txt(line1.o[0]+430*line1.v[0],line1.o[1]+430*line1.v[1]-8,"line 1 ⊥ wall R","#0a6b5e",10);
  }
  if(step>=3&&S.feasible){
    out+=arrow(BS,S.P12,"#7c4dbe",1.3,"5 4",0.6);
    out+=`<circle cx="${S.P12[0]}" cy="${S.P12[1]}" r="4" fill="#7c4dbe"/>`+txt(S.P12[0]+7,S.P12[1]-7,"P₁","#5d3691",10);
    out+=arrow(S.P12,S.E2,"#7c4dbe",1.1,"5 4",0.5);
    out+=`<rect x="${S.E2[0]-4.5}" y="${S.E2[1]-4.5}" width="9" height="9" transform="rotate(45 ${S.E2[0]} ${S.E2[1]})" fill="#7c4dbe" opacity="0.8"/>`+txt(S.E2[0]-9,S.E2[1]-10,"E₂","#5d3691",10,"end");
  }else if(step>=3){
    const q=[BS[0]+150*p2.d[0],BS[1]+150*p2.d[1]];
    out+=arrow(BS,q,"#c22f2f",1.8,"5 4",0.8)+txt(q[0]+8,q[1]-8,`rejected: ${S.reason}`,"#c22f2f",10.5);
  }
  if(step>=4&&S.feasible){
    out+=seg([S.line2.o[0]-900*S.line2.v[0],S.line2.o[1]-900*S.line2.v[1]],[S.line2.o[0]+900*S.line2.v[0],S.line2.o[1]+900*S.line2.v[1]],"#b45607",1.6,"2 6",0.7);
    out+=txt(line1.o[0]+430*line1.v[0],line1.o[1]+430*line1.v[1]+16,Math.abs(dH)<0.05?"line 2 ≡ line 1":"line 2 — wedged |θ|/2 off line 1, still through UE₁","#b45607",10);
  }
  if(step>=5&&solutionFeasible){
    out+=`<circle cx="${U1[0]}" cy="${U1[1]}" r="9" fill="none" stroke="#2ca02c" stroke-width="2" stroke-dasharray="3 3"/>`+txt(U1[0]+12,U1[1]+18,"UE₂ ≔ UE₁","#1d7a1d",10);
    out+=arrow(S.P12,P2s,"#51606e",1.6)+arrow(P2s,U1,"#51606e",1.6);
    out+=`<circle cx="${P2s[0]}" cy="${P2s[1]}" r="4.5" fill="#e8720c"/>`+txt(P2s[0]-8,P2s[1]-8,"P₂","#b45607",10,"end");
    out+=seg([P2s[0]-160*wB[0],P2s[1]-160*wB[1]],[P2s[0]+160*wB[0],P2s[1]+160*wB[1]],"#16222e",3.6,null,0.85);
    out+=txt(P2s[0]-160*wB[0]-6,P2s[1]-160*wB[1]+4,"wall L hyp.","#16222e",10,"end");
  }else if(step>=5&&S.feasible){
    out+=txt(U1[0]+18,U1[1]+38,"rejected: UE₁ lies outside path 2's forward final segment","#c22f2f",10.5);
  }
  if(sig>0&&solutionFeasible){
    for(let k=0;k<24;k++){
      const g1=GD[2*k],g2=GD[2*k+1];
      const w=Math.exp(-0.5*(g1*g1+g2*g2)/2);
      const S1=[RX,Q1t[1]+g1*sig],S2=[LX,Q2t[1]+g2*sig];
      const Ls=dist(BS,S1)+dist(S1,S2)+dist(S2,UE);
      const pp={L:Ls,u:dirOf(aoaOf(unit(sub(S2,UE)))+dH),d:unit(sub(S1,BS))};
      const st=strip2(pp);
      if(!st.feasible)continue;
      const nrm2=[-st.line2.v[1],st.line2.v[0]];
      const dd=nrm2[0]*(U1[0]-st.line2.o[0])+nrm2[1]*(U1[1]-st.line2.o[1]);
      const mk=((U1[0]-st.line2.o[0])*st.line2.v[0]+(U1[1]-st.line2.o[1])*st.line2.v[1])/st.scale;
      if(!(mk>0&&mk<st.rem2&&Math.abs(dd)<1e-5))continue;
      const Uk=[U1[0]-dd*nrm2[0],U1[1]-dd*nrm2[1]];
      out+=`<circle cx="${S1[0]}" cy="${S1[1]}" r="${1.4+1.4*w}" fill="#0e8f7e" opacity="${0.14+0.4*w}"/>`
          +`<circle cx="${S2[0]}" cy="${S2[1]}" r="${1.4+1.4*w}" fill="#0e8f7e" opacity="${0.14+0.4*w}"/>`
          +`<circle cx="${st.E2[0]}" cy="${st.E2[1]}" r="${1.6+1.6*w}" fill="#7c4dbe" opacity="${0.12+0.4*w}"/>`
          +`<circle cx="${Uk[0]}" cy="${Uk[1]}" r="${1.6+1.6*w}" fill="#2ca02c" opacity="${0.12+0.4*w}"/>`;
    }
  }
  out+=bsMark(BS,"BS",-30,4);
  svg.innerHTML=out;
  $("stat34").innerHTML= step<4 ? "" :
    (!S.feasible?`<span class="off">✗ rejected (P⁽¹⁾, θ): ${S.reason}.</span>`:
    !solutionFeasible?`<span class="off">✗ rejected (P⁽¹⁾, θ): the declared UE₂ = UE₁ lies outside path 2's remaining forward segment.</span>`:
    Math.abs(dH)<0.05?
    `∠(line 1, line 2) = <b>${(Math.asin(Math.min(1,sinA))*180/Math.PI).toFixed(2)}°</b> · offset = <b>${(off*M_PER_PX).toFixed(2)} m</b> — coincident for every P⁽¹⁾: no crossing exists; the double bounce adds no UE information in a corridor.`:
    `∠(line 1, line 2) = <b>${(Math.asin(Math.min(1,sinA))*180/Math.PI).toFixed(2)}° = |θ|/2</b> · UE₁ off line 2 = <b>${(off*M_PER_PX).toFixed(2)} m</b> · all segment lengths positive ✓ — θ remains coupled to wall tilt.`);
}
["s34P","s34H","s34L","s34A","s34D","sS34"].forEach(id=>$(id).addEventListener("input",render));
$("b34R").addEventListener("click",()=>{["s34L","s34A","s34D"].forEach(id=>$(id).value=0);render();});
$("b34Prev").addEventListener("click",()=>goStep(step-1));
$("b34Next").addEventListener("click",()=>goStep(step+1));
goStep(1);
})();

/* ---- section 4.3: unknown UE, triple bounce — two rungs of changed foci ---- */
(function(){
const svg=$("svg33");
if(!svg)return;
const GD=gaussians(311,120);
const hitL=(a,b,c,n)=>{const r=sub(b,a);const t=((c[0]-a[0])*n[0]+(c[1]-a[1])*n[1])/((r[0]*n[0]+r[1]*n[1])||1e-12);return [a[0]+t*r[0],a[1]+t*r[1]];};
const refl=(d,n)=>{const k=d[0]*n[0]+d[1]*n[1];return [d[0]-2*k*n[0],d[1]-2*k*n[1]];};
const crossL=(l1,l2)=>{const den=l1.v[0]*l2.v[1]-l1.v[1]*l2.v[0];if(Math.abs(den)<1e-9)return null;
  const w=[l2.o[0]-l1.o[0],l2.o[1]-l1.o[1]];const t=(w[0]*l2.v[1]-w[1]*l2.v[0])/den;
  return [l1.o[0]+t*l1.v[0],l1.o[1]+t*l1.v[1]];};
const fullLine=(l,c,dash,o)=>seg([l.o[0]-1200*l.v[0],l.o[1]-1200*l.v[1]],[l.o[0]+1200*l.v[0],l.o[1]+1200*l.v[1]],c,1.6,dash,o);
const CAP=["",
 "① <b>The unknowns.</b> Three measured paths: path 1 (single, wall A), path 2 (double, A→B), path 3 (triple, A→B→C). Three walls and the UE, all unknown.",
 "② <b>Path 1 = §4.1.</b> E⁽¹⁾ = the full AoD walk; P⁽¹⁾ hypothesizes the bounce (slider); UE₁ sits down the reversed AoA; candidates sweep line 1 ⊥ wall A, and P⁽¹⁾ fixes the wall-A hypothesis.",
 "③ <b>Path 2 = §4.2.</b> Stripped at wall A (P₁, reflect, E₂), its candidates sweep line 2 ⊥ wall B; the crossing with line 1 solves P₂ and the wall-B hypothesis follows.",
 "④ <b>Path 3 — the focus climbs two rungs.</b> Its AoD ray reflects at hypothesized wall A (R₁), again at hypothesized wall B (R₂); the leftover walk lands on E₃, the doubly-stripped mirrored UE.",
 "⑤ <b>Line 3 ⊥ wall C.</b> The candidates sweep the third family; the crossing with line 1 solves the last bounce and the wall-C hypothesis follows — the whole three-wall map from one slider.",
 "⑥ <b>The verdict — after enforcing ray order.</b> Sweep P⁽¹⁾ and θ: all three body-frame AoAs rotate under the same heading candidate, changing every recovered wall and bounce. A member is retained only if P₁, P₂, and P₃ occur in forward order and every leg fits inside L₃. Feasible members preserve the joint ambiguity; red members are algebraic line intersections with negative or reversed path lengths and are not physical triple-bounce solutions. Slide σ only after a feasible member is selected."];
let step=1;
function goStep(n){
  step=Math.max(1,Math.min(6,n));
  $("cap33").innerHTML=CAP[step];
  $("b33Prev").disabled=step===1; $("b33Next").disabled=step===6;
  $("o33Step").textContent=step+" / 6";
  render();
}
function render(){
  const {BS,UE,AY,AX0,AX1,BC,BEND,CC,CEND,BN,CN,VA1,VA2,VA3}=D6;
  const sA1=(AY-UE[1])/(VA1[1]-UE[1]);
  const B1=[UE[0]+sA1*(VA1[0]-UE[0]),AY];
  const Q2t=hitL(UE,VA2,BC,BN);
  const tA2=(AY-Q2t[1])/(VA1[1]-Q2t[1]);
  const Q1t=[Q2t[0]+tA2*(VA1[0]-Q2t[0]),AY];
  const P3t=hitL(UE,VA3,CC,CN);
  const P2t=hitL(P3t,VA2,BC,BN);
  const tA3=(AY-P2t[1])/(VA1[1]-P2t[1]);
  const P1t=[P2t[0]+tA3*(VA1[0]-P2t[0]),AY];
  const dH=+$("s33H").value;
  $("o33H").textContent=(dH>0?"+":"")+dH.toFixed(1)+"°";
  const p1={L:dist(UE,VA1),u:dirOf(aoaOf(unit(sub(VA1,UE)))+dH),d:unit(sub(B1,BS))};
  const p2={L:dist(UE,VA2),u:dirOf(aoaOf(unit(sub(VA2,UE)))+dH),d:unit(sub(Q1t,BS))};
  const L3=dist(UE,VA3)+(+$("s33L").value)/M_PER_PX;
  const phi3=aoaOf(unit(sub(VA3,UE)))+(+$("s33A").value);
  const psi3=aoaOf(unit(sub(P1t,BS)))+(+$("s33D").value);
  const p3={L:L3,u:dirOf(phi3+dH),d:dirOf(psi3)};
  const sig=+$("sS33").value;
  $("oS33").textContent=(sig*M_PER_PX).toFixed(1)+" m";
  $("o33L").textContent=(L3*M_PER_PX).toFixed(1)+" m";
  $("o33A").textContent=degFmt(phi3);
  $("o33D").textContent=degFmt(psi3);
  const E1=[BS[0]+p1.L*p1.d[0],BS[1]+p1.L*p1.d[1]];
  const line1={o:[BS[0]-p1.L*p1.u[0],BS[1]-p1.L*p1.u[1]],v:unit([p1.d[0]+p1.u[0],p1.d[1]+p1.u[1]])};
  const t1=(+$("s33P").value)*p1.L;
  const Pw=[BS[0]+t1*p1.d[0],BS[1]+t1*p1.d[1]];
  const U1=[Pw[0]-(p1.L-t1)*p1.u[0],Pw[1]-(p1.L-t1)*p1.u[1]];
  const nA=unit(sub(U1,E1)), wA=[-nA[1],nA[0]];
  $("o33P").textContent=(t1*M_PER_PX).toFixed(1)+" m";
  const H12=hitRayLine(BS,p2.d,Pw,nA);
  const P12=H12?.p??null, e2=refl(p2.d,nA), rem2=p2.L-(H12?.t??NaN);
  const doubleEntry=!!H12&&H12.t>0&&rem2>0;
  const E2=doubleEntry?[P12[0]+rem2*e2[0],P12[1]+rem2*e2[1]]:null;
  const line2=doubleEntry?{o:[P12[0]-rem2*p2.u[0],P12[1]-rem2*p2.u[1]],v:unit([e2[0]+p2.u[0],e2[1]+p2.u[1]])}:null;
  const X2raw=line2?crossL(line1,line2):null;
  const sc2=Math.hypot(e2[0]+p2.u[0],e2[1]+p2.u[1]);
  const m2=X2raw&&sc2>1e-9?((X2raw[0]-line2.o[0])*line2.v[0]+(X2raw[1]-line2.o[1])*line2.v[1])/sc2:NaN;
  const doubleFeasible=doubleEntry&&!!X2raw&&Number.isFinite(m2)&&m2>0&&m2<rem2;
  const X2=doubleFeasible?X2raw:null;
  let P2s=null,nB=null,wB=null;
  if(doubleFeasible){
    P2s=[P12[0]+m2*e2[0],P12[1]+m2*e2[1]];
    nB=unit(sub(X2,E2)); wB=[-nB[1],nB[0]];
  }
  function strip3(pp){
    if(!doubleFeasible)return {feasible:false,reason:"path 2 is infeasible"};
    const H31=hitRayLine(BS,pp.d,Pw,nA);
    if(!H31||H31.t<=0)return {feasible:false,reason:"bounce 1 lies behind the BS"};
    const R1=H31.p;
    const e31=refl(pp.d,nA);
    const H32=hitRayLine(R1,e31,P2s,nB);
    if(!H32||H32.t<=0)return {feasible:false,reason:"bounce 2 lies behind bounce 1"};
    const R2=H32.p;
    const e32=refl(e31,nB);
    const rem3=pp.L-H31.t-H32.t;
    if(!(rem3>0))return {feasible:false,reason:"the first two legs exhaust L₃"};
    const E3=[R2[0]+rem3*e32[0],R2[1]+rem3*e32[1]];
    const line3={o:[R2[0]-rem3*pp.u[0],R2[1]-rem3*pp.u[1]],v:unit([e32[0]+pp.u[0],e32[1]+pp.u[1]])};
    const X3raw=crossL(line1,line3),scale=Math.hypot(e32[0]+pp.u[0],e32[1]+pp.u[1]);
    const m3=X3raw&&scale>1e-9?((X3raw[0]-line3.o[0])*line3.v[0]+(X3raw[1]-line3.o[1])*line3.v[1])/scale:NaN;
    const feasible=!!X3raw&&Number.isFinite(m3)&&m3>0&&m3<rem3;
    return {R1,R2,E3,e32,scale,line3,X3:feasible?X3raw:null,m3,rem3,feasible,reason:feasible?"":"the line crossing lies outside the forward final segment"};
  }
  const S=strip3(p3);
  let P3s=null,nC=null,wC=null;
  if(S.feasible){
    P3s=[S.R2[0]+S.m3*S.e32[0],S.R2[1]+S.m3*S.e32[1]];
    nC=unit(sub(S.X3,S.E3)); wC=[-nC[1],nC[0]];
  }
  let out="";
  out+=seg([AX0,AY],[AX1,AY],"#8a97a3",3,null,0.3)+seg(BC,BEND,"#8a97a3",3,null,0.3)+seg(CC,CEND,"#8a97a3",3,null,0.3);
  out+=txt(AX0+4,AY-8,"wall A (reference)","#8a97a3",10)+txt(BEND[0]-8,BEND[1]-12,"wall B","#8a97a3",10,"end")+txt(CEND[0]-8,CEND[1]+16,"wall C","#8a97a3",10,"end");
  if(step>=2){
    out+=arrow(BS,E1,"#5d3691",1.3,"6 4",0.6);
    out+=`<rect x="${E1[0]-5}" y="${E1[1]-5}" width="10" height="10" transform="rotate(45 ${E1[0]} ${E1[1]})" fill="#7c4dbe"/>`+txt(E1[0]+9,E1[1]+4,"E⁽¹⁾","#5d3691",10);
    out+=fullLine(line1,"#0e8f7e","7 5",0.5);
    out+=arrow(BS,Pw,"#51606e",1.7)+arrow(Pw,U1,"#51606e",1.7);
    out+=`<circle cx="${Pw[0]}" cy="${Pw[1]}" r="4.5" fill="#e8720c"/>`+txt(Pw[0]+8,Pw[1]-8,"P⁽¹⁾","#b45607",10.5);
    out+=`<circle cx="${U1[0]}" cy="${U1[1]}" r="6" fill="#2ca02c" stroke="#fff" stroke-width="1.6"/>`+txt(U1[0]+10,U1[1]+4,"UE₁","#1d7a1d",11)+headingMark(U1,dH);
    out+=seg([Pw[0]-140*wA[0],Pw[1]-140*wA[1]],[Pw[0]+140*wA[0],Pw[1]+140*wA[1]],"#16222e",3.6,null,0.85);
    out+=txt(Pw[0]-140*wA[0]-6,Pw[1]-140*wA[1]+4,"wall A hyp.","#16222e",10,"end");
  }
  if(step>=3&&doubleFeasible){
    out+=arrow(BS,P12,"#7c4dbe",1.3,"5 4",0.6);
    out+=`<circle cx="${P12[0]}" cy="${P12[1]}" r="4" fill="#7c4dbe"/>`;
    out+=arrow(P12,E2,"#7c4dbe",1.1,"5 4",0.5);
    out+=`<rect x="${E2[0]-4.5}" y="${E2[1]-4.5}" width="9" height="9" transform="rotate(45 ${E2[0]} ${E2[1]})" fill="#7c4dbe" opacity="0.8"/>`+txt(E2[0]+9,E2[1]+4,"E₂","#5d3691",10);
    out+=fullLine(line2,"#0e8f7e","3 5",0.45);
    out+=seg([P2s[0]-110*wB[0],P2s[1]-110*wB[1]],[P2s[0]+110*wB[0],P2s[1]+110*wB[1]],"#16222e",3.6,null,0.85);
    out+=txt(P2s[0]+110*wB[0]+6,P2s[1]+110*wB[1]+4,"wall B hyp.","#16222e",10);
  }else if(step>=3){
    out+=txt(BS[0]+20,BS[1]+42,"path 2 infeasible for selected (P⁽¹⁾, θ)","#c22f2f",11);
  }
  if(step>=4&&S.R1){
    out+=arrow(BS,S.R1,"#b45607",1.2,"4 4",0.6)+arrow(S.R1,S.R2,"#b45607",1.2,"4 4",0.6)+arrow(S.R2,S.E3,"#b45607",1.1,"4 4",0.5);
    out+=`<circle cx="${S.R1[0]}" cy="${S.R1[1]}" r="3.6" fill="#e8720c"/>`+txt(S.R1[0]+7,S.R1[1]-7,"R₁","#b45607",10);
    out+=`<circle cx="${S.R2[0]}" cy="${S.R2[1]}" r="3.6" fill="#e8720c"/>`+txt(S.R2[0]+7,S.R2[1]-7,"R₂","#b45607",10);
    out+=`<rect x="${S.E3[0]-4.5}" y="${S.E3[1]-4.5}" width="9" height="9" transform="rotate(45 ${S.E3[0]} ${S.E3[1]})" fill="#7c4dbe" opacity="0.8"/>`+txt(S.E3[0]+9,S.E3[1]+4,"E₃","#5d3691",10);
  }else if(step>=4){
    out+=txt(BS[0]+20,BS[1]+62,`path 3 infeasible: ${S.reason}`,"#c22f2f",11);
  }
  if(step>=5&&S.feasible){
    out+=fullLine(S.line3,"#0e8f7e","1 5",0.5);
    out+=`<path d="M${S.X3[0]-6} ${S.X3[1]-6}L${S.X3[0]+6} ${S.X3[1]+6}M${S.X3[0]-6} ${S.X3[1]+6}L${S.X3[0]+6} ${S.X3[1]-6}" stroke="#0e8f7e" stroke-width="2.2"/>`;
    if(P3s){
      out+=seg([P3s[0]-100*wC[0],P3s[1]-100*wC[1]],[P3s[0]+100*wC[0],P3s[1]+100*wC[1]],"#16222e",3.6,null,0.85);
      out+=txt(P3s[0]-100*wC[0]-6,P3s[1]-100*wC[1]+14,"wall C hyp.","#16222e",10,"end");
    }
  }
  if(sig>0&&S.feasible){
    const wAd=[1,0], wBd=D6.BD, wCd=D6.CD;
    for(let k=0;k<24;k++){
      const g1=GD[3*k],g2=GD[3*k+1],g3=GD[3*k+2];
      const w=Math.exp(-0.5*(g1*g1+g2*g2+g3*g3)/3);
      const S1=[P1t[0]+g1*sig*wAd[0],P1t[1]+g1*sig*wAd[1]];
      const S2=[P2t[0]+g2*sig*wBd[0],P2t[1]+g2*sig*wBd[1]];
      const S3=[P3t[0]+g3*sig*wCd[0],P3t[1]+g3*sig*wCd[1]];
      const Ls=dist(BS,S1)+dist(S1,S2)+dist(S2,S3)+dist(S3,UE);
      const pp={L:Ls,u:dirOf(aoaOf(unit(sub(S3,UE)))+dH),d:unit(sub(S1,BS))};
      const st=strip3(pp);
      if(!st.feasible)continue;
      out+=`<circle cx="${S1[0]}" cy="${S1[1]}" r="${1.4+1.4*w}" fill="#0e8f7e" opacity="${0.14+0.4*w}"/>`
          +`<circle cx="${S2[0]}" cy="${S2[1]}" r="${1.4+1.4*w}" fill="#0e8f7e" opacity="${0.14+0.4*w}"/>`
          +`<circle cx="${S3[0]}" cy="${S3[1]}" r="${1.4+1.4*w}" fill="#0e8f7e" opacity="${0.14+0.4*w}"/>`
          +`<circle cx="${st.E3[0]}" cy="${st.E3[1]}" r="${1.6+1.6*w}" fill="#7c4dbe" opacity="${0.12+0.4*w}"/>`
          +`<circle cx="${st.X3[0]}" cy="${st.X3[1]}" r="${1.6+1.6*w}" fill="#2ca02c" opacity="${0.12+0.4*w}"/>`;
    }
  }
  out+=bsMark(BS,"BS",-30,4);
  svg.innerHTML=out;
  const g2v=doubleFeasible?dist(X2,U1)*M_PER_PX:NaN, g3v=S.feasible?dist(S.X3,U1)*M_PER_PX:NaN;
  $("stat33").innerHTML= step<3 ? "" :
    (!doubleFeasible?`<span class="off">✗ rejected (P⁽¹⁾, θ): the two-bounce prefix violates forward ray order or its delay budget.</span>`:
     !S.feasible?`path-2 prefix ✓ · <span class="off">path 3 rejected: ${S.reason}.</span>`:
     `ordered path-2 crossing ↔ UE₁: <b>${g2v.toFixed(2)} m</b> · ordered path-3 crossing ↔ UE₁: <b>${g3v.toFixed(2)} m</b> · all segment lengths positive ✓`);
}
["s33P","s33H","s33L","s33A","s33D","sS33"].forEach(id=>$(id).addEventListener("input",render));
$("b33R").addEventListener("click",()=>{["s33L","s33A","s33D"].forEach(id=>$(id).value=0);render();});
$("b33Prev").addEventListener("click",()=>goStep(step-1));
$("b33Next").addEventListener("click",()=>goStep(step+1));
goStep(1);
})();

/* ---- section 4.5: unknown UE, corridor triple — parity seals the slide ---- */
(function(){
const svg=$("svg35");
if(!svg)return;
const GD=gaussians(407,120);
const CAP=["",
 "① <b>The unknowns.</b> Three corridor paths: single (R), double (R→L), triple (R→L→R). Two walls and the UE unknown.",
 "② <b>The ladder so far.</b> Path 1 hypothesizes wall R; path 2 may define wall L only if its forward hit, remaining delay, and declared UE₂ = UE₁ all pass the ordered-segment tests of §4.4.",
 "③ <b>Path 3 — two rungs up.</b> Its AoD ray must reach wall R (R₁), the reflected forward ray must reach wall L (R₂), and a positive leftover walk then lands on E₃. Any failed rung rejects the slice.",
 "④ <b>Line 3 ≡ line 1.</b> The triple's candidates sweep a line ⊥ wall R — at the synthetic reference heading (θ = 0), the <em>same</em> line again, exactly, for every P⁽¹⁾ (statline). That value is not supplied to the estimator. Parity: odd or even, a corridor path's last wall is always parallel to the first.",
 "⑤ <b>The verdict — and illustrative smear.</b> Higher order prunes infeasible (P⁽¹⁾, θ) slices but does not resolve the corridor slide that survives. On each valid slice, line 3 closes at UE₁ and heading stays coupled to recovered-wall tilt; illustrative per-bounce incidence-point smear broadens only those ordered paths and is not surface scattering γ<sup>sc</sup>."];
let step=1;
function goStep(n){
  step=Math.max(1,Math.min(5,n));
  $("cap35").innerHTML=CAP[step];
  $("b35Prev").disabled=step===1; $("b35Next").disabled=step===5;
  $("o35Step").textContent=step+" / 5";
  render();
}
function render(){
  const {BS,UE,LX,RX,WY0,WY1,VA1,VA2,VA3}=D7;
  const sR=(RX-UE[0])/(VA1[0]-UE[0]); const B1=[RX,UE[1]+sR*(VA1[1]-UE[1])];
  const q2=(LX-UE[0])/(VA2[0]-UE[0]); const Q2t=[LX,UE[1]+q2*(VA2[1]-UE[1])];
  const q1=(RX-Q2t[0])/(VA1[0]-Q2t[0]); const Q1t=[RX,Q2t[1]+q1*(VA1[1]-Q2t[1])];
  const t3=(RX-UE[0])/(VA3[0]-UE[0]); const P3t=[RX,UE[1]+t3*(VA3[1]-UE[1])];
  const t2s=(LX-P3t[0])/(VA2[0]-P3t[0]); const P2t=[LX,P3t[1]+t2s*(VA2[1]-P3t[1])];
  const t1s=(RX-P2t[0])/(VA1[0]-P2t[0]); const P1t=[RX,P2t[1]+t1s*(VA1[1]-P2t[1])];
  const dH=+$("s35H").value;
  $("o35H").textContent=(dH>0?"+":"")+dH.toFixed(1)+"°";
  const p1={L:dist(UE,VA1),u:dirOf(aoaOf(unit(sub(VA1,UE)))+dH),d:unit(sub(B1,BS))};
  const p2={L:dist(UE,VA2),u:dirOf(aoaOf(unit(sub(VA2,UE)))+dH),d:unit(sub(Q1t,BS))};
  const L3=dist(UE,VA3)+(+$("s35L").value)/M_PER_PX;
  const phi3=aoaOf(unit(sub(VA3,UE)))+(+$("s35A").value);
  const psi3=aoaOf(unit(sub(P1t,BS)))+(+$("s35D").value);
  const p3={L:L3,u:dirOf(phi3+dH),d:dirOf(psi3)};
  const sig=+$("sS35").value;
  $("oS35").textContent=(sig*M_PER_PX).toFixed(1)+" m";
  $("o35L").textContent=(L3*M_PER_PX).toFixed(1)+" m";
  $("o35A").textContent=degFmt(phi3);
  $("o35D").textContent=degFmt(psi3);
  const E1=[BS[0]+p1.L*p1.d[0],BS[1]+p1.L*p1.d[1]];
  const line1={o:[BS[0]-p1.L*p1.u[0],BS[1]-p1.L*p1.u[1]],v:unit([p1.d[0]+p1.u[0],p1.d[1]+p1.u[1]])};
  const f=+$("s35P").value;
  const t1=f*p1.L;
  const Pw=[BS[0]+t1*p1.d[0],BS[1]+t1*p1.d[1]];
  const U1=[Pw[0]-(p1.L-t1)*p1.u[0],Pw[1]-(p1.L-t1)*p1.u[1]];
  const nA=unit(sub(U1,E1)), wA=[-nA[1],nA[0]];
  $("o35P").textContent=(t1*M_PER_PX).toFixed(1)+" m";
  const H2=hitRayLine(BS,p2.d,Pw,nA);
  const k2=p2.d[0]*nA[0]+p2.d[1]*nA[1];
  const P12=H2?.p??null;
  const e2=[p2.d[0]-2*k2*nA[0],p2.d[1]-2*k2*nA[1]];
  const rem2=p2.L-(H2?.t??NaN);
  const sc2=Math.hypot(e2[0]+p2.u[0],e2[1]+p2.u[1]);
  const doubleEntry=!!H2&&H2.t>0&&rem2>0&&sc2>1e-9;
  const E2=doubleEntry?[P12[0]+rem2*e2[0],P12[1]+rem2*e2[1]]:null;
  const lv2=doubleEntry?{o:[P12[0]-rem2*p2.u[0],P12[1]-rem2*p2.u[1]],v:unit([e2[0]+p2.u[0],e2[1]+p2.u[1]])}:null;
  const nrm2=lv2?[-lv2.v[1],lv2.v[0]]:null;
  const off2=lv2?Math.abs(nrm2[0]*(U1[0]-lv2.o[0])+nrm2[1]*(U1[1]-lv2.o[1])):NaN;
  const m2=lv2?((U1[0]-lv2.o[0])*lv2.v[0]+(U1[1]-lv2.o[1])*lv2.v[1])/sc2:NaN;
  const doubleFeasible=doubleEntry&&Number.isFinite(m2)&&m2>0&&m2<rem2&&off2<1e-5;
  const P2s=doubleFeasible?[P12[0]+m2*e2[0],P12[1]+m2*e2[1]]:null;
  const nB=doubleFeasible?unit(sub(U1,E2)):null, wB=nB?[-nB[1],nB[0]]:null;
  function strip3(pp){
    if(!doubleFeasible)return {feasible:false,reason:"path 2 cannot define a physical wall L"};
    const H31=hitRayLine(BS,pp.d,Pw,nA);
    if(!H31||H31.t<=0)return {feasible:false,reason:"bounce 1 lies behind the BS"};
    const R1=H31.p, k3=pp.d[0]*nA[0]+pp.d[1]*nA[1];
    const e31=[pp.d[0]-2*k3*nA[0],pp.d[1]-2*k3*nA[1]];
    const k31=(e31[0]*nB[0]+e31[1]*nB[1]);
    const H32=hitRayLine(R1,e31,P2s,nB);
    if(!H32||H32.t<=0)return {feasible:false,reason:"bounce 2 lies behind bounce 1"};
    const R2=H32.p;
    const e32=[e31[0]-2*k31*nB[0],e31[1]-2*k31*nB[1]];
    const rem3=pp.L-H31.t-H32.t;
    if(!(rem3>0))return {feasible:false,reason:"the first two legs exhaust L₃"};
    const E3=[R2[0]+rem3*e32[0],R2[1]+rem3*e32[1]];
    const scale=Math.hypot(e32[0]+pp.u[0],e32[1]+pp.u[1]);
    if(!(scale>1e-9))return {feasible:false,reason:"the final-ray family is degenerate"};
    const line3={o:[R2[0]-rem3*pp.u[0],R2[1]-rem3*pp.u[1]],v:unit([e32[0]+pp.u[0],e32[1]+pp.u[1]])};
    const nrm=[-line3.v[1],line3.v[0]];
    const off=Math.abs(nrm[0]*(U1[0]-line3.o[0])+nrm[1]*(U1[1]-line3.o[1]));
    const m3=((U1[0]-line3.o[0])*line3.v[0]+(U1[1]-line3.o[1])*line3.v[1])/scale;
    const feasible=Number.isFinite(m3)&&m3>0&&m3<rem3&&off<1e-5;
    return {R1,R2,E3,line3,e32,rem3,m3,off,feasible,reason:feasible?"":"UE₁ lies outside path 3's forward final segment"};
  }
  const S=strip3(p3);
  const sinA=S.feasible?Math.abs(line1.v[0]*S.line3.v[1]-line1.v[1]*S.line3.v[0]):NaN;
  const off=S.feasible?S.off:NaN;
  let out="";
  out+=seg([LX,WY0],[LX,WY1],"#8a97a3",3,null,0.32)+seg([RX,WY0],[RX,WY1],"#8a97a3",3,null,0.32);
  out+=txt(LX-8,WY0+14,"wall L (reference)","#8a97a3",10,"end")+txt(RX+8,WY0+14,"wall R","#8a97a3",10);
  if(step>=2){
    out+=seg([line1.o[0]-900*line1.v[0],line1.o[1]-900*line1.v[1]],[line1.o[0]+900*line1.v[0],line1.o[1]+900*line1.v[1]],"#0e8f7e",1.6,"7 5",0.55);
    out+=`<rect x="${E1[0]-5}" y="${E1[1]-5}" width="10" height="10" transform="rotate(45 ${E1[0]} ${E1[1]})" fill="#7c4dbe"/>`+txt(E1[0]+9,E1[1]+4,"E⁽¹⁾","#5d3691",10);
    out+=`<circle cx="${Pw[0]}" cy="${Pw[1]}" r="4.5" fill="#e8720c"/>`+txt(Pw[0]+8,Pw[1]-8,"P⁽¹⁾","#b45607",10.5);
    out+=`<circle cx="${U1[0]}" cy="${U1[1]}" r="6" fill="#2ca02c" stroke="#fff" stroke-width="1.6"/>`+txt(U1[0]+10,U1[1]+4,"UE₁","#1d7a1d",11)+headingMark(U1,dH);
    out+=seg([Pw[0]-170*wA[0],Pw[1]-170*wA[1]],[Pw[0]+170*wA[0],Pw[1]+170*wA[1]],"#16222e",3.4,null,0.85);
    out+=txt(Pw[0]+170*wA[0]+6,Pw[1]+170*wA[1]+4,"wall R hyp.","#16222e",10);
    if(doubleFeasible){
      out+=seg([P2s[0]-170*wB[0],P2s[1]-170*wB[1]],[P2s[0]+170*wB[0],P2s[1]+170*wB[1]],"#16222e",3.4,null,0.85);
      out+=txt(P2s[0]-170*wB[0]-6,P2s[1]-170*wB[1]+4,"wall L hyp.","#16222e",10,"end");
    }else{
      out+=txt(U1[0]+18,U1[1]+38,"path 2 rejected: no ordered wall-L solution","#c22f2f",10.5);
    }
    out+=txt(line1.o[0]+430*line1.v[0],line1.o[1]+430*line1.v[1]-8,"line 1","#0a6b5e",10);
  }
  if(step>=3&&S.R1){
    out+=arrow(BS,S.R1,"#b45607",1.2,"4 4",0.6)+arrow(S.R1,S.R2,"#b45607",1.2,"4 4",0.6)+arrow(S.R2,S.E3,"#b45607",1.1,"4 4",0.5);
    out+=`<circle cx="${S.R1[0]}" cy="${S.R1[1]}" r="3.6" fill="#e8720c"/>`+txt(S.R1[0]+7,S.R1[1]-7,"R₁","#b45607",10);
    out+=`<circle cx="${S.R2[0]}" cy="${S.R2[1]}" r="3.6" fill="#e8720c"/>`+txt(S.R2[0]-7,S.R2[1]-7,"R₂","#b45607",10,"end");
    out+=`<rect x="${S.E3[0]-4.5}" y="${S.E3[1]-4.5}" width="9" height="9" transform="rotate(45 ${S.E3[0]} ${S.E3[1]})" fill="#7c4dbe" opacity="0.8"/>`+txt(S.E3[0]+9,S.E3[1]+4,"E₃","#5d3691",10);
  }else if(step>=3){
    out+=txt(BS[0]+20,BS[1]+50,`path 3 rejected: ${S.reason}`,"#c22f2f",10.5);
  }
  if(step>=4&&S.feasible){
    out+=seg([S.line3.o[0]-900*S.line3.v[0],S.line3.o[1]-900*S.line3.v[1]],[S.line3.o[0]+900*S.line3.v[0],S.line3.o[1]+900*S.line3.v[1]],"#b45607",1.6,"2 6",0.7);
    out+=txt(line1.o[0]+430*line1.v[0],line1.o[1]+430*line1.v[1]+16,Math.abs(dH)<0.05?"line 3 ≡ line 1":"line 3 — wedged |θ|/2 off line 1, still through UE₁","#b45607",10);
  }else if(step>=4&&S.R1){
    out+=txt(U1[0]+18,U1[1]+56,`rejected: ${S.reason}`,"#c22f2f",10.5);
  }
  if(sig>0&&S.feasible){
    for(let k=0;k<24;k++){
      const g1=GD[3*k],g2=GD[3*k+1],g3=GD[3*k+2];
      const w=Math.exp(-0.5*(g1*g1+g2*g2+g3*g3)/3);
      const S1=[RX,P1t[1]+g1*sig],S2=[LX,P2t[1]+g2*sig],S3=[RX,P3t[1]+g3*sig];
      const Ls=dist(BS,S1)+dist(S1,S2)+dist(S2,S3)+dist(S3,UE);
      const pp={L:Ls,u:dirOf(aoaOf(unit(sub(S3,UE)))+dH),d:unit(sub(S1,BS))};
      const st=strip3(pp);
      if(!st.feasible)continue;
      const nrm2=[-st.line3.v[1],st.line3.v[0]];
      const dd=nrm2[0]*(U1[0]-st.line3.o[0])+nrm2[1]*(U1[1]-st.line3.o[1]);
      const Uk=[U1[0]-dd*nrm2[0],U1[1]-dd*nrm2[1]];
      out+=`<circle cx="${S1[0]}" cy="${S1[1]}" r="${1.4+1.4*w}" fill="#0e8f7e" opacity="${0.14+0.4*w}"/>`
          +`<circle cx="${S2[0]}" cy="${S2[1]}" r="${1.4+1.4*w}" fill="#0e8f7e" opacity="${0.14+0.4*w}"/>`
          +`<circle cx="${S3[0]}" cy="${S3[1]}" r="${1.4+1.4*w}" fill="#0e8f7e" opacity="${0.14+0.4*w}"/>`
          +`<circle cx="${st.E3[0]}" cy="${st.E3[1]}" r="${1.6+1.6*w}" fill="#7c4dbe" opacity="${0.12+0.4*w}"/>`
          +`<circle cx="${Uk[0]}" cy="${Uk[1]}" r="${1.6+1.6*w}" fill="#2ca02c" opacity="${0.12+0.4*w}"/>`;
    }
  }
  out+=bsMark(BS,"BS",-30,4);
  svg.innerHTML=out;
  $("stat35").innerHTML= step<4 ? "" :
    (!doubleFeasible?`<span class="off">✗ rejected (P⁽¹⁾, θ): path 2 cannot define an ordered wall-L bounce.</span>`:
    !S.feasible?`path-2 prefix ✓ · <span class="off">path 3 rejected: ${S.reason}.</span>`:
    Math.abs(dH)<0.05?
    `∠(line 1, line 3) = <b>${(Math.asin(Math.min(1,sinA))*180/Math.PI).toFixed(2)}°</b> · offset = <b>${(off*M_PER_PX).toFixed(2)} m</b> — coincident for every P⁽¹⁾: parity seals the slide; no corridor order adds UE information.`:
    `∠(line 1, line 3) = <b>${(Math.asin(Math.min(1,sinA))*180/Math.PI).toFixed(2)}° = |θ|/2</b> · UE₁ off line 3 = <b>${(off*M_PER_PX).toFixed(2)} m</b> · all segment lengths positive ✓ — θ remains coupled to wall tilt.`);
}
["s35P","s35H","s35L","s35A","s35D","sS35"].forEach(id=>$(id).addEventListener("input",render));
$("b35R").addEventListener("click",()=>{["s35L","s35A","s35D"].forEach(id=>$(id).value=0);render();});
$("b35Prev").addEventListener("click",()=>goStep(step-1));
$("b35Next").addEventListener("click",()=>goStep(step+1));
goStep(1);
})();

/* =================== SECTION 4.6 special case — globally referenced translations =================== */
(function(){
const svg36=$("svg36"), svg37=$("svg37");
if(!svg36||!svg37)return;
const add=(a,b)=>[a[0]+b[0],a[1]+b[1]];
const scl=(a,s)=>[a[0]*s,a[1]*s];
const dot=(a,b)=>a[0]*b[0]+a[1]*b[1];
const rot9=(v,a)=>[v[0]*Math.cos(a)-v[1]*Math.sin(a),v[0]*Math.sin(a)+v[1]*Math.cos(a)];
const g9=()=>((Math.random()+Math.random()+Math.random())*2-3);   // ~N(0,1)
/* wall = { x : n·x = c } with unit normal n */
const mirror9=(p,n,c)=>sub(p,scl(n,2*(dot(n,p)-c)));

/* ---- scenes: ground truth used ONLY to synthesize data and report errors ---- */
const T9=20*Math.PI/180;
const SCN_J={ BS:[110,330],
  walls:[ {n:[0,1],c:210,seg:[[30,210],[410,210]],lab:"wall A"},
          {n:[Math.cos(T9),-Math.sin(T9)],c:0,seg:[[430,235],[430+Math.sin(T9)*330,235+Math.cos(T9)*330]],lab:"wall B"} ],
  poses:[[175,455],[255,410],[330,428],[398,468]], vis:[0,0,1,1],
  names:["A","B"], railHalf:170 };
SCN_J.walls[1].c=dot(SCN_J.walls[1].n,[430,235]);
const SCN_K={ BS:[160,560],
  walls:[ {n:[1,0],c:330,seg:[[330,95],[330,640]],lab:"wall R"},
          {n:[1,0],c:105,seg:[[105,95],[105,640]],lab:"wall L"} ],
  poses:[[215,470],[258,405],[258,338],[222,268]], vis:[0,0,1,1],
  names:["R","L"], railHalf:150 };

/* clean or jittered single-bounce data at each pose off its visible wall */
function synth9(S,noisy){
  const jL=L=>noisy?L+g9()*3:L, jV=v=>noisy?rot9(v,g9()*0.5*Math.PI/180):v;
  return S.poses.map((p,i)=>{
    const w=S.walls[S.vis[i]], VA=mirror9(S.BS,w.n,w.c);
    const t=(w.c-dot(w.n,p))/dot(w.n,sub(VA,p));
    const P=add(p,scl(sub(VA,p),t));
    return {L:jL(dist(p,VA)), u:jV(unit(sub(VA,p))), d:jV(unit(sub(P,S.BS)))};
  });
}
/* everything the estimator derives WITHOUT ground truth — and WITHOUT UE headings:
   the arrival angles are never used for geometry; wall directions come from motion */
function derive9(S,data){
  const E=data.map(m=>add(S.BS,scl(m.d,m.L)));                      // mirrored UEs (BS side only)
  const o=S.poses.slice(1).map((p,i)=>sub(p,S.poses[i]));           // supplied map-frame displacement vectors
  /* both poses of a side mirror across the same wall, so E_{i+1}-E_i is o reflected
     in the wall direction: the normal is n̂ ∝ (ΔE − o), offset drops out entirely */
  const nH=[0,1].map(side=>{
    const i0=S.vis.indexOf(side);
    return unit(sub(sub(E[i0+1],E[i0]),o[i0]));
  });
  const base=E.map((e,i)=>{const n=nH[S.vis[i]];return sub(e,scl(n,2*dot(n,e)));});
  const ref=[0,1].map(side=>{const i=S.vis.indexOf(side);return {L:data[i].L,d:data[i].d};});
  const cOf=(side,t)=>dot(nH[side],add(S.BS,scl(ref[side].d,t*ref[side].L)));
  const tOf=(side,c)=>(c-dot(nH[side],S.BS))/(ref[side].L*dot(nH[side],ref[side].d));
  const k=S.vis.findIndex((v,i)=>i>0&&v!==S.vis[i-1]);              // the cross-family link
  /* [-2 n̂A | 2 n̂B][cA;cB] = o_{k-1} - (base_k - base_{k-1}) */
  const M=[[-2*nH[0][0],2*nH[1][0]],[-2*nH[0][1],2*nH[1][1]]];
  const rhs=sub(o[k-1],sub(base[k],base[k-1]));
  const G00=M[0][0]*M[0][0]+M[1][0]*M[1][0], G11=M[0][1]*M[0][1]+M[1][1]*M[1][1],
        G01=M[0][0]*M[0][1]+M[1][0]*M[1][1];
  const tr=G00+G11, disc=Math.sqrt(Math.max(0,tr*tr/4-(G00*G11-G01*G01)));
  const l1=tr/2+disc, l2=Math.max(0,tr/2-disc);
  const v1=Math.abs(G01)>1e-9?unit([G01,l1-G00]):(G00>=G11?[1,0]:[0,1]);
  const v2=[-v1[1],v1[0]];
  const b=[M[0][0]*rhs[0]+M[1][0]*rhs[1], M[0][1]*rhs[0]+M[1][1]*rhs[1]];  // Mᵀrhs
  const rankDef=l2<=1e-4;
  let sol=scl(v1,dot(v1,b)/l1);
  if(!rankDef) sol=add(sol,scl(v2,dot(v2,b)/l2));                   // exact LS (full rank)
  /* rank-deficient case: nearest point of the solution LINE {v1·c = v1·b/l1} to a given c */
  const lineProj=c=>add(c,scl(v1,dot(v1,b)/l1-dot(v1,c)));
  return {nH,E,base,cOf,tOf,o,k,s1:Math.sqrt(l1),s2:Math.sqrt(l2),v2,sol,lineProj,rankDef};
}
const pose9=(D,S,i,c)=>add(D.base[i],scl(D.nH[S.vis[i]],2*c[S.vis[i]]));
const gap9=(D,S,c)=>{const q=pose9(D,S,D.k,c),g=add(pose9(D,S,D.k-1,c),D.o[D.k-1]);return sub(q,g);};
function inRes9(D,S,c){                       // within-family residual: blind to the offsets
  let m=0;
  for(let i=1;i<S.poses.length;i++){
    if(S.vis[i]!==S.vis[i-1])continue;
    const d=sub(sub(pose9(D,S,i,c),pose9(D,S,i-1,c)),D.o[i-1]);
    m=Math.max(m,Math.hypot(d[0],d[1]));
  }
  return m;
}
const SIDE_COL=["#e8720c","#7c4dbe"], SIDE_DEEP=["#b45607","#5d3691"];

/* shared painter */
function paint9(S,D,data,c,ck,solved,ghostShift){
  let s="";
  for(const w of S.walls) s+=seg(w.seg[0],w.seg[1],"#8a97a3",3,null,0.32)
    +txt(w.seg[0][0]+6,w.seg[0][1]-8,w.lab+" (truth, reference)","#8a97a3",10);
  S.poses.forEach(p=>{s+=`<circle cx="${p[0]}" cy="${p[1]}" r="7" fill="none" stroke="#2ca02c" stroke-width="1.4" stroke-dasharray="3 3" opacity="0.55"/>`;});
  s+=txt(S.poses[0][0]-12,S.poses[0][1]+24,"true t₁…t₄ (reference)","#1d7a1d",10,"end");
  const q=S.poses.map((_,i)=>pose9(D,S,i,c));
  /* candidate rails: each pose's §4.1 family */
  if(ck.fam) S.poses.forEach((_,i)=>{
    const side=S.vis[i], n=D.nH[side];
    const a=add(D.base[i],scl(n,2*D.cOf(side,0.10))), b2=add(D.base[i],scl(n,2*D.cOf(side,0.92)));
    s+=seg(a,b2,SIDE_COL[side],1.7,i===S.vis.indexOf(side)?null:"5 4",0.5);
    if(i===S.vis.indexOf(side)) s+=txt(b2[0]+7,b2[1]+4,"family "+S.names[side]+" rail (⊥ wall "+S.names[side]+")",SIDE_DEEP[side],10);
  });
  /* mirrored UEs, straight from data */
  if(ck.E){ D.E.forEach(e=>{s+=vaDot(e);});
    s+=txt(D.E[0][0]+9,D.E[0][1]-8,"E₁…E₄ = BS + L·d̂ (mirrored UEs)","#5d3691",10); }
  /* implied walls + per-pose bounce geometry */
  if(ck.geo) [0,1].forEach(side=>{
    const n=D.nH[side], wdir=[-n[1],n[0]], cc=c[side];
    const i0=S.vis.indexOf(side), d0=data[i0].d;
    const P0=add(S.BS,scl(d0,(cc-dot(n,S.BS))/dot(n,d0)));
    s+=seg(sub(P0,scl(wdir,S.railHalf)),add(P0,scl(wdir,S.railHalf)),"#16222e",2.6,null,0.85)
      +txt(P0[0]+10,P0[1]-10,"wall "+S.names[side]+" hypothesis","#16222e",10);
    S.poses.forEach((_,i)=>{ if(S.vis[i]!==side)return;
      const di=data[i].d, Pi=add(S.BS,scl(di,(cc-dot(n,S.BS))/dot(n,di)));
      s+=seg(S.BS,Pi,SIDE_COL[side],1.1,"2 3",0.55)+seg(Pi,q[i],SIDE_COL[side],1.1,"2 3",0.55)
        +`<circle cx="${Pi[0]}" cy="${Pi[1]}" r="3.4" fill="${SIDE_COL[side]}" opacity="0.8"/>`;
    });
  });
  /* globally referenced displacement chain + the cross-family gap */
  const gapv=gap9(D,S,c);
  if(ck.odo){
    for(let i=1;i<S.poses.length;i++) if(S.vis[i]===S.vis[i-1]){
      s+=arrow(q[i-1],q[i],"#2ca02c",1.8,null,0.85)
        +txt((q[i-1][0]+q[i][0])/2+7,(q[i-1][1]+q[i][1])/2-7,"o"+"₁₂₃"[i-1],"#1d7a1d",10);
    }
    const ghost=add(q[D.k-1],D.o[D.k-1]);
    s+=arrow(q[D.k-1],ghost,"#2ca02c",1.8,"6 4",0.85)
      +`<circle cx="${ghost[0]}" cy="${ghost[1]}" r="6.5" fill="none" stroke="#2ca02c" stroke-width="1.8" stroke-dasharray="3 3"/>`
      +txt(ghost[0]+10,ghost[1]-9,"o₂ says t₃ lands here","#1d7a1d",10);
    if(Math.hypot(gapv[0],gapv[1])>1.2) s+=seg(ghost,q[D.k],"#c22f2f",2.6)
      +txt((ghost[0]+q[D.k][0])/2+8,(ghost[1]+q[D.k][1])/2+2,"gap","#c22f2f",11);
  }
  /* ghost solutions along the null direction */
  if(ghostShift&&ck.ghost) for(const sgn of [-1,1]){
    const cg=[c[0]+sgn*ghostShift*D.v2[0], c[1]+sgn*ghostShift*D.v2[1]];
    const qg=S.poses.map((_,i)=>pose9(D,S,i,cg));
    for(let i=1;i<qg.length;i++) s+=seg(qg[i-1],qg[i],"#0e8f7e",1.4,null,0.30);
    qg.forEach(p=>{s+=`<circle cx="${p[0]}" cy="${p[1]}" r="4.5" fill="#0e8f7e" opacity="0.30"/>`;});
    [0,1].forEach(side=>{
      const n=D.nH[side], wdir=[-n[1],n[0]];
      const i0=S.vis.indexOf(side), d0=data[i0].d;
      const P0=add(S.BS,scl(d0,(cg[side]-dot(n,S.BS))/dot(n,d0)));
      s+=seg(sub(P0,scl(wdir,S.railHalf)),add(P0,scl(wdir,S.railHalf)),"#0e8f7e",2,null,0.28);
    });
  }
  /* implied poses */
  q.forEach((p,i)=>{
    s+=`<circle cx="${p[0]}" cy="${p[1]}" r="5.5" fill="#2ca02c" stroke="#fff" stroke-width="1.6"/>`
      +txt(p[0]+9,p[1]+15,"t"+"₁₂₃₄"[i],"#1d7a1d",10.5);
    if(solved) s+=`<circle cx="${p[0]}" cy="${p[1]}" r="11" fill="none" stroke="#0e8f7e" stroke-width="2"/>`;
  });
  s+=bsMark(S.BS,"BS",-32,4);
  return {svg:s,gap:Math.hypot(gapv[0],gapv[1])};
}

/* ---------- figure 1: the corner (svg36) ---------- */
let noisy9=false, solved36=false, guard36=false;
let dataJ=synth9(SCN_J,false), DJ=derive9(SCN_J,dataJ);
const ck36=()=>({fam:$("c36_fam").checked,E:$("c36_E").checked,geo:$("c36_geo").checked,odo:$("c36_odo").checked,ghost:false});
function render36(){
  const c=[DJ.cOf(0,+$("s36A").value), DJ.cOf(1,+$("s36B").value)];
  $("o36A").textContent=(+$("s36A").value*dataJ[SCN_J.vis.indexOf(0)].L*M_PER_PX).toFixed(1)+" m";
  $("o36B").textContent=(+$("s36B").value*dataJ[SCN_J.vis.indexOf(1)].L*M_PER_PX).toFixed(1)+" m";
  const out=paint9(SCN_J,DJ,dataJ,c,ck36(),solved36,0);
  svg36.innerHTML=out.svg;
  const ang=Math.acos(Math.min(1,Math.abs(dot(DJ.nH[0],DJ.nH[1]))))*180/Math.PI;
  const err=solved36?Math.max(...SCN_J.poses.map((p,i)=>dist(p,pose9(DJ,SCN_J,i,c))))*M_PER_PX:null;
  $("stat36").innerHTML=[
    `normals from global displacement (n̂ ∝ ΔE − o) · ∠(n̂A, n̂B) = <b>${ang.toFixed(1)}°</b> · σ(2×2) = [${DJ.s1.toFixed(2)}, ${DJ.s2.toFixed(2)}] — full rank`,
    `within-family parity ‖ΔE‖ = ‖o‖: ${(inRes9(DJ,SCN_J,c)*M_PER_PX).toFixed(2)} m — the angle is bought, the offset untouched`,
    `cross-family gap at o₂: <b class="${out.gap*M_PER_PX<0.05?"ok":"off"}">${(out.gap*M_PER_PX).toFixed(2)} m</b>`,
    solved36?`solved: unique (c_A, c_B) · trajectory error vs truth: <b>${err.toFixed(2)} m</b> <span class="ok">✓</span>`
            :`press solve — least squares on the one informative row`,
    noisy9?"data jittered: σ 0.3 m on L, 0.5° on every angle":"clean specular data"
  ].join("<br>");
}
["s36A","s36B"].forEach(id=>$(id).addEventListener("input",()=>{if(!guard36)solved36=false;render36();}));
["c36_fam","c36_E","c36_geo","c36_odo"].forEach(id=>$(id).addEventListener("change",render36));
$("b36S").addEventListener("click",()=>{
  guard36=true;
  $("s36A").value=Math.max(0.10,Math.min(0.92,DJ.tOf(0,DJ.sol[0])));
  $("s36B").value=Math.max(0.10,Math.min(0.92,DJ.tOf(1,DJ.sol[1])));
  guard36=false; solved36=true; render36();
});
$("b36N").addEventListener("click",()=>{
  noisy9=!noisy9; dataJ=synth9(SCN_J,noisy9); DJ=derive9(SCN_J,dataJ); solved36=false;
  $("b36N").textContent=noisy9?"reset to clean specular data":"jitter the data (σ: 0.3 m, 0.5°)";
  render36();
});

/* ---------- figure 2: the corridor (svg37) ---------- */
let solved37=false, guard37=false, raf9=null;
let dataK=synth9(SCN_K,false), DK=derive9(SCN_K,dataK);
const NULL_SC=55;
const ck37=()=>({fam:$("c37_fam").checked,E:true,geo:true,odo:$("c37_odo").checked,ghost:$("c37_ghost").checked});
function render37(){
  const sN=+$("s37N").value*NULL_SC;
  const c=[DK.cOf(0,+$("s37R").value)+sN*DK.v2[0], DK.cOf(1,+$("s37L").value)+sN*DK.v2[1]];
  $("o37R").textContent=(+$("s37R").value*dataK[SCN_K.vis.indexOf(0)].L*M_PER_PX).toFixed(1)+" m";
  $("o37L").textContent=(+$("s37L").value*dataK[SCN_K.vis.indexOf(1)].L*M_PER_PX).toFixed(1)+" m";
  $("o37N").textContent=(sN*M_PER_PX>=0?"+":"")+(sN*M_PER_PX).toFixed(1)+" m";
  const out=paint9(SCN_K,DK,dataK,c,ck37(),solved37,46);
  svg37.innerHTML=out.svg;
  $("stat37").innerHTML=[
    `normals from global displacement — ∠(n̂R, n̂L) = <b>${(Math.acos(Math.min(1,Math.abs(dot(DK.nH[0],DK.nH[1]))))*180/Math.PI).toFixed(1)}°</b>, parallel · σ(2×2) = [${DK.s1.toFixed(2)}, <b class="off">${DK.s2.toFixed(3)}</b>] — rank deficient`,
    `cross-family gap at o₂: <b class="${out.gap*M_PER_PX<0.05?"ok":"off"}">${(out.gap*M_PER_PX).toFixed(2)} m</b> — invariant as the null slider moves`,
    solved37?`solve returned a <b>line</b>, not a point — snapped to its nearest member · null vector: walls + trajectory translate across the corridor together <span class="ok">— detected, not fixed</span>`
            :`press solve — §4.2's least-squares crossing degenerates into a least-squares line`,
  ].join("<br>");
}
["s37R","s37L","s37N"].forEach(id=>$(id).addEventListener("input",()=>{if(!guard37&&id!=="s37N")solved37=false;render37();}));
["c37_fam","c37_odo","c37_ghost"].forEach(id=>$(id).addEventListener("change",render37));
$("b37S").addEventListener("click",()=>{
  const cs=DK.lineProj([DK.cOf(0,+$("s37R").value), DK.cOf(1,+$("s37L").value)]);
  guard37=true;
  $("s37R").value=Math.max(0.10,Math.min(0.92,DK.tOf(0,cs[0])));
  $("s37L").value=Math.max(0.10,Math.min(0.92,DK.tOf(1,cs[1])));
  $("s37N").value=0;
  guard37=false; solved37=true; render37();
});
$("b37P").addEventListener("click",()=>{
  if(raf9){cancelAnimationFrame(raf9);raf9=null;}
  if(reduced){$("s37N").value=0.6;render37();return;}
  const t0=performance.now(),DUR=3000;
  const tick=now=>{
    const f=Math.min(1,(now-t0)/DUR);
    $("s37N").value=Math.sin(f*2*Math.PI).toFixed(3);
    render37();
    if(f<1)raf9=requestAnimationFrame(tick); else raf9=null;
  };
  raf9=requestAnimationFrame(tick);
});
render36(); render37();
})();

})();
