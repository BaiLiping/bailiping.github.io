(function(root,factory){const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;else root.TrackingGraph=api;})(typeof globalThis==='object'?globalThis:this,function(){
  const R=String.raw,nodes={},edges=[];
  for(let i=0;i<2;i++){
    const y=70+i*155;
    for(const [prefix,x,kind,label] of [['prev',30,'variable',R`y_-^{${i+1}}`],['f',100,'transition',R`f^{${i+1}}`],['y',177,'variable',R`\underline y^{${i+1}}`],['q',252,'factor',R`q^{${i+1}}`],['a',328,'variable',R`a^{${i+1}}`],['b',570,'variable',R`b^{${i+1}}`],['v',643,'factor',R`v^{${i+1}}`],['new',723,'variable',R`\overline y^{${i+1}}`]])nodes[prefix+i]={x,y,kind,label,...(['prev','f','y','q','a'].includes(prefix)?{j:i}:{m:i})};
    for(const [from,to] of [[`prev${i}`,`f${i}`],[`f${i}`,`y${i}`],[`y${i}`,`q${i}`],[`q${i}`,`a${i}`],[`b${i}`,`v${i}`],[`v${i}`,`new${i}`]])edges.push([from,to]);
  }
  const positions=[[[449,60],[421,127]],[[477,169],[449,235]]];
  for(let j=0;j<2;j++)for(let m=0;m<2;m++){
    const [x,y]=positions[j][m],id=`psi${j}${m}`;nodes[id]={x,y,kind:'constraint',label:R`\Psi^{${j+1},${m+1}}`,j,m};edges.push([`a${j}`,id],[id,`b${m}`]);
  }
  function paths(step,j,m){
    const id=typeof step==='string'?step:step.id;
    if(id==='prior')return [[`prev${j}`,`f${j}`]];
    if(id==='predict')return [[`f${j}`,`y${j}`]];
    if(id==='copy')return [[`y${j}`,`q${j}`]];
    if(id==='beta')return [[`q${j}`,`a${j}`]];
    if(id==='xi')return [[`v${m}`,`b${m}`]];
    if(['initial','phi'].includes(id))return [[`a${j}`,`psi${j}${m}`],[`psi${j}${m}`,`b${m}`]];
    if(id==='nu')return [[`b${m}`,`psi${j}${m}`],[`psi${j}${m}`,`a${j}`]];
    if(id==='kappa')return [[`a${j}`,`q${j}`]];
    if(id==='iota')return [[`b${m}`,`v${m}`]];
    if(id==='gamma')return [[`q${j}`,`y${j}`]];
    if(id==='zeta')return [[`v${m}`,`new${m}`]];
    return [];
  }
  function inputs(step,j,m,kind='legacy'){
    const id=typeof step==='string'?step:step.id;
    if(id==='predict')return [[`prev${j}`,`f${j}`]];
    if(id==='copy')return [[`f${j}`,`y${j}`]];
    if(id==='belief')return kind==='legacy'?[[`f${j}`,`y${j}`],[`q${j}`,`y${j}`]]:[[`v${m}`,`new${m}`]];
    if(id==='beta')return [[`y${j}`,`q${j}`]];
    if(id==='initial')return [[`q${j}`,`a${j}`]];
    if(id==='phi')return [[`q${j}`,`a${j}`],[`psi${j}${1-m}`,`a${j}`]];
    if(id==='nu')return [[`v${m}`,`b${m}`],[`psi${1-j}${m}`,`b${m}`]];
    if(id==='kappa')return [0,1].map(n=>[`psi${j}${n}`,`a${j}`]);
    if(id==='iota')return [0,1].map(i=>[`psi${i}${m}`,`b${m}`]);
    if(id==='gamma')return [[`a${j}`,`q${j}`]];
    if(id==='zeta')return [[`b${m}`,`v${m}`]];
    return [];
  }
  function segment(from,to){const a=nodes[from],b=nodes[to],dx=b.x-a.x,dy=b.y-a.y,len=Math.hypot(dx,dy),pad=n=>n.kind==='variable'?18:n.kind==='constraint'?8:15;return {x1:a.x+dx/len*pad(a),y1:a.y+dy/len*pad(a),x2:b.x-dx/len*(pad(b)+3),y2:b.y-dy/len*(pad(b)+3)};}
  function svg({step={id:'none'},j=0,m=0,kind='legacy',interactive=true}={},math=s=>`<span class="math-tex math-inline">\\(${s}\\)</span>`){
    const active=paths(step,j,m),incoming=inputs(step,j,m,kind),seen=new Set(active.flat());
    if(step.id==='belief')seen.add(kind==='legacy'?`y${j}`:`new${m}`);
    let s='<svg viewBox="0 0 760 280" role="group" aria-label="Figure 4, two legacy targets and two measurements"><defs><marker id="out-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#c16023"/></marker><marker id="in-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#168078"/></marker></defs>';
    s+='<path d="M 64 33 V 261" stroke="#cad6dd" stroke-dasharray="5 5"/><text x="9" y="19" class="time-label">Previous</text><text x="98" y="19" class="time-label">Current scan · legacy targets</text><text x="526" y="19" class="time-label">Measurements → possible new targets</text>';
    for(const [a,b] of edges)s+=`<line x1="${nodes[a].x}" y1="${nodes[a].y}" x2="${nodes[b].x}" y2="${nodes[b].y}" stroke="#d6e0e7" stroke-width="2"/>`;
    for(const [list,color,marker,cls] of [[incoming,'#168078','in-arrow','input-path'],[active,'#c16023','out-arrow','active-path']])for(const [a,b] of list){const p=segment(a,b);s+=`<line class="${cls}" data-from="${a}" data-to="${b}" x1="${p.x1}" y1="${p.y1}" x2="${p.x2}" y2="${p.y2}" stroke="${color}" stroke-width="3.3" marker-end="url(#${marker})"/>`;}
    for(const [id,n] of Object.entries(nodes)){
      const circle=n.kind==='variable',size=n.kind==='constraint'?12:28,activeNode=seen.has(id),stroke=activeNode?'#c16023':'#405769',fill=n.kind==='transition'?'#cae2ce':circle?'#fff':'#f4d3d2';
      s+=`<g class="graph-node" ${interactive?`role="button" tabindex="0" aria-label="Inspect ${id}" data-node="${id}"`:''} ${n.j!==undefined?`data-j="${n.j}"`:''} ${n.m!==undefined?`data-m="${n.m}"`:''}><title>${n.j!==undefined?'Legacy '+(n.j+1):''}${n.m!==undefined?' measurement '+(n.m+1):''}</title>`;
      if(activeNode)s+=`<circle cx="${n.x}" cy="${n.y}" r="24" fill="#fff1e6"/>`;
      s+=circle?`<circle cx="${n.x}" cy="${n.y}" r="18" fill="${fill}" stroke="${stroke}" stroke-width="${activeNode?2:1.4}"/>`:`<rect x="${n.x-size/2}" y="${n.y-size/2}" width="${size}" height="${size}" rx="3" fill="${fill}" stroke="${stroke}" stroke-width="${activeNode?2:1.2}"/>`;
      const labelY=n.kind==='constraint'?n.y-34:n.y-12;
      s+=`<foreignObject x="${n.x-37}" y="${labelY}" width="74" height="31"><div xmlns="http://www.w3.org/1999/xhtml" class="node-label ${n.kind==='constraint'?'constraint-label':''}">${math(n.label)}</div></foreignObject></g>`;
    }
    if(active.length){
      const [a,b]=active.at(-1),p=segment(a,b),symbol=(step.symbol||(step.id==='prior'?R`\widetilde f_-^j`:'')).replaceAll('_j','_'+(j+1)).replaceAll('_m','_'+(m+1)).replaceAll('^j','^'+(j+1)).replaceAll('{j,m}',`{${j+1},${m+1}}`).replaceAll('{m,j}',`{${m+1},${j+1}}`),cx=(p.x1+p.x2)/2,cy=(p.y1+p.y2)/2;
      if(symbol)s+=`<foreignObject x="${cx-40}" y="${cy-32}" width="80" height="30"><div xmlns="http://www.w3.org/1999/xhtml" class="message-label">${math(symbol)}</div></foreignObject>`;
    }
    return s+'</svg>';
  }
  return {nodes,edges,paths,inputs,segment,svg};
});
