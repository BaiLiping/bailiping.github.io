// Rendering only; the shared solver and controller own all numerical state.
(function(root){
'use strict';
const colors=['#087f68','#2766b1','#7854a3'],ink='#16273e',muted='#596d80',line='#d8e1e9';
const percent=x=>(100*x).toFixed(1)+'%';
function table(A,{weights=false,reference=null}={}){
 const labels=['Track','Miss',...A[0].slice(1).map((_,j)=>'z'+(j+1))];
 return '<table><thead><tr>'+labels.map(s=>`<th>${s}</th>`).join('')+'</tr></thead><tbody>'+A.map((row,i)=>'<tr><td>T'+(i+1)+'</td>'+row.map((v,j)=>`<td style="background:rgba(8,127,104,${.04+.21*(weights?Math.min(v/2,1):v)})">${weights?(v===0?'0':v.toFixed(2)):percent(v)}${reference?`<small> / ${percent(reference[i][j])}</small>`:''}</td>`).join('')+'</tr>').join('')+'</tbody></table>';
}
function scene(S,gated,GATE){let s='';
 for(let x=0;x<=720;x+=60)s+=`<path d="M${x} 0V350" stroke="#e6edf3"/>`;
 for(let y=0;y<=350;y+=50)s+=`<path d="M0 ${y}H720" stroke="#e6edf3"/>`;
 S.T.forEach((t,i)=>{const a=t.S[0][0],b=t.S[0][1],c=t.S[1][1],mid=(a+c)/2,d=Math.hypot((a-c)/2,b),rx=Math.sqrt(GATE*(mid+d)),ry=Math.sqrt(GATE*(mid-d)),angle=90*Math.atan2(2*b,a-c)/Math.PI;
 if(gated)s+=`<ellipse cx="${t.x}" cy="${t.y}" rx="${rx}" ry="${ry}" transform="rotate(${angle} ${t.x} ${t.y})" fill="${colors[i]}" fill-opacity=".055" stroke="${colors[i]}" stroke-width="1.5" stroke-dasharray="6 5"/>`;
 s+=`<g class="draggable"><circle cx="${t.x}" cy="${t.y}" r="20" fill="transparent"/><circle cx="${t.x}" cy="${t.y}" r="7" fill="${colors[i]}"/><text x="${t.x-25}" y="${t.y-15}" fill="${colors[i]}" font-size="17" font-weight="700">T${i+1}</text></g>`;});
 S.Z.forEach((z,j)=>{const [dx,dy]=[[20,2],[-30,29],[-30,-12],[12,6]][j]||[12,6];s+=`<g class="draggable"><circle cx="${z.x}" cy="${z.y}" r="18" fill="transparent"/><path d="M${z.x-6} ${z.y-6}l12 12m-12 0l12-12" stroke="${ink}" stroke-width="2.5"/><text x="${z.x+dx}" y="${z.y+dy}" font-size="15" fill="${muted}" stroke="#f2f6fa" stroke-width="3" paint-order="stroke">z${j+1}</text></g>`;});return s;
}
function graph(L,h,selected){let s='';const ty=i=>38+i*64,my=j=>22+j*51;
 for(let i=0;i<L.length;i++)for(let j=0;j<L[0].length-1;j++)if(L[i][j+1]>0){const chosen=selected[0]===i&&selected[1]===j;s+=`<line x1="116" y1="${ty(i)}" x2="589" y2="${my(j)}" stroke="${chosen?colors[i]:line}" stroke-width="${chosen?4:2}"/>`;if(chosen){const f=h.kind==='mu'?.62:.38,x=116+473*f,y=ty(i)+(my(j)-ty(i))*f;s+=`<circle cx="${x}" cy="${y}" r="6" fill="${colors[i]}"/><text x="350" y="190" text-anchor="middle" fill="${colors[i]}" font-size="15">Selected: T${i+1} ↔ z${j+1}</text>`;}}
 for(let i=0;i<L.length;i++)s+=`<circle cx="96" cy="${ty(i)}" r="20" fill="#fff" stroke="${colors[i]}" stroke-width="2"/><text x="96" y="${ty(i)+5}" text-anchor="middle" font-size="16" fill="${colors[i]}">T${i+1}</text>`;
 for(let j=0;j<L[0].length-1;j++)s+=`<circle cx="609" cy="${my(j)}" r="20" fill="#fff" stroke="${muted}"/><text x="609" y="${my(j)+5}" text-anchor="middle" font-size="16" fill="${ink}">z${j+1}</text>`;
 return s;
}
function bars(events,k){const max=events[0].p,dx=650/events.length;let s=`<path d="M40 120H700" stroke="${line}"/><text x="12" y="17" fill="${muted}" font-size="12">${percent(max)}</text>`;events.forEach((e,i)=>{const h=96*e.p/max,x=42+i*dx;s+=`<rect x="${x}" y="${120-h}" width="${Math.max(1,dx-3)}" height="${h}" rx="2" fill="${i<k?'#087f68':'#d8e1e9'}"><title>Rank ${i+1}: ${e.a.map((j,t)=>'T'+(t+1)+' to '+(j<0?'miss':'z'+(j+1))).join(', ')}; ${percent(e.p)}</title></rect>`;if(i===0||i===events.length-1||(events.length<35&&(i+1)%5===0))s+=`<text x="${x+dx/2}" y="141" text-anchor="middle" fill="${muted}" font-size="12">${i+1}</text>`;});return s;}
root.AssociationRenderer={table,scene,graph,bars,percent};
})(window);
