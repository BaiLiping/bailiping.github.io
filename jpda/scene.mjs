// The lab and deck use the same scene renderer and probability calculations.
export const COLORS=['#087f68','#b96815'];
export const PLOT={width:900,height:550,left:45,top:20,scale:45,xMin:-9,xMax:9,yMin:-5.5,yMax:5.5};
export const screen=([x,y])=>[PLOT.left+(x-PLOT.xMin)*PLOT.scale,PLOT.top+(PLOT.yMax-y)*PLOT.scale];
export const world=([x,y])=>[Math.max(-8.5,Math.min(8.5,(x-PLOT.left)/PLOT.scale+PLOT.xMin)),Math.max(-5,Math.min(5,PLOT.yMax-(y-PLOT.top)/PLOT.scale))];
const fixed=x=>Number(x.toFixed(3));
const escape=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('"','&quot;');
function ellipse(mean,cov,color,kind) {
  const [x,y]=screen(mean),a=cov[0][0],b=cov[0][1],d=cov[1][1];
  const root=Math.hypot(a-d,2*b),l1=(a+d+root)/2,l2=Math.max(0,(a+d-root)/2);
  const angle=-Math.atan2(2*b,a-d)*90/Math.PI;
  return `<ellipse cx="${fixed(x)}" cy="${fixed(y)}" rx="${fixed(Math.sqrt(5.991*l1)*PLOT.scale)}" ry="${fixed(Math.sqrt(5.991*l2)*PLOT.scale)}" transform="rotate(${fixed(angle)} ${fixed(x)} ${fixed(y)})" fill="${kind==='innovation'?'none':color}" fill-opacity=".09" stroke="${color}" stroke-width="${kind==='innovation'?1.7:2.2}" ${kind==='innovation'?'stroke-dasharray="6 5"':''}/>`;
}
export function sceneSVG(result,{interactive=false,selectedMeasurement=0,selectedEvent=null,showPDA=false,showContours=true,id='scene'}={}) {
  const p=PLOT,clip=`plot-${id}`,event=result.events.find(e=>e.id===selectedEvent);
  const parts=[`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${p.width} ${p.height}" font-family="Arial, Helvetica, sans-serif" role="${interactive?'group':'img'}" aria-labelledby="${id}-title ${id}-desc"><title id="${id}-title">Two-target JPDA association scene</title><desc id="${id}-desc">Open circles are predictions, diamonds are JPDA updated means, and black dots are measurements. Dashed ellipses are innovation contours, not validation gates. ${interactive?'Focus a measurement and use arrow keys to move it.':''}</desc><defs><clipPath id="${clip}"><rect x="45" y="20" width="810" height="495"/></clipPath></defs><rect width="900" height="550" rx="12" fill="#fff"/>`];
  for(let x=-8;x<=8;x+=2){const [sx]=screen([x,0]);parts.push(`<path d="M${sx} 20V515" stroke="#e8edf1"/><text x="${sx}" y="537" text-anchor="middle" font-size="12" fill="#667a8a">${x}</text>`);}
  for(let y=-4;y<=4;y+=2){const [,sy]=screen([0,y]);parts.push(`<path d="M45 ${sy}H855" stroke="#e8edf1"/><text x="31" y="${sy+4}" text-anchor="end" font-size="12" fill="#667a8a">${y}</text>`);}
  parts.push(`<text x="884" y="537" text-anchor="end" font-size="12" fill="#667a8a">x / m</text><text x="15" y="15" font-size="12" fill="#667a8a">y / m</text><g clip-path="url(#${clip})">`);
  result.tracks.forEach((track,t)=>{if(showContours)parts.push(ellipse(track.mean,result.updates[t].S,COLORS[t],'innovation'));parts.push(ellipse(result.updates[t].mean,result.updates[t].cov,COLORS[t],'posterior'));});
  result.tracks.forEach((track,t)=>{
    const [x,y]=screen(track.mean);
    result.measurements.forEach((z,j)=>{
      const probability=result.beta[t][j+1],chosen=event?.assignment[t]===j+1;
      if(event&&!chosen)return;
      if(!event&&probability<.002)return;
      const [zx,zy]=screen(z);
      parts.push(`<path d="M${x} ${y}L${zx} ${zy}" stroke="${COLORS[t]}" stroke-width="${event?4:1+6*probability}" opacity="${event?.9:.15+.7*probability}" fill="none"/>`);
    });
    parts.push(`<circle cx="${x}" cy="${y}" r="9" fill="white" stroke="${COLORS[t]}" stroke-width="3"/><text x="${x}" y="${y-18}" text-anchor="middle" font-size="17" font-weight="700" fill="${COLORS[t]}">${t?'B':'A'}−</text>`);
    const [ux,uy]=screen(result.updates[t].mean);
    parts.push(`<path d="M${ux} ${uy-9}l9 9-9 9-9-9Z" fill="${COLORS[t]}" stroke="white" stroke-width="1.5"/><text x="${ux}" y="${uy+28}" text-anchor="middle" font-size="15" font-weight="700" fill="${COLORS[t]}">${t?'B':'A'}+</text>`);
    if(showPDA){const [px,py]=screen(result.pdaUpdates[t].mean);parts.push(`<rect x="${px-6}" y="${py-6}" width="12" height="12" fill="white" stroke="${COLORS[t]}" stroke-width="2" stroke-dasharray="3 2"/>`);}
  });
  parts.push('</g>');
  result.measurements.forEach((z,j)=>{
    const [x,y]=screen(z),label=`Measurement z${j+1}, x ${z[0].toFixed(2)} metres, y ${z[1].toFixed(2)} metres`;
    parts.push(`<g ${interactive?`class="measurement" data-measurement="${j}" tabindex="0" role="button" aria-label="${escape(label)}. Arrow keys move by 0.1 metres; Shift plus arrow moves by 0.5 metres."`:''} transform="translate(${x} ${y})"><circle r="19" fill="transparent"/><circle class="selection-ring" r="13" fill="none" stroke="${interactive&&selectedMeasurement===j?'#2766b1':'transparent'}" stroke-width="2"/><circle r="6" fill="#16273e" stroke="white" stroke-width="2"/><text x="13" y="-12" font-size="17" font-weight="700" fill="#16273e">z${j+1}</text></g>`);
  });
  parts.push('</svg>');return parts.join('');
}
