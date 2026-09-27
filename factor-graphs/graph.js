(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.FactorGraph = api;
})(globalThis, function () {
  const nodes = { A: [50, 74], x1: [160, 74], B: [50, 276], x2: [160, 276], C: [286, 175], x3: [410, 175], D: [548, 74], x4: [668, 74], E: [548, 276], x5: [668, 276] };
  const tex = n => n.startsWith('x') ? `x_${n.slice(1)}` : `f_${n}`;
  function segment(from, to, offset = 0) {
    const [ax, ay] = nodes[from], [bx, by] = nodes[to], dx = bx - ax, dy = by - ay, length = Math.hypot(dx, dy), ux = dx / length, uy = dy / length;
    return { x1: ax + ux * 31 - uy * offset, y1: ay + uy * 31 + ux * offset, x2: bx - ux * 37 - uy * offset, y2: by - uy * 37 + ux * offset };
  }
  function svg(model, phase, selected, math = s => `<span class="math-tex math-inline">\\(${s}\\)</span>`, options = {}) {
    const bp = globalThis.FactorBP;
    const focus = options.explain ? model.messages[selected] : null;
    const belief = options.belief ? bp.beliefAt(model, options.belief, phase) : null;
    const side = focus && !belief ? bp.senderSide(focus.from, focus.to) : null;
    const sourceNodes = new Set(side?.nodes || belief?.factors || []);
    const inputKeys = new Set(belief ? belief.received.map(m=>bp.key(m.from,m.to)) : focus?.incoming.map(n=>bp.key(n,focus.from)) || []);
    const base = bp.edges.map(([f, x]) => `<line x1="${nodes[f][0]}" y1="${nodes[f][1]}" x2="${nodes[x][0]}" y2="${nodes[x][1]}" stroke="${sourceNodes.has(f)&&sourceNodes.has(x)?'#b0d6d0':'#d5dee5'}" stroke-width="${sourceNodes.has(f)&&sourceNodes.has(x)?5:2}"/>`).join('');
    let valueLabels='';
    const arrows = Object.values(model.messages).filter(m => m.phase <= phase).map(m => {
      const k = bp.key(m.from, m.to), active = m.phase === phase, chosen = k === selected && !belief, input=inputKeys.has(k);
      const important=chosen||input, faint=options.explain&&!important;
      const p = segment(m.from, m.to, 7), marker=chosen?'picked':input?'input':active?'active':'done';
      if(options.explain&&important){
        const edge=[m.from,m.to].sort().join('-');
        const labelPositions={'A-x1':[105,28],'B-x2':[105,232],'C-x1':[211,119],'C-x2':[196,231],'C-x3':[350,128],'D-x3':[507,151],'D-x4':[608,28],'E-x3':[507,224],'E-x5':[608,232]};
        const [lx,ly]=labelPositions[edge];
        const values=m.values.map(v=>Number(v.toFixed(3))).join(', ');
        valueLabels+=`<g pointer-events="none"><rect x="${lx-62}" y="${ly-15}" width="124" height="30" rx="5" fill="${chosen?'#fff3e8':'#e8f4f0'}" stroke="${chosen?'#e5a879':'#8ebcb3'}"/><text x="${lx}" y="${ly+6}" text-anchor="middle" font-family="Arial,sans-serif" font-size="20" fill="${chosen?'#aa4812':'#17695f'}">[${values}]</text></g>`;
      }
      return `<g class="message-edge ${active ? 'active' : ''} ${input?'input-edge':''}" data-message="${k}" role="button" tabindex="0" aria-label="Inspect ${m.from} to ${m.to}, step ${m.phase}"><title>${m.from} → ${m.to} · step ${m.phase}</title><line x1="${p.x1}" y1="${p.y1}" x2="${p.x2}" y2="${p.y2}" stroke="transparent" stroke-width="19"/><line x1="${p.x1}" y1="${p.y1}" x2="${p.x2}" y2="${p.y2}" opacity="${faint ? 0.25 : 1}" stroke="${chosen ? '#ca6218' : input ? '#1e766d' : active ? '#126caa' : '#719393'}" stroke-width="${important?4:active?2.5:1.8}" marker-end="url(#arrow-${marker})"/></g>`;
    }).join('');
    const labels = Object.entries(nodes).map(([n, [x, y]]) => {
      const variable = n.startsWith('x');
      const hit=options.explain&&variable?`class="variable-node" data-variable="${n}" role="button" tabindex="0" aria-label="Inspect belief at ${n}"`:'';
      return `<g ${hit}>${sourceNodes.has(n)?`<circle cx="${x}" cy="${y}" r="34" fill="#e2f2ed"/>`:''}${variable ? `<circle cx="${x}" cy="${y}" r="25" fill="${n===options.belief?'#e8f2f9':'white'}" stroke="${n===options.belief?'#126caa':'#203446'}" stroke-width="${n===options.belief?4:2}"/>` : `<rect x="${x - 17}" y="${y - 17}" width="34" height="34" rx="3" fill="${sourceNodes.has(n)?'#1e766d':'#203446'}"/>`}<foreignObject pointer-events="none" x="${x - 30}" y="${variable ? y - 15 : y + 22}" width="60" height="34"><div xmlns="http://www.w3.org/1999/xhtml" class="graph-label">${math(tex(n))}</div></foreignObject></g>`;
    }).join('');
    return `<svg viewBox="0 0 720 340" role="${options.explain?'group':'img'}" aria-label="The paper's five-variable tree. ${options.explain?'Green arrows supply the selected calculation; orange is the outgoing message. Click a variable to inspect its belief.':'Blue arrows are messages in the current step; orange is the inspected message.'}"><defs>${[['active','#126caa'],['done','#719393'],['picked','#ca6218'],['input','#1e766d']].map(([id,c]) => `<marker id="arrow-${id}" markerWidth="7" markerHeight="7" refX="5.5" refY="3.5" orient="auto"><path d="M0 0 L7 3.5 L0 7Z" fill="${c}"/></marker>`).join('')}</defs>${base}${arrows}${labels}${valueLabels}</svg>`;
  }
  return { nodes, tex, segment, svg };
});
