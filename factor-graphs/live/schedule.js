// Explanatory rendering for the schedule. Arithmetic stays in FactorBP.
window.ScheduleLesson = (() => {
  const R=String.raw;
  const fmt=n=>Number(n.toFixed(4)).toString();
  const tex=n=>n.startsWith('x')?`x_${n.slice(1)}`:`f_${n}`;
  const mu=(from,to)=>R`\mu_{${tex(from)}\to ${tex(to)}}`;
  const stories=[
    '',
    'Leaf factors send their weights. Leaf variables send [1, 1], which changes no product.',
    'The outer branches summarize their information and send it towards the central edge.',
    'The two sides exchange summaries. The opposite arrows contain different sets of factors.',
    'The centre sends information outward, leaving out the message from each recipient.',
    'Every variable has received all its messages. Its normalized product is now exact.'
  ];
  function equation(m,M){
    const input=m.incoming.map(n=>`${mu(n,m.from)}(${tex(window.FactorBP.scopes[m.from]?n:m.from)})`);
    const lhs=`${mu(m.from,m.to)}(${tex(m.variable)})`;
    if(m.from==='C')return M(`${lhs}=${R`\sum_{${m.incoming.map(tex).join(',')}}`}f_C(x_1,x_2,x_3)`)+`<br>`+M(R`\qquad\cdot `+input.join(''));
    if(window.FactorBP.scopes[m.from])return M(`${lhs}=${m.incoming.length?R`\sum_{${m.incoming.map(tex).join(',')}}`:''}${tex(m.from)}(${window.FactorBP.scopes[m.from].map(tex).join(',')})${input.join('')}`);
    return M(`${lhs}=${input.join('')||'1'}`);
  }
  function calculation(m,value,M){
    const BP=window.FactorBP,factor=Boolean(BP.scopes[m.from]),summed=factor?m.incoming:[];
    const headers=[summed.length?M(`(${summed.map(tex).join(',')})`):M(`${tex(m.variable)}`),...(factor?[M(tex(m.from))]:[]),...m.incoming.map(n=>M(`${mu(n,m.from)}`)),M(R`\text{product}`)];
    const rows=m.terms[value].map(row=>{
      const assignment=summed.length?`(${summed.map(v=>row.assignment[v]).join(',')})`:value;
      return `<tr><td>${assignment}</td>${factor?`<td>${fmt(row.factorValue)}</td>`:''}${row.incoming.map(v=>`<td>${fmt(v)}</td>`).join('')}<td class="term">${fmt(row.weight)}</td></tr>`;
    }).join('');
    const operation=factor&&summed.length?'SUM THE ROWS':m.incoming.length?'POINTWISE PRODUCT':factor?'LOCAL FACTOR':'EMPTY PRODUCT = 1';
    return `<table class="message-calculation"><caption>${factor&&summed.length?`Hold ${M(`${tex(m.variable)}=${value}`)} fixed; sum out ${summed.map(n=>M(tex(n))).join(' and ')}.`:`Evaluate the message at ${M(`${tex(m.variable)}=${value}`)}.`}</caption><thead><tr>${headers.map(h=>`<th>${h}</th>`).join('')}</tr></thead><tbody>${rows}</tbody><tfoot><tr><td colspan="${headers.length-1}">${operation}</td><td class="term">${fmt(m.values[value])}</td></tr></tfoot></table>`;
  }
  function reason(m,M){
    const BP=window.FactorBP,side=BP.senderSide(m.from,m.to);
    if(!side.factors.length)return 'No factors lie on this leaf’s side. The identity [1, 1] contributes no preference between the two states.';
    const label=side.factors.map(f=>M(tex(f))).join(', ');
    if(!m.incoming.length)return `Summarizes ${label}. A leaf factor sends its local weights directly.`;
    return `Summarizes ${label}. Excludes the return message ${M(mu(m.to,m.from))}.`;
  }
  function beliefCalculation(b,M){
    const inputRows=b.received.map(m=>`<tr><td>${M(mu(m.from,m.to))}</td><td>${fmt(m.values[0])}</td><td>${fmt(m.values[1])}</td></tr>`).join('');
    const heading=b.complete?'All incoming messages have arrived.':`${b.received.length} of ${b.received.length+b.missing.length} incoming messages have arrived.`;
    const waiting=b.missing.length?`Still missing: ${b.missing.map(m=>M(mu(m.from,m.to))).join(', ')}.`:'Every factor is now represented exactly once.';
    return `<table class="belief-calculation"><caption>${heading}</caption><thead><tr><th>Incoming message</th><th>${M(`${tex(b.variable)}=0`)}</th><th>${M(`${tex(b.variable)}=1`)}</th></tr></thead><tbody>${inputRows||'<tr><td colspan="3">No incoming messages; start from [1, 1].</td></tr>'}</tbody><tfoot><tr><td>MULTIPLY EACH COLUMN</td><td>${fmt(b.raw[0])}</td><td>${fmt(b.raw[1])}</td></tr></tfoot></table><p class="waiting">${waiting}</p>`;
  }
  return {stories,equation,calculation,reason,beliefCalculation,fmt,tex,mu};
})();
