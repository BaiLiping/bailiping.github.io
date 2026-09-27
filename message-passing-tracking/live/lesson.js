window.TrackingLesson=(()=>{
 const R=String.raw,fmt=n=>Math.abs(n)<1e-10?'0':Number(n.toFixed(4)).toString(),Y=[R`\varnothing`,R`L`,R`R`];
 const groupNames=['Predict','Evaluate','Associate','Return','Beliefs'];
 function table(headers,rows,footer){return `<table class="arithmetic"><thead><tr>${headers.map(x=>`<th>${x}</th>`).join('')}</tr></thead><tbody>${rows.map(row=>`<tr>${row.map((x,i)=>`<td${i===row.length-1?' class="term"':''}>${x}</td>`).join('')}</tr>`).join('')}</tbody>${footer?`<tfoot><tr><td colspan="${headers.length-1}">${footer[0]}</td><td class="term">${footer[1]}</td></tr></tfoot>`:''}</table>`;}
 function explain(r,step,{j,m,entry,kind},T){
  const M=window.TrackingBP,id=step.id,l=step.iteration||0,otherM=1-m,otherJ=1-j;
  const A=R`a^{${j+1}}`,B=R`b^{${m+1}}`,Q=R`q^{${j+1}}`,P=R`\Psi^{${j+1},${m+1}}`,alpha=R`\alpha_${j+1}`,beta=R`\beta_${j+1}`,xi=R`\xi_${m+1}`,phi=R`\varphi^{[${l}]}_{${j+1},${m+1}}`,nu=R`\nu^{[${l}]}_{${m+1},${j+1}}`,kappa=R`\kappa_${j+1}`,iota=R`\iota_${m+1}`,gamma=R`\gamma_${j+1}`,zeta=R`\varsigma_${m+1}`;
  let title,description,equation,rows,headers,vector,labels=Y,footer,note,source,domain='State',output;
  if(id==='prior'){
   title=`Previous track ${j+1} → transition factor`;
   description='The previous posterior is the only information sent across time. Its three entries include nonexistence.';
   equation=R`\widetilde f_-^{${j+1}}(y_-)=p(y_-\mid z_{1:k-1})`;
   vector=M.priors[j];headers=['Previous state','Probability'];rows=vector.map((v,y)=>[T(Y[y]),fmt(v)]);note='State order: absent, left cell, right cell. All numbers in this demo are chosen teaching data.';source='Fig. 4; before Eq. (76)';
  }else if(id==='predict'){
   title=`Transition factor → legacy state ${j+1}`;
   description='For each destination state, sum over all previous states. Survival can move probability into the absent state.';
   equation=R`${alpha}(y)=\sum_{y_-}f^{${j+1}}(y\mid y_-)\widetilde f_-^{${j+1}}(y_-)`;
   headers=['Old state','Prior','Transition','Product'];rows=M.priors[j].map((v,y)=>[T(Y[y]),fmt(v),fmt(M.transition[y][entry]),fmt(v*M.transition[y][entry])]);vector=r.alpha[j];footer=['SUM',fmt(vector[entry])];note='Survival = 0.95. A surviving target stays in its cell with probability 0.85; otherwise it switches cells.';source='Eqs. (76)–(77)';
  }else if(id==='copy'){
   title=`Legacy state ${j+1} → local factor`;
   description='A variable multiplies its other incoming messages. Here there is only the prediction, so it passes through unchanged.';
   equation=R`\mu_{\underline y^{${j+1}}\to ${Q}}(y)=${alpha}(y)`;
   vector=r.alpha[j];headers=['State','Incoming prediction','Outgoing'];rows=vector.map((v,y)=>[T(Y[y]),fmt(v),fmt(v)]);note='Exclude the return message γ from q. Feeding γ back here would reuse the current measurement information.';source='Fig. 4; SPA variable rule';
  }else if(id==='beta'){
   title=`Local factor → target association ${j+1}`;
   description='Hold one association choice fixed and sum out state and existence. This creates a weight for each possible measurement.';
   equation=R`${beta}(a)=\sum_y ${Q}(y,a)${alpha}(y)`;
   labels=['0','1','2'];domain='Association';vector=r.beta[j];headers=['State',T(alpha),T(Q),'Product'];rows=r.alpha[j].map((w,y)=>[T(Y[y]),fmt(w),fmt(r.q[j][y][entry]),fmt(w*r.q[j][y][entry])]);footer=['SUM',fmt(vector[entry])];note='a = 0 means missed detection or nonexistence. a = 1 or 2 names the measurement. These weights are not probabilities.';source='Eq. (78)';
  }else if(id==='xi'){
   title=`New-target factor → measurement association ${m+1}`;
   description='Sum over the possible new target. No prior message enters this leaf; its identity message is 1.';
   equation=R`${xi}(b)=\sum_{\bar y}v^{${m+1}}(\bar y,b)`;
   labels=['0','1','2'];domain='Association';vector=r.xi[m];headers=['New state','Local factor','Contribution'];rows=r.v[m].map((row,y)=>[T(Y[y]),fmt(row[entry]),fmt(row[entry])]);footer=['SUM',fmt(vector[entry])];note='b = 0 allows clutter or a new target. For b = 1 or 2, a legacy target owns this measurement, so the new target must be absent.';source='Eqs. (73)–(74), (79)';
  }else if(['initial','phi','nu'].includes(id)){
   const toB=id!=='nu',round=id==='initial'?null:r.da.rounds[l-1],base=toB?r.beta[j]:r.xi[m];
   const other=id==='initial'?[1,1,1]:toB?round.nu[j][otherM]:round.previousPhi[otherJ][m];
   const detail=id==='initial'?r.da.initial[j][m]:toB?round.phiDetail[j][m]:round.nuDetail[j][m];
   const local=toB?beta:xi,otherLabel=id==='initial'?'1':toB?R`\nu^{[${l}]}_{${otherM+1},${j+1}}`:R`\varphi^{[${l-1}]}_{${otherJ+1},${m+1}}`;
   title=toB?`Target ${j+1} → consistency factor → measurement ${m+1}`:`Measurement ${m+1} → consistency factor → target ${j+1}`;
   description=id==='initial'?'Initialize with the local target weights (Eq. 29). The constraint sums all compatible assignments.':'First multiply the other inputs at the variable. Then the constraint sums only assignments consistent with the chosen output.';
   equation=toB?R`${phi}(b)\propto\sum_a\Psi(a,b)\,${beta}(a)`+(id==='initial'?'':R`\,${otherLabel}(a)`):R`${nu}(a)\propto\sum_b\Psi(a,b)\,${xi}(b)\,${otherLabel}(b)`;
   headers=[T(toB?'a':'b'),T(local),T(otherLabel),T(R`\Psi`),'Product'];
   rows=base.map((w,s)=>{const valid=toB?M.psi(j,m,s,entry):M.psi(j,m,entry,s);return [s,fmt(w),fmt(other[s]),valid,fmt(w*other[s]*valid)];});
   labels=['0','1','2'];domain=toB?'Output b':'Output a';vector=detail.values;footer=['RAW SUM',fmt(detail.raw[entry])];
   note=`Shown vector = raw vector ÷ ${fmt(detail.scale)}. The nonmatching entries equal 1. `+(id==='initial'?'This rescaling preserves BP beliefs.':`Exclude the input from Ψ${j+1},${m+1}; use only the other ${toB?'measurement':'target'}’s message.`);
   source=id==='initial'?'Eq. (29), vector initialization':toB?'Eq. (27); normalized as in §VI-B':'Eq. (28); normalized as in §VI-B';
  }else if(id==='kappa'||id==='iota'){
   const aSide=id==='kappa';vector=aSide?r.da.kappa[j]:r.da.iota[m];labels=['0','1','2'];domain='Entry';
   title=aSide?`Target association ${j+1} → local factor`:`Measurement association ${m+1} → new-target factor`;
   description='Multiply all incoming constraint messages, entry by entry. The local weight is excluded on this return path.';
   equation=aSide?R`${kappa}(a)=\prod_m\nu^{[${r.p.iterations}]}_{m,${j+1}}(a)`:R`${iota}(b)=\prod_j\varphi^{[${r.p.iterations}]}_{j,${m+1}}(b)`;
   headers=['Incoming','at 0','at 1','at 2'];rows=[0,1].map(i=>[T(aSide?R`\nu_{${i+1},${j+1}}`:R`\varphi_{${i+1},${m+1}}`),...(aSide?r.da.nu[j][i]:r.da.phi[i][m]).map(fmt)]);rows.push(['PRODUCT',...vector.map(fmt)]);
   note=aSide?'Do not multiply β here: q already contains that local information. κ carries the rest of the association graph back to q.':'Do not multiply ξ here: v already contains that local information. ι carries the association graph back to v.';source=aSide?'Eq. (81)':'Eq. (82)';
  }else if(id==='gamma'||id==='zeta'){
   const old=id==='gamma',factor=old?r.q[j]:r.v[m],incoming=old?r.da.kappa[j]:r.da.iota[m];vector=old?r.gamma[j]:r.zeta[m];
   title=old?`Local factor → legacy state ${j+1}`:`New-target factor → new state ${m+1}`;
   description='Fix a state and sum over association choices. The returning evidence is now a function of state and existence again.';
   equation=old?R`${gamma}(y)=\sum_a ${Q}(y,a)${kappa}(a)`:R`${zeta}(\bar y)=\sum_b v^{${m+1}}(\bar y,b)${iota}(b)`;
   headers=[old?'a':'b','Local factor','Return','Product'];rows=incoming.map((w,a)=>[a,fmt(factor[entry][a]),fmt(w),fmt(w*factor[entry][a])]);footer=['SUM',fmt(vector[entry])];
   note=old?'γ is a likelihood message, not a posterior. Multiply it by α next. For an absent legacy target, γ(∅) = κ(0).':'A present new target requires b = 0. An absent new target allows every b, including assignments to a legacy target.';source='§IX-A4, measurement update';
  }else{
   const old=kind==='legacy',prior=old?r.alpha[j]:[1,1,1],update=old?r.gamma[j]:r.zeta[m],post=old?r.legacy[j]:r.newTargets[m],exact=old?r.exact.legacy[j]:r.exact.newTargets[m];vector=post;
   title=old?`Legacy target ${j+1}: combine α and γ`:`Possible new target ${m+1}: normalize ς`;
   description='Normalize across absent, left and right states. Sum the two present-state entries to obtain existence probability.';
   equation=old?R`\widetilde f(\underline y^{${j+1}})\propto ${alpha}(y)${gamma}(y)`:R`\widetilde f(\overline y^{${m+1}})\propto ${zeta}(\bar y)`;
   headers=['State','Raw weight','BP','Exact'];rows=post.map((w,y)=>[T(Y[y]),fmt(prior[y]*update[y]),fmt(w),fmt(exact[y])]);footer=['EXISTENCE · BP',fmt(1-post[0])];
   note=`Exact existence = ${fmt(1-exact[0])}. This graph has a loop: more iterations can stabilize messages without making the marginals exact.`;source='Eqs. (83)–(86); exact check over 7 matchings';
  }
  output=labels.map((label,i)=>({label:T(label),value:fmt(vector[i])}));
  return {title,description,equation:T(equation),table:table(headers,rows,footer),vector:output,note,source,domain,labels,normalized:id==='prior'||id==='predict'||id==='copy'||id==='belief'};
 }
 return {fmt,Y,groupNames,table,explain};
})();
