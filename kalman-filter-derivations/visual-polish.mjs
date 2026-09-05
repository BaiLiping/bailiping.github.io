/** Presentation-only layer. No model equations or live algorithms are changed.
 * All geometry uses the deck's 1280 × 720 coordinate system.
 */
import { groups } from './family-guide.mjs';
const R=String.raw;
const math=s=>`<span class="math-tex math-display">\\[${s}\\]</span>`;
const palette={paper:'#F8F6F0',panel:'#FFFEFB',ink:'#20372F',muted:'#61756B',rule:'#DCE3DB'};
const sans="Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",serif="Georgia, 'Times New Roman', serif",mono="'SFMono-Regular', Consolas, monospace";
const text=(id,x,y,w,h,html,o={})=>({id,type:'text',x,y,w,h,html,fontFamily:sans,fontSize:16,fontWeight:400,color:palette.ink,lineHeight:1.3,align:'left',valign:'top',rotation:0,opacity:1,...o});
const rect=(id,x,y,w,h,fill=palette.panel,stroke=palette.rule,radius=12)=>({id,type:'shape',shape:'rect',x,y,w,h,fill,stroke,strokeWidth:stroke==='none'?0:1,radius,rotation:0,opacity:1});
function heading(group,title,subtitle){return [text('polish-kicker',72,37,1120,20,group?`${group.n} / ${group.kind}`:'KALMAN FILTER · AN INTERACTIVE DERIVATION ATLAS',{fontFamily:mono,fontSize:11,fontWeight:800,color:group?.color||groups[0].color,letterSpacing:1.2}),text('polish-title',72,68,1136,50,title,{fontFamily:serif,fontSize:38,fontWeight:700,lineHeight:1.08}),text('polish-subtitle',74,125,1124,43,subtitle,{fontSize:15.5,color:palette.muted,lineHeight:1.35})];}
function overview(){
 const els=[text('hero-kicker',74,34,1050,21,'ESTIMATION NOTES / ONE FILTER, MANY DERIVATIONS',{fontSize:11,fontFamily:mono,color:groups[0].color,fontWeight:800,letterSpacing:1.4}),text('hero-line1',72,73,704,67,'One filter.',{fontFamily:serif,fontSize:60,fontWeight:700,lineHeight:1}),text('hero-line2',72,139,727,67,'Many ways to see it.',{fontFamily:serif,fontSize:49,fontWeight:700,color:groups[0].color,lineHeight:1}),rect('hero-result',813,78,395,133,'#E9F0E9','#BCD0C1',16),text('hero-result-label',835,95,351,18,'ONE LINEAR–GAUSSIAN ANSWER',{fontFamily:mono,fontSize:10,fontWeight:800,color:groups[0].color,letterSpacing:.8}),text('hero-result-eq',835,124,351,66,math(R`\begin{aligned}K&=PH^{\mathsf T}S^{-1}\\m^+&=m^-+K\nu\end{aligned}`),{fontSize:26,align:'center',valign:'middle'}),text('hero-deck',76,226,1120,43,'Follow the question, not just the algebra. Explore the principal routes, their assumptions, and what each viewpoint makes visible.',{fontSize:18,color:palette.muted,lineHeight:1.35})];
 const nouns=['DENSITY','GAIN','STATE','BELIEF','MESSAGE','FACTORS'];
 groups.forEach((g,i)=>{const x=72+(i%3)*384,y=291+Math.floor(i/3)*154;els.push(rect(`atlas-card-${i}`,x,y,368,139,palette.panel,palette.rule,13),rect(`atlas-accent-${i}`,x+17,y+19,4,28,g.color,'none',2),text(`atlas-number-${i}`,x+33,y+14,35,23,g.n,{fontSize:15,fontFamily:mono,color:g.color,fontWeight:800}),text(`atlas-noun-${i}`,x+215,y+18,133,19,nouns[i],{fontSize:10,fontFamily:mono,align:'right',color:g.color,fontWeight:700,letterSpacing:1.3}),text(`atlas-name-${i}`,x+32,y+47,313,33,g.short,{fontFamily:serif,fontSize:25,fontWeight:700,link:`group-${g.id}`}),text(`atlas-question-${i}`,x+33,y+89,305,35,g.question,{fontSize:13.5,color:palette.muted,lineHeight:1.28,link:`group-${g.id}`}),text(`atlas-arrow-${i}`,x+332,y+52,20,24,'↗',{fontSize:19,color:g.color,link:`group-${g.id}`}) );});
 els.push(text('atlas-foot',76,611,1122,44,'01–04  Derivation principles     /     05–06  Computational viewpoints\nThe groups overlap. This is a map of principal routes—not six independent filters or an exhaustive count of proofs.',{fontSize:13,color:palette.muted,lineHeight:1.55}));
 return els;
}
function comparison(){
 const equations=groups.map(g=>g.eq);equations[2]=R`\nabla_x\mathcal J(x)=0\quad\Longleftrightarrow\quad Jx=h`;
 const vars=['density','gain','state','distribution','message','factorization'];
 const els=heading(null,'Different questions. One Gaussian answer.','Read across: what changes, which equation governs it, and what the calculation produces.');
 els.push(text('cmp-header-a',89,185,235,19,'QUESTION / UNKNOWN',{fontFamily:mono,fontSize:10,fontWeight:800,color:palette.muted,letterSpacing:1}),text('cmp-header-b',338,185,574,19,'GOVERNING EQUATION',{fontFamily:mono,fontSize:10,fontWeight:800,color:palette.muted,letterSpacing:1}),text('cmp-header-c',969,185,216,19,'OUTPUT',{fontFamily:mono,fontSize:10,fontWeight:800,color:palette.muted,letterSpacing:1}));
 groups.forEach((g,i)=>{const y=214+i*64;els.push(rect(`equation-row-${i}`,72,y,1136,60,i%2?g.soft:palette.panel,'none',8),text(`equation-number-${i}`,88,y+14,28,26,g.n,{fontSize:12,fontFamily:mono,color:g.color,fontWeight:800}),text(`equation-group-${i}`,123,y+9,196,22,g.short,{fontSize:16,fontWeight:700,color:g.color,link:'group-'+g.id}),text(`equation-variable-${i}`,124,y+34,186,17,vars[i],{fontSize:11.5,color:palette.muted}),text(`equation-math-${i}`,329,y+7,606,45,math(equations[i]),{fontSize:17,align:'center',valign:'middle'}),text(`equation-output-${i}`,962,y+10,225,42,g.output,{fontSize:13.3,lineHeight:1.28}));});
 els.push(text('cmp-context',77,611,1128,45,'P = P⁻, ν = z − Hm⁻, S = HPHᵀ + R. Inverse-based forms assume positive-definite covariances.\n𝒥 is the weighted quadratic cost; J and h are its precision and information coefficients. Full notation follows.',{fontSize:13,color:palette.muted,lineHeight:1.45}));
 return els;
}
function summary(g){
 const els=heading(g,g.name,g.question+'  Five principal routes, with their governing equations.');
 g.rows.forEach(([name,eq,why],i)=>{const y=184+i*78;els.push(rect(`summary-card-${i}`,72,y,1136,72,palette.panel,palette.rule,10),rect(`summary-line-${i}`,87,y+18,3,35,g.color,'none',1),text(`summary-name-${i}`,104,y+14,319,46,name,{fontSize:16,fontWeight:700,color:g.color,lineHeight:1.25}),text(`summary-equation-${i}`,442,y+7,741,38,math(eq),{fontSize:18,align:'center',valign:'middle'}),text(`summary-reading-${i}`,451,y+48,724,19,why,{fontSize:12.7,color:palette.muted,align:'center'}));});
 const next=g.id==='numerical'?'implementations':' '+g.id+'-lab';
 els.push(rect('summary-demo-chip',73,586,184,26,g.soft,'none',7),text('summary-demo-link',87,590,157,18,'EXPLORE THE DEMO ↗',{fontSize:10,fontFamily:mono,fontWeight:800,color:g.color,link:next.trim()}),text('summary-demo',275,584,920,35,g.demo,{fontSize:13.1,lineHeight:1.3}),text('summary-limit',79,626,1120,33,g.boundary,{fontSize:11.7,color:palette.muted,lineHeight:1.26}));
 return els;
}
export function applyVisualPolish(slides){
 if(slides.some(s=>s.elements.some(e=>e.id==='polish-progress-line')))throw new Error('Visual polish applied twice');
 let active=null;
 const detailGroup={bayes:'bayes','bayes-equations':'bayes','conditioning-bridge':'bayes',mse:'mse','mse-live':'mse','mse-equations':'mse','covariance-identities':'mse','least-squares':'wls','least-squares-equations':'wls',kl:'kl','kl-equations':'kl',graphs:'graphs',implementations:'numerical','implementations-live':'numerical'};
 slides.forEach((s,index)=>{
  const group=groups.find(g=>s.id==='group-'+g.id||s.id===g.id+'-lab'||detailGroup[s.id]===g.id);
  if(group)active=group;else if(['overview','family-comparison','model','model-live','assumptions-prediction','boundaries','equivalence','references'].includes(s.id))active=null;
  const footer=s.elements.filter(e=>e.id.startsWith('guide-')&&['guide-home','guide-source','guide-rule','guide-progress','guide-toc'].includes(e.id));
  if(s.id==='overview')s.elements=[...overview(),...footer];
  if(s.id==='family-comparison')s.elements=[...comparison(),...footer];
  if(s.id.startsWith('group-'))s.elements=[...summary(group),...footer];
  s.background=palette.paper;
  s.elements.unshift(rect('polish-progress-track',0,0,1280,4,'#E1E8DF','none',0),rect('polish-progress-line',0,0,1280*(index+1)/slides.length,4,active?.color||groups[0].color,'none',0));
  for(const e of s.elements){
   if(e.type==='shape'&&e.fill==='#F7F5EF')e.fill=palette.paper;
   if(e.type==='text'&&e.id==='guide-toc'){e.fontSize=10;e.fontWeight=750;}
  }
  // The context label sits outside every existing heading's authored bounds.
  if(active&&!s.id.startsWith('group-'))s.elements.push(text('polish-group-tag',1030,38,178,19,`${active.n} / ${active.short.toUpperCase()}`,{fontSize:9.5,fontFamily:mono,color:active.color,fontWeight:800,align:'right'}));
 });
}
