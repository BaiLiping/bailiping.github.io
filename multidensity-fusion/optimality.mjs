/** Optimality companion slides for the existing Bento deck.
 * Pure transformation: the original slides and six live experiments are retained.
 * Each claim states its objective, admissible family, and assumptions.
 */
const C={paper:'#F7F5EF',panel:'#FFFEFB',ink:'#203129',muted:'#66756E',rule:'#D8DED7',green:'#2F6B4F',soft:'#E7F0EA'};
const SANS="Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
const SERIF="Georgia, 'Times New Roman', serif";
const MONO="'SFMono-Regular', Consolas, monospace";
const raw=String.raw;
const inline=s=>`<span class="math-tex math-inline">\\(${s}\\)</span>`;
const display=s=>`<span class="math-tex math-display">\\[${s}\\]</span>`;
const text=(id,x,y,w,h,html,fontSize=18,options={})=>({id,type:'text',x,y,w,h,html,fontSize,fontFamily:SANS,fontWeight:400,color:C.ink,lineHeight:1.35,align:'left',valign:'top',rotation:0,opacity:1,...options});
const rect=(id,x,y,w,h,fill=C.panel)=>({id,type:'shape',shape:'rect',x,y,w,h,fill,stroke:C.rule,strokeWidth:1,radius:13,rotation:0,opacity:1});
const formula=(id,x,y,w,h,latex,size=23)=>text(id,x,y,w,h,display(latex),size,{align:'center',valign:'middle'});
const panel=(id,x,y,w,h,title,body)=>[
  rect(`${id}-panel`,x,y,w,h),
  text(`${id}-label`,x+20,y+17,w-40,25,title,12,{fontFamily:MONO,fontWeight:800,color:C.green}),
  text(`${id}-body`,x+20,y+51,w-40,h-65,body,18)
];
const entries=[
  {
    after:'bayes',id:'bayes-optimality',title:'Bayes minimizes a data-fit / prior trade-off',
    subtitle:'Independent private likelihoods, a common prior, and a finite positive evidence Z.',
    equation:raw`\begin{aligned}J_B(q)&=\operatorname{KL}(q\Vert p_0)-\sum_i\mathbb E_q[\log L_i]\\&=\operatorname{KL}(q\Vert p_F)-\log Z\end{aligned}`,
    leftTitle:'THE MINIMIZER',
    left:`${display(raw`p_F=\frac{p_0\prod_i L_i}{Z}`)}The only q-dependent remainder is a nonnegative KL divergence. Equality holds exactly at ${inline('q=p_F')} almost everywhere.`,
    rightTitle:'WHY THE PRIOR APPEARS ONCE',
    right:`${display(raw`p_F\propto\frac{\prod_i p_i}{p_0^{M-1}}`)}Each local posterior already contains p₀. Removing duplicate priors recovers the minimizer above; an uncorrected product solves a different problem.`,
    takeaway:'Optimal for this evidence model—not for arbitrary dependent posteriors.',
    notes:'Minimize over normalized densities q, with q absolutely continuous with respect to p0 and finite objective terms. Z=∫p0∏iLi dμ is finite and positive. Substitute log pF=log p0+Σi log Li−log Z to obtain the identity. KL nonnegativity proves global optimality and uniqueness up to null sets. If all sources share a known common-information density pc plus conditionally independent private data, replace p0 by pc. Arbitrarily overlapping information histories do not generally reduce to dividing by pc^(M−1). Under posterior expected logarithmic loss, the same posterior is optimal because R(q)−R(pF)=KL(pF||q); squared-error point estimation instead selects its mean. This proof is an explanatory derivation using the cited evidence model.'
  },
  {
    after:'known-correlation',id:'known-correlation-optimality',title:'Complete the covariance square',
    subtitle:'Optimality is over linear unbiased estimators; S must be positive definite.',
    equation:raw`\begin{aligned}W_*&=(P_1-C)S^{-1}\\P(W)&=P(W_*)+(W-W_*)S(W-W_*)^{\mathsf T}\end{aligned}`,
    leftTitle:'EXPAND THE ERROR COVARIANCE',
    left:`${inline(raw`e_F=e_1+W(e_2-e_1)`)}. Set ${inline(raw`B=P_1-C`)} and ${inline(raw`S=P_1+P_2-C-C^{\mathsf T}`)}.${display(raw`P(W)=P_1-WB^{\mathsf T}-BW^{\mathsf T}+WSW^{\mathsf T}`)}`,
    rightTitle:'A GLOBAL MATRIX MINIMUM',
    right:`${display(raw`P(W)-P(W_*)\succeq0`)}The remainder is positive semidefinite. Thus W* minimizes every directional error variance, and hence trace or any covariance-monotone cost.`,
    takeaway:'No Gaussian assumption is needed. Known joint error covariance is essential.',
    notes:'Let ei=xi_hat−x be zero-mean estimation errors, with Pi=E[ei ei^T] and C=E[e1 e2^T], from a valid joint covariance. E[e1(e2−e1)^T]=C−P1=−B. Expanding gives P1−WB^T−BW^T+WSW^T. Complete the matrix square at W*=BS^−1. Because S≻0, the remainder is PSD and vanishes only at W*. This establishes the best linear unbiased fusion covariance in Loewner order; it does not establish a general posterior from marginal densities. Singular S needs a separate pseudoinverse/constraint treatment.'
  },
  {
    after:'ci',id:'ci-optimality',title:'CI: optimize a bound that remains valid',
    subtitle:'Separate fixed-weight Gaussian KL pooling from choosing the smallest CI-family bound.',
    equation:raw`P_w=\left(\sum_iw_iP_i^{-1}\right)^{-1},\qquad w_*\in\arg\min_{w\in\Delta}\phi(P_w)`,
    leftTitle:'WHY THE BOUND IS CONSERVATIVE',
    left:`Let ${inline(raw`u_i=P_i^{-1}e_i`)} and ${inline(raw`e_F=P_w\sum_iw_iu_i`)}. Weighted convexity gives${display(raw`\mathbb E[e_Fe_F^{\mathsf T}]\preceq P_w\!\left(\sum_iw_i\mathbb E[u_iu_i^{\mathsf T}]\right)\!P_w\preceq P_w`)}`,
    rightTitle:'WHAT IS ACTUALLY OPTIMIZED?',
    right:`${inline(raw`\Delta=\{w_i\ge0:\sum_iw_i=1\}`)}.${display(raw`\phi(P)=\log\det P\quad\text{or}\quad\operatorname{tr}P`)}This selects the best <b>CI-family bound</b> for the chosen cost. It does not minimize an unknown actual MSE.`,
    takeaway:'Fixed weights + unbiased inputs + valid SPD covariance bounds; arbitrary data-dependent weights need extra justification.',
    notes:'Assume fixed SPD Pi, fixed weights wi, and E[ei ei^T]≼Pi. Pointwise, (Σwi ui)(Σwi ui)^T≼Σwi ui ui^T because their difference is a weighted covariance matrix. Also E[ui ui^T]≼Pi^−1. Multiplication by Pw proves the bound without any assumption on cross-correlation. For Gaussian input densities, the same fixed-weight mean and covariance follow by completing the quadratic in Σwi log pi, and minimize Σwi KL(q||pi). Weight optimization is a distinct problem. The slide only claims optimality within the CI family, not an unrestricted minimax theorem for arbitrary numbers of sources or extra correlation structure. Weights computed from random data require appropriate conditional consistency assumptions.'
  },
  {
    after:'moments',id:'moments-optimality',title:'Moment matching is the best forward-KL Gaussian',
    subtitle:'An exact optimization within the Gaussian family; still an approximation to the full mixture.',
    equation:raw`(\mu_A,P_A)=\arg\min_{m,\,P\succ0}\operatorname{KL}\!\left(q_A\Vert\mathcal N(m,P)\right)`,
    leftTitle:'ONLY THE FIRST TWO MOMENTS ENTER',
    left:`Ignoring constants independent of m and P, the objective is${display(raw`\frac12\left[\log\det P+\operatorname{tr}(P^{-1}P_A)\right]`)}${display(raw`{}+\frac12(m-\mu_A)^{\mathsf T}P^{-1}(m-\mu_A)`)}`,
    rightTitle:'THE NONNEGATIVE OBJECTIVE GAP',
    right:`Write ${inline(raw`g_A=\mathcal N(\mu_A,P_A)`)} and ${inline(raw`g=\mathcal N(m,P)`)}.${display(raw`\operatorname{KL}(q_A\Vert g)-\operatorname{KL}(q_A\Vert g_A)=\operatorname{KL}(g_A\Vert g)\ge0`)}Equality requires m = μA and P = PA.`,
    takeaway:'Best Gaussian under KL(qA ∥ g) does not mean that multiple modes or tail structure are preserved.',
    notes:'Assume finite second moments, PA≻0, and finite KL quantities. The logarithm of a nonsingular Gaussian is quadratic, so its expectation under qA depends only on μA and PA. Since gA has the same moments, the difference of cross-entropies equals KL(gA||g). This proves uniqueness. PA must include the between-source spread Σwi(μi−μA)(μi−μA)^T. A singular PA needs a compatible lower-dimensional Gaussian family; no nonsingular exact moment match exists. The displayed result is a restricted-family projection, not AA full-density closure.'
  },
  {
    after:'wasserstein',id:'wasserstein-optimality',title:'In one dimension, complete a quantile square',
    subtitle:'Let Q and Qi be quantile functions. All input distributions have finite second moments.',
    equation:raw`\sum_iw_iW_2^2(q,p_i)=\int_0^1\!\left[Q(u)-\bar Q(u)\right]^2du+K,\qquad\bar Q=\sum_iw_iQ_i`,
    leftTitle:'WHY THE AVERAGE IS FEASIBLE',
    left:`In one dimension,${display(raw`W_2^2(q,p_i)=\int_0^1[Q(u)-Q_i(u)]^2du`)}A weighted average of nondecreasing quantile functions is again a quantile function. The square vanishes at Q = Q̄.`,
    rightTitle:'GAUSSIANS GIVE A CLOSED FORM',
    right:`${display(raw`Q_i(u)=\mu_i+\sigma_i\Phi^{-1}(u)`)}Therefore${display(raw`\mu_W=\sum_iw_i\mu_i,\qquad\sigma_W=\sum_iw_i\sigma_i`)}Standard deviations—not variances—are averaged.`,
    takeaway:'K is independent of q. This proves the transport optimum, not a Bayesian or covariance-consistency guarantee.',
    notes:'K=Σwi∫0^1(Qi−Qbar)^2du. Expand the weighted square pointwise and integrate; Σwi=1 cancels the cross term. Qbar is nondecreasing and square integrable, so it represents a valid probability distribution in P2(R). The optimizer is unique as a distribution, with quantile functions identified almost everywhere. For Gaussian inputs Φ is the standard normal CDF and σi≥0. This one-dimensional proof is not a general matrix square-root averaging rule. Multidimensional Gaussian barycenters satisfy a matrix fixed-point equation; the cited transport paper treats that case.'
  },
  {
    after:'sets',id:'sets-optimality',title:'The same optimality identities hold on set space',
    subtitle:'Optimize a normalized multi-object density—not a unit-normalized intensity.',
    equation:raw`D_{\rm set}(f\Vert g)=\int f(X)\log\!\frac{f(X)}{g(X)}\,\delta X`,
    leftTitle:'ARITHMETIC FULL-DENSITY POOL',
    left:`${display(raw`f_A=\sum_iw_if_i`)}${display(raw`\sum_iw_iD_{\rm set}(f_i\Vert g)=K_A+D_{\rm set}(f_A\Vert g)`)}The nonnegative remainder is minimized at g = fA.`,
    rightTitle:'GEOMETRIC FULL-DENSITY POOL',
    right:`${display(raw`f_G=Z^{-1}\prod_if_i^{w_i}`)}${display(raw`\sum_iw_iD_{\rm set}(g\Vert f_i)=D_{\rm set}(g\Vert f_G)-\log Z`)}For Z > 0, the minimizer is g = fG.`,
    takeaway:'Bernoulli and Poisson formulas inherit this optimality only when closure or an explicit projection is established.',
    notes:'The KL identities use only a common reference measure and normalization, so replace dμ by the finite-set measure δX. KA=Σwi Dset(fi||fA) is independent of g. Assume fixed nonnegative weights summing to one and omit inactive sources. For the geometric identity require Z>0 and finite relevant KL terms; avoid subtracting infinities. Sources must represent the same object population, coordinate system, and labeling convention. An intensity integrates to expected object count, not one; it cannot be substituted for a full density in this argument.'
  },
  {
    after:'bernoulli',id:'bernoulli-optimality',title:'Optimize existence and location together',
    subtitle:'For interior existence probabilities and positive spatial overlap; boundary cases use the original slide’s conventions.',
    equation:raw`D_{\rm set}(f_{r,p}\Vert f_{s,q})=d_B(r\Vert s)+r\operatorname{KL}(p\Vert q)`,
    leftTitle:'GCI: OVERLAP ENTERS THE EXISTENCE LOSS',
    left:`With ${inline(raw`\eta=\int\prod_i p_i^{w_i}dx`)}, optimizing p gives${display(raw`\min_p J_G(r,p)=\sum_iw_i d_B(r\Vert r_i)-r\log\eta`)}Stationarity in r yields${display(raw`\log\frac{r_G}{1-r_G}=\sum_iw_i\log\frac{r_i}{1-r_i}+\log\eta`)}`,
    rightTitle:'AA: EXISTENCE-WEIGHTED SPATIAL LOSS',
    right:`${display(raw`J_A(r,p)=\sum_iw_i d_B(r_i\Vert r)+\sum_iw_ir_i\operatorname{KL}(p_i\Vert p)`)}The two terms separate, giving${display(raw`r_A=\sum_iw_ir_i,\qquad p_A=\frac{\sum_iw_ir_ip_i}{r_A}`)}`,
    takeaway:'dB is the KL divergence between Bernoulli existence variables; spatial fusion cannot ignore existence weights.',
    notes:'Define dB(r||s)=r log(r/s)+(1−r)log((1−r)/(1−s)). Sum the empty-set contribution and the singleton integral to derive the decomposition. For GCI, pG=∏pi^wi/η and min_p Σwi KL(p||pi)=−log η. The remaining r-objective has second derivative 1/[r(1−r)]>0 and derivative log[r/(1−r)]−Σwi log[ri/(1−ri)]−log η. Its unique stationary point gives the displayed odds equation and hence the original existence formula. For AA, minimizing the first term yields rA=Σwiri; the second is a spatial forward-KL pool with normalized weights wi ri/rA. The on-slide differentiations assume 0<ri<1 and η>0; at rA=0 spatial density is irrelevant, and zero overlap or endpoint inputs require the set-normalization boundary treatment on the original slide.'
  },
  {
    after:'poisson',id:'poisson-optimality',title:'Poisson GCI minimizes an intensity divergence',
    subtitle:'Poisson closure turns the full set-density KL problem into a pointwise convex problem.',
    equation:raw`D_{\rm set}(f_D\Vert f_{D_i})=\int\!\left[D\log\frac{D}{D_i}-D+D_i\right]dx`,
    leftTitle:'DIFFERENTIATE WITH RESPECT TO INTENSITY',
    left:`For ${inline(raw`J_G(D)=\sum_iw_iD_{\rm set}(f_D\Vert f_{D_i})`)},${display(raw`\frac{\delta J_G}{\delta D}=\log D-\sum_iw_i\log D_i`)}Hence${display(raw`D_G(x)=\prod_iD_i(x)^{w_i}`)}`,
    rightTitle:'THE GLOBAL OPTIMALITY CERTIFICATE',
    right:`${display(raw`J_G(D)-J_G(D_G)`)}${display(raw`=\int\!\left[D\log\frac{D}{D_G}-D+D_G\right]dx\ge0`)}The integrand is a generalized KL divergence. Set-density closure makes this the unrestricted GCI solution too.`,
    takeaway:'DG is an intensity: its integral is the fused expected count, not a quantity to normalize to one.',
    notes:'For finite-rate PPPs, log(fD/fDi)=−λ+λi+Σx∈X log[D(x)/Di(x)]. The PPP expectation of the sum is ∫D log(D/Di)dx, giving the displayed divergence. On positive common support, pointwise differentiation yields DG. The exact gap is nonnegative since a log(a/b)−a+b≥0, with equality at a=b. Outside common support, a finite reverse-KL candidate cannot put intensity where an active source has none. Zero overlap gives DG=0 and a valid empty PPP; the full-set normalizer remains exp(−Σwiλi+λG)>0. Rates and objective integrals are assumed finite. The global full-density minimizer is a PPP because the normalized geometric product closes in that family.'
  },
  {
    after:'poisson-aa',id:'poisson-aa-optimality',title:'AA intensity is the best Poisson approximation',
    subtitle:'The full arithmetic mixture remains optimal among all set densities; now restrict the output to a PPP.',
    equation:raw`D_A=\sum_iw_iD_i,\qquad f_{D_A}=\arg\min_{f_D\,\in\,\mathrm{PPP}}D_{\rm set}(f_A\Vert f_D)`,
    leftTitle:'ONLY THE MIXTURE INTENSITY IS NEEDED',
    left:`Using the first-moment identity for fA,${display(raw`D_{\rm set}(f_A\Vert f_D)=K+\int D\,dx-\int D_A\log D\,dx`)}Thus ${inline(raw`\delta J/\delta D=1-D_A/D=0`)} gives D = DA on its positive support.`,
    rightTitle:'A NONNEGATIVE PROJECTION GAP',
    right:`${display(raw`D_{\rm set}(f_A\Vert f_D)-D_{\rm set}(f_A\Vert f_{D_A})`)}${display(raw`=\int\!\left[D_A\log\frac{D_A}{D}-D_A+D\right]dx\ge0`)}This is an exact optimum <b>within PPPs</b>, not a full-density equality.`,
    takeaway:'The KL-best PPP preserves intensity, but discards mixture count overdispersion and object dependencies.',
    notes:'For any candidate PPP, log fD(X)=−∫D dx+Σx∈X log D(x). Taking expectation under the arbitrary finite-first-moment density fA uses E_fA[Σh(x)]=∫DA h dx. The entropy term K does not depend on D. Subtracting the objective at DA yields the generalized KL integral, proving uniqueness up to null sets where finite. At DA=0 the minimum is D=0. Finite expected count and the required KL integrability are assumed. Equivalently minimize Σwi Dset(fi||fD), since it differs from Dset(fA||fD) by a constant. This projection does not make the arithmetic mixture itself Poisson.'
  },
  {
    after:'consensus',id:'consensus-optimality',title:'Consensus computes a network-weighted KL center',
    subtitle:'Fixed primitive row-stochastic A, common positive support, and finite nonzero normalizers.',
    equation:raw`q_\infty=\arg\min_q\sum_j\pi_j\operatorname{KL}(q\Vert q_j^{(0)}),\qquad \pi^{\mathsf T}A=\pi^{\mathsf T}`,
    leftTitle:'NORMALIZATION DISAPPEARS IN LOG RATIOS',
    left:`For a reference state x₀ in common support, define${display(raw`\ell_i^{(t)}(x)=\log\frac{q_i^{(t)}(x)}{q_i^{(t)}(x_0)}`)}Then ${inline(raw`\ell^{(t+1)}=A\ell^{(t)}`)} and ${inline(raw`A^t\to\mathbf1\pi^{\mathsf T}`)}.`,
    rightTitle:'IDENTIFY THE LIMIT AND ITS OBJECTIVE',
    right:`${display(raw`q_\infty\propto\prod_j(q_j^{(0)})^{\pi_j}`)}${display(raw`\sum_j\pi_j\operatorname{KL}(q\Vert q_j^{(0)})=\operatorname{KL}(q\Vert q_\infty)-\log Z_\pi`)}KL nonnegativity supplies the optimality proof.`,
    takeaway:'The network determines π. Uniform weighting requires additional balance; more rounds do not create evidence.',
    notes:'Use finite positive density values at the chosen reference state, or equivalently work with log densities modulo additive constants. At every finite round, qi^(t) is the normalized geometric pool of the initial densities with weights equal to row i of A^t. Primitivity gives a unique positive stationary vector π, sum π=1. Common positive support and finite positive normalizers justify the limiting normalized pool; weighted AM–GM provides an integrable dominating sum of the input densities. The usual GCI KL identity then proves global optimality for the network-determined weights. A primitive doubly stochastic matrix gives πj=1/M. The statement is about repeated pooling of fixed initial beliefs, not filtering with new data or arbitrary time-varying graphs.'
  }
];

export const optimalitySlideIds=Object.freeze(['optimality-guide',...entries.map(e=>e.id)]);

function companion(base,spec){
  const footer=base.elements.filter(e=>e.id.startsWith('source-')||['footer-rule','footer-site','footer-map'].includes(e.id));
  return {id:spec.id,background:C.paper,transition:'morph',notes:spec.notes+'\n\nReferences inherited from the preceding method slide.\n'+base.notes,elements:[
    text('eyebrow',72,34,1120,22,'ESTIMATION NOTES / WHY THIS FUSION IS OPTIMAL',11,{fontFamily:MONO,color:C.green,fontWeight:800,letterSpacing:1.4}),
    text('heading',72,69,1136,57,spec.title,36,{fontFamily:SERIF,fontWeight:700,lineHeight:1.1}),
    text('subtitle',74,128,1132,44,spec.subtitle,17,{color:C.muted}),
    rect('equation-panel',72,184,1136,139,C.soft),formula('main-equation',92,190,1096,127,spec.equation,23),
    ...panel('left',72,340,554,245,spec.leftTitle,spec.left),
    ...panel('right',646,340,562,245,spec.rightTitle,spec.right),
    rect('takeaway-bg',72,592,1136,43,C.soft),text('takeaway',88,601,1104,31,spec.takeaway,14,{color:C.green,fontWeight:600}),
    ...footer.map(e=>({...e})),text('back-method',790,688,170,18,'METHOD ↗',10,{link:spec.after,color:C.green,align:'right'})
  ]};
}
function guide(base){
  const slide=companion(base,{id:'optimality-guide',after:'map',title:'Optimal under which criterion?',subtitle:'Fix the objective and admissible outputs before calling a fusion rule optimal.',equation:'',leftTitle:'',left:'',rightTitle:'',right:'',takeaway:'Pooling weights are fixed unless explicitly optimized. Multi-object rules use set KL; approximations restrict the output family.',notes:'This is a map of distinct optimization problems, not a ranking of fusion methods. Fixed nonnegative pooling weights sum to one; omit zero-weight sources. Density KL uses normalized distributions on a common reference measure. Infinite-divergence and zero-normalizer cases require the support qualifications on each method slide.'});
  slide.elements=slide.elements.filter(e=>!['equation-panel','main-equation'].includes(e.id)&&!e.id.startsWith('left-')&&!e.id.startsWith('right-'));
  const cards=[
    ['BAYES / KNOWN EVIDENCE',raw`\min_q\ \operatorname{KL}(q\Vert p_0)-\sum_i\mathbb E_q\log L_i`,'bayes-optimality'],
    ['KNOWN ERROR CORRELATION',raw`\min_W\ \operatorname{tr}P(W)`,'known-correlation-optimality'],
    ['COVARIANCE INTERSECTION',raw`\min_{w\in\Delta}\ \phi(P_w)`,'ci-optimality'],
    ['GEOMETRIC POOL / GCI',raw`\min_q\ \sum_iw_i\operatorname{KL}(q\Vert p_i)`,'gci-derivation'],
    ['ARITHMETIC POOL / AA',raw`\min_q\ \sum_iw_i\operatorname{KL}(p_i\Vert q)`,'aa'],
    ['WASSERSTEIN BARYCENTER',raw`\min_q\ \sum_iw_iW_2^2(q,p_i)`,'wasserstein-optimality']
  ];
  cards.forEach(([label,math,target],i)=>{const x=72+(i%2)*576,y=184+Math.floor(i/2)*131;
    slide.elements.push(rect(`criterion-${i}`,x,y,560,117),text(`criterion-label-${i}`,x+18,y+14,415,23,label,12,{fontFamily:MONO,color:C.green,fontWeight:800}),
      text(`criterion-link-${i}`,x+430,y+14,108,22,'PROOF ↗',11,{link:target,color:C.green,align:'right'}),formula(`criterion-math-${i}`,x+18,y+32,524,82,math,22));
  });
  return slide;
}

/** Clone, insert proofs, strengthen AA/GCI certificates, and repair all indexes. */
export function withOptimality(baseDeck,baseLiveMap){
  if(!Array.isArray(baseDeck?.slides)||!Array.isArray(baseLiveMap))throw new TypeError('Expected a Bento deck and live-map array.');
  const deck=structuredClone(baseDeck),inlineLiveMap=structuredClone(baseLiveMap);
  const ids=new Set(deck.slides.map(s=>s.id));
  if(ids.size!==deck.slides.length)throw new Error('Base deck has duplicate slide IDs.');
  if(optimalitySlideIds.some(id=>ids.has(id)))throw new Error('Optimality slides have already been installed.');
  const byId=new Map(deck.slides.map(s=>[s.id,s]));
  for(const id of ['map','gci-derivation','aa',...entries.map(e=>e.after)])if(!byId.has(id))throw new Error(`Missing method slide: ${id}`);
  const replace=(slideId,elementId,html)=>{const element=byId.get(slideId).elements.find(e=>e.id===elementId);if(!element)throw new Error(`Missing ${slideId}/${elementId}`);element.html=html;};
  replace('aa','left-body',`${display(raw`J_A(q)=J_A(q_A)+\operatorname{KL}(q_A\Vert q)`)}where ${inline(raw`J_A(q)=\sum_iw_i\operatorname{KL}(p_i\Vert q)`)}.<br><br>The remainder is nonnegative and vanishes exactly when ${inline(raw`q=q_A`)} almost everywhere. This is a global optimum over normalized densities.`);
  replace('gci-derivation','right-body',`${inline(raw`q_G=\arg\min_q\sum_iw_i\operatorname{KL}(q\Vert p_i)`)}.<br><br>KL nonnegativity gives the minimum ${inline(raw`-\log Z_w`)}; equality requires ${inline(raw`q=q_G`)} almost everywhere.<br><br>Require ${inline(raw`Z_w>0`)} and finite relevant terms. Mass outside an active input’s support gives infinite KL.`);
  const additions=new Map(entries.map(spec=>[spec.after,companion(byId.get(spec.after),spec)]));
  additions.set('map',guide(byId.get('map')));
  deck.slides=deck.slides.flatMap(s=>additions.has(s.id)?[s,additions.get(s.id)]:[s]);
  deck.meta={...deck.meta,subject:baseDeck.meta.subject+'; explicit optimality objectives and proofs',optimalityCompanionVersion:1};
  const indexes=new Map(deck.slides.map((s,i)=>[s.id,i]));
  for(const [i,s] of deck.slides.entries()){
    s.elements=s.elements.filter(e=>!['progress-track','progress-fill','footer-page'].includes(e.id));
    s.elements.unshift({...rect('progress-track',0,0,1280,4,C.rule),strokeWidth:0,radius:0},{...rect('progress-fill',0,0,1280*(i+1)/deck.slides.length,4,C.green),strokeWidth:0,radius:0});
    s.elements.push(text('footer-page',1120,687,86,20,`${String(i+1).padStart(2,'0')} / ${deck.slides.length}`,11,{fontFamily:MONO,color:C.muted,align:'right'}));
  }
  for(const lab of inlineLiveMap){if(!indexes.has(lab.slide))throw new Error(`Missing live slide: ${lab.slide}`);lab.slideIndex=indexes.get(lab.slide);}
  return {deck,inlineLiveMap};
}
