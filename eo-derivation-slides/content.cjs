'use strict';

// Presentation content for the restored four-scenario article. TeX macros and
// notation follow eo-derivation/index.html. Appendix I adds GrBP after S4.
const t = String.raw;
const equations = {
  chain: t`f(S,A,Z)=f(S)\,p(A,m\mid S)\,f(Z\mid A,m,S)`,
  constant: t`C(Z,m)=\frac{\mu_c^m e^{-\mu_c}}{m!}\prod_{j=1}^{m}f_c(\boldsymbol z_j)`,
  state: t`\begin{aligned}\text{point: }&\boldsymbol x\qquad\text{extended: }\boldsymbol y=(\boldsymbol x,\boldsymbol e)\\[8pt]\text{uncertain existence: }&r\in\{0,1\}\end{aligned}`,
  association: t`a_i\in\{0,1,\ldots,m\},\qquad a_i=0:\ \text{missed detection}`,
  psi: t`\psi(\boldsymbol a)=\begin{cases}1,&a_i\ne a_j\ \text{whenever }i\ne j,\ a_i,a_j\ne0,\\[5pt]0,&\text{otherwise}.\end{cases}`,
  s1counts: t`\mathcal D=\{i:a_i\ne0\},\qquad n_c=m-|\mathcal D|`,
  likelihood: t`\begin{aligned}f(Z\mid\boldsymbol a,\ub X,m)&=\prod_{i\in\mathcal D}f(\boldsymbol z_{a_i}\mid\boldsymbol x_i)\prod_{j\in\mathcal C}f_c(\boldsymbol z_j)\\[12pt]&=\left[\prod_{i\in\mathcal D}\frac{f(\boldsymbol z_{a_i}\mid\boldsymbol x_i)}{f_c(\boldsymbol z_{a_i})}\right]\prod_{j=1}^{m}f_c(\boldsymbol z_j).\end{aligned}`,
  s1cancel: t`\begin{aligned}{}_{m}P_{m-n_c}&=\frac{m!}{n_c!},\\[9pt]\frac{\mu_c^{n_c}e^{-\mu_c}}{n_c!}\frac{n_c!}{m!}&=\frac{\mu_c^m e^{-\mu_c}}{m!}\prod_{i\in\mathcal D}\frac{1}{\mu_c}.\end{aligned}`,
  pointg: t`g(\boldsymbol x_i,a_i;Z)=\begin{cases}\dfrac{P_d(\boldsymbol x_i)f(\boldsymbol z_{a_i}\mid\boldsymbol x_i)}{\mu_c f_c(\boldsymbol z_{a_i})},&a_i\ne0,\\[12pt]1-P_d(\boldsymbol x_i),&a_i=0.\end{cases}`,
  pointjoint: t`f(\ub X,\boldsymbol a\mid Z)\propto\psi(\boldsymbol a)\prod_{i=1}^{n_t}f(\boldsymbol x_i)g(\boldsymbol x_i,a_i;Z)`,
  legacyg: t`\ub g(\ub x_i,\ub r_i,a_i;Z)=\begin{cases}\dfrac{P_d(\ub x_i)f(\boldsymbol z_{a_i}\mid\ub x_i)}{\mu_c f_c(\boldsymbol z_{a_i})},&a_i\ne0,\ \ub r_i=1,\\[12pt]1-P_d(\ub x_i),&a_i=0,\ \ub r_i=1,\\[5pt]0,&a_i\ne0,\ \ub r_i=0,\\[5pt]1,&a_i=0,\ \ub r_i=0.\end{cases}`,
  s2counts: t`\mathcal N=\{j:\bar r_j=1\},\quad n_n=|\mathcal N|,\quad n_c=m-|\mathcal D|-n_n`,
  s2cancel: t`\begin{aligned}\binom{n_c+n_n}{n_n}^{-1}\!\left({}_{m}P_{m-n_c-n_n}\right)^{-1}&=\frac{n_n!\,n_c!}{m!},\\[12pt]\mu_n^{n_n}\mu_c^{n_c}&=\mu_c^m\left(\frac{\mu_n}{\mu_c}\right)^{n_n}\prod_{i\in\mathcal D}\frac1{\mu_c}.\end{aligned}`,
  newg: t`\bar g(\bar{\boldsymbol x}_j,\bar r_j,\boldsymbol a;\boldsymbol z_j)=\begin{cases}\dfrac{\mu_n f_n(\bar{\boldsymbol x}_j)f(\boldsymbol z_j\mid\bar{\boldsymbol x}_j)}{\mu_c f_c(\boldsymbol z_j)},&j\notin\mathcal T,\ \bar r_j=1,\\[12pt]0,&j\in\mathcal T,\ \bar r_j=1,\\[5pt]f_d(\bar{\boldsymbol x}_j),&\bar r_j=0.\end{cases}`,
  s2joint: t`\begin{aligned}f(\ub Y,\ob Y,\boldsymbol a\mid Z)\propto{}&\psi(\boldsymbol a)\prod_{i=1}^{n_t}f(\ub y_i)\,\ub g(\ub y_i,a_i;Z)\\[12pt]&\times\prod_{j=1}^{m}\bar g(\ob y_j,\boldsymbol a;\boldsymbol z_j).\end{aligned}`,
  eoassociation: t`b_i\in\{0,1,\ldots,K\},\qquad\mathcal T^k=\{i:b_i=k\}`,
  poiss: t`p\!\left(|\mathcal T^k|\mid\boldsymbol y_k\right)=\frac{\mu_m(\boldsymbol y_k)^{|\mathcal T^k|}}{|\mathcal T^k|!}e^{-\mu_m(\boldsymbol y_k)}`,
  eocancel: t`\begin{aligned}\prod_{k=1}^{K}\binom{n_c+\sum_{k'=k}^{K}|\mathcal T^{k'}|}{|\mathcal T^k|}^{-1}&=\frac{n_c!\prod_{k=1}^{K}|\mathcal T^k|!}{m!},\\[12pt]\mu_c^{n_c}\prod_k\mu_m(\boldsymbol y_k)^{|\mathcal T^k|}&=\mu_c^m\prod_k\left(\frac{\mu_m(\boldsymbol y_k)}{\mu_c}\right)^{|\mathcal T^k|}.\end{aligned}`,
  eounary: t`g(\boldsymbol y_k)=f(\boldsymbol y_k)e^{-\mu_m(\boldsymbol y_k)}`,
  eoedge: t`g_{ki}(\boldsymbol y_k,b_i;\boldsymbol z_i)=\begin{cases}\dfrac{\mu_m(\boldsymbol y_k)f(\boldsymbol z_i\mid\boldsymbol y_k)}{\mu_c f_c(\boldsymbol z_i)},&b_i=k,\\[12pt]1,&b_i\ne k.\end{cases}`,
  eojoint: t`f(Y,\boldsymbol b\mid Z)\propto\prod_{k=1}^{K}\left[g(\boldsymbol y_k)\prod_{i=1}^{m}g_{ki}(\boldsymbol y_k,b_i;\boldsymbol z_i)\right]`,
  anchor: t`\begin{aligned}b_i&\in\{0,1,\ldots,\underline K+i\},\\[9pt]\bar r_k=1&\quad\Longleftrightarrow\quad b_k=\underline K+k,\\[9pt]\bar l_k&=\sum_{i=1}^{M}\mathbf1\{b_i=\underline K+k\}.\end{aligned}`,
  truncated: t`p(l\mid\boldsymbol x,\boldsymbol e,\ l\ge1)=\frac{\mu_m(\boldsymbol x,\boldsymbol e)^l}{l!}\frac{e^{-\mu_m(\boldsymbol x,\boldsymbol e)}}{1-e^{-\mu_m(\boldsymbol x,\boldsymbol e)}}`,
  legacycount: t`p(l\mid\boldsymbol y)=\begin{cases}\dfrac{\mu_m(\boldsymbol x,\boldsymbol e)^l}{l!}e^{-\mu_m(\boldsymbol x,\boldsymbol e)},&r=1,\\[12pt]\mathbf1\{l=0\},&r=0.\end{cases}`,
  auxiliaries: t`\begin{aligned}\tb y &: \text{newly detected object states},\\[7pt]\tb b &: \text{their measurement assignments},\\[7pt]\pi(k)&:\text{new PO }k\longmapsto\text{newly detected object}.\end{aligned}`,
  marginalize: t`\begin{aligned}f(\ob y,\boldsymbol b,\overline K,l_{\rm fa}\mid\ub y)=\sum_{\tb b,\boldsymbol\pi}\int f(\ob y,\tb y,\boldsymbol b,\tb b,\boldsymbol\pi,\overline K,l_{\rm fa}\mid\ub y)\,\mathrm d\tb y.\end{aligned}`,
  s4cancel: t`\frac{\overline K!\,l_{\rm fa}!\,\prod l_k!}{M!}\;\frac{\mu_n^{\overline K}e^{-\mu_n}}{\overline K!}\;\frac{\mu_{\rm fa}^{l_{\rm fa}}e^{-\mu_{\rm fa}}}{l_{\rm fa}!}\;\prod\frac1{l_k!}=\frac{\mu_n^{\overline K}\mu_{\rm fa}^{l_{\rm fa}}e^{-(\mu_n+\mu_{\rm fa})}}{M!}`,
  s4counts: t`\overline K=|\mathcal D_{\boldsymbol b}|,\qquad l_{\rm fa}=M-\sum_{k=1}^{\underline K}l_k-\sum_{k=1}^{M}\bar l_k`,
  conditionalprior: t`\begin{aligned}f(\ob y,\boldsymbol b,M\mid\ub y)={}&\frac{\mu_n^{|\mathcal D_{\boldsymbol b}|}\mu_{\rm fa}^{l_{\rm fa}}e^{-(\mu_n+\mu_{\rm fa})}}{M!}\\[9pt]&\times\prod_{k\in\mathcal D_{\boldsymbol b}}\left[\bar r_k f_n(\ob x_k,\ob e_k)\mu_m(\ob x_k,\ob e_k)^{\bar l_k}\frac{e^{-\mu_m(\ob x_k,\ob e_k)}}{1-e^{-\mu_m(\ob x_k,\ob e_k)}}\right]\\[9pt]&\times\prod_{k\notin\mathcal D_{\boldsymbol b}}\left[(1-\bar r_k)f_d(\ob x_k,\ob e_k)\mathbf1\{\bar l_k=0\}\right]\\[9pt]&\times\prod_{k\in\mathcal D_{\ub y}}\left[\mu_m(\ub x_k,\ub e_k)^{l_k}e^{-\mu_m(\ub x_k,\ub e_k)}\right]\prod_{k\notin\mathcal D_{\ub y}}\mathbf1\{l_k=0\}.\end{aligned}`,
  qlegacy: t`\underline q(\ub y)=\begin{cases}f^-(\ub y)e^{-\mu_m(\ub x,\ub e)},&\ub r=1,\\[8pt]f^-(\ub y),&\ub r=0.\end{cases}`,
  qnew: t`\bar q(\ob y)=\begin{cases}\mu_n f_n(\ob x,\ob e)\dfrac{e^{-\mu_m(\ob x,\ob e)}}{1-e^{-\mu_m(\ob x,\ob e)}},&\bar r=1,\\[12pt]f_d(\ob x,\ob e),&\bar r=0.\end{cases}`,
  ratio: t`\rho_{ki}=\frac{\mu_m(\boldsymbol x_k,\boldsymbol e_k)f(\boldsymbol z_i\mid\boldsymbol x_k,\boldsymbol e_k)}{\mu_{\rm fa}f_{\rm fa}(\boldsymbol z_i)}`,
  legacyedge: t`\underline g_{ki}(\ub y_k,b_i;\boldsymbol z_i)=\begin{cases}\rho_{ki},&b_i=k,\ \ub r_k=1,\\[7pt]1,&b_i\ne k,\ \ub r_k=1,\\[7pt]\mathbf1\{b_i\ne k\},&\ub r_k=0.\end{cases}`,
  anchorg: t`\bar g_k(\ob y_k,b_k;\boldsymbol z_k)=\begin{cases}\rho_{kk},&b_k=\underline K+k,\ \bar r_k=1,\\[7pt]0,&b_k\ne\underline K+k,\ \bar r_k=1,\\[7pt]\mathbf1\{b_k\ne\underline K+k\},&\bar r_k=0.\end{cases}`,
  laterh: t`\bar h_{ki}(\ob y_k,b_i;\boldsymbol z_i)=\begin{cases}\rho_{ki},&b_i=\underline K+k,\ \bar r_k=1,\\[7pt]1,&b_i\ne\underline K+k,\ \bar r_k=1,\\[7pt]\mathbf1\{b_i\ne\underline K+k\},&\bar r_k=0.\end{cases}\quad i>k`,
  final: t`\begin{aligned}f(\ub y,\ob y,\boldsymbol b\mid Z)\propto{}&\prod_{k=1}^{\underline K}\left[\underline q(\ub y_k)\prod_{i=1}^{M}\underline g_{ki}(\ub y_k,b_i;\boldsymbol z_i)\right]\\[16pt]&\times\prod_{k=1}^{M}\left[\bar q(\ob y_k)\bar g_k(\ob y_k,b_k;\boldsymbol z_k)\prod_{i=k+1}^{M}\bar h_{ki}(\ob y_k,b_i;\boldsymbol z_i)\right].\end{aligned}`
};

const references = {
  R1: {url:'https://doi.org/10.1109/JPROC.2018.2789427',title:'Meyer et al. · Message passing algorithms for scalable multitarget tracking · 2018'},
  R2: {url:'https://arxiv.org/abs/1604.00970',title:'Granström, Baum & Reuter · Extended Object Tracking: Introduction, Overview and Applications'},
  R3: {url:'https://arxiv.org/abs/2103.11279',title:'Meyer & Williams · Scalable Detection and Tracking of Geometric Extended Objects · 2021'}
};
const slides=[];
function add(id,chapter,title,left,eqs=[],extra={}) {
  slides.push({id,chapter,title,left,right:'',equations:eqs,kind:'normal',section:chapter.startsWith('S')?chapter.slice(0,2).toLowerCase():'technique',refs:[],notes:left.replace(/<[^>]+>/g,' '),...extra});
}
add('cover','Companion to the restored derivation','Joint PDF\nfactorization',
  t`<p>From a generative model to local factors.</p><p>Point and extended objects.<br>Known and unknown object counts.</p>`,[],
  {kind:'cover',right:t`<div class="scenario-grid"><div><b>S1</b>Point · known</div><div><b>S2</b>Point · unknown</div><div><b>S3</b>Extended · known</div><div><b>S4</b>Extended · unknown</div></div>`,notes:'Companion to the restored Joint PDF factorization article. Follow its four scenarios in order. The scope ends at the local factors and factor graphs; the fixed-grouping and explicit BP-update chapters from the reviewed deck are outside this presentation.'});
add('roadmap','The four scenarios','Two modeling choices shape the graph.',
  t`<p><strong>Object model:</strong> one measurement or several per scan.</p><p><strong>Object count:</strong> fixed, or represented by potential objects and existence variables.</p>`,[],
  {right:t`<table class="case-table"><thead><tr><th></th><th>Known count</th><th>Unknown count</th></tr></thead><tbody><tr><th>Point</th><td><a href="#s1-association">S1 · exclusivity</a></td><td><a href="#s2-existence">S2 · existence + new objects</a></td></tr><tr><th>Extended</th><td><a href="#s3-association">S3 · Poisson counts</a></td><td><a href="#s4-anchor">S4 · anchored new objects</a></td></tr></tbody></table>`,notes:'Each matrix entry jumps to the first slide of that scenario. The article uses n_t point targets, K known extended objects, and underlined K legacy potential objects in Scenario 4.'});
add('recipe','Common technique','Prior × allocation × measurements',
  t`<ol><li>Read the counts from the association.</li><li>Insert the Poisson count laws.</li><li>Cancel allocation factorials.</li><li>Collect constants; marginalize auxiliaries.</li></ol>`, ['chain'],
  {right:t`<p class="small">$S$ collects object states; $A$ collects associations and any existence variables needed for the allocation.</p>`,notes:'Measurements are an ordered vector with observed length m. Retain m in the count/allocation kernel before conditioning on the observed data. The Poisson allocation expression is a joint count-and-association probability, not a normalized distribution of associations at a fixed count. This convention makes the restored article’s final factorizations precise.'});
add('notation','Notation and model boundary','What is observed, and what is inferred?',
  t`<p><strong>Observed:</strong> the ordered measurement vector $Z$ and its count.</p><p><strong>Inferred:</strong> states, associations, and existence when applicable.</p><p>Clutter parameters and measurement models are supplied.</p>`, ['state','constant'],
  {right:t`<p class="small">For fixed data and clutter model, $C(Z,m)$ is constant in the inferred states and assignments.</p>`,notes:'Use the restored article’s notation throughout: P_d for point detection, mu_m for the state-dependent EO measurement rate, and mu_c f_c for clutter intensity (mu_fa f_fa in S4). Final products are displayed as posteriors conditioned on Z, including its observed count. The independent predicted-prior assumption is essential.'});
add('s1-association','S1 · Point / known count','Each target makes at most one claim.',
  t`<p>$n_t$ existing point targets; $m$ measurements.</p><p>A nonzero claim must be unique. Several targets may have the null claim.</p>`, ['association','psi','s1counts'],
  {refs:['R1'],notes:'A valid example is a=(1,3,0): two detections and one miss. The assignment (1,1,0) is invalid because two targets claim measurement 1. Clutter is the complement of the claimed measurements.'});
add('s1-likelihood','S1 · Likelihood','Divide by the clutter explanation.',
  t`<p>Claimed measurements use the target likelihood.</p><p>Unclaimed measurements use $f_c$.</p><p>Multiplying and dividing by clutter densities leaves a ratio for each detection.</p>`, ['likelihood'],
  {refs:['R1'],notes:'This algebra assumes a positive clutter density on the measurement support. The product over all observed clutter densities is independent of object states and assignments and can be absorbed into the posterior normalizer.'});
add('s1-counts','S1 · Count cancellation','A permutation removes the clutter factorial.',
  t`<p>Given the detected targets, all consistent assignments to distinct measurement slots are uniform.</p><p>The clutter Poisson factorial cancels the allocation factorial.</p>`, ['s1cancel'],
  {right:t`<p class="small">One factor $1/\mu_c$ remains for each detected target.</p>`,refs:['R1'],notes:'The association/count kernel also contains psi(a), a P_d factor per detection, and a (1-P_d) factor per missed target. Its sum over a is the measurement-count probability, not necessarily one.'});
add('s1-factors','S1 · Local factors','Detection ratios and missed detections',
  t`<p>Each target contributes its prior and one local factor $g$.</p><p>The consistency factor couples their association choices.</p>`, ['pointg','pointjoint'],
  {refs:['R1'],notes:'This is the Scenario 1 result of the restored page, with the fixed measurement vector moved to the conditioning side. Unknown object states are not replaced by point estimates.'});
add('s1-graph','S1 · Factor graph','One chain per target; one shared constraint.',
  t`<p>Prior → state → local likelihood → association.</p><p>$\Psi$ forbids duplicate nonzero claims.</p>`, [],
  {kind:'graph-wide',figure:'fg-point-known.png',caption:'Original Scenario 1 factor graph from the restored article.',refs:['R1'],notes:'Read each horizontal chain left to right. The single global consistency factor touches every a_i. This diagram displays a factorization; deriving or running the message updates is a subsequent inference step.'});
add('s2-existence','S2 · Point / unknown count','Augment each state with existence.',
  t`<p>Legacy potential objects carry $\ub y_i=(\ub x_i,\ub r_i)$.</p><p>An existing object can be detected or missed. A nonexistent object cannot claim a measurement.</p>`, ['legacyg'],
  {refs:['R1'],notes:'The legacy prior f(underlined y_i) carries existence mass. Its nonexistent branch uses a normalized dummy density. A null association alone does not tell us whether the object exists.'});
add('s2-counts','S2 · Count cancellation','Split unclaimed points into new objects and clutter.',
  t`<p>Each measurement has a new potential object $\ob y_j$.</p><p>On the valid support, new-object and clutter counts follow directly from existence and association.</p>`, ['s2counts','s2cancel'],
  {refs:['R1'],notes:'Multiply the inverse binomial split and inverse permutation by the Poisson PMFs for clutter and newly detected objects. Both factorials cancel. In this measurement-anchored convention, mu_n f_n describes newly detected objects; using an intensity of all physical births instead would require the corresponding detection model.'});
add('s2-birth','S2 · New-object factor','A claimed point cannot also start a new object.',
  t`<p>$\mathcal T$ contains points already claimed by legacy objects.</p><p>A valid new-object branch needs both the birth intensity $\mu_n$ and state density $f_n$.</p>`, ['newg'],
  {refs:['R1'],notes:'The bar r=0 branch is the dummy density regardless of whether a legacy object claimed this point. If the point is unclaimed and bar r=0, it is clutter. A new object with bar r=1 must use an unclaimed point.'});
add('s2-result','S2 · Factorized posterior','Legacy chains plus new-object factors',
  t`<p>The same $\psi(\boldsymbol a)$ enforces unique legacy claims.</p><p>Each new-object factor depends on the entire legacy association vector.</p>`, ['s2joint'],
  {refs:['R1'],notes:'This is the restored Scenario 2 final factorization. There is one new potential object per measurement, not an assertion that every measurement creates a confirmed track.'});
add('s2-graph','S2 · Factor graph','New candidates share the association decisions.',
  t`<p>Legacy chains remain on the left.</p><p>New candidates add $\bar g$ factors.</p><p>The colored edges prevent a legacy claim and a new-object claim from explaining the same point.</p>`, [],
  {figure:'fg-point-unknown.png',caption:'Original Scenario 2 graph.',refs:['R1'],notes:'The bar g factors are functions of the association vector. Existence variables stay part of the object state; measurements are observed parameters of the likelihood factors.'});
add('s3-association','S3 · Extended / known count','Let every measurement name its source.',
  t`<p>The extended state is $\boldsymbol y=(\boldsymbol x,\boldsymbol e)$.</p><p>Several measurements can share one object. Each measurement has exactly one source; $b_i=0$ means clutter.</p>`, ['eoassociation','poiss'],
  {refs:['R2','R3'],notes:'T^k is the measurement cell induced by an association hypothesis. It is not a grouping supplied in advance. The count is conditionally Poisson at each object state, including the possibility of zero. No global point-target exclusivity factor is needed.'});
add('s3-counts','S3 · Count cancellation','The allocation binomials telescope.',
  t`<p>Each object selects its cell from the remaining measurements.</p><p>The allocation contributes a factorial for each cell, cancelling the Poisson factorials.</p>`, ['eocancel'],
  {refs:['R2','R3'],notes:'The total count includes object-generated measurements and clutter. Keep all state-dependent exponentials. The only count prefactor removed after conditioning on data is the clutter baseline C(Z,m).'});
add('s3-factors','S3 · Local factors','One unary factor, then one edge per point',
  t`<p>The zero-count exponential belongs to the object’s unary factor.</p><p>An assigned point contributes a measurement-rate likelihood ratio. Every other edge is neutral.</p>`, ['eounary','eoedge'],
  {refs:['R2','R3'],notes:'The exponential exp(-mu_m(y_k)) appears once per object, including when its cell is empty. The pure Poisson model already permits zero detections. A separate object-level P_D gate requires a different empty-cell term and a consistent nonempty branch.'});
add('s3-result','S3 · Factorized posterior','Put all edges inside the object product.',
  t`<p>Each object contributes its unary and all $m$ pairwise factors.</p><p>All objects meet at the measurement-oriented association variables.</p>`, ['eojoint'],
  {refs:['R3'],notes:'This is the bracketed Scenario 3 result in the restored source. A hypothesis b_i=k activates the kth object’s ratio for measurement i; the other objects’ factors for that measurement are one.'});
add('s3-graph','S3 · Factor graph','Many measurements may connect to the same object.',
  t`<p>Each object–measurement pair has a factor $g_{ki}$.</p><p>Variable $b_i$ selects the source of measurement $i$.</p><p>The source choice itself gives each measurement one owner.</p>`, [],
  {figure:'fg-eo-known.png',caption:'Original Scenario 3 graph.',refs:['R3'],notes:'The graph has one object-state variable per known extended object and one association variable per measurement. The unary incorporates both the independent prior and the object-level Poisson exponential.'});
add('s4-anchor','S4 · Extended / unknown count','Anchor each new object to its first point.',
  t`<p>$\underline K$ legacy potential objects and $M$ new potential objects.</p><p>New candidate $k$ is anchored at measurement $k$ and can explain later measurements.</p>`, ['anchor'],
  {refs:['R3'],notes:'First means smallest index in the chosen ordered measurement vector, not physical time. New candidate k exists exactly when b_k=underlined K+k. For K=0, (1,1,1) is valid; (0,1,1) violates the anchor condition even though every entry obeys the domain bound.'});
add('s4-counts','S4 · Measurement counts','A newly detected object has a nonzero count.',
  t`<p>Newly detected objects use a zero-truncated Poisson count.</p><p>Existing legacy objects use an ordinary Poisson count; nonexistent ones generate zero points.</p>`, ['truncated','legacycount'],
  {refs:['R3'],notes:'Here mu_n f_n is the intensity of newly detected objects. The truncated expression is defined at a positive measurement rate. Keep its state-dependent denominator under this birth convention.'});
add('s4-auxiliary','S4 · Auxiliary construction','Expand first, then remove the auxiliaries.',
  t`<ol><li>Introduce newly detected states and their associations.</li><li>Map them onto measurement-anchored candidates.</li><li>Integrate copied states; sum auxiliary labels and mappings.</li></ol>`, ['auxiliaries'],
  {refs:['R3'],notes:'This follows the restored page’s three-step auxiliary construction. State-copy delta functions transfer tilde states to the corresponding bar states. The deterministic mapping indicator remains inside the joint kernel; do not treat a mapping-restricted kernel as a normalized conditional without its normalizer.'});
add('s4-cancellation','S4 · Marginalization','Relabeling cancels the new-object factorial.',
  t`<p>Permuting the auxiliary labels of $\overline K$ newly detected objects gives the same canonical assignment.</p><p>The sum contributes $\overline K!$.</p>`, ['s4cancel','s4counts'],
  {kind:'wide-equation',refs:['R3'],notes:'The displayed product of l_k factorials runs over all legacy and newly detected objects. The state densities, measurement-rate powers, exponentials and zero-truncation denominators remain in the state-dependent factors; only the combinatorial cancellation is shown here.'});
add('s4-prior','S4 · Conditional-prior result','The count model is complete; values come next.',
  t`<p>$\mathcal D_{\boldsymbol b}$: existing new candidates. $\mathcal D_{\ub y}$: existing legacy objects.</p>`, ['conditionalprior'],
  {kind:'full-equation',refs:['R3'],notes:'This is the restored page’s conditional prior f(bar y,b,M | underlined y). It retains the count M. The spatial measurement values Z have not entered yet. The indicator 1{l=0} is the same discrete delta used in the article.'});
add('s4-unaries','S4 · Unary potentials','Separate legacy prediction and new-object evidence.',
  t`<p>$f^-(\ub y)$ includes the legacy existence mass and its conditional state density.</p><p>The new-object unary carries the detected-object intensity and the truncated-count normalization.</p>`, ['qlegacy','qnew'],
  {refs:['R3'],notes:'These are the restored source’s single-scan unaries. For a nonexistent legacy object retain f^-(underlined y), not the scalar one. The paper’s multi-scan pseudo-transition is a different object from this already-predicted single-scan unary.'});
add('s4-legacy-edge','S4 · Legacy measurement edges','An absent object cannot own a point.',
  t`<p>Write the rate-weighted likelihood ratio as $\rho_{ki}$.</p><p>Assignment to an existing object activates that ratio; an absent object forbids the assignment.</p>`, ['ratio','legacyedge'],
  {refs:['R3'],notes:'Rho is only a display abbreviation, not a new model parameter. For a legacy edge evaluate it at the underlined state; for a new edge evaluate it at the bar state. The denominator is the source’s false-alarm intensity mu_fa f_fa.'});
add('s4-anchor-edge','S4 · New-object anchor','The anchor enforces existence in both directions.',
  t`<p>If candidate $k$ exists, measurement $k$ must belong to it.</p><p>If it does not exist, the anchor cannot name it.</p>`, ['anchorg'],
  {refs:['R3'],notes:'Here rho_kk is evaluated at bar y_k for measurement k. The zero branch for an existing candidate whose anchor is assigned elsewhere is essential. A domain bound alone does not enforce this rule.'});
add('s4-later-edge','S4 · Later measurements','An extended new object needs more than its anchor.',
  t`<p>For every $i>k$, candidate $k$ can explain measurement $i$ through $\bar h_{ki}$.</p><p>Earlier measurements cannot refer to a later candidate.</p>`, ['laterh'],
  {refs:['R3'],notes:'Rho_ki is evaluated at bar y_k for measurement i. This factor is neutral if the existing candidate does not own i. If the candidate does not exist, all claims to it are excluded. The original source explicitly includes these later-measurement factors.'});
add('s4-result','S4 · Final factorization','Two families of local products',
  t`<p>Legacy objects connect to every measurement. New candidates connect to their anchor and all later measurements.</p>`, ['final'],
  {kind:'full-equation',refs:['R3'],notes:'This is the complete single-scan posterior factorization in the restored article. Each new candidate contributes bar q, one special anchor bar g, and ordinary later factors bar h. Source variables enforce one source per point while permitting multiple points per object.'});
add('s4-graph','S4 · Factor graph','Prediction, birth, and measurement edges',
  t`<p>$\underline q$: legacy prediction.</p><p>$\bar q$: new-object unary.</p><p>The anchor and later edges encode the canonical labeling.</p><p>Red annotations identify BP messages on this graph.</p>`, [],
  {figure:'fg-eo-unknown.png',caption:'Original Scenario 4 graph. Its h / ḡ labels follow the source figure.',refs:['R3'],notes:'The graph reproduced from the restored article includes message annotations. In this deck underline g names legacy pairwise factors and bar h names later-new-object factors, matching the article’s final equations. The source figure uses the paper’s h/g-bar labels. No new explicit BP iteration scheme is introduced here.'});
add('summary','Summary','Two changes, two modeling tools.',
  t`<p><strong>Unknown object count:</strong> introduce potential objects and existence.</p><p><strong>Multiple points per object:</strong> use measurement-oriented assignments and Poisson counts.</p>`, [],
  {section:'summary',right:t`<table class="case-table"><thead><tr><th>Case</th><th>Association</th><th>Coupling</th></tr></thead><tbody><tr><td>S1</td><td>Target-oriented</td><td>Global exclusivity</td></tr><tr><td>S2</td><td>Target-oriented</td><td>Exclusivity + new objects</td></tr><tr><td>S3</td><td>Measurement-oriented</td><td>Object–point factors</td></tr><tr><td>S4</td><td>Measurement-oriented</td><td>Legacy + anchor + later edges</td></tr></tbody></table>`,notes:'A measurement cell in S3 or S4 is induced by the unknown association. This presentation does not impose an externally fixed grouping. The four cases share the same prior/allocation/likelihood decomposition.'});
add('checks','Before implementation','Check the tiny cases first.',
  t`<ol><li>Keep ordered-vector and finite-set conventions consistent.</li><li>Retain every state-dependent factor.</li><li>Verify counts and normalization by enumeration.</li></ol>`, [],
  {section:'next',right:t`<ol><li>Keep $\mu_n f_n$ in new-object factors.</li><li>Enforce the anchor condition and include every later edge.</li><li>Compare approximate BP marginals against exact inference on small scans.</li></ol>`,notes:'These are the implementation checks at the end of the restored page. Count evidence, birth conventions, and association support should be checked before using the graph in a tracking implementation.'});
add('references','References','The page, the scans, and the primary papers',
  t`<p><strong>R1 · Point-object models</strong><br>Meyer et al., “Message passing algorithms for scalable multitarget tracking,” Proceedings of the IEEE, 2018.</p><p><strong>R2 · Extended-object model</strong><br>Granström, Baum &amp; Reuter, “Extended Object Tracking: Introduction, Overview and Applications.”</p>`, [],
  {kind:'references',section:'next',right:t`<p><strong>R3 · Extended objects with uncertain count</strong><br>Meyer &amp; Williams, “Scalable Detection and Tracking of Geometric Extended Objects,” 2021.</p><div class="callout">The companion page contains the full derivations and original handwritten scans for all four cases.</div>`,refs:['R1','R2','R3'],notes:'This presentation follows the restored Joint PDF factorization page. The original figures are embedded from its assets. Primary papers provide the published model context; handwritten scans are supplementary cross-checks.'});

const appendix = require('./appendix.cjs');
Object.assign(equations, appendix.equations);
references.R4 = appendix.reference;
const grbpIndex = slides.findIndex(slide => slide.id === 'summary');
slides.splice(grbpIndex, 0, ...appendix.slides);
const cover = slides.find(slide => slide.id === 'cover');
cover.chapter = 'Four scenarios + GrBP · Joint PDF derivations';
cover.left += '<p><strong>GrBP:</strong> Precluster, then associate whole groups.</p>';
cover.notes = 'Follow the four raw-measurement scenarios, then GrBP from Appendix I of EO_Writing_Long_Version. The original S1–S4 slides are preserved. GrBP derives the approximate joint PDF after a fixed external partition.';
const roadmap = slides.find(slide => slide.id === 'roadmap');
roadmap.chapter = 'The four scenarios and a GrBP derivation';
roadmap.right += '<div class="callout"><a href="#grbp-preclustering"><strong>GrBP · Preclustering → group association</strong></a><br>Appendix I of the EO manuscript</div>';
roadmap.notes += ' GrBP follows S4 and replaces raw-point assignments with whole-group assignments after a fixed preclustering step.';
const summary = slides.find(slide => slide.id === 'summary');
summary.title = 'From raw measurements to fixed groups';
summary.left += '<p><strong>GrBP:</strong> precluster once, then associate whole groups.</p>';
summary.right = summary.right.replace('</tbody>', '<tr><td>GrBP</td><td>Group-level</td><td>Exclusive target–group factors</td></tr></tbody>');
summary.notes = 'S1–S4 retain raw-measurement assignments. GrBP conditions on one externally supplied partition and uses the appendix’s approximate Poisson clutter-group model. The birth factors and count conventions must remain consistent within each model.';
const referencesSlide = slides.find(slide => slide.id === 'references');
referencesSlide.right = referencesSlide.right.replace('<div class="callout">', '<p><strong>R4 · Preclustering and group association</strong><br>EO_Writing_Long_Version, Appendix I, Eqs. (A.1)–(A.9).</p><div class="callout">').replace('for all four cases.', 'for S1–S4, plus the Appendix I derivation for GrBP.');
referencesSlide.refs.push('R4');
referencesSlide.notes += ' R4 is pinned to manuscript commit d0a380a; the archived Appendix I source was verified byte for byte.';

module.exports={title:'Joint PDF factorization · companion slides',equations,references,slides};
