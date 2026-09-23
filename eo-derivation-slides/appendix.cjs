'use strict';

// Appendix I, verified against EO_Writing_Long_Version at d0a380a9caf81a36b88886efa91b5afd9bb7bdb5.
// Reuse the archived, source-checked equations without restoring the superseded deck.
const t = String.raw;
const sourceURL = 'https://github.com/BaiLiping/EO_Writing_Long_Version/blob/d0a380a9caf81a36b88886efa91b5afd9bb7bdb5/sections/sec_appendixI.tex';
const equations = {
  "pc_setup": "\\boldsymbol Z^G=\\{\\boldsymbol Z^1,\\ldots,\\boldsymbol Z^{n^g}\\},\\qquad m=\\sum_{j=1}^{n^g}|\\boldsymbol Z^j|.",
  "pc_prior": "f(\\underline{\\boldsymbol Y})=\\prod_{p=1}^{n^t}f(\\underline{\\boldsymbol y}^{p}),\\qquad \\underline{\\boldsymbol y}^{p}=(\\underline s^p,\\underline r^p),\\quad s=(\\boldsymbol x,\\boldsymbol E).",
  "pc_choices": "a^p\\in\\{0,\\ldots,n^g\\},\\qquad b^j\\in\\{0,\\ldots,n^t\\},\\qquad a^p=j\\ \\Longleftrightarrow\\ b^j=p.",
  "pc_psi": "\\begin{aligned}\\boldsymbol\\psi(\\boldsymbol a,\\boldsymbol b)&=\\prod_{p=1}^{n^t}\\prod_{j=1}^{n^g}\\Psi_{p,j}(a^p,b^j),\\\\ \\Psi_{p,j}(a,b)&=\\mathbf1[(a=j)\\Longleftrightarrow(b=p)].\\end{aligned}",
  "pc_sets": "\\begin{aligned}\\mathbb D&=\\{p:a^p\\ne0\\},\\qquad\\mathbb N=\\{j:\\overline r^j=1\\},\\\\\\mathbb C&=\\{1,\\ldots,n^g\\}\\setminus\\bigl(\\{a^p:p\\in\\mathbb D\\}\\cup\\mathbb N\\bigr),\\\\ n^d&=|\\mathbb D|,\\quad n^n=|\\mathbb N|,\\quad n^c=n^g-n^d-n^n.\\end{aligned}",
  "pc_chain": "\\begin{aligned}f(\\underline{\\boldsymbol Y},\\overline{\\boldsymbol Y},\\boldsymbol a,\\boldsymbol b,\\boldsymbol Z^G)={}&f(\\underline{\\boldsymbol Y})\\,\\boldsymbol\\psi(\\boldsymbol a,\\boldsymbol b)\\\\&\\times f(\\boldsymbol a,\\boldsymbol b,\\overline{\\boldsymbol Y}\\mid\\underline{\\boldsymbol Y})\\\\&\\times f(\\boldsymbol Z^G\\mid\\underline{\\boldsymbol Y},\\overline{\\boldsymbol Y},\\boldsymbol a,\\boldsymbol b).\\end{aligned}",
  "pc_poisson": "f(n^c)\\approx\\frac{\\mu_g^{n^c}e^{-\\mu_g}}{n^c!},\\qquad f(n^n)=\\frac{\\mu_n^{n^n}e^{-\\mu_n}}{n^n!}.",
  "pc_cancel": "\\begin{aligned}\\binom{n^c+n^n}{n^n}^{-1}&=\\frac{n^c!n^n!}{(n^c+n^n)!},\\\\\\frac1{{}_{n^g}P_{n^d}}&=\\frac{(n^c+n^n)!}{n^g!},\\\\\\text{product}&=\\frac{n^c!n^n!}{n^g!}.\\end{aligned}",
  "pc_birthstates": "B(\\overline{\\mathcal X}\\mid\\overline{\\boldsymbol r})=\\prod_{j\\in\\mathbb N}f_n(\\overline s^j)\\prod_{j\\notin\\mathbb N}f_d(\\overline s^j),\\qquad\\int f_d(s)\\,ds=1.",
  "pc_allocation": "\\begin{aligned}f(\\boldsymbol a,\\boldsymbol b,\\overline{\\boldsymbol Y}\\mid\\underline{\\boldsymbol Y})\\approx{}&\\frac{\\mu_g^{n^c}e^{-\\mu_g}\\mu_n^{n^n}e^{-\\mu_n}}{n^g!}\\,B(\\overline{\\mathcal X}\\mid\\overline{\\boldsymbol r})\\\\ &\\times\\prod_{p\\in\\mathbb D}\\underline r^p\\\\ &\\times\\prod_{p\\notin\\mathbb D}\\left[1-\\underline r^p\\left(1-e^{-\\mu_m(\\underline s^p)}\\right)\\right].\\end{aligned}",
  "pc_Llegacy": "L_{\\mathrm L}(\\boldsymbol Z^j\\mid s)=e^{-\\mu_m(s)}\\prod_{\\boldsymbol z\\in\\boldsymbol Z^j}\\mu_m(s)f(\\boldsymbol z\\mid s).",
  "pc_Lnew": "L_{\\mathrm N}(\\boldsymbol Z^j\\mid s)=\\frac{L_{\\mathrm L}(\\boldsymbol Z^j\\mid s)}{1-e^{-\\mu_m(s)}},\\qquad\\boldsymbol Z^j\\ne\\varnothing.",
  "pc_Fclutter": "F_c(\\boldsymbol Z^j)=\\prod_{\\boldsymbol z\\in\\boldsymbol Z^j}f_c(\\boldsymbol z),\\qquad\\prod_{j=1}^{n^g}F_c(\\boldsymbol Z^j)=\\prod_{i=1}^{m}f_c(\\boldsymbol z^i).",
  "pc_likelihood": "\\begin{aligned}f(\\boldsymbol Z^G\\mid\\underline{\\boldsymbol Y},\\overline{\\boldsymbol Y},\\boldsymbol a,\\boldsymbol b)={}&\\prod_{p\\in\\mathbb D}L_{\\mathrm L}(\\boldsymbol Z^{a^p}\\mid\\underline s^p)\\\\&\\times\\prod_{j\\in\\mathbb N}L_{\\mathrm N}(\\boldsymbol Z^j\\mid\\overline s^j)\\prod_{c\\in\\mathbb C}F_c(\\boldsymbol Z^c).\\end{aligned}",
  "pc_R": "R_j(s)=\\prod_{\\boldsymbol z\\in\\boldsymbol Z^j}\\frac{\\mu_m(s)f(\\boldsymbol z\\mid s)}{f_c(\\boldsymbol z)}.",
  "pc_mu_redistribute": "\\mu_g^{n^c}=\\mu_g^{n^g}\\prod_{p\\in\\mathbb D}\\mu_g^{-1}\\prod_{j\\in\\mathbb N}\\mu_g^{-1}.",
  "pc_C": "C(\\mu_g,\\mu_n,\\boldsymbol Z^G)=\\frac{\\mu_g^{n^g}e^{-\\mu_g}e^{-\\mu_n}}{n^g!}\\prod_{i=1}^{m}f_c(\\boldsymbol z^i).",
  "pc_legacy": "\\underline l((s,r),a;\\boldsymbol Z^G)=\\begin{cases}\\dfrac{e^{-\\mu_m(s)}}{\\mu_g}R_a(s),&a\\ne0,\\ r=1,\\\\ e^{-\\mu_m(s)},&a=0,\\ r=1,\\\\ 1,&a=0,\\ r=0,\\\\ 0,&a\\ne0,\\ r=0.\\end{cases}",
  "pc_new": "\\overline l((s,r),b;\\boldsymbol Z^j)=\\begin{cases}\\dfrac{\\mu_n f_n(s)e^{-\\mu_m(s)}}{\\mu_g[1-e^{-\\mu_m(s)}]}R_j(s),&b=0,\\ r=1,\\\\0,&b\\ne0,\\ r=1,\\\\ f_d(s),&r=0.\\end{cases}",
  "pc_final": "\\begin{aligned}f(\\underline{\\boldsymbol Y},\\overline{\\boldsymbol Y},\\boldsymbol a,\\boldsymbol b,\\boldsymbol Z^G)\\approx{}&C\\,\\boldsymbol\\psi(\\boldsymbol a,\\boldsymbol b)\\\\ &\\times\\prod_{p=1}^{n^t}\\left[f(\\underline{\\boldsymbol y}^p)\\underline l(\\underline{\\boldsymbol y}^p,a^p;\\boldsymbol Z^G)\\right]\\\\&\\times\\prod_{j=1}^{n^g}\\overline l(\\overline{\\boldsymbol y}^j,b^j;\\boldsymbol Z^j).\\end{aligned}"
};
// Line breaks only: the symbols and factors are unchanged from the source transcription.
equations.pc_prior = t`\begin{aligned}s&=(\boldsymbol x,\boldsymbol E),\qquad \underline{\boldsymbol y}^p=(\underline s^p,\underline r^p),\\[8pt]f(\underline{\boldsymbol Y})&=\prod_{p=1}^{n^t}f(\underline{\boldsymbol y}^p).\end{aligned}`;
equations.pc_choices = t`\begin{aligned}a^p&\in\{0,\ldots,n^g\},\qquad b^j\in\{0,\ldots,n^t\},\\[8pt]a^p=j&\quad\Longleftrightarrow\quad b^j=p.\end{aligned}`;
equations.pc_poisson = t`\begin{aligned}f(n^c)&\approx\frac{\mu_g^{n^c}e^{-\mu_g}}{n^c!},\\[14pt]f(n^n)&=\frac{\mu_n^{n^n}e^{-\mu_n}}{n^n!}.\end{aligned}`;
equations.pc_birthstates = t`B(\overline{\mathcal X}\mid\overline{\boldsymbol r})=\prod_{j\in\mathbb N}f_n(\overline s^j)\prod_{j\notin\mathbb N}f_d(\overline s^j)`;
equations.pc_Fclutter = t`\begin{aligned}F_c(\boldsymbol Z^j)&=\prod_{\boldsymbol z\in\boldsymbol Z^j}f_c(\boldsymbol z),\\[12pt]\prod_{j=1}^{n^g}F_c(\boldsymbol Z^j)&=\prod_{i=1}^{m}f_c(\boldsymbol z^i).\end{aligned}`;
equations.pc_legacy = equations.pc_legacy.replaceAll('\\\\', '\\\\[9pt]');
equations.pc_new = equations.pc_new.replaceAll('\\\\', '\\\\[12pt]');

const slides=[];
function add(id,label,title,left,eqs=[],extra={}) {
  slides.push({id:'grbp-'+id,chapter:'Derivation for GrBP / '+label,title,left,
    right:'',equations:eqs,kind:'normal',section:'grbp',refs:['R4'],
    notes:'Appendix I of EO_Writing_Long_Version, commit d0a380a. '+label+'. '+left.replace(/<[^>]+>/g,' '),...extra});
}
add('preclustering','Fixed partition','Derivation for GrBP',
  t`<p>GrBP uses grouped measurements: external clustering fixes one partition.</p><p>Data association then decides which target explains each <strong>whole group</strong>.</p><p>Introduce one newborn candidate per group.</p>`,[],
  {right:t`<div class="grouping-pipeline"><div><span>01 · OBSERVE</span>Raw detections $\boldsymbol Z$</div><b aria-hidden="true">↓</b><div><span>02 · PRECLUSTER</span>Fixed partition $\mathcal P=\mathcal C(\boldsymbol Z)$</div><b aria-hidden="true">↓</b><div><span>03 · ASSOCIATE</span>Groups $\boldsymbol Z^G$ → target choices</div></div>`,
  notes:'Appendix I introduction and footnote to A.2. This is the requested alternative after S4. S4 infers individual measurement assignments; the GrBP derivation conditions on an externally supplied partition. The source calls this an approximate conditional model. It does not model the probability that clustering produced the partition.'});
add('notation','A.1','Points, groups, and targets are different counts.',
  t`<p>$m$ raw detections form $n^g$ disjoint groups.</p><p>$n^t$ legacy components carry predicted beliefs.</p><p>Every component has kinematics, extent, and existence; $s=(\boldsymbol x,\boldsymbol E)$ is shorthand.</p>`,['pc_setup','pc_prior'],
  {notes:'Appendix I.A and Eq. (A.1). The legacy prior factors under independent predicted component beliefs. There are n^t+n^g potential components, not necessarily that many actual targets. A single partition may contain many groups.'});
add('association','A · Association','Two variables record the same claim.',
  t`<p>$a^p=j$: legacy target $p$ claims group $j$.</p><p>$b^j=p$: group $j$ comes from legacy target $p$.</p><p>$b^j=0$ leaves two alternatives: <strong>newborn or clutter</strong>.</p>`,['pc_choices','pc_psi'],
  {notes:'Appendix I.A. Psi_pj is zero exactly when one of a^p=j and b^j=p holds and the other does not. Their product enforces one-to-one legacy-to-group claims. Newborn exclusion is additionally enforced by the newborn factor.'});
add('counts','A · Count identities','Read the counts from the assignments.',
  t`<p>Detected legacy targets, existing newborns, and clutter partition the groups on every feasible hypothesis.</p>`,['pc_sets'],{kind:'wide-equation',
  notes:'Appendix I.A, final paragraph. D contains detected legacy indices; N contains group-indexed existing newborns. Legacy-claimed and newborn groups must be disjoint. On that support, n^c=n^g-n^d-n^n.'});
add('chain','A.2','Separate the prior, allocation, and likelihood.',
  t`<p>Condition throughout on the fixed partition $\mathcal P$. Its notation is suppressed in the source.</p>`,['pc_chain'],{kind:'wide-equation',
  notes:'Eq. (A.2), eq:joint_pdf_structured. This is the source’s chain-rule decomposition on the valid association support. Subsequent group-count and conditional-independence approximations enter its factors. The clustering-output probability is not modeled.'});
add('poisson','A.3','Clutter groups have their own Poisson mean.',
  t`<p>$\mu_g$: mean number of clutter <strong>groups</strong>.</p><p>$\mu_n$: mean number of <strong>detected newborns</strong>.</p><p>Generally $\mu_g\ne\mu_c$: a clutter group can contain several detections.</p>`,['pc_poisson'],
  {notes:'Eq. (A.3), eq:pt_counts_distributions. The induced clutter-group count is approximated by a Poisson distribution. Preserve the approximation sign and do not replace mu_g with the raw clutter-detection mean.'});
add('cancel','A.3 → A.4','The allocation factorials cancel the count factorials.',
  t`<p>Choose which unclaimed groups are newborns.</p><p>Assign detected legacy targets to distinct group slots.</p><p>Multiply by the two Poisson count laws; the remaining denominator is $n^g!$.</p>`,['pc_cancel'],
  {notes:'Paragraph after A.3. The inverse binomial is n^c! n^n!/(n^c+n^n)!. The inverse permutation is (n^c+n^n)!/n^g!, because n^g-n^d=n^c+n^n. Their product cancels both Poisson count factorials.'});
add('allocation','A.4','Keep the existence branches and newborn densities.',
  t`<p>$B$ collects newborn state densities; $f_d$ is normalized. The nonempty legacy count mass enters the likelihood next.</p>`,['pc_allocation','pc_birthstates'],{kind:'full-equation',
  notes:'Eq. (A.4), eq:I13. B is only an abbreviation for the source’s product of f_n for existing newborns and f_d for absent candidates. A detected legacy branch contributes its existence indicator here; the e^-mu_m times product(mu_m f) term enters A.5. Read these terms together as the displayed joint kernel, not as a separately normalized association PMF. Do not insert another detection-probability multiplier.'});
add('group-likelihood','A.5 · Group models','A detected newborn uses a nonempty-group likelihood.',
  t`<p>A legacy group uses the extended-target Poisson likelihood.</p><p>A newborn is detected by definition, so condition on a nonzero count.</p><p>Keep one exponential per target, not one per point.</p>`,['pc_Llegacy','pc_Lnew'],
  {notes:'Paragraph following A.5. L_L and L_N are explicit display abbreviations for the source likelihoods. The zero-truncation denominator 1-exp(-mu_m(s)) belongs with the detected-newborn convention mu_n f_n. The group is nonempty and the truncated form uses positive detection probability.'});
add('likelihood','A.5','Multiply the group likelihoods by source.',
  t`<p>Conditioned on a feasible association, the approximate model treats groups as independent.</p>`,['pc_likelihood'],{kind:'wide-equation',
  notes:'Eq. (A.5), eq:pt_group_like_count_split. Multiply the legacy likelihoods over D, newborn likelihoods over N, and clutter densities F_c over C. Conditional group independence is a model assumption after clustering.'});
add('clutter','A.5 → A.6','Divide each target group by its clutter explanation.',
  t`<p>The product of all group clutter densities is a baseline for the entire scan.</p><p>Redistributing $\mu_g^{n^c}$ leaves <strong>one</strong> $1/\mu_g$ for every target-generated group.</p>`,['pc_Fclutter','pc_mu_redistribute'],
  {notes:'Paragraph before A.6. F_c(Z^j) is the product of raw spatial clutter densities over that group. One clutter-group intensity is removed per target-generated group, not per detection. No additional group-size factorial is inserted.'});
add('constant','A.6','Separate the constant and define the group ratio.',
  t`<p>$C$ is fixed for the observed grouped scan and fixed model parameters.</p><p>$R_j(s)$ abbreviates the per-point likelihood ratios inside group $j$.</p>`,['pc_C','pc_R'],
  {notes:'Eq. (A.6), eq:pt_constant_C. C may be dropped when normalizing the posterior over the stated unknowns. If comparing partitions or estimating clutter/birth parameters, it is no longer automatically constant. R_j is a presentation shorthand whose expansion reproduces A.7–A.8.'});
add('legacy','A.7','A legacy target has four local cases.',
  t`<p>An existing target can claim a group or generate no detections.</p><p>A nonexistent target can only make a null claim.</p><p>The predicted prior remains a separate factor.</p>`,['pc_legacy'],
  {notes:'Eq. (A.7), eq:legacy_l_function, exactly after expanding R_a(s). The assigned/existing branch is exp(-mu_m) R_a/mu_g. The null/existing branch is exp(-mu_m); null/nonexistent is one; assigned/nonexistent is zero.'});
add('newborn','A.8','An unclaimed group can create one newborn.',
  t`<p>$b^j=0$: newborn and clutter compete.</p><p>$b^j\ne0$: a legacy target already owns the group, so the newborn cannot exist.</p><p>The absent candidate carries dummy density $f_d$.</p>`,['pc_new'],
  {notes:'Eq. (A.8), eq:new_l_function. Keep both mu_n f_n and the zero-truncation denominator. The absence branch f_d(s) applies whether or not the group is claimed by a legacy target. The candidate is indexed by the entire group.'});
add('final','A.9','The grouped joint PDF becomes a product of local factors.',
  t`<p>Predicted priors × legacy factors × newborn factors × group-association consistency.</p>`,['pc_final'],{kind:'wide-equation',
  notes:'Eq. (A.9), eq:final_joint_pdf_local_factors. Preserve the source’s approximation sign. Normalizing the displayed product for the fixed observed grouping defines the approximate-model posterior. BP then performs inference on this graph; loopy BP adds a separate inference approximation.'});
add('boundary','Model boundary','Association cannot repair a fixed partition.',
  t`<p><strong>S4:</strong> infer assignments of individual detections; several points can choose the same target.</p><p><strong>GrBP:</strong> infer whole-group assignments after external clustering.</p>`,[],
  {right:t`<div class="callout"><strong>One group per target.<br>One source per group.</strong></div><p>Target splits and cross-target merges are outside this model.</p><p>The number of groups is not the inferred number of targets.</p>`,
  notes:'Appendix I introduction and footnote to A.2. The model conditions on one partition, excludes target splits and cross-target merges, and approximates the induced clutter-group count. It specifies neither a clustering algorithm nor its parameter calibration. Unknown target count is still represented by the legacy and newborn existence indicators.'});

add('graph','Factor graph','Factor graph for GrBP',
  t`<p>Legacy chains and group-indexed newborn chains meet at $\boldsymbol\psi(\boldsymbol a_k,\boldsymbol b_k)$.</p>`,[],
  {kind:'graph-wide',figure:'fg-grbp.svg',
  caption:'Circles: variables · Boxes: factors · Blue: prediction between scans. The central factor collects all pairwise association-consistency factors.',
  alt:'GrBP factor graph: each predicted legacy prior connects to its state, legacy likelihood, and target association variable. The central consistency factor connects target and group association variables. Each group association connects to its newborn likelihood and state. Blue lines indicate prediction between scans.',
  notes:'Rendered from Drawings/graph_drawing.tex at manuscript commit d0a380a. The drawing is present in the manuscript repository but not included by the appendix. Its n_p and n_g labels are normalized to n^t and n^g; the original time index k is retained. The central psi box abbreviates the product of Psi_pj(a^p,b^j), not a claim that the expanded BP graph is loop-free. The newborn density is already inside the newborn likelihood factor, so no extra newborn-prior factor is added. The observation-only constant C is omitted. The pale blue paths describe prediction across scans, separate from the black within-scan factor-graph edges.'});

module.exports = {equations,slides,reference:{url:'../eo-derivation/source/appendix-I-original.tex',title:'EO_Writing_Long_Version · Appendix I · Eqs. (A.1)–(A.9) · commit d0a380a'},
  provenance:{repository:'BaiLiping/EO_Writing_Long_Version',commit:'d0a380a9caf81a36b88886efa91b5afd9bb7bdb5',path:'sections/sec_appendixI.tex',url:sourceURL,
    sourceSHA256:'9bf37e14bf9be93e4e4d9227a1520ccad46c5517793a8477f4040e93b43ce0d4',excerptSHA256:'61201bdb496727ab5eca877e6b5814c66b1ce7f1f7649ec626419871e29265cb',
    factorGraph:{path:'Drawings/graph_drawing.tex',sourceSHA256:'307c379677b4c13f28ebf9a0c02bec7029b2bd44515fac97c6027a7a126f86aa',countLabels:'n_p → n^t; n_g → n^g'}}};
