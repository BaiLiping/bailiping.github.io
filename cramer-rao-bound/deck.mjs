// Native Bento authoring. The original source retains the full notes and derivations.
import fs from 'node:fs';
const root=new URL('./',import.meta.url);
const original=JSON.parse(fs.readFileSync(new URL('src/deck.json',root)));
const formulas=JSON.parse(fs.readFileSync(new URL('src/formula-svg.json',root)));
const tex=JSON.parse(fs.readFileSync(new URL('src/formulas.json',root)));
const C={paper:'#ffffff',ink:'#16273e',muted:'#596d80',rule:'#d8e1e9',wash:'#f2f6fa',green:'#087f68',blue:'#2766b1',orange:'#b96815',purple:'#7854a3'};
const FONT='Arial, Helvetica, sans-serif';
const T=(id,x,y,w,h,html,size=22,color=C.ink,weight=400)=>({id,type:'text',x,y,w,h,html,fontSize:size,fontFamily:FONT,fontWeight:weight,color,align:'left',valign:'top',lineHeight:1.25,rotation:0,opacity:1});
const R=(id,x,y,w,h,fill=C.wash,radius=8)=>({id,type:'shape',shape:'rect',x,y,w,h,fill,stroke:'none',strokeWidth:0,radius,rotation:0,opacity:1});
const image=(id,svg,x,y,w,h,alt)=>({id,type:'image',src:'data:image/svg+xml;base64,'+Buffer.from(svg).toString('base64'),x,y,w,h,alt,fit:'contain',rotation:0,opacity:1});
const F=(name,x,y,w,h)=>image(name,formulas[name].replace(/style="[^"]*"/,'style="color:#16273e"'),x,y,w,h,tex[name]);
const G=(name,x,y,w,h,alt)=>image(name,fs.readFileSync(new URL(`assets/${name}.svg`,root),'utf8'),x,y,w,h,alt);
const link=(label,href,live=false)=>`<a class="${live?'try-live':'deck-link'}" href="${href}"><span>${label}</span><span aria-hidden="true"> →</span></a>`;
const L=(id,x,y,w,label,href,live=false)=>T(id,x,y,w,44,link(label,href,live),20,C.green,600);
const live=(lab,x=804,y=578,w=380)=>L('try-live',x,y,w,'TRY LIVE',`./live/index.html?lab=${lab}`,true);
const H=(id,x,y,w,text,color=C.green)=>T(id,x,y,w,34,text,24,color,700);
const B=(id,x,y,w,text,size=22)=>T(id,x,y,w,100,text,size,C.muted);
const SOURCES={S1:'https://web.stanford.edu/class/archive/stats/stats200/stats200.1172/Lecture15.pdf',S2:'https://arxiv.org/abs/1705.01064',S3:'https://arxiv.org/abs/1006.0888',S4:'https://web.stanford.edu/class/archive/stats/stats200/stats200.1172/Lecture14.pdf'};
function slide(id,body){
 const source=original.find(s=>s.id===id);
 return {id:'s-'+id,background:C.paper,transition:'none',notes:source.notes.replace('The existing site’s warm paper, Georgia headings, and restrained teal/blue/rust palette informed the visual design.','The deck uses the site’s technical Bento presentation format; full explanations and compact labs are linked in the same tab.'),elements:[
  T('kicker',72,32,1136,22,source.section.toUpperCase(),12,C.green,700),
  T('title',72,68,1136,82,source.title,36,C.ink,700),
  T('subtitle',72,155,1136,47,source.subtitle,18,C.muted),R('rule',72,213,1136,1,C.rule,0),
  ...body,T('footer',72,682,720,20,'Cramér–Rao Bound · Precision has a floor',11,C.muted),
  ...source.sources.map((key,i)=>T('source-'+key,872+48*i,682,42,20,link(key,SOURCES[key]),11,C.muted)),
  T('page',1120,682,88,20,'{{page:2}} / {{pages:2}}',11,C.muted)
 ]};
}
export function build(){
 const s=[];
 s.push(slide('overview',[
  T('headline',76,242,560,150,'Precision has<br>a floor.',58,C.green,700),
  B('intro',80,413,510,'Connect what measurements reveal to what estimators can achieve.',27),
  T('path',80,523,520,64,'Fisher information · estimator bias<br>localization geometry',21,C.muted),
  L('guide',80,611,510,'Open the interactive guide','./guide/index.html'),
  G('hero',683,225,509,418,'Repeated sample means at n = 4, 16, and 64. Their spread decreases around the same true mean.')
 ]));
 s.push(slide('repeat',[
  ...[['01','A fixed world','The unknown θ stays fixed across trials.'],['02','New measurements','Each trial produces a fresh random dataset X.'],['03','A new estimate','Apply the same rule T(X). Its spread is the variance.']].flatMap(([n,h,b],i)=>[R('card'+i,72+386*i,244,364,207),T('n'+i,96+386*i,266,300,28,n,17,C.green,700),H('head'+i,96+386*i,310,315,h),B('body'+i,96+386*i,360,315,b,22)]),
  F('m1',92,494,540,62),H('question',706,492,480,'How narrow can the cloud be?'),
  B('unbiased',104,581,508,'Unbiased means correct on average over repeated datasets.',21),
  B('question-copy',706,543,478,'Precision describes repeated estimates. One close estimate does not establish precision.',22)
 ]));
 s.push(slide('distinguish',[
  H('plot-title',92,248,530,'Same separation · different noise'),G('overlap',84,294,553,299,'Densities under θ and θ + δ overlap more for noisy observations.'),
  T('density-note',100,612,525,30,'Data densities; not posterior probabilities.',17,C.muted),
  H('mechanism',718,251,480,'A change must leave a trace.'),B('mechanism-copy',718,309,468,'Almost indistinguishable data cannot support arbitrarily concentrated, locally unbiased estimates.',23),
  F('m2',708,443,478,88),B('more',720,570,458,'More Fisher information permits a lower variance bound.',23)
 ]));
 s.push(slide('score',[
  R('likelihood-card',72,243,540,395),H('likelihood-title',100,268,485,'1 · Likelihood'),F('m3',108,322,469,59),G('likelihood',102,396,478,187,'Two likelihood curves: the narrower curve has greater curvature.'),T('likelihood-note',104,592,475,42,'A likelihood is not a probability density over θ.',18,C.muted),
  H('score-title',683,253,500,'2 · Score'),F('m4',690,296,478,59),T('score-note',688,366,486,33,'Under regularity, the score has mean zero.',19,C.muted),
  H('information-title',683,429,500,'3 · Fisher information'),F('m5',670,480,530,81),B('information-note',688,581,490,'Expected squared score = expected curvature, under the required interchange conditions.',19)
 ]));
 s.push(slide('bound',[
  R('bound-card',72,247,572,387),H('bound-title',103,273,500,'The variance floor'),F('m6',107,340,498,106),F('m7',135,488,443,78),
  T('units',100,591,505,31,'Variance: squared units · SD: parameter units',17,C.muted),
  H('scope',704,251,490,'The theorem has a scope.'),
  ...[['Unbiasedness','Eθ[T] = θ in a neighborhood.'],['Regularity','Differentiate under the expectation; Eθ[Uθ] = 0.'],['Finite information','0 < Iₙ(θ) < ∞ and finite estimator variance.'],['Known model','The likelihood describes the experiment.']].flatMap(([h,b],i)=>[T('scope-head'+i,708,307+82*i,478,28,h,21,C.ink,700),T('scope-body'+i,708,340+82*i,478,48,b,19,C.muted)])
 ]));
 s.push(slide('proof',[
  H('identity',92,245,490,'The score identity'),
  ...[['Differentiate unbiasedness','m8'],['Move the derivative inside','m9'],['Use the zero-mean score','m10']].flatMap(([label,f],i)=>[T('step'+i,98,300+i*108,485,30,`${i+1}   ${label}`,21,C.muted),F(f,121,339+i*108,440,47)]),
  R('proof-card',656,242,552,403),H('cs',684,265,495,'Cauchy–Schwarz'),F('m11',678,325,507,66),F('m12',721,423,420,53),F('m13',717,518,429,67),T('proof-note',689,603,492,29,'An unbiased estimator must respond to the score.',17,C.muted)
 ]));
 s.push(slide('gaussian',[
  H('compute',92,247,520,'Compute the information'),F('m14',88,305,523,80),F('m15',99,426,511,66),F('m16',186,551,319,57),
  R('mean-card',656,244,552,399),H('compare',685,270,488,'Compare with the sample mean'),F('m17',680,340,504,72),F('m18',675,475,511,73),T('exact',689,589,490,39,'Exact equality for every n ≥ 1.',24,C.green,700)
 ]));
 s.push(slide('experiment-intro',[
  H('predict',93,248,524,'Predict → change → explain'),
  ...[['16 → 64 samples','What happens to variance and SD?'],['Double the noise σ','Which quantity becomes four times larger?'],['Use only the first observation','Does collecting unused data help?']].flatMap(([h,b],i)=>[T('change'+i,96,309+i*108,523,33,h,24,C.ink,700),T('explain'+i,96,352+i*108,520,51,b,21,C.muted)]),
  R('baseline',670,243,538,397),H('baseline-title',701,272,471,'Reference experiment'),
  ...[['16','samples / trial'],['2','noise σ'],['1','true μ']].flatMap(([n,t],i)=>[T('value'+i,710+i*153,344,140,61,n,42,C.green,700),T('label'+i,710+i*153,405,145,43,t,16,C.muted)]),
  F('m19',700,487,472,52),T('baseline-note',706,574,465,45,'Truth stays fixed. Each trial uses fresh observations.',20,C.muted)
 ]));
 s.push(slide('gaussian-lab',[
  G('mc-chart',77,258,688,276,'Seeded histogram of 2,000 sample means, with exact sampling-density reference curves.'),
  ...[['0.2570','empirical variance'],['0.2500','exact variance'],['0.2500','unbiased CRB']].flatMap(([v,l],i)=>[R('metric-bg'+i,88+223*i,554,207,82),T('metric'+i,106+223*i,565,180,37,v,29,C.green,700),T('metric-label'+i,106+223*i,607,180,22,l,14,C.muted)]),
  H('experiment',805,255,380,'Change the experiment'),
  B('sampling',807,309,376,'Independent Gaussian datasets.<br>n = 16 · σ = 2 · μ = 1<br>2,000 trials · seed 42',21),
  B('sampling-prompt',807,433,376,'Increase n, change noise, or discard all but one observation. Compare the estimator with the same full-data bound.',21),
  live('gaussian-lab'),T('sampling-note',89,646,1095,22,'A finite simulated variance can fall below the CRB through Monte Carlo fluctuation.',16,C.muted)
 ]));
 s.push(slide('scaling',[
  G('scaling',79,251,557,360,'Log-scale curves: variance bound 4/n and standard-deviation bound 2/sqrt(n).'),
  F('m20',675,246,512,79),
  ...[['n','Variance bound','SD bound'],['4','1.0000','1.00'],['16','0.2500','0.50'],['64','0.0625','0.25']].flatMap((row,i)=>row.map((v,j)=>T('cell'+i+j,684+j*173,370+i*49,165,33,v,i?23:15,i?C.ink:C.muted,i===0?700:400))),
  T('joint-note',686,588,496,66,'Correlated or non-identical data require the information of the joint likelihood.',20,C.muted)
 ]));
 s.push(slide('bias',[
  H('decomposition',92,244,520,'Bias–variance decomposition'),F('m21',113,294,483,48),F('m22',95,367,524,53),G('bias-cloud',90,460,537,198,'An unbiased density centered at truth and a narrower biased density displaced from truth.'),
  R('biased-bound-card',664,243,544,398),H('extension',692,271,483,'The correct biased extension'),F('m23',708,328,457,86),F('m24',678,468,512,87),T('bias-note',695,592,480,36,'The derivative of the bias matters.',22,C.green,700)
 ]));
 s.push(slide('bias-lab',[
  G('bias-chart',79,258,685,247,'Exact distributions of the unbiased sample mean and shrinkage estimator, at alpha = 0.55 and true mean = 0.35.'),G('bias-bars',96,536,648,95,'Exact variance, squared bias, total MSE, and the unbiased CRB for the default shrinkage example.'),
  H('shrink-title',806,254,378,'Shrink toward zero'),T('shrink-model',808,307,374,43,'Tα = α X̄',31,C.green,700),
  T('shrink-baseline',808,369,374,110,'α = 0.55 · μ = 0.35<br>Variance = 0.0756<br>Squared bias = 0.0248<br>MSE = 0.1004',21,C.muted),
  B('shrink-prompt',808,494,374,'Move μ away from zero.<br>Does the MSE advantage remain?',20),live('bias-lab'),
  T('shrink-note',92,646,1096,22,'The unbiased CRB is 0.25. A biased estimator can have a smaller pointwise MSE.',16,C.muted)
 ]));
 s.push(slide('efficiency',[
  R('equality-card',72,245,558,394),H('equality',102,274,494,'Equality in Cauchy–Schwarz'),F('m25',96,351,512,75),F('m26',105,486,487,64),T('equality-note',107,587,488,44,'The same statistic must work throughout the parameter range.',19,C.muted),
  H('claims',692,251,498,'Keep three claims distinct.'),
  ...[['Unbiased','Expectation equals the target.'],['Efficient','Variance attains the CRB.'],['Maximum likelihood','Maximizes likelihood; finite-sample bias is possible.']].flatMap(([h,b],i)=>[T('claim'+i,697,313+i*92,485,31,h,24,C.green,700),T('claim-copy'+i,697,351+i*92,485,52,b,21,C.muted)]),
  T('asymptotic',697,610,485,43,'Regular MLEs may attain the bound asymptotically.',19,C.muted)
 ]));
 s.push(slide('matrix',[
  H('matrix-title',92,247,510,'The matrix CRB'),F('m27',115,297,475,63),F('m28',101,395,512,75),F('m29',90,514,535,57),T('matrix-note',98,609,530,43,'Nonsingular J; regular model; locally unbiased estimator.',18,C.muted),
  G('ellipse',690,261,509,328,'Covariance-bound ellipse with a long axis in the weak-information direction.'),T('direction-note',710,601,460,49,'Larger information eigenvalue → smaller directional variance bound.',21,C.muted)
 ]));
 s.push(slide('range-model',[
  H('model',92,246,524,'Independent Gaussian ranges'),F('m30',87,301,552,60),F('m31',152,415,409,70),F('m32',137,540,449,83),
  G('range-direction',688,253,512,311,'One anchor constrains motion along the radial line of sight; tangential motion has zero first-order range sensitivity.'),
  T('range-note',699,587,493,72,'Each term is rank one. Anchors are known; noise variance is known and independent of position.',20,C.muted)
 ]));
 s.push(slide('geometry-intro',[
  H('configurations',92,248,530,'Predict the weak direction'),
  ...[['Surround','Anchors point toward the target from different directions.'],['Cluster','Anchors share almost the same line of sight.'],['Collinear','Anchors and target lie on the same straight line.']].flatMap(([h,b],i)=>[T('configuration'+i,96,307+i*111,521,30,h,24,C.green,700),T('configuration-copy'+i,96,345+i*111,517,60,b,21,C.muted)]),
  R('peb-card',666,245,542,397),H('peb',697,273,483,'Position error bound'),F('m33',719,344,445,77),T('rmse',703,455,477,74,'For an unbiased position estimator:<br>Euclidean RMSE ≥ PEB.',23),T('local-note',703,569,477,58,'Finite local information does not rule out distant mirror solutions.',21,C.muted)
 ]));
 s.push(slide('geometry-lab',[
  G('geometry-chart',76,252,690,350,'Four surrounding anchors and a true target, with the unit-Mahalanobis covariance-bound ellipse.'),
  T('geometry-baseline',96,616,660,43,'Default: σ = 0.60 m · PEB = 0.625 m',23,C.green,700),
  H('geometry-try',804,254,386,'Reshape the information'),B('geometry-copy',807,311,376,'Move an anchor or the target.<br>Compare Surround, Cluster, and Collinear.',23),
  B('geometry-offset',807,429,376,'Then estimate an unknown common range offset. Watch the equivalent position information decrease.',21),live('geometry-lab'),
  T('ellipse-note',806,641,380,25,'The ellipse is not a confidence region.',16,C.muted)
 ]));
 s.push(slide('nuisance',[
  R('schur-card',72,243,570,397),H('partition',104,270,504,'Partition the information'),F('m34',103,327,506,99),F('m35',105,487,503,79),T('inverse-note',104,591,506,34,'The position block of J⁻¹ is Jₑ⁻¹.',22,C.green,700),
  H('coupling',705,253,483,'Some effects imitate each other.'),B('coupling-copy',709,309,477,'An unknown offset can explain changes in range that would otherwise constrain position.',23),F('m36',815,429,258,54),
  T('coupling-note',708,509,478,49,'No loss when b = 0. Clustered lines of sight can lose much more.',20,C.muted),live('geometry-lab',706,589,468)
 ]));
 s.push(slide('regularity',[
  H('uniform-title',92,245,518,'Xi ∼ Uniform(0, θ)'),F('m37',90,302,539,69),F('m38',110,408,494,55),F('m39',158,500,399,60),T('support-note',102,599,518,49,'The moving boundary invalidates the regular score identity.',21,C.orange,600),
  R('maximum-card',665,245,543,397),H('maximum-title',695,273,480,'Check the estimator directly.'),F('m40',695,336,480,65),F('m41',733,455,407,92),T('maximum-note',699,578,476,61,'The variance follows from the distribution of the sample maximum.',21,C.muted)
 ]));
 s.push(slide('design',[
  R('linear-card',72,244,553,398),H('linear-title',100,272,495,'Linear Gaussian measurements'),F('m42',96,333,504,62),F('m43',125,451,449,75),T('linear-note',102,576,495,66,'H sets sensitivity and geometry.<br>R sets noise and correlation.',22,C.muted),
  H('criteria',685,253,511,'Choose a design objective'),
  ...[['Total variance','Minimize tr(J⁻¹)'],['Ellipsoid volume','Maximize log det J'],['Weakest direction','Maximize λmin(J)']].flatMap(([a,b],i)=>[T('criterion'+i,690,322+86*i,490,30,a,22,C.ink,700),T('objective'+i,690,358+86*i,490,32,b,23,C.green)]),
  T('design-note',690,602,492,50,'Local criteria depend on the parameterization and units.',20,C.muted)
 ]));
 s.push(slide('reviewer',[
  ...[['01 · Estimator class','Is the comparison valid?','Analyze bias before comparing a learned, constrained, or regularized estimator with an unbiased bound.'],['02 · Parameter vector','What was treated as known?','Include unknown offsets, calibration, orientation, and other nuisance parameters before inverting.'],['03 · Metric and model','Are the assumptions aligned?','Match variance, MSE, and RMSE with the correct bound. Model measurement correlations.'],['04 · Local versus global','What can the bound miss?','Inspect rank, conditioning, competing solutions, and low-information regimes.']].flatMap(([k,h,b],i)=>{const x=72+(i%2)*580,y=244+Math.floor(i/2)*208;return [R('review-bg'+i,x,y,556,188),T('review-kicker'+i,x+24,y+18,507,23,k,13,C.green,700),T('review-head'+i,x+24,y+53,507,33,h,24,C.ink,700),T('review-copy'+i,x+24,y+101,507,74,b,20,C.muted)];})
 ]));
 s.push(slide('takeaways',[
  ...[['01 · Sensitivity','Information belongs to the experiment.','m44','Noise and sensing design change the limits before any estimator runs.'],['02 · Precision','Unbiased variance has a floor.','m45','An estimator within the theorem’s class cannot beat the floor.'],['03 · Direction','The matrix tells you where.','m46','Complementary measurement directions improve the weakest constraints.']].flatMap(([k,h,f,b],i)=>{const x=72+386*i;return [R('takeaway-bg'+i,x,250,364,392),T('takeaway-kicker'+i,x+25,275,315,25,k,14,C.green,700),T('takeaway-head'+i,x+25,324,315,98,h,28,C.ink,700),F(f,x+22,439,320,61),T('takeaway-copy'+i,x+25,545,315,84,b,21,C.muted)];})
 ]));
 s.push(slide('references',[
  H('primary',92,246,540,'Primary references'),
  ...[['S1 · Stanford STATS 200, Lecture 15','Fisher information, scalar bound, and efficiency.','S1'],['S2 · Ly et al. (2017)','A Tutorial on Fisher Information.','S2'],['S3 · Shen & Win (2010)','Fundamental Limits of Wideband Localization, Part I.','S3'],['S4 · Stanford STATS 200, Lecture 14','Regular asymptotic normality of the MLE.','S4']].flatMap(([h,b,key],i)=>[T('reference'+i,96,304+i*84,560,35,link(h,SOURCES[key]),20,C.green,600),T('reference-copy'+i,96,343+i*84,561,40,b,17,C.muted)]),
  R('resources-card',694,244,514,397),H('resources',722,269,459,'Explore and take it with you'),
  L('guide',725,323,450,'Interactive guide & experiments','./guide/index.html'),L('pdf',725,381,450,'Download PDF','./Cramer-Rao-Bound.pdf'),L('pptx',725,439,450,'Download PowerPoint','./Cramer-Rao-Bound.pptx'),L('source-package',725,497,450,'Download source package','./Cramer-Rao-Bound-site-package.zip'),
  T('registration',726,563,451,29,link('Frame registration','/frame-registration-slides/'),18,C.green,600),T('radar',726,602,451,29,link('Radar SLAM','/radar-slam/'),18,C.green,600)
 ]));
 if(s.length!==original.length)throw new Error('Slide count changed');
 return s;
}
export default {build,title:'Cramér–Rao Bound: Precision Has a Floor',description:'A 23-slide Bento presentation on Fisher information, estimator bias, and localization geometry, with three interactive experiments and a complete companion guide.',docId:'cramer-rao-bound-bento-v1',...C,fontFamily:FONT};
