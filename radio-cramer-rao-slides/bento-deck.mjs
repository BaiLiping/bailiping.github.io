// Editable, mathematical source for the Radio CRB presentation.
// The neighboring public radio-geometry deck supplies the Bento runtime.
const C={bg:'#F4F6F8',ink:'#16222E',muted:'#51606E',line:'#D7DEE5',blue:'#1874B8',teal:'#0A6B5E',orange:'#D76809',paper:'#FFFFFF',cool:'#E8F2FA',mint:'#E3F2EF',warm:'#FCEBDA',purple:'#7C4DBE'};
const serif="Georgia, 'Times New Roman', serif",sans='Arial, Helvetica, sans-serif',mono='Menlo, Consolas, monospace';
const R=String.raw;
const M=s=>`<span class="math-tex math-display">\\[${s}\\]</span>`;
const I=s=>`<span class="math-tex math-inline">\\(${s}\\)</span>`;
export const references={
 Li:{short:'Li et al. (2023)',title:'Optimal and Robust Waveform Design for MIMO-OFDM Channel Sensing: A Cramér-Rao Bound Perspective',url:'https://arxiv.org/abs/2301.10689'},
 Shah:{short:'Shahmansoori et al. (2018)',title:'Position and Orientation Estimation through Millimeter-Wave MIMO in 5G Systems',url:'https://arxiv.org/abs/1702.01605'},
 LeMagoarou:{short:'Le Magoarou & Paquelet (2020)',title:'Channel estimation: unified view of optimal performance and pilot sequences',url:'https://arxiv.org/abs/2002.04481'},
 Noise:{short:'Texas Instruments: noise figure',title:'Signal Chain Noise Figure Analysis (SLAA652)',url:'https://www.ti.com/lit/pdf/slaa652'},
 Background:{short:'Radio measurements and geometry',title:'Radio Measurements → Radio Map',url:'https://bailiping.com/mpc-detection-to-bounce-count-slides/'},
 Model:{short:'Radio SLAM system model',title:'Radio SLAM system model: physical phase-coded I/Q and receiver timing bias',url:'https://github.com/BaiLiping/radio-SLAM-system-model'}
};
function txt(id,x,y,w,h,html,o={}){return{id,type:'text',x,y,w,h,rotation:0,opacity:1,html,fontFamily:sans,fontSize:20,fontWeight:400,color:C.ink,align:'left',valign:'top',lineHeight:1.35,...o};}
function box(id,x,y,w,h,fill=C.paper,stroke=C.line){return{id,type:'shape',shape:'rect',x,y,w,h,fill,stroke,strokeWidth:1,radius:10,rotation:0,opacity:1};}
function panel(id,x,y,w,h,label,body,{fill=C.paper,size=20,accent=C.blue}={}){return[box(id+'-box',x,y,w,h,fill),txt(id+'-label',x+22,y+18,w-44,24,label,{fontFamily:mono,fontSize:12,fontWeight:700,color:accent}),txt(id+'-body',x+22,y+61,w-44,h-79,body,{fontSize:size})];}
function source(keys){return keys.map(k=>`<a href="${references[k].url}" target="_blank" rel="noopener">${references[k].short}</a>`).join(' · ');}
const slides=[];
function add(id,section,title,subtitle,elements,notes,keys=['Li']){slides.push({id,background:C.bg,transition:'none',notes:notes+'\n\nSources: '+keys.map(k=>references[k].title+' — '+references[k].url).join('\n'),elements:[
 txt('section',96,28,780,22,section.toUpperCase(),{fontSize:12,fontFamily:mono,color:C.orange,fontWeight:700,letterSpacing:1}),
 txt('home',930,28,254,22,'RADIO SLAM · ESTIMATION',{fontSize:11,fontFamily:mono,color:C.muted,align:'right',link:'https://bailiping.com/'}),
 box('rule',96,66,1088,1,C.line,C.line),
 txt('heading',96,86,1088,52,title,{fontFamily:serif,fontSize:38,fontWeight:700,lineHeight:1.1}),
 txt('subtitle',96,142,1088,35,subtitle,{fontSize:17,color:C.muted}),...elements,
 txt('sources',96,676,920,22,source(keys),{fontSize:11,color:C.muted}),
 txt('contents',1034,677,150,20,'CONTENTS ↗',{fontSize:11,fontFamily:mono,color:C.blue,align:'right',link:'overview'})]});}
function two(id,section,title,subtitle,lLabel,lBody,rLabel,rBody,notes,keys=['Li'],sizes=[21,20]){add(id,section,title,subtitle,[...panel('left',96,195,532,443,lLabel,lBody,{fill:C.cool,size:sizes[0]}),...panel('right',652,195,532,443,rLabel,rBody,{size:sizes[1],accent:C.teal})],notes,keys);}
function four(id,section,title,subtitle,items,notes,keys=['Li']){add(id,section,title,subtitle,items.flatMap(([label,body],i)=>panel('panel-'+i,96+(i%2)*556,191+Math.floor(i/2)*233,532,214,label,body,{fill:i===0?C.cool:C.paper,size:19,accent:i===2?C.orange:C.blue})),notes,keys);}
function table(id,headers,rows,{y=194,h=432,fontSize=18,widths}={}){return{id,type:'table',x:96,y,w:1088,h,rotation:0,opacity:1,header:true,columns:(widths||headers.map(()=>1)).map(w=>({w})),rows:[{cells:headers.map(html=>({html}))},...rows.map(row=>({cells:row.map(html=>({html}))}))],style:{headerBg:C.blue,headerColor:'#fff',zebra:C.cool,borderColor:C.line,borderWidth:1,cellPadX:15,cellPadY:10,fontSize,color:C.ink,fontFamily:sans,radius:10}};}
const liveMap=[];
function live(id,intro,title,subtitle,lab,body,base='./live/'){add(id,'Interactive experiment',title,subtitle,[...panel('print-fallback',72,180,1136,475,'INTERACTIVE LAB',body,{size:23,fill:C.cool}),{...box('live-demo-mount',72,180,1136,475,'transparent','transparent'),opacity:0}], 'The interactive panel evaluates the stated statistical model. Every displayed bound is a square root of a marginal CRB. '+body.replace(/<[^>]+>/g,' '));liveMap.push({slide:id,introSlide:intro,inline:true,layout:'region',bounds:{x:72,y:180,width:1136,height:475},src:`${base}?lab=${lab}&embed=1`,source:`${base}?lab=${lab}`,title,sandbox:'allow-scripts allow-same-origin allow-top-navigation-by-user-activation',hideSource:true,readyMessage:true,unloadWhenHidden:true});}

add('overview','Radio detection / precision limits','Cramér–Rao bounds for radio measurements','How much information does a MIMO–OFDM experiment contain?',[
 txt('hero',96,208,1088,118,'Known pilots. Known channel?<br><span style="color:#0A6B5E">Different information, different bounds.</span>',{fontFamily:serif,fontSize:44,fontWeight:700,lineHeight:1.1}),
 txt('intro',96,349,1088,62,'Calculate delay, AoA, AoD and path gain with coded pilots. Explore what remains unknown when the propagation channel is given.',{fontSize:22,color:C.muted}),
 txt('update',96,420,1088,22,'UPDATED 24 SEPTEMBER 2026 · 384-PORT CODED PILOTS + KNOWN-CHANNEL DEMO',{fontSize:12,fontFamily:mono,fontWeight:700,color:C.orange}),
 ...[['01','384 × 50 pilots','project-live',C.cool],['02','Known channel','known-channel-live',C.mint],['03','Reference model','calculator-live',C.warm],['04','Fisher information','fisher-live',C.cool]].flatMap(([n,t,link,fill],i)=>[box('tile-'+i,96+278*i,458,254,118,fill),txt('nav-'+i,117+278*i,479,212,76,n+'<br><b>'+t+'</b>',{fontSize:20,link})]),
 txt('background',96,612,1088,28,'Background: Radio Measurements → Radio Map ↗',{fontSize:17,color:C.teal,link:references.Background.url})
],'A public educational companion to radio multipath geometry. The coded-pilot section follows the physical I/Q observation design with one fixed transmit code matrix across tones. Its representative isolated channel and seeded demonstration pilots do not constitute a saved Sionna scene evaluation. The known-channel section separates simulator truth, exact geometric channel, exact effective channel and noisy CSI. Five reference labs retain the explicitly labeled orthogonal baseline and two-path gain illustration.',['Li','Shah','Background']);

two('meaning','01 · what a bound means','The CRB describes estimation precision','A measurement is an estimate extracted from noisy radio samples.',
 'THE STATISTICAL STATEMENT',M(R`\operatorname{Cov}(\hat{\boldsymbol\eta})\succeq\mathbf J(\boldsymbol\eta)^{-1}`)+'For a locally unbiased estimator under the specified regular model.<br><br>'+M(R`\sigma_{i,\mathrm{LB}}=\sqrt{[\mathbf J^{-1}]_{ii}}`),
 'WHAT THE NUMBER MEANS','A delay bound of <b>0.03 ns</b> is a lower bound on standard deviation at the assumed true channel.<br><br>It describes one local solution. Detection failures, ambiguous solutions, bias, and model mismatch can make actual errors much larger.',
 'The CRB is a matrix lower bound, not a guarantee of an attainable estimator. Detection probability and the unknown number of paths are different statistical problems. A singular direction has no ordinary finite local CRB. Results in this deck are conditional on one path except the explicitly identified multipath gain illustration.');

add('setup','02 · define the experiment','The setup needs more than antenna counts','The calculator uses one transmitting BS and one receiving UE.',[
 table('notation',['Symbol','Meaning','Why it matters'],[
 ['Nₜ = Nₜ,y Nₜ,z','Transmitting array elements / ports','Aperture and distinguishable transmit pilots'],
 ['Nᵣ = Nᵣ,y Nᵣ,z','Receiving array elements / ports','Aperture and received observations'],
 ['L','Coherent OFDM pilot symbols','Useful observation time and pilot rank'],
 ['K, Δf, B = KΔf','Active contiguous tones, spacing, grid bandwidth','Frequency spread and symbol duration'],
 ['f<sub>c</sub>, d, (az, el)','Carrier, element spacing, true path directions','Spatial phase derivatives'],
 ['α, σ², Xₖ','Path gain, complex noise variance, known pilots','Signal strength and information weighting']
 ],{fontSize:18,widths:[.85,1.6,1.55]})
],'Ny and Nz are element counts along the two axes of a rectangular array. This does not assume separate base stations for each transmit element. Antenna polarization, element patterns, mutual coupling, calibration errors and hybrid combiners require an expanded model.',['Li','Shah']);

two('project-setup','Configured radio experiment','The physical coded-pilot observation','One BS array transmits 50 known spatial codes. Every received symbol is retained.',
 'SIGNAL AND RECEIVER CLOCK',M(R`\mathbf Y_k=\widetilde{\mathbf H}_k\mathbf X+\mathbf W_k`)+M(R`\widetilde{\mathbf H}_k=e^{-j2\pi f_kb}\mathbf H_k^{\mathrm g}`)+M(R`\mathbf X\in\mathbb C^{384\times50},\quad\mathbf Y_k\in\mathbb C^{32\times50}`)+'The same X is reused on every tone. '+I('b')+' is the receiver clock bias in seconds.',
 'CONFIGURED DIMENSIONS','<b>BS:</b> 24 horizontal × 16 vertical ports.<br><b>UE:</b> 8 horizontal × 4 vertical ports.<br><br><b>50 symbols, 3300 tones, 400 MHz.</b><br>Carrier: 27.2 GHz. Spacing: λ/2.<br><br>'+M(R`f_k=(k-\lfloor K/2\rfloor)B/K`)+ 'The tensor contains 5,280,000 complex observations per acquisition.',
 'The system model calls the symbol count S and tone count N. This deck uses L and K for the same quantities, and b for the clock bias written beta there. The configured matrices have 384 TX and 32 RX ports on one BS and one UE. No spatial scene, trajectory, or saved RT realization is embedded in this teaching demo. The frequency grid includes DC and is not exactly centered for even K. The fixed physical code matrix is reused at every bin.',['Model','Li'],[23,21]);

add('project-fim','Configured radio experiment','The actual pilots enter the Fisher matrix','Compressed training can identify a structured path even when arbitrary full CSI is unavailable.',[
 ...panel('general-fim',96,190,1088,211,'JOINT INFORMATION FROM THE RECEIVED I/Q',M(R`\mathbf M_k=e^{-j2\pi f_kb}\mathbf H_k^{\mathrm g}(\boldsymbol\eta)\mathbf X`)+M(R`J_{ij}=\frac{2}{\sigma^2}\operatorname{Re}\sum_k\operatorname{tr}\!\left[(\partial_i\mathbf M_k)^H(\partial_j\mathbf M_k)\right]`),{size:23,fill:C.cool}),
 ...panel('pilot-rank',96,423,532,215,'THE TRANSMIT SUBSPACE',M(R`\operatorname{rank}(\mathbf X)\le50<384`)+ 'XXᴴ is singular. Arbitrary full CSI cannot be recovered independently on each tone.',{size:21}),
 ...panel('parameter-rank',652,423,532,215,'THE PARAMETER JACOBIAN','Angle and delay derivatives can remain independent after multiplication by X.<br><br>The full joint FIM gives their bounds.',{size:21,fill:C.mint})
],'For any real parameter eta_i, partial_i M_k = exp(-j2pi f_k b)[partial_i H^g_k - j2pi f_k H^g_k partial_i b] X. A fixed covariance sigma²I gives J=2/sigma² Re(D^H D). The rank relevant for local parameter identifiability is the realified stacked derivative matrix. A pseudoinverse of XX^H recovers only the excited transmit subspace. Free path gains and geometric delays plus an unconstrained common clock create a gauge that transmit diversity alone cannot remove.',['Model','Li','LeMagoarou']);

two('project-codes','Configured radio experiment','A coded-pilot bounds calculator','The next lab evaluates an isolated representative path using the physical observation design.',
 'POWER AND NOISE',M(R`X_{ps}=\sqrt{\frac{P}{N_tK}}\,C_{ps}`)+M(R`C_{ps}\in\{1,j,-1,-j\}`)+M(R`\sigma^2=10^{(-174+\mathrm{NF}-30)/10}\Delta f`)+ 'At NF = 9 dB, independent of P:<br>σ² = 3.8331 × 10⁻¹⁵ W per complex sample.',
 'EXPERIMENTS TO TRY','<b>50 symbols:</b> inspect finite structured-path bounds with fewer symbols than TX ports.<br><br><b>One symbol:</b> AoD can merge into the unknown complex path gain.<br><br><b>Another pilot seed:</b> compare changes in excitation and parameter coupling.',
 'The browser uses a deterministic demonstration QPSK generator, not the actual saved NumPy pilot matrix. Arrays are calibrated, static and planar. Effective attenuation includes element and propagation effects once. The isolated-path amplitude is freely estimated, so angle-dependent amplitude changes can be absorbed into this nuisance parameter. This is an illustrative conditional channel, not an evaluation of the campus RT channel. All 50 symbols remain separate, and noise is not divided by symbol count.',['Model','Li'],[22,21]);
live('project-live','project-codes','Coded pilots and measurement bounds','The full derivative calculation supports the 384-port, 50-symbol design.','coded','Vary antenna dimensions, symbols, bandwidth, power, noise and path direction. The seeded QPSK matrix is reused on every tone.<br><br>The model computes a full seven-parameter Fisher matrix and its marginal bounds. It retains gain and phase nuisance coupling.<br><br>The channel and pilot seed are illustrative. A run-specific result needs the saved X and RT path derivatives.','./project/');

add('known-channel','Known channel','Known to whom, and known in what form?','The information supplied to the estimator determines which uncertainty remains.',[
 table('knowledge',['Available information','Estimator input','Meaning for the bound'],[
 ['Simulator knows the truth','Noisy Y, known X','Evaluate a nonzero CRB at that true channel.'],
 ['Exact geometric Hᵍ','Hᵍ, X, noisy Y','Clock bias and an optional receiver phase can remain unknown.'],
 ['Exact effective H̃','Noiseless H̃','Identifiable functions have no receiver-noise uncertainty. Ambiguities may remain.'],
 ['Noisy channel estimate Ĥ','Ĥ and its uncertainty','Keep its error covariance in the likelihood.']
 ],{fontSize:19,widths:[1.18,1.05,1.8],h:428})
],'A fixed unknown parameter is routinely known to the simulation or evaluator when calculating the deterministic CRB. It remains unknown to the estimator. Exact noiseless channel input changes the statistical experiment: use identifiability analysis or the vanishing-noise limit, rather than substituting a zero covariance into the ordinary inverse. Full effective channel knowledge does not necessarily separate geometric delay from clock bias or uniquely determine a multipath decomposition. A noisy estimate is not exact channel knowledge.',['Li','LeMagoarou','Model']);

two('known-channel-clock','Known geometric channel','Known propagation can leave an unknown clock','The geometric channel provides a template with an absolute delay reference.',
 'MEAN AND CLOCK DERIVATIVE',M(R`\mathbf A_k=\mathbf H_k^{\mathrm g}\mathbf X`)+M(R`\mathbf M_k=e^{-j2\pi f_kb}\mathbf A_k`)+M(R`\partial_b\mathbf M_k=-j2\pi f_k\mathbf M_k`)+ 'The receiver compares this template with noisy received samples.',
 'CLOCK INFORMATION',M(R`w_k=\|\mathbf A_k\|_F^2`)+M(R`J_{bb}=\frac{8\pi^2}{\sigma^2}\sum_k f_k^2w_k`)+M(R`\sigma_{b,\mathrm{LB}}=J_{bb}^{-1/2}`)+ 'This form assumes the complex phase reference is also known.',
 'Holding known H^g fixed, the derivative follows directly from the timing factor in the physical model. Its squared Frobenius norm gives the scalar Gaussian FIM. Frequencies are baseband offsets f_k, not fc+f_k: this experiment cancels the clock bias common carrier rotation. Propagation carrier phase remains in the known H^g exactly once. The bound is conditional on that known template and can be more optimistic than joint channel-and-clock estimation.',['Li','Model'],[25,24]);

add('known-channel-phase','Known geometric channel','Receiver phase changes the clock bound','An unknown common phase '+I(R`\psi`)+' is estimated jointly with the clock bias '+I('b')+'.',[
 ...panel('phase-schur',96,190,1088,178,'INFORMATION AFTER ELIMINATING THE COMMON PHASE',M(R`J_{b,\mathrm{eff}}=\frac{8\pi^2}{\sigma^2}\sum_k w_k(f_k-\bar f_w)^2,\qquad\bar f_w=\frac{\sum_kw_k f_k}{\sum_kw_k}`),{size:23,fill:C.cool}),
 ...panel('weighted-spread',96,391,532,248,'ENERGY AND FREQUENCY SPREAD',M(R`\Gamma=\frac{\sum_kw_k}{\sigma^2}`)+M(R`\beta^2=\frac{\sum_kw_k(f_k-\bar f_w)^2}{\sum_kw_k}`),{size:22}),
 ...panel('clock-bound',652,391,532,248,'CLOCK STANDARD-DEVIATION BOUND',M(R`\sigma_{b,\mathrm{LB}}=\frac{1}{2\pi \beta\sqrt{2\Gamma}}`)+ 'At fixed energy, concentrating the received signal in a narrow part of the band raises the timing bound.',{size:22,fill:C.mint})
],'This is the Schur complement of the two-parameter b,psi Fisher matrix. The weighted frequency mean need not vanish even on an arithmetically centered grid. Under a flat energy spectrum, K=3300, B=400MHz and aggregate Gamma=1000 give beta=115.470MHz and a clock standard-deviation lower bound of 30.82ps. Those are illustrative energy assumptions, not a numerical result from a saved RT run. A single nonzero tone with unknown phase cannot identify clock delay.',['Li','Shah']);

two('known-channel-lab','Known channel experiment','What does exact channel knowledge remove?','The next lab separates a timing benchmark from geometric identifiability.',
 'THE KNOWN-TEMPLATE EXPERIMENT','Change aggregate SNR and where the received energy falls in the band.<br><br>Compare a known phase reference with an unknown common phase.<br><br>The displayed clock bound uses the actual weighted second moment or its centered variance.',
 'THE DELAY–CLOCK AMBIGUITY',M(R`\tau_\ell^{\mathrm a}=\tau_\ell^{\mathrm g}+b`)+M(R`(\tau_\ell^{\mathrm g}+\delta)+(b-\delta)=\tau_\ell^{\mathrm a}`)+ 'With free path gains and no geometric constraint, changing both terms this way preserves the effective channel.',
 'Knowing effective Htilde differs from knowing the geometric template H^g. Exact Htilde alone does not establish an absolute geometric delay origin if clock bias is free. The interactive controls show the invariant sum. For the oracle clock bound, H^g and its delay origin are instead given, so b can be identifiable locally. Uniform tones retain the usual global delay ambiguity period 1/Deltaf.',['Model','Li','Shah'],[22,23]);
live('known-channel-live','known-channel-lab','Known channel and clock uncertainty','Change the information assumption and the received frequency spectrum.','known','Inspect four meanings of known channel. Compare the clock bound with known and unknown receiver phase.<br><br>Move received energy across the band while holding aggregate SNR fixed. The weighted spectrum explains the change in the bound.<br><br>Vary geometric delay and clock bias at fixed apparent delay to see an ambiguity that exact effective-channel knowledge cannot remove.','./project/');

add('signal','Orthogonal reference model','Known pilots, unknown channel','The following reference derivation uses balanced orthogonal pilots.',[
 ...panel('signal',96,191,1088,193,'ONE PATH, ONE SUBCARRIER',M(R`\underbrace{\mathbf Y_k}_{N_r\times L}=\underbrace{\alpha e^{-j2\pi f_k\tau}\mathbf a_r(\mathbf q_r)\mathbf a_t(\mathbf q_t)^H}_{\mathbf H_k}\underbrace{\mathbf X_k}_{N_t\times L}+\mathbf N_k`),{size:26,fill:C.cool}),
 ...panel('unknowns',96,409,532,230,'REAL PARAMETER VECTOR',M(R`\boldsymbol\eta=[\tau,\varphi_r,\epsilon_r,\varphi_t,\epsilon_t,\ell,\phi]^T`)+M(R`\alpha=10^{-\ell/20}e^{j\phi}`),{size:22}),
 ...panel('noise',652,409,532,230,'NOISE AND REFERENCES',M(R`n\sim\mathcal{CN}(0,\sigma^2),\quad\mathbb E|n|^2=\sigma^2`)+ 'Independent I and Q each have variance σ²/2. Frequencies and array coordinates are centered.',{size:20,fill:C.mint})
],'Here varphi is azimuth, epsilon elevation, ell effective path attenuation in dB, phi path phase at the reference carrier and array centroids. f_k is baseband frequency relative to the pilot-band center. Unit-modulus steering vector entries have norm squared N, not one. A consistent change to the conjugation convention does not change the derived CRBs. Static path, far-field plane waves, narrowband array steering at carrier, frequency-flat complex gain and white parameter-independent noise.',['Li','Shah']);

two('pilots','02 · identifiability','Transmit pilots must distinguish the antennas','The closed forms use balanced orthogonal coding across the L symbols.',
 'THE PILOT ASSUMPTION',M(R`\mathbf X_k\mathbf X_k^H=\frac{LP}{N_tK}\mathbf I_{N_t}`)+'P is total transmit power over all tones and ports.<br><br>'+M(R`\operatorname{rank}(\mathbf X_k)=N_t\quad\Longrightarrow\quad L\ge N_t`),
 'IF EVERY ANTENNA SENDS THE SAME PILOT',M(R`\mathbf a_t^H\mathbf x=c(\mathbf q_t)`)+ 'One repeated spatial pilot mixes AoD into an unknown complex scalar αc.<br><br><b>Repeating that same vector cannot identify AoD jointly with a free complex gain.</b>',
 'DFT rows provide an explicit orthogonal time-code construction for L>=Nt. The calculator does not invent separated transmit observations when this rank condition fails. Other training arrangements may estimate a structured channel with fewer symbols, but their actual X must enter the FIM. All-ones symbols after already valid despreading are distinct from sending the same spatial pilot on all physical antennas.',['Li','LeMagoarou'],[22,21]);

two('snr','03 · count the information','Aggregate SNR counts all usable signal energy','Γ is the sum of signal power divided by noise power over the observation.',
 'DEFINITION',M(R`\Gamma=\frac{\sum_k\|\boldsymbol\mu_k\|_F^2}{\sigma^2}`)+M(R`\Gamma=N_rKL\rho`)+ 'ρ is the time-code-average SNR per receiver, tone, and symbol.',
 'THREE WAYS TO SET THE DEMO','<b>Aggregate SNR:</b> hold Γ fixed to isolate frequency spread or aperture.<br><br><b>Per-tone SNR:</b> hold ρ fixed. More tones, receivers, or symbols add information.<br><br><b>Power and noise:</b> calculate ρ from the physical link budget.',
 'For unit-modulus steering, XX^H=(LP/(NtK))I gives ||mu||² summed over k = |alpha|² Nr L P. Dividing by sigma² and writing rho=|alpha|²P/(Ksigma²) gives Gamma=NrKLrho. Nt cancels at fixed total radiated training energy. Additional TX elements improve departure-angle aperture, not an automatic Nt-fold gain in Gamma.',['Li','LeMagoarou']);

two('noise-budget','03 · power and receiver noise','The noise normalization stays explicit','A power-normalized FFT convention keeps pilot power and noise in watts.',
 'PER TONE',M(R`\sigma^2=N_0\Delta f,\qquad N_0=k_BT_0F`)+M(R`P_k=\frac P K,\qquad\rho=\frac{|\alpha|^2P}{N_0B}`)+ 'At T₀ = 290 K, the thermal density is approximately −174 dBm/Hz.',
 'THE PHYSICAL SNR CONTROL',M(R`\rho_{\rm dB}=P_{\rm dBm}-\ell- N_{B,\rm dBm}`)+M(R`N_{B,\rm dBm}\simeq-174+\mathrm{NF}+10\log_{10}B`)+ 'NF and attenuation are explicit assumptions. Calibrated element gains belong in the effective path gain.',
 'N0 is complex baseband noise power per Hz. F=10^(NF/10). The calculator rounds 10 log10(kB*290*1000) to -173.975 dBm/Hz. No 3GPP-mandated NF is asserted. Noise figure 9 dB is an adjustable teaching input. CP, guards, inactive tones, beamforming gain and RF losses are excluded unless incorporated consistently in the inputs.',['Li','Noise'],[21,21]);

two('calculator','04 · the complete calculation','A seven-parameter bounds calculator','Every result is the square root of a marginal CRB.',
 'THE CALCULATION ORDER','<b>1.</b> Specify the array geometry and pilot grid.<br><br><b>2.</b> Compute Γ and the frequency spread β.<br><br><b>3.</b> Assemble the delay, angle, gain and phase information.<br><br><b>4.</b> Invert the identifiable blocks.',
 'EXPERIMENTS TO TRY','Double bandwidth at fixed Γ.<br><br>Double L first at fixed Γ, then at fixed per-tone SNR.<br><br>Compare a square array with a line array.<br><br>Select the 384-port preset and inspect its required training time.',
 'The next slide is fully interactive. Controls persist between labs in this browser session. The 384-port example is an ideal single-polarization 24-by-16 array, not a claim that 384 physical radio ports in a real dual-polarized product have that geometry.');
live('calculator-live','calculator','Radio measurement bounds','Change the setup. The intermediate values explain the result.','calculator','Use the calculator to vary both array dimensions, carrier, spacing, pilot count, symbol count, bandwidth, SNR and direction.<br><br>Delay, both AoA angles, both AoD angles, attenuation and phase update together. The link-budget mode also exposes transmit power and receiver noise figure.<br><br>Open the full-size lab from the deck toolbar if more room is useful.');

two('fisher-likelihood','05 · why these equations','Fisher information measures signal sensitivity','A parameter is easier to estimate when it changes the data more distinctly.',
 'COMPLEX GAUSSIAN LIKELIHOOD',M(R`\log p(\mathbf y\mid\boldsymbol\eta)=C-\frac{\|\mathbf y-\boldsymbol\mu(\boldsymbol\eta)\|^2}{\sigma^2}`)+M(R`\mathbf D=\left[\frac{\partial\boldsymbol\mu}{\partial\eta_1},\ldots,\frac{\partial\boldsymbol\mu}{\partial\eta_7}\right]`),
 'THE FISHER INFORMATION MATRIX',M(R`\boxed{\mathbf J=\frac{2}{\sigma^2}\operatorname{Re}(\mathbf D^H\mathbf D)}`)+ 'Large column norm: strong sensitivity.<br><br>Similar columns: parameters can imitate one another.<br><br>The factor 2 follows the complex-noise convention.',
 'For general known colored covariance C, J=2Re(D^H C^-1 D). Parameter-dependent covariance adds a trace term. This simplified model has constant covariance. The likelihood expression stacks all raw receiver samples. An orthogonal sufficient statistic gives the same information if the transformed noise covariance is retained.',['Li','LeMagoarou'],[22,21]);

add('derivatives','05 · the derivatives','Parameter changes in the received signal','For one separated TX–RX coefficient, write its noise-free mean as μ.',[
 table('derivatives',['Parameter','Derivative of the mean','Information comes from'],[
 ['Delay τ',I(R`\partial_\tau\mu=-j2\pi f_k\mu`),'Phase slope across pilot frequencies'],
 ['Path attenuation ℓ (dB)',I(R`\partial_\ell\mu=-\frac{\ln10}{20}\mu`),'Amplitude change across all samples'],
 ['Path phase '+I(R`\phi`),I(R`\partial_\phi\mu=j\mu`),'Common phase change'],
 ['Receive angle q',I(R`\partial_q\mu=j\frac{2\pi}{\lambda}(\mathbf r_r^T\partial_q\mathbf u_r)\mu`),'Phase slope across receive positions'],
 ['Transmit angle q',I(R`\partial_q\mu=-j\frac{2\pi}{\lambda}(\mathbf r_t^T\partial_q\mathbf u_t)\mu`),'Phase slope across distinguishable TX positions']
 ],{fontSize:18,widths:[1,2,1.4]})
],'Angles in these derivatives are radians. The implementation transforms Jacobian columns into ns, degrees, dB and radians before presenting the FIM. Unit conventions affect numeric FIM entries, so a heatmap needs normalized entries or explicit units. The signs reflect H=alpha ar at^H. Equal weights and centered arrays/frequencies eliminate cross-block couplings under the isolated-path model.');

two('fisher','05 · joint estimation','The inverse includes parameter coupling','The reciprocal of a diagonal entry assumes the other unknowns are known.',
 'ONE COUPLED ANGLE PAIR',M(R`\mathbf J_q=\begin{bmatrix}a&b\\b&d\end{bmatrix}`)+M(R`[\mathbf J_q^{-1}]_{11}=\frac{d}{ad-b^2}`)+ 'When b² approaches ad, the marginal variance grows without bound.',
 'NUISANCE PARAMETERS',M(R`\mathbf J_{\rm eff}=\mathbf J_{qq}-\mathbf J_{qn}\mathbf J_{nn}^{-1}\mathbf J_{nq}`)+ 'The second term removes information that an unknown nuisance can explain.<br><br>Inspect the angle coupling in the next lab.',
 'This Schur complement requires the nuisance block to be invertible. For more general singular models, estimability needs separate analysis. The calculator inverts each valid 2x2 angular block exactly and marks unidentifiable coordinates. It never presents a zero from a Moore-Penrose pseudoinverse as an ordinary finite CRB.',['Li','Shah'],[23,22]);
live('fisher-live','fisher','Inside the Fisher information matrix','Inspect the derivatives, information, and marginal bounds for the same setup.','fisher','The diagonal entries measure sensitivity to each parameter. Off-diagonal entries measure similarity between derivative directions.<br><br>The lab exposes the matrix and the numerical steps used to obtain the bounds. Angles away from broadside can couple azimuth and elevation.<br><br>A normalized matrix describes derivative similarity, not estimator correlation.');

two('delay','06 · delay','Delay precision depends on frequency spread','An unknown complex phase makes the pilot center frequency a nuisance.',
 'WEIGHTED RMS FREQUENCY SPREAD',M(R`\bar f=\sum_k w_k f_k,\qquad\beta^2=\sum_k w_k(f_k-\bar f)^2`)+M(R`\boxed{\sigma_{\tau,\rm LB}=\frac{1}{2\pi\beta\sqrt{2\Gamma}}}`)+ 'Weights sum to one and represent relative pilot information.',
 'EQUALLY WEIGHTED CONTIGUOUS TONES',M(R`\beta^2=\frac{\Delta f^2(K^2-1)}{12}`)+M(R`B=K\Delta f,\quad\beta\approx B/\sqrt{12}`)+ 'A common carrier phase can be absorbed into α. Using f<sub>c</sub> as the delay bandwidth would give an unjustified improvement.',
 'Eliminating unknown phase replaces the second frequency moment by its centered variance. With a single tone and unconstrained complex gain, delay is not identifiable. The bandwidth variable B uses K bins, whereas the distance between the outermost tone centers is (K-1)Deltaf. Sparse pilots change the RMS spread and ambiguity structure.',['Li','Shah']);

add('worked-delay','06 · worked example','A numerical delay bound','The teaching example assumes Γ = 1000, corresponding to 30 dB aggregate SNR.',[
 ...panel('numbers',96,191,1088,173,'SUBSTITUTE THE PILOT GRID',M(R`K=3300,\quad B=400\,\mathrm{MHz},\quad\Delta f=121.212\,\mathrm{kHz}`)+M(R`\beta=115.470\,\mathrm{MHz}`),{size:25,fill:C.cool}),
 ...panel('time',96,389,532,241,'DELAY STANDARD-DEVIATION BOUND',M(R`\frac{1}{2\pi(115.470\times10^6)\sqrt{2000}}`)+M(R`\sigma_{\tau,\rm LB}=0.03082\,\mathrm{ns}`),{size:25}),
 ...panel('distance',652,389,532,241,'TOTAL PROPAGATION PATH LENGTH',M(R`\sigma_{s,\rm LB}=c\sigma_{\tau,\rm LB}=9.24\,\mathrm{mm}`)+ 'For a communication path, s includes every segment. Monostatic radial range uses cτ/2.',{size:22,fill:C.mint})
],'Numbers calculated directly from the stated model using c=299792458 m/s. At Γ=1000 and B=400 MHz, K=3300 gives beta=115470048.5 Hz and sigma_tau approximately 3.0820e-11 s. This is an ideal local noise bound conditional on the model, not expected accuracy of current hardware or a full SLAM system.',['Li','Shah']);

two('bandwidth','06 · bandwidth and subcarriers','More tones can also mean more observation time','B, K, and Δf cannot all vary independently for a contiguous pilot grid.',
 'THE GRID RELATIONS',M(R`B=K\Delta f,\qquad T_u=\frac1{\Delta f}`)+M(R`T_{\rm obs}=\frac L{\Delta f}=\frac{LK}{B}`)+ 'The displayed time excludes the cyclic prefix and scheduling gaps.',
 'WHAT STAYS FIXED?', '<b>Fixed Γ and B:</b> increasing a large K barely changes β or the delay bound.<br><br><b>Fixed P, B and L:</b> more K makes symbols longer and adds energy.<br><br><b>Fixed Γ:</b> doubling B halves the delay bound.',
 'For fixed P, noise density, B and L, rho remains fixed but Gamma=Nr K L rho rises with K because useful duration and transmitted energy rise. For fixed power, K and L, doubling B shortens the observation and lowers Gamma, so delay standard deviation improves as B^-1/2 rather than B^-1. Always state what is held fixed.',['Li','Shah']);
live('bandwidth-live','bandwidth','Bandwidth, tone count, and time','Compare actual calculated curves under different resource assumptions.','bandwidth','At fixed aggregate SNR, the delay bound falls in inverse proportion to bandwidth. The tone-count curve nearly saturates when bandwidth and Γ stay fixed.<br><br>Switch to physical link power to include the change in useful observation time and energy.<br><br>The small-K correction and the grid relation B = KΔf come from the same calculation.');

two('array-model','07 · angles in 3D','Array aperture creates spatial phase information','Both arrays lie in their own local yz planes, with +x as broadside.',
 'DIRECTION AND STEERING',M(R`\mathbf u(\varphi,\epsilon)=\begin{bmatrix}\cos\epsilon\cos\varphi\\\cos\epsilon\sin\varphi\\\sin\epsilon\end{bmatrix}`)+M(R`a_m=e^{j(2\pi/\lambda)\mathbf r_m^T\mathbf u}`),
 'CENTERED SPATIAL SPREAD',M(R`\mathbf C_r=\frac1N\sum_m\mathbf r_m\mathbf r_m^T`)+M(R`\mathbf C_r=\operatorname{diag}(0,v_y,v_z)`)+M(R`v_y=\frac{d^2(N_y^2-1)}{12}`)+ 'The z spread has the same form with N_z.',
 'The array origin is its centroid, so the mean element position is zero. Azimuth is measured from +x toward +y and elevation above the xy plane. The displayed directions are local look directions consistent with the chosen steering convention. Physical transmitter and receiver propagation-vector signs must be mapped consistently from a channel simulator.',['Li','Shah'],[22,23]);

two('geometry','07 · angle bounds','Joint azimuth and elevation information','The same construction gives AoA for the RX array and AoD for the TX array.',
 'THE TWO-BY-TWO INFORMATION BLOCK',M(R`\mathbf Q=[\partial_\varphi\mathbf u,\partial_\epsilon\mathbf u]`)+M(R`\boxed{\mathbf J_q=2\Gamma\left(\frac{2\pi}{\lambda}\right)^2\mathbf Q^T\mathbf C_r\mathbf Q}`)+ 'The two angle bounds come from the diagonal of its inverse.',
 'AT BROADSIDE',M(R`\sigma_{\varphi,\rm LB}=\frac{\lambda}{2\pi\sqrt{2\Gamma v_y}}`)+M(R`\sigma_{\epsilon,\rm LB}=\frac{\lambda}{2\pi\sqrt{2\Gamma v_z}}`)+ 'Increasing the horizontal spread helps azimuth. Increasing the vertical spread helps elevation.',
 'Broadside means varphi=epsilon=0. The formulas output radians. A line array cannot jointly identify two arbitrary direction coordinates. At some symmetry points, one coordinate remains locally estimable while the other is not. Planar arrays also have a global front/back ambiguity even where the local angle FIM is full rank. A front-hemisphere assumption selects a branch but does not change that fact.',['Shah','Li'],[22,22]);
live('geometry-live','geometry','Array geometry and angular uncertainty','Rotate the view and change the array, angles, or element spacing.','geometry','Compare a square array with a line array, then move the path toward a grazing direction.<br><br>Observe the local angular uncertainty contour. Its axes come from the joint angle covariance.<br><br>At fixed d/λ, changing carrier also changes physical aperture, so it does not change the ideal angle CRB. Fixed spacing in millimeters gives a different experiment.');

add('angle-example','07 · worked example','A larger transmitting aperture sharpens AoD','A 30 dB aggregate SNR example at broadside, with half-wavelength spacing.',[
 table('angle-values',['Quantity','4 × 4 RX','24 × 16 TX'],[
 ['Number of modeled ports','16','384'],
 ['Horizontal RMS spread, in wavelengths','0.5590 λ','3.4611 λ'],
 ['Vertical RMS spread, in wavelengths','0.5590 λ','2.3049 λ'],
 ['Azimuth standard-deviation bound','0.3648° (AoA)','0.05891° (AoD)'],
 ['Elevation standard-deviation bound','0.3648° (AoA)','0.08847° (AoD)']
 ],{fontSize:20,h:366,widths:[1.6,1,1]}),
 txt('conditions',96,589,1088,52,'Same aggregate Γ for both arrays. Balanced time-coded pilots require L ≥ 384. This is an ideal 24 × 16 geometry with one modeled polarization.',{fontSize:18,color:C.muted})
],'Computed from v=d²(N²-1)/12 and the broadside angular CRB. These values hold aggregate information fixed, so the improvement in AoD comes from spatial spread. At fc=27.2 GHz, lambda=11.0218 mm, but it cancels because spacing is fixed to lambda/2. No extra gain from counting 384 transmitters is applied.',['Shah','Li']);

two('gain','08 · path gain and pathloss','Gain estimates need an amplitude reference','The demo reports effective per-path attenuation and complex phase.',
 'AMPLITUDE AND PHASE',M(R`\ell=-20\log_{10}|\alpha|`)+M(R`\boxed{\sigma_{\ell,\rm LB}=\frac{20}{\ln10\sqrt{2\Gamma}}}`)+M(R`\sigma_{\phi,\rm LB}=\frac1{\sqrt{2\Gamma}}`),
 'AT Γ = 1000',M(R`\sigma_{\ell,\rm LB}=0.1942\,\mathrm{dB}`)+M(R`\sigma_{\phi,\rm LB}=0.02236\,\mathrm{rad}`)+ 'Equivalent to 1.281° in phase.<br><br>Unknown receiver gain cannot be separated from propagation attenuation using a single uncalibrated link.',
 'The complex path coefficient can include propagation, reflection, antenna, polarization and hardware effects. Estimating alpha does not separately identify all those causes. The bound is for one effective path attenuation, not the large-scale pathloss exponent or total multipath received power. At fixed Gamma, changing ell alone has no effect because the SNR input already specifies received information. In link mode, attenuation changes Gamma.',['Li','Shah']);

add('scaling','09 · resource comparisons','A fair comparison states the fixed resources','These scalings describe the isolated-path model with valid orthogonal training.',[
 table('scaling',['Change','What is held fixed','Effect on standard-deviation bounds'],[
 ['Double B','Γ, K, arrays','Delay ÷ 2. Angle and gain bounds unchanged.'],
 ['Double L','ρ, K, arrays','All identifiable bounds ÷ √2.'],
 ['Double L','Γ, K, arrays','No change. Energy was redistributed over time.'],
 ['Add TX elements','Γ, spacing, other array dimensions','AoD improves with aperture. Γ has no automatic Nₜ factor.'],
 ['Increase RX aperture','Γ','AoA improves with aperture.'],
 ['Raise f<sub>c</sub>','Γ and d/λ','Angle bounds unchanged because physical spacing shrinks.']
 ],{fontSize:18,widths:[1,1,2.1]})
],'These are comparisons, not universal hardware laws. At fixed per-tone SNR, adding receive elements also increases Gamma, improving all identifiable parameters. At fixed physical aperture, adding samples has a different scaling from adding half-wavelength elements. Coherence, calibration, RF-chain count, pilot rank and total energy must remain consistent.',['Li','LeMagoarou']);

two('multipath','10 · more than one path','Nearby paths can imitate one another','The full multipath CRB includes every path and its nuisance parameters.',
 'THE JOINT MEAN',M(R`\boldsymbol\mu=\sum_{p=1}^{P}\alpha_p\mathbf s(\tau_p,\mathbf q_{r,p},\mathbf q_{t,p})`)+ 'Stack derivatives for all paths in D.<br><br>Cross-path terms remain in DᴴD. Assigning an independent isolated-path CRB to each unresolved path loses those couplings.',
 'A CONTROLLED TWO-PATH ILLUSTRATION',M(R`\chi(\Delta\tau)=\frac1K\sum_k e^{-j2\pi f_k\Delta\tau}`)+ 'If both delays are known, two unknown complex gains have a variance inflation factor'+M(R`\frac{1}{1-|\chi|^2}`),
 'The next lab isolates gain separability with known delays, identical spatial signatures, equally weighted contiguous tones and white noise. For y=alpha1 s1+alpha2 s2+n, the complex gain covariance bound is sigma²(S^H S)^-1. Relative to a single gain with equal signature energy, each complex gain MSE bound multiplies by 1/(1-|chi|²). The standard deviation multiplier is its square root. Unknown delays/angles require the full joint FIM and generally incur further loss.',['Li','LeMagoarou'],[21,21]);
live('multipath-live','multipath','Two paths and gain separability','A precisely labeled special case makes the loss of information visible.','multipath','Vary the delay separation in the two-path signature.<br><br>At zero separation, the data can identify the sum of the gains but cannot separate the gains. At large correlation, their estimation errors inflate.<br><br>This experiment holds the two delays known. It is not a full joint bound on unknown multipath delays and angles.');

two('ambiguity','10 · local precision and ambiguity','A small local CRB can coexist with ambiguity','The bound describes curvature near one parameter value.',
 'DELAY AMBIGUITY',M(R`e^{-j2\pi k\Delta f(\tau+1/\Delta f)}=e^{-j2\pi k\Delta f\tau}`)+ 'Uniform pilot spacing repeats the signature up to a common phase.<br><br>Timing knowledge must select an appropriate ambiguity interval.',
 'SPATIAL AND MODEL AMBIGUITIES','A planar array can share a steering vector with a direction behind it.<br><br>Large element spacing can produce grating lobes.<br><br>A known path count and correct associations are assumptions of the local calculation.',
 'The conventional resolution scale around 1/B is a different quantity from the local isolated-path estimation CRB, which can be much smaller at high SNR. Close-path resolution also depends on SNR, relative gains, angles and the estimator. A CRB is not a probability of deciding the correct bounce count.',['Li','Shah']);

two('clock','11 · timing and calibration','Clock offset shares the delay signature','A precise apparent delay does not guarantee precise geometric range.',
 'APPARENT DELAY',M(R`\tau_{\rm obs}=\frac{s}{c}+b`)+M(R`\partial_s\boldsymbol\mu=\frac1c\partial_\tau\boldsymbol\mu,\quad\partial_b\boldsymbol\mu=\partial_\tau\boldsymbol\mu`)+ 'The two derivative columns are proportional.',
 'CONSEQUENCE','With one unconstrained path and a free clock offset, s and b cannot be separated.<br><br>The calculator bounds apparent delay. Converting it to geometric path length assumes synchronization or an independently constrained clock.<br><br>Additional BSs need their geometry and clock relations in the model.',
 'This is direct identifiability analysis of the stated likelihood. Clock offset is in seconds and s total path length in meters. Several arbitrary free path lengths plus a common bias remain underdetermined unless geometry or other constraints help. A timing prior is additional information and must be included explicitly. The c*sigma_tau conversion is conditional on known clock bias.',['Shah','Li']);

two('slam','11 · connection to radio SLAM','Channel bounds become geometric information','Use the same propagation model as the radio SLAM factor.',
 'TRANSFORM THE PARAMETERS',M(R`\boldsymbol\eta=\mathbf h(\mathbf x,\mathbf m,\mathbf b)`)+M(R`\mathbf J_{\rm geom}=\mathbf H^T\mathbf J_{\eta}\mathbf H`)+M(R`\mathbf H=\frac{\partial\mathbf h}{\partial[\mathbf x,\mathbf m,\mathbf b]}`),
 'THE MODEL DETERMINES THE INFORMATION','AoA constrains the last segment. AoD constrains the first segment. Delay constrains total length.<br><br>Sum independent link information after mapping every link to common unknowns.<br><br>Eliminate nuisance variables and fix the coordinate gauge before reading pose bounds.',
 'The geometry Jacobian maps channel parameters to UE state, map variables and clock biases. Nuisance information can be eliminated by a Schur complement when the relevant blocks are invertible. CRB covariance is an optimistic model-based floor, not a calibrated covariance to insert automatically into a SLAM factor. Actual estimator errors, correlations and model mismatch require separate validation. This connects with the provided single- and multiple-bounce background.',['Shah','Background']);

two('extensions','12 · extending the model','The same derivative method handles richer signals','The baseline isolates the seven static path parameters.',
 'DOPPLER AND MOTION',M(R`\mu_{k,l}\propto e^{-j2\pi f_k\tau}e^{j2\pi\nu t_l}`)+M(R`\partial_\nu\mu=j2\pi t_l\mu`)+ 'Time spread supplies Doppler information. Moving channels must retain the actual time-coded pilots in the likelihood.',
 'OTHER EXTENSIONS','<b>Hybrid arrays:</b> include precoders and combiners.<br><br><b>Wideband arrays:</b> evaluate steering at f<sub>c</sub> + fₖ.<br><br><b>Near field:</b> use spherical propagation distances.<br><br><b>Diffuse scattering:</b> specify the random channel or covariance model.',
 'The baseline does not assert a valid Doppler CRB after arbitrary time-code despreading. Doppler can break the assumed static orthogonality. Beam squint, near-field curvature, diffuse scattering, phase noise, interference, unknown covariance and hardware calibration must enter the correct mean/covariance model before evaluating J. The general complex-Gaussian FIM has a covariance-derivative trace term when C depends on the unknowns.',['Li','Shah']);

add('formula-sheet','13 · equation sheet','The isolated-path bounds in one place','Angles use radians here. The demo converts angle outputs to degrees.',[
 ...panel('delay-sheet',96,191,532,214,'DELAY',M(R`\sigma_\tau\ge\frac{1}{2\pi\beta\sqrt{2\Gamma}}`)+M(R`\beta^2=\Delta f^2(K^2-1)/12`),{size:25,fill:C.cool}),
 ...panel('angle-sheet',652,191,532,214,'AoA OR AoD',M(R`\mathbf J_q=2\Gamma(2\pi/\lambda)^2\mathbf Q^T\mathbf C_r\mathbf Q`)+M(R`\sigma_{q_i}\ge\sqrt{[\mathbf J_q^{-1}]_{ii}}`),{size:23,fill:C.mint}),
 ...panel('gain-sheet',96,429,532,214,'ATTENUATION AND PHASE',M(R`\sigma_\ell\ge\frac{20}{\ln10\sqrt{2\Gamma}}`)+M(R`\sigma_\phi\ge\frac1{\sqrt{2\Gamma}}`),{size:21}),
 ...panel('energy-sheet',652,429,532,214,'INFORMATION BUDGET',M(R`\Gamma=N_rKL\rho,\quad\rho=\frac{|\alpha|^2P}{N_0B}`)+M(R`\sigma_{s,\rm LB}=c\sigma_{\tau,\rm LB}\quad\text{(known clock)}`),{size:23})
],'All formulas assume an isolated, static, far-field path, calibrated arrays and power, valid balanced orthogonal training, centered flat pilot weighting and white proper complex Gaussian noise. Singular angle or delay directions require estimability analysis, not blind matrix inversion. These closed forms were derived for the explicit teaching model from the cited Fisher-information framework.',['Li','Shah','LeMagoarou']);

add('references','References and further reading','Sources and modeling assumptions','The live values come from the displayed model and its derivatives.',[
 ...panel('primary',96,190,1088,350,'PRIMARY TECHNICAL SOURCES',[
 ['Li','X. Li, V. C. Andrei, U. J. Mönich and H. Boche, 2023. Complex MIMO–OFDM likelihood, waveform dependence, and Fisher information.'],
 ['Shah','A. Shahmansoori et al., IEEE Transactions on Wireless Communications, 2018. Channel parameters and position/orientation information.'],
 ['LeMagoarou','L. Le Magoarou and S. Paquelet, 2020. Pilot design, complex channel estimation, and performance bounds.']
 ].map(([k,description])=>`<a href="${references[k].url}" target="_blank" rel="noopener"><b>${references[k].title}</b></a><br>${description}`).join('<br><br>'),{size:18}),
 txt('assumptions',96,566,1088,70,'<b>Model:</b> One isolated static path, known pilots, white noise, calibrated amplitudes and arrays. Unknown complex gain is included. The two-path lab explicitly holds delays known.',{fontSize:19,color:C.muted})
],'The lesson contains original explanatory derivations and numerical examples for the model stated in the slides. It does not claim to implement a measured hardware performance predictor. General CRB terminology can also be read at https://en.wikipedia.org/wiki/Cram%C3%A9r%E2%80%93Rao_bound . Thermal noise follows kBT and the cited manufacturer technical note.',['Li','Shah','LeMagoarou','Noise']);

for(let i=0;i<slides.length;i++)slides[i].elements.push(txt('slide-number',970,656,214,16,`${String(i+1).padStart(2,'0')} / ${slides.length}`,{fontSize:11,fontFamily:mono,color:C.muted,align:'right'}));
for(const entry of liveMap)slides.find(s=>s.id===entry.slide).elements.push(txt('open-lab',760,657,240,18,'OPEN FULL-SIZE LAB ↗',{fontSize:11,fontFamily:mono,color:C.blue,link:'https://bailiping.com/radio-cramer-rao-slides/'+entry.source.replace(/^\.\//,'')}));
for(const entry of liveMap)entry.slideIndex=slides.findIndex(s=>s.id===entry.slide);
export const deck={format:'bento/slides',version:1,docId:'radio-cramer-rao-slides',title:'Cramér–Rao Bounds for Radio Measurements',readonly:true,meta:{author:'Bai Liping',subject:'MIMO–OFDM channel-parameter Fisher information and Cramér–Rao bounds',company:'bailiping.com',source:'https://bailiping.com/radio-cramer-rao-slides/'},size:{width:1280,height:720},theme:{background:C.bg,color:C.ink,accent:C.blue,fontFamily:serif},slides};
export const inlineLiveMap=liveMap;
