/** Focused mathematical review of the current Bento deck, 2026-09-01.
 * Keep the original four-family structure and live demonstrations.
 * Called before the exported deck and live-slide indices are constructed.
 */
export function applyMathReview(slides, helpers) {
  const { tex, texBlock, mathLines: lines, mathParagraphs: paras, muted, equationSheetSlide: sheet, C } = helpers;
  const getSlide = id => {
    const slide = slides.find(s => s.id === id);
    if (!slide) throw new Error(`Math review: missing slide ${id}`);
    return slide;
  };
  const edit = (slideId, elementId, html, style = {}) => {
    const element = getSlide(slideId).elements.find(e => e.id === elementId);
    if (!element) throw new Error(`Math review: missing ${slideId}/${elementId}`);
    Object.assign(element, { html }, style);
  };
  const after = (id, slide) => {
    if (slides.some(s => s.id === slide.id)) throw new Error(`Duplicate review slide ${slide.id}`);
    slides.splice(slides.findIndex(s => s.id === id) + 1, 0, slide);
  };
  const note = (id, text) => { getSlide(id).notes += '\n\nMATHEMATICAL REVIEW: ' + text; };

  edit('overview', 'cover-family-name-02', 'Affine minimum MSE');
  edit('model', 'model-correct-body', lines(
    tex`\nu_k=z_k-H_km_k^-`,
    tex`S_k=H_kP_k^-H_k^\mathsf{T}+R_k`,
    tex`K_k=P_k^-H_k^\mathsf{T}S_k^{-1}`,
    tex`m_k^+=m_k^-+K_k\nu_k`,
    tex`P_k^+=P_k^--K_kS_kK_k^\mathsf{T}`
  ), { fontSize: 16, lineHeight: 1.18 });
  note('model', 'The initial state must also be Gaussian for exact Gaussian filtering. All noises are independent across time and independent of the initial state. Matrices and inputs are known and deterministic here. The following assumptions sheet defines the conditioning and derives prediction.');

  after('model-live', sheet({
    id: 'assumptions-prediction', family: 'Shared setup',
    title: 'What is assumed, and why prediction works',
    context: `Write ${tex`Z_{k-1}=z_{1:k-1}`}. The exact Bayesian statements use the full linear–Gaussian model, not Gaussian noise alone.`,
    accent: C.green, soft: C.greenSoft,
    panels: [
      { title: 'Initial state and independence', body: lines(
        tex`x_0\sim\mathcal N(m_0^+,P_0^+)`,
        tex`w_k\sim\mathcal N(0,Q_k),\quad v_k\sim\mathcal N(0,R_k)`,
        muted('All variables in the collection of initial state and noises are mutually independent.'),
        muted('Known deterministic matrices and inputs; finite second moments.')
      ), fontSize: 15, lineHeight: 1.43 },
      { title: 'Conditioning hidden by the superscripts', body: paras(
        lines(tex`m_k^-=\mathbb E[x_k\mid Z_{k-1}]`, tex`P_k^-=\operatorname{Cov}(x_k\mid Z_{k-1})`),
        lines(tex`m_k^+=\mathbb E[x_k\mid Z_{k-1},z_k]`, tex`P_k^+=\operatorname{Cov}(x_k\mid Z_{k-1},z_k)`)
      ), fontSize: 16 },
      { title: 'Prediction follows from the error dynamics', body: lines(
        tex`e_{k-1}^+=x_{k-1}-m_{k-1}^+`,
        tex`m_k^-=F_km_{k-1}^++B_ku_k`,
        tex`e_k^-=F_ke_{k-1}^++w_k`,
        tex`P_k^-=F_kP_{k-1}^+F_k^\mathsf T+Q_k`,
        muted('The cross terms vanish by independence and zero noise mean.')
      ), fontSize: 15, lineHeight: 1.43 },
      { title: 'Which inverses are actually required?', body: paras(
        tex`P_0^+,Q_k\succeq0,\quad R_k\succ0\ \Longrightarrow\ S_k\succ0`,
        muted('The covariance-form filter permits singular state covariances.'),
        muted(`The information, density, and Cholesky formulas below additionally assume ${tex`P^-\succ0`}.`)
      ), fontSize: 15 }
    ],
    notes: 'References [1,2]. Condition on all previous observations before deriving a correction. Gaussian closure requires a Gaussian initial state as well as Gaussian independent noises. Prediction covariance itself only needs second moments and zero cross-covariances. A singular state covariance is legitimate; inverse-precision formulas cannot be used unchanged in that case.'
  }));

  after('bayes-equations', sheet({
    id: 'conditioning-bridge', family: 'Family 01',
    title: 'From Gaussian blocks to the Kalman gain',
    context: `Condition throughout on past observations. Set ${tex`P=P^-`}, ${tex`S=HPH^\mathsf T+R`}, and ${tex`J=P^{-1}+H^\mathsf TR^{-1}H`}.`,
    accent: C.green, soft: C.greenSoft,
    panels: [
      { title: 'The joint state–measurement Gaussian', body: paras(
        tex`\begin{bmatrix}x\\z\end{bmatrix}\sim\mathcal N\!\left(\begin{bmatrix}m^-\\Hm^-\end{bmatrix},\begin{bmatrix}P&PH^\mathsf T\\HP&S\end{bmatrix}\right)`,
        tex`\operatorname{Cov}(x,z)=PH^\mathsf T`
      ), fontSize: 16 },
      { title: 'Condition on the observed z', body: paras(
        lines(tex`m^+=m^-+PH^\mathsf TS^{-1}(z-Hm^-)`, tex`P^+=P-PH^\mathsf TS^{-1}HP`),
        tex`K=\operatorname{Cov}(x,z)\operatorname{Cov}(z)^{-1}`
      ), fontSize: 15.3 },
      { title: 'Bridge back to information form', body: paras(
        lines(tex`J^{-1}=P-PH^\mathsf TS^{-1}HP`, tex`J^{-1}H^\mathsf TR^{-1}=PH^\mathsf TS^{-1}=K`),
        muted('The first line is Woodbury. For the second, multiply by J and use S = HPHᵀ + R.')
      ), fontSize: 15.2 },
      { title: 'The normalization is the evidence', body: paras(
        tex`p(z\mid Z_{k-1})=\mathcal N(z;Hm^-,S)`,
        tex`p^+(x)=\frac{p(z\mid x)p^-(x)}{p(z\mid Z_{k-1})}`,
        muted('For fixed model matrices, the observed innovation changes the mean, not the posterior covariance.')
      ), fontSize: 15 }
    ],
    notes: 'Reference [1], Gaussian-conditioning identity and Chapter 4. All quantities are conditioned on the previous observations. The conditioning formula only requires S invertible; the information-form bridge additionally requires P and R invertible. To verify the gain identity, JPHᵀS⁻¹ = Hᵀ[I + R⁻¹HPHᵀ]S⁻¹ = HᵀR⁻¹. These are algebraic identities, not different statistical models.'
  }));

  edit('mse', 'slide-title', 'Affine minimum-MSE estimation');
  edit('mse', 'slide-subtitle', 'Choose the gain within an affine estimator. Gaussianity is not needed for this restricted optimum.');
  edit('mse', 'mse-right-body', paras(
    `For a scalar measurement ${tex`H=h^\mathsf T`}, the correction follows ${tex`r=P^-h`}, not generally ${tex`h`}.`,
    `${tex`P^- - P^+=rr^\mathsf T/S`}: correlation can reduce uncertainty in coordinates that are not directly measured.`,
    muted('Gaussian model: affine optimum = unrestricted MMSE. Otherwise that equality need not hold.')
  ), { fontSize: 15, lineHeight: 1.4 });
  edit('mse', 'mse-note', `Rotate ${tex`H`} and vary correlation. Compare the measurement normal ${tex`h`} with the gain direction ${tex`P^-h`}. They need not align.`);
  edit('mse-equations', 'slide-title', 'Affine minimum MSE · equations');
  note('mse-equations', 'P(K) is the error second-moment matrix under the specified one-step law; K is fixed with respect to the current random measurement. The positive-semidefinite difference proves optimality among affine estimators, not all measurable estimators. Under the Gaussian model, conditioning on the past is understood. With non-Gaussian noise, the recursive Kalman P is an unconditional LMMSE error covariance, not generally the covariance of each realized conditional posterior.');
  edit('mse-live', 'geometry-metric-copy', 'Covariance reduction follows P⁻h, not generally h.', { fontSize: 15.5 });
  edit('mse-live', 'geometry-fallback-status', 'SCHEMATIC FALLBACK · the live experiment computes the exact covariance ellipses');

  after('mse-equations', sheet({
    id: 'covariance-identities', family: 'Family 02',
    title: 'Covariance identities, with their conditions',
    context: `Assume ${tex`\operatorname{Cov}(e^-,v)=0`}. For a fixed gain, ${tex`e(K)=(I-KH)e^- - Kv`}; the error covariance follows directly.`,
    accent: C.blue, soft: C.blueSoft,
    panels: [
      { title: 'Joseph form: any fixed gain', body: paras(
        lines(tex`P(K)=(I-KH)P^-(I-KH)^\mathsf T`, tex`\qquad\qquad+KRK^\mathsf T`),
        muted('A sum of positive-semidefinite terms in exact arithmetic. Numerically safer, not immune to rounding.')
      ), fontSize: 16 },
      { title: 'Compact forms: use the optimal gain', body: paras(
        tex`K_\star=P^-H^\mathsf TS^{-1}`,
        lines(tex`P^+=(I-K_\star H)P^-`, tex`\phantom{P^+}=P^- - K_\star SK_\star^\mathsf T`),
        muted('These shortened identities are not valid for an arbitrary gain.')
      ), fontSize: 15 },
      { title: 'Scalar observation: a rank-one reduction', body: paras(
        tex`H=h^\mathsf T,\quad r=P^-h,\quad S=h^\mathsf TP^-h+R`,
        tex`P^- - P^+=\frac{rr^\mathsf T}{S}\succeq0`,
        tex`a^\mathsf T(P^- - P^+)a=\frac{(a^\mathsf Tr)^2}{S}`
      ), fontSize: 15.1 },
      { title: 'Measuring x can also update y', body: paras(
        tex`P^-=\begin{bmatrix}4&1\\1&1\end{bmatrix},\quad H=\begin{bmatrix}1&0\end{bmatrix},\quad R=1`,
        tex`S=5,\quad K=\begin{bmatrix}0.8\\0.2\end{bmatrix},\quad P^+=\begin{bmatrix}0.8&0.2\\0.2&0.8\end{bmatrix}`,
        muted('The y correction is 0.2 times the innovation, although only x is measured.')
      ), fontSize: 14.8 }
    ],
    notes: 'References [1,2,3,8]. Expand the error outer product to obtain Joseph form. Substitute K*S=P⁻Hᵀ to obtain the two shortened forms. For scalar H=hᵀ the covariance difference is rank at most one. Among unit Euclidean directions a, the greatest absolute variance reduction is along r=P⁻h; directions orthogonal to r have zero variance reduction. This does not mean all ellipse axes preserve their orientation. The numerical example has P⁺ positive definite and eigenvalues 1 and 0.6.'
  }));

  edit('least-squares-equations', 'least-squares-equations-panel-4-body', lines(
    tex`d=\begin{bmatrix}a\\z\end{bmatrix},\quad G=\begin{bmatrix}I\\H\end{bmatrix},\quad W=\operatorname{diag}(P^-,R)`,
    tex`\widehat x=(G^\mathsf TW^{-1}G)^{-1}G^\mathsf TW^{-1}d`,
    tex`\phantom{\widehat x}=a+K(z-Ha)`,
    muted(`Fixed ${tex`x`}; ${tex`a=x+\varepsilon`}, ${tex`z=Hx+v`}.`),
    muted('Errors are zero-mean and uncorrelated, with covariances P⁻ and R.')
  ), { fontSize: 14, lineHeight: 1.35 });
  note('least-squares-equations', 'The quadratic objective is Gaussian MAP. Its inverse Hessian equals the posterior covariance because that posterior is Gaussian; this is not a generic exact covariance formula for nonlinear MAP. BLUE instead treats a as a random unbiased observation of a fixed x, and requires Eε=Ev=0, Covε=P⁻, Covv=R, and Cov(ε,v)=0. Gaussianity is not required for BLUE.');

  edit('kl-equations', 'kl-equations-panel-1-body', paras(
    tex`\ell_z(x)=-\log p(z\mid x)`,
    tex`F(q)=D_{\rm KL}(q\Vert p^-)+\mathbb E_q[\ell_z(x)]`,
    muted('Constants independent of q may be dropped for optimization, but not when computing the evidence.')
  ), { fontSize: 15.5 });
  edit('kl-equations', 'kl-equations-panel-2-body', paras(
    tex`F(q)=D_{\rm KL}(q\Vert p(x\mid z))-\log p(z)`,
    tex`q_\star(x)=\frac{p^-(x)p(z\mid x)}{p(z)}`,
    tex`p(z)=\int p^-(x)p(z\mid x)\,dx`
  ), { fontSize: 15.2 });
  edit('kl-equations', 'kl-equations-panel-4-body', paras(
    tex`h=(P^-)^{-1}m^-+H^\mathsf TR^{-1}z`,
    lines(tex`\nabla_\mu F=J\mu-h=0`, tex`\nabla_\Sigma F=\tfrac12(J-\Sigma^{-1})=0`),
    tex`m^+=J^{-1}h,\quad P^+=J^{-1},\quad \Sigma\succ0`
  ), { fontSize: 15.5 });
  note('kl-equations', 'Reference [6]. All distributions are conditioned on previous observations, suppressed in this sheet. Require a finite positive normalizer and densities absolutely continuous with respect to the prior. The Gaussian reduction assumes P⁻,R,Σ positive definite. The unrestricted variational identity is Bayes for any prior and likelihood with a proper posterior, including nonlinear and non-Gaussian models. Gaussian-restricted variational inference is exact here because the posterior belongs to that family.');

  edit('implementations', 'impl-qr-body', paras(
    tex`P^-=L_pL_p^\mathsf T,\qquad R=L_rL_r^\mathsf T`,
    tex`A=\begin{bmatrix}L_p^{-1}\\L_r^{-1}H\end{bmatrix},\quad b=\begin{bmatrix}L_p^{-1}m^-\\L_r^{-1}z\end{bmatrix}`,
    lines(tex`A=UT,\quad U^\mathsf TU=I,\quad Tm^+=U^\mathsf Tb`, tex`P^+=(T^\mathsf TT)^{-1}=T^{-1}T^{-\mathsf T}`),
    muted(`Use triangular solves; ${tex`\kappa_2(A^\mathsf TA)=\kappa_2(A)^2`}.`)
  ), { fontSize: 15.3, lineHeight: 1.36 });
  note('implementations', 'Reference [8]. This is a square-root information least-squares update, using a thin QR factorization with upper-triangular nonsingular T. The order T⁻¹T⁻ᵀ in the existing slide was already correct. The LQR dual uses A_c=Fᵀ, B_c=Hᵀ, Q_c=Q, R_c=R; finite-horizon recursions run in opposite time directions. A constant-gain claim requires a convergent time-invariant Riccati recursion, not merely constant F,H,Q,R.');
  edit('implementations-live', 'live-prompt', 'Toy significant-digit rounding, not an IEEE simulator. Compare representations—not a universal ranking of solvers.', { fontSize: 14.2 });
  note('implementations-live', 'The reference is the native-double covariance update, not an exact-arithmetic oracle. QR starts from precomputed covariance factors, while the other paths start from covariance matrices. QR covariance symmetry is enforced by mirroring the Gram matrix; it is not independent evidence of accuracy. The reported minimum eigenvalue is that of the symmetric part.');

  after('implementations-live', sheet({
    id: 'boundaries', family: 'Assumptions and extensions',
    title: 'Where equivalence stops—and where it survives',
    context: 'Separate an exact posterior, an affine estimator, and the numerical representation used to compute them.',
    accent: C.rust, soft: C.rustSoft,
    panels: [
      { title: 'Non-Gaussian: MMSE need not be affine', body: paras(
        tex`\widehat x_{\rm MMSE}=\mathbb E[x\mid z]`,
        tex`\widehat x_{\rm affine}=m^-+\operatorname{Cov}(x,z)S^{-1}(z-\mathbb E[z])`,
        muted('With second moments, the second formula is LMMSE. Neither its mean nor its error covariance need describe the conditional posterior.')
      ), fontSize: 14.1 },
      { title: 'Correlated error and measurement noise', body: lines(
        tex`N=\operatorname{Cov}(e^-,v),\quad C_{xz}=P^-H^\mathsf T+N`,
        tex`S=HP^-H^\mathsf T+HN+N^\mathsf TH^\mathsf T+R`,
        tex`K=C_{xz}S^{-1}`,
        tex`m^+=m^-+K(z-Hm^-)`,
        tex`P^+=P^- - C_{xz}S^{-1}C_{xz}^\mathsf T`
      ), fontSize: 14.6, lineHeight: 1.5 },
      { title: 'Singular priors are not invalid priors', body: paras(
        tex`P^-\succeq0,\ R\succ0\ \Longrightarrow\ S\succ0`,
        muted('Gaussian conditioning still works. The state may live on a lower-dimensional affine support.'),
        muted('Do not use P⁻ inverse, log determinant, or an ordinary density on the full space without adapting the representation.')
      ), fontSize: 15 },
      { title: 'Bayes and unrestricted KL still agree', body: paras(
        tex`F(q)=D_{\rm KL}(q\Vert p^+)-\log p(z)`,
        muted('This identity survives nonlinear likelihoods and non-Gaussian priors, provided the posterior is proper.'),
        muted('What generally fails is Gaussian closure and the equality of MAP, posterior mean, and affine optimum.')
      ), fontSize: 15 }
    ],
    notes: 'References [1,2,6]. The correlated-noise box assumes zero means, a valid positive-semidefinite joint covariance for (e⁻,v), and S positive definite. It gives the Gaussian conditional moments when that pair is jointly Gaussian; otherwise it gives the optimal affine estimate and its error covariance under the chosen one-step law. Cross-time noise correlations also affect prediction and require separate modeling, often state augmentation. Do not identify N automatically with same-index Cov(w_k,v_k) without checking the indexing and other correlations.'
  }));

  edit('equivalence', 'equiv-outside-body', 'With non-Gaussian noise, LMMSE and MAP need not equal the posterior mean.<br><br>Correlated noises modify the covariance blocks. Nonlinear models generally break Gaussian closure—but unrestricted KL updating still equals Bayes.', { fontSize: 15, lineHeight: 1.4 });
  note('equivalence', 'The four families are an editorial organization, not four independent estimators or an exhaustive classification. In particular, the unrestricted KL objective is a variational characterization of Bayes, and information form and Gaussian conditioning are algebraic representations of it. Preserve the distinction between statistical assumptions and numerical algorithms.');
}
