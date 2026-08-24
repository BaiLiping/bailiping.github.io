// Source content recovered from the deployed Kalman derivation atlas.
// The deck generator owns presentation structure; this file owns derivation copy.
export const routes = [
  {
    "n": 1,
    "title": "Bayes + completing the square",
    "family": "Probability",
    "idea": "Multiply the Gaussian prior by the Gaussian likelihood, collect the quadratic and linear terms in x, and complete the square.",
    "steps": [
      "p(x|z) ∝ exp{−½‖x−m⁻‖²<sub>P⁻¹</sub> − ½‖z−Hx‖²<sub>R⁻¹</sub>}",
      "The quadratic coefficient is J⁺ = P⁻¹ + HᵀR⁻¹H; the linear coefficient is h⁺ = P⁻¹m⁻ + HᵀR⁻¹z.",
      "Invert J⁺ and apply the matrix inversion lemma to recover the innovation form."
    ],
    "ref": "R. E. Kalman, ‘A New Approach to Linear Filtering and Prediction Problems,’ ASME J. Basic Eng., 1960.",
    "url": "https://doi.org/10.1115/1.3662552"
  },
  {
    "n": 2,
    "title": "Information-form fusion",
    "family": "Probability",
    "idea": "A Gaussian is an additive information object. Independent evidence simply adds precision matrices and information vectors.",
    "steps": [
      "J⁻ = P⁻¹, h⁻ = J⁻m⁻.",
      "J⁺ = J⁻ + HᵀR⁻¹H, h⁺ = h⁻ + HᵀR⁻¹z.",
      "P⁺ = J⁺⁻¹ and m⁺ = P⁺h⁺; Woodbury converts this to K and S."
    ],
    "ref": "R. E. Kalman and R. S. Bucy, ‘New Results in Linear Filtering and Prediction Theory,’ ASME J. Basic Eng., 1961.",
    "url": "https://doi.org/10.1115/1.3658902"
  },
  {
    "n": 3,
    "title": "Joint-Gaussian conditioning",
    "family": "Probability",
    "idea": "Write x and z as one jointly Gaussian vector. The Kalman update is exactly the conditional mean and covariance formula.",
    "steps": [
      "Cov(x,z)=P⁻Hᵀ and Cov(z)=S=HP⁻Hᵀ+R.",
      "E[x|z]=m⁻+Cov(x,z)Cov(z)⁻¹(z−Hm⁻).",
      "Cov(x|z)=P⁻−P⁻HᵀS⁻¹HP⁻."
    ],
    "ref": "T. Kailath, ‘An Innovations Approach to Least-Squares Estimation—Part I,’ IEEE TAC, 1968.",
    "url": "https://doi.org/10.1109/TAC.1968.1099025"
  },
  {
    "n": 4,
    "title": "Hidden-Markov forward recursion",
    "family": "Probability",
    "idea": "Treat the state as the hidden variable and the measurement as the emission. One forward message is prediction; the next is correction.",
    "steps": [
      "Predict: p(xₖ|z₁:ₖ₋₁)=∫p(xₖ|xₖ₋₁)p(xₖ₋₁|z₁:ₖ₋₁)dxₖ₋₁.",
      "Correct: p(xₖ|z₁:ₖ) ∝ p(zₖ|xₖ)p(xₖ|z₁:ₖ₋₁).",
      "Linear–Gaussian closure turns both integrals/products into the familiar Kalman recursions."
    ],
    "ref": "Y. C. Ho and R. C. K. Lee, ‘A Bayesian Approach to Problems in Stochastic Estimation and Control,’ IEEE TAC, 1964.",
    "url": "https://doi.org/10.1109/TAC.1964.1105763"
  },
  {
    "n": 5,
    "title": "Factor graphs + sum–product",
    "family": "Graphical models",
    "idea": "Represent the prior and likelihood as Gaussian factors. Their product is a variable-node belief; Gaussian messages stay Gaussian.",
    "steps": [
      "Prior message: (J⁻,h⁻). Measurement-factor message: (HᵀR⁻¹H,HᵀR⁻¹z).",
      "The variable node sums canonical parameters, producing (J⁺,h⁺).",
      "Forward–backward scheduling gives filtering and smoothing in one graphical language."
    ],
    "ref": "F. R. Kschischang, B. J. Frey, and H.-A. Loeliger, ‘Factor Graphs and the Sum-Product Algorithm,’ IEEE TIT, 2001.",
    "url": "https://doi.org/10.1109/18.910572"
  },
  {
    "n": 6,
    "title": "Gaussian-process conditioning",
    "family": "Probability",
    "idea": "A linear state-space model induces a Gaussian process over time. Conditioning that process on one noisy observation gives the same update.",
    "steps": [
      "Use the GP block covariance between xₖ and zₖ: Kₓz=P⁻Hᵀ.",
      "The observation covariance is Kzz=S.",
      "The standard GP posterior mean KₓzKzz⁻¹ residual and Schur-complement covariance are the Kalman equations."
    ],
    "ref": "J. Hartikainen and S. Särkkä, ‘Kalman Filtering and Smoothing Solutions to Temporal Gaussian Process Regression Models,’ IEEE MLSP, 2010.",
    "url": "https://doi.org/10.1109/MLSP.2010.5589113"
  },
  {
    "n": 7,
    "title": "MAP / weighted least squares",
    "family": "Optimization",
    "idea": "The posterior mode minimizes the negative log posterior. For Gaussians, that objective is a weighted least-squares problem.",
    "steps": [
      "minₓ ½‖x−m⁻‖²<sub>P⁻¹</sub> + ½‖z−Hx‖²<sub>R⁻¹</sub>.",
      "Set the gradient to zero: (P⁻¹+HᵀR⁻¹H)x=P⁻¹m⁻+HᵀR⁻¹z.",
      "Solve the normal equations and use Woodbury to expose the Kalman gain."
    ],
    "ref": "B. M. Bell and F. W. Cathey, ‘The Iterated Kalman Filter Update as a Gauss–Newton Method,’ IEEE TAC, 1993.",
    "url": "https://doi.org/10.1109/9.250476"
  },
  {
    "n": 8,
    "title": "Recursive least squares + Woodbury",
    "family": "Optimization",
    "idea": "Append the new measurement rows to an existing least-squares system, then update the inverse normal matrix recursively.",
    "steps": [
      "The new normal matrix is J⁺=J⁻+HᵀR⁻¹H.",
      "Apply the Woodbury identity instead of reinverting J⁺ from scratch.",
      "The rank-m correction is P⁺=P⁻−P⁻HᵀS⁻¹HP⁻, with the same gain K."
    ],
    "ref": "R. L. Plackett, ‘Some Theorems in Least Squares,’ Biometrika, 1950.",
    "url": "https://doi.org/10.1093/biomet/37.1-2.149"
  },
  {
    "n": 9,
    "title": "Square-root / QR derivation",
    "family": "Numerical linear algebra",
    "idea": "Whiten the prior and measurement equations, stack them, and solve by orthogonal triangularization rather than forming covariance products.",
    "steps": [
      "Stack P⁻¹/²(x−m⁻) and R⁻¹/²(z−Hx).",
      "QR factorization turns the stacked system into an upper-triangular posterior information square root.",
      "Back-substitution gives m⁺; the triangular factor gives P⁺ while preserving positive definiteness better."
    ],
    "ref": "P. G. Kaminski, A. E. Bryson Jr., and S. F. Schmidt, ‘Discrete Square Root Filtering,’ IEEE TAC, 1971.",
    "url": "https://doi.org/10.1109/TAC.1971.1099816"
  },
  {
    "n": 10,
    "title": "LMMSE + orthogonality principle",
    "family": "Estimation",
    "idea": "Restrict the estimate to m⁺=m⁻+Kν and choose K so that the estimation error is orthogonal to every linear function of the innovation.",
    "steps": [
      "ν=z−Hm⁻, Cov(x−m⁻,ν)=P⁻Hᵀ, Cov(ν)=S.",
      "Orthogonality requires E[(x−m⁻−Kν)νᵀ]=0.",
      "Therefore K=P⁻HᵀS⁻¹; substituting gives the posterior covariance."
    ],
    "ref": "T. Kailath, ‘An Innovations Approach to Least-Squares Estimation—Part I,’ IEEE TAC, 1968.",
    "url": "https://doi.org/10.1109/TAC.1968.1099025"
  },
  {
    "n": 11,
    "title": "Direct covariance minimization",
    "family": "Estimation",
    "idea": "Write the posterior error covariance as a function of an arbitrary gain and minimize its trace (or any positive weighted trace).",
    "steps": [
      "P⁺(K)=(I−KH)P⁻(I−KH)ᵀ+KRKᵀ.",
      "Differentiate tr P⁺(K) with respect to K and set the matrix gradient to zero.",
      "K(HP⁻Hᵀ+R)=P⁻Hᵀ, so K=P⁻HᵀS⁻¹."
    ],
    "ref": "R. E. Kalman, ‘A New Approach to Linear Filtering and Prediction Problems,’ ASME J. Basic Eng., 1960.",
    "url": "https://doi.org/10.1115/1.3662552"
  },
  {
    "n": 12,
    "title": "BLUE / Gauss–Markov",
    "family": "Estimation",
    "idea": "Among all linear unbiased combinations of the prior estimate and the new measurement, choose the one with minimum covariance.",
    "steps": [
      "Let m⁺=Bm⁻+Az and impose unbiasedness B+AH=I.",
      "Substitute B=I−AH and minimize the error covariance over A.",
      "The optimum A is P⁻HᵀS⁻¹, hence A=K and B=I−KH."
    ],
    "ref": "A. C. Aitken, ‘On Least Squares and Linear Combination of Observations,’ Proc. Royal Soc. Edinburgh, 1936.",
    "url": "https://doi.org/10.1017/S0370164600014346"
  },
  {
    "n": 13,
    "title": "Wiener filtering + innovations",
    "family": "Estimation",
    "idea": "Project the state onto the space generated by past measurements, then add the component carried by the new orthogonal innovation.",
    "steps": [
      "The innovation νₖ is orthogonal to all earlier measurements.",
      "Its projection coefficient is Cov(xₖ,νₖ)Cov(νₖ)⁻¹.",
      "Those covariances are P⁻Hᵀ and S, so the causal Wiener projection becomes the Kalman update."
    ],
    "ref": "T. Kailath, ‘An Innovations Approach to Least-Squares Estimation—Part I,’ IEEE TAC, 1968.",
    "url": "https://doi.org/10.1109/TAC.1968.1099025"
  },
  {
    "n": 14,
    "title": "Kalman–LQR duality",
    "family": "Control duality",
    "idea": "The estimator Riccati equation is the transpose-dual of the regulator Riccati equation. The observer gain is the dual feedback gain.",
    "steps": [
      "Exchange A↔Aᵀ, H↔Bᵀ, process covariance↔state cost, and R↔control cost.",
      "The LQR completion-of-squares/Riccati argument maps to minimum-variance estimation.",
      "Under the dual map, the feedback gain becomes the Kalman gain."
    ],
    "ref": "R. E. Kalman, ‘Contributions to the Theory of Optimal Control,’ Bol. Soc. Mat. Mexicana, 1960.",
    "url": "https://www.cs.unc.edu/~welch/kalman/media/pdf/Kalman1960.pdf"
  },
  {
    "n": 15,
    "title": "Recursive Bayesian filtering",
    "family": "Probability",
    "idea": "Start from the general Bayesian filter, then specialize the transition and likelihood to linear Gaussians. Closure supplies the finite-dimensional recursion.",
    "steps": [
      "Prediction is Chapman–Kolmogorov integration through the transition density.",
      "Correction is Bayes’ rule with the current likelihood.",
      "Evaluating the Gaussian integral and product yields the standard prediction and measurement updates."
    ],
    "ref": "Y. C. Ho and R. C. K. Lee, ‘A Bayesian Approach to Problems in Stochastic Estimation and Control,’ IEEE TAC, 1964.",
    "url": "https://doi.org/10.1109/TAC.1964.1105763"
  },
  {
    "n": 16,
    "title": "Minimum surprise / maximum relative entropy",
    "family": "Information theory",
    "idea": "Update the prior as little as possible—measured by KL divergence—while enforcing the information supplied by the observation and the linear–Gaussian constraints.",
    "steps": [
      "q* = arg min<sub>q∈C</sub> D<sub>KL</sub>(q ‖ p⁻), where C encodes normalization and the measurement/moment constraints.",
      "The variational solution is an exponential tilt of p⁻; for quadratic Gaussian constraints it remains Gaussian.",
      "Its canonical parameters become J⁺=J⁻+HᵀR⁻¹H and h⁺=h⁻+HᵀR⁻¹z, hence the same K, m⁺ and P⁺."
    ],
    "ref": "A. Giffin and R. Urniezius, ‘The Kalman Filter Revisited Using Maximum Relative Entropy,’ Entropy, 2014; P. R. Kalata and R. Priemer, Information Sciences, 1979.",
    "url": "https://doi.org/10.3390/e16021047"
  }
];
