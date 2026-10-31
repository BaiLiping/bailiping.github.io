# Cramér–Rao Bound — presenter notes

23 slides · three live labs · approximately 30–40 minutes.

## 01. Cramér–Rao Bound

How precise could an estimate possibly be?

Suggested pacing: 30–40 minutes including experiments. The main prerequisites are expectation, variance, derivatives, and basic matrices. Start by asking: “Could a cleverer algorithm always beat today’s error?” The answer depends on the measurement model and the estimator class. The scalar CRB is a pointwise frequentist lower bound, not a guarantee that an estimator attaining it exists. The browser deck is self-contained and works offline; the PDF and PowerPoint are static snapshots. Use arrow keys to navigate, O for overview, N for these notes, and F for fullscreen. Each lab preserves state while you navigate.

## 02. An estimate is one outcome. Precision is a distribution.

Keep the unknown parameter fixed. Repeat the entire measurement process.

Make the distinction between a measurement distribution and a sampling distribution explicit. A sample of 16 observations is one trial, not 16 estimates. Monte Carlo repeats the entire trial many times. The CRB constrains the estimator variance at a fixed true parameter when unbiasedness holds in a neighborhood and regularity conditions permit the score identity. It does not impose a minimum absolute error on each realized estimate.

## 03. Nearby worlds are hard to tell apart.

Information measures how sensitively the data distribution changes when the parameter moves.

The word local matters. Fisher information describes infinitesimal changes in the distribution around the true parameter; it can miss distant competing likelihood modes. The two curve pairs use the same mean displacement and different known Gaussian noise levels. This is an intuition illustration, not a classifier or a posterior. In the Gaussian location model the information per observation is exactly 1/σ². That calculation will appear shortly.

## 04. Likelihood → score → Fisher information.

The likelihood is a function of the candidate parameter, with the observed dataset held fixed.

I_n always means total information in the whole dataset. I_1 is reserved for one observation. The expected Fisher information is a model property evaluated at the true parameter. Observed information is minus the second derivative of the realized log likelihood and is generally random. They coincide in the known-variance Gaussian mean example, but not in every model. A normalized likelihood plotted with maximum 1 is not a posterior density.

## 05. The scalar Cramér–Rao Bound.

A floor for the variance of locally unbiased estimators in a regular model.

An accessible sufficient setting is an interior parameter in a smooth family with parameter-independent support, dominated derivatives that justify differentiation under integration, finite positive Fisher information, and a finite-variance estimator. These are sufficient conditions rather than a complete necessary checklist. Local unbiasedness at θ means both the correct expectation and derivative equal to one there. A bound does not promise attainability. No single realized error or finite Monte Carlo variance is required to sit above the theoretical floor.

## 06. One score identity. One Cauchy–Schwarz inequality.

An unbiased estimator must move, on average, at the same rate as the true parameter.

Because T is a statistic that does not itself depend on the unknown θ, differentiation acts on the density. Formally, ∂θ Eθ[T] = ∫ T(x) ∂θp(x;θ) dx = Eθ[TUθ]. Score mean zero follows by differentiating ∫ p(x;θ)dx = 1 under the integral. Cauchy–Schwarz then gives the result. For a biased statistic with expectation m(θ), replace 1 by m′(θ). That yields the biased extension used later. Regularity is not decorative: the uniform example breaks this step.

## 07. For a Gaussian mean, averaging is already optimal.

Assume independent measurements Xi ∼ N(μ, σ²), with σ known.

The Gaussian location family with known positive variance is regular. Each observation contributes information 1/σ². The sample mean has exact distribution N(μ,σ²/n), so it is unbiased and efficient at every sample size. This is not a generic statement about maximum likelihood estimators. The first-observation estimator X₁ is also unbiased, but discards n−1 measurements and has variance σ². Its efficiency relative to the full-data CRB is 1/n.

## 08. What will happen when you collect more measurements?

In the next slide, every histogram bar comes from genuinely simulated datasets.

Ask the audience to give both variance and standard-deviation answers. Quadrupling n divides variance by four and standard deviation by two. Doubling σ multiplies variance by four and standard deviation by two. Collecting more data does not help an estimator that ignores it. A finite empirical variance can fall slightly below the theoretical bound simply from Monte Carlo fluctuation. The live display distinguishes exact estimator variance, sample variance over trials, and the full-data bound.

## 09. Can you reach the precision floor?

Change the experiment, repeat it, and compare empirical spread with the exact bound.

Begin with the sample mean, n=16 and σ=2. The empirical variance should be near 0.25. Slide n to 64: the exact variance and CRB become 0.0625. Select “First observation”: its variance is now 4 even though the full-data CRB remains 0.0625. The teal curve is the exact efficient sample-mean density for this Gaussian model, not a universal probability distribution associated with a CRB. The rust dashed curve is the chosen estimator’s exact density. At the default estimator these two curves coincide. The code draws all n measurements independently using a seeded Box–Muller generator. Empirical variance uses divisor R−1; more trials reduce Monte Carlo variation. Reset restores defaults and exactly replays the initial 2,000 trials.

## 10. Four times the data. Half the standard deviation.

Information adds for independent measurements; variance and standard deviation scale differently.

For independent but non-identical regular observations, information is the sum of their individual information contributions. The nI₁ formula additionally assumes identical information. Correlation does not automatically imply less information for every possible model, but it invalidates counting repeated correlated observations as independent. In a Gaussian mean model with known covariance R, information is 1ᵀR⁻¹1. The two plotted curves have different units and are separately labeled: variance in squared parameter units, SD in parameter units.

## 11. A narrow cloud can be centered in the wrong place.

The unbiased CRB is not a universal lower bound on mean squared error.

Derivation: write m(θ)=Eθ[T]=θ+b(θ), so Cov(T,U)=m′(θ)=1+b′(θ). Cauchy–Schwarz gives the biased variance lower bound, and adding squared bias gives the MSE bound. This is a fixed-estimator, differentiable-bias statement; neither the estimator nor its tuning rule can secretly use the unknown true θ. A biased estimator can have MSE below the unbiased CRB at some θ, without contradiction. In the lab the shrinkage factor is selected by the viewer to inspect pointwise performance, not optimized by an estimator that knows μ.

## 12. Can you “beat” the bound by shrinking toward zero?

Watch the variance fall, then move the true mean away from the shrinkage target.

The model has n=16, σ=2, so v=σ²/n=0.25. The shrinkage estimator is Tα=αX̄. It has expectation αμ, bias (α−1)μ, variance α²v, and MSE α²v+(1−α)²μ². Its bias derivative is α−1, so the biased variance bound is exactly α²v and is attained. At α=0 the distribution is a point mass at zero: the plot draws a spike, not a Gaussian density with zero variance. Try μ=0.35 then μ=2 with α=0.55; the same reduction in variance becomes a large bias penalty. The plots and metrics are analytic, not simulated. A prior is not needed for this example; the zero anchor is just a fixed shrinkage target.

## 13. “Lower bound” does not mean “always achievable.”

Equality requires a very specific relationship between the estimator and the score.

The displayed equality condition is for the scalar unbiased bound at the parameter value being considered, with nonzero finite information. Equality in Cauchy–Schwarz forces T−θ to be proportional to U; covariance 1 fixes the constant to 1/I. To attain the bound for every parameter, the right side plus θ must not require knowing the unknown θ. Asymptotic normality is a distributional statement; translating it into variance convergence requires appropriate moment control. Avoid claiming finite-sample MLE optimality merely from the asymptotic theorem.

## 14. Information has directions.

A vector parameter has a covariance matrix—not a single uncertainty number.

Here u=∇θ log p(X;θ) is the score vector, and J is the total Fisher information matrix. The symbol ≽ denotes positive-semidefinite order: every quadratic form of Cov−J⁻¹ is nonnegative. It is not an element-by-element comparison. For positive-definite J, covariance-bound eigenvalues are reciprocal information eigenvalues. A covariance ellipse is a geometric visualization; probability coverage requires a distributional assumption and the appropriate radius. We use unit Mahalanobis radius and do not label it as a confidence region.

## 15. One range measurement constrains one local direction.

A simplified 2D ranging model: known anchor positions, independent Gaussian range noise.

Differentiate h_i(p)=||p−a_i||. Its gradient is the unit vector from anchor i to the target. For independent additive Gaussian noise of known, position-independent variance, information is the sum of σ_i⁻² times the outer product of that gradient. This toy model starts from noisy ranges, not raw waveforms; it is inspired by localization-information geometry but is not a reproduction of the full wideband model in Shen and Win. With constant range-noise variance, moving an anchor along the same ray changes neither its information direction nor weight. Real range accuracy may depend on distance, SNR, multipath, or visibility.

## 16. Same sensors. Different geometry. Different limits.

Four accurate ranges can still leave one position direction poorly constrained.

The PEB is the square root of trace J⁻¹ and has distance units. It lower-bounds RMSE of the full Euclidean position error, under the relevant unbiasedness and regularity conditions. It is neither a 95% radius nor the mean Euclidean error. Four anchors arranged in a line can still give full local rank when the target is off that line, even though a global reflection ambiguity remains. The preset named Collinear puts both anchors and target on the same line to deliberately produce local rank one. The demo reports singularity rather than drawing a falsely finite uncertainty ellipse.

## 17. Move the sensors. Reshape the bound.

Drag any anchor or the target. The Fisher matrix and its inverse update immediately.

Drag or use the selection and coordinate sliders; every operation is accessible without pointer dragging. Coordinates and σ are in meters. The four anchor directions define J=Σuᵢuᵢᵀ/σ². The target is the true evaluation point, not a fitted estimate. With the common-offset toggle, the displayed matrix is the equivalent position information J_e after eliminating β by a Schur complement. The ellipse is centered at the truth and has principal radii 1/√λ; it is not simulated estimator output or a confidence ellipse. If eigenvalues are numerically zero, the UI gives a rank-deficient status and no finite PEB. Near-anchor coincidence is flagged because the range derivative fails there. Long ellipses can extend beyond the field of view; this is explicitly labeled. No nonlinear localization solver is being claimed.

## 18. An unknown clock offset consumes position information.

The same measurements must now explain both the target position and a common range offset β.

For z_i=||p−a_i||+β+ε_i with common σ, A=Σuᵢuᵢᵀ/σ², b=Σuᵢ/σ², and c=m/σ². Thus J_e=(Σuᵢuᵢᵀ−(Σuᵢ)(Σuᵢ)ᵀ/m)/σ². This is exactly the toggle implemented in the geometry lab. The clock-offset analogy uses β in distance units, so a time offset would be multiplied by propagation speed first. The block A is conditional information when β is known; inverting only A when β is unknown is overly optimistic. Orthogonal parameter coupling can make the loss zero.

## 19. A moving support breaks the familiar proof.

Not every model allows differentiation to pass through the expectation.

Here X_(n) is the sample maximum. Its expectation is nθ/(n+1) and its variance is nθ²/((n+1)²(n+2)); multiplying by (n+1)/n gives the displayed unbiased estimator and variance. For example n=4 gives variance θ²/24. Blindly using E[U²]=n²/θ² and then applying the regular unbiased inequality would suggest θ²/n², which is contradicted by this unbiased estimator precisely because E[U]≠0 and the regular score identity fails. Parameter-independent support is a common sufficient regularity condition, not a necessary condition for every possible information inequality. A nonregular model needs a suitable alternative argument.

## 20. Design the experiment, not only the estimator.

A new independent, correctly modeled measurement adds a positive-semidefinite information term.

The displayed linear Gaussian formula assumes R is known, positive definite, and independent of θ. With parameter-dependent covariance the general Gaussian information also contains covariance-derivative terms. For full column-rank H, generalized least squares is unbiased and attains this CRB exactly. For nonlinear h(θ), replace H by the Jacobian evaluated at θ to compute the local information for this additive Gaussian model; this does not guarantee that a nonlinear estimator attains the bound. Trace, determinant, and minimum-eigenvalue criteria describe different uncertainty summaries and can depend strongly on scaling the parameter coordinates. The objective choices are derived from the covariance ellipsoid.

## 21. Read a CRB result like a careful reviewer.

Before comparing an algorithm with a bound, make sure they solve the same statistical problem.

Four common mistakes are comparing the unbiased CRB against a biased estimator’s MSE without qualification; omitting nuisance variables; plotting an RMSE against a variance lower bound; and interpreting a well-conditioned local Hessian as proof of global identifiability. A finite Monte Carlo estimate is also random: use enough trials and report uncertainty. The demo intentionally makes these caveats visible rather than hiding them behind attractive ellipses. The CRB is a diagnostic and design tool, not an algorithm or a universal confidence interval.

## 22. Three ideas to keep.

More information lowers the bound. Bias changes the comparison. Geometry decides the weak directions.

Return to the opening question. There are two distinct levers: choose a better estimator for the available data, or improve what the data reveal through sensing design. A CRB gap does not automatically show algorithmic weakness: finite-sample nonattainability, bias, nuisance variables, or model mismatch may explain it. Close with the demonstration that moving anchors changes the information before running any localization algorithm. Optional discussion: which nuisance parameter or correlation is easiest to overlook in the audience’s own application?

## 23. Continue from the bound to the bigger picture.

The equations and examples are teaching material; the sources below give the wider statistical context.

Source mapping: S1 grounds the scalar proof, Gaussian calculation, and efficiency discussion. S2 gives broad information and estimation context. S3 grounds localization-information geometry and equivalent Fisher information; this deck’s independent Gaussian-range toy model is deliberately simpler than that paper’s waveform-level model. S4 supports the carefully qualified asymptotic MLE statement. The biased extension, uniform-maximum calculation, linear Gaussian design consequences, figures, and numerical demos are independently derived teaching examples. Links navigate in the same tab, following the site convention. The existing site’s warm paper, Georgia headings, and restrained teal/blue/rust palette informed the visual design.

