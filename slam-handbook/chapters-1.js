/* Original teaching material; section references point to the SLAM Handbook. */
window.SLAM_CHAPTERS = (window.SLAM_CHAPTERS || []).concat([
  {
    "n": 1,
    "title": "Factor Graphs for SLAM",
    "pages": "19–52",
    "goal": "Understand how many uncertain local measurements become one globally consistent estimate.",
    "ideas": [
      {
        "title": "Measurements become factors",
        "text": "A factor graph separates unknown variables (robot poses, landmarks, biases) from the functions constraining them. A loop closure connects a later pose to an earlier one, providing a long-range consistency check missing from odometry alone. Factors need only involve the states necessary to predict their measurements.",
        "equation": "p(x|z) ∝ ∏ᵢ φᵢ(xᵢ)",
        "ref": "§1.1, pp.20–25"
      },
      {
        "title": "Probabilities become weighted residuals",
        "text": "For Gaussian measurement errors, maximizing the posterior is equivalent to minimizing the sum of squared, whitened residuals. Dividing by uncertainty is not cosmetic: it determines whether a precise measurement or a noisy one dominates an estimate. A prior also removes otherwise unobservable global frame freedom.",
        "equation": "x* = argminₓ Σᵢ ‖zᵢ − hᵢ(xᵢ)‖²_Σᵢ",
        "ref": "§1.2, pp.25–29"
      },
      {
        "title": "Sparsity makes large problems solvable",
        "text": "Each factor touches only a few variables, creating sparse Jacobians. Eliminating variables introduces fill-in; the elimination order changes computation, not the answer. iSAM2 uses a Bayes tree to refactor affected cliques rather than rebuild everything for every new measurement.",
        "equation": "Λ = AᵀA;  Λ = RᵀR",
        "ref": "§1.5–1.7, pp.35–51"
      }
    ],
    "demos": [
      {
        "key": "loop",
        "title": "Close the loop",
        "question": "How strongly should a final visit pull a drifting odometry chain toward its starting point?",
        "params": [
          {
            "key": "drift",
            "label": "Cumulative odometry drift",
            "min": 0,
            "max": 8,
            "step": 0.2,
            "value": 3,
            "unit": "m"
          },
          {
            "key": "loop",
            "label": "Loop-closure weight",
            "min": 0,
            "max": 30,
            "step": 1,
            "value": 12,
            "unit": ""
          }
        ],
        "tryThis": "Set loop weight to zero, then increase it. Does the correction spread across the chain?",
        "limit": "1D linear-Gaussian chain; this is not full SE(2) pose-graph optimization.",
        "ref": "§1.2, 1.4"
      },
      {
        "key": "precision",
        "title": "Let precise data speak",
        "question": "Which of two contradictory observations controls a fused scalar estimate?",
        "params": [
          {
            "key": "z2",
            "label": "Second measurement",
            "min": 2,
            "max": 12,
            "step": 0.2,
            "value": 9,
            "unit": "m"
          },
          {
            "key": "sigma",
            "label": "Second-measurement σ",
            "min": 0.2,
            "max": 6,
            "step": 0.2,
            "value": 2,
            "unit": "m"
          }
        ],
        "tryThis": "Reduce σ and observe the weighted mean move toward the second observation.",
        "limit": "Independent scalar Gaussian observations with a flat prior.",
        "ref": "§1.2.2"
      },
      {
        "key": "fill",
        "title": "Order the elimination",
        "question": "How many graph edges are created when a hub is eliminated early?",
        "params": [
          {
            "key": "nodes",
            "label": "Number of leaves",
            "min": 3,
            "max": 12,
            "step": 1,
            "value": 8,
            "unit": ""
          },
          {
            "key": "hubFirst",
            "label": "Eliminate hub first (0/1)",
            "min": 0,
            "max": 1,
            "step": 1,
            "value": 1,
            "unit": ""
          }
        ],
        "tryThis": "Compare hub-first and leaf-first elimination; both target the same solution.",
        "limit": "Counts induced fill edges in a star graph, not factorization FLOPs.",
        "ref": "§1.5–1.6"
      }
    ],
    "checks": [
      {
        "q": "Why is an anchor usually required?",
        "a": "Relative pose and landmark measurements do not determine a unique global origin and orientation; fixing one pose or adding a prior removes gauge freedom."
      },
      {
        "q": "Does a sparser Jacobian guarantee a sparse Cholesky factor?",
        "a": "No. Sparse elimination may create fill-in. Choosing a good ordering can dramatically reduce the factorization cost."
      }
    ],
    "watch": "Loop closures can be false; Chapter 3 explains why an inconsistent constraint cannot always be trusted."
  },
  {
    "n": 2,
    "title": "Advanced State Variable Representations",
    "pages": "53–74",
    "goal": "Use appropriate geometry for poses and make sense of trajectories between measurement times.",
    "ideas": [
      {
        "title": "Rotations are not free vectors",
        "text": "A rotation matrix belongs to SO(2) or SO(3): its columns are orthonormal and its determinant is +1. Adding matrix entries arbitrarily leaves this space. Instead, apply an update in a local tangent vector and map it through the exponential map so the estimate stays on the manifold.",
        "equation": "R_new = R_old Exp(δθ)",
        "ref": "§2.1.1–2.1.3, pp.54–59"
      },
      {
        "title": "Pose composition remembers frames",
        "text": "SE(2) and SE(3) combine translation and orientation. Rotating then translating is generally not the same as translating then rotating; left and right perturbations have different frame interpretations. Frame labels and retractions are part of the mathematics, not mere notation.",
        "equation": "T = [ R  t ; 0  1 ]",
        "ref": "§2.1.1–2.1.5, pp.54–62"
      },
      {
        "title": "Time is a continuous variable",
        "text": "Asynchronous IMUs, rolling-shutter cameras and spinning LiDARs observe the world at many different instants. A spline or an SDE-derived Gaussian process lets us query a smooth trajectory at arbitrary timestamps without storing a pose per measurement. Local spline support and Markov priors preserve sparsity.",
        "equation": "p(t) = Σₖ Ψₖ(t)cₖ",
        "ref": "§2.2.1–2.2.4, pp.62–74"
      }
    ],
    "demos": [
      {
        "key": "angles",
        "title": "Navigate the ±π seam",
        "question": "What happens when you average headings just below +180° and just above −180°?",
        "params": [
          {
            "key": "a",
            "label": "Heading A",
            "min": 150,
            "max": 180,
            "step": 1,
            "value": 175,
            "unit": "°"
          },
          {
            "key": "b",
            "label": "Heading B",
            "min": -180,
            "max": -150,
            "step": 1,
            "value": -175,
            "unit": "°"
          }
        ],
        "tryThis": "Compare raw arithmetic mean to the circular mean.",
        "limit": "Two planar angles; ambiguous antipodal averages are flagged.",
        "ref": "§2.1"
      },
      {
        "key": "rigid",
        "title": "Compose a planar pose",
        "question": "Where does a body-frame point land after rotation and translation?",
        "params": [
          {
            "key": "theta",
            "label": "Heading",
            "min": 0,
            "max": 180,
            "step": 5,
            "value": 55,
            "unit": "°"
          },
          {
            "key": "tx",
            "label": "Translation x",
            "min": -4,
            "max": 4,
            "step": 0.2,
            "value": 1,
            "unit": "m"
          },
          {
            "key": "ty",
            "label": "Translation y",
            "min": -4,
            "max": 4,
            "step": 0.2,
            "value": 0,
            "unit": "m"
          }
        ],
        "tryThis": "Change heading without moving the translation: compare the transformed point.",
        "limit": "One SE(2) transform applied to a single landmark.",
        "ref": "§2.1.1"
      },
      {
        "key": "spline",
        "title": "Sample between keyframes",
        "question": "How do measurement time and knot spacing affect interpolation?",
        "params": [
          {
            "key": "t",
            "label": "Query time",
            "min": 0,
            "max": 10,
            "step": 0.1,
            "value": 4.3,
            "unit": "s"
          },
          {
            "key": "spacing",
            "label": "Knot interval",
            "min": 1,
            "max": 5,
            "step": 0.5,
            "value": 2,
            "unit": "s"
          },
          {
            "key": "wiggle",
            "label": "Trajectory curvature",
            "min": 0,
            "max": 2,
            "step": 0.1,
            "value": 1,
            "unit": ""
          }
        ],
        "tryThis": "Increase knot spacing and inspect where the interpolation loses detail.",
        "limit": "Piecewise-linear approximation of a smooth synthetic trajectory, not a full GP.",
        "ref": "§2.2"
      }
    ],
    "checks": [
      {
        "q": "Why use a Lie-algebra perturbation instead of adding entries to a rotation matrix?",
        "a": "The exponential of a tangent perturbation composes with a valid rotation and keeps the new estimate on SO(d). Arbitrary entrywise addition does not."
      },
      {
        "q": "What makes locally supported splines computationally useful?",
        "a": "Each observation depends on only a small number of nearby control points, so the measurement Jacobian remains sparse."
      }
    ],
    "watch": "Be explicit about coordinate frames, perturbation side, units and interpolation validity whenever adapting these models."
  },
  {
    "n": 3,
    "title": "Robustness to Incorrect Data Association and Outliers",
    "pages": "75–100",
    "goal": "Distinguish uncertainty from a wrong correspondence, and prevent a handful of bad measurements from distorting a map.",
    "ideas": [
      {
        "title": "A large residual may be a broken hypothesis",
        "text": "Gaussian least squares squares every error; one badly associated loop closure can dominate thousands of small residuals. Outliers arise from perceptual aliasing, moving objects and sensor failures, not only random noise. More optimization iterations cannot fix an incorrect data-association assumption by themselves.",
        "equation": "J(x) = Σᵢ rᵢ(x)²",
        "ref": "§3.1, pp.75–78"
      },
      {
        "title": "Reject inconsistencies before optimization",
        "text": "RANSAC samples minimal hypotheses and counts supporting measurements. Its chance of an all-inlier sample falls as inlier rate or sample size falls. Pairwise Consistency Maximization instead builds a graph connecting mutually consistent observations and seeks a large clique.",
        "equation": "P(success in N) = 1 − (1 − wˢ)ᴺ",
        "ref": "§3.2, pp.78–85"
      },
      {
        "title": "Use bounded influence in the back end",
        "text": "Robust M-estimation substitutes a slower-growing loss for squared error. IRLS alternates weighted least-squares updates with residual-dependent weights. Graduated nonconvexity starts with a gentler loss before approaching a more selective one; none of these heuristics universally guarantees a correct map.",
        "equation": "minₓ Σᵢ ρ(rᵢ(x))",
        "ref": "§3.3, pp.85–98"
      }
    ],
    "demos": [
      {
        "key": "ransac",
        "title": "How many RANSAC trials?",
        "question": "How does an inlier fraction influence the trials needed for 99% confidence?",
        "params": [
          {
            "key": "inliers",
            "label": "Inlier fraction",
            "min": 0.1,
            "max": 0.95,
            "step": 0.05,
            "value": 0.6,
            "unit": ""
          },
          {
            "key": "sample",
            "label": "Minimal sample size",
            "min": 2,
            "max": 10,
            "step": 1,
            "value": 5,
            "unit": ""
          }
        ],
        "tryThis": "Lower the inlier fraction to 0.2 and observe how quickly the budget grows.",
        "limit": "Independent uniform sampling approximation with a known inlier ratio.",
        "ref": "§3.2.1"
      },
      {
        "key": "robust",
        "title": "Square or bound the damage",
        "question": "Where does a least-squares estimate go if one reading is grossly incorrect?",
        "params": [
          {
            "key": "bad",
            "label": "Outlier value",
            "min": 0,
            "max": 25,
            "step": 0.5,
            "value": 16,
            "unit": ""
          },
          {
            "key": "cutoff",
            "label": "Huber threshold",
            "min": 0.3,
            "max": 5,
            "step": 0.1,
            "value": 2,
            "unit": ""
          }
        ],
        "tryThis": "Move the outlier far away. Compare mean and iteratively reweighted Huber location.",
        "limit": "One-dimensional location M-estimation with a fixed set of inliers.",
        "ref": "§3.3"
      },
      {
        "key": "pcm",
        "title": "Build a consistency clique",
        "question": "How can mutually consistent measurements survive many contradictory proposals?",
        "params": [
          {
            "key": "outliers",
            "label": "Injected bad proposals",
            "min": 0,
            "max": 9,
            "step": 1,
            "value": 5,
            "unit": ""
          },
          {
            "key": "tolerance",
            "label": "Consistency tolerance",
            "min": 0.2,
            "max": 3,
            "step": 0.2,
            "value": 1,
            "unit": ""
          }
        ],
        "tryThis": "Increase tolerance; can a too-permissive consistency test let wrong hypotheses through?",
        "limit": "Scalar pairwise agreement toy, illustrating a clique proxy rather than exact 3D registration.",
        "ref": "§3.2.2"
      }
    ],
    "checks": [
      {
        "q": "Why does a 5-point RANSAC solver struggle if only 10% of proposed matches are correct?",
        "a": "The probability of sampling five inliers is 0.1⁵ = 0.00001 per attempt; many attempts are needed."
      },
      {
        "q": "Can a robust loss alone recover from a severely wrong odometry backbone?",
        "a": "Not reliably. The optimizer may start in the wrong basin and classify valid measurements as outliers; front-end checks and good initialization still matter."
      }
    ],
    "watch": "Treat an accepted loop closure as a hypothesis until geometry and robust estimation support it."
  },
  {
    "n": 4,
    "title": "Differentiable Optimization",
    "pages": "101–118",
    "goal": "Reason about how a back-end optimum changes as a learning-based front-end changes.",
    "ideas": [
      {
        "title": "Optimization can be part of a computation graph",
        "text": "A learned front-end may predict matches, confidences or depth, which determine a geometric least-squares objective. The optimized pose is an implicit function of those predictions. A task-level loss can be differentiated through the solve to improve front-end parameters rather than treating the back end as a black box.",
        "equation": "x*(y) = argminₓ L(x,y)",
        "ref": "§4.1–4.2, pp.102–108"
      },
      {
        "title": "Unrolling differentiates the iterations",
        "text": "By storing the sequence of solver updates, automatic differentiation can backpropagate through each iteration. Truncating the history uses less memory but changes the derivative. It is important to distinguish the derivative of a T-step algorithm from the derivative of its converged solution.",
        "equation": "xₜ₊₁ = xₜ − η ∇ₓL(xₜ,y)",
        "ref": "§4.2.1–4.2.2, pp.104–107"
      },
      {
        "title": "Implicit derivatives use optimality conditions",
        "text": "At a smooth stationary optimum, ∇ₓL(x*,y)=0. Differentiating this condition yields a linear system involving the Hessian. This avoids storing a long solver history, but it assumes an appropriate stationary point and a nonsingular or properly regularized Hessian; manifold states require tangent-space derivatives.",
        "equation": "dx*/dy = −Hₓₓ⁻¹ Hₓᵧ",
        "ref": "§4.2.3–4.4, pp.107–118"
      }
    ],
    "demos": [
      {
        "key": "unroll",
        "title": "Watch unrolled gradients converge",
        "question": "How many gradient-descent steps are needed before the derivative resembles its converged value?",
        "params": [
          {
            "key": "steps",
            "label": "Solver steps",
            "min": 1,
            "max": 40,
            "step": 1,
            "value": 8,
            "unit": ""
          },
          {
            "key": "rate",
            "label": "Step size",
            "min": 0.05,
            "max": 0.95,
            "step": 0.05,
            "value": 0.3,
            "unit": ""
          }
        ],
        "tryThis": "Raise step count with a stable step size. Then try an aggressive step size.",
        "limit": "Quadratic scalar loss L=(x−y)²/2; exact analytic derivative available.",
        "ref": "§4.2.1"
      },
      {
        "key": "implicit",
        "title": "Compare three derivatives",
        "question": "When do finite differences and implicit differentiation agree?",
        "params": [
          {
            "key": "weight",
            "label": "Observation weight",
            "min": 0.1,
            "max": 5,
            "step": 0.1,
            "value": 1.8,
            "unit": ""
          },
          {
            "key": "eps",
            "label": "Finite-difference step",
            "min": 0.0001,
            "max": 0.5,
            "step": 0.0001,
            "value": 0.01,
            "unit": ""
          }
        ],
        "tryThis": "Change the finite-difference step. Tiny steps are not always numerically best.",
        "limit": "Smooth scalar weighted quadratic with a unique minimum.",
        "ref": "§4.2.3"
      },
      {
        "key": "condition",
        "title": "See a nearly singular Hessian",
        "question": "What happens to the solution sensitivity as curvature approaches zero?",
        "params": [
          {
            "key": "lambda",
            "label": "Regularizing curvature",
            "min": 0.01,
            "max": 2,
            "step": 0.01,
            "value": 0.2,
            "unit": ""
          },
          {
            "key": "noise",
            "label": "Input perturbation",
            "min": -0.5,
            "max": 0.5,
            "step": 0.05,
            "value": 0.2,
            "unit": ""
          }
        ],
        "tryThis": "Decrease curvature while keeping input perturbation fixed.",
        "limit": "1D quadratic sensitivity, not a full matrix SLAM Hessian.",
        "ref": "§4.4"
      }
    ],
    "checks": [
      {
        "q": "When is unrolled differentiation not exactly the same as implicit differentiation?",
        "a": "With a finite number of iterations, unrolling differentiates the finite solver algorithm; implicit differentiation differentiates the converged stationary solution under smoothness and invertibility assumptions."
      },
      {
        "q": "Why avoid directly inverting a large Hessian?",
        "a": "It is expensive and often numerically unwise. Instead solve Hessian-vector linear systems, typically using iterative methods."
      }
    ],
    "watch": "At a gauge-degenerate optimum the Hessian is singular; fix the gauge or use an appropriate regularization before interpreting gradients."
  },
  {
    "n": 5,
    "title": "Dense Map Representations",
    "pages": "119–148",
    "goal": "Choose a spatial representation that matches sensors, memory constraints and downstream tasks.",
    "ideas": [
      {
        "title": "Map values express different physical quantities",
        "text": "An occupancy grid stores evidence that a cell is filled; a signed-distance map stores a surface's distance and sign; point clouds, meshes and surfels preserve different details. A polished rendering does not imply known free space, and a missing point need not mean empty space.",
        "equation": "lₜ = lₜ₋₁ + logit(p(zₜ|cell)) − l₀",
        "ref": "§5.2–5.3, pp.122–135"
      },
      {
        "title": "Fusion should model uncertainty",
        "text": "Repeated measurements can accumulate evidence in log-odds cells or combine truncated signed distances by confidence. Sensor noise, pose error, occlusion and changing objects alter what can be trusted. More scans may sharpen a map only when their assumptions are appropriate.",
        "equation": "D = (Σᵢ wᵢ dᵢ)/(Σᵢ wᵢ)",
        "ref": "§5.1, 5.4, pp.119–145"
      },
      {
        "title": "Resolution trades fidelity for cost",
        "text": "A 3D voxel grid's memory grows with the inverse cube of voxel size for a fixed volume. Sparse voxel hashing, trees, surfels and meshes can reduce storage, but performance depends on spatial occupancy and the required queries. Downstream planning may need conservative free-space treatment.",
        "equation": "N_voxels ∝ (L / voxel_size)³",
        "ref": "§5.3–5.5, pp.126–148"
      }
    ],
    "demos": [
      {
        "key": "occupancy",
        "title": "Accumulate occupancy evidence",
        "question": "How many consistent hits does it take to change a neutral cell into a confident occupied cell?",
        "params": [
          {
            "key": "hits",
            "label": "Hit observations",
            "min": 0,
            "max": 12,
            "step": 1,
            "value": 4,
            "unit": ""
          },
          {
            "key": "misses",
            "label": "Free-ray observations",
            "min": 0,
            "max": 12,
            "step": 1,
            "value": 2,
            "unit": ""
          },
          {
            "key": "reliability",
            "label": "Sensor hit probability",
            "min": 0.55,
            "max": 0.95,
            "step": 0.05,
            "value": 0.75,
            "unit": ""
          }
        ],
        "tryThis": "Add several misses after hits; posterior belief changes, not the world itself.",
        "limit": "Independent inverse-sensor log-odds cell update with fixed prior.",
        "ref": "§5.2"
      },
      {
        "key": "tsdf",
        "title": "Fuse noisy surface samples",
        "question": "How does confidence weighting change the estimated zero crossing?",
        "params": [
          {
            "key": "depth",
            "label": "Second observed surface",
            "min": 2,
            "max": 8,
            "step": 0.2,
            "value": 5,
            "unit": "m"
          },
          {
            "key": "weight",
            "label": "Second sample weight",
            "min": 0.1,
            "max": 10,
            "step": 0.1,
            "value": 4,
            "unit": ""
          },
          {
            "key": "trunc",
            "label": "Truncation distance",
            "min": 0.2,
            "max": 3,
            "step": 0.2,
            "value": 1,
            "unit": "m"
          }
        ],
        "tryThis": "Make the second surface disagree strongly and change truncation.",
        "limit": "1D clipped signed-distance fusion along a ray.",
        "ref": "§5.4"
      },
      {
        "key": "voxel",
        "title": "Pay for resolution",
        "question": "What is the memory penalty of cutting voxel size in half?",
        "params": [
          {
            "key": "size",
            "label": "Voxel side length",
            "min": 0.05,
            "max": 1,
            "step": 0.05,
            "value": 0.2,
            "unit": "m"
          },
          {
            "key": "extent",
            "label": "Cube side length",
            "min": 2,
            "max": 30,
            "step": 1,
            "value": 12,
            "unit": "m"
          },
          {
            "key": "bytes",
            "label": "Bytes per cell",
            "min": 2,
            "max": 16,
            "step": 2,
            "value": 8,
            "unit": "B"
          }
        ],
        "tryThis": "Half the voxel width and compare the new memory requirement.",
        "limit": "Dense cubic grid estimate; sparse allocation can differ radically.",
        "ref": "§5.3, 5.5"
      }
    ],
    "checks": [
      {
        "q": "Is the absence of a LiDAR return sufficient evidence that a space is empty?",
        "a": "No. A ray can traverse observed free space, but cells beyond an occluder or outside the sensor view remain unknown."
      },
      {
        "q": "How does dense grid memory change when resolution doubles in each dimension?",
        "a": "At fixed physical extent, each voxel edge halves, producing about 2³ = 8 times as many 3D cells."
      }
    ],
    "watch": "Separate occupied, free and unobserved cells; downstream consumers must not conflate them."
  },
  {
    "n": 6,
    "title": "Certifiably Optimal Solvers and Theoretical Properties of SLAM",
    "pages": "149–180",
    "goal": "Tell apart finding a good local answer, proving it is globally optimal, and quantifying its precision.",
    "ideas": [
      {
        "title": "A low cost is not a proof",
        "text": "Nonlinear pose estimation has nonconvex landscapes. A local solver may reach a stationary point that depends on initialization. Convex relaxations and dual certificates can sometimes prove that a candidate attains the global optimum, but a failed certificate is not evidence that the candidate is wrong.",
        "equation": "primal objective ≥ dual lower bound",
        "ref": "§6.1, pp.150–173"
      },
      {
        "title": "Constraints and gauge matter",
        "text": "Rotation matrices must satisfy orthogonality; relative measurements alone leave global frame freedom. Strong measurements or additional geometry can eliminate ambiguities. Feasibility, global optimality, local stability and physical truth are different claims.",
        "equation": "RᵀR = I; det R = 1",
        "ref": "§6.1, pp.150–173"
      },
      {
        "title": "Precision has an information limit",
        "text": "The Cramér–Rao perspective asks how accurately an unbiased estimator could estimate identifiable parameters, given a model and information matrix. Repeated independent observations add Fisher information, but an unobservable direction gives singular information, not merely a large finite error bar.",
        "equation": "Cov(x̂) ⪰ I(x)⁻¹  (regular identifiable case)",
        "ref": "§6.2, pp.173–180"
      }
    ],
    "demos": [
      {
        "key": "certify",
        "title": "Measure an optimality gap",
        "question": "When does a candidate upper bound meet a relaxation lower bound?",
        "params": [
          {
            "key": "candidate",
            "label": "Candidate cost",
            "min": 2,
            "max": 20,
            "step": 0.5,
            "value": 7,
            "unit": ""
          },
          {
            "key": "bound",
            "label": "Certified lower bound",
            "min": 0,
            "max": 12,
            "step": 0.5,
            "value": 5,
            "unit": ""
          }
        ],
        "tryThis": "Bring the candidate cost down to the lower bound. What can you conclude?",
        "limit": "Illustrative primal-dual gap bookkeeping, not an SDP certificate construction.",
        "ref": "§6.1"
      },
      {
        "key": "geometry",
        "title": "Choose between two basins",
        "question": "How can a nonconvex objective produce different answers from different initializations?",
        "params": [
          {
            "key": "start",
            "label": "Initial estimate",
            "min": -3,
            "max": 3,
            "step": 0.1,
            "value": 1.8,
            "unit": ""
          },
          {
            "key": "bias",
            "label": "Asymmetry",
            "min": -1,
            "max": 1,
            "step": 0.1,
            "value": 0.2,
            "unit": ""
          }
        ],
        "tryThis": "Move the initial guess across the center and watch gradient descent select a different well.",
        "limit": "Scalar double-well optimization, an analogy for nonconvex rotation problems.",
        "ref": "§6.1"
      },
      {
        "key": "crlb",
        "title": "See an uncertainty floor",
        "question": "How does another independent observation change the variance lower bound?",
        "params": [
          {
            "key": "count",
            "label": "Independent measurements",
            "min": 1,
            "max": 30,
            "step": 1,
            "value": 5,
            "unit": ""
          },
          {
            "key": "sigma",
            "label": "Measurement σ",
            "min": 0.2,
            "max": 5,
            "step": 0.2,
            "value": 1.5,
            "unit": ""
          },
          {
            "key": "strength",
            "label": "Relative geometry strength",
            "min": 0,
            "max": 1,
            "step": 0.05,
            "value": 0.8,
            "unit": ""
          }
        ],
        "tryThis": "Reduce geometry strength toward zero and inspect the bound.",
        "limit": "Scalar Gaussian Fisher-information model; ignores bias and correlations.",
        "ref": "§6.2"
      }
    ],
    "checks": [
      {
        "q": "Does failure of an optimality certificate prove the estimate is suboptimal?",
        "a": "No. A certificate can be inconclusive because a relaxation is loose or numerical conditions fail. Successful certificate gives a guarantee under stated assumptions."
      },
      {
        "q": "What happens to the CRLB in an unobservable direction?",
        "a": "The Fisher information has a zero eigenvalue; its ordinary inverse is undefined, reflecting a non-identifiable parameter."
      }
    ],
    "watch": "A certificate applies to the stated mathematical problem and its assumptions, not automatically to ground-truth accuracy."
  }
]);
