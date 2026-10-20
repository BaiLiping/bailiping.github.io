/* Part III: original spatial-AI lessons and interactive experiment specifications. */
window.SLAM_CHAPTERS = (window.SLAM_CHAPTERS || []).concat([
  {
    "n": 13,
    "title": "Boosting SLAM with Deep Learning",
    "pages": "366–396",
    "goal": "Understand what learning adds to SLAM and how geometry keeps predictions consistent across views.",
    "ideas": [
      {
        "title": "Learned predictions provide priors, not guarantees",
        "text": "Networks can estimate depth, image correspondences, optical flow, confidence and pose cues. They learn statistical regularities from training data, but a sharp prediction can still be wrong out of distribution. Geometry constrains how those predictions must agree with camera motion and with one another.",
        "equation": "learned proposal + geometric residual → refined estimate",
        "ref": "§13.1–13.2, pp.368–379"
      },
      {
        "title": "Learned correspondences can feed differentiable BA",
        "text": "Systems such as DROID-SLAM use learned update operators together with dense correspondence and optimization structure. Iterative refinement can propagate geometric error back toward learned components. Improvement depends on sufficient geometric signal, accurate modeling and suitable supervision.",
        "equation": "min_{poses,depth} Σ reprojection error",
        "ref": "§13.3, pp.379–386"
      },
      {
        "title": "Point-map models reconstruct jointly",
        "text": "Recent scene inference methods infer 3D point maps and matches from pairs of images; extending such predictions to larger scenes still requires coherent reference frames and robust associations. There is a difference between an attractive single-pair reconstruction and globally consistent SLAM.",
        "equation": "cross-view consistency constrains point maps",
        "ref": "§13.4–13.6, pp.386–395"
      }
    ],
    "demos": [
      {
        "key": "features",
        "title": "Weight uncertain learned matches",
        "question": "How does a matching model's confidence change which correspondence controls a pose estimate?",
        "params": [
          {
            "key": "wrong",
            "label": "Incorrect match offset",
            "min": 0,
            "max": 12,
            "step": 0.5,
            "value": 6,
            "unit": "px"
          },
          {
            "key": "confidence",
            "label": "Wrong-match confidence",
            "min": 0,
            "max": 1,
            "step": 0.05,
            "value": 0.6,
            "unit": ""
          },
          {
            "key": "good",
            "label": "Good-match confidence",
            "min": 0.1,
            "max": 1,
            "step": 0.05,
            "value": 0.8,
            "unit": ""
          }
        ],
        "tryThis": "Make a wrong feature highly confident. Can geometry alone repair an incorrect association?",
        "limit": "Scalar confidence-weighted offset estimate, not a neural feature matcher.",
        "ref": "§13.2"
      },
      {
        "key": "scale",
        "title": "Resolve depth-scale ambiguity",
        "question": "How can a metric cue change a monocular depth prediction?",
        "params": [
          {
            "key": "pred",
            "label": "Predicted relative depth",
            "min": 1,
            "max": 12,
            "step": 0.5,
            "value": 6,
            "unit": ""
          },
          {
            "key": "scale",
            "label": "Metric scale cue",
            "min": 0.2,
            "max": 3,
            "step": 0.1,
            "value": 1.5,
            "unit": ""
          },
          {
            "key": "error",
            "label": "Scale cue noise",
            "min": 0,
            "max": 1,
            "step": 0.05,
            "value": 0.25,
            "unit": ""
          }
        ],
        "tryThis": "Modify scale while holding monocular structure fixed. The pixel geometry can remain equivalent.",
        "limit": "Single-view positive scale ambiguity and toy cue fusion.",
        "ref": "§13.1"
      },
      {
        "key": "learnedba",
        "title": "Refine a learned initialization",
        "question": "How much can an iterative geometric update correct a biased initial pose?",
        "params": [
          {
            "key": "init",
            "label": "Learned pose bias",
            "min": -4,
            "max": 4,
            "step": 0.2,
            "value": 2,
            "unit": "m"
          },
          {
            "key": "iters",
            "label": "Refinement iterations",
            "min": 0,
            "max": 10,
            "step": 1,
            "value": 4,
            "unit": ""
          },
          {
            "key": "gain",
            "label": "Update gain",
            "min": 0.1,
            "max": 0.95,
            "step": 0.05,
            "value": 0.5,
            "unit": ""
          }
        ],
        "tryThis": "Start far away and increase iterations; distinguish this convex toy from true multi-view BA.",
        "limit": "Linear scalar measurement with contraction updates; no deep network executes.",
        "ref": "§13.3"
      }
    ],
    "checks": [
      {
        "q": "Why should learned depth be checked against multi-view geometry?",
        "a": "A network predicts likely depth from training statistics. Multiple views provide projective consistency constraints, which may expose incompatible predictions."
      },
      {
        "q": "Does differentiable BA guarantee that learned features will generalize?",
        "a": "No. It can provide a training signal under the chosen losses and data, but model mismatch and distribution shift remain."
      }
    ],
    "watch": "Do not interpret these sliders as neural network inference; they isolate the geometric roles of learned outputs."
  },
  {
    "n": 14,
    "title": "Map Representations with Differentiable Volume Rendering",
    "pages": "397–416",
    "goal": "Understand how a renderable 3D map becomes a parameterized function of view and scene.",
    "ideas": [
      {
        "title": "Volume rendering accumulates transmittance",
        "text": "A ray samples density and color. Each sample contributes its color multiplied by the chance that the ray reaches the sample and terminates there. Early opaque samples suppress contributions from later ones. This is the core numerical idea behind NeRF-like volume rendering.",
        "equation": "C = Σᵢ Tᵢ αᵢ cᵢ,  Tᵢ = ∏ⱼ₍₍₍i(1−αⱼ)",
        "ref": "§14.1–14.2, pp.398–409"
      },
      {
        "title": "Radiance fields encode continuous appearance",
        "text": "A neural field maps positions and viewing directions to density and color, enabling view synthesis from camera observations. Training requires calibrated camera poses or joint pose refinement and a photometric objective; many explanations for the observed images are possible without enough viewpoints.",
        "equation": "(x,direction) ↦ (σ,color)",
        "ref": "§14.2, pp.401–409"
      },
      {
        "title": "Gaussian splatting makes explicit elements renderable",
        "text": "3D Gaussian splatting represents scenes as anisotropic Gaussian primitives with position, opacity, scale and appearance. Projecting and alpha-compositing these blobs offers fast view-dependent rendering. A visually sharp result is not automatically a collision-safe metric surface.",
        "equation": "project 3D Gaussian → screen ellipse; alpha composite",
        "ref": "§14.3, pp.409–414"
      }
    ],
    "demos": [
      {
        "key": "volume",
        "title": "Composite a ray",
        "question": "Why does moving opacity to the front change the final pixel color?",
        "params": [
          {
            "key": "alpha1",
            "label": "Front opacity",
            "min": 0,
            "max": 1,
            "step": 0.05,
            "value": 0.35,
            "unit": ""
          },
          {
            "key": "alpha2",
            "label": "Back opacity",
            "min": 0,
            "max": 1,
            "step": 0.05,
            "value": 0.75,
            "unit": ""
          },
          {
            "key": "background",
            "label": "Background intensity",
            "min": 0,
            "max": 1,
            "step": 0.05,
            "value": 0.2,
            "unit": ""
          }
        ],
        "tryThis": "Make the front layer opaque. How much can the back layer contribute?",
        "limit": "Two homogeneous scalar-colored volume samples; ignores view-dependent color.",
        "ref": "§14.2"
      },
      {
        "key": "splat",
        "title": "Project a Gaussian footprint",
        "question": "How do depth and 3D size change the Gaussian's screen footprint?",
        "params": [
          {
            "key": "size",
            "label": "Gaussian world σ",
            "min": 0.05,
            "max": 2,
            "step": 0.05,
            "value": 0.65,
            "unit": "m"
          },
          {
            "key": "depth",
            "label": "Gaussian depth",
            "min": 1,
            "max": 12,
            "step": 0.5,
            "value": 5,
            "unit": "m"
          },
          {
            "key": "f",
            "label": "Focal length",
            "min": 200,
            "max": 900,
            "step": 50,
            "value": 500,
            "unit": "px"
          }
        ],
        "tryThis": "Double depth. A fixed world-size primitive should occupy fewer pixels.",
        "limit": "Isotropic Gaussian small-angle perspective approximation, not full anisotropic covariance projection.",
        "ref": "§14.3"
      },
      {
        "key": "view",
        "title": "Make appearance view-dependent",
        "question": "How does a specular-looking component change with viewing angle?",
        "params": [
          {
            "key": "view",
            "label": "Viewing angle",
            "min": 0,
            "max": 90,
            "step": 5,
            "value": 25,
            "unit": "°"
          },
          {
            "key": "rough",
            "label": "Lobe roughness",
            "min": 0.1,
            "max": 1,
            "step": 0.05,
            "value": 0.3,
            "unit": ""
          },
          {
            "key": "base",
            "label": "Diffuse intensity",
            "min": 0.1,
            "max": 0.8,
            "step": 0.05,
            "value": 0.4,
            "unit": ""
          }
        ],
        "tryThis": "Increase roughness: the highlight broadens and peaks less.",
        "limit": "One-lobe intensity proxy; no physically calibrated BRDF or learned field.",
        "ref": "§14.1–14.2"
      }
    ],
    "checks": [
      {
        "q": "Why does a high-opacity sample reduce the influence of samples behind it?",
        "a": "Its absorption decreases transmittance, so later samples are less likely to contribute to the observed ray color."
      },
      {
        "q": "Does photorealistic rendering imply a metrically reliable obstacle map?",
        "a": "No. Rendering quality and geometric free-space accuracy are different objectives and require separate evaluation."
      }
    ],
    "watch": "These illustrations are original toy renderers; they do not train a NeRF or implement GPU splatting."
  },
  {
    "n": 15,
    "title": "Dynamic and Deformable SLAM",
    "pages": "417–453",
    "goal": "Separate camera motion, moving entities and persistent environmental change.",
    "ideas": [
      {
        "title": "Static-world assumptions can fail systematically",
        "text": "If a person or vehicle moves while tracked as a fixed landmark, the back-end may bend camera poses to explain its motion. Robust masks, multibody estimation or explicit moving-object tracks help. Whether to exclude motion or model it depends on the map's intended use.",
        "equation": "z = h(robot pose, object state(t)) + noise",
        "ref": "§15.1–15.2, pp.418–432"
      },
      {
        "title": "Long-term maps need change awareness",
        "text": "The same location can differ by lighting, furniture, seasons or construction. A lifelong map may retain stable structure while updating changed regions. Blindly accumulating observations creates ghost surfaces; blindly forgetting old observations discards useful place memory.",
        "equation": "p(occupancy at t | temporal observations)",
        "ref": "§15.3, pp.432–446"
      },
      {
        "title": "Deformable objects require richer state",
        "text": "A bending sheet or tissue cannot be represented by one rigid transform. Deformation models use local regularity, control nodes or physical constraints while reconciling observations. Strong regularization may erase genuine deformation; weak regularization makes the problem underdetermined.",
        "equation": "E = E_data + λ E_deformation",
        "ref": "§15.4, pp.446–451"
      }
    ],
    "demos": [
      {
        "key": "dynamic",
        "title": "Reject moving landmarks",
        "question": "How does tracking a moving object as fixed bias a pose estimate?",
        "params": [
          {
            "key": "motion",
            "label": "Moving-object shift",
            "min": 0,
            "max": 8,
            "step": 0.2,
            "value": 4,
            "unit": "m"
          },
          {
            "key": "dynamicFraction",
            "label": "Dynamic feature fraction",
            "min": 0,
            "max": 1,
            "step": 0.05,
            "value": 0.4,
            "unit": ""
          },
          {
            "key": "reject",
            "label": "Dynamic rejection",
            "min": 0,
            "max": 1,
            "step": 0.05,
            "value": 0.5,
            "unit": ""
          }
        ],
        "tryThis": "Increase moving-feature prevalence. Then reject dynamic measurements more reliably.",
        "limit": "Weighted mean alignment proxy; no object tracking or geometric segmentation.",
        "ref": "§15.2"
      },
      {
        "key": "changes",
        "title": "Update a changing occupancy cell",
        "question": "How does decay rate trade memory for adaptation to a newly empty scene?",
        "params": [
          {
            "key": "past",
            "label": "Past occupied observations",
            "min": 0,
            "max": 15,
            "step": 1,
            "value": 10,
            "unit": ""
          },
          {
            "key": "new",
            "label": "Recent free observations",
            "min": 0,
            "max": 15,
            "step": 1,
            "value": 5,
            "unit": ""
          },
          {
            "key": "forget",
            "label": "Forgetting factor",
            "min": 0,
            "max": 1,
            "step": 0.05,
            "value": 0.8,
            "unit": ""
          }
        ],
        "tryThis": "Lower the forgetting factor. What happens to long-lived but now outdated evidence?",
        "limit": "Scalar log-odds time-decay model, not a full lifelong occupancy map.",
        "ref": "§15.3"
      },
      {
        "key": "deform",
        "title": "Balance data against rigidity",
        "question": "How far will a deformable surface bend when smoothness becomes dominant?",
        "params": [
          {
            "key": "pull",
            "label": "Observed displacement",
            "min": 0,
            "max": 5,
            "step": 0.2,
            "value": 3,
            "unit": "m"
          },
          {
            "key": "stiffness",
            "label": "Deformation stiffness",
            "min": 0,
            "max": 10,
            "step": 0.5,
            "value": 3,
            "unit": ""
          },
          {
            "key": "nodes",
            "label": "Control nodes",
            "min": 3,
            "max": 12,
            "step": 1,
            "value": 7,
            "unit": ""
          }
        ],
        "tryThis": "Increase stiffness: does the model fit the observation or preserve its original shape?",
        "limit": "1D smoothness-regularized chain with one displaced center node.",
        "ref": "§15.4"
      }
    ],
    "checks": [
      {
        "q": "Why can moving-object features damage camera motion estimates?",
        "a": "A solver assuming all landmarks are static may explain actual object motion by changing camera poses."
      },
      {
        "q": "Why isn't the strongest possible deformation regularization always best?",
        "a": "It would suppress genuine deformations and fit observations poorly."
      }
    ],
    "watch": "Robot motion, object motion, illumination shifts and structural changes must not be conflated."
  },
  {
    "n": 16,
    "title": "Metric-Semantic SLAM",
    "pages": "454–489",
    "goal": "Connect geometry to object identity, relationships and task-driven spatial queries.",
    "ideas": [
      {
        "title": "Semantic labels are uncertain map variables",
        "text": "A metric map provides positions and shapes; semantic SLAM augments that geometry with class, instance or object attributes. A semantic detector's label is not ground truth. Repeated observations can update categorical beliefs, but correlated predictions should not be counted as independent evidence.",
        "equation": "p(class | observations) ∝ p(observations | class)p(class)",
        "ref": "§16.1–16.2, pp.455–471"
      },
      {
        "title": "Objects have geometry and relationships",
        "text": "Sparse object landmarks, dense labeled voxels and hierarchical scene graphs answer different questions. A scene graph can link rooms, surfaces, objects and their containment or adjacency relationships, making a query such as 'which cup is on the table?' structurally meaningful.",
        "equation": "room → contains → table → supports → cup",
        "ref": "§16.2–16.4, pp.456–487"
      },
      {
        "title": "Task utility determines representation",
        "text": "For navigation, traversability and persistent obstacles may matter more than every pixel label. For manipulation, object extent and contact geometry are essential. Semantics can help data association, but a confidently misclassified or moving object can also create a false long-term constraint.",
        "equation": "utility(map) depends on query and uncertainty",
        "ref": "§16.3–16.4, pp.471–487"
      }
    ],
    "demos": [
      {
        "key": "semantic",
        "title": "Fuse uncertain class evidence",
        "question": "When does repeated class evidence override an initially strong prior?",
        "params": [
          {
            "key": "prior",
            "label": "Prior probability: chair",
            "min": 0.05,
            "max": 0.95,
            "step": 0.05,
            "value": 0.4,
            "unit": ""
          },
          {
            "key": "observations",
            "label": "Chair-positive observations",
            "min": 0,
            "max": 10,
            "step": 1,
            "value": 4,
            "unit": ""
          },
          {
            "key": "quality",
            "label": "Detector sensitivity",
            "min": 0.55,
            "max": 0.95,
            "step": 0.05,
            "value": 0.75,
            "unit": ""
          }
        ],
        "tryThis": "Increase positive evidence; reflect on why correlated repeat frames may overstate confidence.",
        "limit": "Binary independent-observation Bayes model; no learned semantic model.",
        "ref": "§16.2"
      },
      {
        "key": "scenegraph",
        "title": "Traverse a spatial hierarchy",
        "question": "How does graph depth affect the number of relationship hops in a semantic query?",
        "params": [
          {
            "key": "rooms",
            "label": "Rooms in building",
            "min": 2,
            "max": 12,
            "step": 1,
            "value": 6,
            "unit": ""
          },
          {
            "key": "objects",
            "label": "Objects per room",
            "min": 2,
            "max": 15,
            "step": 1,
            "value": 6,
            "unit": ""
          },
          {
            "key": "depth",
            "label": "Relationship levels",
            "min": 1,
            "max": 4,
            "step": 1,
            "value": 3,
            "unit": ""
          }
        ],
        "tryThis": "Increase objects/room. Compare searching a structured room with a flat global scan.",
        "limit": "Synthetic containment graph counts, not an implemented 3D scene-graph network.",
        "ref": "§16.4"
      },
      {
        "key": "wayfind",
        "title": "Choose an affordance-aware route",
        "question": "Why might the shortest path not be the preferred robot route?",
        "params": [
          {
            "key": "short",
            "label": "Direct path length",
            "min": 3,
            "max": 15,
            "step": 0.5,
            "value": 6,
            "unit": "m"
          },
          {
            "key": "detour",
            "label": "Safe detour length",
            "min": 3,
            "max": 20,
            "step": 0.5,
            "value": 10,
            "unit": "m"
          },
          {
            "key": "risk",
            "label": "Direct-path risk",
            "min": 0,
            "max": 1,
            "step": 0.05,
            "value": 0.5,
            "unit": ""
          }
        ],
        "tryThis": "Raise collision risk of the direct route; see when expected cost prefers the detour.",
        "limit": "Two route alternatives with a chosen risk penalty, no geometric path planner.",
        "ref": "§16.1, 16.4"
      }
    ],
    "checks": [
      {
        "q": "How does an object-centered scene graph differ from a point cloud?",
        "a": "It stores entities and relations (such as room containment and object support), rather than only metric points."
      },
      {
        "q": "Can repeated semantic predictions be treated as independent observations?",
        "a": "Only if their conditional independence assumption is reasonable; repeated similar frames may otherwise lead to overconfidence."
      }
    ],
    "watch": "Object classes, geometry and relations can change over time and must carry uncertainty."
  },
  {
    "n": 17,
    "title": "Towards Open-World Spatial AI",
    "pages": "490–520",
    "goal": "Understand what open-vocabulary map embeddings enable, and what they do not establish.",
    "ideas": [
      {
        "title": "Language can address map elements by meaning",
        "text": "Multimodal foundation models encode images, words and spatial observations in related feature spaces. A text query can rank map regions by embedding similarity instead of relying on a fixed list of labels. Such scores are associations, not calibrated object-existence probabilities by default.",
        "equation": "similarity(t,v) = (t⋅v)/(‖t‖‖v‖)",
        "ref": "§17.1–17.3, pp.491–514"
      },
      {
        "title": "Spatial grounding needs geometry",
        "text": "A command such as 'find somewhere to sit' may retrieve several similar regions. To execute it, the system still needs object positions, extents, reachability, coordinate frames and temporal validity. Grounded actions combine language relevance with metric constraints and physical affordances.",
        "equation": "score = semantic relevance × spatial feasibility",
        "ref": "§17.3, pp.499–514"
      },
      {
        "title": "Open world does not mean infallible",
        "text": "Open-set queries can describe unfamiliar categories and relationships, but embeddings can be biased, brittle to phrasing or affected by illumination and occlusion. Thresholding and calibration should leave room for 'unknown' and human verification on consequential actions.",
        "equation": "accept only if score exceeds task-calibrated threshold",
        "ref": "§17.2–17.4, pp.495–514"
      }
    ],
    "demos": [
      {
        "key": "vocab",
        "title": "Search a semantic map",
        "question": "How does prompt wording shift which synthetic room embedding scores highest?",
        "params": [
          {
            "key": "concept",
            "label": "Query concept index",
            "min": 0,
            "max": 3,
            "step": 1,
            "value": 1,
            "unit": ""
          },
          {
            "key": "temperature",
            "label": "Similarity temperature",
            "min": 0.1,
            "max": 2,
            "step": 0.1,
            "value": 0.7,
            "unit": ""
          }
        ],
        "tryThis": "Switch query index and make softmax sharper. Does that create better underlying evidence?",
        "limit": "Deterministic four-vector toy embedding and softmax, not CLIP inference.",
        "ref": "§17.2–17.3"
      },
      {
        "key": "grounding",
        "title": "Rank feasible destinations",
        "question": "When do geometry and accessibility override a semantically relevant target?",
        "params": [
          {
            "key": "similarity",
            "label": "Semantic relevance",
            "min": 0,
            "max": 1,
            "step": 0.05,
            "value": 0.9,
            "unit": ""
          },
          {
            "key": "distance",
            "label": "Distance to target",
            "min": 1,
            "max": 30,
            "step": 1,
            "value": 12,
            "unit": "m"
          },
          {
            "key": "blocked",
            "label": "Obstruction",
            "min": 0,
            "max": 1,
            "step": 0.05,
            "value": 0.3,
            "unit": ""
          }
        ],
        "tryThis": "Keep similarity high but block the route. Should the robot still choose the target?",
        "limit": "Scalar multi-objective route score, not a trained language-conditioned planner.",
        "ref": "§17.3"
      },
      {
        "key": "calibration",
        "title": "Know when to abstain",
        "question": "How does acceptance threshold control false positive selections?",
        "params": [
          {
            "key": "threshold",
            "label": "Similarity threshold",
            "min": 0.2,
            "max": 0.95,
            "step": 0.05,
            "value": 0.65,
            "unit": ""
          },
          {
            "key": "novelty",
            "label": "Out-of-distribution shift",
            "min": 0,
            "max": 0.6,
            "step": 0.05,
            "value": 0.2,
            "unit": ""
          }
        ],
        "tryThis": "Increase novelty. Compare number of known and unknown queries accepted.",
        "limit": "Fixed synthetic similarity score sets, not calibrated model confidence.",
        "ref": "§17.4"
      }
    ],
    "checks": [
      {
        "q": "Is cosine similarity a calibrated probability that an object exists?",
        "a": "No. It is a normalized vector similarity; probabilistic interpretation requires additional modeling and calibration."
      },
      {
        "q": "Why does a language-grounded query still need SLAM?",
        "a": "Semantic retrieval alone does not determine physical pose, traversability, metric distance or whether the observation is current."
      }
    ],
    "watch": "Keep an explicit unknown outcome and avoid allowing semantic similarity alone to authorize physical actions."
  },
  {
    "n": 18,
    "title": "The Computational Structure of Spatial AI Systems",
    "pages": "521–547",
    "goal": "Reason about computational graphs, scheduling and distributed estimation on constrained hardware.",
    "ideas": [
      {
        "title": "Different perception loops need different deadlines",
        "text": "Odometry may need millisecond latency for control, while global map optimization can run less often. Task reasoning may tolerate still longer delays. The engineering challenge is to schedule dataflow so a slow semantic stage does not block fast feedback.",
        "equation": "latency budget = sensing + compute + communication",
        "ref": "§18.1–18.3, pp.521–528"
      },
      {
        "title": "Computation follows graph structure",
        "text": "SLAM and Spatial AI naturally form sparse graphs of factors and states. Hardware architectures exploit locality, parallelism, memory bandwidth and accelerator compatibility. A faster arithmetic unit cannot offset unlimited data transfers or serial dependencies.",
        "equation": "throughput ≤ min(compute capacity, memory bandwidth constraint)",
        "ref": "§18.4–18.5, pp.528–540"
      },
      {
        "title": "Belief propagation trades global barriers for local messages",
        "text": "On suitable Gaussian factor graphs, nodes exchange local information that can be updated asynchronously and in parallel. Gaussian belief propagation may converge rapidly for some models, but convergence and exactness depend on graph structure, weights and schedule. Continual learning adds extra update paths and stability concerns.",
        "equation": "m_{i→j} = combine(local factor, incoming messages)",
        "ref": "§18.6–18.8, pp.540–547"
      }
    ],
    "demos": [
      {
        "key": "budget",
        "title": "Allocate a compute budget",
        "question": "What happens if sensing and inference demand more time than one control cycle allows?",
        "params": [
          {
            "key": "hz",
            "label": "Control update rate",
            "min": 10,
            "max": 200,
            "step": 10,
            "value": 100,
            "unit": "Hz"
          },
          {
            "key": "front",
            "label": "Front-end latency",
            "min": 1,
            "max": 20,
            "step": 1,
            "value": 5,
            "unit": "ms"
          },
          {
            "key": "back",
            "label": "Back-end latency",
            "min": 0,
            "max": 50,
            "step": 2,
            "value": 16,
            "unit": "ms"
          }
        ],
        "tryThis": "Increase update rate while holding latency fixed. Decide what must run asynchronously.",
        "limit": "Serial/parallel timing budget proxy; actual schedules overlap.",
        "ref": "§18.1–18.3"
      },
      {
        "key": "pipeline",
        "title": "Pipeline sensor and compute stages",
        "question": "Why can pipelining improve throughput without lowering single-sample latency?",
        "params": [
          {
            "key": "sense",
            "label": "Sensing stage",
            "min": 1,
            "max": 20,
            "step": 1,
            "value": 6,
            "unit": "ms"
          },
          {
            "key": "infer",
            "label": "Inference stage",
            "min": 1,
            "max": 40,
            "step": 1,
            "value": 14,
            "unit": "ms"
          },
          {
            "key": "map",
            "label": "Mapping stage",
            "min": 1,
            "max": 40,
            "step": 1,
            "value": 20,
            "unit": "ms"
          }
        ],
        "tryThis": "Identify the slowest stage; it caps sustained pipeline throughput.",
        "limit": "Three-stage deterministic pipeline with zero transfer overhead.",
        "ref": "§18.4–18.5"
      },
      {
        "key": "gbp",
        "title": "Iterate local Gaussian messages",
        "question": "How quickly can neighbors agree on a scalar consensus with different coupling?",
        "params": [
          {
            "key": "coupling",
            "label": "Neighbor coupling",
            "min": 0,
            "max": 0.95,
            "step": 0.05,
            "value": 0.6,
            "unit": ""
          },
          {
            "key": "rounds",
            "label": "Message rounds",
            "min": 0,
            "max": 25,
            "step": 1,
            "value": 8,
            "unit": ""
          },
          {
            "key": "nodes",
            "label": "Number of variables",
            "min": 3,
            "max": 10,
            "step": 1,
            "value": 6,
            "unit": ""
          }
        ],
        "tryThis": "Increase coupling and iteration count. Check convergence of local means.",
        "limit": "Synchronous Jacobi updates of a strictly convex Gaussian chain; instructional analogue, not full GBP message schedules.",
        "ref": "§18.6"
      }
    ],
    "checks": [
      {
        "q": "Why are control-rate and global map update-rate often different?",
        "a": "Low-level feedback requires tight latency; global optimization can be computationally costly and can update asynchronously without blocking fast odometry."
      },
      {
        "q": "Is asynchronous message passing automatically convergent on any loopy graph?",
        "a": "No. Gaussian message-passing convergence depends on the linear system/graph properties and update schedule."
      }
    ],
    "watch": "Computational savings must be evaluated against latency, memory transfers, energy and estimation quality."
  }
]);
