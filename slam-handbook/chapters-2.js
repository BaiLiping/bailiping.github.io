/* Part II: original lessons and experiments corresponding to the published table of contents. */
window.SLAM_CHAPTERS = (window.SLAM_CHAPTERS || []).concat([
  {
    "n": 7,
    "title": "Visual SLAM",
    "pages": "193–223",
    "goal": "See how image correspondence, geometry and bundle adjustment reconstruct both cameras and points.",
    "ideas": [
      {
        "title": "Pixels constrain rays, not depth by themselves",
        "text": "A pinhole camera maps a 3D scene point into a 2D image through perspective division. The image gives a bearing ray; depth must come from baseline, additional views, motion or a depth sensor. Calibrated focal length changes the relation between world angle and pixel displacement.",
        "equation": "u = fX/Z + cₓ;  v = fY/Z + cᵧ",
        "ref": "§7.2–7.3, pp.196–207"
      },
      {
        "title": "Parallax makes triangulation possible",
        "text": "Matching the same landmark across views yields intersecting rays in ideal geometry; real noisy rays do not intersect exactly. A small camera baseline or a distant target produces tiny disparity and poor depth conditioning, especially when pose estimates are uncertain.",
        "equation": "Z ≈ fB/d  (rectified stereo)",
        "ref": "§7.3, pp.197–207"
      },
      {
        "title": "Bundle adjustment couples cameras and landmarks",
        "text": "The front end detects or tracks features and proposes associations; bundle adjustment jointly refines camera poses and map points by reducing reprojection errors. Robust weighting, gauge handling, keyframe selection and loop closure decide whether an elegant least-squares fit becomes a useful real-time system.",
        "equation": "min_{T,P} Σᵢⱼ ‖uᵢⱼ − π(TᵢPⱼ)‖²",
        "ref": "§7.4–7.8, pp.207–222"
      }
    ],
    "demos": [
      {
        "key": "camera",
        "title": "Move a point through a camera",
        "question": "How does pixel position change with lateral position, depth and focal length?",
        "params": [
          {
            "key": "x",
            "label": "Point X",
            "min": -3,
            "max": 3,
            "step": 0.1,
            "value": 1,
            "unit": "m"
          },
          {
            "key": "z",
            "label": "Depth Z",
            "min": 1,
            "max": 12,
            "step": 0.2,
            "value": 5,
            "unit": "m"
          },
          {
            "key": "f",
            "label": "Focal length",
            "min": 150,
            "max": 900,
            "step": 25,
            "value": 500,
            "unit": "px"
          }
        ],
        "tryThis": "Double depth while holding X fixed; note how projection contracts toward the principal point.",
        "limit": "Ideal calibrated pinhole camera, no lens distortion.",
        "ref": "§7.3"
      },
      {
        "key": "parallax",
        "title": "Triangulate from disparity",
        "question": "How does depth uncertainty explode when disparity approaches zero?",
        "params": [
          {
            "key": "baseline",
            "label": "Stereo baseline",
            "min": 0.1,
            "max": 2,
            "step": 0.1,
            "value": 0.5,
            "unit": "m"
          },
          {
            "key": "depth",
            "label": "True depth",
            "min": 2,
            "max": 30,
            "step": 1,
            "value": 12,
            "unit": "m"
          },
          {
            "key": "pixels",
            "label": "Disparity σ",
            "min": 0.1,
            "max": 2,
            "step": 0.1,
            "value": 0.5,
            "unit": "px"
          }
        ],
        "tryThis": "Increase depth until disparity becomes very small; inspect sensitivity.",
        "limit": "Rectified stereo small-noise first-order propagation.",
        "ref": "§7.3"
      },
      {
        "key": "ba",
        "title": "Refine a landmark by reprojection",
        "question": "How do multiple camera views constrain a scalar landmark coordinate?",
        "params": [
          {
            "key": "noise",
            "label": "Pixel noise",
            "min": 0,
            "max": 12,
            "step": 1,
            "value": 4,
            "unit": "px"
          },
          {
            "key": "views",
            "label": "Number of camera views",
            "min": 2,
            "max": 12,
            "step": 1,
            "value": 5,
            "unit": ""
          },
          {
            "key": "iterations",
            "label": "Gauss–Newton steps",
            "min": 0,
            "max": 8,
            "step": 1,
            "value": 3,
            "unit": ""
          }
        ],
        "tryThis": "Compare 0 steps with several updates from a biased initialization.",
        "limit": "1D pinhole-like nonlinear measurement with fixed known camera poses.",
        "ref": "§7.4"
      }
    ],
    "checks": [
      {
        "q": "Why can one calibrated camera view not determine arbitrary scene depth?",
        "a": "A projected pixel determines a ray. Many 3D points along that ray have the same ideal projection."
      },
      {
        "q": "What changes when triangulating a very distant point with a short baseline?",
        "a": "Disparity becomes tiny, so pixel noise creates a large depth error."
      }
    ],
    "watch": "Feature matching failures, moving objects and poor initialization can defeat pure reprojection-error optimization."
  },
  {
    "n": 8,
    "title": "LiDAR SLAM",
    "pages": "224–249",
    "goal": "Understand how range scans are aligned, motion-compensated and recognized as places.",
    "ideas": [
      {
        "title": "Scan matching estimates relative motion",
        "text": "A LiDAR measures range and bearing to surfaces. Local odometry typically aligns overlapping scans or scans to a map, often with point-to-point or point-to-plane residuals. Incorrect nearest-neighbor matches and limited structural variation can make an apparently precise fit misleading.",
        "equation": "min_T Σᵢ ‖Tpᵢ − q_{match(i)}‖²",
        "ref": "§8.1–8.2, pp.225–236"
      },
      {
        "title": "A scan is not an instantaneous photograph",
        "text": "Spinning and scanning sensors acquire points over a nonzero interval while the robot moves. If all returns are placed at one pose, walls warp. De-skewing reprojects each return using a time-dependent trajectory; synchronization and extrinsic calibration matter.",
        "equation": "p_ref = T_ref⁻¹ T(tᵢ) pᵢ",
        "ref": "§8.2.2, pp.227–236"
      },
      {
        "title": "Recognizing a place closes long-term drift",
        "text": "A place descriptor proposes scan candidates; geometric registration checks their consistency. A genuine loop closure corrects accumulated odometry drift, while a false loop closure can fold the map onto itself. A 2D histogram or descriptor alone is not a geometric proof.",
        "equation": "pose graph = odometry + checked loop factors",
        "ref": "§8.3–8.4, pp.236–248"
      }
    ],
    "demos": [
      {
        "key": "icp",
        "title": "Align two noisy scans",
        "question": "What happens to translation estimation when correspondences contain mismatches?",
        "params": [
          {
            "key": "shift",
            "label": "True shift",
            "min": -3,
            "max": 3,
            "step": 0.2,
            "value": 1.4,
            "unit": "m"
          },
          {
            "key": "noise",
            "label": "Range jitter",
            "min": 0,
            "max": 1,
            "step": 0.05,
            "value": 0.2,
            "unit": "m"
          },
          {
            "key": "bad",
            "label": "Wrong matches",
            "min": 0,
            "max": 10,
            "step": 1,
            "value": 2,
            "unit": ""
          }
        ],
        "tryThis": "Add wrong matches; does mean point-to-point alignment remain reliable?",
        "limit": "1D fixed-correspondence least-squares translation, not iterative 3D ICP.",
        "ref": "§8.2"
      },
      {
        "key": "deskew",
        "title": "Undo motion distortion",
        "question": "How far does a moving platform displace late scan points if treated as simultaneous?",
        "params": [
          {
            "key": "speed",
            "label": "Platform speed",
            "min": 0,
            "max": 6,
            "step": 0.2,
            "value": 2,
            "unit": "m/s"
          },
          {
            "key": "duration",
            "label": "Sweep time",
            "min": 0.02,
            "max": 0.5,
            "step": 0.02,
            "value": 0.2,
            "unit": "s"
          },
          {
            "key": "time",
            "label": "Point timestamp fraction",
            "min": 0,
            "max": 1,
            "step": 0.05,
            "value": 0.8,
            "unit": ""
          }
        ],
        "tryThis": "At fixed speed double sweep time, then use per-point compensation.",
        "limit": "Constant-velocity translational scan in 2D; excludes rotation.",
        "ref": "§8.2.2"
      },
      {
        "key": "place",
        "title": "Test a place descriptor",
        "question": "How does a similarity threshold trade missed closures against false matches?",
        "params": [
          {
            "key": "threshold",
            "label": "Acceptance threshold",
            "min": 0.3,
            "max": 0.95,
            "step": 0.05,
            "value": 0.7,
            "unit": ""
          },
          {
            "key": "noise",
            "label": "Descriptor noise",
            "min": 0,
            "max": 0.3,
            "step": 0.02,
            "value": 0.1,
            "unit": ""
          }
        ],
        "tryThis": "Lower the threshold to capture more true places; watch aliasing increase.",
        "limit": "Synthetic deterministic scores, not a learned Scan Context/descriptor.",
        "ref": "§8.3"
      }
    ],
    "checks": [
      {
        "q": "Why can nearest-neighbor ICP fail even when the optimizer converges?",
        "a": "Incorrect correspondences or repeated geometry may create a local minimum that explains the wrong matches."
      },
      {
        "q": "Why are timestamps important for a spinning LiDAR?",
        "a": "Different points arrive at different robot poses; motion-compensating each point avoids introducing scan distortion."
      }
    ],
    "watch": "Geometric degeneracy, moving objects, reflective surfaces and calibration errors can all corrupt a LiDAR factor."
  },
  {
    "n": 9,
    "title": "Radar SLAM",
    "pages": "250–281",
    "goal": "Connect radar range, Doppler and detection uncertainty to robust localization.",
    "ideas": [
      {
        "title": "Radar measures more than distance",
        "text": "Many radars resolve target range from delay, radial velocity from Doppler frequency and direction from antenna geometry. A target's Doppler speed is only the velocity projected onto the line of sight; tangential motion can be invisible in a single look direction.",
        "equation": "v_r = v ⋅ r̂",
        "ref": "§9.1, pp.250–261"
      },
      {
        "title": "Detection is a statistical decision",
        "text": "Radar returns often have speckle, clutter and multipath. A CFAR-like detector adapts a threshold to a local noise estimate instead of applying one absolute cutoff, trading false alarms against missed targets. Downstream odometry must respect the resulting uncertainty and sparse structure.",
        "equation": "threshold = scale × local clutter power",
        "ref": "§9.1–9.2, pp.250–269"
      },
      {
        "title": "Multipath tests what a detection means",
        "text": "Reflections may create ghost ranges or mirrored geometry; even a consistent-looking return is not always a direct path to a real landmark. Radar place recognition and SLAM need robust correspondence, motion compensation and cross-view consistency, especially in feature-poor conditions.",
        "equation": "τ_path = path_length/c",
        "ref": "§9.2–9.4, pp.261–279"
      }
    ],
    "demos": [
      {
        "key": "doppler",
        "title": "Project motion onto a beam",
        "question": "At which target aspect angle does Doppler fail to observe tangential motion?",
        "params": [
          {
            "key": "speed",
            "label": "Relative speed",
            "min": 0,
            "max": 20,
            "step": 1,
            "value": 12,
            "unit": "m/s"
          },
          {
            "key": "angle",
            "label": "Beam aspect angle",
            "min": 0,
            "max": 180,
            "step": 5,
            "value": 65,
            "unit": "°"
          }
        ],
        "tryThis": "Move angle toward 90°. The object can move rapidly while measured radial speed vanishes.",
        "limit": "Single monostatic radial-velocity projection without sensor ego-motion.",
        "ref": "§9.1"
      },
      {
        "key": "cfar",
        "title": "Adapt a radar threshold",
        "question": "How can a local clutter increase make a previously detectable target disappear?",
        "params": [
          {
            "key": "clutter",
            "label": "Local clutter power",
            "min": 0.2,
            "max": 8,
            "step": 0.2,
            "value": 2,
            "unit": ""
          },
          {
            "key": "scale",
            "label": "CFAR multiplier",
            "min": 1.2,
            "max": 5,
            "step": 0.2,
            "value": 2.6,
            "unit": ""
          },
          {
            "key": "signal",
            "label": "Target power",
            "min": 1,
            "max": 20,
            "step": 0.5,
            "value": 9,
            "unit": ""
          }
        ],
        "tryThis": "Increase clutter with fixed target power, then change threshold scaling.",
        "limit": "Power-threshold illustration; not a calibrated CA-CFAR false-alarm analysis.",
        "ref": "§9.1"
      },
      {
        "key": "ghost",
        "title": "Find the reflected range",
        "question": "How much longer is a reflected path than the direct line of sight?",
        "params": [
          {
            "key": "x",
            "label": "Target horizontal distance",
            "min": 2,
            "max": 15,
            "step": 0.5,
            "value": 8,
            "unit": "m"
          },
          {
            "key": "height",
            "label": "Target offset from reflecting wall",
            "min": 1,
            "max": 8,
            "step": 0.5,
            "value": 3,
            "unit": "m"
          },
          {
            "key": "wall",
            "label": "Wall lateral offset",
            "min": 0,
            "max": 5,
            "step": 0.5,
            "value": 2,
            "unit": "m"
          }
        ],
        "tryThis": "Move reflector farther and compare direct and image-path lengths.",
        "limit": "2D single-bounce image-method geometry; real radar paths are more complicated.",
        "ref": "§9.1, 9.4"
      }
    ],
    "checks": [
      {
        "q": "Does zero Doppler mean a target is stationary?",
        "a": "No. Its velocity may be perpendicular to the line of sight, or relative radial motions may cancel."
      },
      {
        "q": "Why is a strong reflected return not proof of a nearby direct-path landmark?",
        "a": "Multipath can add path length or create virtual image positions not occupied by a physical object."
      }
    ],
    "watch": "Radar resolution and clutter differ dramatically by hardware and waveform; these experiments abstract those details."
  },
  {
    "n": 10,
    "title": "Event-based SLAM",
    "pages": "282–303",
    "goal": "Understand why asynchronous brightness changes support fast motion estimation but require different inference.",
    "ideas": [
      {
        "title": "Events describe changes, not frames",
        "text": "An event camera emits a timestamped pixel event when log brightness changes by a contrast threshold; its polarity reports the sign. A static, unchanging patch may produce no events even if objects are present. Processing must work with sparse asynchronous signals.",
        "equation": "Δ log I(x,y,t) ≈ ±C",
        "ref": "§10.1, pp.282–286"
      },
      {
        "title": "Temporal alignment reveals motion",
        "text": "Events caused by a moving edge trace a trajectory in space-time. Warping event coordinates under a candidate motion can concentrate them into a sharp edge; incorrect velocity smears the accumulation. Contrast objectives can help estimate motion without conventional frames.",
        "equation": "x_ref = x_event − v(t−t_ref)",
        "ref": "§10.3–10.4, pp.287–293"
      },
      {
        "title": "Speed and threshold change event density",
        "text": "A higher contrast threshold suppresses small changes; fast motion or high contrast can produce many events. Noise, refractory effects, feature sparsity and mixed motions require careful front-end treatment; a high event rate is not automatically more geometric information.",
        "equation": "rate ∝ |d log I/dt| / C  (idealized)",
        "ref": "§10.1–10.6, pp.282–294"
      }
    ],
    "demos": [
      {
        "key": "events",
        "title": "Trigger an event camera",
        "question": "At what log-contrast threshold does a moving edge stop triggering events?",
        "params": [
          {
            "key": "contrast",
            "label": "Log-brightness change",
            "min": 0,
            "max": 2,
            "step": 0.05,
            "value": 1.1,
            "unit": ""
          },
          {
            "key": "threshold",
            "label": "Event threshold",
            "min": 0.1,
            "max": 0.7,
            "step": 0.05,
            "value": 0.25,
            "unit": ""
          }
        ],
        "tryThis": "Increase C while holding total contrast fixed. Count the ideal events.",
        "limit": "Single pixel with a deterministic step in log intensity.",
        "ref": "§10.1"
      },
      {
        "key": "warp",
        "title": "Focus an event streak",
        "question": "Which candidate velocity aligns an event track into one sharp x coordinate?",
        "params": [
          {
            "key": "speed",
            "label": "Actual edge velocity",
            "min": -4,
            "max": 4,
            "step": 0.2,
            "value": 2,
            "unit": "px/ms"
          },
          {
            "key": "guess",
            "label": "Warp velocity",
            "min": -4,
            "max": 4,
            "step": 0.2,
            "value": 1,
            "unit": "px/ms"
          },
          {
            "key": "window",
            "label": "Accumulation window",
            "min": 1,
            "max": 10,
            "step": 1,
            "value": 6,
            "unit": "ms"
          }
        ],
        "tryThis": "Match the guessed velocity to actual velocity; dispersion should fall.",
        "limit": "Single edge with constant velocity, no background activity.",
        "ref": "§10.4"
      },
      {
        "key": "refractory",
        "title": "Trade event rate for throughput",
        "question": "How many ideal events are suppressed by a pixel refractory interval?",
        "params": [
          {
            "key": "rate",
            "label": "Raw event rate",
            "min": 100,
            "max": 2000,
            "step": 100,
            "value": 900,
            "unit": "Hz"
          },
          {
            "key": "dead",
            "label": "Refractory interval",
            "min": 0,
            "max": 5,
            "step": 0.2,
            "value": 1,
            "unit": "ms"
          }
        ],
        "tryThis": "Increase refractory interval; examine retained rate.",
        "limit": "Non-paralyzable dead-time approximation for one pixel.",
        "ref": "§10.1"
      }
    ],
    "checks": [
      {
        "q": "What does an event-camera pixel primarily report?",
        "a": "The time, location and sign of a sufficiently large change in log brightness, not an absolute brightness measurement."
      },
      {
        "q": "Why can correct motion compensation sharpen an accumulated event image?",
        "a": "Events from one moving edge are shifted to the same reference-time coordinate instead of being spread along its trajectory."
      }
    ],
    "watch": "Event generation also depends on texture, illumination, sensor bias, latency and noise."
  },
  {
    "n": 11,
    "title": "Inertial Odometry for SLAM",
    "pages": "304–332",
    "goal": "See how high-rate IMU integration builds motion constraints and where its ambiguity comes from.",
    "ideas": [
      {
        "title": "Inertial measurements accumulate error",
        "text": "A gyroscope observes angular velocity plus bias and noise; an accelerometer measures specific force, not world-frame acceleration directly. Pose integrates these signals over time. A small constant acceleration bias produces position error growing roughly quadratically with time.",
        "equation": "Δp_bias ≈ ½ bₐ t²",
        "ref": "§11.1, pp.304–308"
      },
      {
        "title": "Preintegration avoids reintegrating everything",
        "text": "An IMU may sample far more frequently than cameras or LiDARs. Preintegrating measurements between two keyframes compresses hundreds of samples into a relative-motion factor and uncertainty, with bias corrections when linearization changes. It does not magically eliminate bias.",
        "equation": "ΔR, Δv, Δp over [tᵢ,tⱼ]",
        "ref": "§11.2, pp.308–319"
      },
      {
        "title": "Observability sets limits on what can be learned",
        "text": "IMU-only navigation lacks a stable global position, heading and often other gauge-related quantities. Vision or other aids can provide some constraints, but yaw, scale and biases depend on motion and sensor configuration. A well-converged least squares solution need not make unobservable directions identifiable.",
        "equation": "rank(Fisher information) < state dimension",
        "ref": "§11.3–11.4, pp.319–330"
      }
    ],
    "demos": [
      {
        "key": "imu",
        "title": "Integrate a biased accelerometer",
        "question": "How quickly does position drift with a small acceleration bias?",
        "params": [
          {
            "key": "bias",
            "label": "Acceleration bias",
            "min": 0,
            "max": 0.5,
            "step": 0.01,
            "value": 0.12,
            "unit": "m/s²"
          },
          {
            "key": "time",
            "label": "Elapsed time",
            "min": 1,
            "max": 30,
            "step": 1,
            "value": 15,
            "unit": "s"
          }
        ],
        "tryThis": "Double elapsed time: does error double or quadruple?",
        "limit": "Constant bias, rest start, one spatial axis and no orientation error.",
        "ref": "§11.1"
      },
      {
        "key": "preint",
        "title": "Accumulate IMU delta",
        "question": "How do sample rate and bias affect a preintegrated velocity and displacement?",
        "params": [
          {
            "key": "freq",
            "label": "Sampling rate",
            "min": 20,
            "max": 400,
            "step": 20,
            "value": 200,
            "unit": "Hz"
          },
          {
            "key": "duration",
            "label": "Keyframe interval",
            "min": 0.1,
            "max": 2,
            "step": 0.1,
            "value": 0.9,
            "unit": "s"
          },
          {
            "key": "accel",
            "label": "True specific acceleration",
            "min": -2,
            "max": 3,
            "step": 0.1,
            "value": 1.2,
            "unit": "m/s²"
          }
        ],
        "tryThis": "Double sampling rate. For constant noiseless acceleration, should the physical delta change?",
        "limit": "Discrete constant-acceleration integration, without gravity rotation and stochastic covariance.",
        "ref": "§11.2"
      },
      {
        "key": "yaw",
        "title": "Find a heading gauge",
        "question": "Can relative gyro turns reveal a global heading with no compass?",
        "params": [
          {
            "key": "initial",
            "label": "Unknown initial heading",
            "min": -180,
            "max": 180,
            "step": 10,
            "value": 40,
            "unit": "°"
          },
          {
            "key": "turn",
            "label": "Integrated turn",
            "min": -90,
            "max": 90,
            "step": 5,
            "value": 35,
            "unit": "°"
          },
          {
            "key": "anchor",
            "label": "Absolute heading observation (0/1)",
            "min": 0,
            "max": 1,
            "step": 1,
            "value": 0,
            "unit": ""
          }
        ],
        "tryThis": "Disable heading observation and change initial angle. Relative turns remain identical.",
        "limit": "Planar heading-only observability illustration.",
        "ref": "§11.3"
      }
    ],
    "checks": [
      {
        "q": "Why can constant acceleration bias cause position error proportional to time squared?",
        "a": "Acceleration is integrated once into velocity error and a second time into position error: Δp ≈ ½ bₐt²."
      },
      {
        "q": "Does IMU preintegration eliminate the need to estimate biases?",
        "a": "No. Bias is explicitly modeled and can be corrected or relinearized; preintegration primarily reduces repeated integration work."
      }
    ],
    "watch": "Strong vibrations, time offsets, gravity misalignment and biases complicate inertial factors considerably."
  },
  {
    "n": 12,
    "title": "Leg Odometry for SLAM",
    "pages": "333–358",
    "goal": "Connect contact reasoning and robot kinematics to motion estimates, especially under slip.",
    "ideas": [
      {
        "title": "Contacts give temporary world anchors",
        "text": "When a foot remains planted, its world-frame position is approximately constant. Joint encoders and a kinematic model then constrain body motion relative to that foot. When the foot swings or slips, that stationarity assumption fails and the corresponding factor should not be trusted equally.",
        "equation": "p_foot,world ≈ constant during stance",
        "ref": "§12.1–12.2, pp.333–345"
      },
      {
        "title": "Contact detection is probabilistic",
        "text": "Force, joint dynamics and motion cues can indicate whether each foot is in stance. Hard thresholds are straightforward but brittle near touchdown; soft contact probabilities allow uncertainty-weighted fusion. Wrongly asserting contact can generate large odometry errors.",
        "equation": "w_contact ∈ [0,1]",
        "ref": "§12.3, pp.345–348"
      },
      {
        "title": "Leg odometry is another sensor in the graph",
        "text": "A leg odometry estimate can complement inertial and visual motion, especially where camera texture or lighting fails. Its accuracy hinges on calibration, joint encoder bias, impact behavior, compliant feet, slipping terrain and the support configuration.",
        "equation": "Σ_fused⁻¹ = Σ₁⁻¹ + Σ₂⁻¹  (independent scalar example)",
        "ref": "§12.4–12.5, pp.348–355"
      }
    ],
    "demos": [
      {
        "key": "contact",
        "title": "Recognize a stance phase",
        "question": "When should a stationary-foot constraint be activated?",
        "params": [
          {
            "key": "force",
            "label": "Normalized foot force",
            "min": 0,
            "max": 1,
            "step": 0.05,
            "value": 0.65,
            "unit": ""
          },
          {
            "key": "vel",
            "label": "Foot speed",
            "min": 0,
            "max": 2,
            "step": 0.1,
            "value": 0.35,
            "unit": "m/s"
          },
          {
            "key": "strict",
            "label": "Contact strictness",
            "min": 0.2,
            "max": 1,
            "step": 0.05,
            "value": 0.6,
            "unit": ""
          }
        ],
        "tryThis": "Lower force and increase foot speed; how does stance confidence respond?",
        "limit": "Heuristic logistic contact score, not learned or calibrated foot-state inference.",
        "ref": "§12.3"
      },
      {
        "key": "slip",
        "title": "Detect a broken foot anchor",
        "question": "How does terrain slip corrupt an estimate that assumes a planted foot?",
        "params": [
          {
            "key": "slip",
            "label": "Actual foot slip velocity",
            "min": 0,
            "max": 1.5,
            "step": 0.1,
            "value": 0.6,
            "unit": "m/s"
          },
          {
            "key": "stance",
            "label": "Stance duration",
            "min": 0.1,
            "max": 3,
            "step": 0.1,
            "value": 1.5,
            "unit": "s"
          },
          {
            "key": "conf",
            "label": "Assumed contact confidence",
            "min": 0,
            "max": 1,
            "step": 0.1,
            "value": 0.9,
            "unit": ""
          }
        ],
        "tryThis": "Increase confidence while a foot is actually slipping. Does the bias get better?",
        "limit": "Constant-velocity 1D slip bias model.",
        "ref": "§12.2–12.4"
      },
      {
        "key": "fusion",
        "title": "Fuse kinematics and IMU",
        "question": "What does uncertainty weighting do when contact measurements disagree with inertial motion?",
        "params": [
          {
            "key": "leg",
            "label": "Leg-derived displacement",
            "min": 0,
            "max": 6,
            "step": 0.2,
            "value": 2.5,
            "unit": "m"
          },
          {
            "key": "sigmaLeg",
            "label": "Leg uncertainty",
            "min": 0.1,
            "max": 3,
            "step": 0.1,
            "value": 0.8,
            "unit": "m"
          },
          {
            "key": "sigmaImu",
            "label": "IMU uncertainty",
            "min": 0.1,
            "max": 3,
            "step": 0.1,
            "value": 1.4,
            "unit": "m"
          }
        ],
        "tryThis": "Make the leg reading precise, then deliberately inconsistent. What assumption would break?",
        "limit": "Two independent Gaussian scalar estimates, not a dynamic filter.",
        "ref": "§12.4"
      }
    ],
    "checks": [
      {
        "q": "Why must a leg odometry system distinguish stance from swing?",
        "a": "A stance foot may be modeled as stationary against the world. A swinging foot is not a fixed world reference."
      },
      {
        "q": "What happens if a slipping foot is modeled as firmly planted?",
        "a": "The body motion estimate absorbs foot slip as apparent robot motion, producing biased odometry."
      }
    ],
    "watch": "Contact state can be uncertain and surfaces deform; avoid treating all foot contacts as perfect landmarks."
  }
]);
