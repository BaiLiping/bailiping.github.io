# Cramér–Rao Bounds for Radio Measurements

Public presentation: <https://bailiping.com/radio-cramer-rao-slides/>.

Updated September 29, 2026: **40 slides and six embedded experiments**. The active acquisition is now analog beam sweeping at both the BS and UE, with one OFDM pilot symbol per tested beam pair.

- [Current analog beam-sweep calculator](https://bailiping.com/radio-cramer-rao-slides/analog/)
- [Acquisition setup](https://bailiping.com/radio-cramer-rao-slides/#project-setup)
- [Reference papers](https://bailiping.com/radio-cramer-rao-slides/#references)
- [Orthogonal-training reference calculator](https://bailiping.com/radio-cramer-rao-slides/live/?lab=calculator)
- [Historical coded-pilot calculator](https://bailiping.com/radio-cramer-rao-slides/project/?lab=coded)

## Current analog acquisition

The BS codebook is 10 by 10 (100 beams); the UE codebook is 8 by 4 (32 beams). The model assumes one active RF stream at each end and one OFDM pilot symbol at each tested pair. An exhaustive Cartesian sweep therefore comprises 3,200 sequential symbols, not one symbol for the entire sweep. Each pair and tone yields one complex scalar after analog combining:

$$y_{mnk}=s_k\mathbf w_n^H\widetilde{\mathbf H}_k\mathbf v_m+z_{mnk}.$$

For one isolated path,

$$\mu_{mnk}=s_k\alpha e^{-j2\pi f_k\tau^{\rm a}}(\mathbf w_n^H\mathbf a_r)(\mathbf a_t^H\mathbf v_m),\qquad \tau^{\rm a}=\tau^{\rm g}+b.$$

The calculator estimates apparent delay, two arrival angles, two departure angles, effective attenuation and reference phase jointly. Unknown gain/phase coupling is retained. It forms the full seven-by-seven Fisher matrix from the scalar observation derivatives:

$$J_{ij}=2\operatorname{Re}\sum_{(m,n)\in\mathcal S}\sum_k\frac{(\partial_i\mu_{mnk})^*(\partial_j\mu_{mnk})}{\sigma^2}.$$

The tone and spatial factors are accumulated using exact frequency moments, equivalent to summing all raw samples. Marginal bounds use a rank-aware normalized eigendecomposition. A coordinate with a component in the null space is reported as unidentifiable, never as a pseudoinverse zero.

One fixed beam pair generally cannot separate the four angles from a freely unknown complex gain in this narrowband-array model. One symbol at each of many different beam pairs can retain sufficient spatial diversity. The relevant rank is the realified structured-parameter Jacobian, not a requirement for 384 independent symbols per pair.

## Retained parameters and explicit assumptions

Beam counts are **not antenna counts**. The previous physical arrays are retained: 24 horizontal by 16 vertical BS elements and 8 horizontal by 4 vertical UE elements, at half-wavelength spacing. The retained OFDM configuration is 27.2 GHz carrier, 3,300 contiguous active tones and 400 MHz grid bandwidth. With delta-f = B/K, one useful symbol is 8.25 microseconds; a full sweep is 26.4 milliseconds and 10,560,000 complex samples. CP, switching time and scheduling gaps are additional.

The actual measured beam weights and angular coverage have not been supplied. The browser uses **illustrative ideal unit-norm phase-shifter beams**, uniformly spaced in azimuth from -60 to 60 degrees and elevation from -30 to 30 degrees at both ends. Coverage, physical arrays, beam-grid counts and path directions are independent inputs. Defaults of 30 dBm transmit power, 9 dB noise figure and 120 dB effective attenuation are teaching inputs, not measured hardware performance.

The path is isolated, static and far field. The sweep retains complex I/Q and assumes a common complex path gain with stable or compensated phase across pairs. Narrowband array steering at the carrier excludes beam squint. Motion requires actual timestamps and appropriate Doppler/channel evolution; phase drift and switching calibration may require nuisance parameters. Power-only beam measurements require a different likelihood. No channel coherence guarantee follows from the sweep duration.

The full schedule can be compared with one fixed pair, a BS-only sweep, or a UE-only sweep. Partial-sweep examples freeze a strongest beam at the true evaluation point. They do not simulate adaptive beam selection or charge its search overhead.

The analog weights have unit norm and array steering entries have unit magnitude. Pilot power per tone is P/K. For spatially white pre-combining noise, the combined complex variance is sigma-squared = N0 delta-f, with N0 = 10^((-174 + NF - 30)/10) W/Hz. At the default tone grid and NF=9 dB, sigma-squared is 3.8330638305071264e-15 W. Do not divide this by 32 beams or 3,200 pairs, and do not multiply antenna gains into responses that already include them. Increasing beam density at fixed power increases both duration and transmitted energy; it is not a fixed-energy comparison.

## Unknown channel and clock

Pilots, beam weights, schedule, calibration and assumed noise are known. The propagation channel is not known to the estimator. True parameters only specify the operating point at which the deterministic CRB is evaluated. The isolated-path calculation is conditional on one path; an unknown number of paths is a separate model-order problem.

Beam sweeping does not remove the clock gauge. Adding delta to b and subtracting delta from every geometric delay leaves the observation unchanged. Channel-level delay bounds are on apparent delay; path-delay differences are clock-free. Geometry, timing references or explicit priors provide additional constraints.

A local CRB is an optimistic estimation benchmark, not a detection probability, ambiguity probability, expected hardware error or automatically calibrated SLAM covariance. A full unknown-multipath bound must retain cross-path derivatives and nuisance coupling.

## Reference and historical models

The five original embedded labs in `live/` remain explicitly identified in the deck as reference models. Their balanced orthogonal transmit training and per-antenna receive observations are **not** the analog acquisition. In particular, their L >= Nt time-code condition, aggregate-SNR normalization and array-covariance angular closed forms must not be substituted for the actual swept-beam FIM.

The two-path reference lab has known delays and unknown complex gains with identical spatial signatures; it is not a joint unknown-delay/angle multipath bound. The previous 384-port, 50-symbol QPSK calculator in `project/` remains accessible as a historical model and links to the current analog calculator.

## Editable source, build and tests

`analog-deck.mjs` is the active acquisition revision. It imports the preserved `bento-deck.mjs` reference derivations, replaces the current setup and equations, labels reference-only sections and installs native clickable Bento paper links. `build.mjs` builds the existing licensed Bento presentation using the neighboring public radio-geometry deck and its local MathJax. Stable entry IDs include `project-setup`, `project-live`, `unknown-channel` and `references`.

The current mathematical engine is `analog/model.js` (browser and CommonJS). The interface is `analog/index.html`; exported JSON includes codebook directions, actual tested pair indices, assumptions, parameter units and the full FIM.

```sh
node radio-cramer-rao-slides/tests/model.test.cjs
node radio-cramer-rao-slides/tests/project-model.test.cjs
node radio-cramer-rao-slides/tests/analog-model.test.cjs
node radio-cramer-rao-slides/build.mjs
python3 -m http.server 8765 --bind 127.0.0.1
# In another shell, with Playwright and Chromium installed:
python3 radio-cramer-rao-slides/tests/analog-browser.py
```

The analog numerical regression compares all FIM entries with independent raw-I/Q finite differences and checks power scaling, normalization, joint inversion and singular configurations. Browser tests check changed-slide layout, math rendering, native paper links, the sandboxed embedded calculator, keyboard navigation and mobile controls. `.github/workflows/build-radio-crb.yml` builds and tests these sources and commits only generated deck assets.

## Reference papers — direct links

- F. Sohrabi and W. Yu, *Hybrid Analog and Digital Beamforming for mmWave OFDM Large-Scale Antenna Arrays*, 2017. <https://arxiv.org/abs/1711.08408>. Used for the analog/hybrid OFDM architecture, not a CRB result or its known-CSI optimization premise.
- X. Li, V. C. Andrei, U. J. Mönich and H. Boche, *Optimal and Robust Waveform Design for MIMO-OFDM Channel Sensing: A Cramér-Rao Bound Perspective*, 2023. <https://arxiv.org/abs/2301.10689>.
- A. Shahmansoori et al., *Position and Orientation Estimation through Millimeter Wave MIMO in 5G Systems*, IEEE TWC, 2018. <https://arxiv.org/abs/1702.01605>.
- L. Le Magoarou and S. Paquelet, *Channel estimation: unified view of optimal performance and pilot sequences*, 2020. <https://arxiv.org/abs/2002.04481>.

Supporting manufacturer technical note, not a research paper: Texas Instruments, *Signal Chain Noise Figure Analysis*, SLAA652. <https://www.ti.com/lit/pdf/slaa652>.

The 100-by-32 sweep is the configured experiment; it is not attributed to a reference paper. Slide-footer references and the reference-slide paper titles/URLs use native Bento links, so the renderer does not strip their navigation.
