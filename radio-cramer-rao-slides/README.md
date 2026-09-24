# Cramér–Rao Bounds for Radio Measurements

Public presentation: <https://bailiping.com/radio-cramer-rao-slides/>.

Updated September 24, 2026: **40 slides and seven live experiments**, including physical coded pilots and known-channel timing bounds.

- [Coded-pilot calculator](https://bailiping.com/radio-cramer-rao-slides/project/?lab=coded)
- [Known channel and receiver clock](https://bailiping.com/radio-cramer-rao-slides/project/?lab=known)
- [Orthogonal-pilot reference calculator](https://bailiping.com/radio-cramer-rao-slides/live/?lab=calculator)
- [Fisher information](https://bailiping.com/radio-cramer-rao-slides/live/?lab=fisher)
- [Bandwidth and observation time](https://bailiping.com/radio-cramer-rao-slides/live/?lab=bandwidth)
- [3D array geometry](https://bailiping.com/radio-cramer-rao-slides/live/?lab=geometry)
- [Two-path gain separability](https://bailiping.com/radio-cramer-rao-slides/live/?lab=multipath)

This companion to **Radio Measurements → Radio Map** derives local bounds for delay, two arrival angles, two departure angles, effective path attenuation, and reference phase. The cover links directly to both new experiments. The slide IDs `project-setup` and `known-channel` are stable entry points.

## Physical coded-pilot model

The configured observation design follows the [radio SLAM system model](https://github.com/BaiLiping/radio-SLAM-system-model):

$$Y_k=e^{-j2\pi f_k\beta}H_k^{\mathrm g}X+W_k.$$

The default dimensions are 384 TX ports, 32 RX ports, 50 symbols, and 3300 active tones over 400 MHz at 27.2 GHz. The arrays use half-wavelength spacing with 24 horizontal and 16 vertical TX elements, and eight horizontal and four vertical RX elements. One known QPSK matrix `X` is reused on every tone. Physical pilot entries have magnitude `sqrt(P / (Nt K))`. The noise density is `10^((-174 + NF - 30)/10)` W/Hz, and the complex sample variance is this density times `B/K`. At NF = 9 dB the default variance is `3.8330638305071264e-15` W. Noise is not divided by symbol count or pilot power.

The calculator uses a representative isolated static far-field path with calibrated effective attenuation. It computes the full seven-parameter Fisher matrix from the derivatives of `H_k X`, including the coded-pilot gain/phase cross terms. The parameter units are ns, degrees, degrees, degrees, degrees, dB, and radians. Its per-coordinate marginal bounds account for nuisance coupling and local unobservable directions.

The browser's deterministic QPSK generator supplies **illustrative pilots**. It does not reproduce a saved NumPy pilot matrix or a campus RT channel realization. Consequently, the displayed values are conditional examples for this observation design. A bound for an actual acquisition requires its saved `X`, path realization, and physical parameter derivatives. The effective path attenuation includes calibrated propagation and element effects once. No extra element gain or RT steering factor should be multiplied into already complete exported coefficients.

With 50 symbols, the pilot rank is at most 50. Full arbitrary per-tone CSI recovery needs row rank 384, while structured angle/delay estimation depends on the rank of the much smaller real parameter Jacobian. The new calculator therefore supports `L < Nt`. A single repeated spatial excitation cannot identify AoD jointly with a free complex gain.

## Known-channel interpretation

The slides and second new experiment distinguish:

1. **Simulator truth:** evaluate a CRB at the true channel while the estimator receives noisy I/Q and known pilots.
2. **Exact geometric channel:** the receiver knows `H^g`, but may still need to estimate clock bias and an optional common receiver phase.
3. **Exact effective channel:** identifiable functions of `Htilde` have no receiver-noise uncertainty, while delay/clock or path-decomposition ambiguities can remain.
4. **Noisy CSI:** retain its error covariance in the likelihood. It is not exact channel knowledge.

For known `H^g`, let `w_k = ||H_k^g X||_F²`. The clock information with a known phase reference is

$$J_{\beta\beta}=\frac{8\pi^2}{\sigma^2}\sum_k w_k f_k^2.$$

An unknown common receiver phase changes this to

$$J_{\beta,\mathrm{eff}}=\frac{8\pi^2}{\sigma^2}\sum_k w_k(f_k-\bar f_w)^2,$$

where `fbar_w` is the power-weighted mean frequency. These use baseband offsets, consistent with the stated clock model. The known-channel demo varies an illustrative received-energy spectrum at fixed aggregate SNR and shows both bounds. Its delay/clock controls demonstrate the invariant apparent delay `tau_a = tau_g + beta`, a separate identifiability question from the known-geometric-template timing experiment.

## Orthogonal reference model

The five original labs use one static, isolated far-field path, calibrated arrays, narrowband array steering, flat contiguous pilot tones, white circular complex Gaussian noise, and balanced orthogonal transmit training. `Nt` and `Nr` count antenna elements at one BS and one UE. This time-code design requires `L >= Nt`. This is a requirement of that training scheme, not a universal symbol-count requirement for parametric channel estimation.

The total information SNR is `Gamma = Nr K L rho`, where `rho` averages over the transmit code and is the SNR per receiver, tone, and symbol. Transmit energy is shared among `Nt` elements, so no extra `Nt` multiplier appears. The three conventions are fixed aggregate SNR, fixed average per-tone SNR, and a physical power/noise link budget. Grid bandwidth is `B = K deltaF`. Useful duration `L / deltaF` excludes CP and scheduling gaps.

The implementation retains the full azimuth/elevation information blocks and inverts them jointly. It marks singular directions instead of reporting pseudoinverse zeros as finite CRBs. Complex noise variance means `E|n|² = sigma²`, with `sigma²/2` in each real quadrature. Delay refers to apparent propagation delay unless clock offset has been constrained. Per-path attenuation is an effective calibrated channel-gain parameter, not separately an intrinsic wall-loss parameter.

The multipath lab has a narrower model: two known delays and unknown complex gains with identical spatial signatures. Its gain-variance inflation is `1/(1-|chi|²)`, where `chi` is normalized frequency-signature correlation. It is not a joint unknown-delay/angle multipath bound.

## Editable source and build

`bento-deck.mjs` contains the slide text, equations, references, and layout. `build.mjs` uses the existing public radio-geometry deck's licensed Bento runtime and its local MathJax distribution. Generated `index.html`, `deck.json`, and `live-demos.json` should be committed with the source.

The orthogonal reference application is in `live/`, with the `RadioCRB` engine. The coded-pilot and known-channel application is in `project/`, with the `RadioProjectCRB` engine. Both `model.js` files are usable from Node via CommonJS. The applications use local scripts and have no runtime network dependencies for their calculations.

```sh
node radio-cramer-rao-slides/build.mjs
node radio-cramer-rao-slides/tests/model.test.cjs
node radio-cramer-rao-slides/tests/project-model.test.cjs
python3 -m http.server 8767 --bind 127.0.0.1
```

The reference test constructs complex raw-I/Q derivatives for DFT-coded transmit pilots and compares the entire Fisher matrix with the analytical engine. The project test compares the coded-pilot matrix with independent raw-I/Q finite differences and checks physical normalization, nuisance elimination, and singular configurations. Visual checks cover the changed Bento slides and both new labs at desktop, embedded and mobile sizes.

Arrow keys navigate slides. Page Up / Page Down also navigate from inside a live lab. Escape returns focus to the presentation. The standalone lab supports narrower screens. Live panels preserve inputs for the browser session, and inactive iframes unload.

## Primary sources

- X. Li et al., *Optimal and Robust Waveform Design for MIMO-OFDM Channel Sensing: A Cramér-Rao Bound Perspective*, 2023. <https://arxiv.org/abs/2301.10689>
- A. Shahmansoori et al., *Position and Orientation Estimation through Millimeter-Wave MIMO in 5G Systems*, IEEE TWC, 2018. <https://arxiv.org/abs/1702.01605>
- L. Le Magoarou and S. Paquelet, *Channel Estimation: Unified View of Optimal Performance and Pilot Sequences*, IEEE TSP, 2020. <https://arxiv.org/abs/2002.04481>
- P. Poshala, Rushil KK, and R. Gupta, *Signal Chain Noise Figure Analysis*, Texas Instruments SLAA652, October 2014. <https://www.ti.com/lit/pdf/slaa652>

Closed forms and numeric examples are derived for the explicit model above. A local CRB is an optimistic estimation benchmark. It does not specify detection probability, bounce count, ambiguity probability, or automatically supply a calibrated covariance for a SLAM estimator.
