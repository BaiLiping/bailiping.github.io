# Cramér–Rao Bounds for Radio Measurements

Public presentation: <https://bailiping.com/radio-cramer-rao-slides/>.

Interactive calculator: <https://bailiping.com/radio-cramer-rao-slides/live/>.

This companion to **Radio Measurements → Radio Map** derives local bounds for delay, two arrival angles, two departure angles, effective path attenuation, and reference phase. Five interactive labs expose the full calculator, Fisher information, bandwidth and observation time, 3D array geometry, and two-path gain separability.

## Model

The default model is one static, isolated far-field path, calibrated arrays, narrowband array steering, flat contiguous pilot tones, white circular complex Gaussian noise, and balanced orthogonal transmit training. `Nt` and `Nr` count antenna elements at one BS and one UE. The time-code design requires `L >= Nt`. This is a requirement of this training scheme, not a universal symbol-count requirement for every parametric channel estimator.

The total information SNR is `Gamma = Nr K L rho`, where `rho` averages over the transmit code and is the SNR per receiver, tone, and symbol. Transmit energy is shared among `Nt` elements, so no extra `Nt` multiplier appears. The three conventions are fixed aggregate SNR, fixed average per-tone SNR, and a physical power/noise link budget. Grid bandwidth is `B = K deltaF`. Useful duration `L / deltaF` excludes CP and scheduling gaps.

The implementation retains the full azimuth/elevation information blocks and inverts them jointly. It marks singular directions instead of reporting pseudoinverse zeros as finite CRBs. Complex noise variance means `E|n|² = sigma²`, with `sigma²/2` in each real quadrature. Delay refers to apparent propagation delay unless clock offset has been constrained. Per-path attenuation is an effective calibrated channel-gain parameter, not separately an intrinsic wall-loss parameter.

The multipath lab has a narrower model: two known delays and unknown complex gains with identical spatial signatures. Its gain-variance inflation is `1/(1-|chi|²)`, where `chi` is normalized frequency-signature correlation. It is not a joint unknown-delay/angle multipath bound.

## Editable source and build

`bento-deck.mjs` contains the slide text, equations, references, and layout. `build.mjs` uses the existing public radio-geometry deck's licensed Bento runtime and its local MathJax distribution. Generated `index.html`, `deck.json`, and `live-demos.json` should be committed with the source.

The live application is in `live/`. Its numerical engine is `model.js`, usable from a browser as `RadioCRB` or from Node via CommonJS. There are no runtime network dependencies for the calculator.

```sh
node radio-cramer-rao-slides/build.mjs
node radio-cramer-rao-slides/tests/model.test.cjs
python3 -m http.server 8767 --bind 127.0.0.1
```

The numerical test independently constructs complex raw-I/Q derivatives for DFT-coded transmit pilots and compares the entire Fisher information matrix with the analytical engine. It also verifies marginal covariance, resource normalization, and singular configurations.

Arrow keys navigate slides. Page Up / Page Down also navigate from inside a live lab. Escape returns focus to the presentation. The standalone lab supports narrower screens. Live panels preserve inputs for the browser session, and inactive iframes unload.

## Primary sources

- X. Li et al., *Optimal and Robust Waveform Design for MIMO-OFDM Channel Sensing: A Cramér-Rao Bound Perspective*, 2023. <https://arxiv.org/abs/2301.10689>
- A. Shahmansoori et al., *Position and Orientation Estimation through Millimeter-Wave MIMO in 5G Systems*, IEEE TWC, 2018. <https://arxiv.org/abs/1702.01605>
- L. Le Magoarou and S. Paquelet, *Channel Estimation: Unified View of Optimal Performance and Pilot Sequences*, IEEE TSP, 2020. <https://arxiv.org/abs/2002.04481>
- P. Poshala, Rushil KK, and R. Gupta, *Signal Chain Noise Figure Analysis*, Texas Instruments SLAA652, October 2014. <https://www.ti.com/lit/pdf/slaa652>

Closed forms and numeric examples are derived for the explicit model above. A local CRB is an optimistic estimation benchmark. It does not specify detection probability, bounce count, ambiguity probability, or automatically supply a calibrated covariance for a SLAM estimator.
