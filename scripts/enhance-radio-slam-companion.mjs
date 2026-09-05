import { readFileSync, writeFileSync } from 'node:fs'
import { resolve } from 'node:path'

const pagePath = resolve('mpc-detection-to-bounce-count/index.html')
let html = readFileSync(pagePath, 'utf8')
let changed = false

function replaceRequired(before, after, label, { all = false } = {}) {
  if (html.includes(after)) return
  if (!html.includes(before)) throw new Error(`Could not find ${label}`)
  html = all ? html.split(before).join(after) : html.replace(before, after)
  changed = true
}

function replaceOptional(before, after, { all = false } = {}) {
  if (html.includes(after) || !html.includes(before)) return
  html = all ? html.split(before).join(after) : html.replace(before, after)
  changed = true
}

function insertBefore(marker, addition, sentinel, label) {
  if (html.includes(sentinel)) return
  const index = html.indexOf(marker)
  if (index < 0) throw new Error(`Could not find ${label}`)
  html = html.slice(0, index) + addition + html.slice(index)
  changed = true
}

function replaceBlock(startMarker, endMarker, replacement, sentinel, label) {
  if (html.includes(sentinel)) return
  const start = html.indexOf(startMarker)
  const end = html.indexOf(endMarker, start + startMarker.length)
  if (start < 0 || end < 0) throw new Error(`Could not find ${label}`)
  html = html.slice(0, start) + replacement + html.slice(end)
  changed = true
}

replaceRequired(
  '<meta name="description" content="Interactive note: convert resolved multipath-component measurements into bounce counts with known or unknown UE pose and map, anchored by a known BS pose.">',
  '<meta name="description" content="Detailed interactive companion: derive radio-SLAM measurement factors from delay, AoA, AoD, and path gain; recover UE trajectory and propagation map; and express bistatic radio sensing as GraphSLAM.">',
  'page description'
)

insertBefore(
  '</head>',
  '<link rel="stylesheet" href="companion.css">\n',
  'href="companion.css"',
  'head closing tag for companion stylesheet'
)

const mathJaxHead = String.raw`<script id="radio-companion-mathjax-config">
window.MathJax={
  tex:{inlineMath:[['\\(','\\)']],displayMath:[['\\[','\\]']],processEscapes:true},
  svg:{fontCache:'global'}
};
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-svg-full.js"></script>
`
insertBefore(
  '</head>',
  mathJaxHead,
  'id="radio-companion-mathjax-config"',
  'head closing tag for MathJax'
)

replaceOptional(
  '    <a href="#measurement">Measurement</a>',
  '    <a href="#notation">Notation</a>\n    <a href="#measurement">Measurement</a>'
)
replaceOptional(
  '    <a href="#unknown-pose-map">Unknown UE + map</a>',
  '    <a href="#unknown-pose-map">Unknown UE + map</a>\n    <a href="#bistatic-graphslam">GraphSLAM bridge</a>'
)

replaceRequired(
  '<p class="thesis">A resolved multipath component can provide four observables: a <strong>delay</strong> (a path length), an <strong>arrival angle</strong> at the UE, a <strong>departure angle</strong> at the BS, and a <strong>path loss</strong>. These measurements constrain the physical route behind each MPC detection and its <strong>bounce count</strong>. Section 2 tests LoS and reflection sequences against a known pose and map; Section 3 keeps the poses known while inferring the map; Section 4 fixes only the BS pose and jointly hypothesizes the UE position, UE heading, and walls.</p>',
  '<p class="thesis">A resolved multipath component can provide a <strong>delay</strong>, a UE-frame <strong>angle of arrival</strong>, a BS-frame <strong>angle of departure</strong>, and a complex or power-related <strong>path gain</strong>. After timing calibration, delay constrains total propagation length; the endpoint angles constrain bearing rays; calibrated gain can help rank competing routes. Radio SLAM jointly reconstructs the <strong>UE trajectory</strong>, a persistent <strong>radio map</strong>, and the discrete path explanation behind each MPC.</p>',
  'hero thesis'
)

replaceRequired(
  '<div class="accuracy"><strong>Context.</strong> The theory uses the same kind of per-path tuple that a SAGE estimator can produce, but every scene and number on this page is illustrative rather than a replay of measured SAGE output. The current <code>Gaussian_Splatting_Test</code> benchmark consumes synthetic ray-tracer path observables with world-aligned, translation-only pose states. The virtual-anchor reading follows E. Leitinger et&nbsp;al., <a href="https://arxiv.org/abs/1801.04463">“A Belief Propagation Algorithm for Multipath-Based SLAM,”</a> <i>IEEE Trans. Wireless Commun.</i> 18(12), 2019. Scale in the 2D drawings: 1&nbsp;px ≙ 0.1&nbsp;m.</div>',
  '<div class="accuracy"><strong>How to use this page.</strong> The slide deck gives the visual argument; this companion spells out variables, coordinate frames, likelihoods, factor-graph structure, and observability assumptions. The scenes remain illustrative rather than replays of measured SAGE output. The current <code>Gaussian_Splatting_Test</code> benchmark uses synthetic ray-tracer observables and a simplified world-aligned, translation-only implementation, while the derivation below states the more general joint pose–map model. Scale in the 2D drawings: 1&nbsp;px ≙ 0.1&nbsp;m.</div>',
  'hero context'
)

replaceOptional(
  '<a class="is-measurement" href="#measurement"><span>01 · input</span><strong>Measurement</strong><small>delay · angles · path loss</small></a>',
  '<a class="is-measurement" href="#measurement"><span>01 · input</span><strong>Measurement</strong><small>delay · local angles · path gain</small></a>'
)

const notationSection = String.raw`
<!-- ============ 00 notation and reading guide ============ -->
<section class="sec companion-section" id="notation">
  <h2><span class="no">00</span>Notation and reading guide</h2>
  <p class="lede">The interactive figures are two-dimensional, but the factor-graph construction is not tied to 2D. This guide distinguishes <em>measured quantities</em>, <em>derived world-frame rays</em>, <em>unknown variables</em>, and <em>known calibration parameters</em> before the geometry begins.</p>

  <div class="accuracy"><strong>Frame convention.</strong> Superscripts identify the coordinate frame: \(W\) is the world/map frame, \(B_s\) is the local frame of base station \(s\), and \(U_t\) is the UE body/array frame at time \(t\). Measured AoA and AoD are local-frame angles. They become world-frame bearing rays only after adding the corresponding known or hypothesized heading.</div>

  <nav class="subsection-tiles is-four" aria-label="Notation subsections">
    <a href="#indices-and-states"><span>0.1</span><strong>Indices &amp; states</strong><small>time · BS · MPC · map</small></a>
    <a href="#angle-frames"><span>0.2</span><strong>Frames &amp; angles</strong><small>local measurement → world ray</small></a>
    <a href="#measurement-noise"><span>0.3</span><strong>Noise &amp; weights</strong><small>covariance · whitening</small></a>
    <a href="#model-boundary"><span>0.4</span><strong>Model boundary</strong><small>what the demos assume</small></a>
  </nav>

  <div class="subsection-block" id="indices-and-states">
    <h3 class="subh"><span class="no">0.1</span>Indices, variables, and known quantities</h3>
    <table class="companion-symbols">
      <thead><tr><th>Symbol</th><th>Type</th><th>Meaning</th></tr></thead>
      <tbody>
        <tr><td>\(t=1,\ldots,T\)</td><td>time index</td><td>One UE state or scan time.</td></tr>
        <tr><td>\(s=1,\ldots,S\)</td><td>BS index</td><td>The transmitting base station, also called a physical anchor.</td></tr>
        <tr><td>\(\ell=1,\ldots,n_{ts}\)</td><td>MPC index</td><td>One resolved component in the measurement set at time \(t\) from BS \(s\). It is not automatically a persistent path label.</td></tr>
        <tr><td>\(j=1,\ldots,J\)</td><td>map index</td><td>A persistent scatterer, wall, virtual anchor, or reflector-chain state.</td></tr>
        <tr><td>\(q\)</td><td>path class</td><td>LoS, one-bounce, two-bounce, or another explicitly modeled propagation family.</td></tr>
        <tr><td>\(a_{ts\ell}\)</td><td>association</td><td>The complete path assigned to MPC \(\ell\): LoS or an ordered wall chain; \(a=0\) denotes clutter.</td></tr>
        <tr><td>\(\mathbf B_s=(\mathbf b_s,\theta_s)\)</td><td>known or calibrated</td><td>BS position and array heading. Promote it to a graph variable only when uncertain.</td></tr>
        <tr><td>\(\mathbf T_t=(\mathbf p_t,\theta_t)\)</td><td>unknown UE pose</td><td>UE position and orientation in 2D; use an \(SE(3)\) pose in 3D.</td></tr>
        <tr><td>\(\mathcal M=\{\mathbf m_j\}_{j=1}^{J}\)</td><td>unknown map</td><td>The collection of persistent radio-map variables.</td></tr>
        <tr><td>\(\boldsymbol\kappa\)</td><td>calibration / nuisance</td><td>Clock bias, hardware delay, antenna calibration, transmit-power offset, or material parameters.</td></tr>
      </tbody>
    </table>
    <div class="eq math-eq">
      \[
      \mathbf z_{ts\ell}=
      \begin{bmatrix}
        \tau_{ts\ell} & \varphi^{U}_{ts\ell} & \psi^{B}_{ts\ell} & g^{\mathrm{dB}}_{ts\ell}
      \end{bmatrix}^{\mathsf T}.
      \]
    </div>
    <p class="eq-note">Here \(\tau\) is measured in seconds, \(\varphi^U\) is AoA in the UE frame, \(\psi^B\) is AoD in the BS frame, and \(g^{\mathrm{dB}}\) is an optional calibrated gain/loss observable. A channel estimator may instead return a complex coefficient \(\alpha\), with received power proportional to \(|\alpha|^2\).</p>
  </div>

  <div class="subsection-block" id="angle-frames">
    <h3 class="subh"><span class="no">0.2</span>Measured local angles versus world-frame rays</h3>
    <p class="lede">Let \(\angle([x,y]^{\mathsf T})=\operatorname{atan2}(y,x)\), and let \(\operatorname{wrap}_{\pi}\) map an angle into \((-\pi,\pi]\). This page adopts the <em>direction from which the wave arrives</em> convention: the AoA ray points from the UE toward the previous interaction or source.</p>
    <div class="eq math-eq">
      \[
      \psi^{W}_{ts\ell}=\operatorname{wrap}_{\pi}(\theta_s+\psi^{B}_{ts\ell}),
      \qquad
      \varphi^{W}_{ts\ell}=\operatorname{wrap}_{\pi}(\theta_t+\varphi^{U}_{ts\ell}).
      \]
    </div>
    <div class="companion-math-grid">
      <article class="companion-math-card"><h4>Known headings</h4><p>When \(\theta_s\) and \(\theta_t\) are known, the local measurements can be drawn as world-frame rays. Sections 2 and 3 operate in this regime.</p></article>
      <article class="companion-math-card"><h4>Unknown UE heading</h4><p>When \(\theta_t\) is unknown, \(\varphi^U\) must stay in the UE frame. Every heading hypothesis rotates all UE-frame AoAs together. Section 4 therefore estimates heading jointly with position and map.</p></article>
    </div>
  </div>

  <div class="subsection-block" id="measurement-noise">
    <h3 class="subh"><span class="no">0.3</span>Measurement model, covariance, and residual weighting</h3>
    <p class="lede">A factor is a probabilistic constraint, not an exact equation. Its covariance determines how strongly the back end trusts delay relative to angles and gain.</p>
    <div class="eq math-eq">
      \[
      \mathbf z_i=\mathbf h_i(\boldsymbol\Theta_i)+\boldsymbol\varepsilon_i,
      \qquad
      \boldsymbol\varepsilon_i\sim\mathcal N(\mathbf 0,\boldsymbol\Sigma_i),
      \qquad
      \|\mathbf r_i\|^2_{\boldsymbol\Omega_i}
      =\mathbf r_i^{\mathsf T}\boldsymbol\Sigma_i^{-1}\mathbf r_i,
      \quad \boldsymbol\Omega_i=\boldsymbol\Sigma_i^{-1}.
      \]
    </div>
    <p class="eq-note">The covariance whitens mixed units. A metre of path-length error and a radian of angular error are not added directly; each residual component is scaled by its uncertainty and any cross-correlation in \(\boldsymbol\Sigma_i\).</p>
  </div>

  <div class="subsection-block" id="model-boundary">
    <h3 class="subh"><span class="no">0.4</span>What is assumed—and what is merely visualized</h3>
    <div class="assumption-strip">
      <article><span>Geometry</span><strong>Ideal specular baseline</strong>The VA equations are exact for flat specular reflectors. Rough or distributed scattering needs a different likelihood.</article>
      <article><span>Timing</span><strong>Calibrated demos</strong>The figures use \(L=c\tau\). In a real one-way system use \(L=c(\tau-\delta_{ts})\) or estimate the clock/hardware offset.</article>
      <article><span>Visibility</span><strong>Finite surfaces matter</strong>An unfolded algebraic solution is accepted only when every folded hit lies on the modeled surface and every leg is physically visible.</article>
      <article><span>Truth overlay</span><strong>Not an estimator input</strong>Faint walls, path colors, and reference UEs explain the synthetic construction; the unknown-map estimator does not receive them.</article>
    </div>
  </div>
</section>

`

insertBefore(
  '<!-- ============ 01 the measurement ============ -->',
  notationSection,
  'id="indices-and-states"',
  'measurement section marker'
)

replaceRequired(
  '<p class="lede">The channel estimator resolves a multipath component into geometric and radiometric observations. Read the delay and angles first, then use the power–delay profile and path-loss factors to understand which physical route produced them.</p>',
  '<p class="lede">The channel estimator produces noisy per-component parameters. Delay becomes geometric path length only after timing calibration; AoA and AoD remain local-frame measurements until the corresponding headings are applied; gain is radiometric evidence rather than a direct bounce counter.</p>',
  'measurement-section introduction'
)

const pathTupleBlock = String.raw`  <div class="subsection-block" id="path-tuple">
    <h3 class="subh"><span class="no">1.1</span>The resolved path tuple</h3>
    <p class="lede">For one resolvable MPC, a parametric channel estimator may return delay, two endpoint angles, and a complex gain. The index \(\ell\) only identifies a row in the current scan; temporal association is a separate inference problem.</p>
    <div class="eq math-eq">
      \[
      \mathbf z_{ts\ell}=
      \begin{bmatrix}
        \tau_{ts\ell} & \varphi^{U}_{ts\ell} & \psi^{B}_{ts\ell} & \alpha_{ts\ell}
      \end{bmatrix}^{\mathsf T},
      \qquad
      L_{ts\ell}=c\big(\tau_{ts\ell}-\delta_{ts}\big).
      \]
    </div>
    <div class="companion-math-grid">
      <article class="companion-math-card"><h4>Delay \(\tau\)</h4><p>One-way propagation time. \(c\) is the speed of light and \(\delta_{ts}\) collects clock and hardware delay. The simplified figures set \(\delta_{ts}=0\).</p></article>
      <article class="companion-math-card"><h4>AoA \(\varphi^U\)</h4><p>Arrival bearing in the UE array/body frame. This page points the ray from the UE back toward the previous interaction or source.</p></article>
      <article class="companion-math-card"><h4>AoD \(\psi^B\)</h4><p>Departure bearing in the transmitting BS array frame. A known BS heading converts it into a world-frame ray.</p></article>
      <article class="companion-math-card"><h4>Complex gain \(\alpha\)</h4><p>Amplitude and phase of the resolved component. \(|\alpha|^2\) is power-related; calibrated path loss additionally requires transmit-power, antenna, and receiver-chain normalization.</p></article>
    </div>
    <div class="accuracy"><strong>Do not collapse the tuple too early.</strong> Delay, angles, and gain have different units, noise levels, calibration requirements, and failure modes. The factor graph combines them through an explicit likelihood and covariance rather than an unweighted sum.</div>
  </div>

`
replaceBlock(
  '  <div class="subsection-block" id="path-tuple">',
  '  <div class="subsection-block" id="power-delay">',
  pathTupleBlock,
  'Do not collapse the tuple too early.',
  'path-tuple subsection'
)

replaceOptional(
  '    <h3 class="subh"><span class="no">1.2</span>The power–delay profile</h3>\n    <figure class="fig"',
  '    <h3 class="subh"><span class="no">1.2</span>The power–delay profile</h3>\n    <p class="lede">Finite bandwidth gives finite delay resolution. As a useful scale, \\(\\Delta\\tau\\sim 1/B\\) and \\(\\Delta L\\sim c/B\\), although waveform, windowing, SNR, array processing, and super-resolution estimation determine the practical limit. Two physical paths inside one resolution cell may appear as a single component.</p>\n    <figure class="fig"'
)
replaceOptional(
  'A random power-delay profile',
  'Illustrative power–delay profile. Treat peaks as separate MPCs only when the estimator can resolve them.'
)
replaceOptional(
  'Paths with the same travel distance span ~20&nbsp;dB of excess loss. What differs is what they hit—which surface, how many times, and whether the interaction is specular or diffuse.',
  'Each interaction often adds reflection or scattering loss, but bounce count does not impose a strict power ordering. Distance, material, incidence angle, antenna response, and blockage may dominate.'
)
replaceOptional(
  'Polarization exposes the difference: on matched reflected paths, half of all matched pairs differ by more than 3&nbsp;dB between polarizations.',
  'The reflection coefficient depends on material, carrier frequency, polarization, and incidence angle. A calibrated propagation model is required before gain becomes a quantitative map factor.'
)

replaceRequired(
  '<div class="accuracy"><strong>Scope of §2.</strong> The map, finite wall segments, BS pose, UE pose, and both array headings are known. For each MPC, test ordered wall sequences directly with the image-source method; no lower-order prefix path is required. These 2D checks assume synchronized delay and globally oriented AoA/AoD. In 3D, wall lines become planes and bearings live on S².</div>',
  '<div class="accuracy"><strong>Scope of §2.</strong> The map, finite wall segments, BS pose, UE pose, and both array headings are known. For each MPC, test ordered wall sequences directly with the image-source method; no lower-order prefix path is required. The demos use synchronized/calibrated path length, and convert UE-frame AoA plus BS-frame AoD into world-frame rays with the known headings. In 3D, wall lines become planes and angular measurements become unit directions or azimuth/elevation pairs.</div>',
  'Section 2 scope'
)
replaceOptional(
  '<div class="step"><b>04 · predict</b>Read the path\'s delay, global AoA, and global AoD.</div>',
  '<div class="step"><b>04 · predict</b>Predict calibrated delay, UE-frame AoA, and BS-frame AoD.</div>'
)

replaceRequired(
  '<div class="accuracy"><strong>Scope of §3.</strong> The BS and UE positions and both array headings are known, but the walls are not. Each 2D construction assumes synchronized delay and globally oriented AoA/AoD. Higher-order recovery additionally needs correctly associated nested prefix MPCs—for example, the one-bounce path associated with a two-bounce path. Wall lines and hollow anchors shown as reference geometry explain the synthetic scene; they are not estimator inputs. In 3D, delay ellipses become prolate ellipsoids and inferred wall lines become planes.</div>',
  '<div class="accuracy"><strong>Scope of §3.</strong> The BS and UE positions and both array headings are known, but the walls are not. Each 2D construction conditions on synchronized/calibrated timing and first converts local AoA/AoD to world-frame rays using the known headings. Higher-order recovery additionally needs correctly associated nested prefix MPCs—for example, the one-bounce path associated with a two-bounce path. Reference walls and hollow anchors explain the synthetic scene; they are not estimator inputs. In 3D, delay ellipses become prolate ellipsoids and inferred wall lines become planes.</div>',
  'Section 3 scope'
)
replaceOptional(
  '<div class="step"><b>01 · resolve</b>Take one measured path tuple (L, φ, ψ) in the global frame.</div>',
  '<div class="step"><b>01 · resolve</b>Take one tuple \(L,\varphi^U,\psi^B\) and rotate its local angles with the known headings.</div>'
)

replaceRequired(
  '<div class="accuracy"><strong>Scope of §4.</strong> The BS position and array heading are known, so AoD ψ is available in the global frame. The UE position, UE heading θ, and wall map are unknown. Delays are synchronized ranges; UE AoA φ<sub>body</sub> remains body-frame data, so each hypothesis uses φ<sub>global</sub> = φ<sub>body</sub> + θ. Higher-order cases require correctly associated prefix paths. Faint reference walls and UEs explain the synthetic scene only—they are never estimator inputs. The sliders show one candidate slice; the solution is the union of all feasible slices. Each tile opens on the completed construction; use <strong>Back</strong> to replay it step by step.</div>',
  '<div class="accuracy"><strong>Scope of §4.</strong> The BS position and array heading are known, so the measured BS-frame AoD can be rotated into a world-frame departure ray. The UE position, UE heading \(\theta_t\), and wall map are unknown. The demos condition on calibrated timing, while a real one-way model must retain \(\delta_{ts}\). UE AoA remains body-frame data, so each hypothesis uses \(\varphi^W=\operatorname{wrap}_{\pi}(\theta_t+\varphi^U)\). Higher-order cases require associated prefix paths. Faint reference walls and UEs are explanatory truth overlays, never estimator inputs; the displayed slider is one slice of the full feasible family.</div>',
  'Section 4 scope'
)
replaceOptional(
  '<div class="step"><b>01 · anchor</b>Start from the fixed BS pose and its global AoD ray.</div>',
  '<div class="step"><b>01 · anchor</b>Start from the fixed BS pose and rotate its measured BS-frame AoD into the world frame.</div>'
)

for (const [before, after] of [
  ['delay L = cτ', 'calibrated length L = c(τ−δ)'],
  ['AoA φ (at the UE, global)', 'world-frame AoA ray φᵂ (UE heading known)'],
  ['AoD ψ (at the BS, global)', 'world-frame AoD ray ψᵂ (BS heading known)'],
  ['global AoA φ (at the UE)', 'world-frame AoA ray φᵂ (UE heading known)'],
  ['global AoD ψ (at the BS)', 'world-frame AoD ray ψᵂ (BS heading known)']
]) {
  if (html.includes(before)) {
    html = html.split(before).join(after)
    changed = true
  }
}

const graphSlamSection = String.raw`
<!-- ============ 05 bistatic radio to GraphSLAM ============ -->
<section class="sec companion-section" id="bistatic-graphslam">
  <h2><span class="no">05</span>Bistatic radio SLAM as GraphSLAM</h2>
  <p class="lede">The transmitter being off the UE does not require a different SLAM backend. It changes the radio measurement function and, when transmitter calibration is uncertain, the number of variables touched by each factor.</p>
  <div class="accuracy"><strong>Core correction.</strong> GraphSLAM does not assume that every physical device is onboard. A factor graph records which <em>unknown variables</em> enter each prior or likelihood. A calibrated BS is therefore a fixed argument of a radio factor; an uncertain BS pose, clock, or array orientation is promoted to a variable with its own prior.</div>

  <nav class="subsection-tiles is-four" aria-label="Bistatic GraphSLAM subsections">
    <a href="#factor-language"><span>5.1</span><strong>Factor language</strong><small>known BS as a parameter</small></a>
    <a href="#bistatic-factor"><span>5.2</span><strong>MPC factor</strong><small>delay · AoA · AoD · gain</small></a>
    <a href="#va-factor"><span>5.3</span><strong>VA / wall factor</strong><small>unfold specular paths</small></a>
    <a href="#radio-graph-objective"><span>5.4</span><strong>Joint objective</strong><small>trajectory · map · association</small></a>
  </nav>

  <div class="subsection-block" id="factor-language">
    <h3 class="subh"><span class="no">5.1</span>The sensor placement changes \(h(\cdot)\), not the graph machinery</h3>
    <div class="companion-math-grid">
      <article class="companion-math-card"><h4>Typical onboard observation</h4><p>A sensor rigidly attached to the robot gives</p><p>\(\mathbf z_{tj}=\mathbf h(\mathbf T_t,\mathbf m_j)+\boldsymbol\varepsilon.\)</p><p>The likelihood touches the current pose and the observed landmark.</p></article>
      <article class="companion-math-card"><h4>Bistatic radio observation</h4><p>A path begins at BS \(s\), interacts with the map, and ends at UE pose \(t\):</p><p>\(\mathbf z_{ts\ell}=\mathbf h_q(\mathbf T_t,\mathbf m_j;\mathbf B_s,\boldsymbol\kappa)+\boldsymbol\varepsilon.\)</p><p>The semicolon separates optimized variables from known parameters.</p></article>
    </div>
    <div class="factor-bridge" role="img" aria-label="A known base station parameter and two unknown nodes connected by one radio factor">
      <div class="node known">\(\mathbf B_s\)<br>known BS</div>
      <div class="edge">→</div>
      <div class="factor">\(f^{\mathrm{rad}}_{ts\ell}\)<br>conditioned likelihood</div>
      <div class="edge">↔</div>
      <div style="display:grid;gap:8px"><div class="node">\(\mathbf T_t\)<br>UE pose</div><div class="node map">\(\mathbf m_j\)<br>radio-map entity</div></div>
    </div>
    <p class="factor-caption">Known values can feed a factor without appearing as optimized variable nodes. If \(\mathbf B_s\) or \(\boldsymbol\kappa\) is uncertain, draw an additional variable node and attach a prior.</p>
  </div>

  <div class="subsection-block" id="bistatic-factor">
    <h3 class="subh"><span class="no">5.2</span>One MPC becomes one sparse bistatic likelihood factor</h3>
    <p class="lede">A point-scatterer model makes the endpoint geometry explicit. Let \(\mathbf b_s\) be the known BS position, \(\mathbf p_t\) the UE position, and \(\mathbf s_j\) a persistent point interaction.</p>
    <div class="eq math-eq">
      \[
      \begin{aligned}
      \widehat L_{tsj}
        &=\|\mathbf s_j-\mathbf b_s\|+\|\mathbf p_t-\mathbf s_j\|,\\
      \widehat\tau_{tsj}
        &=\widehat L_{tsj}/c+\delta_{ts},\\
      \widehat\psi^{B}_{tsj}
        &=\operatorname{wrap}_{\pi}\!\left(\angle(\mathbf s_j-\mathbf b_s)-\theta_s\right),\\
      \widehat\varphi^{U}_{tsj}
        &=\operatorname{wrap}_{\pi}\!\left(\angle(\mathbf s_j-\mathbf p_t)-\theta_t\right).
      \end{aligned}
      \]
    </div>
    <p class="eq-note">For LoS, replace \(\mathbf s_j\) by the opposite endpoint: \(\widehat L=\|\mathbf p_t-\mathbf b_s\|\), AoD points BS→UE, and the adopted AoA convention points UE→BS.</p>
    <div class="eq math-eq">
      \[
      \mathbf r^{\mathrm{geom}}_{ts\ell}(j,q)=
      \begin{bmatrix}
        c(\tau_{ts\ell}-\delta_{ts})-\widehat L_q\\
        \operatorname{wrap}_{\pi}(\varphi^U_{ts\ell}-\widehat\varphi^U_q)\\
        \operatorname{wrap}_{\pi}(\psi^B_{ts\ell}-\widehat\psi^B_q)
      \end{bmatrix},
      \qquad
      f^{\mathrm{rad}}_{ts\ell}\propto
      \exp\!\left[-\tfrac12
      \|\mathbf r^{\mathrm{geom}}_{ts\ell}\|^2_{\boldsymbol\Sigma_{ts\ell}^{-1}}\right].
      \]
    </div>
    <div class="companion-math-grid">
      <article class="companion-math-card"><h4>Angular residual</h4><p>Subtracting angles without wrapping creates artificial errors at the \(-\pi/\pi\) boundary. In 3D, compare unit directions on the sphere or use azimuth/elevation with the appropriate manifold convention.</p></article>
      <article class="companion-math-card"><h4>Gain residual</h4><p>Only after calibration should gain enter as, for example, \(r_g=g^{\mathrm{dB}}-\gamma_q(\mathbf T_t,\mathcal M,\mathbf B_s;\boldsymbol\xi)\). Otherwise use gain as association or gating evidence rather than treating it as a deterministic bounce counter.</p></article>
    </div>
  </div>

  <div class="subsection-block" id="va-factor">
    <h3 class="subh"><span class="no">5.3</span>A specular wall becomes a landmark through the image-source construction</h3>
    <p class="lede">For one ideal reflecting line, parameterize the wall by a unit normal \(\mathbf n_j\) and signed offset \(d_j\). Reflecting the known BS across that wall produces a virtual anchor.</p>
    <div class="eq math-eq">
      \[
      \begin{aligned}
      \mathcal W_j&=\{\mathbf x:\mathbf n_j^{\mathsf T}\mathbf x=d_j\},
      \qquad \|\mathbf n_j\|=1,\\
      \mathcal R_j(\mathbf x)&=\mathbf x+2\big(d_j-\mathbf n_j^{\mathsf T}\mathbf x\big)\mathbf n_j,\\
      \mathbf v_{sj}&=\mathcal R_j(\mathbf b_s),
      \qquad
      \widehat L_{tsj}=\|\mathbf p_t-\mathbf v_{sj}\|,\\
      \lambda_{tsj}&=\frac{d_j-\mathbf n_j^{\mathsf T}\mathbf p_t}
      {\mathbf n_j^{\mathsf T}(\mathbf v_{sj}-\mathbf p_t)},
      \qquad
      \mathbf r_{tsj}=\mathbf p_t+\lambda_{tsj}(\mathbf v_{sj}-\mathbf p_t).
      \end{aligned}
      \]
    </div>
    <p class="eq-note">\(\mathbf v_{sj}\) is the VA and \(\mathbf r_{tsj}\) is the physical reflection point. A valid finite-wall path needs \(0&lt;\lambda&lt;1\), the hit inside the wall support, and every physical leg free of occlusion.</p>
    <div class="eq math-eq">
      \[
      \widehat\varphi^U_{tsj}=\operatorname{wrap}_{\pi}\!\left(\angle(\mathbf v_{sj}-\mathbf p_t)-\theta_t\right),
      \qquad
      \widehat\psi^B_{tsj}=\operatorname{wrap}_{\pi}\!\left(\angle(\mathbf r_{tsj}-\mathbf b_s)-\theta_s\right).
      \]
    </div>
    <h4 style="margin:22px 0 8px;font:700 11px var(--mono);letter-spacing:.08em;text-transform:uppercase">Ordered multi-bounce path</h4>
    <div class="eq math-eq">
      \[
      \mathbf v_s^{(0)}=\mathbf b_s,
      \qquad
      \mathbf v_s^{(k)}=\mathcal R_{j_k}\!\left(\mathbf v_s^{(k-1)}\right),
      \quad k=1,\ldots,K,
      \qquad
      \widehat L_q=\|\mathbf p_t-\mathbf v_s^{(K)}\|.
      \]
    </div>
    <p>The ordered wall sequence \(q=(j_1,\ldots,j_K)\) is part of the hypothesis. Build image sources in transmitter-to-receiver order, then fold the UE→final-VA line through the walls in reverse order. Reject the full chain when one folded point is off-surface, behind its ray origin, or occluded.</p>
    <table class="companion-symbols">
      <thead><tr><th>Map state</th><th>Best use</th><th>Important limitation</th></tr></thead>
      <tbody>
        <tr><td>Virtual anchor \(\mathbf v_{sj}\)</td><td>Compact one-BS, one-wall factor; behaves like a landmark for path length and AoA.</td><td>The same physical wall creates different VAs for different BSs.</td></tr>
        <tr><td>Wall / plane \((\mathbf n_j,d_j)\)</td><td>Shares one physical surface across poses, paths, and BSs.</td><td>Needs finite-support and visibility logic; the normal sign has a parameterization symmetry.</td></tr>
        <tr><td>Point scatterer \(\mathbf s_j\)</td><td>Diffuse or point-like interaction model.</td><td>Does not enforce equal-angle specular reflection.</td></tr>
        <tr><td>Composite VA / chain</td><td>Compact representation of a known ordered multi-bounce transform.</td><td>May identify the composite transform without uniquely identifying every individual wall.</td></tr>
      </tbody>
    </table>
  </div>

  <div class="subsection-block" id="radio-graph-objective">
    <h3 class="subh"><span class="no">5.4</span>Joint trajectory–map posterior and GraphSLAM objective</h3>
    <p class="lede">Collect continuous unknowns in \(\boldsymbol\Theta\). Each entry of \(A\) selects a complete path hypothesis: clutter, LoS, or an ordered wall chain. Known BS states are conditioned on; uncertain BS and calibration states are included in \(\boldsymbol\Theta\).</p>
    <div class="eq math-eq">
      \[
      \boldsymbol\Theta=\{\mathbf T_{1:T},\mathcal M,\boldsymbol\delta,\boldsymbol\xi,\mathbf B_{\mathrm{uncertain}}\},
      \]
      \[
      a_{ts\ell}\in\mathcal H=\{0,\mathrm{LoS}\}\cup
      \{(k,j_1,\ldots,j_k):k\ge1,\text{ ordered wall chain}\}.
      \]
    </div>
    <p class="eq-note">Here \(0\) denotes clutter, not a map entity. LoS has no reflector; a reflected path selects every wall in its ordered chain. Bounce count is determined by the hypothesis, not an additional independent association variable. Finite-support and visibility tests determine admissible candidates.</p>
    <div class="eq math-eq">
      \[
      p(\boldsymbol\Theta,A\mid Z,U,\mathbf B_{\mathrm{known}})
      \propto p(\boldsymbol\Theta)
      \prod_{t=2}^{T}f_t^{\mathrm{rel}}
      \prod_{t,s}\mathcal L_{\mathrm{set}}(\mathcal Z_{ts},A_{ts}\mid\boldsymbol\Theta,\mathbf B_s).
      \]
    </div>
    <p class="eq-note">\(U\) is optional motion information, and \(f_t^{\mathrm{rel}}\) is the corresponding relative-pose factor. The joint set likelihood \(\mathcal L_{\mathrm{set}}\) includes association weights, clutter, missed detections, count terms, and the admissibility constraints of the declared observation model. A product of independent per-MPC mixture factors is not automatically this set likelihood. One-to-one constraints, when assumed, apply to a physical path hypothesis within a scan, not to every use of the same wall.</p>
    <div class="eq math-eq">
      \[
      \begin{aligned}
      \boldsymbol\Theta^*=\arg\min_{\boldsymbol\Theta}\;&
      \|\mathbf r^{\mathrm{prior}}\|^2_{\boldsymbol\Omega_0}
      +\sum_{t=2}^{T}\|\mathbf r_t^{\mathrm{rel}}\|^2_{\boldsymbol\Omega_t}\\
      &+\sum_{(t,s,\ell):a_{ts\ell}\ne0}
      \rho\!\left(\|\mathbf r^{\mathrm{rad}}_{ts\ell}(\mathbf x_t,\mathcal M,a_{ts\ell})\|^2_{\boldsymbol\Omega_{ts\ell}}\right).
      \end{aligned}
      \]
    </div>
    <p class="eq-note">This is the fixed-association geometric back-end objective; it includes LoS. Ordinary Gaussian least squares uses \(\rho(s)=s\), fixed covariances, and state-independent omitted likelihood terms. A robust loss is a modeling choice. Retain state-dependent log-determinants, detection/count, association, and visibility terms when claiming the full MAP objective.</p>
    <div class="companion-steps">
      <article><span>Front end</span><strong>Propose the discrete explanation</strong>Resolve MPCs, normalize angle conventions, generate associations and bounce-order candidates, test visibility, initialize new variables, and attach covariances.</article>
      <article><span>Factor graph</span><strong>State the probabilistic problem</strong>Pose, map, clock, and calibration nodes are connected only to the priors and likelihoods in which they participate. Reobserving one map entity couples distant UE poses.</article>
      <article><span>Back end</span><strong>Optimize or marginalize</strong>With fixed complete path assignments \(A\), solve the conditioned problem. With uncertain \(A\), alternate, marginalize, or branch; mixture/max-mixture and robust factors are practical approximations. \(a=0\) uses a clutter likelihood, not \(\mathbf m_0\).</article>
    </div>

    <h4 style="margin:22px 0 8px;font:700 11px var(--mono);letter-spacing:.08em;text-transform:uppercase">Bundle adjustment, GraphSLAM, and iSAM2</h4>
    <div class="companion-math-grid">
      <article class="companion-math-card"><h4>Measurement-only core</h4><p>Pose–map radio factors alone form a bundle-adjustment-like problem. No governing motion law is required: two poses become coupled when they observe the same persistent map entity.</p></article>
      <article class="companion-math-card"><h4>Additional GraphSLAM factors</h4><p>Odometry, IMU, smoothness, loop/revisit, clock, and calibration factors may be added when available. An odometry factor is a noisy relative-pose likelihood; it is not automatically a physical dynamics model.</p></article>
      <article class="companion-math-card"><h4>Batch solution</h4><p>Gauss–Newton or Levenberg–Marquardt repeatedly linearizes the sparse residuals. Eliminating independent map blocks gives the usual Schur-complement pose system.</p></article>
      <article class="companion-math-card"><h4>Incremental solution</h4><p>iSAM2 solves the same posterior incrementally by updating the affected part of a Bayes tree. It does not decide MPC association or invent a radio measurement model.</p></article>
    </div>

    <div class="accuracy"><strong>Gauge versus observability.</strong> A calibrated BS pose can remove the ordinary free global frame, but it cannot remove a genuine null space caused by corridor geometry, an unidentified reflector decomposition, an unknown clock, insufficient heading information, or poor path diversity. Inspect Jacobian rank or posterior covariance instead of forcing a unique answer.</div>

    <h4 style="margin:24px 0 8px;font:700 11px var(--mono);letter-spacing:.08em;text-transform:uppercase">Implementation checklist</h4>
    <div class="assumption-strip">
      <article><span>01</span><strong>Time reference</strong>State whether delay is ToA, excess delay, TDoA, or calibrated path length; include clock and hardware variables when needed.</article>
      <article><span>02</span><strong>Angle convention</strong>Name the BS and UE frames and whether AoA points toward the source or along the arriving wave vector.</article>
      <article><span>03</span><strong>Map primitive</strong>Choose point scatterers, VAs, physical walls, or ordered chains according to the propagation physics and sharing required.</article>
      <article><span>04</span><strong>Association</strong>Separate MPC row index from persistent identity; model clutter, missed detections, births, and alternative bounce orders.</article>
      <article><span>05</span><strong>Uncertainty</strong>Use calibrated covariances, wrap angular residuals, and model correlations when the channel estimator couples delay and angle errors.</article>
      <article><span>06</span><strong>Visibility</strong>Enforce finite support, positive ordered legs, occlusion, and path availability rather than accepting every image-source intersection.</article>
      <article><span>07</span><strong>Radiometry</strong>Use path gain quantitatively only with a propagation and calibration model; otherwise keep it as soft association evidence.</article>
      <article><span>08</span><strong>Rank</strong>Anchor the intended gauge and report geometry-induced weak or null directions explicitly.</article>
    </div>

    <h4 style="margin:24px 0 8px;font:700 11px var(--mono);letter-spacing:.08em;text-transform:uppercase">References used for this companion</h4>
    <ul class="reference-list">
      <li><strong>SLAM Handbook, Chapters I and 1.</strong> Front-end/back-end terminology, factor graphs, MAP-to-nonlinear-least-squares, covariance whitening, bundle adjustment versus landmark SLAM, sparsity, and incremental inference.</li>
      <li><a href="https://www.cs.cmu.edu/~kaess/pub/Dellaert17fnt.html"><strong>Dellaert and Kaess, Factor Graphs for Robot Perception</strong></a> — sensor-agnostic factor-graph modeling and sparse nonlinear inference.</li>
      <li><a href="https://arxiv.org/abs/1801.04463"><strong>Leitinger et al., A Belief Propagation Algorithm for Multipath-Based SLAM</strong></a> — physical and virtual anchors, MPC association, and joint agent–map estimation.</li>
      <li><a href="https://arxiv.org/abs/2211.09241"><strong>Leitinger et al., Data Fusion for Multipath-Based SLAM</strong></a> — sharing physical surfaces across paths and base stations through master virtual anchors.</li>
      <li><a href="https://arxiv.org/abs/2304.05680"><strong>Wielandner et al., Multipath-based SLAM for Non-Ideal Reflective Surfaces</strong></a> — multiple measurements per surface and dispersion beyond an ideal one-MPC-per-VA model.</li>
      <li><a href="https://doi.org/10.1177/0278364911430419"><strong>Kaess et al., iSAM2</strong></a> — incremental solution of nonlinear factor graphs using the Bayes tree.</li>
    </ul>
  </div>
</section>

`

insertBefore(
  '<footer>\n  Interactive note by Bai Liping',
  graphSlamSection,
  'id="factor-language"',
  'footer insertion point'
)

replaceOptional(
  'Interactive note by Bai Liping · scale 1 px ≙ 0.1 m, c ≈ 0.3 m/ns ·\n  virtual-anchor background:',
  'Interactive note by Bai Liping · detailed companion to the radio-SLAM slides · scale 1 px ≙ 0.1 m, c ≈ 0.3 m/ns ·\n  virtual-anchor background:'
)

for (const marker of [
  'href="companion.css"',
  'id="radio-companion-mathjax-config"',
  'id="notation"',
  'id="indices-and-states"',
  'id="angle-frames"',
  'id="measurement-noise"',
  'id="bistatic-graphslam"',
  'id="factor-language"',
  'id="bistatic-factor"',
  'id="va-factor"',
  'id="radio-graph-objective"',
  'GraphSLAM does not assume that every physical device is onboard',
  'L_{ts\\ell}=c\\big(\\tau_{ts\\ell}-\\delta_{ts}\\big)',
  '\\(a=0\\) uses a clutter likelihood',
  'Only after calibration should gain enter'
]) {
  if (!html.includes(marker)) throw new Error(`Companion-page validation failed: ${marker}`)
}

for (const legacy of [
  'a <strong>delay</strong> (a path length)',
  'These 2D checks assume synchronized delay and globally oriented AoA/AoD.',
  'Each 2D construction assumes synchronized delay and globally oriented AoA/AoD.',
  'AoD ψ is available in the global frame.',
  'Paths with the same travel distance span ~20&nbsp;dB',
  'half of all matched pairs differ by more than 3&nbsp;dB'
]) {
  if (html.includes(legacy)) throw new Error(`Legacy or overstrong wording remains: ${legacy}`)
}

if (html.includes('\0')) throw new Error('The generated HTML contains a null byte')

if (changed) {
  writeFileSync(pagePath, html)
  console.log('Expanded the radio-SLAM companion page and corrected measurement/frame wording.')
} else {
  console.log('No radio-SLAM companion-page changes needed.')
}
