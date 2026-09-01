(() => {
  'use strict'

  const COLORS = {
    bg: '#0b1220', panel: '#111a2b', inner: '#162238', ink: '#f8fafc', soft: '#b9c4d6', faint: '#7f8da3', line: '#26354d',
    cyan: '#22d3ee', blue: '#5b8cff', green: '#58d68d', amber: '#f5c542', red: '#f97373', violet: '#a78bfa'
  }

  const params = new URLSearchParams(window.location.search)
  const demo = params.get('demo') || 'scalar'
  const app = document.getElementById('app')
  const model = window.KalmanModel

  function fmt(value, digits = 3) {
    if (!Number.isFinite(value)) return '—'
    const abs = Math.abs(value)
    if (abs !== 0 && (abs >= 1e4 || abs < 1e-3)) return value.toExponential(Math.max(1, digits - 1))
    return value.toFixed(digits).replace(/\.?0+$/, '')
  }

  function escapeHtml(value) {
    return String(value).replace(/[&<>"]/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[char])
  }

  function mathHtml(source, display = false) {
    const escaped = escapeHtml(source)
    const open = display ? '\\[' : '\\('
    const close = display ? '\\]' : '\\)'
    return `<span class="math-tex ${display ? 'math-display' : 'math-inline'}">${open}${escaped}${close}</span>`
  }

  function tex(strings, ...values) {
    return mathHtml(String.raw(strings, ...values))
  }

  function setMath(node, source, display = false) {
    node.innerHTML = mathHtml(source, display)
    window.typesetDynamicMath?.()
  }

  function createShell({ accent, eyebrow, title, intro, controls, stageTitle, stageBody }) {
    app.style.setProperty('--accent', accent)
    app.innerHTML = `
      <aside class="control-panel">
        <p class="eyebrow">${escapeHtml(eyebrow)}</p>
        <h1>${escapeHtml(title)}</h1>
        <p class="intro">${intro}</p>
        <div class="control-stack">${controls}</div>
      </aside>
      <section class="stage-panel">
        <div class="stage-bar">
          <strong>${escapeHtml(stageTitle)}</strong>
          <p class="status" id="status"><i></i><span>exact agreement</span></p>
        </div>
        <div class="stage-body">${stageBody}</div>
      </section>`
    return {
      controls: app.querySelector('.control-stack'),
      stage: app.querySelector('.stage-body'),
      status: app.querySelector('#status')
    }
  }

  function rangeControl(id, label, min, max, step, value, output) {
    return `
      <div class="control">
        <div class="control-head"><label for="${id}">${label}</label><output id="${id}-out" for="${id}">${escapeHtml(output)}</output></div>
        <input id="${id}" type="range" min="${min}" max="${max}" step="${step}" value="${value}">
      </div>`
  }

  function setStatus(node, text, tone = 'good', isMath = false) {
    const label = node.querySelector('span')
    if (isMath) setMath(label, text)
    else label.textContent = text
    const dot = node.querySelector('i')
    dot.style.background = tone === 'bad' ? COLORS.red : tone === 'warn' ? COLORS.amber : COLORS.green
  }

  function fitCanvas(canvas) {
    const rect = canvas.getBoundingClientRect()
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    const width = Math.max(1, Math.round(rect.width))
    const height = Math.max(1, Math.round(rect.height))
    if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
      canvas.width = Math.round(width * dpr)
      canvas.height = Math.round(height * dpr)
    }
    const ctx = canvas.getContext('2d')
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    return { ctx, width, height }
  }

  function watchCanvas(canvas, draw) {
    let raf = 0
    const request = () => {
      cancelAnimationFrame(raf)
      raf = requestAnimationFrame(draw)
    }
    const observer = new ResizeObserver(request)
    observer.observe(canvas)
    window.addEventListener('resize', request)
    request()
    return request
  }

  function mountScalar() {
    const shell = createShell({
      accent: COLORS.cyan,
      eyebrow: 'SCALAR FUSION · SHARED POSTERIOR',
      title: 'Prior × likelihood',
      intro: 'Change the two Gaussian sources. Four equivalent formulas agree numerically; this is a consistency check, not four independent derivations.',
      controls: [
        rangeControl('prior-mean', `Prior mean ${tex`m^-`}`, -4, 4, .1, -1.2, '−1.2'),
        rangeControl('prior-sigma', `Prior standard deviation ${tex`\sigma_p`}`, .2, 3, .05, 1.35, '1.35'),
        rangeControl('measurement', `Measurement ${tex`z`}`, -4, 4, .1, 2.1, '2.1'),
        rangeControl('measurement-sigma', `Measurement standard deviation ${tex`\sigma_r`}`, .2, 3, .05, .75, '0.75'),
        '<div class="button-row"><button id="scalar-reset" type="button">Reset</button><button id="scalar-swap" class="primary" type="button">Swap certainty</button></div>'
      ].join(''),
      stageTitle: 'POSTERIOR DENSITY AND NUMERICAL AGREEMENT',
      stageBody: `
        <div class="canvas-layout">
          <div class="canvas-card">
            <canvas id="scalar-canvas" aria-label="Prior, likelihood, and posterior Gaussian density curves"></canvas>
            <div class="legend" aria-hidden="true">
              <span style="color:${COLORS.cyan}"><i></i>prior</span>
              <span style="color:${COLORS.amber}"><i></i>likelihood</span>
              <span style="color:${COLORS.green}"><i></i>posterior</span>
            </div>
          </div>
          <aside class="metric-rail">
            <div class="metric-card accent"><span class="metric-k">Kalman gain</span><div class="metric-v" id="scalar-gain">—</div><p class="metric-copy">${tex`K=P^-/(P^-+R)`}</p></div>
            <div class="metric-card"><span class="metric-k">Posterior</span><div class="metric-v" id="scalar-post">—</div></div>
            <div class="metric-card"><span class="metric-k">Derivation check</span><div class="metric-v good">Bayes · WLS<br>information · conditioning</div><p class="metric-copy">One posterior, four routes.</p></div>
            <div class="metric-card"><span class="metric-k">Maximum disagreement</span><div class="metric-v good" id="scalar-delta">0</div></div>
          </aside>
        </div>`
    })

    const canvas = document.getElementById('scalar-canvas')
    const controls = {
      priorMean: document.getElementById('prior-mean'),
      priorSigma: document.getElementById('prior-sigma'),
      measurement: document.getElementById('measurement'),
      measurementSigma: document.getElementById('measurement-sigma')
    }
    const outputs = Object.fromEntries(Object.entries(controls).map(([key, node]) => [key, document.getElementById(`${node.id}-out`)]))
    let state = null

    function update() {
      const m = Number(controls.priorMean.value)
      const sp = Number(controls.priorSigma.value)
      const z = Number(controls.measurement.value)
      const sr = Number(controls.measurementSigma.value)
      state = model.scalarPosterior({ priorMean: m, priorSigma: sp, measurement: z, measurementSigma: sr })
      const { K, postMean, postVar, delta } = state
      outputs.priorMean.textContent = fmt(m, 2)
      outputs.priorSigma.textContent = fmt(sp, 2)
      outputs.measurement.textContent = fmt(z, 2)
      outputs.measurementSigma.textContent = fmt(sr, 2)
      document.getElementById('scalar-gain').textContent = fmt(K, 5)
      setMath(document.getElementById('scalar-post'), String.raw`\begin{aligned}m^+&=${fmt(postMean, 4)}\\\sigma^+&=${fmt(Math.sqrt(postVar), 4)}\end{aligned}`, true)
      document.getElementById('scalar-delta').textContent = delta === 0 ? '0 (same arithmetic)' : delta.toExponential(2)
      setStatus(shell.status, 'derivations agree', 'good')
      draw()
    }

    function gaussian(x, mean, sigma) {
      const q = (x - mean) / sigma
      return Math.exp(-.5 * q * q) / (sigma * Math.sqrt(2 * Math.PI))
    }

    function draw() {
      if (!state) return
      const { ctx, width, height } = fitCanvas(canvas)
      ctx.clearRect(0, 0, width, height)
      ctx.fillStyle = '#0d1626'
      ctx.fillRect(0, 0, width, height)
      const pad = { l: 44, r: 18, t: 18, b: 38 }
      const xmin = -6, xmax = 6
      const plotW = Math.max(1, width - pad.l - pad.r)
      const plotH = Math.max(1, height - pad.t - pad.b)
      const maxY = Math.max(gaussian(state.m, state.m, state.sp), gaussian(state.z, state.z, state.sr), gaussian(state.postMean, state.postMean, Math.sqrt(state.postVar))) * 1.18
      const X = x => pad.l + (x - xmin) / (xmax - xmin) * plotW
      const Y = y => pad.t + plotH - y / maxY * plotH

      ctx.strokeStyle = COLORS.line
      ctx.lineWidth = 1
      ctx.font = '9px Menlo, Consolas, monospace'
      ctx.fillStyle = COLORS.faint
      ctx.textAlign = 'center'
      for (let x = -6; x <= 6; x += 1) {
        ctx.beginPath(); ctx.moveTo(X(x), pad.t); ctx.lineTo(X(x), pad.t + plotH); ctx.stroke()
        ctx.fillText(String(x), X(x), height - 13)
      }
      ctx.beginPath(); ctx.moveTo(pad.l, pad.t + plotH); ctx.lineTo(width - pad.r, pad.t + plotH); ctx.strokeStyle = COLORS.soft; ctx.stroke()

      const curves = [
        { mean: state.m, sigma: state.sp, color: COLORS.cyan, label: 'm⁻' },
        { mean: state.z, sigma: state.sr, color: COLORS.amber, label: 'z' },
        { mean: state.postMean, sigma: Math.sqrt(state.postVar), color: COLORS.green, label: 'm⁺' }
      ]
      for (const curve of curves) {
        ctx.beginPath()
        for (let px = 0; px <= plotW; px += 2) {
          const x = xmin + px / plotW * (xmax - xmin)
          const y = gaussian(x, curve.mean, curve.sigma)
          if (px === 0) ctx.moveTo(X(x), Y(y)); else ctx.lineTo(X(x), Y(y))
        }
        ctx.strokeStyle = curve.color
        ctx.lineWidth = curve.label === 'm⁺' ? 3 : 2
        ctx.stroke()
        ctx.setLineDash([4, 4])
        ctx.beginPath(); ctx.moveTo(X(curve.mean), pad.t + 8); ctx.lineTo(X(curve.mean), pad.t + plotH); ctx.stroke()
        ctx.setLineDash([])
        ctx.fillStyle = curve.color
        ctx.font = '700 10px Menlo, Consolas, monospace'
        ctx.fillText(curve.label, X(curve.mean), pad.t + 10)
      }

      const innovationY = pad.t + plotH - 18
      ctx.strokeStyle = COLORS.green
      ctx.lineWidth = 2
      ctx.beginPath(); ctx.moveTo(X(state.m), innovationY); ctx.lineTo(X(state.postMean), innovationY); ctx.stroke()
      ctx.fillStyle = COLORS.green
      const direction = Math.sign(state.postMean - state.m) || 1
      ctx.beginPath(); ctx.moveTo(X(state.postMean), innovationY); ctx.lineTo(X(state.postMean) - 7 * direction, innovationY - 4); ctx.lineTo(X(state.postMean) - 7 * direction, innovationY + 4); ctx.closePath(); ctx.fill()
      ctx.fillStyle = COLORS.soft
      ctx.font = '9px Menlo, Consolas, monospace'
      ctx.fillText(`K(z−m⁻) = ${fmt(state.K * (state.z - state.m), 3)}`, (X(state.m) + X(state.postMean)) / 2, innovationY - 8)
    }

    Object.values(controls).forEach(node => node.addEventListener('input', update))
    document.getElementById('scalar-reset').addEventListener('click', () => {
      controls.priorMean.value = -1.2; controls.priorSigma.value = 1.35; controls.measurement.value = 2.1; controls.measurementSigma.value = .75; update()
    })
    document.getElementById('scalar-swap').addEventListener('click', () => {
      const a = controls.priorSigma.value
      controls.priorSigma.value = controls.measurementSigma.value
      controls.measurementSigma.value = a
      update()
    })
    watchCanvas(canvas, draw)
    update()
  }

  function mountGeometry() {
    const shell = createShell({
      accent: COLORS.green,
      eyebrow: '2D COVARIANCE GEOMETRY',
      title: 'One scalar slice of a 2D prior',
      intro: `Rotate ${tex`H`}, change ${tex`R`}, and move the measured hyperplane. The Kalman gain follows the state–innovation cross-covariance.`,
      controls: [
        rangeControl('geo-sx', `Prior ${tex`\sigma_x`}`, .35, 2.8, .05, 1.8, '1.8'),
        rangeControl('geo-sy', `Prior ${tex`\sigma_y`}`, .35, 2.8, .05, 1.0, '1.0'),
        rangeControl('geo-rho', `Prior correlation ${tex`\rho`}`, -.9, .9, .05, .65, '0.65'),
        rangeControl('geo-angle', `Measurement angle ${tex`\varphi`}`, 0, 180, 1, 28, '28°'),
        rangeControl('geo-z', `Measured value ${tex`z`}`, -3.5, 3.5, .1, 1.7, '1.7'),
        rangeControl('geo-sigma', `Measurement ${tex`\sigma_r`}`, .15, 2.2, .05, .45, '0.45'),
        '<div class="button-row"><button id="geo-reset" type="button">Reset</button><button id="geo-orthogonal" class="primary" type="button">Rotate 90°</button></div>'
      ].join(''),
      stageTitle: 'PRIOR ELLIPSE → MEASUREMENT STRIP → POSTERIOR ELLIPSE',
      stageBody: `
        <div class="canvas-layout">
          <div class="canvas-card">
            <canvas id="geo-canvas" aria-label="Prior and posterior covariance ellipses with a linear measurement strip"></canvas>
            <div class="legend" aria-hidden="true">
              <span style="color:${COLORS.cyan}"><i></i>prior ${tex`2\sigma`}</span>
              <span style="color:${COLORS.amber}"><i></i>measurement</span>
              <span style="color:${COLORS.green}"><i></i>posterior ${tex`2\sigma`}</span>
            </div>
          </div>
          <aside class="metric-rail">
            <div class="metric-card accent"><span class="metric-k">Gain vector</span><div class="metric-v" id="geo-k">—</div><p class="metric-copy">${tex`K=P^-H^\mathsf{T}/S`}</p></div>
            <div class="metric-card"><span class="metric-k">Innovation</span><div class="metric-v" id="geo-innovation">—</div></div>
            <div class="metric-card"><span class="metric-k">Posterior mean</span><div class="metric-v" id="geo-mean">—</div></div>
            <div class="metric-card"><span class="metric-k">Uncertainty area</span><div class="metric-v good" id="geo-area">—</div><p class="metric-copy">${tex`\sqrt{\det P^+}/\sqrt{\det P^-}`}</p></div>
            <div class="metric-card"><span class="metric-k">Interpretation</span><p class="metric-copy" id="geo-copy">—</p></div>
          </aside>
        </div>`
    })

    const canvas = document.getElementById('geo-canvas')
    const ids = ['geo-sx', 'geo-sy', 'geo-rho', 'geo-angle', 'geo-z', 'geo-sigma']
    const controls = Object.fromEntries(ids.map(id => [id, document.getElementById(id)]))
    let state = null

    function update() {
      const sx = Number(controls['geo-sx'].value)
      const sy = Number(controls['geo-sy'].value)
      const rho = Number(controls['geo-rho'].value)
      const angleDeg = Number(controls['geo-angle'].value)
      const z = Number(controls['geo-z'].value)
      const sr = Number(controls['geo-sigma'].value)
      state = model.covarianceGeometry({ sx, sy, rho, angleDeg, z, measurementSigma: sr })
      const { K, innovation, S, mp, areaRatio, gainAngleSine } = state
      document.getElementById('geo-sx-out').textContent = fmt(sx, 2)
      document.getElementById('geo-sy-out').textContent = fmt(sy, 2)
      document.getElementById('geo-rho-out').textContent = fmt(rho, 2)
      document.getElementById('geo-angle-out').textContent = `${Math.round(angleDeg)}°`
      document.getElementById('geo-z-out').textContent = fmt(z, 2)
      document.getElementById('geo-sigma-out').textContent = fmt(sr, 2)
      setMath(document.getElementById('geo-k'), String.raw`\begin{bmatrix}${fmt(K[0], 4)}&${fmt(K[1], 4)}\end{bmatrix}^{\mathsf{T}}`, true)
      setMath(document.getElementById('geo-innovation'), String.raw`\begin{aligned}\nu&=${fmt(innovation, 4)}\\S&=${fmt(S, 4)}\end{aligned}`, true)
      setMath(document.getElementById('geo-mean'), String.raw`\begin{bmatrix}${fmt(mp[0], 3)}&${fmt(mp[1], 3)}\end{bmatrix}^{\mathsf{T}}`, true)
      document.getElementById('geo-area').textContent = `${fmt(100 * areaRatio, 2)}% remains`
      document.getElementById('geo-copy').textContent = gainAngleSine > .15
        ? 'Prior anisotropy or correlation rotates the gain away from the measurement normal, so one measurement updates coupled state components.'
        : 'For this measurement direction, the gain is nearly parallel to the measurement normal. Alignment alone does not imply weak prior correlation.'
      setStatus(shell.status, 'posterior remains positive definite', 'good')
      draw()
    }

    function draw() {
      if (!state) return
      const { ctx, width, height } = fitCanvas(canvas)
      ctx.clearRect(0, 0, width, height)
      ctx.fillStyle = '#0d1626'; ctx.fillRect(0, 0, width, height)
      const pad = 26
      const scale = Math.min((width - 2 * pad) / 10, (height - 2 * pad) / 8)
      const origin = [width / 2, height / 2]
      const toPx = p => [origin[0] + p[0] * scale, origin[1] - p[1] * scale]

      ctx.strokeStyle = COLORS.line; ctx.lineWidth = 1; ctx.font = '8px Menlo, Consolas, monospace'; ctx.fillStyle = COLORS.faint
      for (let x = -5; x <= 5; x += 1) {
        const px = toPx([x, 0])[0]
        ctx.beginPath(); ctx.moveTo(px, pad); ctx.lineTo(px, height - pad); ctx.stroke()
      }
      for (let y = -4; y <= 4; y += 1) {
        const py = toPx([0, y])[1]
        ctx.beginPath(); ctx.moveTo(pad, py); ctx.lineTo(width - pad, py); ctx.stroke()
      }
      ctx.strokeStyle = COLORS.faint
      ctx.beginPath(); ctx.moveTo(pad, origin[1]); ctx.lineTo(width - pad, origin[1]); ctx.stroke()
      ctx.beginPath(); ctx.moveTo(origin[0], pad); ctx.lineTo(origin[0], height - pad); ctx.stroke()

      const h = state.h
      const d = [-h[1], h[0]]
      const center = [h[0] * state.z, h[1] * state.z]
      const halfLength = 9
      const halfBand = state.sr
      const corners = [
        [center[0] + d[0] * halfLength + h[0] * halfBand, center[1] + d[1] * halfLength + h[1] * halfBand],
        [center[0] - d[0] * halfLength + h[0] * halfBand, center[1] - d[1] * halfLength + h[1] * halfBand],
        [center[0] - d[0] * halfLength - h[0] * halfBand, center[1] - d[1] * halfLength - h[1] * halfBand],
        [center[0] + d[0] * halfLength - h[0] * halfBand, center[1] + d[1] * halfLength - h[1] * halfBand]
      ].map(toPx)
      ctx.fillStyle = 'rgba(245,197,66,.10)'
      ctx.beginPath(); corners.forEach((p, i) => i ? ctx.lineTo(...p) : ctx.moveTo(...p)); ctx.closePath(); ctx.fill()
      const lineA = toPx([center[0] + d[0] * halfLength, center[1] + d[1] * halfLength])
      const lineB = toPx([center[0] - d[0] * halfLength, center[1] - d[1] * halfLength])
      ctx.strokeStyle = COLORS.amber; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(...lineA); ctx.lineTo(...lineB); ctx.stroke()

      function ellipse(mean, P, color, fill, widthLine) {
        const eig = model.eigen2(P)
        const c = toPx(mean)
        ctx.save()
        ctx.translate(c[0], c[1])
        ctx.rotate(-eig.angle)
        ctx.scale(2 * Math.sqrt(eig.l1) * scale, 2 * Math.sqrt(eig.l2) * scale)
        ctx.beginPath(); ctx.arc(0, 0, 1, 0, Math.PI * 2)
        ctx.restore()
        ctx.fillStyle = fill; ctx.fill()
        ctx.strokeStyle = color; ctx.lineWidth = widthLine; ctx.stroke()
      }
      ellipse(state.m, state.P, COLORS.cyan, 'rgba(34,211,238,.08)', 2)
      ellipse(state.mp, state.Pp, COLORS.green, 'rgba(88,214,141,.12)', 3)

      const from = toPx(state.m), to = toPx(state.mp)
      ctx.strokeStyle = COLORS.green; ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(...from); ctx.lineTo(...to); ctx.stroke()
      const angle = Math.atan2(to[1] - from[1], to[0] - from[0])
      ctx.fillStyle = COLORS.green; ctx.beginPath(); ctx.moveTo(...to); ctx.lineTo(to[0] - 9 * Math.cos(angle - .45), to[1] - 9 * Math.sin(angle - .45)); ctx.lineTo(to[0] - 9 * Math.cos(angle + .45), to[1] - 9 * Math.sin(angle + .45)); ctx.closePath(); ctx.fill()

      for (const [p, color, label] of [[state.m, COLORS.cyan, 'm⁻'], [state.mp, COLORS.green, 'm⁺']]) {
        const q = toPx(p)
        ctx.fillStyle = color; ctx.beginPath(); ctx.arc(q[0], q[1], 4.5, 0, Math.PI * 2); ctx.fill()
        ctx.font = '700 10px Menlo, Consolas, monospace'; ctx.fillText(label, q[0] + 8, q[1] - 8)
      }
      ctx.fillStyle = COLORS.amber; ctx.font = '700 9px Menlo, Consolas, monospace'
      ctx.fillText(`H x = z,  φ=${Math.round(state.angleDeg)}°`, lineA[0] + 8, Math.max(18, lineA[1] + 14))
    }

    ids.forEach(id => controls[id].addEventListener('input', update))
    document.getElementById('geo-reset').addEventListener('click', () => {
      const values = { 'geo-sx': 1.8, 'geo-sy': 1, 'geo-rho': .65, 'geo-angle': 28, 'geo-z': 1.7, 'geo-sigma': .45 }
      for (const [id, value] of Object.entries(values)) controls[id].value = value
      update()
    })
    document.getElementById('geo-orthogonal').addEventListener('click', () => {
      let angle = Number(controls['geo-angle'].value) + 90
      if (angle > 180) {
        angle -= 180
        controls['geo-z'].value = -Number(controls['geo-z'].value)
      }
      controls['geo-angle'].value = angle
      update()
    })
    watchCanvas(canvas, draw)
    update()
  }

  // ---------- Small matrix library used by the equivalence experiment ----------
  function zeros(rows, cols) { return Array.from({ length: rows }, () => Array(cols).fill(0)) }
  function eye(n) { const A = zeros(n, n); for (let i = 0; i < n; i += 1) A[i][i] = 1; return A }
  function clone(A) { return A.map(row => row.slice()) }
  function transpose(A) { return A[0].map((_, j) => A.map(row => row[j])) }
  function maxAbsDiff(A, B) {
    let max = 0
    for (let i = 0; i < A.length; i += 1) for (let j = 0; j < A[i].length; j += 1) max = Math.max(max, Math.abs(A[i][j] - B[i][j]))
    return max
  }
  function maxAbsDiffVec(a, b) { return Math.max(...a.map((value, i) => Math.abs(value - b[i]))) }

  function makeOps(digits = 16) {
    const round = value => {
      if (!Number.isFinite(value)) return value
      if (digits >= 16 || value === 0) return value
      return Number(value.toPrecision(digits))
    }
    const rmat = A => A.map(row => row.map(round))
    const rvec = a => a.map(round)
    const add = (A, B) => A.map((row, i) => row.map((value, j) => round(value + B[i][j])))
    const sub = (A, B) => A.map((row, i) => row.map((value, j) => round(value - B[i][j])))
    const addVec = (a, b) => a.map((value, i) => round(value + b[i]))
    const subVec = (a, b) => a.map((value, i) => round(value - b[i]))
    const mul = (A, B) => {
      const C = zeros(A.length, B[0].length)
      for (let i = 0; i < A.length; i += 1) {
        for (let j = 0; j < B[0].length; j += 1) {
          let sum = 0
          for (let k = 0; k < B.length; k += 1) sum = round(sum + round(A[i][k] * B[k][j]))
          C[i][j] = sum
        }
      }
      return C
    }
    const matVec = (A, b) => A.map(row => {
      let sum = 0
      for (let i = 0; i < row.length; i += 1) sum = round(sum + round(row[i] * b[i]))
      return sum
    })
    const inverse = A => {
      const n = A.length
      const M = A.map((row, i) => [...row.map(round), ...eye(n)[i]])
      for (let col = 0; col < n; col += 1) {
        let pivot = col
        for (let row = col + 1; row < n; row += 1) if (Math.abs(M[row][col]) > Math.abs(M[pivot][col])) pivot = row
        if (Math.abs(M[pivot][col]) < 1e-15) throw new Error('numerically singular')
        ;[M[col], M[pivot]] = [M[pivot], M[col]]
        const scale = M[col][col]
        for (let j = 0; j < 2 * n; j += 1) M[col][j] = round(M[col][j] / scale)
        for (let row = 0; row < n; row += 1) {
          if (row === col) continue
          const factor = M[row][col]
          for (let j = 0; j < 2 * n; j += 1) M[row][j] = round(M[row][j] - round(factor * M[col][j]))
        }
      }
      return M.map(row => row.slice(n).map(round))
    }
    const cholesky = A => {
      const n = A.length
      const L = zeros(n, n)
      for (let i = 0; i < n; i += 1) {
        for (let j = 0; j <= i; j += 1) {
          let sum = A[i][j]
          for (let k = 0; k < j; k += 1) sum = round(sum - round(L[i][k] * L[j][k]))
          if (i === j) {
            if (sum <= 1e-15) throw new Error('Cholesky lost positive definiteness')
            L[i][j] = round(Math.sqrt(sum))
          } else {
            L[i][j] = round(sum / L[j][j])
          }
        }
      }
      return L
    }
    const solveLower = (L, b) => {
      const n = L.length, x = Array(n).fill(0)
      for (let i = 0; i < n; i += 1) {
        let sum = b[i]
        for (let j = 0; j < i; j += 1) sum = round(sum - round(L[i][j] * x[j]))
        if (Math.abs(L[i][i]) < 1e-15) throw new Error('singular triangular system')
        x[i] = round(sum / L[i][i])
      }
      return x
    }
    const solveLowerMatrix = (L, B) => {
      const columns = transpose(B).map(column => solveLower(L, column))
      return transpose(columns)
    }
    const qrLeastSquares = (A, b) => {
      const M = rmat(A), y = rvec(b)
      const rows = M.length, cols = M[0].length
      for (let col = 0; col < cols; col += 1) {
        let norm2 = 0
        for (let row = col; row < rows; row += 1) norm2 = round(norm2 + round(M[row][col] * M[row][col]))
        const norm = Math.sqrt(Math.max(0, norm2))
        if (norm < 1e-15) throw new Error('QR lost rank')
        const alpha = round(M[col][col] >= 0 ? -norm : norm)
        const v = Array(rows - col).fill(0)
        v[0] = round(M[col][col] - alpha)
        for (let row = col + 1; row < rows; row += 1) v[row - col] = M[row][col]
        let vtv = 0
        for (const value of v) vtv = round(vtv + round(value * value))
        if (vtv < 1e-30) throw new Error('QR reflector vanished')
        const beta = round(2 / vtv)
        for (let j = col; j < cols; j += 1) {
          let dot = 0
          for (let row = col; row < rows; row += 1) dot = round(dot + round(v[row - col] * M[row][j]))
          const scale = round(beta * dot)
          for (let row = col; row < rows; row += 1) M[row][j] = round(M[row][j] - round(scale * v[row - col]))
        }
        let dot = 0
        for (let row = col; row < rows; row += 1) dot = round(dot + round(v[row - col] * y[row]))
        const scale = round(beta * dot)
        for (let row = col; row < rows; row += 1) y[row] = round(y[row] - round(scale * v[row - col]))
        M[col][col] = alpha
        for (let row = col + 1; row < rows; row += 1) M[row][col] = 0
      }
      return { R: M.slice(0, cols).map(row => row.slice(0, cols)), qtb: y.slice(0, cols) }
    }
    const solveUpper = (R, b) => {
      const n = R.length, x = Array(n).fill(0)
      for (let i = n - 1; i >= 0; i -= 1) {
        let sum = b[i]
        for (let j = i + 1; j < n; j += 1) sum = round(sum - round(R[i][j] * x[j]))
        if (Math.abs(R[i][i]) < 1e-15) throw new Error('singular triangular system')
        x[i] = round(sum / R[i][i])
      }
      return x
    }
    return { round, rmat, rvec, add, sub, addVec, subVec, mul, matVec, inverse, cholesky, solveLower, solveLowerMatrix, qrLeastSquares, solveUpper }
  }

  function jacobiEigenvalues(A) {
    const M = clone(A).map((row, i) => row.map((value, j) => (value + A[j][i]) / 2))
    const n = M.length
    for (let iter = 0; iter < 80; iter += 1) {
      let p = 0, q = 1, max = 0
      for (let i = 0; i < n; i += 1) for (let j = i + 1; j < n; j += 1) if (Math.abs(M[i][j]) > max) { max = Math.abs(M[i][j]); p = i; q = j }
      if (max < 1e-13) break
      const angle = .5 * Math.atan2(2 * M[p][q], M[q][q] - M[p][p])
      const c = Math.cos(angle), s = Math.sin(angle)
      for (let k = 0; k < n; k += 1) {
        const mkp = M[k][p], mkq = M[k][q]
        M[k][p] = c * mkp - s * mkq
        M[k][q] = s * mkp + c * mkq
      }
      for (let k = 0; k < n; k += 1) {
        const mpk = M[p][k], mqk = M[q][k]
        M[p][k] = c * mpk - s * mqk
        M[q][k] = s * mpk + c * mqk
      }
    }
    return M.map((row, i) => row[i]).sort((a, b) => a - b)
  }

  function rngFromSeed(seed) {
    let x = seed >>> 0
    return () => {
      x ^= x << 13; x ^= x >>> 17; x ^= x << 5
      return (x >>> 0) / 4294967296
    }
  }

  function normal(rng) {
    const u = Math.max(1e-12, rng()), v = rng()
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
  }

  function orthogonal(n, rng) {
    const Q = zeros(n, n)
    for (let j = 0; j < n; j += 1) {
      let v = Array.from({ length: n }, () => normal(rng))
      for (let i = 0; i < j; i += 1) {
        const dot = v.reduce((sum, value, k) => sum + value * Q[k][i], 0)
        v = v.map((value, k) => value - dot * Q[k][i])
      }
      let norm = Math.hypot(...v)
      if (norm < 1e-10) { v = Array.from({ length: n }, (_, i) => i === j ? 1 : 0); norm = 1 }
      for (let i = 0; i < n; i += 1) Q[i][j] = v[i] / norm
    }
    return Q
  }

  function spdWithCondition(n, exponent, rng, scale = 1) {
    const Q = orthogonal(n, rng)
    const eigen = Array.from({ length: n }, (_, i) => scale * Math.pow(10, -exponent * i / Math.max(1, n - 1)))
    const D = zeros(n, n); eigen.forEach((value, i) => { D[i][i] = value })
    const ops = makeOps(16)
    return ops.mul(ops.mul(Q, D), transpose(Q))
  }

  function generateProblem(n, m, exponent, seed) {
    const rng = rngFromSeed(seed)
    const P = spdWithCondition(n, exponent, rng, 1.4)
    const R = spdWithCondition(m, 1, rng, .35)
    const full = makeOps(16)
    const LP = full.cholesky(P)
    const LR = full.cholesky(R)
    const H = Array.from({ length: m }, () => Array.from({ length: n }, () => normal(rng) / Math.sqrt(n)))
    const priorMean = Array.from({ length: n }, () => .6 * normal(rng))
    // Draw from the same Gaussian model used by every algebraic route.
    const xi = Array.from({ length: n }, () => normal(rng))
    const eta = Array.from({ length: m }, () => normal(rng))
    const trueState = full.addVec(priorMean, full.matVec(LP, xi))
    const z = full.addVec(full.matVec(H, trueState), full.matVec(LR, eta))
    return { P, R, LP, LR, H, priorMean, z }
  }

  function covarianceMethod(problem, ops, joseph = false) {
    const P = ops.rmat(problem.P), R = ops.rmat(problem.R), H = ops.rmat(problem.H)
    const m = ops.rvec(problem.priorMean), z = ops.rvec(problem.z)
    const HPHt = ops.mul(ops.mul(H, P), transpose(H))
    const S = ops.add(HPHt, R)
    const K = ops.mul(ops.mul(P, transpose(H)), ops.inverse(S))
    const innovation = ops.subVec(z, ops.matVec(H, m))
    const mean = ops.addVec(m, ops.matVec(K, innovation))
    let covariance
    if (joseph) {
      const A = ops.sub(eye(P.length), ops.mul(K, H))
      covariance = ops.add(ops.mul(ops.mul(A, P), transpose(A)), ops.mul(ops.mul(K, R), transpose(K)))
    } else {
      covariance = ops.sub(P, ops.mul(ops.mul(K, S), transpose(K)))
    }
    return { mean, covariance, K }
  }

  function informationMethod(problem, ops) {
    const P = ops.rmat(problem.P), R = ops.rmat(problem.R), H = ops.rmat(problem.H)
    const m = ops.rvec(problem.priorMean), z = ops.rvec(problem.z)
    const Pinv = ops.inverse(P), Rinv = ops.inverse(R)
    const HtRinv = ops.mul(transpose(H), Rinv)
    const Lambda = ops.add(Pinv, ops.mul(HtRinv, H))
    const eta = ops.addVec(ops.matVec(Pinv, m), ops.matVec(HtRinv, z))
    const covariance = ops.inverse(Lambda)
    const mean = ops.matVec(covariance, eta)
    return { mean, covariance, K: null }
  }

  function qrMethod(problem, ops) {
    const LP = ops.rmat(problem.LP), LR = ops.rmat(problem.LR), H = ops.rmat(problem.H)
    const m = ops.rvec(problem.priorMean), z = ops.rvec(problem.z)
    const priorRows = ops.solveLowerMatrix(LP, eye(LP.length))
    const measurementRows = ops.solveLowerMatrix(LR, H)
    const A = [...priorRows, ...measurementRows]
    const b = [...ops.solveLower(LP, m), ...ops.solveLower(LR, z)]
    const { R: Rx, qtb } = ops.qrLeastSquares(A, b)
    const mean = ops.solveUpper(Rx, qtb)
    const inverseColumns = []
    for (let column = 0; column < LP.length; column += 1) {
      const basis = Array.from({ length: LP.length }, (_, row) => row === column ? 1 : 0)
      inverseColumns.push(ops.solveUpper(Rx, basis))
    }
    const W = transpose(inverseColumns)
    const covariance = zeros(LP.length, LP.length)
    for (let row = 0; row < LP.length; row += 1) {
      for (let column = 0; column <= row; column += 1) {
        let value = 0
        for (let k = 0; k < LP.length; k += 1) value = ops.round(value + ops.round(W[row][k] * W[column][k]))
        covariance[row][column] = value
        covariance[column][row] = value
      }
    }
    return { mean, covariance, K: null }
  }

  function methodDiagnostics(result, reference) {
    const symmetry = maxAbsDiff(result.covariance, transpose(result.covariance))
    const minEigen = jacobiEigenvalues(result.covariance)[0]
    const delta = Math.max(maxAbsDiff(result.covariance, reference.covariance), maxAbsDiffVec(result.mean, reference.mean))
    return { symmetry, minEigen, delta }
  }

  function tone(value, kind = 'delta') {
    if (!Number.isFinite(value)) return 'bad'
    if (kind === 'eigen') return value > -1e-10 ? 'good' : value > -1e-5 ? 'warn' : 'bad'
    return value < 1e-8 ? 'good' : value < 1e-4 ? 'warn' : 'bad'
  }

  function matrixText(A, digits = 3) {
    return A.map(row => row.map(value => fmt(value, digits).padStart(8)).join(' ')).join('\n')
  }

  function mountEquivalence() {
    const shell = createShell({
      accent: COLORS.violet,
      eyebrow: 'MATRIX IDENTITIES · FINITE PRECISION',
      title: 'Same target, different arithmetic',
      intro: 'Toy rounding, not IEEE emulation. QR starts from covariance factors; other paths start from matrices. The reference is native double, not exact truth.',
      controls: [
        `<div class="control"><div class="control-head"><label for="eq-n">State dimension ${tex`n`}</label></div><select id="eq-n"><option>2</option><option selected>3</option><option>4</option></select></div>`,
        `<div class="control"><div class="control-head"><label for="eq-m">Measurement dimension ${tex`m`}</label></div><select id="eq-m"><option selected>1</option><option>2</option><option>3</option></select></div>`,
        rangeControl('eq-condition', 'Prior condition number', 0, 8, .5, 3, '10^3'),
        rangeControl('eq-digits', 'Simulated significant digits', 5, 16, 1, 16, 'native double'),
        '<div class="button-row"><button id="eq-benign" type="button">Benign case</button><button id="eq-random" class="primary" type="button">New problem</button></div>'
      ].join(''),
      stageTitle: 'DIFFERENCE FROM NATIVE-DOUBLE REFERENCE',
      stageBody: `
        <div class="eq-layout">
          <p class="review-numerics-note">P_s = (P + Pᵀ)/2. QR symmetry is enforced by mirroring. Δ is the largest entrywise difference across the toy mean and covariance.</p>
          <div class="eq-grid" id="eq-grid"></div>
          <div class="matrix-strip">
            <div class="matrix-card"><strong>Reference posterior covariance ${tex`P^+`}</strong><pre id="eq-posterior">—</pre></div>
            <div class="matrix-card"><strong>Reference Kalman gain ${tex`K`}</strong><pre id="eq-gain">—</pre></div>
          </div>
        </div>`
    })

    const nSelect = document.getElementById('eq-n')
    const mSelect = document.getElementById('eq-m')
    const condition = document.getElementById('eq-condition')
    const digits = document.getElementById('eq-digits')
    let seed = 1949
    let problem = null

    function syncMOptions() {
      const n = Number(nSelect.value)
      Array.from(mSelect.options).forEach(option => { option.disabled = Number(option.value) > n })
      if (Number(mSelect.value) > n) mSelect.value = String(n)
    }

    function regenerate() {
      syncMOptions()
      problem = generateProblem(Number(nSelect.value), Number(mSelect.value), Number(condition.value), seed)
      evaluate()
    }

    function cardHtml(name, subtitle, diagnostics, extraClass = '', error = null) {
      if (error) {
        return `<article class="eq-card ${extraClass}"><div class="eq-head"><strong>${name}</strong></div><div class="eq-formula">${mathHtml(subtitle, true)}</div><div class="metric-v bad">${escapeHtml(error.message)}</div><p class="metric-copy">The simulated arithmetic lost rank or positive definiteness.</p></article>`
      }
      const values = [
        [mathHtml(String.raw`\max\Delta`), diagnostics.delta, tone(diagnostics.delta)],
        ['symmetry', diagnostics.symmetry, tone(diagnostics.symmetry)],
        [mathHtml(String.raw`\lambda_{\min}(P_s)`), diagnostics.minEigen, tone(diagnostics.minEigen, 'eigen')]
      ]
      return `<article class="eq-card ${extraClass}">
        <div class="eq-head"><strong>${name}</strong></div>
        <div class="eq-formula">${mathHtml(subtitle, true)}</div>
        <div class="eq-metrics">${values.map(([label, value, cls]) => `<div class="eq-metric"><small>${label}</small><code class="${cls}">${fmt(value, 3)}</code></div>`).join('')}</div>
      </article>`
    }

    function evaluate() {
      if (!problem) return
      const exp = Number(condition.value)
      const d = Number(digits.value)
      setMath(document.getElementById('eq-condition-out'), exp === 0 ? '1' : String.raw`10^{${fmt(exp, 1)}}`)
      document.getElementById('eq-digits-out').textContent = d >= 16 ? 'native double' : `${d} digits`
      const full = makeOps(16)
      let reference
      try {
        reference = covarianceMethod(problem, full, false)
      } catch (error) {
        setStatus(shell.status, 'reference solve failed', 'bad')
        return
      }
      const specs = [
        ['Covariance', String.raw`P^- - KSK^\mathsf{T}`, ops => covarianceMethod(problem, ops, false), 'reference'],
        ['Information', String.raw`\left((P^-)^{-1}+H^\mathsf{T}R^{-1}H\right)^{-1}`, ops => informationMethod(problem, ops), ''],
        ['Joseph', String.raw`(I-KH)P^-(I-KH)^\mathsf{T}+KRK^\mathsf{T}`, ops => covarianceMethod(problem, ops, true), 'stable'],
        ['QR / square root', String.raw`\text{whiten}\to\mathrm{QR}\to\text{triangular solve}`, ops => qrMethod(problem, ops), 'stable']
      ]
      const rendered = []
      let worst = 0, failures = 0, negative = false
      for (const [name, subtitle, solve, cls] of specs) {
        try {
          const result = solve(makeOps(d))
          const diagnostics = methodDiagnostics(result, reference)
          worst = Math.max(worst, diagnostics.delta)
          negative ||= diagnostics.minEigen < -1e-10
          rendered.push(cardHtml(name, subtitle, diagnostics, cls))
        } catch (error) {
          failures += 1
          rendered.push(cardHtml(name, subtitle, null, cls, error))
        }
      }
      document.getElementById('eq-grid').innerHTML = rendered.join('')
      window.typesetDynamicMath?.()
      document.getElementById('eq-posterior').textContent = matrixText(reference.covariance, 4)
      document.getElementById('eq-gain').textContent = matrixText(reference.K, 4)
      if (failures) setStatus(shell.status, `${failures} formulation${failures === 1 ? '' : 's'} failed`, 'bad')
      else if (negative || worst > 1e-4) setStatus(shell.status, 'finite precision separates the forms', 'warn')
      else if (worst < 1e-10) setStatus(shell.status, 'all forms agree', 'good')
      else setStatus(shell.status, String.raw`\max\Delta=${fmt(worst, 2)}`, 'good', true)
    }

    nSelect.addEventListener('change', regenerate)
    mSelect.addEventListener('change', regenerate)
    condition.addEventListener('input', regenerate)
    digits.addEventListener('input', evaluate)
    document.getElementById('eq-random').addEventListener('click', () => { seed = (seed + 7919) >>> 0; regenerate() })
    document.getElementById('eq-benign').addEventListener('click', () => { condition.value = 1; digits.value = 16; seed = 1949; regenerate() })
    regenerate()
  }

  if (demo === 'geometry') mountGeometry()
  else if (demo === 'equivalence') mountEquivalence()
  else mountScalar()
})()
