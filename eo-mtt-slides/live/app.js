(() => {
  'use strict'

  const NS = 'http://www.w3.org/2000/svg'
  const VALID_VIEWS = new Set(['partition', 'hypotheses', 'inference'])
  const VIEW_COPY = {
    partition: ['LIVE EXAMPLE · CANDIDATE PARTITIONS', 'What can a distance sweep see?'],
    hypotheses: ['LIVE EXAMPLE · HYPOTHESIS MANAGEMENT', 'What survives branch, prune, and projection?'],
    inference: ['LIVE EXAMPLE · DIRECT APPROXIMATE INFERENCE', 'Joint search is not marginal inference']
  }
  const COLORS = ['#2456c8', '#e08607', '#0e9f6e', '#6d4fc2', '#b7355c', '#0694a2', '#65a30d']
  const CLUTTER = '#8b929c'
  const AREA = 100 * 66
  const MEASUREMENTS = [
    { x: 28, y: 42.5 }, { x: 31, y: 48 }, { x: 36.5, y: 44.5 }, { x: 29.5, y: 46 },
    { x: 59, y: 49.5 }, { x: 66, y: 53.5 }, { x: 67.5, y: 49 }, { x: 62, y: 52.5 },
    { x: 48.5, y: 48.5 }, { x: 86, y: 18 }
  ]
  const OBJECTS = [
    { x: 32, y: 45.2, rx: 9, ry: 5, angle: -12, color: '#2456c8', label: 'object 1' },
    { x: 63.5, y: 51, rx: 8.5, ry: 4.6, angle: 15, color: '#e08607', label: 'object 2' }
  ]
  const MISSING_PARTITION = [[0, 1, 2, 3, 8], [4, 5, 6, 7], [9]]

  const elements = {
    appShell: document.getElementById('appShell'),
    viewKicker: document.getElementById('viewKicker'),
    viewTitle: document.getElementById('viewTitle'),
    backButton: document.getElementById('backButton'),
    viewLinks: Array.from(document.querySelectorAll('[data-view-link]')),
    partitionScene: document.getElementById('partitionScene'),
    bellM: document.getElementById('bellM'),
    bellMValue: document.getElementById('bellMValue'),
    bellOutput: document.getElementById('bellOutput'),
    distanceThreshold: document.getElementById('distanceThreshold'),
    distanceOutput: document.getElementById('distanceOutput'),
    cellCount: document.getElementById('cellCount'),
    candidateCount: document.getElementById('candidateCount'),
    missingButton: document.getElementById('missingButton'),
    partitionNote: document.getElementById('partitionNote'),
    branchButton: document.getElementById('branchButton'),
    pruneButton: document.getElementById('pruneButton'),
    hypothesisReset: document.getElementById('hypothesisReset'),
    hypothesisStatus: document.getElementById('hypothesisStatus'),
    hypothesisCards: document.getElementById('hypothesisCards'),
    ambiguousMark: document.getElementById('ambiguousMark'),
    marginalBars: document.getElementById('marginalBars'),
    gibbsScene: document.getElementById('gibbsScene'),
    gibbsMoves: document.getElementById('gibbsMoves'),
    gibbsRun: document.getElementById('gibbsRun'),
    gibbsStep: document.getElementById('gibbsStep'),
    gibbsReset: document.getElementById('gibbsReset'),
    visitList: document.getElementById('visitList'),
    bpScene: document.getElementById('bpScene'),
    bpIteration: document.getElementById('bpIteration'),
    bpRun: document.getElementById('bpRun'),
    bpStep: document.getElementById('bpStep'),
    bpReset: document.getElementById('bpReset'),
    bpReadout: document.getElementById('bpReadout')
  }

  const state = {
    view: validatedView(new URLSearchParams(window.location.search).get('view')),
    showMissing: false,
    hypotheses: [],
    hypothesisStage: 0,
    hypothesisScan: 0,
    gibbs: null,
    bp: null,
    paused: document.hidden
  }

  function validatedView(value) {
    return VALID_VIEWS.has(value) ? value : 'partition'
  }

  function svgElement(tag, attributes = {}, parent) {
    const element = document.createElementNS(NS, tag)
    Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, String(value)))
    if (parent) parent.appendChild(element)
    return element
  }

  function screenPoint(point, width = 720, height = 355) {
    return {
      x: width * (0.06 + point.x / 112),
      y: height * (0.08 + point.y / 78)
    }
  }

  function clear(element) {
    while (element.firstChild) element.removeChild(element.firstChild)
  }

  function grid(svg, width, height) {
    const defs = svgElement('defs', {}, svg)
    const pattern = svgElement('pattern', { id: `grid-${svg.id}`, width: 36, height: 36, patternUnits: 'userSpaceOnUse' }, defs)
    svgElement('path', { d: 'M36 0H0V36', fill: 'none', stroke: '#d8dcd1', 'stroke-width': 1 }, pattern)
    svgElement('rect', { x: 1, y: 1, width: width - 2, height: height - 2, rx: 12, fill: '#fdfdfb', stroke: '#d8dcd1' }, svg)
    svgElement('rect', { x: 1, y: 1, width: width - 2, height: height - 2, rx: 12, fill: `url(#grid-${svg.id})`, opacity: 0.7 }, svg)
  }

  function drawTruth(svg, width, height) {
    OBJECTS.forEach((object) => {
      const center = screenPoint(object, width, height)
      const radiusX = width * object.rx / 112
      const radiusY = height * object.ry / 78
      svgElement('ellipse', {
        cx: center.x,
        cy: center.y,
        rx: radiusX,
        ry: radiusY,
        transform: `rotate(${object.angle} ${center.x} ${center.y})`,
        fill: object.color,
        'fill-opacity': 0.05,
        stroke: object.color,
        'stroke-opacity': 0.55,
        'stroke-width': 2,
        'stroke-dasharray': '7 6'
      }, svg)
      const label = svgElement('text', {
        x: center.x,
        y: center.y - radiusY - 12,
        'text-anchor': 'middle',
        fill: object.color,
        'font-size': 11,
        'font-family': 'system-ui, sans-serif',
        'font-weight': 800
      }, svg)
      label.textContent = object.label
    })
  }

  function cross(origin, a, b) {
    return (a.x - origin.x) * (b.y - origin.y) - (a.y - origin.y) * (b.x - origin.x)
  }

  function convexHull(points) {
    const sorted = points.slice().sort((a, b) => a.x - b.x || a.y - b.y)
    if (sorted.length < 3) return sorted
    const lower = []
    sorted.forEach((point) => {
      while (lower.length > 1 && cross(lower[lower.length - 2], lower[lower.length - 1], point) <= 0) lower.pop()
      lower.push(point)
    })
    const upper = []
    sorted.slice().reverse().forEach((point) => {
      while (upper.length > 1 && cross(upper[upper.length - 2], upper[upper.length - 1], point) <= 0) upper.pop()
      upper.push(point)
    })
    lower.pop()
    upper.pop()
    return lower.concat(upper)
  }

  function drawCell(svg, indexes, color, width, height) {
    const points = indexes.map((index) => screenPoint(MEASUREMENTS[index], width, height))
    if (points.length === 1) {
      svgElement('circle', {
        cx: points[0].x,
        cy: points[0].y,
        r: 18,
        fill: color,
        'fill-opacity': 0.09,
        stroke: color,
        'stroke-opacity': 0.72,
        'stroke-width': 2
      }, svg)
      return
    }
    if (points.length === 2) {
      svgElement('line', {
        x1: points[0].x,
        y1: points[0].y,
        x2: points[1].x,
        y2: points[1].y,
        stroke: color,
        'stroke-opacity': 0.12,
        'stroke-width': 35,
        'stroke-linecap': 'round'
      }, svg)
      svgElement('line', {
        x1: points[0].x,
        y1: points[0].y,
        x2: points[1].x,
        y2: points[1].y,
        stroke: color,
        'stroke-opacity': 0.65,
        'stroke-width': 2
      }, svg)
      return
    }
    const hull = convexHull(points)
    const center = hull.reduce((sum, point) => ({ x: sum.x + point.x, y: sum.y + point.y }), { x: 0, y: 0 })
    center.x /= hull.length
    center.y /= hull.length
    const padded = hull.map((point) => {
      const dx = point.x - center.x
      const dy = point.y - center.y
      const length = Math.hypot(dx, dy) || 1
      return { x: point.x + dx / length * 18, y: point.y + dy / length * 18 }
    })
    const path = `M${padded.map((point) => `${point.x.toFixed(1)},${point.y.toFixed(1)}`).join('L')}Z`
    svgElement('path', {
      d: path,
      fill: color,
      'fill-opacity': 0.09,
      stroke: color,
      'stroke-opacity': 0.72,
      'stroke-width': 2,
      'stroke-linejoin': 'round'
    }, svg)
  }

  function drawMeasurements(svg, cells, clutter = [], width = 720, height = 355, options = {}) {
    clear(svg)
    grid(svg, width, height)
    drawTruth(svg, width, height)
    const colorByIndex = new Map()
    cells.forEach((cell, cellIndex) => {
      const color = COLORS[cellIndex % COLORS.length]
      cell.forEach((index) => colorByIndex.set(index, color))
      drawCell(svg, cell, color, width, height)
    })

    if (options.edges) options.edges(svg, width, height)

    MEASUREMENTS.forEach((measurement, index) => {
      const point = screenPoint(measurement, width, height)
      const isClutter = clutter.includes(index)
      const fill = colorByIndex.get(index) || '#3a4148'
      svgElement('circle', {
        cx: point.x,
        cy: point.y,
        r: index === 8 ? 7.5 : 6.5,
        fill: isClutter ? '#fff' : fill,
        stroke: isClutter ? CLUTTER : '#fff',
        'stroke-width': isClutter ? 2.4 : 2.2
      }, svg)
      const label = svgElement('text', {
        x: point.x + 10,
        y: point.y - 9,
        fill: index === 8 ? '#1b2320' : '#69746d',
        'font-size': index === 8 ? 12 : 10,
        'font-family': 'system-ui, sans-serif',
        'font-weight': index === 8 ? 900 : 700
      }, svg)
      label.textContent = `m${index + 1}`
    })
  }

  function distance(i, j) {
    return Math.hypot(MEASUREMENTS[i].x - MEASUREMENTS[j].x, MEASUREMENTS[i].y - MEASUREMENTS[j].y)
  }

  function singleLinkage(threshold) {
    const parent = MEASUREMENTS.map((_, index) => index)
    function find(index) {
      while (parent[index] !== index) {
        parent[index] = parent[parent[index]]
        index = parent[index]
      }
      return index
    }
    for (let i = 0; i < MEASUREMENTS.length; i += 1) {
      for (let j = i + 1; j < MEASUREMENTS.length; j += 1) {
        if (distance(i, j) <= threshold) {
          const a = find(i)
          const b = find(j)
          if (a !== b) parent[a] = b
        }
      }
    }
    const groups = new Map()
    MEASUREMENTS.forEach((_, index) => {
      const root = find(index)
      if (!groups.has(root)) groups.set(root, [])
      groups.get(root).push(index)
    })
    return Array.from(groups.values()).sort((a, b) => a[0] - b[0])
  }

  function signature(cells) {
    return cells.map((cell) => `{${cell.map((index) => index + 1).join(',')}}`).join('')
  }

  function candidateSet() {
    const seen = new Set()
    for (let threshold = 0.5; threshold <= 40.001; threshold += 0.25) {
      seen.add(signature(singleLinkage(threshold)))
    }
    return seen
  }

  function bellNumbers(maximum) {
    const bells = [1n]
    let row = [1n]
    for (let n = 1; n <= maximum; n += 1) {
      const next = [row[row.length - 1]]
      for (let index = 0; index < row.length; index += 1) next.push(next[index] + row[index])
      row = next
      bells.push(row[0])
    }
    return bells
  }

  const BELLS = bellNumbers(30)
  const CANDIDATES = candidateSet()

  function formatBigInt(value) {
    return value.toString().replace(/\B(?=(\d{3})+(?!\d))/g, '\u2009')
  }

  function renderPartition() {
    const measurementCount = Number(elements.bellM.value)
    elements.bellMValue.textContent = String(measurementCount)
    elements.bellOutput.textContent = `B(${measurementCount}) = ${formatBigInt(BELLS[measurementCount])}`

    const threshold = Number(elements.distanceThreshold.value)
    elements.distanceOutput.textContent = `d = ${threshold.toFixed(2).replace(/0$/, '')}`
    elements.candidateCount.textContent = String(CANDIDATES.size)

    const cells = state.showMissing ? MISSING_PARTITION : singleLinkage(threshold)
    drawMeasurements(elements.partitionScene, cells, [], 720, 355)
    elements.cellCount.textContent = String(cells.length)
    elements.missingButton.setAttribute('aria-pressed', String(state.showMissing))
    elements.missingButton.textContent = state.showMissing
      ? 'Return to the distance sweep'
      : 'Show a plausible partition the sweep misses'
    elements.partitionNote.textContent = state.showMissing
      ? 'Missing candidate: {1,2,3,4,9}{5,6,7,8}{10}. Single linkage attaches m₉ to the right cluster first, so no swept d can produce this grouping.'
      : `At d = ${threshold.toFixed(2).replace(/0$/, '')}, single linkage yields ${cells.length} cells. Across the full sweep it produces only ${CANDIDATES.size} distinct candidates out of 115,975.`
  }

  const BASE_HYPOTHESES = [
    { weight: 0.46, label: 'm₉ from object 2', code: '{1–4}{5–9} · m₁₀ clutter' },
    { weight: 0.31, label: 'm₉ from object 1', code: '{1–4,9}{5–8} · m₁₀ clutter' },
    { weight: 0.14, label: 'm₉ births new object', code: '{1–4}{5–8}{9} · m₁₀ clutter' },
    { weight: 0.09, label: 'm₉ is clutter', code: '{1–4}{5–8} · m₉,m₁₀ clutter' }
  ]
  const BRANCH_FACTORS = [
    { weight: 0.55, label: 'association α' },
    { weight: 0.30, label: 'association β' },
    { weight: 0.15, label: 'association γ' }
  ]

  function resetHypotheses() {
    state.hypotheses = BASE_HYPOTHESES.map((hypothesis) => ({ ...hypothesis }))
    state.hypothesisStage = 0
    state.hypothesisScan = 0
    renderHypotheses('4 global hypotheses · current scan')
  }

  function branchHypotheses() {
    state.hypothesisScan += 1
    state.hypotheses = state.hypotheses.flatMap((hypothesis) => BRANCH_FACTORS.map((factor) => ({
      weight: hypothesis.weight * factor.weight,
      label: hypothesis.label,
      code: `${hypothesis.code} · ${factor.label} @ t+${state.hypothesisScan}`
    })))
    state.hypothesisStage = 1
    renderHypotheses(`${state.hypotheses.length} children · each parent branched 3 ways`)
  }

  function pruneHypotheses() {
    const sorted = state.hypotheses.slice().sort((a, b) => b.weight - a.weight)
    const kept = sorted.slice(0, 4)
    const retainedWeight = kept.reduce((sum, hypothesis) => sum + hypothesis.weight, 0)
    const totalWeight = sorted.reduce((sum, hypothesis) => sum + hypothesis.weight, 0)
    state.hypotheses = kept.map((hypothesis) => ({ ...hypothesis, weight: hypothesis.weight / retainedWeight }))
    state.hypothesisStage = 0
    renderHypotheses(`capped at 4 · discarded ${(totalWeight - retainedWeight).toFixed(3)} weight, then renormalized`)
  }

  function renderHypotheses(status) {
    elements.hypothesisCards.replaceChildren()
    const total = state.hypotheses.reduce((sum, hypothesis) => sum + hypothesis.weight, 0) || 1
    state.hypotheses.forEach((hypothesis, index) => {
      const card = document.createElement('article')
      card.className = 'hypothesis-card'
      card.innerHTML = `
        <header><span>H${index + 1}</span><strong>${hypothesis.weight.toFixed(3)}</strong></header>
        <div class="weight-track"><span style="width:${Math.max(2, hypothesis.weight / total * 100).toFixed(1)}%"></span></div>
        <p><b>${hypothesis.label}</b><br>${hypothesis.code}</p>
      `
      elements.hypothesisCards.appendChild(card)
    })
    elements.hypothesisStatus.textContent = status
    elements.branchButton.disabled = state.hypothesisStage !== 0
    elements.pruneButton.disabled = state.hypothesisStage !== 1
    renderMarginals()
  }

  function renderMarginals() {
    const categories = [
      ['m₉ from object 2', 'object 2', '#e08607'],
      ['m₉ from object 1', 'object 1', '#2456c8'],
      ['m₉ births new object', 'new object', '#0e9f6e'],
      ['m₉ is clutter', 'clutter', '#8b929c']
    ]
    const total = state.hypotheses.reduce((sum, hypothesis) => sum + hypothesis.weight, 0) || 1
    const rows = categories.map(([hypothesisLabel, displayLabel, color]) => [
      displayLabel,
      state.hypotheses
        .filter((hypothesis) => hypothesis.label === hypothesisLabel)
        .reduce((sum, hypothesis) => sum + hypothesis.weight, 0) / total,
      color
    ])
    elements.marginalBars.innerHTML = rows.map(([label, value, color]) => `
      <div class="marginal-row">
        <span>${label}</span>
        <div class="bar"><span style="width:${value * 100}%;background:${color}"></span></div>
        <strong>${value.toFixed(2)}</strong>
      </div>
    `).join('')
    let cursor = 0
    const stops = rows.map(([, value, color]) => {
      const start = cursor
      cursor += value * 100
      return `${color} ${start.toFixed(2)}% ${cursor.toFixed(2)}%`
    }).join(', ')
    elements.ambiguousMark.style.background = `linear-gradient(#fff, #fff) padding-box, conic-gradient(${stops}) border-box`
    elements.ambiguousMark.setAttribute('aria-label', `Current m9 marginals: ${rows.map(([label, value]) => `${label} ${value.toFixed(2)}`).join(', ')}`)
  }

  function gaussian2(squaredDistance, variance) {
    return Math.exp(-squaredDistance / (2 * variance)) / (2 * Math.PI * variance)
  }

  function mulberry32(seed) {
    return function random() {
      seed |= 0
      seed = seed + 0x6D2B79F5 | 0
      let value = Math.imul(seed ^ seed >>> 15, 1 | seed)
      value = value + Math.imul(value ^ value >>> 7, 61 | value) ^ value
      return ((value ^ value >>> 14) >>> 0) / 4294967296
    }
  }

  function gibbsCells() {
    const groups = new Map()
    state.gibbs.assignments.forEach((assignment, index) => {
      if (assignment < 0) return
      if (!groups.has(assignment)) groups.set(assignment, [])
      groups.get(assignment).push(index)
    })
    return Array.from(groups.values()).sort((a, b) => a[0] - b[0])
  }

  function gibbsSignature() {
    const clutter = []
    state.gibbs.assignments.forEach((assignment, index) => { if (assignment < 0) clutter.push(index + 1) })
    return signature(gibbsCells()) + (clutter.length ? ` ∅{${clutter.join(',')}}` : '')
  }

  function recordGibbs() {
    const current = gibbsSignature()
    state.gibbs.visits.set(current, (state.gibbs.visits.get(current) || 0) + 1)
  }

  function resetGibbs() {
    state.gibbs = {
      assignments: MEASUREMENTS.map(() => 0),
      visits: new Map(),
      moves: 0,
      pointer: 0,
      last: null,
      random: mulberry32(20260816)
    }
    recordGibbs()
    renderGibbs()
  }

  function stepGibbs() {
    const sample = state.gibbs
    const index = sample.pointer
    sample.pointer = (sample.pointer + 1) % MEASUREMENTS.length
    sample.last = index
    const groups = new Map()
    let maximumId = -1
    sample.assignments.forEach((assignment, measurementIndex) => {
      if (measurementIndex === index || assignment < 0) return
      if (!groups.has(assignment)) groups.set(assignment, [])
      groups.get(assignment).push(measurementIndex)
      maximumId = Math.max(maximumId, assignment)
    })
    const options = []
    const weights = []
    groups.forEach((members, groupId) => {
      const mean = members.reduce((sum, member) => ({
        x: sum.x + MEASUREMENTS[member].x,
        y: sum.y + MEASUREMENTS[member].y
      }), { x: 0, y: 0 })
      mean.x /= members.length
      mean.y /= members.length
      const variance = 5.5 * 5.5 * (1 + 1 / members.length) + 2
      const squaredDistance = Math.pow(MEASUREMENTS[index].x - mean.x, 2) + Math.pow(MEASUREMENTS[index].y - mean.y, 2)
      options.push(groupId)
      weights.push(members.length * gaussian2(squaredDistance, variance))
    })
    options.push(maximumId + 1)
    weights.push(1 / AREA)
    options.push(-1)
    weights.push(2 / AREA)
    let draw = sample.random() * weights.reduce((sum, weight) => sum + weight, 0)
    let selected = 0
    while (selected < weights.length - 1 && draw > weights[selected]) {
      draw -= weights[selected]
      selected += 1
    }
    sample.assignments[index] = options[selected]
    sample.moves += 1
    recordGibbs()
  }

  function renderGibbs() {
    const clutter = []
    state.gibbs.assignments.forEach((assignment, index) => { if (assignment < 0) clutter.push(index) })
    drawMeasurements(elements.gibbsScene, gibbsCells(), clutter, 620, 225)
    if (state.gibbs.last !== null) {
      const point = screenPoint(MEASUREMENTS[state.gibbs.last], 620, 225)
      svgElement('circle', { cx: point.x, cy: point.y, r: 12, fill: 'none', stroke: '#1b2320', 'stroke-width': 2 }, elements.gibbsScene)
    }
    elements.gibbsMoves.textContent = `${state.gibbs.moves} moves`
    const entries = Array.from(state.gibbs.visits.entries()).sort((a, b) => b[1] - a[1])
    const total = entries.reduce((sum, entry) => sum + entry[1], 0)
    elements.visitList.innerHTML = entries.slice(0, 2).map(([label, count]) => {
      const share = count / total
      return `<div class="visit-row" title="${label}"><span>${label}</span><div class="bar"><span style="width:${share * 100}%"></span></div><strong>${(share * 100).toFixed(0)}%</strong></div>`
    }).join('')
  }

  function bpLikelihood(index, objectIndex) {
    const object = OBJECTS[objectIndex]
    const variance = objectIndex === 0 ? 57 : 51
    const squaredDistance = Math.pow(MEASUREMENTS[index].x - object.x, 2) + Math.pow(MEASUREMENTS[index].y - object.y, 2)
    return gaussian2(squaredDistance, variance)
  }

  function computeBeliefs() {
    const gamma = 4
    const newObject = 1 / AREA
    const clutter = 2 / AREA
    state.bp.probabilities = MEASUREMENTS.map((_, index) => {
      const weights = [
        state.bp.existence[0] * gamma * bpLikelihood(index, 0),
        state.bp.existence[1] * gamma * bpLikelihood(index, 1),
        newObject,
        clutter
      ]
      const normalization = weights.reduce((sum, weight) => sum + weight, 0)
      return weights.map((weight) => weight / normalization)
    })
  }

  function resetBp() {
    state.bp = { existence: [0.5, 0.5], iteration: 0, probabilities: [] }
    computeBeliefs()
    renderBp()
  }

  function stepBp() {
    const support = [0, 0]
    state.bp.probabilities.forEach((probabilities) => {
      support[0] += probabilities[0]
      support[1] += probabilities[1]
    })
    state.bp.existence = support.map((value) => Math.min(0.95, value / (value + 0.7)))
    state.bp.iteration += 1
    computeBeliefs()
  }

  function renderBp() {
    drawMeasurements(elements.bpScene, [], [9], 620, 225, {
      edges(svg, width, height) {
        MEASUREMENTS.forEach((measurement, index) => {
          const point = screenPoint(measurement, width, height)
          OBJECTS.forEach((object, objectIndex) => {
            const probability = state.bp.probabilities[index][objectIndex]
            if (probability <= 0.03) return
            const target = screenPoint(object, width, height)
            svgElement('line', {
              x1: point.x,
              y1: point.y,
              x2: target.x,
              y2: target.y,
              stroke: object.color,
              'stroke-width': 1 + probability * 3,
              'stroke-opacity': Math.min(0.9, probability)
            }, svg)
          })
          const newProbability = state.bp.probabilities[index][2]
          svgElement('circle', {
            cx: point.x,
            cy: point.y,
            r: 11,
            fill: 'none',
            stroke: '#0e9f6e',
            'stroke-width': 1.5,
            'stroke-dasharray': '4 3',
            'stroke-opacity': Math.min(0.9, 0.12 + newProbability * 2.2)
          }, svg)
        })
      }
    })
    elements.bpIteration.textContent = `iteration ${state.bp.iteration}`
    const rows = [
      ['object 1 existence', state.bp.existence[0], '#2456c8'],
      ['object 2 existence', state.bp.existence[1], '#e08607'],
      ['m₉ → object 1', state.bp.probabilities[8][0], '#2456c8'],
      ['m₉ → object 2', state.bp.probabilities[8][1], '#e08607'],
      ['m₁₀ → clutter', state.bp.probabilities[9][3], '#8b929c']
    ]
    elements.bpReadout.innerHTML = rows.map(([label, value, color]) => `
      <div class="bp-row"><span>${label}</span><div class="bar"><span style="width:${value * 100}%;background:${color}"></span></div><strong>${value.toFixed(2)}</strong></div>
    `).join('')
  }

  function runBp() {
    for (let pass = 0; pass < 15; pass += 1) {
      const previous = state.bp.probabilities.map((row) => row.slice())
      stepBp()
      let maximumChange = 0
      state.bp.probabilities.forEach((row, rowIndex) => {
        row.forEach((value, columnIndex) => {
          maximumChange = Math.max(maximumChange, Math.abs(value - previous[rowIndex][columnIndex]))
        })
      })
      if (maximumChange < 0.0001) break
    }
    renderBp()
  }

  function setView(view, updateHistory = false) {
    state.view = validatedView(view)
    document.body.dataset.view = state.view
    elements.viewKicker.textContent = VIEW_COPY[state.view][0]
    elements.viewTitle.textContent = VIEW_COPY[state.view][1]
    elements.viewLinks.forEach((link) => {
      const isCurrent = link.dataset.viewLink === state.view
      if (isCurrent) link.setAttribute('aria-current', 'page')
      else link.removeAttribute('aria-current')
    })
    if (updateHistory) {
      const url = new URL(window.location.href)
      url.searchParams.set('view', state.view)
      window.history.replaceState(null, '', url)
    }
  }

  function requestBack() {
    window.parent.postMessage({ type: 'bento-live-back' }, '*')
    if (window.parent === window) window.location.href = '../'
  }

  function setPaused(paused) {
    state.paused = paused
    document.body.classList.toggle('is-paused', paused)
  }

  elements.viewLinks.forEach((link) => {
    link.addEventListener('click', (event) => {
      event.preventDefault()
      setView(link.dataset.viewLink, true)
    })
  })
  elements.backButton.addEventListener('click', (event) => {
    if (window.parent !== window) {
      event.preventDefault()
      requestBack()
    }
  })
  elements.bellM.addEventListener('input', renderPartition)
  elements.distanceThreshold.addEventListener('input', () => {
    state.showMissing = false
    renderPartition()
  })
  elements.missingButton.addEventListener('click', () => {
    state.showMissing = !state.showMissing
    renderPartition()
  })
  elements.branchButton.addEventListener('click', branchHypotheses)
  elements.pruneButton.addEventListener('click', pruneHypotheses)
  elements.hypothesisReset.addEventListener('click', resetHypotheses)
  elements.gibbsStep.addEventListener('click', () => {
    stepGibbs()
    renderGibbs()
  })
  elements.gibbsRun.addEventListener('click', () => {
    for (let move = 0; move < 100; move += 1) stepGibbs()
    renderGibbs()
  })
  elements.gibbsReset.addEventListener('click', resetGibbs)
  elements.bpStep.addEventListener('click', () => {
    stepBp()
    renderBp()
  })
  elements.bpRun.addEventListener('click', runBp)
  elements.bpReset.addEventListener('click', resetBp)

  document.addEventListener('keydown', (event) => {
    if (event.key !== 'Escape') return
    event.preventDefault()
    requestBack()
  })

  document.addEventListener('visibilitychange', () => setPaused(document.hidden))
  window.addEventListener('pagehide', () => setPaused(true))
  window.addEventListener('pageshow', () => setPaused(document.hidden))
  window.addEventListener('message', (event) => {
    if (event.source !== window.parent || event.origin !== window.location.origin || !event.data || typeof event.data !== 'object') return
    if (event.data.type === 'bento-live-pause') setPaused(true)
    if (event.data.type === 'bento-live-resume') setPaused(false)
  })

  ;[
    elements.bellM,
    elements.distanceThreshold,
    elements.missingButton,
    elements.branchButton,
    elements.pruneButton,
    elements.hypothesisReset,
    elements.gibbsRun,
    elements.gibbsStep,
    elements.gibbsReset,
    elements.bpRun,
    elements.bpStep,
    elements.bpReset
  ].forEach((control) => { control.disabled = false })

  setView(state.view)
  renderPartition()
  resetHypotheses()
  resetGibbs()
  resetBp()
})()
