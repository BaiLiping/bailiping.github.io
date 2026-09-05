(() => {
  'use strict'

  const MAX_FRAME = 100
  const FRAME_MS = 125
  const SCALE = 1.9
  const ORIGIN_X = 255
  const ORIGIN_Y = 236
  const FOV_RADIUS = 120
  const requestedView = new URLSearchParams(window.location.search).get('view')
  const initialView = requestedView === 'timeline' ? 'timeline' : 'rule'

  const stations = {
    BS1: { id: 'BS1', x: 0, y: 0 },
    BS2: { id: 'BS2', x: 150, y: 0 }
  }

  const targets = [
    { id: 'A', tx: 'BS1', rx: 'BS2', phase: 0.15 },
    { id: 'B', tx: 'BS2', rx: 'BS1', phase: 1.25 }
  ]

  const elements = {
    appShell: document.getElementById('appShell'),
    backButton: document.getElementById('backButton'),
    playButton: document.getElementById('playButton'),
    playLabel: document.querySelector('#playButton span'),
    stepButton: document.getElementById('stepButton'),
    eventButton: document.getElementById('eventButton'),
    timeline: document.getElementById('timeline'),
    timeOutput: document.getElementById('timeOutput'),
    priorMode: document.getElementById('priorMode'),
    measurementMode: document.getElementById('measurementMode'),
    modeHint: document.getElementById('modeHint'),
    existenceThreshold: document.getElementById('existenceThreshold'),
    visibilityThreshold: document.getElementById('visibilityThreshold'),
    existenceOutput: document.getElementById('existenceOutput'),
    visibilityOutput: document.getElementById('visibilityOutput'),
    gateSummary: document.getElementById('gateSummary'),
    priorCount: document.getElementById('priorCount'),
    measurementCount: document.getElementById('measurementCount'),
    trackCount: document.getElementById('trackCount'),
    tradeoffNote: document.getElementById('tradeoffNote'),
    sceneCallout: document.getElementById('sceneCallout'),
    eventText: document.getElementById('eventText')
  }

  const targetElements = Object.fromEntries(targets.map((target) => [target.id, {
    target: document.getElementById(`target${target.id}`),
    targetLabel: document.querySelector(`#target${target.id} .target-label text`),
    transfer: document.getElementById(`transfer${target.id}`),
    transferLabel: document.querySelector(`#transfer${target.id} .packet text`),
    measurement: document.getElementById(`measurement${target.id}`),
    measurementLink: document.querySelector(`#measurement${target.id} .measurement-link`),
    measurementMark: document.querySelector(`#measurement${target.id} .measurement-mark`),
    estimate: document.getElementById(`estimate${target.id}`),
    belief: document.querySelector(`#estimate${target.id} .belief`),
    estimateMark: document.querySelector(`#estimate${target.id} .estimate-mark`),
    gateRow: document.getElementById(`gate${target.id}`),
    existence: document.getElementById(`exist${target.id}`),
    visibility: document.getElementById(`visible${target.id}`),
    existenceBar: document.getElementById(`existBar${target.id}`),
    visibilityBar: document.getElementById(`visibleBar${target.id}`),
    gateState: document.getElementById(`gateState${target.id}`)
  }]))

  const state = {
    frame: 0,
    playIntent: false,
    raf: 0,
    lastTick: 0,
    parentPaused: false,
    pageHidden: document.hidden,
    offscreen: false,
    events: [],
    eventByTarget: new Map(),
    lastAnnouncement: ''
  }

  function clamp(value, min = 0, max = 1) {
    return Math.min(max, Math.max(min, value))
  }

  function positionAt(target, frame) {
    const angle = frame / MAX_FRAME * Math.PI * 2
    if (target.id === 'A') {
      return {
        x: -100 + 3.5 * frame,
        y: -15 + 10 * Math.sin(angle - 0.55)
      }
    }
    return {
      x: 250 - 3.2 * frame,
      y: 15 - 10 * Math.sin(angle + 0.35)
    }
  }

  function screenPoint(position) {
    return {
      x: ORIGIN_X + SCALE * position.x,
      y: ORIGIN_Y - SCALE * position.y
    }
  }

  function distance(a, b) {
    return Math.hypot(a.x - b.x, a.y - b.y)
  }

  // Illustrative smooth finite-FoV visibility curve, not the paper's exact
  // detection model. At 120 m E[pD] is about 0.47 and approaches 0.93 inside.
  function expectedDetection(station, position) {
    const radiusOffset = (distance(station, position) - FOV_RADIUS) / 8
    return 0.93 / (1 + Math.exp(radiusOffset))
  }

  function existenceAt(target, frame) {
    const ownPosition = positionAt(target, frame)
    const other = targets.find((candidate) => candidate.id !== target.id)
    const otherPosition = positionAt(other, frame)
    const separation = distance(ownPosition, otherPosition)
    const ambiguityPenalty = 0.105 * Math.exp(-Math.pow(separation / 34, 2))
    const txDetection = expectedDetection(stations[target.tx], ownPosition)
    const edgePenalty = 0.14 * clamp((0.58 - txDetection) / 0.58)
    const deterministicRipple = 0.012 * Math.sin(frame * 0.17 + target.phase)
    return clamp(0.955 - ambiguityPenalty - edgePenalty + deterministicRipple, 0.05, 0.975)
  }

  function thresholds() {
    return {
      existence: Number(elements.existenceThreshold.value),
      visibility: Number(elements.visibilityThreshold.value)
    }
  }

  function gateAt(target, frame) {
    const values = thresholds()
    const position = positionAt(target, frame)
    const existence = existenceAt(target, frame)
    const visibility = expectedDetection(stations[target.rx], position)
    const transmitterVisibility = expectedDetection(stations[target.tx], position)
    const existenceOpen = existence > values.existence
    const visibilityOpen = visibility > values.visibility

    return {
      existence,
      visibility,
      transmitterVisibility,
      existenceOpen,
      visibilityOpen,
      open: existenceOpen && visibilityOpen
    }
  }

  function rebuildEvents() {
    state.events = []
    state.eventByTarget.clear()

    targets.forEach((target) => {
      for (let frame = 0; frame <= MAX_FRAME; frame += 1) {
        if (gateAt(target, frame).open) {
          const event = { frame, target }
          state.events.push(event)
          state.eventByTarget.set(target.id, event)
          break
        }
      }
    })

    state.events.sort((left, right) => left.frame - right.frame || left.target.id.localeCompare(right.target.id))
    elements.eventButton.disabled = state.events.length === 0
    elements.eventButton.title = state.events.length === 0
      ? 'No handover meets the current thresholds'
      : 'Jump to the next handover event'
  }

  function measurementEnabled() {
    return elements.measurementMode.checked
  }

  function measurementIsAvailable(target, frame) {
    const event = state.eventByTarget.get(target.id)
    if (!event || frame < event.frame) return false
    const gate = gateAt(target, frame)
    return gate.open && gate.transmitterVisibility >= 0.24
  }

  function countMeasurements(frame) {
    if (!measurementEnabled()) return 0

    return targets.reduce((total, target) => {
      const event = state.eventByTarget.get(target.id)
      if (!event || frame < event.frame) return total

      let count = 0
      for (let current = event.frame; current <= frame; current += 1) {
        if (measurementIsAvailable(target, current)) count += 1
      }
      return total + count
    }, 0)
  }

  function deterministicMeasurement(target, frame) {
    const truth = positionAt(target, frame)
    const phase = target.id === 'A' ? 0.7 : 1.8
    return {
      x: truth.x + 2.8 * Math.sin(frame * 0.71 + phase),
      y: truth.y + 2.3 * Math.cos(frame * 0.53 + phase)
    }
  }

  function receiverEstimate(target, frame, eventFrame) {
    const truth = positionAt(target, frame)
    const age = Math.max(0, frame - eventFrame)
    const withMeasurement = measurementEnabled()
    const amplitude = withMeasurement
      ? 2.8 + 4.2 * Math.exp(-age / 7)
      : 8.5 + 9.5 * Math.exp(-age / 13)
    const phase = target.id === 'A' ? 0.4 : 2.1

    return {
      x: truth.x + amplitude * Math.sin(frame * 0.23 + phase),
      y: truth.y + amplitude * 0.65 * Math.cos(frame * 0.19 + phase),
      radiusX: withMeasurement ? 9 + 11 * Math.exp(-age / 8) : 17 + 20 * Math.exp(-age / 15),
      radiusY: withMeasurement ? 6 + 7 * Math.exp(-age / 8) : 10 + 13 * Math.exp(-age / 15)
    }
  }

  function setTransform(element, point) {
    element.setAttribute('transform', `translate(${point.x.toFixed(2)} ${point.y.toFixed(2)})`)
  }

  function setPath(element, from, to) {
    element.setAttribute('d', `M${from.x.toFixed(2)} ${from.y.toFixed(2)}L${to.x.toFixed(2)} ${to.y.toFixed(2)}`)
  }

  function renderTarget(target) {
    const dom = targetElements[target.id]
    const position = positionAt(target, state.frame)
    const point = screenPoint(position)
    const event = state.eventByTarget.get(target.id)
    const hasTransferred = Boolean(event && state.frame >= event.frame)
    const isEventFrame = Boolean(event && state.frame >= event.frame && state.frame <= event.frame + 2)

    setTransform(dom.target, point)
    dom.targetLabel.textContent = `${target.id} · ${hasTransferred ? target.rx : target.tx}`

    dom.transfer.classList.toggle('is-hidden', !isEventFrame)
    dom.transfer.classList.toggle('is-active', isEventFrame)
    dom.transfer.setAttribute('aria-hidden', String(!isEventFrame))
    dom.transferLabel.textContent = measurementEnabled() ? 'PRIOR + z*' : 'PRIOR'

    if (hasTransferred) {
      const estimate = receiverEstimate(target, state.frame, event.frame)
      const estimatePoint = screenPoint(estimate)
      dom.estimate.classList.remove('is-hidden')
      dom.estimate.setAttribute('aria-hidden', 'false')
      dom.belief.setAttribute('cx', estimatePoint.x.toFixed(2))
      dom.belief.setAttribute('cy', estimatePoint.y.toFixed(2))
      dom.belief.setAttribute('rx', (estimate.radiusX * SCALE).toFixed(2))
      dom.belief.setAttribute('ry', (estimate.radiusY * SCALE).toFixed(2))
      setTransform(dom.estimateMark, estimatePoint)
    } else {
      dom.estimate.classList.add('is-hidden')
      dom.estimate.setAttribute('aria-hidden', 'true')
    }

    const showMeasurement = measurementEnabled() && measurementIsAvailable(target, state.frame)
    if (showMeasurement) {
      const measurementPoint = screenPoint(deterministicMeasurement(target, state.frame))
      dom.measurement.classList.remove('is-hidden')
      dom.measurement.setAttribute('aria-hidden', 'false')
      setPath(dom.measurementLink, point, measurementPoint)
      setTransform(dom.measurementMark, measurementPoint)
    } else {
      dom.measurement.classList.add('is-hidden')
      dom.measurement.setAttribute('aria-hidden', 'true')
    }
  }

  function renderGate(target) {
    const dom = targetElements[target.id]
    const gate = gateAt(target, state.frame)
    const event = state.eventByTarget.get(target.id)
    const alreadySent = Boolean(event && state.frame >= event.frame)

    dom.existence.textContent = gate.existence.toFixed(2)
    dom.visibility.textContent = gate.visibility.toFixed(2)
    dom.existenceBar.style.width = `${(gate.existence * 100).toFixed(1)}%`
    dom.visibilityBar.style.width = `${(gate.visibility * 100).toFixed(1)}%`
    dom.existenceBar.style.background = gate.existenceOpen ? 'var(--green)' : 'var(--danger)'
    dom.visibilityBar.style.background = gate.visibilityOpen ? 'var(--green)' : 'var(--danger)'
    dom.gateRow.classList.toggle('is-open', gate.open)
    dom.gateState.textContent = alreadySent ? 'SENT' : (gate.open ? 'SEND' : 'WAIT')
    dom.gateState.title = gate.open
      ? `Both gates are open for target ${target.id}`
      : `Target ${target.id} is waiting for ${[
          gate.existenceOpen ? '' : 'existence',
          gate.visibilityOpen ? '' : 'receiver visibility'
        ].filter(Boolean).join(' and ')}`

    return Number(gate.existenceOpen) + Number(gate.visibilityOpen)
  }

  function eventAnnouncement() {
    const eventsNow = state.events.filter((event) => event.frame === state.frame)
    if (eventsNow.length > 0) {
      const names = eventsNow.map((event) => `Target ${event.target.id}`).join(' and ')
      const payload = measurementEnabled() ? 'a prior and one associated measurement' : 'one prior'
      return `${names}: both gates open. Send ${payload}; the receiver seeds a declared track.`
    }

    const futureEvent = state.events.find((event) => event.frame > state.frame)
    if (futureEvent) {
      return `Next: Target ${futureEvent.target.id} passes both gates at t = ${futureEvent.frame} s.`
    }

    if (state.events.length === 0) {
      return 'No target passes both gates at these thresholds. Lower Pₜₕ or Γ to recover a handover window.'
    }

    const sentCount = state.events.filter((event) => event.frame <= state.frame).length
    if (sentCount === state.events.length) {
      return `All ${sentCount} available priors were sent exactly once; receiver tracks continue locally.`
    }

    return 'Ready — press Play or jump to the next handover.'
  }

  function renderCounters() {
    const sent = state.events.filter((event) => event.frame <= state.frame).length
    elements.priorCount.textContent = `${sent} / 2`
    elements.trackCount.textContent = `${sent} / 2`
    elements.measurementCount.textContent = String(countMeasurements(state.frame))
  }

  function renderTradeoff() {
    const values = thresholds()
    const missingTargets = 2 - state.events.length
    let visibilityText
    let existenceText

    if (values.visibility < 0.4) {
      visibilityText = 'Low Γ permits an earlier boundary handover, increasing exposure to weak-visibility false tracks.'
    } else if (values.visibility > 0.7) {
      visibilityText = 'High Γ waits for stronger receiver visibility, shortening overlap support and increasing missed-track risk.'
    } else if (Math.abs(values.visibility - 0.5) < 0.005) {
      visibilityText = 'Γ = 0.50 is the paper’s simulated operating point: credible receiver visibility without waiting for the FoV centre.'
    } else {
      visibilityText = 'Moderate Γ balances early boundary handover against waiting too deep into the receiving FoV.'
    }

    if (values.existence > 0.93) {
      existenceText = ' The strict existence gate can delay or suppress a transfer during ambiguous association.'
    } else if (values.existence > 0.7) {
      existenceText = ' The stronger declaration gate rejects lower-confidence tracks but narrows the transfer window.'
    } else {
      existenceText = ' Pₜₕ = 0.50 matches the deck’s declared-track threshold.'
    }

    const missingText = missingTargets > 0
      ? ` With these settings, ${missingTargets === 2 ? 'neither target' : 'one target'} finds a valid window.`
      : ''

    elements.tradeoffNote.textContent = visibilityText + existenceText + missingText
  }

  function renderTransport() {
    elements.timeline.value = String(state.frame)
    elements.timeline.style.setProperty('--timeline-progress', `${state.frame}%`)
    elements.timeline.setAttribute('aria-valuetext', `${state.frame} seconds`)
    elements.timeOutput.textContent = `t = ${state.frame} s`
    elements.playButton.setAttribute('aria-pressed', String(state.playIntent && !isSuspended()))
    elements.playLabel.textContent = state.playIntent && !isSuspended()
      ? 'Pause'
      : (state.frame >= MAX_FRAME ? 'Replay' : 'Play')
  }

  function render() {
    let openGateCount = 0
    targets.forEach((target) => {
      renderTarget(target)
      openGateCount += renderGate(target)
    })

    elements.gateSummary.textContent = `${openGateCount} of 4 open`
    renderCounters()
    renderTradeoff()
    renderTransport()

    const announcement = eventAnnouncement()
    if (announcement !== state.lastAnnouncement) {
      elements.eventText.textContent = announcement
      state.lastAnnouncement = announcement
    }
    const isEvent = state.events.some((event) => event.frame === state.frame)
    elements.sceneCallout.classList.toggle('is-event', isEvent)
  }

  function isSuspended() {
    return state.parentPaused || state.pageHidden || state.offscreen
  }

  function animationTick(timestamp) {
    if (!state.playIntent || isSuspended()) {
      state.raf = 0
      state.lastTick = 0
      renderTransport()
      return
    }

    if (!state.lastTick) state.lastTick = timestamp
    const elapsed = timestamp - state.lastTick
    if (elapsed >= FRAME_MS) {
      const frames = Math.max(1, Math.floor(elapsed / FRAME_MS))
      state.lastTick += frames * FRAME_MS
      state.frame = Math.min(MAX_FRAME, state.frame + frames)
      if (state.frame >= MAX_FRAME) state.playIntent = false
      render()
    }

    if (state.playIntent && !isSuspended()) {
      state.raf = window.requestAnimationFrame(animationTick)
    } else {
      state.raf = 0
      state.lastTick = 0
      renderTransport()
    }
  }

  function syncPlayback() {
    const shouldRun = state.playIntent && !isSuspended()
    if (shouldRun && !state.raf) {
      state.lastTick = 0
      state.raf = window.requestAnimationFrame(animationTick)
    } else if (!shouldRun && state.raf) {
      window.cancelAnimationFrame(state.raf)
      state.raf = 0
      state.lastTick = 0
    }
    renderTransport()
  }

  function setFrame(frame, pause = true) {
    if (pause) state.playIntent = false
    state.frame = clamp(Math.round(frame), 0, MAX_FRAME)
    state.lastAnnouncement = ''
    syncPlayback()
    render()
  }

  function togglePlayback() {
    if (state.playIntent) {
      state.playIntent = false
    } else {
      if (state.frame >= MAX_FRAME) state.frame = 0
      state.playIntent = true
    }
    syncPlayback()
    render()
  }

  function jumpToNextEvent() {
    if (state.events.length === 0) return
    let next = state.events.find((event) => event.frame > state.frame)
    if (!next) next = state.events[0]
    setFrame(next.frame)
  }

  function updateMode() {
    elements.modeHint.textContent = measurementEnabled()
      ? 'The prior seeds the receiver track; the most likely associated measurement tightens the illustrated position belief.'
      : 'The prior seeds a declared track at the receiver; uncertainty contracts through later local updates.'
    state.lastAnnouncement = ''
    render()
  }

  function updateThresholds() {
    elements.existenceOutput.value = Number(elements.existenceThreshold.value).toFixed(2)
    elements.visibilityOutput.value = Number(elements.visibilityThreshold.value).toFixed(2)
    rebuildEvents()
    state.lastAnnouncement = ''
    render()
  }

  function requestBack() {
    state.playIntent = false
    syncPlayback()
    window.parent.postMessage({ type: 'bento-live-back' }, '*')
    if (window.parent === window) window.location.href = '../'
  }

  function isTypingTarget(target) {
    if (!(target instanceof Element)) return false
    return Boolean(target.closest('input, button, a, select, textarea, [contenteditable="true"]'))
  }

  function enableControls() {
    [
      elements.playButton,
      elements.stepButton,
      elements.timeline,
      elements.priorMode,
      elements.measurementMode,
      elements.existenceThreshold,
      elements.visibilityThreshold
    ].forEach((control) => { control.disabled = false })
  }

  elements.playButton.addEventListener('click', togglePlayback)
  elements.stepButton.addEventListener('click', () => setFrame(Math.min(MAX_FRAME, state.frame + 1)))
  elements.eventButton.addEventListener('click', jumpToNextEvent)
  elements.timeline.addEventListener('input', () => setFrame(Number(elements.timeline.value)))
  elements.priorMode.addEventListener('change', updateMode)
  elements.measurementMode.addEventListener('change', updateMode)
  elements.existenceThreshold.addEventListener('input', updateThresholds)
  elements.visibilityThreshold.addEventListener('input', updateThresholds)

  elements.backButton.addEventListener('click', (event) => {
    if (window.parent !== window) {
      event.preventDefault()
      requestBack()
    }
  })

  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') {
      event.preventDefault()
      requestBack()
      return
    }

    if (isTypingTarget(event.target)) return

    if (event.code === 'Space') {
      event.preventDefault()
      togglePlayback()
    } else if (event.key === 'ArrowRight') {
      event.preventDefault()
      setFrame(Math.min(MAX_FRAME, state.frame + 1))
    } else if (event.key === 'ArrowLeft') {
      event.preventDefault()
      setFrame(Math.max(0, state.frame - 1))
    }
  })

  document.addEventListener('visibilitychange', () => {
    state.pageHidden = document.hidden
    syncPlayback()
  })

  window.addEventListener('pagehide', () => {
    state.pageHidden = true
    syncPlayback()
  })

  window.addEventListener('pageshow', () => {
    state.pageHidden = document.hidden
    syncPlayback()
  })

  window.addEventListener('message', (event) => {
    if (event.source !== window.parent || !event.data || typeof event.data !== 'object') return
    if (event.data.type === 'bento-live-pause') {
      state.parentPaused = true
      syncPlayback()
    } else if (event.data.type === 'bento-live-resume') {
      state.parentPaused = false
      syncPlayback()
    }
  })

  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver((entries) => {
      const entry = entries[0]
      state.offscreen = !entry.isIntersecting || entry.intersectionRatio < 0.05
      syncPlayback()
    }, { threshold: [0, 0.05, 0.25] })
    observer.observe(elements.appShell)
  }

  enableControls()
  elements.existenceOutput.value = Number(elements.existenceThreshold.value).toFixed(2)
  elements.visibilityOutput.value = Number(elements.visibilityThreshold.value).toFixed(2)
  rebuildEvents()
  if (initialView === 'timeline' && state.events.length > 0) {
    state.frame = Math.max(0, state.events[0].frame - 5)
  }
  render()
})()
