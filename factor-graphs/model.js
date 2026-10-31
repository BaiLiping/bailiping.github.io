(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.FactorBP = api;
})(typeof globalThis === 'object' ? globalThis : this, function () {
  'use strict';
  const defaults = Object.freeze({ a: 0.3, b: 0.6, q: 0.85 });
  const variables = ['x1', 'x2', 'x3', 'x4', 'x5'];
  const scopes = { A: ['x1'], B: ['x2'], C: ['x1', 'x2', 'x3'], D: ['x3', 'x4'], E: ['x3', 'x5'] };
  const edges = Object.entries(scopes).flatMap(([f, xs]) => xs.map(x => [f, x]));
  // Exactly the five parallel phases in Fig. 7 / Section II-D, p. 503.
  const schedule = [
    [['A', 'x1'], ['B', 'x2'], ['x4', 'D'], ['x5', 'E']],
    [['x1', 'C'], ['x2', 'C'], ['D', 'x3'], ['E', 'x3']],
    [['C', 'x3'], ['x3', 'C']],
    [['C', 'x1'], ['C', 'x2'], ['x3', 'D'], ['x3', 'E']],
    [['x1', 'A'], ['x2', 'B'], ['D', 'x4'], ['E', 'x5']]
  ];
  const key = (from, to) => `${from}>${to}`;
  const neighbors = node => scopes[node] || Object.keys(scopes).filter(f => scopes[f].includes(node));
  const normalize = v => { const z = v.reduce((s, x) => s + x, 0); return v.map(x => x / z); };
  function senderSide(from, to) {
    const nodes = new Set([from]), queue = [from];
    while (queue.length) {
      const node = queue.shift();
      for (const next of neighbors(node)) {
        if ((node === from && next === to) || nodes.has(next)) continue;
        nodes.add(next); queue.push(next);
      }
    }
    return { nodes: [...nodes], factors: Object.keys(scopes).filter(f => nodes.has(f)) };
  }
  function beliefAt(result, variable, phase) {
    const all = neighbors(variable).map(f => result.messages[key(f, variable)]);
    const received = all.filter(m => m.phase <= phase), missing = all.filter(m => m.phase > phase);
    const raw = [0, 1].map(a => received.reduce((w, m) => w * m.values[a], 1));
    const factors = [...new Set(received.flatMap(m => senderSide(m.from, m.to).factors))].sort();
    return { variable, received, missing, raw, belief: normalize(raw), factors, complete: missing.length === 0 };
  }
  function parameters(input = {}) {
    const p = { ...defaults, ...input };
    for (const name of ['a', 'b', 'q']) if (!Number.isFinite(p[name]) || p[name] < 0 || p[name] > 1) throw new RangeError(`Invalid ${name}`);
    return p;
  }
  function factor(f, x, p) {
    if (f === 'A') return x.x1 ? p.a : 1 - p.a;
    if (f === 'B') return x.x2 ? p.b : 1 - p.b;
    if (f === 'C') return x.x3 === (x.x1 ^ x.x2) ? p.q : 1 - p.q;
    if (f === 'D') return [[3, 1], [1, 2]][x.x3][x.x4];
    if (f === 'E') return [[2, 1], [1, 4]][x.x3][x.x5];
    throw new Error(`Unknown factor ${f}`);
  }
  function assignments(xs) {
    return Array.from({ length: 2 ** xs.length }, (_, n) => Object.fromEntries(xs.map((x, i) => [x, (n >> (xs.length - i - 1)) & 1])));
  }
  function message(from, to, messages, p) {
    const incoming = neighbors(from).filter(n => n !== to);
    incoming.forEach(n => { if (!messages[key(n, from)]) throw new Error(`Missing dependency ${n}>${from} for ${from}>${to}`); });
    const variable = scopes[from] ? to : from;
    const terms = [0, 1].map(value => {
      if (!scopes[from]) {
        const values = incoming.map(n => messages[key(n, from)].values[value]);
        return [{ assignment: { [variable]: value }, factorValue: 1, incoming: values, weight: values.reduce((a, b) => a * b, 1) }];
      }
      return assignments(incoming).map(other => {
        const x = { ...other, [variable]: value };
        const local = factor(from, x, p);
        const values = incoming.map(n => messages[key(n, from)].values[x[n]]);
        return { assignment: x, factorValue: local, incoming: values, weight: values.reduce((a, b) => a * b, local) };
      });
    });
    return { from, to, variable, incoming, terms, values: terms.map(rows => rows.reduce((s, r) => s + r.weight, 0)) };
  }
  function enumerate(input = {}) {
    const p = parameters(input);
    const rows = assignments(variables).map(x => ({ x, weight: Object.keys(scopes).reduce((w, f) => w * factor(f, x, p), 1) }));
    const z = rows.reduce((s, row) => s + row.weight, 0);
    if (!(z > 0)) throw new RangeError('The global function has zero mass');
    const raw = Object.fromEntries(variables.map(v => [v, [0, 1].map(a => rows.reduce((s, r) => s + (r.x[v] === a ? r.weight : 0), 0))]));
    return { z, rows, raw, beliefs: Object.fromEntries(variables.map(v => [v, normalize(raw[v])])) };
  }
  function run(input = {}) {
    const p = parameters(input), messages = {}, phases = [];
    for (const [index, phase] of schedule.entries()) {
      // Read only the previous phase; each parallel phase is dependency-safe.
      const next = phase.map(([from, to]) => ({ ...message(from, to, messages, p), phase: index + 1 }));
      next.forEach(m => { messages[key(m.from, m.to)] = m; });
      phases.push(next);
    }
    const raw = Object.fromEntries(variables.map(v => [v, [0, 1].map(a => neighbors(v).reduce((w, f) => w * messages[key(f, v)].values[a], 1))]));
    const beliefs = Object.fromEntries(variables.map(v => [v, normalize(raw[v])]));
    const readyAt = Object.fromEntries(variables.map(v => [v, Math.max(...neighbors(v).map(f => messages[key(f, v)].phase))]));
    const exact = enumerate(p);
    const maxError = Math.max(...variables.flatMap(v => beliefs[v].map((b, a) => Math.abs(b - exact.beliefs[v][a]))));
    return { p, phases, messages, raw, beliefs, readyAt, exact, maxError };
  }
  return { defaults, variables, scopes, edges, schedule, key, neighbors, normalize, senderSide, beliefAt, parameters, factor, assignments, message, enumerate, run };
});
