// Both engines follow the same opening timing, with offsets normalized to each engine’s maximum reach.
export function createOpening() {
  return { on: false, t: 0, hold: 1, span: 2.4, settle: 1 };
}

export function remainingOffset(opening, relativeReach) {
  const start = opening.hold + relativeReach * opening.span;
  const u = Math.min(1, Math.max(0, (opening.t - start) / opening.settle));
  return 1 - u * u * (3 - 2 * u);
}
