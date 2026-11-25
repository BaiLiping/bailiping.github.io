/* Progressive enhancement: all listing content is present in the published HTML. */
(() => {
  'use strict';
  const form = document.getElementById('job-filters');
  if (!form) return;
  const cards = Array.from(document.querySelectorAll('#job-list [data-job]'));
  const search = document.getElementById('job-search');
  const category = document.getElementById('job-category');
  const type = document.getElementById('job-type');
  const region = document.getElementById('job-region');
  const status = document.getElementById('job-status');
  const sponsorship = document.getElementById('job-sponsorship');
  const access = document.getElementById('job-access');
  const addedFrom = document.getElementById('job-added-from');
  const addedThrough = document.getElementById('job-added-through');
  const sort = document.getElementById('job-sort');
  const dateError = document.getElementById('job-date-error');
  const list = document.getElementById('job-list');
  const count = document.getElementById('results-count');
  const empty = document.getElementById('no-results');
  let previousSort;

  function update() {
    const terms = search.value.trim().toLocaleLowerCase().split(/\s+/).filter(Boolean);
    const from = addedFrom?.value || '';
    const through = addedThrough?.value || '';
    const invalidRange = Boolean(from && through && from > through);
    if (dateError) dateError.hidden = !invalidRange;
    addedFrom?.setCustomValidity(invalidRange ? 'The start date must be on or before the end date.' : '');
    let visible = 0;
    cards.forEach(card => {
      const d = card.dataset;
      const hiringRegions = (d.regions || '').split(/\s+/).filter(Boolean);
      const locationMatches = region.value === 'all'
        || (region.value === 'Sweden' ? d.sweden === 'true'
          : d.region === region.value || hiringRegions.includes(region.value)
            || (region.value === 'Europe' && d.sweden === 'true'));
      const dateMatches = (!from && !through) || Boolean(d.addedDate
        && (!from || d.addedDate >= from) && (!through || d.addedDate <= through));
      const matches = !invalidRange && dateMatches && terms.every(term => d.search.includes(term))
        && (category.value === 'all' || d.category === category.value)
        && (type.value === 'all' || d.type === type.value)
        && locationMatches
        && (!sponsorship || sponsorship.value === 'all' || d.sponsorship === sponsorship.value)
        && (!access || access.value === 'all' || d.application === access.value)
        && (status.value === 'all'
          || (status.value === 'current' ? d.status !== 'closed' : d.status === status.value));
      card.hidden = !matches;
      visible += Number(matches);
    });
    const order = sort?.value || 'recommended';
    if (list && order !== previousSort) {
      const ordered = cards.map((card, index) => ({card, index, added: Date.parse(card.dataset.addedAt)}));
      if (order === 'newest' || order === 'oldest') {
        ordered.sort((a, b) => {
          const aKnown = Number.isFinite(a.added), bKnown = Number.isFinite(b.added);
          if (aKnown !== bKnown) return aKnown ? -1 : 1;
          if (aKnown && a.added !== b.added) return (order === 'newest' ? -1 : 1) * (a.added - b.added);
          return a.index - b.index;
        });
      }
      ordered.forEach(({card}) => list.appendChild(card));
      previousSort = order;
    }
    count.textContent = `${visible} of ${cards.length} ${cards.length === 1 ? 'listing' : 'listings'}`;
    empty.hidden = invalidRange || visible !== 0;
  }

  form.addEventListener('submit', event => event.preventDefault());
  form.addEventListener('input', update);
  form.addEventListener('change', update);
  form.addEventListener('reset', () => requestAnimationFrame(update));
  form.hidden = false;
  update();

  const timestamp = document.querySelector('[data-snapshot-time]');
  if (timestamp) {
    const checked = Date.parse(timestamp.dataset.snapshotTime);
    if (Number.isFinite(checked) && Date.now() - checked > 9 * 24 * 60 * 60 * 1000) {
      document.getElementById('stale-notice').hidden = false;
    }
  }
})();

