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
  const count = document.getElementById('results-count');
  const empty = document.getElementById('no-results');

  function update() {
    const terms = search.value.trim().toLocaleLowerCase().split(/\s+/).filter(Boolean);
    let visible = 0;
    cards.forEach(card => {
      const d = card.dataset;
      const matches = terms.every(term => d.search.includes(term))
        && (category.value === 'all' || d.category === category.value)
        && (type.value === 'all' || d.type === type.value)
        && (region.value === 'all' || d.region === region.value)
        && (status.value === 'all'
          || (status.value === 'current' ? d.status !== 'closed' : d.status === status.value));
      card.hidden = !matches;
      visible += Number(matches);
    });
    count.textContent = `${visible} of ${cards.length} ${cards.length === 1 ? 'listing' : 'listings'}`;
    empty.hidden = visible !== 0;
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
    if (Number.isFinite(checked) && Date.now() - checked > 48 * 60 * 60 * 1000) {
      document.getElementById('stale-notice').hidden = false;
    }
  }
})();
