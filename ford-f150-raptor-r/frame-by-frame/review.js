const display = document.querySelector('.display');
const report = document.querySelector('main');
display.hidden = false;
function showLayout(layout) {
  document.body.classList.toggle('wipe-mode', layout === 'wipe');
  document.body.classList.toggle('shape-mode', layout === 'shape');
  if (layout === 'shape') {
    for (const trace of report.querySelectorAll('.shape-trace[data-src]')) {
      trace.src = trace.dataset.src;
      delete trace.dataset.src;
    }
  }
  for (const option of display.querySelectorAll('button')) {
    option.setAttribute('aria-pressed', String(option.dataset.layout === layout));
  }
}
function followReference() {
  if (location.hash === '#geometry-audit') {
    document.getElementById('geometry-audit').open = true;
  }
  if (location.hash === '#photo-flat-tonneau-cover') showLayout('pairs');
}
showLayout(location.hash === '#photo-flat-tonneau-cover' ? 'pairs' : 'shape');
followReference();
window.addEventListener('hashchange', followReference);
display.addEventListener('click', event => {
  const button = event.target.closest('button');
  if (button) showLayout(button.dataset.layout);
});
document.addEventListener('input', event => {
  if (event.target.matches('.wiper input')) {
    event.target.closest('article').style.setProperty('--wipe', event.target.value + '%');
  } else if (event.target.matches('.trace-control input')) {
    event.target.closest('article').style.setProperty('--trace-opacity', event.target.value / 100);
  }
});

function moveDivider(pair, clientX) {
  const bounds = pair.getBoundingClientRect();
  const input = pair.closest('article').querySelector('.wiper input');
  input.value = 100 * (clientX - bounds.left) / bounds.width;
  input.dispatchEvent(new Event('input', { bubbles: true }));
}

report.addEventListener('pointerdown', event => {
  const pair = event.target.closest('.wipe-mode .pair');
  if (!pair || !event.isPrimary || event.button !== 0) return;
  pair.setPointerCapture(event.pointerId);
  moveDivider(pair, event.clientX);
});
report.addEventListener('pointermove', event => {
  const pair = event.target.closest('.wipe-mode .pair');
  if (pair?.hasPointerCapture(event.pointerId)) moveDivider(pair, event.clientX);
});
