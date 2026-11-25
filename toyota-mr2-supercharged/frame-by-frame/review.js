const display = document.querySelector('.display');
const report = document.querySelector('main');
display.hidden = false;
display.addEventListener('click', event => {
  const button = event.target.closest('button');
  if (!button) return;
  document.body.classList.toggle('wipe-mode', button.dataset.layout === 'wipe');
  for (const option of display.querySelectorAll('button')) {
    option.setAttribute('aria-pressed', String(option === button));
  }
});
document.addEventListener('input', event => {
  if (event.target.matches('.wiper input')) {
    event.target.closest('article').style.setProperty('--wipe', event.target.value + '%');
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
