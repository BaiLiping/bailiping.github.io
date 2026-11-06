(() => {
  const tablist = document.querySelector('.category-tabs');
  const tabs = [...tablist.querySelectorAll('.category-tab')];
  const panels = tabs.map(tab => document.querySelector(tab.getAttribute('href')));

  // Without JavaScript these remain ordinary anchors and every section is readable.
  tablist.setAttribute('role', 'tablist');
  tablist.setAttribute('aria-orientation', 'horizontal');
  tabs.forEach((tab, index) => {
    tab.setAttribute('role', 'tab');
    tab.setAttribute('aria-controls', panels[index].id);
    panels[index].setAttribute('role', 'tabpanel');
    panels[index].setAttribute('aria-labelledby', tab.id);
    panels[index].tabIndex = 0;
  });

  function activate(index, { focus = false, historyMode } = {}) {
    const focusedPanel = panels.find(panel => panel.contains(document.activeElement));
    tabs.forEach((tab, position) => {
      const selected = position === index;
      tab.setAttribute('aria-selected', String(selected));
      tab.tabIndex = selected ? 0 : -1;
      panels[position].hidden = !selected;
    });

    const tab = tabs[index];
    if (focus || (focusedPanel && focusedPanel !== panels[index])) {
      tab.focus({ preventScroll: true });
    }

    // Reveal a selected tab on small screens without moving the page vertically.
    const bounds = tab.getBoundingClientRect();
    const rail = tablist.getBoundingClientRect();
    if (bounds.left < rail.left) tablist.scrollLeft -= rail.left - bounds.left;
    if (bounds.right > rail.right) tablist.scrollLeft += bounds.right - rail.right;

    const hash = tab.getAttribute('href');
    if (historyMode && location.hash !== hash) {
      history[historyMode](null, '', hash);
    }
  }

  function restore() {
    const id = location.hash.slice(1).replace(/^group-/, '');
    const index = panels.findIndex(panel => panel.id === id);
    activate(index < 0 ? 0 : index);
  }

  tabs.forEach((tab, index) => {
    tab.addEventListener('click', event => {
      if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
      event.preventDefault();
      activate(index, { historyMode: 'pushState' });
    });

    tab.addEventListener('keydown', event => {
      let next;
      if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
      if (event.key === 'ArrowLeft') next = (index - 1 + tabs.length) % tabs.length;
      if (event.key === 'Home') next = 0;
      if (event.key === 'End') next = tabs.length - 1;
      if (event.key === ' ') next = index;
      if (next === undefined) return;
      event.preventDefault();
      activate(next, { focus: true, historyMode: 'replaceState' });
    });
  });

  window.addEventListener('popstate', restore);
  window.addEventListener('hashchange', restore);
  restore();
})();
