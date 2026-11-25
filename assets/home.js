/* Tabbed navigation for the homepage category sections.
   Progressive enhancement: without JavaScript, tabs are plain anchor
   links and every section stays visible. */
(() => {
  const tablist = document.querySelector('.category-tabs');
  const tabs = [...tablist.querySelectorAll('.category-tab')];
  const panels = tabs.map(tab => document.querySelector(tab.getAttribute('href')));

  /* Upgrade the plain anchor links to ARIA tabs and tab-panels so
     assistive technology announces them correctly.
     Takes nothing, returns nothing. */
  tablist.setAttribute('role', 'tablist');
  tablist.setAttribute('aria-orientation', 'horizontal');
  tabs.forEach((tab, index) => {
    tab.setAttribute('role', 'tab');
    tab.setAttribute('aria-controls', panels[index].id);
    panels[index].setAttribute('role', 'tabpanel');
    panels[index].setAttribute('aria-labelledby', tab.id);
    /* Panels are NOT given tabindex=0.  Every panel contains focusable
       links, so keyboard users can Tab into the content.  Omitting
       tabindex prevents the browser from focusing the panel on
       hash-fragment navigation, which drew an unwanted focus ring. */
  });

  /* Show the panel at the given index, hide every other panel, and
     optionally focus the selected tab.  Updates the URL hash when
     historyMode is 'pushState' or 'replaceState'.
     Takes index (number) and an options object { focus, historyMode }.
     Returns nothing. */
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

    /* Scroll the selected tab into view on narrow screens without
       shifting the page vertically. */
    const bounds = tab.getBoundingClientRect();
    const rail = tablist.getBoundingClientRect();
    if (bounds.left < rail.left) tablist.scrollLeft -= rail.left - bounds.left;
    if (bounds.right > rail.right) tablist.scrollLeft += bounds.right - rail.right;

    const hash = tab.getAttribute('href');
    if (historyMode && location.hash !== hash) {
      history[historyMode](null, '', hash);
    }
  }

  /* Read the current URL hash and activate the matching tab.
     Strips the legacy "group-" prefix so that old links such as
     #group-life still work.
     Takes nothing, returns nothing. */
  function restore() {
    const id = location.hash.slice(1).replace(/^group-/, '');
    const index = panels.findIndex(panel => panel.id === id);
    activate(index < 0 ? 0 : index);
  }

  /* Attach click and keyboard handlers to each tab.
     Click: selects the tab and pushes a history entry.
     Keyboard: ArrowRight/ArrowLeft cycle, Home/End jump to first/last,
     Space re-confirms the current tab. */
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

  /* Restore the correct tab when the user navigates with
     the browser Back or Forward buttons. */
  window.addEventListener('popstate', restore);
  window.addEventListener('hashchange', restore);

  /* Activate the tab matching the initial URL hash,
     or default to Work. */
  restore();
})();
