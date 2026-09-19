(function () {
  'use strict';

  var root = document.documentElement;
  var embedded = false;

  try {
    embedded = window.self !== window.top;
  } catch (error) {
    embedded = true;
  }

  var embedMode = new URLSearchParams(window.location.search).get('embed');
  if (embedMode === '1' || embedMode === 'region') {
    embedded = true;
  }

  root.classList.add(embedded ? 'is-bento-embedded' : 'is-slide-standalone');
  if (embedded) root.classList.add('live-region-mode');
}());
