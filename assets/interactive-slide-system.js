(function () {
  'use strict';

  var root = document.documentElement;
  var embedded = false;

  try {
    embedded = window.self !== window.top;
  } catch (error) {
    embedded = true;
  }

  if (new URLSearchParams(window.location.search).get('embed') === '1') {
    embedded = true;
  }

  root.classList.add(embedded ? 'is-bento-embedded' : 'is-slide-standalone');
}());
