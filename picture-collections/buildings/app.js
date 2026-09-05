(() => {
  "use strict";

  const tiles = Array.from(document.querySelectorAll(".tile"));
  const lightbox = document.getElementById("lightbox");
  const stage = document.getElementById("lightboxStage");
  const caption = document.getElementById("lightboxCaption");
  const closeButton = document.getElementById("lightboxClose");
  const previousButton = document.getElementById("lightboxPrevious");
  const nextButton = document.getElementById("lightboxNext");

  let currentIndex = -1;
  let returnFocus = null;
  let touchStartX = 0;

  function preload(index) {
    const tile = tiles[(index + tiles.length) % tiles.length];
    if (!tile) return;
    const image = new Image();
    image.src = tile.href;
  }

  function show(index) {
    if (!tiles.length) return;

    const isOpening = lightbox.hidden;
    currentIndex = (index + tiles.length) % tiles.length;
    const tile = tiles[currentIndex];
    const image = document.createElement("img");

    image.className = "lightbox-image";
    image.src = tile.href;
    image.alt = tile.querySelector("img").alt;
    image.decoding = "async";

    stage.replaceChildren(image);
    caption.textContent = `${tile.dataset.collection} — ${tile.dataset.position}`;
    lightbox.hidden = false;
    lightbox.setAttribute("aria-hidden", "false");
    document.body.classList.add("lightbox-open");
    if (isOpening) closeButton.focus({ preventScroll: true });

    preload(currentIndex - 1);
    preload(currentIndex + 1);
  }

  function hide() {
    lightbox.hidden = true;
    lightbox.setAttribute("aria-hidden", "true");
    stage.replaceChildren();
    document.body.classList.remove("lightbox-open");
    currentIndex = -1;

    if (returnFocus) returnFocus.focus({ preventScroll: true });
  }

  function previous() {
    show(currentIndex - 1);
  }

  function next() {
    show(currentIndex + 1);
  }

  tiles.forEach((tile, index) => {
    tile.addEventListener("click", (event) => {
      event.preventDefault();
      returnFocus = tile;
      show(index);
    });
  });

  closeButton.addEventListener("click", hide);
  previousButton.addEventListener("click", previous);
  nextButton.addEventListener("click", next);

  lightbox.addEventListener("click", (event) => {
    if (event.target === lightbox || event.target === stage) hide();
  });

  lightbox.addEventListener("touchstart", (event) => {
    touchStartX = event.changedTouches[0].clientX;
  }, { passive: true });

  lightbox.addEventListener("touchend", (event) => {
    const distance = event.changedTouches[0].clientX - touchStartX;
    if (Math.abs(distance) < 50) return;
    if (distance > 0) previous();
    else next();
  }, { passive: true });

  document.addEventListener("keydown", (event) => {
    if (lightbox.hidden) return;

    if (event.key === "Escape") hide();
    else if (event.key === "ArrowLeft") previous();
    else if (event.key === "ArrowRight") next();
    else if (event.key === "Tab") {
      const controls = [closeButton, previousButton, nextButton];
      const activeIndex = controls.indexOf(document.activeElement);
      const direction = event.shiftKey ? -1 : 1;
      const nextIndex = (activeIndex + direction + controls.length) % controls.length;
      event.preventDefault();
      controls[nextIndex].focus();
    }
  });
})();
