(() => {
  "use strict";
  const form = document.getElementById("publication-filters");
  const search = document.getElementById("publication-search");
  const publisher = document.getElementById("publisher-filter");
  const list = document.getElementById("publication-list");
  const count = document.getElementById("publication-count");
  const empty = document.getElementById("no-results");
  if (!form || !search || !publisher || !list || !count || !empty) return;

  const normalize = (value) => value.normalize("NFKC").toLocaleLowerCase().trim();
  const entries = Array.from(list.querySelectorAll(".publication"), (element) => ({
    element,
    text: normalize(element.textContent),
    publisher: element.dataset.publisher
  }));

  function filterPublications() {
    const words = normalize(search.value).split(/\s+/).filter(Boolean);
    let visible = 0;
    for (const entry of entries) {
      const matches = (publisher.value === "all" || entry.publisher === publisher.value)
        && words.every((word) => entry.text.includes(word));
      entry.element.hidden = !matches;
      if (matches) visible += 1;
    }
    count.textContent = visible + " of " + entries.length + " publications";
    list.hidden = visible === 0;
    empty.hidden = visible !== 0;
  }

  search.addEventListener("input", filterPublications);
  publisher.addEventListener("change", filterPublications);
  form.addEventListener("submit", (event) => event.preventDefault());
  form.addEventListener("reset", () => {
    search.value = "";
    publisher.value = "all";
    filterPublications();
  });
  form.hidden = false;
  filterPublications();
})();
