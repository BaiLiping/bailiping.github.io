(function () {
  const counterNodes = Array.from(document.querySelectorAll("[data-visitor-counter]"));
  if (!counterNodes.length) {
    return;
  }

  const apiBase = "https://api.counterapi.dev/v1/bailiping-com/unique-visitors";
  const productionHosts = ["bailiping.com", "www.bailiping.com"];
  const storageKey = "bailiping.com.uniqueVisitorCounted.v1";
  const countedValue = "1";
  const storage = getStorage();

  function getStorage() {
    const testKey = "__visitor_counter_test__";
    try {
      window.localStorage.setItem(testKey, testKey);
      window.localStorage.removeItem(testKey);
      return window.localStorage;
    } catch (error) {
      return null;
    }
  }

  function hasCounted() {
    return storage ? storage.getItem(storageKey) === countedValue : false;
  }

  function markCounted() {
    if (storage) {
      storage.setItem(storageKey, countedValue);
    }
  }

  function setCounterState(state, value) {
    counterNodes.forEach((node) => {
      node.dataset.state = state;
      const valueNode = node.querySelector("[data-visitor-count]");
      if (valueNode) {
        valueNode.textContent = value;
      }
    });
  }

  function formatCount(value) {
    const count = Number(value);
    if (!Number.isFinite(count)) {
      return "Unavailable";
    }
    return count.toLocaleString("en-US");
  }

  async function fetchCounter(url) {
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Visitor counter request failed: ${response.status}`);
    }
    return response.json();
  }

  async function updateCounter() {
    setCounterState("loading", "...");

    const shouldIncrement =
      productionHosts.includes(window.location.hostname) && storage && !hasCounted();
    const url = shouldIncrement ? `${apiBase}/up` : `${apiBase}/`;
    const data = await fetchCounter(url);

    if (shouldIncrement) {
      markCounted();
    }

    setCounterState("ready", formatCount(data.count));
  }

  updateCounter().catch((error) => {
    console.warn(error);
    setCounterState("error", "Unavailable");
  });
})();
