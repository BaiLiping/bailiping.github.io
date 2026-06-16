(function () {
  const data = window.SALE_DATA || { sellers: [] };
  const sellers = Array.isArray(data.sellers) ? data.sellers : [];
  const statusLabels = {
    available: "Available",
    reserved: "Reserved",
    sold: "Sold"
  };

  const sellerGrid = document.querySelector("#sellerGrid");
  const itemGrid = document.querySelector("#itemGrid");
  const emptyState = document.querySelector("#emptyState");
  const summaryTotal = document.querySelector("#summaryTotal");
  const summaryAvailable = document.querySelector("#summaryAvailable");
  const contactPanel = document.querySelector("#contactPanel");
  const filterButtons = Array.from(document.querySelectorAll(".filter-button"));
  const searchInput = document.querySelector("#searchInput");
  const dialog = document.querySelector("#itemDialog");
  const dialogClose = document.querySelector("#dialogClose");
  const dialogImage = document.querySelector("#dialogImage");
  const dialogTitle = document.querySelector("#dialogTitle");
  const dialogSeller = document.querySelector("#dialogSeller");
  const dialogStatus = document.querySelector("#dialogStatus");
  const dialogPrice = document.querySelector("#dialogPrice");
  const dialogDescription = document.querySelector("#dialogDescription");
  const dialogNotes = document.querySelector("#dialogNotes");
  const galleryPrev = document.querySelector("#galleryPrev");
  const galleryNext = document.querySelector("#galleryNext");
  const imageCounter = document.querySelector("#imageCounter");
  const thumbStrip = document.querySelector("#thumbStrip");

  let activeSellerId = sellers[0] ? sellers[0].id : "";
  let activeFilter = "all";
  let searchTerm = "";
  let currentItems = [];
  let galleryImages = [];
  let galleryIndex = 0;

  function getActiveSeller() {
    return sellers.find((seller) => seller.id === activeSellerId) || sellers[0] || null;
  }

  function getSellerItems(seller) {
    return Array.isArray(seller && seller.items) ? seller.items : [];
  }

  function getVisibleItems() {
    const seller = getActiveSeller();
    const items = getSellerItems(seller);
    const normalizedSearch = searchTerm.trim().toLowerCase();

    return items.filter((item) => {
      const statusMatch = activeFilter === "all" || item.status === activeFilter;
      if (!statusMatch) {
        return false;
      }

      if (!normalizedSearch) {
        return true;
      }

      return [
        item.title,
        item.category,
        item.condition,
        item.description,
        item.notes,
        seller && seller.name
      ]
        .filter(Boolean)
        .join(" ")
        .toLowerCase()
        .includes(normalizedSearch);
    });
  }

  function statusClass(status) {
    return status === "reserved" || status === "sold" ? status : "available";
  }

  function statusLabel(status) {
    return statusLabels[status] || "Available";
  }

  function itemImages(item) {
    return Array.isArray(item.images) ? item.images.filter(Boolean) : [];
  }

  function render() {
    renderSellers();
    renderSummary();
    renderContact();
    renderItems();
  }

  function renderSellers() {
    sellerGrid.innerHTML = "";

    sellers.forEach((seller) => {
      const items = getSellerItems(seller);
      const available = items.filter((item) => item.status === "available").length;
      const isActive = seller.id === activeSellerId;
      const button = document.createElement("button");
      button.type = "button";
      button.role = "tab";
      button.className = `seller-card${isActive ? " is-active" : ""}`;
      button.setAttribute("aria-selected", String(isActive));
      button.innerHTML = `
        <span class="seller-card-kicker">${escapeHtml(seller.name || "Seller")}</span>
        <strong>${escapeHtml(seller.location || "Pickup location to be confirmed")}</strong>
        <span>${items.length} items · ${available} available</span>
      `;
      button.addEventListener("click", () => {
        activeSellerId = seller.id;
        render();
      });
      sellerGrid.appendChild(button);
    });
  }

  function renderSummary() {
    const seller = getActiveSeller();
    const items = getSellerItems(seller);
    const available = items.filter((item) => item.status === "available").length;
    summaryTotal.textContent = `${items.length} item${items.length === 1 ? "" : "s"}`;
    summaryAvailable.textContent = `${available} available`;
  }

  function renderContact() {
    const seller = getActiveSeller();
    const contact = data.contact || {};

    if (!seller) {
      contactPanel.hidden = true;
      contactPanel.textContent = "";
      return;
    }

    const rows = [
      seller.location ? `Pickup: ${seller.location}` : "",
      seller.email ? `Email: ${seller.email}` : "",
      seller.phone ? `Phone: ${seller.phone}` : "",
      contact.note || ""
    ].filter(Boolean);

    contactPanel.hidden = false;
    contactPanel.innerHTML = `
      <div class="contact-copy">
        <p class="contact-kicker">Contact</p>
        <h3>${escapeHtml(contact.title || "Interested in an item?")}</h3>
        <p class="contact-text">${escapeHtml(contact.message || "Contact the seller with the item name.")}</p>
        ${rows.length ? `<div class="contact-meta">${rows.map((row) => `<p>${escapeHtml(row)}</p>`).join("")}</div>` : ""}
      </div>
      <div class="contact-action">
        ${seller.email ? `<a class="primary-action" href="mailto:${escapeAttr(seller.email)}">Email seller</a>` : ""}
        ${seller.phone ? `<a class="secondary-action" href="tel:${escapeAttr(seller.phone)}">Call seller</a>` : ""}
      </div>
    `;
  }

  function renderItems() {
    const seller = getActiveSeller();
    currentItems = getVisibleItems();
    itemGrid.innerHTML = "";
    emptyState.hidden = currentItems.length > 0;

    currentItems.forEach((item, index) => {
      const images = itemImages(item);
      const image = images[0] || "";
      const status = statusClass(item.status);
      const card = document.createElement("article");
      card.className = `item-card status-${status}`;
      card.tabIndex = 0;
      card.role = "button";
      card.setAttribute("aria-label", `View details for ${item.title}`);
      card.innerHTML = `
        <div class="card-image-wrap">
          ${image ? `<img class="card-image" src="${escapeAttr(image)}" alt="${escapeAttr(item.title)}" loading="lazy">` : `<div class="image-fallback">No image</div>`}
          <span class="status-badge status-${status}">${statusLabel(item.status)}</span>
        </div>
        <div class="card-body">
          <div class="card-title-row">
            <h2>${escapeHtml(item.title || "Untitled item")}</h2>
            <span class="price">${escapeHtml(item.price || "Price TBC")}</span>
          </div>
          <p class="description">${escapeHtml(item.description || "")}</p>
          <div class="card-meta">
            ${item.category ? `<span>${escapeHtml(item.category)}</span>` : ""}
            ${item.condition ? `<span>${escapeHtml(item.condition)}</span>` : ""}
            ${seller && seller.location ? `<span>${escapeHtml(seller.location)}</span>` : ""}
          </div>
        </div>
      `;
      card.addEventListener("click", () => openItem(index));
      card.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          openItem(index);
        }
      });
      itemGrid.appendChild(card);
    });
  }

  function openItem(index) {
    const item = currentItems[index];
    const seller = getActiveSeller();
    if (!item) {
      return;
    }

    galleryImages = itemImages(item);
    galleryIndex = 0;
    dialogTitle.textContent = item.title || "Untitled item";
    dialogSeller.textContent = seller ? seller.name : "";
    dialogPrice.textContent = item.price || "Price TBC";
    dialogDescription.textContent = item.description || "";
    dialogNotes.textContent = item.notes || "";
    dialogNotes.hidden = !item.notes;
    dialogStatus.className = `status-badge status-${statusClass(item.status)}`;
    dialogStatus.textContent = statusLabel(item.status);
    renderGallery();

    if (typeof dialog.showModal === "function") {
      dialog.showModal();
    } else {
      dialog.setAttribute("open", "");
    }
  }

  function renderGallery() {
    const hasManyImages = galleryImages.length > 1;
    const image = galleryImages[galleryIndex] || "";
    dialogImage.src = image;
    dialogImage.alt = dialogTitle.textContent || "Item image";
    dialogImage.hidden = !image;
    galleryPrev.hidden = !hasManyImages;
    galleryNext.hidden = !hasManyImages;
    imageCounter.hidden = !hasManyImages;
    imageCounter.textContent = hasManyImages ? `${galleryIndex + 1} / ${galleryImages.length}` : "";
    thumbStrip.innerHTML = "";

    galleryImages.forEach((src, index) => {
      const thumb = document.createElement("button");
      thumb.type = "button";
      thumb.className = `thumb${index === galleryIndex ? " is-active" : ""}`;
      thumb.setAttribute("aria-label", `Show image ${index + 1}`);
      thumb.innerHTML = `<img src="${escapeAttr(src)}" alt="">`;
      thumb.addEventListener("click", () => {
        galleryIndex = index;
        renderGallery();
      });
      thumbStrip.appendChild(thumb);
    });
  }

  function showGalleryOffset(offset) {
    if (!galleryImages.length) {
      return;
    }
    galleryIndex = (galleryIndex + offset + galleryImages.length) % galleryImages.length;
    renderGallery();
  }

  function bindEvents() {
    filterButtons.forEach((button) => {
      button.addEventListener("click", () => {
        activeFilter = button.dataset.filter || "all";
        filterButtons.forEach((item) => {
          const isActive = item === button;
          item.classList.toggle("is-active", isActive);
          item.setAttribute("aria-selected", String(isActive));
        });
        renderSummary();
        renderItems();
      });
    });

    searchInput.addEventListener("input", () => {
      searchTerm = searchInput.value;
      renderItems();
    });

    dialogClose.addEventListener("click", () => dialog.close());
    galleryPrev.addEventListener("click", () => showGalleryOffset(-1));
    galleryNext.addEventListener("click", () => showGalleryOffset(1));
    dialog.addEventListener("click", (event) => {
      if (event.target === dialog) {
        dialog.close();
      }
    });
    document.addEventListener("keydown", (event) => {
      if (!dialog.open) {
        return;
      }
      if (event.key === "ArrowLeft") {
        showGalleryOffset(-1);
      }
      if (event.key === "ArrowRight") {
        showGalleryOffset(1);
      }
    });
  }

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function escapeAttr(value) {
    return escapeHtml(value).replace(/`/g, "&#096;");
  }

  bindEvents();
  render();
})();
