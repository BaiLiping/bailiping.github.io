(function () {
  const data = window.SALE_DATA || { sellers: [] };
  const sellers = Array.isArray(data.sellers) ? data.sellers : [];
  const statusLabels = {
    available: "Available",
    later: "For later"
  };

  const itemGrid = document.querySelector("#itemGrid");
  const emptyState = document.querySelector("#emptyState");
  const summaryTotal = document.querySelector("#summaryTotal");
  const summaryAvailable = document.querySelector("#summaryAvailable");
  const contactPanel = document.querySelector("#contactPanel");
  const statusFilterButtons = Array.from(document.querySelectorAll("[data-status]"));
  const roomFilterButtons = Array.from(document.querySelectorAll("[data-room]"));
  const searchInput = document.querySelector("#searchInput");
  const dialog = document.querySelector("#itemDialog");
  const dialogClose = document.querySelector("#dialogClose");
  const dialogImage = document.querySelector("#dialogImage");
  const dialogTitle = document.querySelector("#dialogTitle");
  const dialogSeller = document.querySelector("#dialogSeller");
  const dialogStatus = document.querySelector("#dialogStatus");
  const dialogPrice = document.querySelector("#dialogPrice");
  const dialogDescription = document.querySelector("#dialogDescription");
  const dialogDetails = document.querySelector("#dialogDetails");
  const dialogWebsite = document.querySelector("#dialogWebsite");
  const dialogNotes = document.querySelector("#dialogNotes");
  const galleryPrev = document.querySelector("#galleryPrev");
  const galleryNext = document.querySelector("#galleryNext");
  const imageCounter = document.querySelector("#imageCounter");
  const thumbStrip = document.querySelector("#thumbStrip");

  let activeSellerId = sellers[0] ? sellers[0].id : "";
  let activeStatusFilter = "all";
  let activeRoomFilter = "all";
  let searchTerm = "";
  let currentItems = [];
  let galleryImages = [];
  let galleryIndex = 0;

  function getActiveSeller() {
    return sellers.find((seller) => seller.id === activeSellerId) || sellers[0] || null;
  }

  function getSellerItems(seller) {
    const items = Array.isArray(seller && seller.items) ? seller.items : [];
    return items.filter((item) => !item.hidden);
  }

  function getVisibleItems() {
    const seller = getActiveSeller();
    const items = getSellerItems(seller);
    const normalizedSearch = searchTerm.trim().toLowerCase();

    return items.filter((item) => {
      const statusMatch = activeStatusFilter === "all" || item.status === activeStatusFilter;
      const roomMatch = activeRoomFilter === "all" || itemRooms(item).includes(activeRoomFilter);
      if (!statusMatch) {
        return false;
      }

      if (!roomMatch) {
        return false;
      }

      if (!normalizedSearch) {
        return true;
      }

      return [
        item.title,
        item.category,
        item.description,
        shouldShowPrice(item) ? item.price : "",
        shouldShowPrice(item) ? item.originalPrice : "",
        item.availableTime,
        item.website,
        formatQuantity(item.quantity),
        formatRoomList(item),
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
    return status === "later" || status === "reserved" || status === "sold" ? status : "available";
  }

  function statusLabel(status) {
    return statusLabels[status] || "Available";
  }

  function itemImages(item) {
    return Array.isArray(item.images) ? item.images.filter(Boolean) : [];
  }

  function itemRooms(item) {
    return Array.isArray(item.rooms) ? item.rooms.filter(Boolean) : [];
  }

  function formatQuantity(quantity) {
    return quantity || quantity === 0 ? `Qty ${quantity}` : "";
  }

  function formatRoomList(item) {
    const rooms = itemRooms(item);
    return rooms.map((room) => room.charAt(0).toUpperCase() + room.slice(1)).join(", ");
  }

  function shouldShowPrice(item) {
    return item && item.status !== "later";
  }

  function detailRows(item) {
    const showPrice = shouldShowPrice(item);
    const rows = [
      { label: "Status", value: statusLabel(item.status) },
      { label: "Quantity", value: item.quantity || item.quantity === 0 ? String(item.quantity) : "" },
      { label: "Rooms", value: formatRoomList(item) },
      { label: "Available time", value: item.availableTime || "" },
      { label: "Original price", value: showPrice ? item.originalPrice || "" : "" },
      { label: "Reference price", value: showPrice ? [item.euroPrice, item.rmbPrice].filter(Boolean).join(" / ") : "" }
    ];
    return rows.filter((row) => row.value);
  }

  function render() {
    renderSummary();
    renderContact();
    renderItems();
  }

  function renderSummary() {
    const seller = getActiveSeller();
    const items = getSellerItems(seller);
    const available = items.filter((item) => item.status === "available").length;
    const later = items.filter((item) => item.status === "later").length;
    summaryTotal.textContent = `${items.length} item${items.length === 1 ? "" : "s"}`;
    summaryAvailable.textContent = `${available} available · ${later} for later`;
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
      seller.email ? `Email: ${seller.email}` : "",
      seller.location ? `Pickup: ${seller.location}` : "",
      seller.phone ? `Phone: ${seller.phone}` : "",
      contact.note || ""
    ].filter(Boolean);
    const title = contact.title || "";
    const message = contact.message || "";

    contactPanel.hidden = false;
    contactPanel.innerHTML = `
      <div class="contact-copy">
        <p class="contact-kicker">Contact</p>
        ${title ? `<h3>${escapeHtml(title)}</h3>` : ""}
        ${message ? `<p class="contact-text">${escapeHtml(message)}</p>` : ""}
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
      const rooms = itemRooms(item);
      const showPrice = shouldShowPrice(item);
      const facts = [
        formatQuantity(item.quantity),
        item.availableTime || "",
        showPrice && item.originalPrice ? `Original ${item.originalPrice}` : ""
      ].filter(Boolean);
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
            ${showPrice ? `<span class="price">${escapeHtml(item.price || "Price TBC")}</span>` : ""}
          </div>
          <p class="description">${escapeHtml(item.description || "")}</p>
          ${facts.length ? `<div class="card-facts">${facts.map((fact) => `<span>${escapeHtml(fact)}</span>`).join("")}</div>` : ""}
          <div class="card-meta">
            ${rooms.map((room) => `<span>${escapeHtml(room.charAt(0).toUpperCase() + room.slice(1))}</span>`).join("")}
            ${item.website ? "<span>Product link</span>" : ""}
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
    dialogSeller.textContent = seller ? seller.location || seller.name : "";
    if (shouldShowPrice(item)) {
      dialogPrice.hidden = false;
      dialogPrice.textContent = item.price ? `Price: ${item.price}` : "Price TBC";
    } else {
      dialogPrice.hidden = true;
      dialogPrice.textContent = "";
    }
    dialogDescription.textContent = item.description || "";
    renderDetails(item);
    renderProductLink(item);
    const note = item.notes && !item.notes.startsWith("Original price:") ? item.notes : "";
    dialogNotes.textContent = note;
    dialogNotes.hidden = !note;
    dialogStatus.className = `status-badge status-${statusClass(item.status)}`;
    dialogStatus.textContent = statusLabel(item.status);
    renderGallery();

    if (typeof dialog.showModal === "function") {
      dialog.showModal();
    } else {
      dialog.setAttribute("open", "");
    }
  }

  function renderDetails(item) {
    if (!dialogDetails) {
      return;
    }

    const rows = detailRows(item);
    dialogDetails.innerHTML = rows
      .map((row) => `
        <div>
          <dt>${escapeHtml(row.label)}</dt>
          <dd>${escapeHtml(row.value)}</dd>
        </div>
      `)
      .join("");
  }

  function renderProductLink(item) {
    if (!dialogWebsite) {
      return;
    }

    if (!item.website) {
      dialogWebsite.hidden = true;
      dialogWebsite.removeAttribute("href");
      return;
    }

    dialogWebsite.hidden = false;
    dialogWebsite.href = item.website;
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
    statusFilterButtons.forEach((button) => {
      button.addEventListener("click", () => {
        activeStatusFilter = button.dataset.status || "all";
        statusFilterButtons.forEach((item) => {
          const isActive = item === button;
          item.classList.toggle("is-active", isActive);
          item.setAttribute("aria-selected", String(isActive));
        });
        renderSummary();
        renderItems();
      });
    });

    roomFilterButtons.forEach((button) => {
      button.addEventListener("click", () => {
        activeRoomFilter = button.dataset.room || "all";
        roomFilterButtons.forEach((item) => {
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
