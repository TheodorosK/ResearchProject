/* Lumen front-end: a small hash-routed single page app, no build step. */
"use strict";

// ------------------------------------------------------------------ utilities
const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const h = (html) => { const t = document.createElement("template"); t.innerHTML = html.trim(); return t.content.firstElementChild; };
const fmt = (n) => Number(n || 0).toLocaleString();
const plural = (n, one, many = one + "s") => `${fmt(n)} ${n === 1 ? one : many}`;
const debounce = (fn, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };
const previewUrl = (id) => `/api/photos/${id}/preview`;
const faceUrl = (id) => `/api/faces/${id}/thumb`;
const fmtClock = (s) => { s = Math.round(s || 0); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`; };
const fmtBytes = (b) => !b ? "" : b >= 1048576 ? (b / 1048576).toFixed(1) + " MB" : Math.max(1, Math.round(b / 1024)) + " KB";
const fmtDate = (ts, opts = { year: "numeric", month: "short", day: "numeric" }) => ts ? new Date(ts * 1000).toLocaleDateString(undefined, opts) : "";
const fmtDuration = (s) => {
  if (s == null) return "";
  if (s < 90) return "under 2 min";
  if (s < 3600) return `~${Math.round(s / 60)} min`;
  return `~${(s / 3600).toFixed(s < 36000 ? 1 : 0)} h`;
};
const store = {
  get(k, d) { try { const v = localStorage.getItem("lumen:" + k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem("lumen:" + k, JSON.stringify(v)); } catch { /* private mode */ } },
};

async function api(path, opts = {}) {
  const init = { ...opts, headers: { ...(opts.headers || {}) } };
  if (opts.body !== undefined && typeof opts.body !== "string") {
    init.body = JSON.stringify(opts.body);
    init.headers["Content-Type"] = "application/json";
  }
  const res = await fetch(path, init);
  let data = null;
  try { data = await res.json(); } catch { /* empty body */ }
  if (!res.ok) {
    const d = data && data.detail;
    const err = new Error(typeof d === "string" ? d : Array.isArray(d) ? d.map((x) => x.msg).join(", ") : `Request failed (${res.status})`);
    err.status = res.status;
    throw err;
  }
  return data;
}

const I = {
  search: '<svg class="icon" viewBox="0 0 24 24"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>',
  similar: '<svg class="icon" viewBox="0 0 24 24"><path d="M12 3l1.9 5.6L19.5 10l-5.6 1.9L12 17.5l-1.9-5.6L4.5 10l5.6-1.4z"/><path d="M19 15l.8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8z"/></svg>',
  plus: '<svg class="icon" viewBox="0 0 24 24"><path d="M12 5v14M5 12h14"/></svg>',
  check: '<svg viewBox="0 0 24 24"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>',
  close: '<svg class="icon" viewBox="0 0 24 24"><path d="M6 6l12 12M18 6 6 18"/></svg>',
  left: '<svg class="icon" viewBox="0 0 24 24"><path d="M15 5l-7 7 7 7"/></svg>',
  right: '<svg class="icon" viewBox="0 0 24 24"><path d="M9 5l7 7-7 7"/></svg>',
  folder: '<svg class="icon" viewBox="0 0 24 24"><path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/></svg>',
  download: '<svg class="icon" viewBox="0 0 24 24"><path d="M12 4v11m0 0-4-4m4 4 4-4M5 20h14"/></svg>',
  trash: '<svg class="icon" viewBox="0 0 24 24"><path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"/></svg>',
  edit: '<svg class="icon" viewBox="0 0 24 24"><path d="M4 20h4L19 9l-4-4L4 16z"/></svg>',
  star: '<svg class="icon" viewBox="0 0 24 24"><path d="M12 3l2.7 5.6 6.1.9-4.4 4.3 1 6.1L12 17l-5.4 2.9 1-6.1L3.2 9.5l6.1-.9z"/></svg>',
  pause: '<svg class="icon" viewBox="0 0 24 24"><path d="M8 5v14M16 5v14"/></svg>',
  play: '<svg class="icon" viewBox="0 0 24 24"><path d="M7 5l12 7-12 7z"/></svg>',
  up: '<svg class="icon" viewBox="0 0 24 24"><path d="M7 11v9H4v-9zM7 11l4-8a2 2 0 0 1 3 2l-1 5h6a2 2 0 0 1 2 2.3l-1.2 6A2 2 0 0 1 17.8 20H7"/></svg>',
  down: '<svg class="icon" viewBox="0 0 24 24"><path d="M17 13V4h3v9zM17 13l-4 8a2 2 0 0 1-3-2l1-5H5a2 2 0 0 1-2-2.3l1.2-6A2 2 0 0 1 6.2 4H17"/></svg>',
  slides: '<svg class="icon" viewBox="0 0 24 24"><rect x="3" y="5" width="18" height="14" rx="2"/><path d="M10 9.5v5l4-2.5z"/></svg>',
  refresh: '<svg class="icon" viewBox="0 0 24 24"><path d="M20 11a8 8 0 1 0-2.3 5.7M20 4v7h-7"/></svg>',
  collection: '<svg class="icon" viewBox="0 0 24 24"><rect x="3" y="7" width="14" height="13" rx="2"/><path d="M7 4h12a2 2 0 0 1 2 2v11"/></svg>',
  moon: '<svg class="icon" viewBox="0 0 24 24"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>',
  person: '<svg class="icon" viewBox="0 0 24 24"><circle cx="12" cy="8" r="4"/><path d="M4 21a8 8 0 0 1 16 0"/></svg>',
  pin: '<svg class="icon" viewBox="0 0 24 24"><path d="M12 21s-7-6.2-7-11.5a7 7 0 0 1 14 0C19 14.8 12 21 12 21z"/><circle cx="12" cy="9.5" r="2.5"/></svg>',
  copy: '<svg class="icon" viewBox="0 0 24 24"><rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2"/></svg>',
  merge: '<svg class="icon" viewBox="0 0 24 24"><path d="M6 3v6a6 6 0 0 0 6 6h6M15 12l3 3-3 3M6 21v-4"/></svg>',
  eyeoff: '<svg class="icon" viewBox="0 0 24 24"><path d="M3 3l18 18M10.6 6.1A10 10 0 0 1 12 6c6 0 9.5 6 9.5 6a17 17 0 0 1-3 3.7M6.6 6.6A17 17 0 0 0 2.5 12S6 18 12 18a9.6 9.6 0 0 0 4.4-1"/></svg>',
  sun: '<svg class="icon" viewBox="0 0 24 24"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>',
};

const EMOJIS = ["📸", "🏖️", "🏔️", "🌅", "🎉", "❤️", "👨‍👩‍👧", "🐶", "🐱", "🍝", "✈️", "🚗", "🎄", "🎂", "💍", "🌸", "🌊", "🏛️", "⭐", "🌙", "🎨", "⚽", "🎶", "🏝️", "🍷", "👶", "🧳", "🏡"];
const COLORS = ["#ff8a5b", "#ff6f91", "#c77dff", "#7c9cff", "#38bdf8", "#3ddc97", "#facc15", "#a3a3a3"];
const COLLECTION_IDEAS = ["Summer 2024", "Best of the kids", "Food diary", "Wall-worthy", "Our trips", "Family portraits"];

// ------------------------------------------------------------------ app state
const state = {
  status: null,
  selection: new Map(),     // id -> item, survives navigation so you can gather from several searches
  target: null,             // collection being filled via "Add photos"
  trayActions: [],          // per-view extra tray buttons
  view: null,
};

// ------------------------------------------------------------------ toasts
function toast(msg, { action, onAction, error = false, ms = 4200 } = {}) {
  const el = h(`<div class="toast${error ? " err" : ""}"><span>${msg}</span>${action ? `<button>${esc(action)}</button>` : ""}</div>`);
  if (action) $("button", el).onclick = () => { onAction?.(); el.remove(); };
  $("#toasts").append(el);
  setTimeout(() => el.remove(), ms);
}
const fail = (e) => toast(esc(e.message || e), { error: true });

// ------------------------------------------------------------------ selection
function isSelected(id) { return state.selection.has(id); }
function setSelected(item, on) {
  if (on) state.selection.set(item.id, item); else state.selection.delete(item.id);
  $$(`.tile[data-id="${item.id}"]`).forEach((t) => t.classList.toggle("selected", on));
}
function clearSelection() {
  state.selection.clear();
  $$(".tile.selected").forEach((t) => t.classList.remove("selected"));
  renderTray();
}
function renderTray() {
  const n = state.selection.size;
  document.body.classList.toggle("selecting", n > 0);
  $("#tray").hidden = n === 0;
  $("#tray-n").textContent = fmt(n);
  const actions = $("#tray-actions");
  actions.innerHTML = "";
  const ids = () => [...state.selection.keys()];
  const add = (label, icon, fn, cls = "") => {
    const b = h(`<button class="btn ${cls}">${icon}<span>${esc(label)}</span></button>`);
    b.onclick = fn;
    actions.append(b);
  };
  if (state.target) {
    add(`Add to ${state.target.emoji || ""} ${state.target.name}`, I.plus, () => addPhotosTo(state.target, ids()), "primary");
  }
  for (const a of state.trayActions) add(a.label, a.icon, () => a.run(ids()), a.cls || "");
  add("Add to collection", I.collection, () => openAddToCollection(ids()), state.target ? "" : "primary");
  add("More like these", I.similar, () => go(`#/like?ids=${ids().slice(0, 60).join(",")}`));
}

// ------------------------------------------------------------------ justified photo grid
class JGrid {
  constructor(items = [], { rowHeight, showTier = false, coverId = null, onOpen, feedback = null } = {}) {
    this.feedback = feedback;
    this.items = [];
    this.tiles = [];
    this.showTier = showTier;
    this.coverId = coverId;
    this.onOpen = onOpen || ((idx) => openLightbox(this.items, idx));
    this.rowHeight = rowHeight || (() => (innerWidth < 600 ? 120 : innerWidth < 1100 ? 180 : 220));
    this.el = h('<div class="jgrid"></div>');
    this.lastIdx = null;
    this.el.addEventListener("click", (e) => this.onClick(e));
    let raf = 0, lastW = 0;
    this.ro = new ResizeObserver(() => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => { if (this.el.clientWidth !== lastW) { lastW = this.el.clientWidth; this.layout(); } });
    });
    this.ro.observe(this.el);
    this.add(items);
  }
  add(items) {
    const frag = document.createDocumentFragment();
    for (const it of items) {
      const t = h(`<div class="tile${isSelected(it.id) ? " selected" : ""}" data-id="${it.id}">
        <img loading="lazy" decoding="async" alt="${esc(it.name)}" src="${previewUrl(it.id)}">
        <button class="check" data-act="select" aria-label="Select">${I.check}</button>
        <div class="quick">${this.feedback ? `<button data-act="like" title="More like this" aria-label="More like this" class="${this.feedback.like.has(it.id) ? "on" : ""}">${I.up}</button><button data-act="dislike" title="Less like this" aria-label="Less like this" class="${this.feedback.dislike.has(it.id) ? "on" : ""}">${I.down}</button>` : ""}<button data-act="similar" title="Find similar photos" aria-label="Find similar">${I.similar}</button></div>
        ${it.kind === "video" ? `<span class="vid-badge">${I.play}${it.t != null ? `${fmtClock(it.t)} / ` : ""}${fmtClock(it.duration)}</span>` : ""}
        ${this.showTier && it.tier ? `<i class="tier ${it.tier}"></i>` : ""}
        ${this.coverId === it.id ? '<span class="cover-flag">Cover</span>' : ""}
      </div>`);
      const img = $("img", t);
      img.onload = () => img.classList.add("loaded");
      img.onerror = () => { img.classList.add("loaded"); img.style.visibility = "hidden"; };
      this.tiles.push(t);
      this.items.push(it);
      frag.append(t);
    }
    this.el.append(frag);
    this.layout();
  }
  layout() {
    const W = this.el.clientWidth;
    if (!W) return;
    const gap = 6, target = this.rowHeight();
    let row = [], sum = 0;
    const size = (tiles, hgt) => tiles.forEach(([t, ar]) => { t.style.height = hgt + "px"; t.style.width = ar * hgt + "px"; });
    this.items.forEach((it, i) => {
      const ar = Math.min(3, Math.max(0.4, (it.w || 4) / (it.h || 3)));
      row.push([this.tiles[i], ar]);
      sum += ar;
      if (sum * target + gap * (row.length - 1) >= W) {
        size(row, (W - gap * (row.length - 1) - 1) / sum);
        row = []; sum = 0;
      }
    });
    if (row.length) size(row, target);
  }
  onClick(e) {
    const tile = e.target.closest(".tile");
    if (!tile) return;
    const idx = this.tiles.indexOf(tile);
    const item = this.items[idx];
    const act = e.target.closest("[data-act]")?.dataset.act;
    if (act === "similar") { go(`#/similar/${item.id}`); return; }
    if (act === "like" || act === "dislike") { this.feedback.toggle(item.id, act); return; }
    if (act === "select" || state.selection.size > 0 || e.metaKey || e.ctrlKey) {
      if (e.shiftKey && this.lastIdx != null) {
        const [a, b] = [Math.min(this.lastIdx, idx), Math.max(this.lastIdx, idx)];
        for (let i = a; i <= b; i++) setSelected(this.items[i], true);
      } else {
        setSelected(item, !isSelected(item.id));
      }
      this.lastIdx = idx;
      renderTray();
      return;
    }
    this.onOpen(idx);
  }
  selectAll() { this.items.forEach((it) => setSelected(it, true)); renderTray(); }
}

function emptyState(emoji, title, text, actionsHtml = "") {
  return h(`<div class="empty fade-in"><div class="big">${emoji}</div><h3>${title}</h3><p>${text}</p>${actionsHtml}</div>`);
}

// ------------------------------------------------------------------ router
function go(hash) { if (location.hash === hash) route(); else location.hash = hash; }

function parseHash() {
  const raw = location.hash.replace(/^#/, "") || "/";
  const [path, qs] = raw.split("?");
  return { parts: path.split("/").filter(Boolean), params: new URLSearchParams(qs || "") };
}

async function route() {
  const { parts, params } = parseHash();
  const view = $("#view");
  const name = parts[0] || "home";
  state.view = name;
  state.trayActions = [];
  document.body.classList.toggle("on-home", name === "home");
  $$(".nav a").forEach((a) => a.classList.toggle("active", a.dataset.nav === name || (name === "collection" && a.dataset.nav === "collections") || (name === "person" && a.dataset.nav === "people") || (name === "memory" && a.dataset.nav === "memories")));
  if (name !== "search") $("#top-search-input").value = "";
  view.innerHTML = "";
  closeLightbox();
  window.scrollTo(0, 0);
  try {
    if (name === "home") await renderHome(view);
    else if (name === "search") await renderSearch(view, params);
    else if (name === "similar") await renderSimilar(view, Number(parts[1]));
    else if (name === "like") await renderLike(view, params);
    else if (name === "library") await renderLibrary(view, params);
    else if (name === "collections") await renderCollections(view);
    else if (name === "people") await renderPeople(view, params);
    else if (name === "person") await renderPerson(view, Number(parts[1]));
    else if (name === "map") await renderMap(view, params);
    else if (name === "tidy") await renderTidy(view, params);
    else if (name === "memories") await renderMemories(view, params);
    else if (name === "memory") await renderMemory(view, parts.slice(1).join("/"));
    else if (name === "collection") await renderCollection(view, Number(parts[1]));
    else if (name === "sources") await renderSources(view);
    else view.append(emptyState("🧭", "Nothing here", "That page doesn't exist.", '<a class="btn" href="#/">Go home</a>'));
  } catch (e) {
    view.append(e.status === 503
      ? emptyState("⏳", "Almost ready", esc(e.message), '<button class="btn" onclick="route()">Try again</button>')
      : emptyState("😵", "Something went wrong", esc(e.message)));
  }
  renderTray();
  renderTargetBanner();
}

function renderTargetBanner() {
  $("#target-banner")?.remove();
  if (!state.target || state.view === "collection") return;
  const b = h(`<div id="target-banner" class="panel fade-in" style="display:flex;align-items:center;justify-content:space-between;gap:12px;padding:12px 16px;margin-bottom:18px">
    <span>Collecting photos for <strong>${esc(state.target.emoji || "")} ${esc(state.target.name)}</strong> — search, select, then press <em>Add</em>.</span>
    <a class="btn small" href="#/collection/${state.target.id}">Done</a></div>`);
  $("#view").prepend(b);
}

// ------------------------------------------------------------------ home
async function renderHome(view) {
  const st = state.status || (await refreshStatus());
  if (!st.folders) return renderWelcome(view);
  const data = await api("/api/home");
  const mosaic = data.recent.slice(0, 12).map((p) => `<img src="${previewUrl(p.id)}" alt="">`).join("");
  const hero = h(`<section class="hero fade-in">
      <div class="hero-mosaic" aria-hidden="true">${mosaic}</div>
      <h1>What would you like to <span class="grad-text">remember</span>?</h1>
      <p class="lede">Describe a moment, a place, a feeling — in your own words. Lumen finds the photos that fit.</p>
      <form class="hero-search" role="search">
        ${I.search}
        <input type="search" name="q" autocomplete="off" aria-label="Describe what you're looking for">
        <button class="btn primary" type="submit">${I.search}<span>Search</span></button>
      </form>
      <div class="chips"></div>
      <div class="hint">Tip: add <kbd>-word</kbd> to leave something out, like <em>beach -people</em>. Press <kbd>/</kbd> to search from anywhere.</div>
    </section>`);
  const input = $("input", hero);
  $("form", hero).onsubmit = (e) => { e.preventDefault(); if (input.value.trim()) go(`#/search?q=${encodeURIComponent(input.value.trim())}`); };
  const chips = $(".chips", hero);
  for (const s of data.suggestions) {
    const c = h(`<a class="chip" href="#/search?q=${encodeURIComponent(s)}">${esc(s)}</a>`);
    chips.append(c);
  }
  const surprise = h('<button class="chip" type="button">🎲 Surprise me</button>');
  surprise.onclick = () => {
    const pool = data.explore.length ? data.explore.map((g) => g.query) : data.suggestions;
    go(`#/search?q=${encodeURIComponent(pool[Math.floor(Math.random() * pool.length)])}`);
  };
  chips.append(surprise);
  view.append(hero);
  typePlaceholder(input, data.suggestions);
  setTimeout(() => input.focus({ preventScroll: true }), 50);

  if (!st.ready && st.pending) {
    view.append(h(`<section class="section panel fade-in"><h2>Lumen is getting to know your photos ✨</h2>
      <p class="muted">The first photos will show up here in a moment. You can already search — results grow as indexing continues.</p>
      <div class="bar"><i style="width:${st.total ? (100 * st.ready) / st.total : 2}%"></i></div></section>`));
  }

  if (data.on_this_day.length) {
    const years = [...new Set(data.on_this_day.map((p) => new Date(p.taken_at * 1000).getFullYear()))];
    const sec = h(`<section class="section fade-in"><div class="section-head"><h2>🕰️ On this day</h2><span class="muted">${years.join(" · ")}</span></div></section>`);
    const g = new JGrid(data.on_this_day, { rowHeight: () => (innerWidth < 600 ? 110 : 170) });
    sec.append(g.el);
    view.append(sec);
  }

  if (data.best.length >= 6) {
    const sec = h(`<section class="section fade-in"><div class="section-head"><h2>✨ Your best shots</h2><a href="#/library?order=best">See more</a></div></section>`);
    sec.append(new JGrid(data.best.slice(0, 12), { rowHeight: () => (innerWidth < 600 ? 120 : 200) }).el);
    view.append(sec);
  }

  const cols = await api("/api/collections");
  const csec = h(`<section class="section fade-in"><div class="section-head"><h2>📚 Your collections</h2><a href="#/collections">See all</a></div><div class="scroller"></div></section>`);
  const sc = $(".scroller", csec);
  cols.slice(0, 10).forEach((c) => sc.append(collectionCard(c)));
  sc.append(newCollectionCard());
  view.append(csec);

  if (data.people.length) {
    const psec = h(`<section class="section fade-in"><div class="section-head"><h2>🙂 People</h2><a href="#/people">See everyone</a></div><div class="people-strip"></div></section>`);
    data.people.forEach((p) => $(".people-strip", psec).append(personChip(p)));
    view.append(psec);
  }

  if (data.memories.length) {
    const msec = h(`<section class="section fade-in"><div class="section-head"><h2>🧳 Memories</h2><a href="#/memories">See all</a></div><div class="scroller memories"></div></section>`);
    data.memories.forEach((m) => $(".scroller", msec).append(memoryCard(m)));
    view.append(msec);
  }

  const hints = [];
  if (data.places) hints.push(`<a class="hint-card" href="#/map"><span class="hint-e">🗺️</span><span><strong>${plural(data.places, "photo")} on the map</strong><span class="muted">See where your memories happened</span></span></a>`);
  if (data.tidy.duplicate_extra) hints.push(`<a class="hint-card" href="#/tidy"><span class="hint-e">🧹</span><span><strong>${plural(data.tidy.duplicate_extra, "duplicate copy", "duplicate copies")}</strong><span class="muted">Free up space by clearing the extras</span></span></a>`);
  if (data.tidy.burst_groups) hints.push(`<a class="hint-card" href="#/tidy?kind=bursts"><span class="hint-e">🎞️</span><span><strong>${plural(data.tidy.burst_groups, "burst")} of similar shots</strong><span class="muted">Keep the best of each moment</span></span></a>`);
  if (hints.length) view.append(h(`<section class="section fade-in hint-row">${hints.join("")}</section>`));

  if (data.explore.length) {
    const sec = h(`<section class="section fade-in"><div class="section-head"><h2>🧭 Your world at a glance</h2><span class="muted">Themes Lumen spotted in your library</span></div><div class="card-grid"></div></section>`);
    const grid = $(".card-grid", sec);
    for (const g of data.explore.slice(0, 16)) {
      const cover = g.photos[0];
      grid.append(h(`<a class="explore-card" href="#/search?q=${encodeURIComponent(g.query)}&title=${encodeURIComponent(g.name)}">
        ${cover ? `<img loading="lazy" src="${previewUrl(cover.id)}" alt="">` : ""}
        <div class="label"><span class="e">${g.emoji}</span><strong>${esc(g.name)}</strong><span>${plural(g.count, "photo")}</span></div></a>`));
    }
    view.append(sec);
  }

  if (data.moods.length) {
    const sec = h(`<section class="section fade-in"><div class="section-head"><h2>🎭 Moods & styles</h2><span class="muted">The feeling of your photos</span></div><div class="card-grid moods"></div></section>`);
    const grid = $(".card-grid", sec);
    for (const g of data.moods.slice(0, 12)) {
      const cover = g.photos[0];
      grid.append(h(`<a class="explore-card mood" href="#/search?q=${encodeURIComponent(g.query)}&title=${encodeURIComponent(g.name)}&sort=quality">
        ${cover ? `<img loading="lazy" src="${previewUrl(cover.id)}" alt="">` : ""}
        <div class="label"><span class="e">${g.emoji}</span><strong>${esc(g.name)}</strong><span>${plural(g.count, "photo")}</span></div></a>`));
    }
    view.append(sec);
  }

  if (data.recent.length) {
    const sec = h(`<section class="section fade-in"><div class="section-head"><h2>🆕 Recently added</h2><a href="#/library">Browse library</a></div></section>`);
    sec.append(new JGrid(data.recent).el);
    view.append(sec);
  }
}

function typePlaceholder(input, phrases) {
  let p = 0, i = 0, deleting = false;
  const tick = () => {
    if (!input.isConnected) return;
    const word = phrases[p % phrases.length];
    i += deleting ? -1 : 1;
    input.placeholder = `Try “${word.slice(0, i)}”`;
    let delay = deleting ? 28 : 55 + Math.random() * 40;
    if (!deleting && i >= word.length) { deleting = true; delay = 1800; }
    else if (deleting && i <= 0) { deleting = false; p++; delay = 300; }
    setTimeout(tick, delay);
  };
  tick();
}

function renderWelcome(view) {
  const el = h(`<section class="hero fade-in">
    <h1>Your photos, <span class="grad-text">understood</span>.</h1>
    <p class="lede">Point Lumen at the folders where your photos live. It looks at every picture — on this computer, nothing is uploaded — so you can find any moment just by describing it.</p>
    <div class="row-actions"><button class="btn primary" id="welcome-add">${I.folder} Choose photo folders</button></div>
    <div class="steps">
      <div class="step"><b>1</b><h4>Pick folders</h4><p class="muted">Add as many as you like — your Pictures folder, an external drive, old phone backups.</p></div>
      <div class="step"><b>2</b><h4>Lumen learns</h4><p class="muted">It builds a smart preview and an understanding of each photo. You can search while it works.</p></div>
      <div class="step"><b>3</b><h4>Just ask</h4><p class="muted">“Grandma's garden in spring”, “sunset from the ferry”, “dogs in the snow”. Then gather favourites into collections.</p></div>
    </div></section>`);
  $("#welcome-add", el).onclick = () => openFolderBrowser();
  view.append(el);
}

// ------------------------------------------------------------------ search
const TIERS = [
  ["best", "Spot on", "Photos that match your description closely"],
  ["good", "Strong matches", "Very likely what you mean"],
  ["maybe", "Might be it", "Looser matches worth a glance"],
];

const legendItems = '<span><i class="tier best"></i>Spot on</span><span><i class="tier good"></i>Strong</span><span><i class="tier maybe"></i>Possible</span>';

function renderTiered(container, items, emptyMsg, gridOpts = {}) {
  if (!items.length) { container.append(emptyMsg); return []; }
  const grids = [];
  const strong = items.some((i) => i.tier !== "maybe");
  for (const [tier, title, sub] of TIERS) {
    const group = items.filter((i) => i.tier === tier);
    if (!group.length) continue;
    const label = !strong ? "Closest we could find" : title;
    const subtitle = !strong ? "Nothing matched strongly — try describing it differently, or with more detail." : sub;
    container.append(h(`<div class="group-head"><i class="dot tier ${tier}"></i><h3>${label}</h3><span class="sub">${plural(group.length, "photo")} · ${subtitle}</span></div>`));
    const g = new JGrid(group, { showTier: false, ...gridOpts });
    container.append(g.el);
    grids.push(g);
  }
  return grids;
}

async function renderSearch(view, params) {
  const q = params.get("q") || "";
  const title = params.get("title");
  $("#top-search-input").value = title ? "" : q;
  const filters = new URLSearchParams();
  for (const k of ["year_from", "year_to", "folder_id", "sort", "like", "dislike"]) if (params.get(k)) filters.set(k, params.get(k));
  const head = h(`<div class="search-head fade-in">
      <div class="search-top"><h1 class="search-title">${title ? `${esc(title)}` : `<q>${esc(q)}</q>`}</h1><div class="row-actions"></div></div>
      <div class="filters"></div>
    </div>`);
  view.append(head);
  const results = h('<div><div class="skeleton">' + "<div></div>".repeat(12) + "</div></div>");
  view.append(results);

  const [res, years, folders] = await Promise.all([
    api(`/api/search?q=${encodeURIComponent(q)}&${filters}`),
    api("/api/years"),
    api("/api/folders"),
  ]);
  const f = $(".filters", head);
  const setParam = (k, v) => { const p = new URLSearchParams(params); v ? p.set(k, v) : p.delete(k); go(`#/search?${p}`); };
  if (years.length > 1) {
    const yf = h(`<select aria-label="From year"><option value="">Any year</option>${years.map((y) => `<option ${params.get("year_from") == y.year ? "selected" : ""}>${y.year}</option>`).join("")}</select>`);
    yf.onchange = () => {
      const p = new URLSearchParams(params);
      yf.value ? (p.set("year_from", yf.value), p.set("year_to", yf.value)) : (p.delete("year_from"), p.delete("year_to"));
      go(`#/search?${p}`);
    };
    f.append(yf);
  }
  if (folders.length > 1) {
    const fs = h(`<select aria-label="Folder"><option value="">All folders</option>${folders.map((x) => `<option value="${x.id}" ${params.get("folder_id") == x.id ? "selected" : ""}>${esc(x.name)}</option>`).join("")}</select>`);
    fs.onchange = () => setParam("folder_id", fs.value);
    f.append(fs);
  }
  const sortSel = h(`<select aria-label="Order"><option value="relevance">Best match first</option><option value="quality" ${params.get("sort") === "quality" ? "selected" : ""}>Best looking first</option></select>`);
  sortSel.onchange = () => setParam("sort", sortSel.value === "quality" ? "quality" : "");
  f.append(sortSel);
  const ex = res.parsed.expression;
  if (ex) f.append(h(`<span class="chip" title="Lumen reads the expressions on faces for this">${ex.emoji} ${ex.everyone ? "everyone " : ""}${esc({ happiness: "smiling", sadness: "sad", surprise: "surprised", anger: "angry", fear: "scared", disgust: "disgusted", neutral: "serious", contempt: "unimpressed" }[ex.label])} faces</span>`));
  if (res.parsed.places?.length) f.append(h(`<span class="chip" title="Only photos taken here">${I.pin} ${res.parsed.places.map(esc).join(", ")}</span>`));
  if (res.parsed.when) f.append(h(`<span class="chip" title="Only photos from then">📅 ${esc(res.parsed.when)}</span>`));
  if (res.parsed.people?.length) f.append(h(`<span class="muted">with ${res.parsed.people.map((p) => `<a class="chip" href="#/person/${p.id}">${I.person} ${esc(p.name)}</a>`).join(" ")}</span>`));
  const neg = res.parsed.negatives;
  if (neg.length) f.append(h(`<span class="muted">excluding ${neg.map((n) => `<strong>${esc(n)}</strong>`).join(", ")}</span>`));
  const saveBtn = h(`<button class="btn small">${I.star} Save as smart collection</button>`);
  saveBtn.onclick = () => openCollectionEditor({ kind: "smart", query: q, name: title || q.replace(/^./, (c) => c.toUpperCase()) });
  const selAll = h('<button class="btn small">Select all</button>');
  $(".row-actions", head).append(selAll, saveBtn);

  results.innerHTML = "";
  if (!res.indexed) {
    results.append(emptyState("⏳", "Still learning your photos", "Nothing is indexed yet. Results will appear as soon as the first photos are ready.", '<a class="btn" href="#/sources">See progress</a>'));
    return;
  }
  const strongCount = res.items.filter((i) => i.tier !== "maybe").length;
  head.append(h(`<div class="legend"><span class="muted">${strongCount ? plural(strongCount, "strong match", "strong matches") : "No strong matches"} among ${fmt(res.indexed)} photos</span>${legendItems}</div>`));
  // 👍/👎 refinement: kept in the URL so Back undoes it.
  const like = new Set(res.parsed.like), dislike = new Set(res.parsed.dislike);
  const feedback = {
    like, dislike,
    toggle: (id, act) => {
      const [on, off] = act === "like" ? [like, dislike] : [dislike, like];
      on.has(id) ? on.delete(id) : on.add(id);
      off.delete(id);
      const p = new URLSearchParams(params);
      like.size ? p.set("like", [...like].join(",")) : p.delete("like");
      dislike.size ? p.set("dislike", [...dislike].join(",")) : p.delete("dislike");
      go(`#/search?${p}`);
    },
  };
  if (like.size || dislike.size) {
    const bar = h(`<div class="refine-bar fade-in">${I.similar}<span>Refined with <strong>${like.size} 👍</strong> and <strong>${dislike.size} 👎</strong>. Keep rating to get closer.</span><button class="link">Start over</button></div>`);
    $("button", bar).onclick = () => { const p = new URLSearchParams(params); p.delete("like"); p.delete("dislike"); go(`#/search?${p}`); };
    head.append(bar);
  } else {
    head.append(h(`<p class="hint" style="margin:0">Not quite it? Hover a photo and press 👍 or 👎. Lumen will steer the results.</p>`));
  }
  const slideBtn = h(`<button class="btn small">${I.slides} Slideshow</button>`);
  slideBtn.onclick = () => playSlideshow(res.items.filter((i) => i.tier !== "maybe").length ? res.items.filter((i) => i.tier !== "maybe") : res.items, title || q);
  $(".row-actions", head).prepend(slideBtn);
  const grids = renderTiered(results, res.items, emptyState("🔍", "No photos found", "Try other words, or fewer of them."), { feedback });
  selAll.onclick = () => grids.forEach((g) => g.selectAll());
}

async function renderSimilar(view, id) {
  const [items, photo] = await Promise.all([api(`/api/photos/${id}/similar?limit=120`), api(`/api/photos/${id}`)]);
  const head = h(`<div class="page-head fade-in"><div style="display:flex;gap:16px;align-items:center">
      <img src="${previewUrl(id)}" alt="" style="width:84px;height:84px;object-fit:cover;border-radius:16px;box-shadow:var(--shadow);cursor:pointer">
      <div><h1>Photos like this one</h1><p>${esc(photo.name)} · ${fmtDate(photo.taken_at)}</p></div></div>
      <div class="row-actions"><button class="btn small">Select all</button></div></div>`);
  $("img", head).onclick = () => openLightbox([photo], 0);
  view.append(head);
  const grids = renderTiered(view, items, emptyState("🌫️", "Nothing similar yet", "As more photos are indexed, look-alikes will show up here."));
  $(".btn", head).onclick = () => grids.forEach((g) => g.selectAll());
}

async function renderLike(view, params) {
  const ids = (params.get("ids") || "").split(",").map(Number).filter(Boolean);
  const items = await api("/api/similar", { method: "POST", body: { photo_ids: ids } });
  const head = h(`<div class="page-head fade-in"><div><h1>More like your selection</h1><p>Based on ${plural(ids.length, "photo")} you picked</p></div>
     <div class="row-actions"><button class="btn small">Select all</button></div></div>`);
  view.append(head);
  const grids = renderTiered(view, items, emptyState("🌫️", "Nothing similar found", "Try selecting different photos."));
  $(".btn", head).onclick = () => grids.forEach((g) => g.selectAll());
}

// ------------------------------------------------------------------ library
async function renderLibrary(view, params) {
  const year = params.get("year");
  const order = params.get("order") || "newest";
  const years = await api("/api/years");
  const head = h(`<div class="page-head fade-in"><div><h1>Library</h1><p class="lib-count">&nbsp;</p></div>
     <div class="row-actions"><select class="input" style="height:36px;width:auto;border-radius:999px" aria-label="Order">
       <option value="newest">Newest first</option><option value="oldest" ${order === "oldest" ? "selected" : ""}>Oldest first</option><option value="best" ${order === "best" ? "selected" : ""}>Best looking first</option></select></div></div>`);
  $("select", head).onchange = (e) => go(`#/library?order=${e.target.value}${year ? `&year=${year}` : ""}`);
  view.append(head);
  if (years.length) {
    const chips = h('<div class="chips" style="margin-bottom:20px"></div>');
    chips.append(h(`<a class="chip ${!year ? "active" : ""}" href="#/library?order=${order}">All</a>`));
    years.forEach((y) => chips.append(h(`<a class="chip ${year == y.year ? "active" : ""}" href="#/library?order=${order}&year=${y.year}">${y.year} <span class="n">${fmt(y.count)}</span></a>`)));
    view.append(chips);
  }
  const body = h("<div></div>");
  const sentinel = h('<div style="height:1px"></div>');
  view.append(body, sentinel);
  const months = new Map();
  const all = [];
  let offset = 0, total = Infinity, loading = false;
  const load = async () => {
    if (loading || offset >= total) return;
    loading = true;
    const res = await api(`/api/photos?offset=${offset}&limit=300&order=${order}${year ? `&year=${year}` : ""}`);
    total = res.total;
    offset += res.items.length;
    $(".lib-count", head).textContent = total ? plural(total, "photo") : "";
    if (!total) body.append(emptyState("🖼️", "No photos yet", "Add a folder in Sources and they'll appear here.", '<a class="btn primary" href="#/sources">Add folders</a>'));
    const fresh = new Map();
    for (const it of res.items) {
      const key = order === "best" ? "Best looking first" : it.taken_at ? fmtDate(it.taken_at, { year: "numeric", month: "long" }) : "Undated";
      if (!fresh.has(key)) fresh.set(key, []);
      fresh.get(key).push(it);
    }
    for (const [key, items] of fresh) {
      let m = months.get(key);
      if (!m) {
        const sec = h(`<section class="month"><div class="group-head"><h3>${esc(key)}</h3><span class="sub"></span></div></section>`);
        const grid = new JGrid([], { onOpen: (idx) => openLightbox(all, all.indexOf(grid.items[idx])) });
        sec.append(grid.el);
        body.append(sec);
        m = { grid, sec };
        months.set(key, m);
      }
      m.grid.add(items);
      all.push(...items);
      $(".sub", m.sec).textContent = plural(m.grid.items.length, "photo");
    }
    loading = false;
  };
  const io = new IntersectionObserver((e) => { if (e[0].isIntersecting) load().catch(fail); }, { rootMargin: "1200px" });
  io.observe(sentinel);
  await load();
}

// ------------------------------------------------------------------ collections
function collectionCard(c) {
  const ids = c.preview_ids || [];
  const media = ids.length
    ? `<div class="mosaic n${Math.min(ids.length, 3)}">${ids.slice(0, 3).map((id) => `<img loading="lazy" src="${previewUrl(id)}" alt="">`).join("")}</div>`
    : `<div class="mosaic-empty">${esc(c.emoji || "📁")}</div>`;
  return h(`<a class="ccard fade-in" href="#/collection/${c.id}">
    ${media}${c.kind === "smart" ? '<span class="badge">✨ Smart</span>' : ""}
    <div class="ccard-body"><div class="ccard-emoji" style="${c.color ? `background:${c.color}33` : ""}">${esc(c.emoji || "📁")}</div>
    <div style="min-width:0"><div class="ccard-title">${esc(c.name)}</div><div class="ccard-sub">${plural(c.count, "photo")}</div></div></div></a>`);
}

function newCollectionCard(preset = {}) {
  const el = h(`<button class="ccard new"><span><span class="plus">+</span>New collection</span></button>`);
  el.onclick = () => openCollectionEditor(preset);
  return el;
}

async function renderCollections(view) {
  const cols = await api("/api/collections");
  const head = h(`<div class="page-head fade-in"><div><h1>Collections</h1><p>Your own albums and catalogues — hand-picked, or smart ones that fill themselves.</p></div>
    <div class="row-actions"><button class="btn primary">${I.plus} New collection</button></div></div>`);
  $(".btn", head).onclick = () => openCollectionEditor();
  view.append(head);
  if (!cols.length) {
    const e = emptyState("📚", "No collections yet", "Group photos however you like: a trip, a person, a year, a feeling. Pick a name to start:");
    const chips = h('<div class="chips" style="justify-content:center"></div>');
    COLLECTION_IDEAS.forEach((n) => {
      const c = h(`<button class="chip">${esc(n)}</button>`);
      c.onclick = () => openCollectionEditor({ name: n });
      chips.append(c);
    });
    e.append(chips);
    view.append(e);
    return;
  }
  const grid = h('<div class="card-grid"></div>');
  cols.forEach((c) => grid.append(collectionCard(c)));
  grid.append(newCollectionCard());
  view.append(grid);
}

async function renderCollection(view, id) {
  const c = await api(`/api/collections/${id}`);
  const cover = c.cover_photo_id && c.items.some((i) => i.id === c.cover_photo_id) ? c.cover_photo_id : c.items[0]?.id;
  const heroEl = h(`<section class="c-hero fade-in ${cover ? "" : "noimg"}">
      ${cover ? `<img class="bg" src="${previewUrl(cover)}" alt="">` : ""}
      <div class="c-hero-inner">
        <div><div class="e">${esc(c.emoji || "📁")}</div><h1>${esc(c.name)}</h1>
          <p>${plural(c.count, "photo")}${c.kind === "smart" ? ` · ✨ Smart: “${esc(c.query)}”` : ""}${c.description ? ` · ${esc(c.description)}` : ""}</p></div>
        <div class="row-actions"></div>
      </div></section>`);
  const actions = $(".row-actions", heroEl);
  const btn = (label, icon, fn, extra = "") => { const b = h(`<button class="btn small ${extra}">${icon} ${label}</button>`); b.onclick = fn; actions.append(b); return b; };
  if (c.kind === "manual") {
    btn("Add photos", I.plus, () => {
      state.target = { id: c.id, name: c.name, emoji: c.emoji };
      toast(`Search for photos, select them, then press <strong>Add to ${esc(c.name)}</strong>.`);
      go("#/");
    });
  }
  if (c.count) btn("Play", I.slides, () => playSlideshow(c.items, c.name, c.description || ""));
  btn("Edit", I.edit, () => openCollectionEditor({ existing: c }));
  if (c.count) {
    btn("Copy to folder", I.copy, () => openFolderBrowser({
      mode: "pick", title: `Copy “${c.name}” to a folder`, subfolder: c.name,
      onPick: (dest, subfolder) => copyCollection(c, dest, subfolder),
    }));
    const a = h(`<a class="btn small" href="/api/collections/${c.id}/export" download>${I.download} Zip</a>`);
    actions.append(a);
  }
  if (c.kind === "smart") {
    btn("Keep as is", I.star, async () => {
      if (!(await confirmDialog("Freeze this collection?", "It becomes a regular collection with the photos it holds right now, and stops updating itself. You can then add and remove photos by hand.", "Freeze"))) return;
      await api(`/api/collections/${c.id}/freeze`, { method: "POST" }).catch(fail);
      route();
    });
  }
  btn("Delete", I.trash, async () => {
    if (!(await confirmDialog(`Delete “${c.name}”?`, "Only the collection goes away. Your photos stay exactly where they are.", "Delete collection", true))) return;
    await api(`/api/collections/${c.id}`, { method: "DELETE" });
    if (state.target?.id === c.id) state.target = null;
    toast(`Deleted “${esc(c.name)}”`);
    go("#/collections");
  });
  view.append(heroEl);

  if (state.target?.id === c.id) state.target = null;

  if (!c.items.length) {
    view.append(c.kind === "smart"
      ? emptyState("✨", "Nothing matches yet", "This collection fills itself with photos that match its description. Try editing the description, or wait until more photos are indexed.")
      : emptyState(c.emoji || "📁", "This collection is empty", "Find photos with a search, select them, and add them here.", ""));
    if (c.kind === "manual") {
      const b = h(`<div style="text-align:center;margin-top:-40px;position:relative"><button class="btn primary">${I.plus} Find photos to add</button></div>`);
      $("button", b).onclick = () => actions.querySelector(".btn").click();
      view.append(b);
    }
    return;
  }
  const head = h(`<div class="section-head"><span class="muted">${c.kind === "smart" ? "Updates automatically as new photos are indexed." : "Select photos to remove them or pick a cover."}</span><button class="link">Select all</button></div>`);
  view.append(head);
  const grid = new JGrid(c.items, { coverId: c.cover_photo_id });
  view.append(grid.el);
  $(".link", head).onclick = () => grid.selectAll();

  const inThis = (ids) => ids.filter((i) => c.items.some((x) => x.id === i));
  state.trayActions = [
    { label: "Set as cover", icon: I.star, run: async (ids) => {
      const mine = inThis(ids);
      if (!mine.length) return;
      await api(`/api/collections/${c.id}`, { method: "PATCH", body: { cover_photo_id: mine[mine.length - 1] } }).catch(fail);
      clearSelection();
      route();
    } },
  ];
  if (c.kind === "manual") {
    state.trayActions.unshift({ label: "Remove from collection", icon: I.trash, run: async (ids) => {
      const mine = inThis(ids);
      await api(`/api/collections/${c.id}/photos/remove`, { method: "POST", body: { photo_ids: mine } }).catch(fail);
      toast(`Removed ${plural(mine.length, "photo")} from “${esc(c.name)}”`, {
        action: "Undo", onAction: async () => { await api(`/api/collections/${c.id}/photos`, { method: "POST", body: { photo_ids: mine } }); route(); },
      });
      clearSelection();
      route();
    } });
  }
}

async function addPhotosTo(col, ids) {
  try {
    const res = await api(`/api/collections/${col.id}/photos`, { method: "POST", body: { photo_ids: ids } });
    const dup = ids.length - res.added;
    toast(`Added ${plural(res.added, "photo")} to ${esc(col.emoji || "")} <strong>${esc(col.name)}</strong>${dup ? ` (${dup} already there)` : ""}`, {
      action: "View", onAction: () => go(`#/collection/${col.id}`),
    });
    clearSelection();
  } catch (e) { fail(e); }
}

// ------------------------------------------------------------------ modals
function openModal(html, { wide = false } = {}) {
  const root = $("#modal-root");
  const back = h(`<div class="modal-backdrop"><div class="modal ${wide ? "wide" : ""}" role="dialog" aria-modal="true">${html}</div></div>`);
  const close = () => { back.remove(); document.removeEventListener("keydown", onKey, true); };
  const onKey = (e) => { if (e.key === "Escape") { e.stopPropagation(); close(); } };
  back.addEventListener("mousedown", (e) => { if (e.target === back) close(); });
  document.addEventListener("keydown", onKey, true);
  root.append(back);
  $$("[data-close]", back).forEach((b) => (b.onclick = close));
  const modal = $(".modal", back);
  setTimeout(() => $("input:not([type=radio]), textarea, button.primary", modal)?.focus(), 30);
  return { el: modal, close };
}

function confirmDialog(title, text, ok = "OK", danger = false) {
  return new Promise((resolve) => {
    const m = openModal(`<div class="modal-head"><div><h2>${esc(title)}</h2><p>${esc(text)}</p></div></div>
      <div class="modal-foot"><button class="btn ghost" data-close>Cancel</button><button class="btn primary ok" style="${danger ? "background:#e5484d;color:#fff" : ""}">${esc(ok)}</button></div>`);
    const origClose = m.close;
    let done = false;
    m.close = () => { origClose(); if (!done) resolve(false); };
    $$("[data-close]", m.el).forEach((b) => (b.onclick = () => m.close()));
    $(".ok", m.el).onclick = () => { done = true; origClose(); resolve(true); };
  });
}

function openCollectionEditor({ existing = null, name = "", kind = "manual", query = "", photoIds = [], onCreated } = {}) {
  const c = existing || {};
  let emoji = c.emoji || EMOJIS[Math.floor(Math.random() * 8)];
  let color = c.color || COLORS[0];
  const isEdit = !!existing;
  kind = c.kind || kind;
  const m = openModal(`
    <div class="modal-head"><div><h2>${isEdit ? "Edit collection" : "New collection"}</h2>
      <p>${isEdit ? "Rename it, change its look or what it's about." : photoIds.length ? `Starting with ${plural(photoIds.length, "selected photo")}.` : "Give it a name you'll recognise."}</p></div>
      <button class="icon-btn" data-close aria-label="Close">${I.close}</button></div>
    <form class="modal-body" id="ced">
      <div class="field"><label for="c-name">Name</label><input class="input" id="c-name" maxlength="120" required placeholder="e.g. Summer in Crete" value="${esc(c.name || name)}"></div>
      ${isEdit ? "" : `<div class="field"><label>How should it fill up?</label><div class="seg">
        <label><input type="radio" name="kind" value="manual" ${kind === "manual" ? "checked" : ""}><strong>🖐️ I'll pick photos</strong><span>Add exactly the ones you choose.</span></label>
        <label><input type="radio" name="kind" value="smart" ${kind === "smart" ? "checked" : ""}><strong>✨ Smart</strong><span>Describe it; matching photos join automatically.</span></label>
      </div></div>`}
      <div class="field" id="q-field"><label for="c-query">Describe what belongs in it</label>
        <textarea class="input" id="c-query" placeholder="e.g. sunsets over the sea, golden sky">${esc(c.query || query)}</textarea>
        <small id="q-count"></small><div class="preview-strip" id="q-prev"></div></div>
      <div class="field"><label>Icon</label><div class="emoji-grid">${EMOJIS.map((e) => `<button type="button" class="${e === emoji ? "on" : ""}">${e}</button>`).join("")}</div></div>
      <div class="field"><label>Colour</label><div class="swatches">${COLORS.map((x) => `<button type="button" aria-label="${x}" style="background:${x}" class="${x === color ? "on" : ""}" data-c="${x}"></button>`).join("")}</div></div>
      ${isEdit ? `<div class="field"><label for="c-desc">Note <span class="muted">(optional)</span></label><input class="input" id="c-desc" maxlength="300" value="${esc(c.description || "")}" placeholder="A few words about it"></div>` : ""}
    </form>
    <div class="modal-foot"><button class="btn ghost" data-close>Cancel</button><button class="btn primary" form="ced" type="submit">${isEdit ? "Save" : "Create collection"}</button></div>`);
  const el = m.el;
  const qField = $("#q-field", el);
  const currentKind = () => (isEdit ? kind : $("input[name=kind]:checked", el).value);
  const syncKind = () => { qField.hidden = currentKind() !== "smart"; if (!qField.hidden) preview(); };
  $$("input[name=kind]", el).forEach((r) => (r.onchange = syncKind));
  $$(".emoji-grid button", el).forEach((b) => (b.onclick = () => { emoji = b.textContent; $$(".emoji-grid button", el).forEach((x) => x.classList.toggle("on", x === b)); }));
  $$(".swatches button", el).forEach((b) => (b.onclick = () => { color = b.dataset.c; $$(".swatches button", el).forEach((x) => x.classList.toggle("on", x === b)); }));
  const preview = debounce(async () => {
    const q = $("#c-query", el).value.trim();
    const box = $("#q-prev", el), count = $("#q-count", el);
    if (!q) { box.innerHTML = ""; count.textContent = "Lumen will keep it up to date as new photos arrive."; return; }
    try {
      const res = await api(`/api/search?q=${encodeURIComponent(q)}&limit=500`);
      const strong = res.items.filter((i) => i.tier !== "maybe");
      count.textContent = strong.length ? `≈ ${plural(strong.length, "photo")} match right now` : "No strong matches yet — try different words.";
      box.innerHTML = strong.slice(0, 12).map((i) => `<img src="${previewUrl(i.id)}" alt="">`).join("");
    } catch { /* ignore while typing */ }
  }, 450);
  $("#c-query", el).oninput = preview;
  syncKind();
  $("#ced", el).onsubmit = async (e) => {
    e.preventDefault();
    const body = { name: $("#c-name", el).value.trim(), emoji, color };
    if (currentKind() === "smart") body.query = $("#c-query", el).value.trim();
    try {
      let res;
      if (isEdit) {
        if ($("#c-desc", el)) body.description = $("#c-desc", el).value.trim() || null;
        res = await api(`/api/collections/${c.id}`, { method: "PATCH", body });
      } else {
        res = await api("/api/collections", { method: "POST", body: { ...body, kind: currentKind(), photo_ids: photoIds } });
      }
      m.close();
      if (onCreated) { onCreated(res); return; }
      toast(isEdit ? "Saved" : `Created ${esc(emoji)} <strong>${esc(res.name)}</strong>`);
      if (isEdit) route(); else go(`#/collection/${res.id}`);
    } catch (err) { fail(err); }
  };
}

async function openAddToCollection(ids) {
  const cols = (await api("/api/collections")).filter((c) => c.kind === "manual");
  const m = openModal(`
    <div class="modal-head"><div><h2>Add ${plural(ids.length, "photo")} to…</h2><p>Pick a collection, or start a new one.</p></div>
      <button class="icon-btn" data-close aria-label="Close">${I.close}</button></div>
    <div class="modal-body">
      ${cols.length > 5 ? '<input class="input" id="pick-filter" placeholder="Filter collections…" style="margin-bottom:10px">' : ""}
      <div class="pick-list"></div>
    </div>
    <div class="modal-foot"><button class="btn primary" id="pick-new">${I.plus} New collection with these</button></div>`);
  const list = $(".pick-list", m.el);
  const draw = (filter = "") => {
    list.innerHTML = "";
    cols.filter((c) => c.name.toLowerCase().includes(filter.toLowerCase())).forEach((c) => {
      const media = c.preview_ids?.length ? `<img src="${previewUrl(c.preview_ids[0])}" alt="">` : `<span class="ph">${esc(c.emoji || "📁")}</span>`;
      const b = h(`<button class="pick">${media}<span class="t"><strong>${esc(c.emoji || "")} ${esc(c.name)}</strong><span>${plural(c.count, "photo")}</span></span>${I.plus}</button>`);
      b.onclick = async () => { m.close(); await addPhotosTo(c, ids); };
      list.append(b);
    });
    if (!cols.length) list.append(h('<p class="muted" style="text-align:center;margin:20px 0">You have no collections yet — create your first one below.</p>'));
  };
  draw();
  $("#pick-filter", m.el)?.addEventListener("input", (e) => draw(e.target.value));
  $("#pick-new", m.el).onclick = () => {
    m.close();
    openCollectionEditor({ photoIds: ids, onCreated: (res) => {
      clearSelection();
      toast(`Created ${esc(res.emoji || "")} <strong>${esc(res.name)}</strong> with ${plural(res.count, "photo")}`, { action: "View", onAction: () => go(`#/collection/${res.id}`) });
      if (state.view === "collections" || state.view === "home") route();
    } });
  };
}

async function openFolderBrowser({ mode = "add", title, subfolder = "", onPick } = {}) {
  const added = [];
  const pick = mode === "pick";
  const m = openModal(`
    <div class="modal-head"><div><h2>${esc(title || "Choose photo folders")}</h2><p>${pick ? "Browse to where the copies should go. Your originals stay where they are." : "Browse to a folder and add it. Subfolders are included. Add as many as you like."}</p></div>
      <button class="icon-btn" data-close aria-label="Close">${I.close}</button></div>
    <div class="modal-body">
      <div class="fb"><div class="fb-roots"></div>
        <div class="fb-main"><div class="crumbs"></div><div class="fb-dirs"></div>
          <div class="fb-here"><span class="here-text"></span><button class="btn primary small" id="fb-add">${pick ? I.copy + " Copy here" : I.plus + " Add this folder"}</button></div>
          ${pick ? `<label class="fb-sub"><input type="checkbox" id="fb-sub-on" ${subfolder ? "checked" : ""}> Put them in a new folder called <input class="input" id="fb-sub" value="${esc(subfolder)}"></label>` : ""}
          <form id="fb-path" style="display:flex;gap:8px;margin-top:10px"><input class="input" placeholder="…or paste a folder path" style="height:38px"><button class="btn small">Go</button></form>
          <div class="added-list"></div>
        </div></div>
    </div>
    <div class="modal-foot"><span class="muted" style="margin-right:auto;font-size:13px">${pick ? "Existing files are never overwritten." : "🔒 Lumen only reads your photos. Nothing leaves this computer."}</span><button class="btn" data-close>${pick ? "Cancel" : "Done"}</button></div>`, { wide: true });
  const el = m.el;
  let cur = null;
  const open = async (path) => {
    try {
      const d = await api(`/api/fs${path ? `?path=${encodeURIComponent(path)}` : ""}`);
      cur = d;
      $(".fb-roots", el).innerHTML = "";
      d.roots.forEach((r) => {
        const b = h(`<button title="${esc(r.path)}">${r.name === "Home" ? "🏠" : r.name === "Pictures" || r.name === "Photos" ? "🖼️" : "💽"} ${esc(r.name)}</button>`);
        b.onclick = () => open(r.path);
        $(".fb-roots", el).append(b);
      });
      const crumbs = $(".crumbs", el);
      crumbs.innerHTML = "";
      const sep = d.path.includes("\\") ? "\\" : "/";
      const segs = d.path.split(/[\\/]/).filter(Boolean);
      let acc = d.path.startsWith("/") ? "" : "";
      const rootBtn = h(`<button>${sep === "/" ? "/" : segs[0]}</button>`);
      rootBtn.onclick = () => open(sep === "/" ? "/" : segs[0] + "\\");
      crumbs.append(rootBtn);
      (sep === "/" ? segs : segs.slice(1)).forEach((s, i, arr) => {
        acc = sep === "/" ? acc + "/" + s : (i === 0 ? segs[0] + "\\" + s : acc + "\\" + s);
        const target = acc;
        const b = h(`<button>${esc(s)}</button>`);
        b.onclick = () => open(target);
        crumbs.append(h('<span class="muted">›</span>'), b);
      });
      const dirs = $(".fb-dirs", el);
      dirs.innerHTML = "";
      if (d.parent) { const up = h("<button>⬆️ ..</button>"); up.onclick = () => open(d.parent); dirs.append(up); }
      d.dirs.forEach((x) => { const b = h(`<button>📁 ${esc(x.name)}</button>`); b.onclick = () => open(x.path); dirs.append(b); });
      if (!d.dirs.length) dirs.append(h('<p class="muted" style="padding:10px">No subfolders here.</p>'));
      $(".here-text", el).innerHTML = `<strong>${esc(d.path.split(/[\\/]/).filter(Boolean).pop() || d.path)}</strong> · ${d.images ? plural(d.images, "photo") + " right here" : "no photos directly here"}${d.dirs.length ? `, ${plural(d.dirs.length, "subfolder")}` : ""}`;
    } catch (e) { fail(e); }
  };
  $("#fb-add", el).onclick = async () => {
    if (pick) {
      const sub = $("#fb-sub-on", el).checked ? $("#fb-sub", el).value.trim() : "";
      m.close();
      onPick(cur.path, sub);
      return;
    }
    try {
      const r = await api("/api/folders", { method: "POST", body: { path: cur.path } });
      added.push(r.path);
      $(".added-list", el).append(h(`<span class="chip active">✓ ${esc(r.path.split(/[\\/]/).filter(Boolean).pop() || r.path)}</span>`));
      toast("Folder added — Lumen is scanning it now ✨");
      refreshStatus();
    } catch (e) { fail(e); }
  };
  $("#fb-path", el).onsubmit = (e) => { e.preventDefault(); const v = $("input", e.target).value.trim(); if (v) open(v); };
  const origClose = m.close;
  m.close = () => { origClose(); if (added.length && ["home", "sources"].includes(state.view)) route(); };
  $$("[data-close]", el).forEach((b) => (b.onclick = () => m.close()));
  const pics = (await api("/api/fs")).roots.find((r) => r.name === (pick ? "Desktop" : "Pictures"));
  open(pics?.path);
}

// ------------------------------------------------------------------ lightbox
const lb = { items: [], idx: 0, extra: null };

function openLightbox(items, idx) {
  lb.items = items;
  lb.idx = idx;
  $("#lightbox").hidden = false;
  document.body.style.overflow = "hidden";
  drawLightbox();
}
function closeLightbox() {
  $("#lightbox").hidden = true;
  $("#lightbox").innerHTML = "";  // stops any playing video
  document.body.style.overflow = "";
}
async function drawLightbox() {
  const it = lb.items[lb.idx];
  if (!it) return closeLightbox();
  const box = $("#lightbox");
  box.innerHTML = `
    <div class="lb-stage">
      ${it.kind === "video"
        ? `<video class="lb-video" src="/api/photos/${it.id}/original#t=${it.t || 0}" controls autoplay playsinline poster="${previewUrl(it.id)}"></video>`
        : `<img class="lb-low" src="${previewUrl(it.id)}" alt="">`}
      <button class="icon-btn lb-close" aria-label="Close (Esc)">${I.close}</button>
      ${lb.idx > 0 ? `<button class="lb-nav prev" aria-label="Previous">${I.left}</button>` : ""}
      ${lb.idx < lb.items.length - 1 ? `<button class="lb-nav next" aria-label="Next">${I.right}</button>` : ""}
    </div>
    <aside class="lb-side"><h3>${esc(it.name)}</h3><div class="lb-details muted">Loading…</div></aside>`;
  $(".lb-close", box).onclick = closeLightbox;
  $(".prev", box)?.addEventListener("click", () => step(-1));
  $(".next", box)?.addEventListener("click", () => step(1));
  $(".lb-stage", box).onclick = (e) => { if (e.target.classList.contains("lb-stage")) closeLightbox(); };
  const vid = $(".lb-video", box);
  if (vid) vid.onerror = () => vid.replaceWith(h(`<div class="lb-novideo"><img src="${previewUrl(it.id)}" alt=""><p>Your browser can't play this video format.</p><button class="btn primary small">Open in your video player</button></div>`));
  $(".lb-stage", box).addEventListener("click", (e) => { if (e.target.closest(".lb-novideo button")) api(`/api/photos/${it.id}/open`, { method: "POST" }).catch(fail); });
  const full = new Image();
  full.alt = it.name;
  full.onload = () => { const low = $(".lb-low", box); if (low && lb.items[lb.idx] === it) low.replaceWith(full); };
  if (!vid) full.src = `/api/photos/${it.id}/original`;

  const [d, sim] = await Promise.all([api(`/api/photos/${it.id}`), api(`/api/photos/${it.id}/similar?limit=9`).catch(() => [])]);
  if (lb.items[lb.idx] !== it) return;
  const side = $(".lb-side", box);
  const sel = isSelected(it.id);
  side.innerHTML = `
    <h3>${esc(d.name)}</h3>
    <div class="lb-actions">
      <button class="btn primary small" data-a="add">${I.collection} Add to collection</button>
      <button class="btn small" data-a="select">${sel ? "✓ Selected" : "Select"}</button>
      <button class="btn small" data-a="similar">${I.similar} Find similar</button>
      <button class="btn small" data-a="reveal">${I.folder} Show in folder</button>
      <a class="btn small" href="/api/photos/${it.id}/download">${I.download} Download</a>
    </div>
    <dl>
      ${d.taken_at ? `<dt>📅</dt><dd>${new Date(d.taken_at * 1000).toLocaleString(undefined, { dateStyle: "full", timeStyle: "short" })}</dd>` : ""}
      ${d.camera ? `<dt>📷</dt><dd>${esc(d.camera)}</dd>` : ""}
      <dt>📁</dt><dd>${esc(d.folder)}</dd>
      ${d.lat != null ? `<dt>📍</dt><dd><a href="#/map?photo=${d.id}">${esc(d.place || `${d.lat.toFixed(4)}, ${d.lon.toFixed(4)}`)} · View on map</a></dd>` : ""}
      ${d.kind === "video" ? `<dt>🎬</dt><dd>Video · ${fmtClock(d.duration)}${it.t != null ? ` · best moment at ${fmtClock(it.t)}` : ""}</dd>` : ""}
      ${d.size ? `<dt>💾</dt><dd>${d.size >= 1048576 ? (d.size / 1048576).toFixed(1) + " MB" : Math.max(1, Math.round(d.size / 1024)) + " KB"}</dd>` : ""}
      ${d.quality != null ? `<dt>⭐</dt><dd>${d.quality >= 90 ? "One of your best shots · " : d.quality >= 70 ? "Looks great · " : ""}<span class="muted">Quality ${d.quality}/100</span></dd>` : ""}
      ${it.tier ? `<dt>🎯</dt><dd>${{ best: "Spot-on match", good: "Strong match", maybe: "Possible match" }[it.tier]}</dd>` : ""}
    </dl>
    ${d.people.length ? `<div><h4>People</h4><div class="lb-people">${d.people.map((p) => `<a class="lb-person" href="${p.id ? `#/person/${p.id}` : "#/people"}"><span class="lb-face"><img src="${faceUrl(p.face_id)}" alt="">${p.expression_emoji ? `<i title="${esc(p.expression)}">${p.expression_emoji}</i>` : ""}</span><span>${p.id ? esc(p.name || "Add a name") : "Not grouped yet"}</span></a>`).join("")}</div></div>` : ""}
    ${d.collections.length ? `<div><h4>In collections</h4><div class="chips">${d.collections.map((c) => `<a class="chip" href="#/collection/${c.id}">${esc(c.emoji || "📁")} ${esc(c.name)}</a>`).join("")}</div></div>` : ""}
    ${sim.length ? `<div><h4>Looks similar</h4><div class="lb-similar">${sim.map((s, i) => `<img src="${previewUrl(s.id)}" data-i="${i}" alt="${esc(s.name)}">`).join("")}</div></div>` : ""}
    <div class="muted" style="font-size:12px;margin-top:auto">← → to browse · Esc to close</div>`;
  side.onclick = async (e) => {
    const a = e.target.closest("[data-a]")?.dataset.a;
    const simImg = e.target.closest(".lb-similar img");
    if (simImg) { openLightbox(sim, Number(simImg.dataset.i)); return; }
    if (a === "add") openAddToCollection([it.id]);
    if (a === "similar") go(`#/similar/${it.id}`);
    if (a === "reveal") api(`/api/photos/${it.id}/reveal`, { method: "POST" }).then(() => toast("Opened in your file manager")).catch(fail);
    if (a === "select") { setSelected(it, !isSelected(it.id)); renderTray(); e.target.closest("button").textContent = isSelected(it.id) ? "✓ Selected" : "Select"; }
  };
}
function step(d) {
  const n = lb.idx + d;
  if (n >= 0 && n < lb.items.length) { lb.idx = n; drawLightbox(); }
}

// ------------------------------------------------------------------ sources
async function renderSources(view) {
  const [folders, settings] = await Promise.all([api("/api/folders"), api("/api/settings")]);
  view.append(h(`<div class="page-head fade-in"><div><h1>Sources</h1><p>The folders Lumen looks after, and how it's getting on.</p></div>
    <div class="row-actions"><button class="btn" id="rescan">${I.refresh} Check for new photos</button><button class="btn primary" id="add-folder">${I.folder} Add folders</button></div></div>`));
  const grid = h('<div class="two-col"></div>');
  const left = h(`<div class="panel fade-in"><h2>Photo folders</h2><p class="muted" style="margin:0">Subfolders are included. Removing a folder only forgets it — your files are never touched.</p><div class="folder-list"></div></div>`);
  const list = $(".folder-list", left);
  if (!folders.length) list.append(emptyState("🗂️", "No folders yet", "Add the folders where your photos live.", ""));
  for (const f of folders) {
    const pct = f.photos ? (100 * f.ready) / f.photos : 0;
    const row = h(`<div class="folder"><div class="ficon">${f.exists ? "📁" : "⚠️"}</div>
      <div class="meta"><strong>${esc(f.name)}</strong><div class="path" title="${esc(f.path)}">&lrm;${esc(f.path)}</div>
        <div class="muted" style="font-size:13px">${f.exists ? "" : "Not reachable right now (unplugged drive?) · "}${plural(f.photos, "photo")}${f.pending ? ` · ${fmt(f.pending)} to go` : ""}${f.errors ? ` · ${fmt(f.errors)} couldn't be read` : ""}</div>
        ${f.pending ? `<div class="bar"><i style="width:${pct}%"></i></div>` : ""}</div>
      <button class="icon-btn" title="Remove folder" aria-label="Remove folder">${I.trash}</button></div>`);
    $("button", row).onclick = async () => {
      if (!(await confirmDialog(`Stop watching “${f.name}”?`, `Lumen will forget its ${plural(f.photos, "photo")} and remove them from your collections. The files on disk are not touched.`, "Remove folder", true))) return;
      await api(`/api/folders/${f.id}`, { method: "DELETE" }).catch(fail);
      toast(`Removed “${esc(f.name)}”`);
      route();
    };
    list.append(row);
  }
  const right = h(`<div><div class="panel fade-in" id="index-panel"></div>
    <div class="panel fade-in"><h2>Understanding model</h2><p class="muted" style="margin:0">Runs entirely on this computer. Switching re-reads every photo (previews are reused, so it's quicker than the first time).</p>
      <form id="model-form">${settings.models.map((x) => `<label class="model-opt"><input type="radio" name="model" value="${x.key}" ${x.key === settings.model ? "checked" : ""}><span><strong>${esc(x.label)}</strong><br><span class="muted">${esc(x.description)}</span></span></label>`).join("")}</form>
      <p class="muted" style="font-size:13px;margin:14px 0 0">🔒 Library and previews are stored in <code>${esc(settings.data_dir)}</code></p></div></div>`);
  $$("input[name=model]", right).forEach((r) => (r.onchange = async () => {
    const ok = await confirmDialog("Switch model?", "Lumen will re-index all photos with the new model. Search keeps working with the photos already done.", "Switch");
    if (!ok) { $(`input[value="${settings.model}"]`, right).checked = true; return; }
    await api("/api/settings", { method: "PUT", body: { model: r.value } }).catch(fail);
    toast("Switching model — re-indexing in the background");
    route();
  }));
  grid.append(left, right);
  view.append(grid);
  $("#add-folder").onclick = () => openFolderBrowser();
  $("#rescan").onclick = async () => { await api("/api/scan", { method: "POST" }); toast("Looking for new and changed photos…"); refreshStatus(); };
  drawIndexPanel();
}

function drawIndexPanel() {
  const p = $("#index-panel");
  const s = state.status;
  if (!p || !s) return;
  const pct = s.total ? Math.round((100 * s.ready) / s.total) : 0;
  const label = { idle: s.pending || s.enrich_pending ? "Waiting" : "All caught up ✓", scanning: "Looking for photos…", indexing: "Learning your photos…", enriching: "Finding faces & places…", paused: "Paused" }[s.state] || s.state;
  p.innerHTML = `<div style="display:flex;justify-content:space-between;align-items:center;gap:10px"><h2>${label}</h2>
      ${s.pending || s.enrich_pending || s.state === "paused" ? `<button class="btn small" id="pp">${s.state === "paused" ? I.play + " Resume" : I.pause + " Pause"}</button>` : ""}</div>
    <div class="bar" style="height:10px"><i style="width:${pct}%"></i></div>
    <div class="stats">
      <div class="stat"><b>${fmt(s.ready)}</b><span>photos understood</span></div>
      <div class="stat"><b>${fmt(s.pending)}</b><span>still to go</span></div>
      ${s.enrich_pending ? `<div class="stat"><b>${fmt(s.enrich_pending)}</b><span>to check for faces & places</span></div>` : ""}
      <div class="stat"><b>${fmt(s.people)}</b><span>${s.faces_state === "ready" || s.faces_state === "not loaded" ? "people recognised" : esc(s.faces_state.startsWith("downloading") ? "people · downloading face models" : s.faces_state === "off" ? "people · face grouping off" : "people · faces " + s.faces_state)}</span></div>
      <div class="stat"><b>${s.rate ? s.rate.toFixed(1) + "/s" : "—"}</b><span>${s.eta_seconds ? fmtDuration(s.eta_seconds) + " left" : "speed"}</span></div>
      ${s.errors ? `<div class="stat"><b>${fmt(s.errors)}</b><span>unreadable · <button class="link" id="retry">retry</button></span></div>` : ""}
    </div>
    <p class="muted" style="font-size:13px;margin:12px 0 0;word-break:break-all">${s.state === "scanning" ? `Scanned ${fmt(s.scanned_files)} files · ` : ""}${s.current ? esc(s.current) : ""}${s.model_state !== "ready" ? ` · Model: ${esc(s.model_state)}` : ""}${s.last_error ? `<br>⚠️ ${esc(s.last_error)}` : ""}</p>`;
  $("#pp", p)?.addEventListener("click", async () => {
    state.status = await api(`/api/index/${s.state === "paused" ? "resume" : "pause"}`, { method: "POST" });
    refreshStatus();
  });
  $("#retry", p)?.addEventListener("click", async () => { await api("/api/index/retry-errors", { method: "POST" }); refreshStatus(); });
}

// ------------------------------------------------------------------ copy to folder
async function copyCollection(c, dest, subfolder) {
  try {
    const r = await api(`/api/collections/${c.id}/copy`, { method: "POST", body: { dest, subfolder: subfolder || null } });
    toast(`Copying ${plural(r.total, "photo")} to <strong>${esc(r.dest)}</strong>…`);
    for (;;) {
      await new Promise((ok) => setTimeout(ok, 700));
      const j = await api(`/api/jobs/${r.job_id}`);
      if (j.state === "done") {
        const miss = j.result.missing ? ` (${j.result.missing} originals were missing)` : "";
        toast(`✅ Copied ${plural(j.result.copied, "photo")} to <strong>${esc(j.result.dest)}</strong>${miss}`, { ms: 7000 });
        return;
      }
      if (j.state === "failed") throw new Error(j.error);
    }
  } catch (e) { fail(e); }
}

// ------------------------------------------------------------------ people
function personChip(p) {
  return h(`<a class="person fade-in" href="#/person/${p.id}">
    <span class="avatar">${p.face_id ? `<img loading="lazy" src="${faceUrl(p.face_id)}" alt="">` : I.person}</span>
    <strong class="${p.name ? "" : "muted"}">${esc(p.name || "Add a name")}</strong>
    <span class="muted">${plural(p.count, "photo")}</span></a>`);
}

async function renderPeople(view, params) {
  const hidden = params.get("hidden") === "1";
  const [people, st] = await Promise.all([api(`/api/people${hidden ? "?hidden=true" : ""}`), refreshStatus()]);
  view.append(h(`<div class="page-head fade-in"><div><h1>${hidden ? "Hidden people" : "People"}</h1>
    <p>${hidden ? "People you've hidden. Open one to bring them back." : "Lumen groups faces it recognises across your photos. Give them names, then search for them: “Maria at the beach”."}</p></div>
    <div class="row-actions"><a class="btn small" href="#/people${hidden ? "" : "?hidden=1"}">${hidden ? "Back to people" : I.eyeoff + " Hidden"}</a></div></div>`));
  if (!people.length) {
    const busy = st.enrich_pending || st.pending;
    const off = st.faces_state === "off" || st.faces_state.startsWith("unavailable");
    view.append(hidden ? emptyState("🙈", "Nobody is hidden", "")
      : off ? emptyState("🙂", "Face grouping is unavailable", esc(st.faces_state === "off" ? "It was switched off (LUMEN_FACES=0)." : st.faces_state))
      : emptyState(busy ? "⏳" : "🙂", busy ? "Looking for faces…" : "No people yet", busy
        ? `Lumen is still going through your photos (${fmt(st.enrich_pending + st.pending)} to go). People appear here as soon as someone shows up in two or more photos.`
        : "Nobody appears in more than one photo yet. As you add more folders, people will show up here."));
    return;
  }
  const named = people.filter((p) => p.name), unnamed = people.filter((p) => !p.name);
  for (const [title, list] of [["Named", named], [named.length ? "Who are they?" : "", unnamed]]) {
    if (!list.length) continue;
    const sec = h(`<section class="section fade-in" style="margin-top:0">${title && named.length ? `<div class="section-head"><h2>${title}</h2></div>` : ""}<div class="people-grid"></div></section>`);
    list.forEach((p) => $(".people-grid", sec).append(personChip(p)));
    view.append(sec);
  }
}

async function renderPerson(view, id) {
  const p = await api(`/api/people/${id}`);
  const head = h(`<section class="person-hero fade-in">
      <span class="avatar big">${p.face_id ? `<img src="${faceUrl(p.face_id)}" alt="">` : I.person}</span>
      <div class="person-meta">
        <input class="name-input" maxlength="80" placeholder="Who is this?" value="${esc(p.name || "")}" aria-label="Name">
        <p class="muted">${plural(p.count, "photo")}${p.hidden ? " · hidden" : ""} · <span class="save-hint">${p.name ? "Click the name to rename" : "Type a name and press Enter"}</span></p>
      </div>
      <div class="row-actions"></div></section>`);
  const input = $(".name-input", head);
  const save = async () => {
    const name = input.value.trim();
    if (name === (p.name || "")) return;
    try {
      await api(`/api/people/${id}`, { method: "PATCH", body: { name } });
      p.name = name;
      $(".save-hint", head).textContent = name ? "Click the name to rename" : "Type a name and press Enter";
      toast(name ? `Saved. Try searching “${esc(name)} smiling” ✨` : "Name removed");
    } catch (e) {
      if (e.status === 409) {
        const other = (await api("/api/people")).find((x) => (x.name || "").toLowerCase() === name.toLowerCase());
        if (other && (await confirmDialog(`Merge with ${name}?`, `Someone is already called “${name}”. Are these the same person?`, "Yes, merge"))) {
          await api("/api/people/merge", { method: "POST", body: { into: other.id, person_ids: [id] } });
          go(`#/person/${other.id}`);
          return;
        }
        input.value = p.name || "";
      } else fail(e);
    }
  };
  input.onkeydown = (e) => { if (e.key === "Enter") input.blur(); if (e.key === "Escape") { input.value = p.name || ""; input.blur(); } };
  input.onblur = save;
  const actions = $(".row-actions", head);
  const btn = (label, icon, fn) => { const b = h(`<button class="btn small">${icon} ${label}</button>`); b.onclick = fn; actions.append(b); };
  btn("Same as…", I.merge, () => openMergePeople(p));
  btn(p.hidden ? "Unhide" : "Hide", I.eyeoff, async () => {
    await api(`/api/people/${id}`, { method: "PATCH", body: { hidden: !p.hidden } }).catch(fail);
    toast(p.hidden ? "Back in People" : "Hidden. Find them under People → Hidden");
    go(p.hidden ? `#/person/${id}` : "#/people");
  });
  if (p.name) {
    const moods = [["😊", "smiling"], ["😂", "laughing"], ["😮", "surprised"], ["😐", "serious"], ["😢", "sad"]];
    $(".person-meta", head).append(h(`<div class="chips" style="margin-top:10px">${moods.map(([e, w]) =>
      `<a class="chip" href="#/search?q=${encodeURIComponent(p.name + " " + w)}">${e} ${esc(p.name)} ${w}</a>`).join("")}</div>`));
  }
  view.append(head);
  if (!p.name) setTimeout(() => input.focus(), 50);

  const sub = h(`<div class="section-head"><span class="muted">Spotted someone else? Select those photos and choose <strong>Not them</strong>.</span><button class="link">Select all</button></div>`);
  const grid = new JGrid(p.items);
  $(".link", sub).onclick = () => grid.selectAll();
  view.append(sub, grid.el);
  state.trayActions = [{ label: "Not them", icon: I.close, run: async (ids) => {
    const mine = ids.filter((i) => p.items.some((x) => x.id === i));
    const r = await api(`/api/people/${id}/remove-photos`, { method: "POST", body: { photo_ids: mine } }).catch(fail);
    clearSelection();
    toast(`Removed ${plural(mine.length, "photo")} from ${esc(p.name || "this person")}`);
    if (r && !r.remaining) go("#/people"); else route();
  } }];
}

async function openMergePeople(p) {
  const others = (await api("/api/people")).filter((x) => x.id !== p.id);
  const chosen = new Set();
  const m = openModal(`
    <div class="modal-head"><div><h2>Who else is ${esc(p.name || "this")}?</h2><p>Pick the groups that are the same person. Lumen sometimes splits one person into several groups (e.g. as a child and as an adult).</p></div>
      <button class="icon-btn" data-close aria-label="Close">${I.close}</button></div>
    <div class="modal-body"><div class="people-grid small"></div></div>
    <div class="modal-foot"><button class="btn ghost" data-close>Cancel</button><button class="btn primary" id="do-merge" disabled>Merge</button></div>`, { wide: true });
  const grid = $(".people-grid", m.el);
  if (!others.length) grid.append(h('<p class="muted">There is nobody else to merge with yet.</p>'));
  others.forEach((o) => {
    const c = personChip(o);
    c.removeAttribute("href");
    c.onclick = () => {
      chosen.has(o.id) ? chosen.delete(o.id) : chosen.add(o.id);
      c.classList.toggle("on", chosen.has(o.id));
      $("#do-merge", m.el).disabled = !chosen.size;
      $("#do-merge", m.el).textContent = chosen.size ? `Merge ${chosen.size + 1} groups` : "Merge";
    };
    grid.append(c);
  });
  $("#do-merge", m.el).onclick = async () => {
    await api("/api/people/merge", { method: "POST", body: { into: p.id, person_ids: [...chosen] } }).catch(fail);
    m.close();
    toast("Merged ✨");
    route();
  };
}

// ------------------------------------------------------------------ places (map)
let leafletReady = null;
function loadLeaflet() {
  leafletReady ||= new Promise((resolve, reject) => {
    for (const href of ["vendor/leaflet/leaflet.css", "vendor/leaflet/MarkerCluster.css"]) {
      document.head.append(h(`<link rel="stylesheet" href="${href}">`));
    }
    const s1 = document.createElement("script");
    s1.src = "vendor/leaflet/leaflet.js";
    s1.onload = () => {
      const s2 = document.createElement("script");
      s2.src = "vendor/leaflet/leaflet.markercluster.js";
      s2.onload = resolve;
      s2.onerror = reject;
      document.head.append(s2);
    };
    s1.onerror = reject;
    document.head.append(s1);
  });
  return leafletReady;
}

function isDark() {
  const t = document.documentElement.dataset.theme;
  return t ? t === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
}

async function renderMap(view, params) {
  const [{ points }] = await Promise.all([api("/api/map"), loadLeaflet()]);
  view.append(h(`<div class="page-head fade-in"><div><h1>Places</h1><p>${points.length ? `${plural(points.length, "photo")} with a location. Zoom in to explore; the photos below follow the map.` : ""}</p></div></div>`));
  if (!points.length) {
    const st = state.status || {};
    view.append(emptyState("🗺️", st.enrich_pending ? "Still reading locations…" : "No locations yet",
      st.enrich_pending ? `Lumen is reading where your photos were taken (${fmt(st.enrich_pending)} to go).`
        : "None of your photos has GPS information. Phones usually add it when location is enabled for the camera."));
    return;
  }
  const placeList = await api("/api/places").catch(() => []);
  const chips = h('<div class="chips" style="margin-bottom:14px"></div>');
  view.append(chips);
  const mapEl = h('<div class="map fade-in" id="map"></div>');
  const below = h(`<section class="section"><div class="group-head"><h3>In this area</h3><span class="sub"></span><span style="flex:1"></span><button class="link">Select all</button></div><div class="area"></div></section>`);
  view.append(mapEl, below);
  const map = L.map(mapEl, { zoomControl: true, worldCopyJump: true });
  L.tileLayer(`https://{s}.basemaps.cartocdn.com/${isDark() ? "dark_all" : "rastertiles/voyager"}/{z}/{x}/{y}{r}.png`, {
    maxZoom: 19, subdomains: "abcd",
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/attributions">CARTO</a>',
  }).addTo(map);
  const thumbIcon = (id, count) => L.divIcon({
    className: "map-pin",
    html: `<div class="map-thumb"><img src="${previewUrl(id)}" alt="">${count > 1 ? `<b>${count > 999 ? Math.round(count / 1000) + "k" : count}</b>` : ""}</div>`,
    iconSize: [52, 52], iconAnchor: [26, 26],
  });
  const cluster = L.markerClusterGroup({
    showCoverageOnHover: false, maxClusterRadius: 60,
    iconCreateFunction: (c) => thumbIcon(c.getAllChildMarkers()[0].options.photoId, c.getChildCount()),
  });
  for (const [id, lat, lon] of points) {
    const mk = L.marker([lat, lon], { icon: thumbIcon(id, 1), photoId: id });
    mk.on("click", async () => openLightbox(await api("/api/photos/batch", { method: "POST", body: { photo_ids: [id] } }), 0));
    cluster.addLayer(mk);
  }
  map.addLayer(cluster);
  placeList.slice(0, 16).forEach((pl) => {
    const c = h(`<button class="chip">${esc(pl.city)}${pl.country ? ` <span class="n">${esc(pl.country)}</span>` : ""} <span class="n">${fmt(pl.n)}</span></button>`);
    c.onclick = () => map.flyTo([pl.lat, pl.lon], 12);
    chips.append(c);
  });
  const focus = Number(params.get("photo"));
  const fp = focus && points.find((p) => p[0] === focus);
  if (fp) map.setView([fp[1], fp[2]], 15);
  else map.fitBounds(L.latLngBounds(points.map((p) => [p[1], p[2]])), { padding: [40, 40], maxZoom: 13 });

  let grid = null;
  const refresh = debounce(async () => {
    if (!mapEl.isConnected) return;
    const b = map.getBounds();
    const inside = points.filter((p) => b.contains([p[1], p[2]])).map((p) => p[0]);
    $(".sub", below).textContent = inside.length > 300 ? `${fmt(inside.length)} photos · showing the 300 most recent` : plural(inside.length, "photo");
    const items = inside.length ? await api("/api/photos/batch", { method: "POST", body: { photo_ids: inside.slice(0, 300) } }) : [];
    const area = $(".area", below);
    area.innerHTML = "";
    grid = new JGrid(items);
    area.append(items.length ? grid.el : h('<p class="muted">Move the map to see photos here.</p>'));
  }, 350);
  $(".link", below).onclick = () => grid?.selectAll();
  map.on("moveend", refresh);
  refresh();
}

// ------------------------------------------------------------------ tidy up (duplicates & similar shots)
async function renderTidy(view, params) {
  const kind = params.get("kind") === "bursts" ? "bursts" : "duplicates";
  const res = await api(`/api/duplicates?kind=${kind}`);
  const sm = res.summary;
  view.append(h(`<div class="page-head fade-in"><div><h1>Tidy up</h1>
    <p>Copies and near-identical shots, with the best one picked for you. Anything you remove goes to your computer's Trash / Recycle Bin, so you can always get it back.</p></div></div>`));
  view.append(h(`<div class="chips" style="margin-bottom:22px">
    <a class="chip ${kind === "duplicates" ? "active" : ""}" href="#/tidy">🧬 Duplicates <span class="n">${fmt(sm.duplicate_groups)}</span></a>
    <a class="chip ${kind === "bursts" ? "active" : ""}" href="#/tidy?kind=bursts">🎞️ Similar shots <span class="n">${fmt(sm.burst_groups)}</span></a></div>`));
  if (!res.groups.length) {
    view.append(emptyState("✨", kind === "duplicates" ? "No duplicates" : "No similar shots",
      kind === "duplicates" ? "Every photo in your library is one of a kind." : "No bursts of near-identical photos found."));
    return;
  }
  // Which photos are marked for removal. Duplicates default to "keep the best"; similar shots to nothing.
  const marked = new Set();
  if (kind === "duplicates") res.groups.forEach((g) => g.photo_ids.slice(1).forEach((i) => marked.add(i)));
  const bar = h(`<div class="tidy-bar panel"><span class="tidy-count"></span><button class="btn primary small" id="trash-all">${I.trash} Move to Trash</button></div>`);
  const syncBar = () => {
    $(".tidy-count", bar).innerHTML = marked.size ? `<strong>${plural(marked.size, "photo")}</strong> marked to remove` : "Click photos to mark the ones you don't need";
    $("#trash-all", bar).disabled = !marked.size;
  };
  view.append(bar);
  const list = h('<div class="tidy-list"></div>');
  view.append(list);
  for (const g of res.groups) {
    const card = h(`<div class="tidy-group panel fade-in">
      <div class="tidy-head"><strong>${g.kind === "identical" ? "Identical copies" : g.kind === "near" ? "Same photo, different versions" : "Similar shots"}</strong>
        <span class="muted">${plural(g.photos.length, "photo")}${g.photos[0]?.taken_at ? " · " + fmtDate(g.photos[0].taken_at) : ""}</span>
        <span style="flex:1"></span><button class="link" data-a="ignore">${kind === "duplicates" ? "Not duplicates" : "Keep all"}</button></div>
      <div class="tidy-row"></div></div>`);
    const row = $(".tidy-row", card);
    g.photos.forEach((p, idx) => {
      const t = h(`<button class="tidy-photo ${marked.has(p.id) ? "remove" : ""}" title="${esc(p.name)}">
        <img loading="lazy" src="${previewUrl(p.id)}" alt="">
        ${idx === 0 ? '<span class="best">★ Best</span>' : ""}
        <span class="verdict"></span>
        <span class="info"><b>${esc(p.name)}</b><span>${fmtBytes(p.size)} · ${esc(p.dir)}</span></span></button>`);
      const sync = () => { t.classList.toggle("remove", marked.has(p.id)); $(".verdict", t).textContent = marked.has(p.id) ? "Remove" : "Keep"; };
      sync();
      t.onclick = () => {
        if (marked.has(p.id)) marked.delete(p.id);
        else if (g.photo_ids.filter((i) => !marked.has(i)).length > 1) marked.add(p.id);
        else { toast("Keep at least one photo from each group"); return; }
        sync(); syncBar();
      };
      t.ondblclick = () => openLightbox(g.photos, idx);
      row.append(t);
    });
    $("[data-a=ignore]", card).onclick = async () => {
      await api("/api/duplicates/ignore", { method: "POST", body: { key: g.key } }).catch(fail);
      g.photo_ids.forEach((i) => marked.delete(i));
      card.remove();
      syncBar();
    };
    list.append(card);
  }
  if (res.total_groups > res.groups.length) list.append(h(`<p class="muted">Showing the first ${res.groups.length} of ${fmt(res.total_groups)} groups. Clear these to see more.</p>`));
  $("#trash-all", bar).onclick = async () => {
    const ids = [...marked];
    if (!(await confirmDialog(`Move ${plural(ids.length, "photo")} to the Trash?`, "The files go to your computer's Trash / Recycle Bin, where you can restore them. Lumen will forget them.", "Move to Trash", true))) return;
    try {
      const r = await api("/api/photos/trash", { method: "POST", body: { photo_ids: ids } });
      toast(`🧹 Moved ${plural(r.trashed, "photo")} to the Trash${r.failed.length ? ` · ${r.failed.length} couldn't be moved` : ""}`);
      route();
    } catch (e) { fail(e); }
  };
  syncBar();
}

// ------------------------------------------------------------------ memories (trips & events)
function memoryCard(m) {
  const ids = m.covers.map((c) => c.id);
  return h(`<a class="memory fade-in" href="#/memory/${m.key}">
    <div class="memory-media mosaic n${Math.min(ids.length, 3)}">${ids.slice(0, 3).map((id) => `<img loading="lazy" src="${previewUrl(id)}" alt="">`).join("")}</div>
    <span class="badge">${m.kind === "trip" ? "🧳 Trip" : "✨ Event"}</span>
    <div class="memory-body"><strong>${esc(m.title)}</strong><span>${esc(m.when)} · ${plural(m.count, "photo")}</span></div></a>`);
}

async function renderMemories(view, params) {
  const kind = params.get("kind") || "";
  const res = await api(`/api/memories?limit=200${kind ? `&kind=${kind}` : ""}`);
  view.append(h(`<div class="page-head fade-in"><div><h1>Memories</h1><p>Your library, told as trips and days out. Lumen builds these from when and where photos were taken, and who is in them.</p></div></div>`));
  view.append(h(`<div class="chips" style="margin-bottom:22px">
    <a class="chip ${!kind ? "active" : ""}" href="#/memories">All</a>
    <a class="chip ${kind === "trip" ? "active" : ""}" href="#/memories?kind=trip">🧳 Trips</a>
    <a class="chip ${kind === "event" ? "active" : ""}" href="#/memories?kind=event">✨ Events</a></div>`));
  if (!res.items.length) {
    view.append(emptyState("🧳", "No memories yet", "Memories appear once there are groups of photos taken close together in time. Trips need photos with locations."));
    return;
  }
  let year = null, grid = null;
  for (const m of res.items) {
    const y = new Date(m.start * 1000).getFullYear();
    if (y !== year) {
      year = y;
      view.append(h(`<div class="group-head"><h3>${y}</h3></div>`));
      grid = h('<div class="memory-grid"></div>');
      view.append(grid);
    }
    grid.append(memoryCard(m));
  }
}

async function renderMemory(view, key) {
  const m = await api(`/api/memories/${key}`);
  const cover = m.covers[0]?.id;
  const heroEl = h(`<section class="c-hero fade-in ${cover ? "" : "noimg"}">
      ${cover ? `<img class="bg" src="${previewUrl(cover)}" alt="">` : ""}
      <div class="c-hero-inner"><div><div class="e">${m.kind === "trip" ? "🧳" : "✨"}</div><h1>${esc(m.title)}</h1>
        <p>${esc(m.when)} · ${plural(m.count, "photo")}${m.place ? ` · ${esc(m.place)}` : ""}</p></div>
        <div class="row-actions"></div></div></section>`);
  const actions = $(".row-actions", heroEl);
  const play = h(`<button class="btn small">${I.slides} Play</button>`);
  play.onclick = () => playSlideshow(m.items, m.title, m.when);
  const save = h(`<button class="btn small">${I.collection} Save as collection</button>`);
  save.onclick = async () => {
    try {
      const c = await api(`/api/memories/${key}/save`, { method: "POST" });
      toast(`Saved as <strong>${esc(c.name)}</strong>`, { action: "Open", onAction: () => go(`#/collection/${c.id}`) });
    } catch (e) { fail(e); }
  };
  actions.append(play, save);
  view.append(heroEl);
  const head = h('<div class="section-head"><span class="muted">Photos in order, as the day unfolded.</span><button class="link">Select all</button></div>');
  const grid = new JGrid(m.items);
  $(".link", head).onclick = () => grid.selectAll();
  view.append(head, grid.el);
}

// ------------------------------------------------------------------ slideshow
function playSlideshow(items, title = "", subtitle = "") {
  items = items.filter(Boolean);
  if (!items.length) return;
  const el = h(`<div class="slideshow" role="dialog" aria-label="Slideshow">
      <div class="ss-stage"></div>
      <div class="ss-title"><h2>${esc(title)}</h2>${subtitle ? `<p>${esc(subtitle)}</p>` : ""}</div>
      <div class="ss-bar"><i></i></div>
      <div class="ss-controls">
        <button class="icon-btn" data-a="prev" aria-label="Previous">${I.left}</button>
        <button class="icon-btn" data-a="pause" aria-label="Pause">${I.pause}</button>
        <button class="icon-btn" data-a="next" aria-label="Next">${I.right}</button>
        <span class="ss-count"></span>
        <button class="icon-btn" data-a="close" aria-label="Close">${I.close}</button>
      </div></div>`);
  document.body.append(el);
  document.body.style.overflow = "hidden";
  el.requestFullscreen?.().catch(() => {});
  const stage = $(".ss-stage", el);
  const SLIDE_MS = 5000;
  let i = -1, timer = null, paused = false;
  const src = (it) => (it.kind === "video" ? previewUrl(it.id) : `/api/photos/${it.id}/original`);
  const preload = (k) => { if (items[k]) { const im = new Image(); im.src = src(items[k]); } };
  const show = (k) => {
    i = (k + items.length) % items.length;
    // A real document element: images inside a <template> don't load until adopted.
    const img = document.createElement("img");
    img.className = "ss-img";
    img.alt = "";
    // Ken Burns: a slow zoom towards a random corner.
    const ox = 30 + Math.random() * 40, oy = 30 + Math.random() * 40;
    img.style.transformOrigin = `${ox}% ${oy}%`;
    img.style.animationDuration = `${SLIDE_MS + 1500}ms`;
    img.onload = () => {
      stage.append(img);
      requestAnimationFrame(() => img.classList.add("in"));
      [...stage.children].slice(0, -1).forEach((old) => { old.classList.remove("in"); setTimeout(() => old.remove(), 1400); });
    };
    img.onerror = () => { img.src = previewUrl(items[i].id); };
    img.src = src(items[i]);
    $(".ss-count", el).textContent = `${i + 1} / ${items.length}`;
    const bar = $(".ss-bar i", el);
    bar.style.transition = "none";
    bar.style.width = "0";
    requestAnimationFrame(() => { bar.style.transition = paused ? "none" : `width ${SLIDE_MS}ms linear`; bar.style.width = paused ? "0" : "100%"; });
    preload(i + 1);
    clearTimeout(timer);
    if (!paused) timer = setTimeout(() => show(i + 1), SLIDE_MS);
  };
  const close = () => {
    clearTimeout(timer);
    document.removeEventListener("keydown", onKey, true);
    if (document.fullscreenElement) document.exitFullscreen?.().catch(() => {});
    el.remove();
    document.body.style.overflow = "";
  };
  const toggle = () => {
    paused = !paused;
    $("[data-a=pause]", el).innerHTML = paused ? I.play : I.pause;
    show(i);
  };
  const onKey = (e) => {
    const k = { Escape: close, ArrowRight: () => show(i + 1), ArrowLeft: () => show(i - 1), " ": toggle }[e.key];
    if (k) { e.preventDefault(); e.stopPropagation(); k(); }
  };
  document.addEventListener("keydown", onKey, true);
  el.onclick = (e) => {
    const a = e.target.closest("[data-a]")?.dataset.a;
    if (a === "close") close();
    if (a === "next") show(i + 1);
    if (a === "prev") show(i - 1);
    if (a === "pause") toggle();
  };
  let idle;
  el.onmousemove = () => { el.classList.add("show-ui"); clearTimeout(idle); idle = setTimeout(() => el.classList.remove("show-ui"), 2200); };
  setTimeout(() => $(".ss-title", el).classList.add("gone"), 2600);
  show(0);
}

// ------------------------------------------------------------------ status polling
let pollTimer = null;
async function refreshStatus() {
  clearTimeout(pollTimer);
  const prev = state.status;
  try {
    state.status = await api("/api/status");
  } catch {
    pollTimer = setTimeout(refreshStatus, 5000);
    return prev;
  }
  const s = state.status;
  const pill = $("#index-pill");
  const busy = ["scanning", "indexing", "enriching", "paused"].includes(s.state) || s.pending > 0 || s.enrich_pending > 0;
  pill.hidden = !busy;
  if (busy) {
    const enriching = !s.pending && s.enrich_pending > 0;
    const pct = enriching ? 1 - s.enrich_pending / Math.max(s.total, 1) : s.total ? s.ready / s.total : 0;
    pill.classList.toggle("spinning", s.state === "scanning" || (s.model_state !== "ready" && s.state === "indexing"));
    $(".ring-fill", pill).style.strokeDashoffset = 94.25 * (1 - pct);
    $(".index-pill-text", pill).textContent = s.state === "scanning" ? `Scanning… ${fmt(s.scanned_files)}`
      : s.state === "paused" ? `Paused · ${fmt(s.ready)} / ${fmt(s.total)}`
      : enriching ? `Faces & places · ${fmt(s.enrich_pending)} left`
      : `${fmt(s.ready)} / ${fmt(s.total)}${s.eta_seconds ? " · " + fmtDuration(s.eta_seconds) : ""}`;
  }
  drawIndexPanel();
  // First photos became searchable, or a big batch finished: refresh home if the user isn't typing.
  if (prev && state.view === "home" && !document.activeElement?.matches("input") &&
      ((prev.ready === 0 && s.ready > 0) || (prev.pending > 0 && s.pending === 0) || prev.folders !== s.folders ||
       (prev.enrich_pending > 0 && s.enrich_pending === 0))) route();
  pollTimer = setTimeout(refreshStatus, busy ? 2000 : 10000);
  return s;
}

// ------------------------------------------------------------------ theme & global keys
function applyTheme(t) {
  if (t) document.documentElement.dataset.theme = t; else delete document.documentElement.dataset.theme;
  const dark = t ? t === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  $("#theme-toggle").innerHTML = dark ? I.sun : I.moon;
}
$("#theme-toggle").onclick = () => {
  const dark = document.documentElement.dataset.theme ? document.documentElement.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  const t = dark ? "light" : "dark";
  store.set("theme", t);
  applyTheme(t);
};

$("#top-search").onsubmit = (e) => {
  e.preventDefault();
  const q = $("#top-search-input").value.trim();
  if (q) go(`#/search?q=${encodeURIComponent(q)}`);
};
$("#tray-clear").onclick = () => { clearSelection(); };

document.addEventListener("keydown", (e) => {
  const typing = e.target.matches("input, textarea, select");
  if (!$("#lightbox").hidden) {
    if (e.key === "Escape") closeLightbox();
    if (e.key === "ArrowLeft") step(-1);
    if (e.key === "ArrowRight") step(1);
    return;
  }
  if (e.key === "/" && !typing) {
    e.preventDefault();
    (state.view === "home" ? $(".hero-search input") : $("#top-search-input"))?.focus();
  }
  if (e.key === "Escape" && !typing && state.selection.size) clearSelection();
});

window.addEventListener("hashchange", route);
applyTheme(store.get("theme", null));
refreshStatus().then(route);
