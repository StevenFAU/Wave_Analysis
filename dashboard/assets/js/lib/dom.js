// Small DOM helpers. Text always goes in through textContent/createTextNode:
// catalog and status strings come from upstream data and are never parsed as HTML.

const SVG_NS = "http://www.w3.org/2000/svg";

function applyAttrs(el, attrs, isSvg) {
  if (!attrs) return;
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") el.setAttribute("class", Array.isArray(v) ? v.filter(Boolean).join(" ") : v);
    else if (k === "text") el.textContent = v;
    else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
    else if (k === "dataset") Object.assign(el.dataset, v);
    else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
    else if (!isSvg && k in el && typeof v !== "string" && k !== "list") el[k] = v;
    else el.setAttribute(k, v === true ? "" : String(v));
  }
}

function append(el, children) {
  for (const c of children.flat(Infinity)) {
    if (c == null || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
}

/** h("a", {href, class}, "text", child, [more]) */
export function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  applyAttrs(el, attrs, false);
  append(el, children);
  return el;
}

export function s(tag, attrs, ...children) {
  const el = document.createElementNS(SVG_NS, tag);
  applyAttrs(el, attrs, true);
  append(el, children);
  return el;
}

export function clear(el) {
  while (el.firstChild) el.removeChild(el.firstChild);
  return el;
}

/** External link that opens in a new tab without leaking the opener. */
export function ext(href, text, attrs = {}) {
  return h("a", { href, target: "_blank", rel: "noopener noreferrer", ...attrs }, text ?? href);
}

export function cssVar(name, el = document.documentElement) {
  return getComputedStyle(el).getPropertyValue(name).trim();
}

export function card(title, ...children) {
  return h("section", { class: "card" }, title ? h("h2", null, title) : null, ...children);
}

export function tile(label, value, sub, { hero = false } = {}) {
  return h(
    "div",
    { class: "tile" },
    h("div", { class: "tile-label" }, label),
    h("div", { class: ["tile-value", hero && "hero"] }, value),
    sub ? h("div", { class: "tile-sub" }, sub) : null,
  );
}

export function statusLabel(level, text) {
  return h("span", { class: ["status", level] }, text);
}

/** Legend: items = [{label, color, kind: 'rect'|'line'|'dot'|'ring'|'square'}] */
export function legend(items) {
  return h(
    "div",
    { class: "legend" },
    items.map((it) =>
      h(
        "span",
        { class: "legend-item" },
        h("span", {
          class: ["swatch", it.kind && it.kind !== "rect" ? it.kind : null],
          style: { background: it.color, borderColor: it.color },
        }),
        it.label,
      ),
    ),
  );
}

/**
 * Sortable table. columns = [{key, label, num, render(row) -> node|string,
 * sort(row) -> comparable}], rows = objects. onRow(row) makes rows clickable.
 */
export function dataTable(columns, rows, { onRow, initialSort, caption, rowHref } = {}) {
  let sortKey = initialSort?.key ?? null;
  let dir = initialSort?.dir ?? 1;
  const tbody = h("tbody");
  const ths = columns.map((c) => {
    const th = h("th", { class: c.num ? "num" : null, scope: "col" });
    if (c.sortable === false) th.textContent = c.label;
    else
      th.append(
        h("button", {
          class: "sort",
          type: "button",
          text: c.label,
          onclick: () => {
            if (sortKey === c.key) dir = -dir;
            else {
              sortKey = c.key;
              dir = c.num ? -1 : 1;
            }
            draw();
          },
        }),
      );
    return th;
  });
  const valueOf = (c, r) => (c.sort ? c.sort(r) : r[c.key]);
  function draw() {
    ths.forEach((th, i) => {
      if (columns[i].key === sortKey) th.setAttribute("aria-sort", dir > 0 ? "ascending" : "descending");
      else th.removeAttribute("aria-sort");
    });
    const col = columns.find((c) => c.key === sortKey);
    const sorted = col
      ? [...rows].sort((a, b) => {
          const va = valueOf(col, a);
          const vb = valueOf(col, b);
          if (va == null && vb == null) return 0;
          if (va == null) return 1;
          if (vb == null) return -1;
          return (va < vb ? -1 : va > vb ? 1 : 0) * dir;
        })
      : rows;
    clear(tbody);
    for (const r of sorted) {
      const tr = h(
        "tr",
        onRow ? { class: "clickable", tabindex: 0 } : null,
        columns.map((c) => h("td", { class: c.num ? "num" : null }, c.render ? c.render(r) : r[c.key] ?? "")),
      );
      if (onRow) {
        tr.addEventListener("click", (e) => {
          if (e.target.closest("a, button")) return;
          onRow(r);
        });
        tr.addEventListener("keydown", (e) => {
          if (e.key === "Enter") onRow(r);
        });
        if (rowHref) tr.title = "Open";
      }
      tbody.append(tr);
    }
  }
  draw();
  return h(
    "div",
    { class: "table-wrap" },
    h("table", null, caption ? h("caption", { class: "sr-only" }, caption) : null, h("thead", null, h("tr", null, ths)), tbody),
  );
}

/** The accessible table twin of a chart, collapsed by default. */
export function tableView(columns, rows, summary = "Table view") {
  return h(
    "details",
    { class: "table-view" },
    h("summary", null, summary),
    dataTable(columns, rows),
  );
}

export function copyButton(text, label = "Copy") {
  const btn = h("button", { class: "icon-btn", type: "button" }, label);
  btn.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(text);
      btn.textContent = "Copied";
    } catch {
      btn.textContent = "Copy failed";
    }
    setTimeout(() => {
      btn.textContent = label;
    }, 1500);
  });
  return btn;
}
