// One shared tooltip. Content is built from DOM nodes (never HTML strings).
import { clear, h } from "./dom.js";

const el = () => document.getElementById("tooltip");

/**
 * rows = [{label, value, color, kind}] ; title = string
 * Values lead (strong), labels follow.
 */
export function showTooltip(x, y, title, rows = [], note = null) {
  const tip = el();
  if (!tip) return;
  clear(tip);
  if (title) tip.append(h("div", { class: "tt-title" }, title));
  for (const r of rows) {
    tip.append(
      h(
        "div",
        { class: "tt-row" },
        r.color ? h("span", { class: ["tt-key", r.kind === "cell" && "cell"], style: { background: r.color } }) : null,
        h("strong", null, r.value),
        h("span", { class: "muted" }, r.label),
      ),
    );
  }
  if (note) tip.append(h("div", { class: "tt-title", style: { marginTop: "4px" } }, note));
  tip.hidden = false;
  const pad = 14;
  const { innerWidth: w, innerHeight: hgt } = window;
  const rect = tip.getBoundingClientRect();
  let left = x + pad;
  let top = y + pad;
  if (left + rect.width > w - 8) left = x - rect.width - pad;
  if (top + rect.height > hgt - 8) top = y - rect.height - pad;
  tip.style.left = `${Math.max(8, left)}px`;
  tip.style.top = `${Math.max(8, top)}px`;
}

export function hideTooltip() {
  const tip = el();
  if (tip) tip.hidden = true;
}
