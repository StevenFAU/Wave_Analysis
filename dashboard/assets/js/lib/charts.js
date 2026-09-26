// Minimal SVG/canvas charts: line/dot time series with a snapping crosshair,
// stacked columns with per-column hover, a canvas heatmap, and small helpers.
// Marks follow the dashboard's chart spec: 2px lines, <=24px columns with a
// 2px surface gap and 4px rounded data ends, hairline solid grid.

import { cssVar, h, s } from "./dom.js";
import { fmtDateUtc, fmtHourUtc, fmtNum, fmtUtcShort } from "./format.js";
import { hideTooltip, showTooltip } from "./tooltip.js";

const HOUR = 3600e3;
const DAY = 24 * HOUR;
const TIME_STEPS = [HOUR, 3 * HOUR, 6 * HOUR, 12 * HOUR, DAY, 2 * DAY, 7 * DAY, 14 * DAY, 30 * DAY];

/** Time-axis ticks at least `minPx` apart; midnight ticks are labelled with the date. */
export function timeTicks(t0, t1, width, minPx = 64, minStep = 0) {
  const span = t1 - t0;
  if (!(span > 0) || width <= 0) return [];
  if (span > 2 * 365 * DAY) {
    // Calendar-year ticks for long spans (e.g. publication years).
    const y0 = new Date(t0).getUTCFullYear();
    const y1 = new Date(t1).getUTCFullYear();
    const k = [1, 2, 5, 10, 20, 50, 100].find((n) => ((n * 365.25 * DAY) / span) * width >= minPx) ?? 100;
    const ticks = [];
    for (let y = Math.ceil(y0 / k) * k; y <= y1; y += k) {
      const t = Date.UTC(y, 0, 1);
      if (t >= t0 && t <= t1) ticks.push({ t, label: String(y), major: false });
    }
    return ticks;
  }
  const step =
    TIME_STEPS.find((st) => st >= minStep && (st / span) * width >= minPx) ?? TIME_STEPS[TIME_STEPS.length - 1];
  const first = Math.ceil(t0 / step) * step;
  const ticks = [];
  for (let t = first; t <= t1; t += step) {
    const midnight = t % DAY === 0;
    ticks.push({ t, label: midnight || step >= DAY ? fmtDateUtc(t) : fmtHourUtc(t), major: midnight });
  }
  return ticks;
}

/** "Nice" linear ticks covering [min, max]. */
export function niceTicks(min, max, count = 4) {
  if (min === max) {
    max = min + 1;
  }
  const span = max - min;
  const raw = span / Math.max(count, 1);
  const mag = 10 ** Math.floor(Math.log10(raw));
  const norm = raw / mag;
  const step = (norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1) * mag;
  const lo = Math.floor(min / step) * step;
  const hi = Math.ceil(max / step) * step;
  const out = [];
  for (let v = lo; v <= hi + step / 2; v += step) out.push(Number(v.toFixed(10)));
  return out;
}

/** Re-run draw(width) whenever the container's width changes. Returns disconnect(). */
export function responsive(container, draw) {
  let last = -1;
  const run = () => {
    const w = Math.floor(container.clientWidth);
    if (w > 0 && w !== last) {
      last = w;
      draw(w);
    }
  };
  const ro = new ResizeObserver(run);
  ro.observe(container);
  run();
  return () => ro.disconnect();
}

/** Crosshair sync between small multiples. */
export function createSync() {
  const subs = new Set();
  return {
    subscribe(fn) {
      subs.add(fn);
      return () => subs.delete(fn);
    },
    publish(t, source) {
      for (const fn of subs) fn(t, source);
    },
  };
}

function nearestIndex(sorted, t) {
  let lo = 0;
  let hi = sorted.length - 1;
  if (hi < 0) return -1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (sorted[mid] < t) lo = mid + 1;
    else hi = mid;
  }
  if (lo > 0 && Math.abs(sorted[lo - 1] - t) <= Math.abs(sorted[lo] - t)) return lo - 1;
  return lo;
}

/**
 * Time-series chart.
 * opts: {series:[{key,label,color,kind:'line'|'dots',points:[{t,v}]}], xMin, xMax,
 *        height, yMin, yMax, yTicks, yFormat, yTitle, maxGapMs, cursor, onPick, sync, ariaLabel}
 */
export function lineChart(container, opts) {
  const {
    series,
    xMin,
    xMax,
    height = 150,
    yFormat = (v) => fmtNum(v, 1),
    maxGapMs = 3 * HOUR,
    onPick,
    sync,
    ariaLabel,
  } = opts;
  let cursor = opts.cursor ?? null;
  const wrap = h("div", { class: "chart" });
  container.append(wrap);
  const values = series.flatMap((sr) => sr.points.map((p) => p.v)).filter((v) => v != null);
  if (!values.length) {
    wrap.append(h("div", { class: "empty" }, "No observations in this period."));
    return { setCursor() {}, destroy() {} };
  }
  const yMin = opts.yMin ?? Math.min(0, ...values);
  const yMax = opts.yMax ?? Math.max(...values) * 1.08;
  const yTicks = opts.yTicks ?? niceTicks(yMin, yMax, 4);
  const y0 = Math.min(yMin, yTicks[0]);
  const y1 = Math.max(yMax, yTicks[yTicks.length - 1]);
  const allTimes = [...new Set(series.flatMap((sr) => sr.points.filter((p) => p.v != null).map((p) => p.t)))].sort(
    (a, b) => a - b,
  );
  const m = { top: 10, right: 12, bottom: 22, left: 44 };
  let xs = null;
  let hover = null;
  let cursorLine = null;
  let hoverT = null;

  function draw(width) {
    wrap.querySelector("svg")?.remove();
    const iw = Math.max(40, width - m.left - m.right);
    const ih = height - m.top - m.bottom;
    const x = (t) => m.left + ((t - xMin) / (xMax - xMin)) * iw;
    const y = (v) => m.top + ih - ((v - y0) / (y1 - y0)) * ih;
    xs = { x, iw, ih };
    const svg = s("svg", {
      viewBox: `0 0 ${width} ${height}`,
      height,
      role: "img",
      "aria-label": ariaLabel || "Time series",
      tabindex: 0,
    });
    for (const v of yTicks) {
      svg.append(s("line", { class: "grid-line", x1: m.left, x2: m.left + iw, y1: y(v), y2: y(v) }));
      svg.append(s("text", { class: "tick-label", x: m.left - 6, y: y(v) + 4, "text-anchor": "end" }, yFormat(v)));
    }
    svg.append(s("line", { class: "axis-line", x1: m.left, x2: m.left + iw, y1: m.top + ih, y2: m.top + ih }));
    for (const tk of timeTicks(xMin, xMax, iw)) {
      svg.append(
        s("text", { class: "tick-label", x: x(tk.t), y: height - 6, "text-anchor": "middle" }, tk.label),
      );
      if (tk.major) svg.append(s("line", { class: "grid-line", x1: x(tk.t), x2: x(tk.t), y1: m.top, y2: m.top + ih }));
    }
    for (const sr of series) {
      const pts = sr.points.filter((p) => p.v != null && p.t >= xMin && p.t <= xMax);
      if (sr.kind === "dots") {
        const g = s("g", { style: `fill:${sr.color}` });
        for (const p of pts) g.append(s("circle", { cx: x(p.t), cy: y(p.v), r: 3 }));
        svg.append(g);
      } else {
        let d = "";
        let prev = null;
        for (const p of pts) {
          d += `${prev == null || p.t - prev > maxGapMs ? "M" : "L"}${x(p.t).toFixed(1)},${y(p.v).toFixed(1)}`;
          prev = p.t;
        }
        svg.append(s("path", { class: "series-line", d, style: `stroke:${sr.color}` }));
        // Isolated points (between gaps) would be invisible as a path.
        const lone = pts.filter(
          (p, i) => (i === 0 || p.t - pts[i - 1].t > maxGapMs) && (i === pts.length - 1 || pts[i + 1].t - p.t > maxGapMs),
        );
        for (const p of lone) svg.append(s("circle", { cx: x(p.t), cy: y(p.v), r: 2.5, style: `fill:${sr.color}` }));
      }
    }
    cursorLine = s("line", { class: "cursor-line", y1: m.top, y2: m.top + ih, visibility: "hidden" });
    hover = s("line", { class: "crosshair", y1: m.top, y2: m.top + ih, visibility: "hidden" });
    svg.append(cursorLine, hover);
    const hit = s("rect", { class: "hit", x: m.left, y: m.top, width: iw, height: ih });
    svg.append(hit);
    hit.addEventListener("pointermove", (e) => {
      const rect = svg.getBoundingClientRect();
      const px = ((e.clientX - rect.left) / rect.width) * width;
      const t = xMin + ((px - m.left) / iw) * (xMax - xMin);
      const i = nearestIndex(allTimes, t);
      if (i < 0) return;
      setHover(allTimes[i], true, e.clientX, e.clientY);
      sync?.publish(allTimes[i], api);
    });
    hit.addEventListener("pointerleave", () => {
      setHover(null);
      sync?.publish(null, api);
    });
    if (onPick) {
      hit.style.cursor = "pointer";
      hit.addEventListener("click", () => hoverT != null && onPick(hoverT));
    }
    svg.addEventListener("keydown", (e) => {
      if (!allTimes.length) return;
      let i = hoverT == null ? allTimes.length - 1 : nearestIndex(allTimes, hoverT);
      if (e.key === "ArrowLeft") i = Math.max(0, i - 1);
      else if (e.key === "ArrowRight") i = Math.min(allTimes.length - 1, i + 1);
      else if (e.key === "Enter" && onPick && hoverT != null) {
        onPick(hoverT);
        return;
      } else return;
      e.preventDefault();
      const r = svg.getBoundingClientRect();
      setHover(allTimes[i], true, r.left + (x(allTimes[i]) / width) * r.width, r.top + 20);
      sync?.publish(allTimes[i], api);
    });
    svg.addEventListener("blur", () => setHover(null));
    wrap.append(svg);
    setCursor(cursor);
  }

  function setHover(t, withTip = false, cx = 0, cy = 0) {
    hoverT = t;
    if (!hover || !xs) return;
    if (t == null || t < xMin || t > xMax) {
      hover.setAttribute("visibility", "hidden");
      hideTooltip();
      return;
    }
    const px = xs.x(t);
    hover.setAttribute("x1", px);
    hover.setAttribute("x2", px);
    hover.setAttribute("visibility", "visible");
    if (!withTip) return;
    const rows = series.map((sr) => {
      const pts = sr.points.filter((p) => p.v != null);
      const i = nearestIndex(
        pts.map((p) => p.t),
        t,
      );
      const p = i >= 0 && Math.abs(pts[i].t - t) <= maxGapMs / 2 ? pts[i] : null;
      return { label: sr.label, value: p ? yFormat(p.v) : "–", color: sr.color };
    });
    showTooltip(cx, cy, fmtUtcShort(t), rows, onPick ? "Click to show the image nearest this time" : null);
  }

  function setCursor(t) {
    cursor = t;
    if (!cursorLine || !xs) return;
    if (t == null || t < xMin || t > xMax) {
      cursorLine.setAttribute("visibility", "hidden");
      return;
    }
    const px = xs.x(t);
    cursorLine.setAttribute("x1", px);
    cursorLine.setAttribute("x2", px);
    cursorLine.setAttribute("visibility", "visible");
  }

  const api = { setCursor };
  const unsync = sync?.subscribe((t, source) => {
    if (source !== api) setHover(t, false);
  });
  const stop = responsive(wrap, draw);
  api.destroy = () => {
    stop();
    unsync?.();
  };
  return api;
}

function roundedTop(x, y, w, hgt, r) {
  const rr = Math.max(0, Math.min(r, w / 2, hgt));
  return `M${x},${y + hgt}V${y + rr}Q${x},${y} ${x + rr},${y}H${x + w - rr}Q${x + w},${y} ${x + w},${y + rr}V${y + hgt}Z`;
}

/**
 * Stacked columns on a time axis.
 * opts: {rows:[{t, values:{key:n}}], keys:[{key,label,color}], xMin, xMax, bandMs,
 *        height, yFormat, title(row), note(row), onPick(row), ariaLabel}
 */
export function columnChart(container, opts) {
  const { rows, keys, xMin, xMax, bandMs = HOUR, height = 180, yFormat = (v) => fmtNum(v, 0), onPick, ariaLabel } = opts;
  // Tick no finer than one band: daily columns get date ticks, not 12:00.
  const minTick = opts.minTickMs ?? bandMs;
  const wrap = h("div", { class: "chart" });
  container.append(wrap);
  const totals = rows.map((r) => keys.reduce((a, k) => a + (r.values[k.key] || 0), 0));
  const maxTotal = Math.max(1, ...totals);
  if (!rows.length) {
    wrap.append(h("div", { class: "empty" }, "No data yet."));
    return { destroy() {} };
  }
  const ticks = niceTicks(0, maxTotal, 4);
  const yTop = ticks[ticks.length - 1];
  const m = { top: 10, right: 12, bottom: 22, left: 44 };

  function draw(width) {
    wrap.querySelector("svg")?.remove();
    const iw = Math.max(40, width - m.left - m.right);
    const ih = height - m.top - m.bottom;
    const x = (t) => m.left + ((t - xMin) / (xMax - xMin)) * iw;
    const y = (v) => m.top + ih - (v / yTop) * ih;
    const band = (bandMs / (xMax - xMin)) * iw;
    const colW = Math.max(1, Math.min(24, band - 2));
    const svg = s("svg", { viewBox: `0 0 ${width} ${height}`, height, role: "img", "aria-label": ariaLabel || "" });
    for (const v of ticks) {
      svg.append(s("line", { class: "grid-line", x1: m.left, x2: m.left + iw, y1: y(v), y2: y(v) }));
      svg.append(s("text", { class: "tick-label", x: m.left - 6, y: y(v) + 4, "text-anchor": "end" }, yFormat(v)));
    }
    for (const tk of timeTicks(xMin, xMax, iw, 64, minTick)) {
      svg.append(s("text", { class: "tick-label", x: x(tk.t) + (minTick >= DAY ? band / 2 : 0), y: height - 6, "text-anchor": "middle" }, tk.label));
    }
    const gap = colW >= 4 ? 2 : 0;
    rows.forEach((r, idx) => {
      const cx = x(r.t) + (band - colW) / 2;
      let base = 0;
      const g = s("g", { class: "bar" });
      const present = keys.filter((k) => (r.values[k.key] || 0) > 0);
      present.forEach((k, j) => {
        const v = r.values[k.key];
        const top = y(base + v);
        const bottom = y(base);
        const hgt = Math.max(0, bottom - top - (j > 0 ? gap : 0));
        const yy = top;
        if (j === present.length - 1) g.append(s("path", { d: roundedTop(cx, yy, colW, hgt, 4), style: `fill:${k.color}` }));
        else g.append(s("rect", { x: cx, y: yy, width: colW, height: hgt, style: `fill:${k.color}` }));
        base += v;
      });
      const hit = s("rect", { class: "hit", x: x(r.t), y: m.top, width: Math.max(band, 6), height: ih });
      hit.addEventListener("pointermove", (e) => {
        g.classList.add("active");
        showTooltip(
          e.clientX,
          e.clientY,
          opts.title ? opts.title(r) : fmtUtcShort(r.t),
          [
            ...keys.map((k) => ({ label: k.label, value: yFormat(r.values[k.key] || 0), color: k.color, kind: "cell" })),
            ...(keys.length > 1 ? [{ label: "Total", value: yFormat(totals[idx]) }] : []),
          ],
          opts.note ? opts.note(r) : null,
        );
      });
      hit.addEventListener("pointerleave", () => {
        g.classList.remove("active");
        hideTooltip();
      });
      if (onPick) {
        hit.style.cursor = "pointer";
        hit.addEventListener("click", () => onPick(r));
      }
      svg.append(g, hit);
    });
    svg.append(s("line", { class: "axis-line", x1: m.left, x2: m.left + iw, y1: m.top + ih, y2: m.top + ih }));
    wrap.append(svg);
  }
  const stop = responsive(wrap, draw);
  return { destroy: stop };
}

/**
 * Canvas heatmap (rows x columns).
 * opts: {rows:[{id,label}], nCols, color(r,c)->css, info(r,c)->{title, rows, note}, onPick(r,c),
 *        colTicks:[{i,label,major}], cellH, labelW}
 */
export function heatmap(container, opts) {
  const { rows, nCols, color, info, onPick, colTicks = [], cellH = 12, labelW = 64 } = opts;
  const wrap = h("div", { class: "chart" });
  container.append(wrap);
  if (!rows.length) {
    wrap.append(h("div", { class: "empty" }, "No stations match the filters."));
    return { destroy() {} };
  }
  const top = 18;
  const canvas = h("canvas", { role: "img", "aria-label": opts.ariaLabel || "Coverage heatmap", tabindex: 0 });
  wrap.append(canvas);
  let geom = null;
  function draw(width) {
    const dpr = window.devicePixelRatio || 1;
    const cw = Math.max(1, (width - labelW - 4) / nCols);
    const heightPx = top + rows.length * cellH + 4;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(heightPx * dpr);
    canvas.style.height = `${heightPx}px`;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, heightPx);
    ctx.font = `11px ${cssVar("--font") || "system-ui"}`;
    ctx.fillStyle = cssVar("--muted");
    ctx.textBaseline = "middle";
    const gridColor = cssVar("--grid");
    for (const tk of colTicks) {
      const px = labelW + tk.i * cw;
      if (tk.label) {
        ctx.textAlign = "left";
        ctx.fillText(tk.label, px + 2, 8);
      }
      if (tk.major) {
        ctx.fillStyle = gridColor;
        ctx.fillRect(Math.round(px) - 1, top - 4, 1, rows.length * cellH + 6);
        ctx.fillStyle = cssVar("--muted");
      }
    }
    const gapX = cw >= 4 ? 1 : 0;
    const colorCache = new Map();
    const resolve = (c) => {
      if (!c.startsWith("var(")) return c;
      if (!colorCache.has(c)) colorCache.set(c, cssVar(c.slice(4, -1)));
      return colorCache.get(c);
    };
    rows.forEach((r, ri) => {
      const yy = top + ri * cellH;
      ctx.fillStyle = cssVar("--ink-2");
      ctx.textAlign = "right";
      ctx.fillText(r.label, labelW - 6, yy + cellH / 2);
      for (let c = 0; c < nCols; c += 1) {
        const col = color(ri, c);
        if (!col) continue;
        ctx.fillStyle = resolve(col);
        ctx.fillRect(labelW + c * cw, yy + 1, Math.max(1, cw - gapX), cellH - 2);
      }
    });
    geom = { cw, width, heightPx };
  }
  function cellAt(e) {
    if (!geom) return null;
    const rect = canvas.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * geom.width;
    const py = ((e.clientY - rect.top) / rect.height) * geom.heightPx;
    const c = Math.floor((px - labelW) / geom.cw);
    const r = Math.floor((py - top) / cellH);
    if (c < 0 || c >= nCols || r < 0 || r >= rows.length) return null;
    return { r, c };
  }
  canvas.addEventListener("pointermove", (e) => {
    const cell = cellAt(e);
    if (!cell) {
      hideTooltip();
      canvas.style.cursor = "default";
      return;
    }
    const inf = info(cell.r, cell.c);
    canvas.style.cursor = onPick && inf.pickable ? "pointer" : "default";
    showTooltip(e.clientX, e.clientY, inf.title, inf.rows || [], inf.note);
  });
  canvas.addEventListener("pointerleave", hideTooltip);
  if (onPick)
    canvas.addEventListener("click", (e) => {
      const cell = cellAt(e);
      if (cell) onPick(cell.r, cell.c);
    });
  const stop = responsive(wrap, draw);
  return { destroy: stop, redraw: () => geom && draw(geom.width) };
}

/** A 1px-per-hour coverage strip for tables (canvas). colorOf(i) -> css var or color. */
export function stripCanvas(n, colorOf, { width = 168, height = 12 } = {}) {
  const canvas = h("canvas", { width: n, height: 1, style: { width: `${width}px`, height: `${height}px` } });
  canvas.className = "strip-canvas";
  const ctx = canvas.getContext("2d");
  const cache = new Map();
  for (let i = 0; i < n; i += 1) {
    let c = colorOf(i);
    if (!c) continue;
    if (c.startsWith("var(")) {
      if (!cache.has(c)) cache.set(c, cssVar(c.slice(4, -1)));
      c = cache.get(c);
    }
    ctx.fillStyle = c;
    ctx.fillRect(i, 0, 1, 1);
  }
  canvas.style.imageRendering = "pixelated";
  return canvas;
}

/**
 * Horizontal year-range bars (e.g. archive history per product).
 * items: [{label, first, last, years, color}]
 */
export function yearRanges(container, items, { minYear, maxYear, height = null } = {}) {
  const wrap = h("div", { class: "chart" });
  container.append(wrap);
  if (!items.length) {
    wrap.append(h("div", { class: "empty" }, "No archive history recorded."));
    return { destroy() {} };
  }
  const lo = minYear ?? Math.min(...items.map((i) => i.first));
  const hi = (maxYear ?? Math.max(...items.map((i) => i.last))) + 1;
  const rowH = 22;
  const m = { top: 6, right: 12, bottom: 22, left: 70 };
  const hgt = height ?? m.top + items.length * rowH + m.bottom;
  function draw(width) {
    wrap.querySelector("svg")?.remove();
    const iw = Math.max(40, width - m.left - m.right);
    const x = (yr) => m.left + ((yr - lo) / (hi - lo)) * iw;
    const svg = s("svg", { viewBox: `0 0 ${width} ${hgt}`, height: hgt, role: "img", "aria-label": "Archive years by product" });
    const ticks = niceTicks(lo, hi, Math.max(2, Math.floor(iw / 70))).filter((v) => v >= lo && v <= hi);
    for (const v of ticks) {
      svg.append(s("line", { class: "grid-line", x1: x(v), x2: x(v), y1: m.top, y2: hgt - m.bottom }));
      svg.append(s("text", { class: "tick-label", x: x(v), y: hgt - 6, "text-anchor": "middle" }, String(v)));
    }
    items.forEach((it, i) => {
      const yy = m.top + i * rowH + 5;
      svg.append(s("text", { class: "tick-label", x: m.left - 8, y: yy + 8, "text-anchor": "end" }, it.label));
      const x0 = x(it.first);
      const w = Math.max(3, x(it.last + 1) - x0 - 1);
      const bar = s("rect", { x: x0, y: yy, width: w, height: 12, rx: 3, style: `fill:${it.color || "var(--s1)"}` });
      bar.addEventListener("pointermove", (e) =>
        showTooltip(e.clientX, e.clientY, it.label, [
          { label: "first year", value: String(it.first) },
          { label: "last year", value: String(it.last) },
          { label: "years with files", value: String(it.years) },
        ]),
      );
      bar.addEventListener("pointerleave", hideTooltip);
      svg.append(bar);
    });
    wrap.append(svg);
  }
  const stop = responsive(wrap, draw);
  return { destroy: stop };
}
