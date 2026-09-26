// Decoding of the status.json coverage encoding (see
// src/wave_analysis/dashboard/status.py). Pure; unit-tested in
// tests/js/coverage.test.mjs.

export const HOUR_MS = 3600 * 1000;
export const IMAGE_BASE = "https://www.ndbc.noaa.gov/images/buoycam";
// NDBC keeps superseded images about 72 h; stay inside that with a margin.
export const UPSTREAM_RETENTION_H = 71;

export const ILLUMINATION = {
  d: { label: "Day", token: "--il-day", order: 0 },
  l: { label: "Low sun", token: "--il-low", order: 1 },
  t: { label: "Twilight", token: "--il-twi", order: 2 },
  n: { label: "Night", token: "--il-night", order: 3 },
  "-": { label: "Unknown", token: "--il-twi", order: 4 },
};

export const CELL_STATES = {
  archived: { label: "Archived" },
  not_published: { label: "Not published by NDBC", token: "--cell-notpub" },
  gap: { label: "Missed by the collector", token: "--cell-gap" },
  none: { label: "Not expected / pending", token: "--cell-empty" },
};

/** One coverage character -> {state, minute}. */
export function decodeCell(ch) {
  if (ch >= "0" && ch <= "5") return { state: "archived", minute: Number(ch) * 10 };
  if (ch === "x") return { state: "not_published", minute: null };
  if (ch === "?") return { state: "gap", minute: null };
  return { state: "none", minute: null };
}

/** Start times (ms) of the status window's hours. */
export function windowHours(window) {
  const start = Date.parse(window.start);
  return Array.from({ length: window.hours }, (_, i) => start + i * HOUR_MS);
}

const pad = (n) => String(n).padStart(2, "0");

/** NDBC file name, e.g. W04A_2026_09_26_1310.jpg */
export function imageFileName(camera, ms) {
  const d = new Date(ms);
  return `${camera}_${d.getUTCFullYear()}_${pad(d.getUTCMonth() + 1)}_${pad(d.getUTCDate())}_${pad(d.getUTCHours())}${pad(d.getUTCMinutes())}.jpg`;
}

export function imageUrl(fileName) {
  return `${IMAGE_BASE}/${fileName}`;
}

/** Counts of each state in a coverage string. */
export function summarize(coverage) {
  const out = { archived: 0, not_published: 0, gap: 0, none: 0 };
  for (const ch of coverage) out[decodeCell(ch).state] += 1;
  return out;
}

/**
 * Archived images of one station within the status window, oldest first:
 * [{t, file, url, illum, upstream}] where `upstream` says whether NDBC still
 * serves the file (only those can be shown on a public page).
 */
export function stationImages(station, window, now = Date.now()) {
  if (!station.camera) return [];
  const hours = windowHours(window);
  const out = [];
  for (let i = 0; i < hours.length; i += 1) {
    const { state, minute } = decodeCell(station.coverage[i]);
    if (state !== "archived") continue;
    const t = hours[i] + minute * 60 * 1000;
    const file = imageFileName(station.camera, t);
    out.push({
      t,
      file,
      url: imageUrl(file),
      illum: station.illumination ? station.illumination[i] : "-",
      upstream: now - t < UPSTREAM_RETENTION_H * HOUR_MS,
    });
  }
  return out;
}

/**
 * Per-hour totals across stations: archived images by illumination code,
 * plus not-published and gap counts. `filter(station)` selects stations.
 */
export function hourlyTotals(stations, window, filter = () => true) {
  const n = window.hours;
  const rows = Array.from({ length: n }, () => ({ d: 0, l: 0, t: 0, n: 0, "-": 0, not_published: 0, gap: 0 }));
  for (const s of stations) {
    if (!filter(s)) continue;
    for (let i = 0; i < n; i += 1) {
      const { state } = decodeCell(s.coverage[i]);
      if (state === "archived") rows[i][(s.illumination || "")[i] || "-"] += 1;
      else if (state === "not_published") rows[i].not_published += 1;
      else if (state === "gap") rows[i].gap += 1;
    }
  }
  return rows;
}

/** Region label from the NDBC station-id prefix (WMO area numbering). */
export function region(stationId) {
  const p = String(stationId).slice(0, 2);
  return REGIONS[p] ? p : "other";
}

export const REGIONS = {
  41: "41xxx: SE Atlantic & Caribbean",
  42: "42xxx: Gulf of Mexico & Caribbean",
  44: "44xxx: NE Atlantic coast",
  45: "45xxx: Great Lakes",
  46: "46xxx: NE Pacific & Alaska",
  51: "51xxx: Hawaii & central Pacific",
  52: "52xxx: Western Pacific",
  other: "Other",
};

/** Significant-wave-height bin (0-4) for the ordinal map scale. */
export const HS_BINS = [1, 2, 3, 4];
export function hsBin(hs) {
  if (hs == null || Number.isNaN(hs)) return null;
  let i = 0;
  while (i < HS_BINS.length && hs >= HS_BINS[i]) i += 1;
  return i;
}

export function hsBinLabel(i) {
  if (i === 0) return `< ${HS_BINS[0]} m`;
  if (i === HS_BINS.length) return `≥ ${HS_BINS[HS_BINS.length - 1]} m`;
  return `${HS_BINS[i - 1]}–${HS_BINS[i]} m`;
}

/** Collector health from the age of the last run (minutes). */
export function collectorHealth(lastRunMs, now = Date.now()) {
  if (lastRunMs == null) return { level: "critical", label: "No runs recorded", short: "No runs" };
  const age = (now - lastRunMs) / 60000;
  if (age <= 80) return { level: "good", label: "Collector on schedule", short: "Collector OK" };
  if (age <= 180) return { level: "warning", label: "Collector late", short: "Collector late" };
  return { level: "critical", label: "Collector stalled", short: "Collector stalled" };
}
