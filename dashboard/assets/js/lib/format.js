// Pure formatting helpers (unit-tested in tests/js/format.test.mjs).

const intFmt = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });

export const DASH = "–";

export function fmtInt(n) {
  return n == null || Number.isNaN(n) ? DASH : intFmt.format(Math.round(n));
}

export function fmtNum(n, digits = 1) {
  if (n == null || Number.isNaN(n)) return DASH;
  return new Intl.NumberFormat("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(n);
}

/** 1,284 / 12.9K / 4.2M (stat tiles). */
export function fmtCompact(n) {
  if (n == null || Number.isNaN(n)) return DASH;
  const a = Math.abs(n);
  if (a < 10000) return fmtInt(n);
  if (a < 1e6) return `${fmtNum(n / 1e3, a < 1e5 ? 1 : 0)}K`;
  return `${fmtNum(n / 1e6, 1)}M`;
}

/** Decimal byte units (1 MB = 10^6 B), matching the collection docs. */
export function fmtBytes(b) {
  if (b == null || Number.isNaN(b)) return DASH;
  const units = ["B", "KB", "MB", "GB", "TB"];
  let v = b;
  let i = 0;
  while (Math.abs(v) >= 1000 && i < units.length - 1) {
    v /= 1000;
    i += 1;
  }
  return `${i === 0 ? fmtInt(v) : fmtNum(v, v < 10 ? 1 : 0)} ${units[i]}`;
}

export function fmtPct(part, whole, digits = 1) {
  if (!whole) return DASH;
  return `${fmtNum((100 * part) / whole, digits)} %`;
}

/** Parse an ISO time; returns milliseconds or null. */
export function parseTime(s) {
  if (s == null) return null;
  const t = Date.parse(s);
  return Number.isNaN(t) ? null : t;
}

const pad = (n) => String(n).padStart(2, "0");
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

/** "2026-09-26 14:42Z" */
export function fmtUtc(ms, { seconds = false } = {}) {
  if (ms == null) return DASH;
  const d = new Date(ms);
  const base = `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())} ${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}`;
  return `${base}${seconds ? `:${pad(d.getUTCSeconds())}` : ""}Z`;
}

/** "Sep 26 14:42Z" */
export function fmtUtcShort(ms) {
  if (ms == null) return DASH;
  const d = new Date(ms);
  return `${MONTHS[d.getUTCMonth()]} ${d.getUTCDate()} ${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}Z`;
}

export function fmtDateUtc(ms) {
  if (ms == null) return DASH;
  const d = new Date(ms);
  return `${MONTHS[d.getUTCMonth()]} ${d.getUTCDate()}`;
}

export function fmtHourUtc(ms) {
  const d = new Date(ms);
  return `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}`;
}

/** "just now", "12 min ago", "3 h ago", "2 d ago" (future times read "in ..."). */
export function fmtAgo(ms, now = Date.now()) {
  if (ms == null) return DASH;
  const s = (now - ms) / 1000;
  const a = Math.abs(s);
  let txt;
  if (a < 60) return "just now";
  if (a < 3600) txt = `${Math.round(a / 60)} min`;
  else if (a < 48 * 3600) txt = `${fmtNum(a / 3600, a < 10 * 3600 ? 1 : 0)} h`;
  else txt = `${Math.round(a / 86400)} d`;
  return s >= 0 ? `${txt} ago` : `in ${txt}`;
}

export function fmtDuration(seconds) {
  if (seconds == null) return DASH;
  if (seconds < 90) return `${Math.round(seconds)} s`;
  if (seconds < 5400) return `${Math.round(seconds / 60)} min`;
  return `${fmtNum(seconds / 3600, 1)} h`;
}

const POINTS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"];

/** 16-point compass name for a direction in degrees (direction *from*). */
export function compass(deg) {
  if (deg == null || Number.isNaN(deg)) return DASH;
  const i = Math.round((((deg % 360) + 360) % 360) / 22.5) % 16;
  return POINTS[i];
}

export function fmtLatLon(lat, lon) {
  if (lat == null || lon == null) return DASH;
  const ns = lat >= 0 ? "N" : "S";
  const ew = lon >= 0 ? "E" : "W";
  return `${fmtNum(Math.abs(lat), 3)}°${ns} ${fmtNum(Math.abs(lon), 3)}°${ew}`;
}

/** Case- and accent-insensitive "does haystack contain every word of needle". */
export function matchesQuery(haystack, query) {
  if (!query) return true;
  const fold = (s) => s.normalize("NFKD").replace(/[̀-ͯ]/g, "").toLowerCase();
  const h = fold(haystack);
  return fold(query)
    .split(/\s+/)
    .filter(Boolean)
    .every((w) => h.includes(w));
}
