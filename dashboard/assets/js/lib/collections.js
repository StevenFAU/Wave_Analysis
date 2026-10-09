// Summaries of the status.json `collections` list (built by
// src/wave_analysis/dashboard/collections.py and status.py). Pure; unit-tested
// in tests/js/collections.test.mjs.
import { fmtInt } from "./format.js";

const HOUR_MS = 3600 * 1000;
const DAY_MS = 24 * HOUR_MS;

export const MODES = {
  hourly: "Collecting hourly",
  twice_monthly: "Snapshots twice a month",
  complete: "Complete (historical archive)",
  on_request: "Downloaded on request",
};

/** Files on disk against the collection's verified ledger rows. */
export function checkStatus(c) {
  if (!c.check) return { level: null, label: "–", detail: "No per-file ledger" };
  const { on_disk, ledger, missing_files, unledgered_files } = c.check;
  if (!missing_files && !unledgered_files)
    return { level: "good", label: "Match", detail: `${fmtInt(on_disk)} files, each with a verified ledger row` };
  const parts = [];
  if (missing_files) parts.push(`${fmtInt(missing_files)} ledgered files missing`);
  if (unledgered_files) parts.push(`${fmtInt(unledgered_files)} files without a ledger row`);
  return { level: "critical", label: "Mismatch", detail: `${parts.join("; ")} (${fmtInt(on_disk)} on disk, ${fmtInt(ledger)} in ledger)` };
}

/** How current a scheduled collection is (null for complete or on-request ones). */
export function freshness(c, now = Date.now()) {
  const limits = { hourly: [3, 24], twice_monthly: [17 * 24, 32 * 24] }[c.mode];
  if (!limits) return null;
  const t = Date.parse(c.updated);
  if (Number.isNaN(t)) return { level: "critical", label: "No run recorded" };
  const h = (now - t) / HOUR_MS;
  if (h <= limits[0]) return { level: "good", label: "Up to date" };
  if (h <= limits[1]) return { level: "warning", label: "Late" };
  return { level: "critical", label: "Stalled" };
}

/** "1 file", "8,145 images", "69 site-months". */
export function countLabel(n, unit) {
  const singular = { images: "image", files: "file", "site-months": "site-month" }[unit];
  return `${fmtInt(n)} ${n === 1 && singular ? singular : unit}`;
}

/** Parts (cameras) of an hourly collection whose newest item is older than `days`.
 *  Parts marked `hourly: false` (past years only) are not expected to be current. */
export function staleParts(c, now = Date.now(), days = 2) {
  if (c.mode !== "hourly" || !c.parts) return [];
  return c.parts.filter((p) => p.hourly !== false && p.last && now - Date.parse(p.last) > days * DAY_MS);
}

/** Image count and total bytes across collections. */
export function totals(collections) {
  let images = 0;
  let bytes = 0;
  let imageCollections = 0;
  for (const c of collections) {
    bytes += c.bytes || 0;
    if (c.unit === "images") {
      images += c.count || 0;
      imageCollections += 1;
    }
  }
  return { images, bytes, imageCollections, collections: collections.length };
}

/** "2009-02-06 – 2013-10-15" (dates) or "2009-01 – 2026-09" (months). */
export function periodLabel(first, last) {
  if (!first) return "–";
  const d = (s) => String(s).slice(0, 10);
  return d(first) === d(last) || !last ? d(first) : `${d(first)} – ${d(last)}`;
}

const ERA5_STATES = { final: "final", preliminary: "preliminary (ERA5T)", incomplete: "incomplete (month not over)", unknown: "state unknown" };

/** Short plain-language notes about one collection. */
export function collectionNotes(c, now = Date.now()) {
  const notes = [];
  if (c.id === "ndbc_buoycam") {
    notes.push(`${fmtInt(c.parts_count)} cameras; ${fmtInt(c.not_found)} camera-hours never published by NDBC`);
  } else if (c.id === "ndbc_realtime") {
    notes.push(`45-day spectral and meteorological files for ${fmtInt(c.parts_count)} camera stations`);
  } else if (c.id === "ndbc_history") {
    notes.push(`${fmtInt(c.parts_count)} stations; ${(c.parts || []).map((p) => p.id).join(", ")} files`);
  } else if (c.id === "external") {
    notes.push(`${fmtInt((c.parts || []).length)} datasets, each file checked against its repository's published sizes and checksums`);
  } else if (c.not_found) {
    notes.push(`${fmtInt(c.not_found)} listed images no longer served (HTTP 404)`);
  }
  if (c.pairing) {
    const p = c.pairing;
    notes.push(
      `${fmtInt(p.trusted_time)} with a verified capture time; ${fmtInt(p.paired)} paired with ${p.reference} on ${fmtInt(p.paired_days)} days`,
    );
  }
  if (c.states) {
    const order = ["final", "preliminary", "incomplete", "unknown"];
    notes.push(
      order
        .filter((k) => c.states[k])
        .map((k) => `${fmtInt(c.states[k])} ${ERA5_STATES[k]}`)
        .join(", "),
    );
  }
  const stale = staleParts(c, now);
  if (stale.length) {
    const since = stale.map((p) => String(p.last).slice(0, 10)).sort()[0];
    notes.push(`No new images since ${since}: ${stale.map((p) => p.id).join(", ")}`);
  }
  if (c.failed) notes.push(`${fmtInt(c.failed)} requests failed (will be retried)`);
  return notes;
}
