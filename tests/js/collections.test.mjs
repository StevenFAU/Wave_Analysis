// Collection summaries from status.json (built by src/wave_analysis/dashboard/collections.py).
import assert from "node:assert/strict";
import { test } from "node:test";

import {
  checkStatus,
  collectionNotes,
  countLabel,
  freshness,
  periodLabel,
  staleParts,
  totals,
} from "../../dashboard/assets/js/lib/collections.js";

const NOW = Date.parse("2026-09-27T14:00:00Z");
const check = (over = {}) => ({ on_disk: 10, ledger: 10, missing_files: 0, unledgered_files: 0, ...over });

const webcoos = {
  id: "webcoos",
  mode: "hourly",
  unit: "images",
  count: 30,
  bytes: 3000,
  updated: "2026-09-27T13:26:00Z",
  not_found: 0,
  failed: 0,
  check: check(),
  parts: [
    { id: "oakisland_east", last: "2026-09-27T13:00:05Z" },
    { id: "jennette_north", last: "2026-09-16T18:29:57Z" },
    { id: "jennette_south", last: "2026-09-16T18:29:36Z" },
  ],
};

test("files against ledger", () => {
  assert.equal(checkStatus(webcoos).level, "good");
  const bad = checkStatus({ check: check({ on_disk: 9, missing_files: 1 }) });
  assert.equal(bad.level, "critical");
  assert.match(bad.detail, /1 ledgered files missing/);
  assert.equal(checkStatus({}).level, null);
});

test("freshness only for scheduled collections", () => {
  assert.equal(freshness(webcoos, NOW).level, "good");
  assert.equal(freshness({ ...webcoos, updated: "2026-09-27T06:00:00Z" }, NOW).level, "warning");
  assert.equal(freshness({ ...webcoos, updated: "2026-09-25T06:00:00Z" }, NOW).level, "critical");
  assert.equal(freshness({ mode: "complete", updated: "2020-01-01T00:00:00Z" }, NOW), null);
  // Realtime snapshots are due on the 1st and 15th.
  assert.equal(freshness({ mode: "twice_monthly", updated: "2026-09-15" }, NOW).level, "good");
  assert.equal(freshness({ mode: "twice_monthly", updated: "2026-09-05" }, NOW).level, "warning");
  assert.equal(freshness({ mode: "twice_monthly", updated: "2026-08-15" }, NOW).level, "critical");
});

test("count labels", () => {
  assert.equal(countLabel(1, "files"), "1 file");
  assert.equal(countLabel(8145, "images"), "8,145 images");
  assert.equal(countLabel(69, "site-months"), "69 site-months");
});

test("cameras without new images are named", () => {
  assert.deepEqual(
    staleParts(webcoos, NOW).map((p) => p.id),
    ["jennette_north", "jennette_south"],
  );
  assert.deepEqual(staleParts({ ...webcoos, mode: "complete" }, NOW), []);
  const notes = collectionNotes(webcoos, NOW);
  assert.ok(notes.includes("No new images since 2026-09-16: jennette_north, jennette_south"));
});

test("totals count images only from image collections", () => {
  const t = totals([webcoos, { unit: "site-months", count: 69, bytes: 5 }, { unit: "images", count: 2, bytes: 1 }]);
  assert.deepEqual(t, { images: 32, bytes: 3006, imageCollections: 2, collections: 3 });
});

test("notes for pairing and ERA5 states", () => {
  const waimea = {
    id: "pacioos_waimea",
    mode: "complete",
    not_found: 165,
    pairing: { reference: "CDIP 106", paired: 16404, paired_days: 800, trusted_time: 17051 },
  };
  assert.deepEqual(collectionNotes(waimea, NOW), [
    "165 listed images no longer served (HTTP 404)",
    "17,051 with a verified capture time; 16,404 paired with CDIP 106 on 800 days",
  ]);
  const era5 = { id: "era5_waves", mode: "on_request", states: { final: 60, incomplete: 3, preliminary: 6 } };
  assert.deepEqual(collectionNotes(era5, NOW), [
    "60 final, 6 preliminary (ERA5T), 3 incomplete (month not over)",
  ]);
});

test("period labels", () => {
  assert.equal(periodLabel("2009-02-06T03:00:00Z", "2013-10-15T03:00:00Z"), "2009-02-06 – 2013-10-15");
  assert.equal(periodLabel("2009-01", "2026-09"), "2009-01 – 2026-09");
  assert.equal(periodLabel("2026-09-26", "2026-09-26"), "2026-09-26");
  assert.equal(periodLabel(null, null), "–");
});

test("notes for the NDBC history and research-dataset collections", () => {
  const hist = { id: "ndbc_history", mode: "on_request", parts_count: 104, parts: [{ id: "stdmet" }] };
  assert.deepEqual(collectionNotes(hist, NOW), ["104 stations; stdmet files"]);
  const ext = { id: "external", mode: "complete", parts: [{ id: "a" }, { id: "b" }] };
  assert.match(collectionNotes(ext, NOW)[0], /^2 datasets/);
});
