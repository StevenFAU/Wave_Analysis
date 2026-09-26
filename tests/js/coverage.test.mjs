// Decoding of status.json coverage strings (encoding: src/wave_analysis/dashboard/status.py).
import assert from "node:assert/strict";
import { test } from "node:test";

import {
  collectorHealth,
  decodeCell,
  hourlyTotals,
  hsBin,
  hsBinLabel,
  imageFileName,
  region,
  stationImages,
  summarize,
  windowHours,
} from "../../dashboard/assets/js/lib/coverage.js";

const window = { start: "2026-09-26T07:00:00Z", hours: 6, as_of: "2026-09-26T12:40:00Z" };
const station = { station_id: "41010", camera: "W04A", coverage: "01x?1.", illumination: "ddddln" };

test("cells", () => {
  assert.deepEqual(decodeCell("1"), { state: "archived", minute: 10 });
  assert.deepEqual(decodeCell("0"), { state: "archived", minute: 0 });
  assert.deepEqual(decodeCell("5"), { state: "archived", minute: 50 });
  assert.equal(decodeCell("x").state, "not_published");
  assert.equal(decodeCell("?").state, "gap");
  assert.equal(decodeCell(".").state, "none");
  assert.deepEqual(summarize(station.coverage), { archived: 3, not_published: 1, gap: 1, none: 1 });
});

test("file names match NDBC's", () => {
  assert.equal(imageFileName("W04A", Date.UTC(2026, 8, 26, 13, 10)), "W04A_2026_09_26_1310.jpg");
});

test("images still served upstream", () => {
  const now = Date.parse("2026-09-26T12:45:00Z");
  const imgs = stationImages(station, window, now);
  assert.deepEqual(
    imgs.map((i) => i.file),
    ["W04A_2026_09_26_0700.jpg", "W04A_2026_09_26_0810.jpg", "W04A_2026_09_26_1110.jpg"],
  );
  assert.ok(imgs.every((i) => i.upstream));
  assert.equal(imgs[2].illum, "l");
  const later = stationImages(station, window, now + 72 * 3600e3);
  assert.ok(later.every((i) => !i.upstream));
  assert.deepEqual(stationImages({ ...station, camera: null }, window, now), []);
});

test("hourly totals by illumination", () => {
  const rows = hourlyTotals([station, { ...station, station_id: "41011" }], window, (s) => s.station_id === "41010");
  assert.equal(rows.length, 6);
  assert.equal(rows[0].d, 1);
  assert.equal(rows[2].not_published, 1);
  assert.equal(rows[3].gap, 1);
  assert.equal(rows[4].l, 1);
  assert.equal(windowHours(window)[5], Date.parse("2026-09-26T12:00:00Z"));
});

test("wave-height bins", () => {
  assert.equal(hsBin(0.4), 0);
  assert.equal(hsBin(1), 1);
  assert.equal(hsBin(5.3), 4);
  assert.equal(hsBin(null), null);
  assert.equal(hsBinLabel(0), "< 1 m");
  assert.equal(hsBinLabel(2), "2–3 m");
  assert.equal(hsBinLabel(4), "≥ 4 m");
});

test("collector health thresholds", () => {
  const now = Date.UTC(2026, 8, 26, 12);
  assert.equal(collectorHealth(now - 30 * 60e3, now).level, "good");
  assert.equal(collectorHealth(now - 2 * 3600e3, now).level, "warning");
  assert.equal(collectorHealth(now - 5 * 3600e3, now).level, "critical");
  assert.equal(collectorHealth(null, now).level, "critical");
});

test("regions from WMO station-id prefix", () => {
  assert.equal(region("41010"), "41");
  assert.equal(region("51201"), "51");
  assert.equal(region("ABC12"), "other");
});
