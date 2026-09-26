// Front-end formatting helpers. Run: node --test tests/js/
import assert from "node:assert/strict";
import { test } from "node:test";

import {
  compass,
  fmtAgo,
  fmtBytes,
  fmtCompact,
  fmtInt,
  fmtLatLon,
  fmtNum,
  fmtPct,
  fmtUtc,
  fmtUtcShort,
  matchesQuery,
  parseTime,
} from "../../dashboard/assets/js/lib/format.js";

test("numbers", () => {
  assert.equal(fmtInt(6335), "6,335");
  assert.equal(fmtInt(null), "–");
  assert.equal(fmtNum(2.345, 1), "2.3");
  assert.equal(fmtNum(Number.NaN), "–");
  assert.equal(fmtCompact(1284), "1,284");
  assert.equal(fmtCompact(12900), "12.9K");
  assert.equal(fmtCompact(4.2e6), "4.2M");
  assert.equal(fmtPct(99, 100), "99.0 %");
  assert.equal(fmtPct(1, 0), "–");
});

test("bytes use decimal units", () => {
  assert.equal(fmtBytes(512), "512 B");
  assert.equal(fmtBytes(43000), "43 KB");
  assert.equal(fmtBytes(266848508), "267 MB");
  assert.equal(fmtBytes(3.2e10), "32 GB");
  assert.equal(fmtBytes(1.5e9), "1.5 GB");
});

test("times are UTC", () => {
  const t = parseTime("2026-09-26T14:42:12Z");
  assert.equal(fmtUtc(t), "2026-09-26 14:42Z");
  assert.equal(fmtUtc(t, { seconds: true }), "2026-09-26 14:42:12Z");
  assert.equal(fmtUtcShort(t), "Sep 26 14:42Z");
  assert.equal(parseTime("not a time"), null);
  assert.equal(parseTime(null), null);
});

test("relative ages", () => {
  const now = Date.UTC(2026, 8, 26, 12);
  assert.equal(fmtAgo(now - 20e3, now), "just now");
  assert.equal(fmtAgo(now - 12 * 60e3, now), "12 min ago");
  assert.equal(fmtAgo(now - 3.5 * 3600e3, now), "3.5 h ago");
  assert.equal(fmtAgo(now - 3 * 86400e3, now), "3 d ago");
  assert.equal(fmtAgo(now + 30 * 60e3, now), "in 30 min");
});

test("compass points (direction from)", () => {
  assert.equal(compass(0), "N");
  assert.equal(compass(359), "N");
  assert.equal(compass(22.5), "NNE");
  assert.equal(compass(-90), "W");
  assert.equal(compass(null), "–");
});

test("coordinates and search", () => {
  assert.equal(fmtLatLon(28.86, -78.478), "28.860°N 78.478°W");
  assert.ok(matchesQuery("Guimarães et al. stereo", "guimaraes STEREO"));
  assert.ok(!matchesQuery("41010 CANAVERAL", "hatteras"));
  assert.ok(matchesQuery("anything", ""));
});
