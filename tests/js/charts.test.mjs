// Pure chart helpers (axis ticks). DOM-free, so they run under Node.
import assert from "node:assert/strict";
import { test } from "node:test";

import { niceTicks, timeTicks } from "../../dashboard/assets/js/lib/charts.js";

const H = 3600e3;
const D = 24 * H;

test("nice linear ticks cover the range", () => {
  assert.deepEqual(niceTicks(0, 5.3, 4), [0, 2, 4, 6]);
  assert.deepEqual(niceTicks(0, 360, 4), [0, 100, 200, 300, 400]);
  assert.deepEqual(niceTicks(2, 2, 4), [2, 2.5, 3]);
});

test("time ticks are spaced and labelled", () => {
  const t0 = Date.UTC(2026, 8, 23);
  const ticks = timeTicks(t0, t0 + 7 * D, 700);
  assert.ok(ticks.length >= 3 && ticks.length <= 12);
  const gaps = ticks.slice(1).map((tk, i) => tk.t - ticks[i].t);
  assert.ok(gaps.every((g) => g === gaps[0]));
  assert.ok(ticks.some((tk) => tk.major && /^Sep \d+$/.test(tk.label)));
  // Never finer than the requested minimum step.
  assert.ok(timeTicks(t0, t0 + 3 * D, 2000, 64, D).every((tk) => tk.t % D === 0));
  assert.deepEqual(timeTicks(t0, t0, 500), []);
});

test("long spans use calendar years", () => {
  const ticks = timeTicks(Date.UTC(1960, 0, 1), Date.UTC(2027, 0, 1), 600);
  assert.ok(ticks.length >= 3);
  assert.ok(ticks.every((tk) => /^\d{4}$/.test(tk.label)));
  assert.ok(ticks.every((tk) => Number(tk.label) % 10 === 0));
});
