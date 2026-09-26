import { columnChart, heatmap } from "../lib/charts.js";
import { CELL_STATES, decodeCell, HOUR_MS, ILLUMINATION, region, REGIONS, summarize, windowHours } from "../lib/coverage.js";
import { card, dataTable, h, legend, statusLabel, tableView, tile } from "../lib/dom.js";
import {
  fmtAgo,
  fmtBytes,
  fmtDateUtc,
  fmtDuration,
  fmtInt,
  fmtPct,
  fmtUtc,
  fmtUtcShort,
  matchesQuery,
  parseTime,
} from "../lib/format.js";
import { cellColor } from "./cameras.js";

export const title = () => "Archive";
export const refreshOnLive = true;

const DAY_MS = 86400000;

export async function render(root, ctx, _args, params) {
  root.append(
    h("h1", null, "Archive"),
    h(
      "p",
      { class: "lede" },
      "What the collector has archived, hour by hour, and what NDBC never published. Every request is recorded in a ledger (URL, SHA-256, HTTP metadata), so a gap is always explained: either NDBC returned 404 at every candidate minute, or the collector missed it.",
    ),
  );
  const st = ctx.live?.status;
  if (!st) {
    root.append(h("div", { class: "notice critical" }, "Live archive status is unavailable right now."));
    return undefined;
  }
  const destroyers = [];
  const hours = windowHours(st.window);

  // Filters (scope everything below).
  const win = h(
    "select",
    { "aria-label": "Time window" },
    [
      ["24", "Last 24 hours"],
      ["72", "Last 72 hours"],
      [String(st.window.hours), `Last ${st.window.hours / 24} days`],
    ].map(([v, l]) => h("option", { value: v }, l)),
  );
  win.value = params.get("h") || "72";
  const reg = h(
    "select",
    { "aria-label": "Region" },
    h("option", { value: "" }, "All regions"),
    Object.entries(REGIONS).map(([k, v]) => h("option", { value: k }, v)),
  );
  reg.value = params.get("region") || "";
  const q = h("input", { type: "search", placeholder: "Filter stations", "aria-label": "Filter stations", value: params.get("q") || "" });
  const onlyGaps = h("input", { type: "checkbox" });
  root.append(h("div", { class: "filters" }, h("label", null, "Window ", win), h("label", null, "Region ", reg), q, h("label", null, onlyGaps, "Only stations with gaps")));

  const body = h("div");
  root.append(body);

  function draw() {
    for (const d of destroyers.splice(0)) d();
    body.replaceChildren();
    const nH = Math.min(Number(win.value) || 72, st.window.hours);
    const off = st.window.hours - nH;
    const stations = st.stations
      .filter((s) => s.coverage && s.images > 0)
      .filter((s) => !reg.value || region(s.station_id) === reg.value)
      .filter((s) => matchesQuery(`${s.station_id} ${s.name || ""}`, q.value))
      .filter((s) => !onlyGaps.checked || /[?x]/.test(s.coverage.slice(off)))
      .sort((a, b) => a.station_id.localeCompare(b.station_id));

    const counts = { archived: 0, not_published: 0, gap: 0 };
    for (const s of stations) {
      const c = summarize(s.coverage.slice(off));
      counts.archived += c.archived;
      counts.not_published += c.not_published;
      counts.gap += c.gap;
    }
    const expected = counts.archived + counts.not_published + counts.gap;
    body.append(
      h(
        "div",
        { class: "tiles" },
        tile("Camera-hours archived", fmtPct(counts.archived, expected), `${fmtInt(counts.archived)} of ${fmtInt(expected)} due`),
        tile("Never published by NDBC", fmtInt(counts.not_published), "404 at every candidate minute stamp"),
        tile(
          "Missed by the collector",
          fmtInt(counts.gap),
          counts.gap ? "no image and no 404 recorded" : "none: every due hour is explained",
        ),
        tile("Stations shown", fmtInt(stations.length), `of ${fmtInt(st.stations.filter((s) => s.images > 0).length)} with images`),
      ),
    );

    // Heatmap.
    const hmCard = card(
      "Hourly coverage by station",
      h("p", { class: "card-sub" }, "One row per camera, one column per UTC hour. Archived cells are coloured by illumination. Click a cell from the last ~72 h to open that image."),
      legend([
        ...["d", "l", "t", "n"].map((k) => ({ label: ILLUMINATION[k].label, color: `var(${ILLUMINATION[k].token})` })),
        { label: CELL_STATES.not_published.label, color: "var(--cell-notpub)" },
        { label: CELL_STATES.gap.label, color: "var(--cell-gap)" },
        { label: CELL_STATES.none.label, color: "var(--cell-empty)" },
      ]),
    );
    body.append(hmCard);
    const colTicks = [];
    for (let c = 0; c < nH; c += 1) {
      const t = hours[off + c];
      if (t % DAY_MS === 0) colTicks.push({ i: c, label: fmtDateUtc(t), major: true });
      else if (nH <= 24 && t % (6 * HOUR_MS) === 0) colTicks.push({ i: c, label: `${String(new Date(t).getUTCHours()).padStart(2, "0")}Z` });
    }
    const now = Date.now();
    const hm = heatmap(hmCard, {
      rows: stations.map((s) => ({ id: s.station_id, label: s.station_id })),
      nCols: nH,
      cellH: 11,
      labelW: 52,
      colTicks,
      ariaLabel: "Coverage heatmap: stations by hour",
      color: (r, c) => cellColor(stations[r].coverage[off + c], (stations[r].illumination || "")[off + c]),
      info: (r, c) => {
        const s = stations[r];
        const i = off + c;
        const cell = decodeCell(s.coverage[i]);
        const t = hours[i];
        const pickable = cell.state === "archived" && now - (t + cell.minute * 60000) < 71 * HOUR_MS;
        const il = ILLUMINATION[(s.illumination || "")[i]] || ILLUMINATION["-"];
        return {
          title: `${s.station_id} · ${fmtUtc(t)} hour`,
          rows: [
            {
              label: cell.state === "archived" ? `stamped :${String(cell.minute).padStart(2, "0")}` : "",
              value: CELL_STATES[cell.state].label,
              color: cellColor(s.coverage[i], (s.illumination || "")[i]),
              kind: "cell",
            },
            { label: "sun", value: il.label },
          ],
          note: pickable ? "Click to view this image" : "Click to open the station",
          pickable: true,
        };
      },
      onPick: (r, c) => {
        const s = stations[r];
        const cell = decodeCell(s.coverage[off + c]);
        const t = hours[off + c] + (cell.minute ?? 0) * 60000;
        ctx.navigate(cell.state === "archived" ? `station/${s.station_id}/${Math.round(t / 1000)}` : `station/${s.station_id}`);
      },
    });
    destroyers.push(hm.destroy);
    hmCard.append(
      tableView(
        [
          { key: "station_id", label: "Station" },
          { key: "archived", label: "Archived", num: true, render: (r) => fmtInt(r.archived) },
          { key: "not_published", label: "Not published", num: true, render: (r) => fmtInt(r.not_published) },
          { key: "gap", label: "Missed", num: true, render: (r) => fmtInt(r.gap) },
        ],
        stations.map((s) => ({ station_id: s.station_id, ...summarize(s.coverage.slice(off)) })),
        "Table view (per station)",
      ),
    );

    // Daily totals over the whole archive.
    const dates = st.daily.dates.map((d) => Date.parse(`${d}T00:00:00Z`));
    const ids = new Set(stations.map((s) => s.station_id));
    const dayRows = dates.map((t, i) => ({
      t,
      values: { images: stations.reduce((a, s) => a + (ids.has(s.station_id) ? s.daily[i] || 0 : 0), 0) },
      bytes: st.daily.bytes[i],
    }));
    const dayCard = card(
      "Images per day, whole archive",
      h("p", { class: "card-sub" }, "For the stations selected above. The first and the current day are partial."),
    );
    const cc = columnChart(dayCard, {
      rows: dayRows,
      keys: [{ key: "images", label: "Images", color: "var(--s1)" }],
      xMin: dates[0],
      xMax: dates[dates.length - 1] + DAY_MS,
      bandMs: DAY_MS,
      height: 170,
      title: (r) => fmtDateUtc(r.t),
      ariaLabel: "Images archived per day",
    });
    destroyers.push(cc.destroy);
    dayCard.append(
      tableView(
        [
          { key: "t", label: "Date (UTC)", render: (r) => new Date(r.t).toISOString().slice(0, 10) },
          { key: "images", label: "Images", num: true, sort: (r) => r.values.images, render: (r) => fmtInt(r.values.images) },
          { key: "bytes", label: "Bytes (all stations)", num: true, render: (r) => fmtBytes(r.bytes) },
        ],
        [...dayRows].reverse(),
      ),
    );

    const a = st.archive;
    const minutes = Object.entries(a.window_minute_stamps || {}).sort((x, y) => y[1] - x[1]);
    const totalMin = minutes.reduce((acc, [, v]) => acc + v, 0);
    const storeCard = card(
      "Storage and file names",
      h(
        "dl",
        { class: "kv" },
        h("dt", null, "Archive size"),
        h("dd", null, `${fmtBytes(a.bytes)} in ${fmtInt(a.images)} images (${fmtBytes(a.images ? a.bytes / a.images : 0)} each on average)`),
        h("dt", null, "First / last image"),
        h("dd", null, `${fmtUtcShort(parseTime(a.first_image))} / ${fmtUtcShort(parseTime(a.last_image))}`),
        h("dt", null, "Minute stamps (7 days)"),
        h("dd", null, minutes.map(([m, v]) => `:${m.padStart(2, "0")} ${fmtPct(v, totalMin)}`).join(" · ") || "–"),
        h("dt", null, "Ledger"),
        h("dd", null, `${fmtInt(a.ledger_rows)} requests recorded; ${fmtInt(a.ledger_verified_images)} verified image downloads`),
      ),
      h(
        "p",
        { class: "card-foot" },
        "NDBC's minute stamp varies per camera and per hour (mostly :10, sometimes :00, rarely :50), so the backfill tries several candidates before recording a 404.",
      ),
    );
    body.append(h("div", { class: "grid grid-2" }, dayCard, storeCard));
  }
  function sync() {
    const p = new URLSearchParams();
    if (win.value !== "72") p.set("h", win.value);
    if (reg.value) p.set("region", reg.value);
    if (q.value) p.set("q", q.value);
    const hash = `#/archive${p.toString() ? `?${p}` : ""}`;
    if (location.hash !== hash) history.replaceState(null, "", hash);
    draw();
  }
  win.addEventListener("change", sync);
  reg.addEventListener("change", sync);
  q.addEventListener("input", sync);
  onlyGaps.addEventListener("change", sync);
  draw();

  // Static panels (not affected by the filters).
  const integrity = [];
  const a = st.archive;
  integrity.push({
    check: "Every archived image has a verified ledger row",
    ok: a.ledger_verified_images >= a.images,
    detail: `${fmtInt(a.images)} files on disk; ${fmtInt(a.ledger_verified_images)} verified downloads in the ledger`,
  });
  integrity.push({
    check: "No unexplained gaps in the last 7 days",
    ok: a.window_coverage.gap === 0,
    detail: `${fmtInt(a.window_coverage.gap)} camera-hours without an image or a recorded 404`,
  });
  const lastRun = parseTime(st.collector.last_run);
  integrity.push({
    check: "Collector ran in the last 80 minutes (as of this snapshot)",
    ok: parseTime(st.generated_at) - lastRun <= 80 * 60000,
    detail: `last run ${fmtUtc(lastRun)}`,
  });
  integrity.push({
    check: "Offsite copy is current",
    ok: st.offsite?.last_sync ? Date.now() - parseTime(st.offsite.last_sync) < 2 * DAY_MS : false,
    detail: st.offsite?.last_sync ? `last copy ${fmtAgo(parseTime(st.offsite.last_sync))}` : "not configured yet (see Operations docs)",
    warnOnly: true,
  });
  const runs = st.collector.runs.map((r) => ({ ...r, t: parseTime(r.at) })).reverse();
  root.append(
    h(
      "div",
      { class: "grid grid-2" },
      card(
        "Integrity checks",
        dataTable(
          [
            { key: "check", label: "Check", sortable: false },
            {
              key: "ok",
              label: "Result",
              sortable: false,
              render: (r) => statusLabel(r.ok ? "good" : r.warnOnly ? "warning" : "critical", r.ok ? "Pass" : r.warnOnly ? "Attention" : "Fail"),
            },
            { key: "detail", label: "Detail", sortable: false },
          ],
          integrity,
        ),
      ),
      card(
        "Realtime spectra snapshots",
        h("p", { class: "card-sub" }, "45-day NDBC realtime files (stdmet, spec, swden, swdir, swdir2, swr1, swr2) for every camera station, gzip-compressed, twice a month."),
        st.realtime.length
          ? dataTable(
              [
                { key: "date", label: "Snapshot" },
                { key: "stations", label: "Stations", num: true, render: (r) => fmtInt(r.stations) },
                { key: "files", label: "Files", num: true, render: (r) => fmtInt(r.files) },
                { key: "bytes", label: "Size", num: true, render: (r) => fmtBytes(r.bytes) },
              ],
              [...st.realtime].reverse(),
            )
          : h("p", { class: "muted" }, "No snapshot yet."),
      ),
    ),
    card(
      "Collector runs, last 7 days",
      tableView(
        [
          { key: "t", label: "Run (UTC)", render: (r) => fmtUtc(r.t) },
          { key: "new_images", label: "New images", num: true, render: (r) => fmtInt(r.new_images) },
          { key: "not_found", label: "404", num: true, render: (r) => fmtInt(r.not_found) },
          { key: "failed", label: "Failed", num: true, render: (r) => fmtInt(r.failed) },
          { key: "duration_s", label: "Duration", num: true, render: (r) => fmtDuration(r.duration_s) },
        ],
        runs,
        `${fmtInt(runs.length)} runs`,
      ),
    ),
  );
  return () => {
    for (const d of destroyers) d();
  };
}
