import { columnChart } from "../lib/charts.js";
import { checkStatus, collectionNotes, countLabel, freshness, MODES, periodLabel, totals } from "../lib/collections.js";
import { collectorHealth, HOUR_MS, hourlyTotals, ILLUMINATION, imageUrl, windowHours } from "../lib/coverage.js";
import { card, dataTable, h, legend, statusLabel, tableView, tile } from "../lib/dom.js";
import {
  compass,
  fmtAgo,
  fmtBytes,
  fmtDuration,
  fmtInt,
  fmtNum,
  fmtPct,
  fmtUtc,
  fmtUtcShort,
  parseTime,
} from "../lib/format.js";

export const title = () => "Overview";
export const refreshOnLive = true;

const ILLUM_KEYS = ["d", "l", "t", "n"].map((k) => ({
  key: k,
  label: ILLUMINATION[k].label,
  color: `var(${ILLUMINATION[k].token})`,
}));

export function projectedBytesPerYear(daily) {
  // Use complete days only: drop the first (partial) and the current day.
  const n = daily.bytes.length;
  const full = daily.bytes.slice(1, Math.max(1, n - 1));
  if (!full.length) return null;
  return (full.reduce((a, b) => a + b, 0) / full.length) * 365;
}

function shortName(name) {
  if (!name) return "";
  return name.split(/ - | \(|, /)[0].trim();
}

export function obsSummary(o) {
  if (!o) return null;
  const parts = [];
  if (o.WVHT != null) parts.push(`Hs ${fmtNum(o.WVHT, 1)} m`);
  if (o.DPD != null) parts.push(`Tp ${fmtNum(o.DPD, 0)} s`);
  if (o.MWD != null) parts.push(`from ${compass(o.MWD)}`);
  if (!parts.length && o.WSPD != null) parts.push(`wind ${fmtNum(o.WSPD, 0)} m/s`);
  return parts.join(" · ") || null;
}

function collectedTable(st) {
  const last = parseTime(st.collector.last_run);
  const health = collectorHealth(last);
  const rt = st.realtime?.length ? st.realtime[st.realtime.length - 1] : null;
  const ss = st.seastate;
  const off = st.offsite?.last_sync ? parseTime(st.offsite.last_sync) : null;
  const rows = [
    {
      stream: "Buoy-camera images",
      schedule: "hourly, minute 40 (70 h backfill)",
      last: [statusLabel(health.level, health.label), h("div", { class: "small muted" }, `last run ${fmtAgo(last)}`)],
    },
    webcoosRow(st),
    {
      stream: "Realtime wave files (stdmet, spec, 5 spectral)",
      schedule: "1st and 15th of each month",
      last: rt
        ? `${rt.date}: ${fmtInt(rt.stations)} stations, ${fmtInt(rt.files)} files, ${fmtBytes(rt.bytes)}`
        : "no snapshot yet",
    },
    {
      stream: "Recent sea state (display only)",
      schedule: "hourly, minute 52",
      last: ss?.refresh
        ? `${fmtInt(ss.stations.length)} stations; ${fmtInt(ss.refresh.requests)} range requests, ${fmtBytes(ss.refresh.bytes_received)}${
            ss.refresh.failures ? `, ${ss.refresh.failures} failed` : ""
          }`
        : `${fmtInt(ss?.stations?.length || 0)} stations (cache only)`,
    },
    {
      stream: "Offsite copy (object storage)",
      schedule: "daily, 04:15",
      last: off ? `last copy ${fmtAgo(off)}` : statusLabel("warning", "Not configured yet"),
    },
  ];
  return dataTable(
    [
      { key: "stream", label: "Stream", sortable: false },
      { key: "schedule", label: "Schedule", sortable: false },
      { key: "last", label: "Latest", sortable: false, render: (r) => h("div", null, r.last) },
    ],
    rows.filter(Boolean),
  );
}

function webcoosRow(st) {
  const c = (st.collections || []).find((x) => x.id === "webcoos");
  if (!c) return null;
  const f = freshness(c);
  const t = parseTime(c.updated);
  return {
    stream: `WebCOOS camera stills (${fmtInt(c.parts.length)} cameras, 30-min grid)`,
    schedule: "hourly, minute 25 (3-day lookback)",
    last: [statusLabel(f.level, f.label), h("div", { class: "small muted" }, `last run ${fmtAgo(t)}`)],
  };
}

const nowrap = (text) => h("span", { style: { whiteSpace: "nowrap" } }, text);

function period(first, last) {
  const label = periodLabel(first, last);
  const [a, b] = label.split(" – ");
  return b ? [nowrap(a), " – ", nowrap(b)] : nowrap(label);
}

function collectionsCard(st) {
  const cols = st.collections || [];
  if (!cols.length) return null;
  const tot = totals(cols);
  const checked = cols.filter((c) => c.check);
  const mismatched = checked.filter((c) => checkStatus(c).level !== "good");
  const rows = cols.map((c) => ({ ...c, t_first: c.first, notes: collectionNotes(c) }));
  const parts = cols.flatMap((c) => (c.parts || []).map((p) => ({ ...p, collection: c.name, unit: c.unit })));
  return card(
    "Data held",
    h(
      "p",
      { class: "card-sub" },
      "Every collection on the collector host, counted from the files on disk and checked against the collection's request ledger. Counts only: PacIOOS and WebCOOS images are not republished here.",
    ),
    h(
      "div",
      { class: "tiles" },
      tile("Images held", fmtInt(tot.images), `in ${fmtInt(tot.imageCollections)} image collections`, { hero: true }),
      tile("Total size", fmtBytes(tot.bytes), `${fmtInt(tot.collections)} collections`),
      h(
        "div",
        { class: "tile" },
        h("div", { class: "tile-label" }, "Files against ledgers"),
        h(
          "div",
          { class: "tile-value", style: { fontSize: "1.15rem", marginTop: "8px" } },
          mismatched.length
            ? statusLabel("critical", `${mismatched.length} mismatch${mismatched.length > 1 ? "es" : ""}`)
            : statusLabel("good", "All match"),
        ),
        h("div", { class: "tile-sub" }, `${fmtInt(checked.length)} collections with a per-file ledger`),
      ),
    ),
    dataTable(
      [
        {
          key: "name",
          label: "Collection",
          render: (c) =>
            h(
              "div",
              null,
              h("a", { href: `#/sources/${c.dataset_id}` }, c.name),
              h("div", { class: "small muted" }, `${c.provider} · ${MODES[c.mode] || c.mode}`),
              c.notes.map((n) => h("div", { class: "small muted" }, n)),
            ),
        },
        { key: "count", label: "Held", num: true, render: (c) => nowrap(countLabel(c.count, c.unit)) },
        { key: "bytes", label: "Size", num: true, render: (c) => nowrap(fmtBytes(c.bytes)) },
        { key: "t_first", label: "Data period (UTC)", render: (c) => period(c.first, c.last) },
        {
          key: "updated",
          label: "Status",
          sortable: false,
          render: (c) => {
            const f = freshness(c);
            const t = parseTime(c.updated);
            return h(
              "div",
              null,
              f ? statusLabel(f.level, f.label) : null,
              h(
                "div",
                { class: "small muted" },
                !t ? "–" : String(c.updated).length === 10 ? `latest ${c.updated}` : `updated ${fmtAgo(t)}`,
              ),
            );
          },
        },
        {
          key: "check",
          label: "Files vs ledger",
          sortable: false,
          render: (c) => {
            const s = checkStatus(c);
            return s.level ? h("span", { title: s.detail }, statusLabel(s.level, s.label)) : h("span", { class: "muted", title: s.detail }, s.label);
          },
        },
      ],
      rows,
      { caption: "Data collections held" },
    ),
    tableView(
      [
        { key: "collection", label: "Collection" },
        { key: "id", label: "Camera / site / station" },
        { key: "count", label: "Held", num: true, render: (p) => fmtInt(p.count) },
        { key: "bytes", label: "Size", num: true, render: (p) => nowrap(fmtBytes(p.bytes)) },
        { key: "first", label: "Data period (UTC)", render: (p) => period(p.first, p.last) },
      ],
      parts,
      `Per camera, site and station (${fmtInt(parts.length)})`,
    ),
  );
}

function gallery(st) {
  const idx = st.window.hours - 1;
  const listed = st.stations.filter((s) => s.listed && s.latest_listed_image);
  let pick = listed.filter((s) => "dl".includes((s.illumination || "")[idx] || "-"));
  const daylight = pick.length > 0;
  if (!daylight) pick = listed;
  pick.sort((a, b) => (b.latest_obs?.WVHT ?? -1) - (a.latest_obs?.WVHT ?? -1));
  pick = pick.slice(0, 8);
  const note = daylight
    ? "Cameras where the sun is up now, largest waves first. Images load from NDBC."
    : "No camera is in daylight right now; showing the latest images. Images load from NDBC.";
  return card(
    "Latest daylight images",
    h("p", { class: "card-sub" }, note),
    h(
      "div",
      { class: "gallery" },
      pick.map((s) =>
        h(
          "a",
          { class: "shot", href: `#/station/${s.station_id}` },
          h(
            "div",
            { class: "frame" },
            h("img", {
              src: imageUrl(s.latest_listed_image),
              alt: `Buoy camera ${s.station_id}, six views`,
              loading: "lazy",
              decoding: "async",
            }),
          ),
          h(
            "div",
            { class: "meta" },
            h("span", { class: "shot-name" }, h("strong", null, s.station_id), ` ${shortName(s.name)}`),
            h("span", { class: "muted" }, obsSummary(s.latest_obs) || "no recent wave observation"),
          ),
        ),
      ),
    ),
  );
}

function researchCards(catalog) {
  const byStatus = { verified: 0, documented: 0, candidate: 0 };
  for (const d of catalog.datasets) byStatus[d.verification.status] = (byStatus[d.verification.status] || 0) + 1;
  const byLevel = {};
  for (const r of catalog.references) {
    const lv = r.verification?.level || "not logged";
    byLevel[lv] = (byLevel[lv] || 0) + 1;
  }
  const levelOrder = ["V1", "V1/V2", "V2", "V2/V3", "not logged"];
  return h(
    "div",
    { class: "grid grid-3" },
    card(
      "Data sources",
      h("p", { class: "card-sub" }, `${catalog.datasets.length} public sources in the registry (${catalog.registry_version ? `v${catalog.registry_version}` : ""}).`),
      h(
        "div",
        { class: "chips" },
        Object.entries(byStatus).map(([k, v]) => h("span", { class: ["badge", k] }, `${v} ${k}`)),
      ),
      h("p", { class: "card-foot" }, h("a", { href: "#/sources" }, "Browse sources →")),
    ),
    card(
      "Literature",
      h("p", { class: "card-sub" }, `${catalog.references.length} references; ${catalog.screened.length} more screened, not yet read.`),
      h(
        "div",
        { class: "chips" },
        levelOrder
          .filter((k) => byLevel[k])
          .map((k) => h("span", { class: ["badge", `lv-${k.replace("/", "-")}`] }, `${byLevel[k]} ${k}`)),
      ),
      h("p", { class: "card-foot" }, h("a", { href: "#/literature" }, "Browse references →")),
    ),
    card(
      "Camera sites and wave references",
      h(
        "p",
        { class: "card-sub" },
        `${catalog.sites.length} public shore-camera sites paired with the nearest wave buoys, plus every NDBC buoy camera (same-hull reference).`,
      ),
      h("p", { class: "card-foot" }, h("a", { href: "#/cameras" }, "Open the map →")),
    ),
  );
}

export async function render(root, ctx) {
  const { catalog, live } = ctx;
  root.append(
    h("h1", null, "Overview"),
    h(
      "p",
      { class: "lede" },
      "This project continuously archives NOAA buoy-camera images (NDBC keeps them only about 72 hours) and pairs them with wave measurements from the same buoys, to train and evaluate vision-based sea-state estimation against instrument ground truth. It also holds shore-camera, buoy and reanalysis collections (Data held, below). The live panels update hourly from the collector.",
    ),
  );
  const st = live?.status;
  if (!st) {
    root.append(
      h(
        "div",
        { class: "notice critical" },
        "Live archive status is unavailable right now",
        live?.error ? ` (${live.error.message || live.error})` : "",
        ". Sources and literature below come from the site build.",
      ),
      researchCards(catalog),
    );
    return;
  }
  if (live.source === "bundled")
    root.append(
      h(
        "div",
        { class: "notice" },
        `Showing the copy of the live data bundled with this site (generated ${fmtUtc(parseTime(st.generated_at))}); the live branch could not be reached.`,
      ),
    );

  const a = st.archive;
  const cov = a.window_coverage;
  const expected = cov.archived + cov.not_published + cov.gap;
  const last = parseTime(st.collector.last_run);
  const health = collectorHealth(last);
  const perYear = projectedBytesPerYear(st.daily);
  root.append(
    h(
      "div",
      { class: "tiles" },
      tile("Buoy-camera images archived", fmtInt(a.images), `since ${fmtUtc(parseTime(a.first_image))}`, { hero: true }),
      tile(
        "Cameras with images",
        `${fmtInt(a.stations_with_images)} of ${fmtInt(a.stations_listed)}`,
        "listed by NDBC right now",
      ),
      tile(
        "Hours archived, last 7 days",
        fmtPct(cov.archived, expected),
        `${fmtInt(cov.not_published)} never published by NDBC · ${fmtInt(cov.gap)} missed`,
      ),
      tile(
        "Archive size",
        fmtBytes(a.bytes),
        perYear ? `≈ ${fmtBytes(perYear)} per year at the recent rate` : "rate available after a full day",
      ),
      h(
        "div",
        { class: "tile" },
        h("div", { class: "tile-label" }, "Collector"),
        h("div", { class: "tile-value", style: { fontSize: "1.15rem", marginTop: "8px" } }, statusLabel(health.level, health.label)),
        h("div", { class: "tile-sub" }, `last run ${fmtUtcShort(last)} (${fmtAgo(last)})`),
      ),
    ),
  );

  const held = collectionsCard(st);
  if (held) root.append(held);

  // Images per hour by illumination, from the first archived hour in the window.
  const hours = windowHours(st.window);
  const totals = hourlyTotals(st.stations, st.window);
  const firstImage = parseTime(a.first_image);
  const start = Math.max(hours[0], firstImage == null ? hours[0] : Math.floor(firstImage / HOUR_MS) * HOUR_MS);
  const rows = hours.map((t, i) => ({ t, values: totals[i], raw: totals[i] })).filter((r) => r.t >= start);
  const chartCard = card(
    "Images archived per hour, last 7 days",
    h(
      "p",
      { class: "card-sub" },
      "By illumination at each camera (sun elevation at minute 10). Night images are kept: ADR 0001 annotates, it does not delete.",
    ),
    legend(ILLUM_KEYS.map((k) => ({ ...k, kind: "rect" }))),
  );
  root.append(chartCard);
  const colChart = columnChart(chartCard, {
    rows,
    keys: ILLUM_KEYS,
    xMin: start,
    xMax: hours[hours.length - 1] + HOUR_MS,
    bandMs: HOUR_MS,
    height: 200,
    ariaLabel: "Stacked columns of images archived per hour by illumination",
    note: (r) =>
      r.raw.not_published || r.raw.gap
        ? `${r.raw.not_published} not published by NDBC, ${r.raw.gap} missed`
        : null,
  });
  chartCard.append(
    tableView(
      [
        { key: "t", label: "Hour (UTC)", render: (r) => fmtUtc(r.t) },
        ...ILLUM_KEYS.map((k) => ({ key: k.key, label: k.label, num: true, sort: (r) => r.values[k.key], render: (r) => fmtInt(r.values[k.key]) })),
        { key: "np", label: "Not published", num: true, sort: (r) => r.raw.not_published, render: (r) => fmtInt(r.raw.not_published) },
        { key: "gap", label: "Missed", num: true, sort: (r) => r.raw.gap, render: (r) => fmtInt(r.raw.gap) },
      ],
      [...rows].reverse(),
    ),
  );

  // Collector runs.
  const runs = st.collector.runs.map((r) => ({ ...r, t: parseTime(r.at) }));
  const runsCard = card(
    "Collector runs",
    h(
      "p",
      { class: "card-sub" },
      "New images per hourly run. A tall column after a quiet period is the backfill recovering images after downtime (up to 70 h).",
    ),
    legend([
      { label: "New images", color: "var(--s1)" },
      { label: "404 (never published)", color: "var(--cell-notpub)" },
    ]),
  );
  const colRuns = columnChart(runsCard, {
    rows: runs.map((r) => ({ t: r.t, values: { new_images: r.new_images, not_found: r.not_found }, run: r })),
    keys: [
      { key: "new_images", label: "New images", color: "var(--s1)" },
      { key: "not_found", label: "404", color: "var(--cell-notpub)" },
    ],
    xMin: Math.min(start, ...runs.map((r) => r.t)),
    xMax: hours[hours.length - 1] + HOUR_MS,
    bandMs: HOUR_MS,
    height: 160,
    ariaLabel: "Images fetched per collector run",
    title: (r) => `Run at ${fmtUtcShort(r.t)}`,
    note: (r) => `took ${fmtDuration(r.run.duration_s)}${r.run.failed ? `, ${r.run.failed} failed requests` : ""}`,
  });
  runsCard.append(
    tableView(
      [
        { key: "t", label: "Run (UTC)", render: (r) => fmtUtc(r.t) },
        { key: "new_images", label: "New images", num: true, render: (r) => fmtInt(r.new_images) },
        { key: "not_found", label: "404", num: true, render: (r) => fmtInt(r.not_found) },
        { key: "failed", label: "Failed", num: true, render: (r) => fmtInt(r.failed) },
        { key: "duration_s", label: "Duration", num: true, render: (r) => fmtDuration(r.duration_s) },
      ],
      [...runs].reverse(),
    ),
  );

  root.append(
    h("div", { class: "grid grid-2" }, runsCard, card("What is collected", collectedTable(st))),
  );
  root.append(gallery(st));
  root.append(researchCards(catalog));

  return () => {
    colChart.destroy();
    colRuns.destroy();
  };
}
