import { createSync, heatmap, lineChart, yearRanges } from "../lib/charts.js";
import { CELL_STATES, decodeCell, HOUR_MS, ILLUMINATION, stationImages, summarize, windowHours } from "../lib/coverage.js";
import { blockRows, loadSeaState, nearestRow } from "../lib/data.js";
import { card, clear, dataTable, ext, h, legend, tableView } from "../lib/dom.js";
import {
  compass,
  DASH,
  fmtAgo,
  fmtBytes,
  fmtDateUtc,
  fmtInt,
  fmtLatLon,
  fmtNum,
  fmtPct,
  fmtUtc,
  fmtUtcShort,
  parseTime,
} from "../lib/format.js";
import { cellColor } from "./cameras.js";

export const title = ([id]) => `Station ${id}`;

const PRODUCT_LABELS = {
  stdmet: "Met + bulk waves",
  swden: "Spectral density",
  swdir: "Direction α1",
  swdir2: "Direction α2",
  swr1: "r1",
  swr2: "r2",
  adcp: "ADCP currents",
};

function obsBox(label, value, unit) {
  return h("div", { class: "obs" }, h("div", { class: "k" }, label), h("div", { class: "v" }, value, unit ? h("span", { class: "u" }, ` ${unit}`) : null));
}

function dirText(deg) {
  return deg == null ? DASH : `${fmtNum(deg, 0)}° ${compass(deg)}`;
}

function obsPanel(stdRows, specRows, t) {
  // NDBC reports variables on different rows (e.g. DPD only on some wave rows),
  // so each value comes from its own nearest row within 40 minutes.
  const near = (rows, key) => nearestRow(rows, t, key);
  const val = (rows, key) => near(rows, key)?.[key] ?? null;
  const wave = near(stdRows, "WVHT");
  const wind = near(stdRows, "WSPD");
  const spec = near(specRows, "SwH");
  if (!wave && !wind && !spec)
    return h("p", { class: "muted small" }, "No buoy observation within 40 minutes of this image.");
  const lag = (row) => {
    if (!row) return " (none)";
    const m = Math.round((row.t - t) / 60000);
    return m === 0 ? " (same minute)" : ` (${m > 0 ? "+" : ""}${m} min)`;
  };
  const steep = val(specRows, "STEEPNESS");
  return h(
    "div",
    null,
    h(
      "div",
      { class: "obs-grid" },
      obsBox("Significant height Hs", fmtNum(val(stdRows, "WVHT"), 1), "m"),
      obsBox("Dominant period", fmtNum(val(stdRows, "DPD"), 0), "s"),
      obsBox("Average period", fmtNum(val(stdRows, "APD"), 1), "s"),
      obsBox("Mean direction (from)", dirText(val(stdRows, "MWD"))),
      obsBox("Swell H / T", spec ? `${fmtNum(spec.SwH, 1)} m / ${fmtNum(spec.SwP, 0)} s` : DASH),
      obsBox("Wind-sea H / T", spec ? `${fmtNum(spec.WWH, 1)} m / ${fmtNum(spec.WWP, 0)} s` : DASH),
      obsBox("Steepness", steep ? String(steep).replace("_", " ").toLowerCase() : DASH),
      obsBox("Wind (gust)", wind ? `${fmtNum(wind.WSPD, 0)} (${fmtNum(val(stdRows, "GST"), 0)})` : DASH, "m/s"),
      obsBox("Wind from", dirText(val(stdRows, "WDIR"))),
      obsBox("Water temp.", fmtNum(val(stdRows, "WTMP"), 1), "°C"),
    ),
    h(
      "p",
      { class: "card-foot" },
      `Offset of the nearest NDBC realtime rows from the image time: waves${lag(wave)}, wind${lag(wind)}, spectral summary${lag(spec)}. Provisional, not quality-controlled.`,
    ),
  );
}

function viewer(rec, st, stdRowsRef, specRowsRef, initialT, onChange) {
  const images = stationImages(rec, st.window).filter((im) => im.upstream);
  const box = h("div");
  if (!images.length) {
    box.append(
      h(
        "p",
        { class: "muted" },
        rec.images
          ? "All of this camera's archived images are older than NDBC's ~72 h retention, so none can be shown here. They are kept in the project archive."
          : "No images archived for this camera yet.",
      ),
    );
    return { el: box, select() {}, current: () => null, destroy() {} };
  }
  let idx = images.length - 1;
  if (initialT) {
    let best = Infinity;
    images.forEach((im, i) => {
      const d = Math.abs(im.t - initialT);
      if (d < best) {
        best = d;
        idx = i;
      }
    });
  }
  // On phones the 2880x300 strip is ~36 px tall: start with the six views split.
  let split = window.matchMedia("(max-width: 720px)").matches;
  const stage = h("div", { class: "viewer", tabindex: 0, "aria-label": "Image viewer; use the arrow keys to step through images" });
  const caption = h("div", { class: "small" });
  const slider = h("input", { type: "range", min: 0, max: images.length - 1, step: 1, value: idx, "aria-label": "Image time" });
  const prev = h("button", { class: "icon-btn", type: "button", "aria-label": "Previous image" }, "◀");
  const next = h("button", { class: "icon-btn", type: "button", "aria-label": "Next image" }, "▶");
  const latest = h("button", { class: "icon-btn", type: "button" }, "Latest");
  const splitBtn = h("button", { class: "icon-btn", type: "button", "aria-pressed": String(split) }, "Split six views");
  const orig = h("a", { target: "_blank", rel: "noopener noreferrer", class: "small" }, "Open original");
  const obsHost = h("div");

  function show() {
    const im = images[idx];
    clear(stage);
    if (split) {
      const grid = h("div", { class: "views6" });
      for (let v = 0; v < 6; v += 1)
        grid.append(
          h("div", {
            role: "img",
            "aria-label": `View ${v + 1} of 6`,
            style: { backgroundImage: `url("${im.url}")`, backgroundPosition: `${(v / 5) * 100}% 0` },
          }),
        );
      stage.append(grid);
    } else {
      const img = h("img", { class: "strip", src: im.url, alt: `Buoy ${rec.station_id} camera, six views, ${fmtUtc(im.t)}`, decoding: "async" });
      img.addEventListener("error", () => {
        clear(stage);
        stage.append(h("div", { class: "placeholder" }, "This image could not be loaded from NDBC (it may have just aged out of the ~72 h window)."));
      });
      stage.append(img);
    }
    for (const j of [idx - 1, idx + 1]) if (images[j]) new Image().src = images[j].url;
    const il = ILLUMINATION[im.illum] || ILLUMINATION["-"];
    caption.replaceChildren(
      h("strong", null, fmtUtc(im.t)),
      ` · ${fmtAgo(im.t)} · ${il.label.toLowerCase()} · image ${idx + 1} of ${images.length} still served by NDBC`,
    );
    slider.value = String(idx);
    orig.href = im.url;
    prev.disabled = idx === 0;
    next.disabled = idx === images.length - 1;
    obsHost.replaceChildren(obsPanel(stdRowsRef(), specRowsRef(), im.t));
    onChange(im);
  }
  const go = (i) => {
    idx = Math.max(0, Math.min(images.length - 1, i));
    show();
  };
  slider.addEventListener("input", () => go(Number(slider.value)));
  prev.addEventListener("click", () => go(idx - 1));
  next.addEventListener("click", () => go(idx + 1));
  latest.addEventListener("click", () => go(images.length - 1));
  splitBtn.addEventListener("click", () => {
    split = !split;
    splitBtn.setAttribute("aria-pressed", String(split));
    show();
  });
  const onKey = (e) => {
    if (e.target.closest("input, textarea, select")) return;
    if (e.key === "ArrowLeft") go(idx - 1);
    else if (e.key === "ArrowRight") go(idx + 1);
    else return;
    e.preventDefault();
  };
  stage.addEventListener("keydown", onKey);
  box.append(stage, h("div", { class: "viewer-bar" }, prev, slider, next, latest, splitBtn, orig), caption, obsHost);
  show();
  return {
    el: box,
    current: () => images[idx],
    select(t) {
      let best = Infinity;
      let bi = idx;
      images.forEach((im, i) => {
        const d = Math.abs(im.t - t);
        if (d < best) {
          best = d;
          bi = i;
        }
      });
      go(bi);
    },
    refreshObs() {
      obsHost.replaceChildren(obsPanel(stdRowsRef(), specRowsRef(), images[idx].t));
    },
    destroy() {},
  };
}

function coverageGrid(host, rec, st, onPick) {
  const first = parseTime(rec.first_image);
  const hours = windowHours(st.window);
  // Start at the day of the station's first archived image (no empty rows before it).
  const firstDay = first == null ? -Infinity : Math.floor(first / 86400000);
  const days = [...new Set(hours.map((t) => Math.floor(t / 86400000)))].filter((d) => d >= firstDay);
  const cellIndex = new Map(hours.map((t, i) => [t, i]));
  const rows = days.map((d) => ({ id: d, label: fmtDateUtc(d * 86400000) }));
  const now = Date.now();
  const at = (r, c) => {
    const t = days[r] * 86400000 + c * HOUR_MS;
    return { t, i: cellIndex.get(t) };
  };
  return heatmap(host, {
    rows,
    nCols: 24,
    cellH: 18,
    labelW: 56,
    colTicks: [0, 3, 6, 9, 12, 15, 18, 21].map((c) => ({ i: c, label: `${String(c).padStart(2, "0")}Z`, major: c === 0 })),
    ariaLabel: `Hourly coverage of station ${rec.station_id} by day`,
    color: (r, c) => {
      const { i } = at(r, c);
      if (i == null) return null;
      return cellColor(rec.coverage[i], (rec.illumination || "")[i]);
    },
    info: (r, c) => {
      const { t, i } = at(r, c);
      if (i == null) return { title: fmtUtc(t), rows: [{ label: "outside the 7-day window", value: DASH }] };
      const cell = decodeCell(rec.coverage[i]);
      const il = ILLUMINATION[(rec.illumination || "")[i]] || ILLUMINATION["-"];
      const upstream = cell.state === "archived" && now - (t + cell.minute * 60000) < 71 * HOUR_MS;
      return {
        title: `${fmtUtc(t)} hour`,
        rows: [
          { label: cell.state === "archived" ? `stamped :${String(cell.minute).padStart(2, "0")}` : "", value: CELL_STATES[cell.state].label, color: cellColor(rec.coverage[i], (rec.illumination || "")[i]), kind: "cell" },
          { label: "sun", value: il.label },
        ],
        note: upstream ? "Click to view this image" : null,
        pickable: upstream,
      };
    },
    onPick: (r, c) => {
      const { t, i } = at(r, c);
      if (i == null) return;
      const cell = decodeCell(rec.coverage[i]);
      if (cell.state === "archived") onPick(t + cell.minute * 60000);
    },
  });
}

export async function render(root, ctx, [id, tParam]) {
  const st = ctx.live?.status;
  const rec = st?.stations.find((s) => s.station_id === id);
  const meta = ctx.catalog.stations[id];
  const sitesUsing = ctx.catalog.sites.filter((site) => site.references.some((r) => r.station_id === id));
  const name = rec?.name || meta?.name;
  root.append(h("div", { class: "crumbs" }, h("a", { href: "#/cameras" }, "Cameras"), " / ", id));
  if (!rec && !meta && !sitesUsing.length) {
    root.append(h("h1", null, `Station ${id}`), h("p", null, "This station is not in the camera listing or the station catalog."));
    return undefined;
  }
  const lat = rec?.lat ?? meta?.lat;
  const lon = rec?.lon ?? meta?.lon;
  root.append(
    h("h1", null, `${id} `, h("span", { class: "muted", style: { fontWeight: 400 } }, name || "")),
    h(
      "p",
      { class: "lede" },
      [
        fmtLatLon(lat, lon),
        meta?.owner,
        meta?.platform_type,
        meta?.hull ? `hull ${meta.hull}` : null,
        rec?.camera ? `camera ${rec.camera}` : null,
        meta?.cdip_id ? `CDIP station ${meta.cdip_id}` : null,
      ]
        .filter(Boolean)
        .join(" · "),
      " · ",
      ext(`https://www.ndbc.noaa.gov/station_page.php?station=${encodeURIComponent(id.toLowerCase())}`, "NDBC station page"),
      meta?.cdip_id ? [" · ", ext(`https://cdip.ucsd.edu/m/products/?stn=${meta.cdip_id}p1`, "CDIP page")] : null,
    ),
  );

  const destroyers = [];
  let stdRows = [];
  let specRows = [];
  const sync = createSync();
  let viewerApi = null;
  const charts = [];

  if (rec?.camera && st) {
    const c = summarize(rec.coverage);
    const vCard = card(
      "Camera images",
      h(
        "p",
        { class: "card-sub" },
        `${fmtInt(rec.images)} images archived (${fmtBytes(rec.bytes)}) since ${fmtUtc(parseTime(rec.first_image))}; last 7 days ${fmtPct(c.archived, c.archived + c.not_published + c.gap, 1)} of hours archived. `,
        "Only the last ~72 h can be shown here: NDBC serves the images, and the project archive is not publicly hosted.",
      ),
    );
    root.append(vCard);
    const tInit = tParam ? Number(tParam) * 1000 : null;
    viewerApi = viewer(
      rec,
      st,
      () => stdRows,
      () => specRows,
      tInit,
      (im) => {
        for (const ch of charts) ch.setCursor(im.t);
        const hash = `#/station/${id}/${Math.round(im.t / 1000)}`;
        if (location.hash !== hash) history.replaceState(null, "", hash);
      },
    );
    vCard.append(viewerApi.el);
  } else if (rec && !rec.camera) {
    root.append(h("div", { class: "notice info" }, "NDBC lists this buoy in its camera feed, but no image has been published yet."));
  }

  // Sea state.
  const seaCard = card("Sea state, recent days", h("p", { class: "card-sub" }, "Loading NDBC realtime observations…"));
  root.append(seaCard);
  const seaSub = seaCard.querySelector(".card-sub");
  try {
    const sea = await loadSeaState(ctx.live, id);
    if (!sea) {
      seaSub.textContent = "No recent realtime observations are published for this station.";
    } else {
      stdRows = blockRows(sea.products.stdmet);
      specRows = blockRows(sea.products.spec);
      viewerApi?.refreshObs();
      const gen = parseTime(sea.generated_at) || Date.now();
      const xMax = gen;
      const xMin = gen - (st?.seastate?.hours || 84) * HOUR_MS;
      seaSub.textContent =
        "NDBC realtime data (provisional, not quality-controlled). Hover to read values; click a chart to show the image nearest that time. The dark line marks the image shown above.";
      const pts = (rows, k) => rows.map((r) => ({ t: r.t, v: r[k] }));
      const pick = viewerApi ? (t) => viewerApi.select(t) : null;
      const blocks = [
        {
          title: "Wave height (m)",
          series: [
            { key: "WVHT", label: "Hs (total)", color: "var(--s1)", points: pts(stdRows, "WVHT") },
            { key: "SwH", label: "Swell", color: "var(--s2)", points: pts(specRows, "SwH") },
            { key: "WWH", label: "Wind sea", color: "var(--s3)", points: pts(specRows, "WWH") },
          ],
          fmt: (v) => fmtNum(v, 1),
        },
        {
          title: "Period (s)",
          series: [
            { key: "DPD", label: "Dominant", color: "var(--s1)", points: pts(stdRows, "DPD") },
            { key: "APD", label: "Average", color: "var(--s2)", points: pts(stdRows, "APD") },
          ],
          fmt: (v) => fmtNum(v, 0),
        },
        {
          title: "Direction, coming from (°T)",
          series: [
            { key: "MWD", label: "Waves (mean)", color: "var(--s1)", kind: "dots", points: pts(stdRows, "MWD") },
            { key: "WDIR", label: "Wind", color: "var(--s2)", kind: "dots", points: pts(stdRows, "WDIR") },
          ],
          fmt: (v) => `${fmtNum(v, 0)}°`,
          yMin: 0,
          yMax: 360,
          yTicks: [0, 90, 180, 270, 360],
        },
        {
          title: "Wind (m/s)",
          series: [
            { key: "WSPD", label: "Speed", color: "var(--s1)", points: pts(stdRows, "WSPD") },
            { key: "GST", label: "Gust", color: "var(--s2)", points: pts(stdRows, "GST") },
          ],
          fmt: (v) => fmtNum(v, 0),
        },
      ];
      const grid = h("div", { class: "grid grid-2" });
      seaCard.append(grid);
      for (const b of blocks) {
        const present = b.series.filter((sr) => sr.points.some((p) => p.v != null));
        const cell = h("div", null, h("h3", null, b.title), present.length > 1 ? legend(present.map((sr) => ({ label: sr.label, color: sr.color, kind: sr.kind === "dots" ? "dot" : "line" }))) : null);
        grid.append(cell);
        const chart = lineChart(cell, {
          series: present,
          xMin,
          xMax,
          height: 150,
          yFormat: b.fmt,
          yMin: b.yMin,
          yMax: b.yMax,
          yTicks: b.yTicks,
          cursor: viewerApi?.current()?.t ?? null,
          onPick: pick,
          sync,
          ariaLabel: `${b.title} at station ${id}`,
        });
        charts.push(chart);
        destroyers.push(chart.destroy);
      }
      seaCard.append(
        tableView(
          [
            { key: "t", label: "Time (UTC)", render: (r) => fmtUtc(r.t) },
            { key: "WVHT", label: "Hs (m)", num: true, render: (r) => fmtNum(r.WVHT, 1) },
            { key: "DPD", label: "DPD (s)", num: true, render: (r) => fmtNum(r.DPD, 0) },
            { key: "APD", label: "APD (s)", num: true, render: (r) => fmtNum(r.APD, 1) },
            { key: "MWD", label: "MWD (°)", num: true, render: (r) => fmtNum(r.MWD, 0) },
            { key: "WSPD", label: "Wind (m/s)", num: true, render: (r) => fmtNum(r.WSPD, 1) },
            { key: "WDIR", label: "Wind dir (°)", num: true, render: (r) => fmtNum(r.WDIR, 0) },
            { key: "WTMP", label: "Water (°C)", num: true, render: (r) => fmtNum(r.WTMP, 1) },
          ],
          [...stdRows].reverse(),
          `Table view (${fmtInt(stdRows.length)} rows)`,
        ),
      );
    }
  } catch (e) {
    seaSub.textContent = `Could not load sea-state data: ${e.message || e}`;
  }

  // Hourly coverage grid.
  if (rec?.coverage && st) {
    const covCard = card(
      "Hourly coverage, last 7 days",
      h("p", { class: "card-sub" }, "Each cell is one UTC hour. Click an archived hour from the last ~72 h to view its image."),
      legend([
        { label: "Day", color: "var(--il-day)" },
        { label: "Low sun", color: "var(--il-low)" },
        { label: "Twilight", color: "var(--il-twi)" },
        { label: "Night", color: "var(--il-night)" },
        { label: "Not published by NDBC", color: "var(--cell-notpub)" },
        { label: "Missed by collector", color: "var(--cell-gap)" },
        { label: "Not expected / pending", color: "var(--cell-empty)" },
      ]),
    );
    root.append(covCard);
    const hm = coverageGrid(covCard, rec, st, (t) => {
      if (viewerApi) {
        viewerApi.select(t);
        root.querySelector(".viewer")?.scrollIntoView({ behavior: "smooth", block: "center" });
      }
    });
    destroyers.push(hm.destroy);
  }

  // Archive history and references.
  const side = h("div", { class: "grid grid-2" });
  if (meta) {
    const hist = card(
      "NDBC archive history",
      h(
        "p",
        { class: "card-sub" },
        "Years with historical files per product (from the project's NDBC inventory). Realtime (45-day) products now: ",
        meta.realtime_products.length ? meta.realtime_products.join(", ") : "none",
        ".",
      ),
    );
    side.append(hist);
    const items = Object.entries(meta.history).map(([p, v]) => ({ label: PRODUCT_LABELS[p] || p, ...v }));
    destroyers.push(yearRanges(hist, items).destroy);
  }
  const facts = card("Details");
  const kv = h("dl", { class: "kv" });
  const add = (k, v) => v != null && v !== "" && kv.append(h("dt", null, k), h("dd", null, v));
  add("Position", fmtLatLon(lat, lon));
  add("Owner", meta?.owner);
  add("Program", meta?.program);
  add("Platform", [meta?.platform_type, meta?.hull].filter(Boolean).join(", "));
  add("Payload", meta?.payload);
  add("Camera code", rec?.camera);
  add("In NDBC camera listing", rec ? (rec.listed ? "yes" : "no") : null);
  add("Latest listed image", rec?.latest_listed_image);
  add("First / last archived", rec?.first_image ? `${fmtUtcShort(parseTime(rec.first_image))} / ${fmtUtcShort(parseTime(rec.last_image))}` : null);
  facts.append(kv);
  if (sitesUsing.length) {
    facts.append(
      h("h3", { style: { marginTop: "14px" } }, "Reference for camera sites"),
      dataTable(
        [
          { key: "name", label: "Site" },
          { key: "role", label: "Role", render: (r) => r.references.find((x) => x.station_id === id).role },
          { key: "km", label: "Distance", num: true, render: (r) => `${fmtNum(r.references.find((x) => x.station_id === id).distance_km, 1)} km` },
        ],
        sitesUsing,
      ),
    );
  }
  side.append(facts);
  root.append(side);

  return () => {
    for (const d of destroyers) d();
    viewerApi?.destroy();
  };
}
