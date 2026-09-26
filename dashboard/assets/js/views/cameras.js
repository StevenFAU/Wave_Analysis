import { stripCanvas } from "../lib/charts.js";
import { decodeCell, HS_BINS, hsBin, hsBinLabel, region, REGIONS, summarize } from "../lib/coverage.js";
import { card, cssVar, dataTable, h } from "../lib/dom.js";
import { compass, DASH, fmtAgo, fmtInt, fmtNum, fmtPct, matchesQuery, parseTime } from "../lib/format.js";

export const title = () => "Cameras";
export const refreshOnLive = true;

export function cellColor(ch, illum) {
  const { state } = decodeCell(ch);
  if (state === "archived")
    return { d: "var(--il-day)", l: "var(--il-low)", t: "var(--il-twi)", n: "var(--il-night)" }[illum] || "var(--il-twi)";
  if (state === "not_published") return "var(--cell-notpub)";
  if (state === "gap") return "var(--cell-gap)";
  return "var(--cell-empty)";
}

/** Keep a Pacific-spanning set contiguous: east longitudes become < -180. */
export function unwrapLon(lon) {
  return lon > 90 ? lon - 360 : lon;
}

function shortName(name) {
  return name ? name.split(/ - | \(/)[0].trim() : "";
}

function stationRows(ctx) {
  const st = ctx.live?.status;
  if (st) return st.stations;
  // No live data: fall back to the camera registry snapshot in the catalog.
  return Object.entries(ctx.catalog.stations)
    .filter(([, m]) => m.lat != null)
    .map(([id, m]) => ({ station_id: id, name: m.name, lat: m.lat, lon: m.lon, listed: false, coverage: "", images: 0 }));
}

function hsFill(hs) {
  const b = hsBin(hs);
  return b == null ? null : cssVar(`--hs-${b}`);
}

function popupFor(nodes) {
  const div = document.createElement("div");
  div.append(...nodes);
  return div;
}

function buildMap(el, ctx, stations, onFilterChange) {
  const L = window.L;
  if (!L) {
    el.append(h("p", { class: "empty" }, "The map library did not load."));
    return null;
  }
  const map = L.map(el, { worldCopyJump: false, zoomSnap: 0.5, scrollWheelZoom: false });
  // OpenStreetMap tiles need no key; in dark mode CSS inverts them (.osm-tiles).
  // GEBCO's WMS gives bathymetry, which matters for wave transformation.
  const bases = {
    "Streets (OpenStreetMap)": L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      className: "osm-tiles",
      attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    }),
    "Bathymetry (GEBCO)": L.tileLayer.wms("https://wms.gebco.net/mapserv?", {
      layers: "GEBCO_LATEST",
      format: "image/png",
      version: "1.3.0",
      maxZoom: 12,
      attribution: 'Bathymetry: <a href="https://www.gebco.net">GEBCO Compilation Group</a>',
    }),
  };
  bases["Streets (OpenStreetMap)"].addTo(map);

  const ring = cssVar("--surface");
  const muted = cssVar("--muted");
  const buoyLayer = L.layerGroup().addTo(map);
  const siteLayer = L.layerGroup().addTo(map);
  const refLayer = L.layerGroup().addTo(map);
  const markers = new Map();
  const bounds = [];

  for (const s of stations) {
    if (s.lat == null || s.lon == null) continue;
    const ll = [s.lat, unwrapLon(s.lon)];
    bounds.push(ll);
    const fill = hsFill(s.latest_obs?.WVHT);
    const hasImages = s.images > 0;
    const m = L.circleMarker(ll, {
      radius: 7,
      color: hasImages ? ring : muted,
      weight: 2,
      fillColor: fill || muted,
      fillOpacity: hasImages ? 1 : 0,
      opacity: 1,
    });
    const o = s.latest_obs || {};
    m.bindTooltip(
      popupFor([
        h("strong", null, `${s.station_id} `),
        shortName(s.name),
        h("br"),
        o.WVHT != null ? `Hs ${fmtNum(o.WVHT, 1)} m · Tp ${fmtNum(o.DPD, 0)} s · from ${compass(o.MWD)}` : "no recent wave observation",
        h("br"),
        hasImages ? `${fmtInt(s.images)} images archived` : "listed without an image",
      ]),
      { direction: "top", offset: [0, -6] },
    );
    m.on("click", () => ctx.navigate(`station/${s.station_id}`));
    m.addTo(buoyLayer);
    markers.set(s.station_id, m);
  }

  const refColor = cssVar("--s3");
  const seenRefs = new Set();
  for (const site of ctx.catalog.sites) {
    const sll = [site.latitude, unwrapLon(site.longitude)];
    bounds.push(sll);
    const icon = L.divIcon({ className: "", html: '<div class="site-marker"></div>', iconSize: [14, 14], iconAnchor: [7, 7] });
    const refsList = site.references.map((r) =>
      h(
        "div",
        null,
        `${r.role}: `,
        r.network === "USGS" ? r.name : h("a", { href: `#/station/${r.station_id}` }, `${r.network} ${r.cdip_id ? `${r.cdip_id} / ` : ""}${r.station_id}`),
        ` · ${fmtNum(r.distance_km, 1)} km`,
      ),
    );
    L.marker(sll, { icon, keyboard: true, title: site.name })
      .bindPopup(
        popupFor([
          h("strong", null, site.name),
          h("div", { class: "muted" }, `${site.provider} · ${site.period}`),
          ...refsList,
          site.approximate ? h("div", { class: "muted" }, "Position approximate.") : null,
          site.dataset_id ? h("div", null, h("a", { href: `#/sources/${site.dataset_id}` }, "Dataset entry →")) : null,
        ].filter(Boolean)),
      )
      .addTo(siteLayer);
    for (const r of site.references) {
      const rll = [r.latitude, unwrapLon(r.longitude)];
      L.polyline([sll, rll], { color: refColor, weight: 1.5, opacity: 0.8, interactive: false }).addTo(refLayer);
      if (seenRefs.has(r.station_id) || markers.has(r.station_id)) continue;
      seenRefs.add(r.station_id);
      const rm = L.circleMarker(rll, { radius: 5, color: ring, weight: 2, fillColor: refColor, fillOpacity: 1 });
      rm.bindTooltip(
        popupFor([h("strong", null, `${r.network} ${r.cdip_id ? `${r.cdip_id} (${r.station_id})` : r.station_id}`), h("br"), r.name || ""]),
        { direction: "top" },
      );
      if (r.network !== "USGS") rm.on("click", () => ctx.navigate(`station/${r.station_id}`));
      rm.addTo(refLayer);
    }
  }

  L.control
    .layers(bases, { "NDBC buoy cameras": buoyLayer, "Shore camera sites": siteLayer, "Reference buoys": refLayer }, { collapsed: true, position: "topleft" })
    .addTo(map);
  const legendCtl = L.control({ position: "topright" });
  legendCtl.onAdd = () => {
    const div = h("div", { class: "map-legend" }, h("strong", null, "Hs now (buoy cameras)"));
    for (let i = 0; i <= HS_BINS.length; i += 1)
      div.append(
        h("div", { class: "row" }, h("span", { class: "swatch dot", style: { background: cssVar(`--hs-${i}`) } }), hsBinLabel(i)),
      );
    div.append(
      h("div", { class: "row" }, h("span", { class: "swatch dot", style: { background: muted } }), "no recent observation"),
      h("div", { class: "row" }, h("span", { class: "swatch ring" }), "listed, no image"),
      h("div", { class: "row" }, h("span", { class: "swatch square", style: { background: cssVar("--s2") } }), "shore camera site"),
      h("div", { class: "row" }, h("span", { class: "swatch dot", style: { background: refColor } }), "reference buoy"),
    );
    L.DomEvent.disableClickPropagation(div);
    return div;
  };
  legendCtl.addTo(map);
  if (bounds.length) map.fitBounds(bounds, { padding: [24, 24] });
  else map.setView([30, -100], 3);
  el.addEventListener("click", () => map.scrollWheelZoom.enable(), { once: true });

  onFilterChange((visible) => {
    for (const [id, m] of markers) {
      if (visible.has(id)) m.addTo(buoyLayer);
      else buoyLayer.removeLayer(m);
    }
  });
  return map;
}

export async function render(root, ctx) {
  const stations = stationRows(ctx);
  const st = ctx.live?.status;
  root.append(
    h("h1", null, "Cameras and wave references"),
    h(
      "p",
      { class: "lede" },
      "NDBC buoy cameras (colour: significant wave height now, from the same buoy), public shore-camera sites (diamonds), and the wave buoys proposed as their references. Click a buoy for its images and sea state. Click the map to enable scroll-zoom.",
    ),
  );
  if (!st) root.append(h("div", { class: "notice" }, "Live data unavailable: positions from the camera registry, without coverage or wave heights."));

  const q = h("input", { type: "search", placeholder: "Filter stations (id or name)", "aria-label": "Filter stations" });
  const reg = h(
    "select",
    { "aria-label": "Region" },
    h("option", { value: "" }, "All regions"),
    Object.entries(REGIONS).map(([k, v]) => h("option", { value: k }, v)),
  );
  const onlyImages = h("input", { type: "checkbox", checked: true });
  root.append(h("div", { class: "filters" }, q, h("label", null, "Region ", reg), h("label", null, onlyImages, "Only cameras with archived images")));

  const mapEl = h("div", { class: "map", role: "region", "aria-label": "Map of camera stations" });
  root.append(h("section", null, mapEl));
  const listeners = [];
  const map = buildMap(mapEl, ctx, stations, (fn) => listeners.push(fn));

  const tableHost = h("div");
  const tableCard = card("Buoy cameras", h("p", { class: "card-sub" }, "7-day coverage strip: one pixel per hour, coloured by illumination; grey = never published by NDBC; red = missed by the collector."), tableHost);
  root.append(tableCard);

  const columns = [
    { key: "station_id", label: "Station", render: (r) => h("a", { href: `#/station/${r.station_id}` }, r.station_id) },
    { key: "name", label: "Name", render: (r) => shortName(r.name) },
    { key: "hs", label: "Hs (m)", num: true, sort: (r) => r.latest_obs?.WVHT, render: (r) => fmtNum(r.latest_obs?.WVHT, 1) },
    { key: "tp", label: "Tp (s)", num: true, sort: (r) => r.latest_obs?.DPD, render: (r) => fmtNum(r.latest_obs?.DPD, 0) },
    { key: "mwd", label: "From", sort: (r) => r.latest_obs?.MWD, render: (r) => (r.latest_obs?.MWD != null ? compass(r.latest_obs.MWD) : DASH) },
    { key: "wind", label: "Wind (m/s)", num: true, sort: (r) => r.latest_obs?.WSPD, render: (r) => fmtNum(r.latest_obs?.WSPD, 0) },
    {
      key: "coverage",
      label: "Last 7 days",
      sort: (r) => (r.coverage ? summarize(r.coverage).archived : -1),
      render: (r) => (r.coverage ? stripCanvas(r.coverage.length, (i) => cellColor(r.coverage[i], (r.illumination || "")[i])) : DASH),
    },
    {
      key: "pct",
      label: "Archived",
      num: true,
      sort: (r) => {
        const c = summarize(r.coverage || "");
        const e = c.archived + c.not_published + c.gap;
        return e ? c.archived / e : null;
      },
      render: (r) => {
        const c = summarize(r.coverage || "");
        return fmtPct(c.archived, c.archived + c.not_published + c.gap, 0);
      },
    },
    { key: "images", label: "Images", num: true, render: (r) => fmtInt(r.images) },
    { key: "last_image", label: "Last image", sort: (r) => r.last_image, render: (r) => (r.last_image ? fmtAgo(parseTime(r.last_image)) : DASH) },
  ];

  function apply() {
    const visible = stations.filter(
      (s) =>
        matchesQuery(`${s.station_id} ${s.name || ""}`, q.value) &&
        (!reg.value || region(s.station_id) === reg.value) &&
        (!onlyImages.checked || s.images > 0),
    );
    tableHost.replaceChildren(
      dataTable(columns, visible, {
        onRow: (r) => ctx.navigate(`station/${r.station_id}`),
        initialSort: { key: "station_id", dir: 1 },
        caption: "Buoy cameras",
      }),
      h("p", { class: "card-foot" }, `${visible.length} of ${stations.length} stations shown.`),
    );
    const ids = new Set(visible.map((s) => s.station_id));
    for (const fn of listeners) fn(ids);
  }
  q.addEventListener("input", apply);
  reg.addEventListener("change", apply);
  onlyImages.addEventListener("change", apply);
  apply();

  // Shore-camera sites and their references.
  const siteRows = ctx.catalog.sites.map((s) => ({
    ...s,
    primary: s.references.find((r) => r.role === "primary"),
    offshore: s.references.find((r) => r.role === "offshore"),
  }));
  const refCell = (r) =>
    r
      ? h(
          "span",
          null,
          r.network === "USGS" ? r.name : h("a", { href: `#/station/${r.station_id}` }, `${r.network} ${r.cdip_id ? `${r.cdip_id} (${r.station_id})` : r.station_id}`),
          h("span", { class: "muted" }, ` · ${fmtNum(r.distance_km, 1)} km`),
        )
      : DASH;
  root.append(
    card(
      "Shore camera sites",
      h(
        "p",
        { class: "card-sub" },
        "Distances are great-circle distances from the camera to the buoy's listed position. Distance alone does not make a buoy representative: depth, exposure and sheltering matter, and offshore labels need a transformation model for surf-zone imagery.",
      ),
      dataTable(
        [
          { key: "name", label: "Site", render: (r) => h("span", null, r.name, r.approximate ? h("span", { class: "muted" }, " (position approx.)") : null) },
          { key: "provider", label: "Provider" },
          { key: "period", label: "Imagery period" },
          { key: "primary", label: "Primary reference", sort: (r) => r.primary?.distance_km, render: (r) => refCell(r.primary) },
          { key: "offshore", label: "Offshore buoy", sort: (r) => r.offshore?.distance_km, render: (r) => refCell(r.offshore) },
          { key: "dataset_id", label: "Dataset", render: (r) => (r.dataset_id ? h("a", { href: `#/sources/${r.dataset_id}` }, r.dataset_id) : DASH) },
        ],
        siteRows,
        {
          onRow: (r) => {
            if (map) {
              map.flyTo([r.latitude, unwrapLon(r.longitude)], 10);
              mapEl.scrollIntoView({ behavior: "smooth", block: "center" });
            }
          },
        },
      ),
    ),
  );

  return () => {
    if (map) map.remove();
  };
}
