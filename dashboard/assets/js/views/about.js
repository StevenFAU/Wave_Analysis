import { card, ext, h } from "../lib/dom.js";
import { fmtUtc, parseTime } from "../lib/format.js";

export const title = () => "About";

function navList(items) {
  return h(
    "ul",
    null,
    items.map((it) =>
      h("li", null, it.url ? (it.external ? ext(it.url, it.title) : h("a", { href: it.url }, it.title)) : h("strong", null, it.title), it.children ? navList(it.children) : null),
    ),
  );
}

export async function render(root, ctx) {
  const b = ctx.catalog.build;
  const repo = `https://github.com/${b.repository}`;
  const live = ctx.live;
  root.append(
    h("h1", null, "About this dashboard"),
    h(
      "p",
      { class: "lede" },
      "A window onto the Wave_Analysis research repository: what is being collected, how complete it is, which public sources can supply training data, and which papers the project's claims rest on. Every number here is generated from files in the repository or the collector's ledger; nothing is typed in by hand.",
    ),
    card(
      "How the data gets here",
      h(
        "div",
        { class: "flow" },
        h("div", null, h("strong", null, "1 · Collect"), "systemd timers on the collector host archive NDBC buoy-camera images hourly and realtime wave files twice a month. Every request goes into a ledger with its SHA-256."),
        h("div", null, h("strong", null, "2 · Summarise"), "Each hour, ", h("code", null, "wave-analysis dashboard live"), " builds status.json from the archive and ledger, and fetches recent sea state (range requests on NDBC realtime files)."),
        h("div", null, h("strong", null, "3 · Publish"), h("code", null, "scripts/publish_dashboard.sh"), " force-pushes a single commit to the ", h("code", null, "dashboard-data"), " branch. No history accumulates."),
        h("div", null, h("strong", null, "4 · Build"), "GitHub Actions builds this site from main: the front end, catalog.json (registry, bibliography, verification log, camera sites) and the MkDocs documentation."),
        h("div", null, h("strong", null, "5 · Read"), "Your browser loads the catalog from the site and the live status from the branch (5-minute CDN cache), falling back to the copy bundled at the last deploy."),
      ),
    ),
    h(
      "div",
      { class: "grid grid-2" },
      card(
        "Freshness and provenance",
        h(
          "dl",
          { class: "kv" },
          h("dt", null, "Site built"),
          h("dd", null, `${fmtUtc(parseTime(b.built_at))} from `, b.commit ? ext(`${repo}/commit/${b.commit}`, b.commit.slice(0, 7)) : "an uncommitted tree"),
          h("dt", null, "Live data"),
          h(
            "dd",
            null,
            live?.status
              ? `generated ${fmtUtc(parseTime(live.status.generated_at))} (${live.source === "live" ? "dashboard-data branch" : "bundled copy"})`
              : "unavailable",
          ),
          h("dt", null, "Status schema"),
          h("dd", null, h("code", null, live?.status?.schema || "–")),
          h("dt", null, "Catalog schema"),
          h("dd", null, h("code", null, ctx.catalog.schema)),
          h("dt", null, "Camera registry"),
          h("dd", null, b.camera_registry || "–"),
        ),
        h(
          "p",
          { class: "card-foot" },
          "The collector is judged on schedule if its last run is under 80 minutes old, late up to 3 hours, and stalled after that.",
        ),
      ),
      card(
        "Caveats",
        h(
          "ul",
          null,
          h("li", null, "Images are shown from NDBC's own server, so only the last ~72 hours are viewable. The project archive itself is not publicly hosted; citable releases will go to Zenodo (ADR 0008)."),
          h("li", null, "Sea-state values are NDBC realtime data: provisional and not quality-controlled. Research labels come from archived realtime snapshots and NDBC historical files."),
          h("li", null, "Camera-to-buoy distances use the buoy's listed position; a moored buoy moves within its watch circle."),
          h("li", null, "“Mentioned in” links match bibliography keys and first-author/year; they can over-match authors with two papers in one year."),
        ),
      ),
    ),
    card(
      "Documentation",
      h("p", { class: "card-sub" }, "The full project documentation (MkDocs), including the decisions (ADRs) behind this dashboard."),
      navList(ctx.catalog.docs.nav),
    ),
    card(
      "Source and licences",
      h(
        "p",
        null,
        "Code: ",
        ext(repo, b.repository),
        " (MIT). Buoy-camera images and observations: NOAA National Data Buoy Center, U.S. Government work in the public domain. Map tiles © OpenStreetMap contributors; bathymetry © GEBCO Compilation Group. Map library: Leaflet 1.9.4 (BSD-2-Clause), vendored.",
      ),
    ),
  );
}
