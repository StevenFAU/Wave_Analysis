// Data inventory: what every collection is for and what its variables mean
// (catalog.dictionary, from data/registry/data_dictionary.yaml), with live
// counts from status.json. The long-form version, with sources, is
// docs/datasets/inventory.md.
import { checkStatus, collectionNotes, countLabel, freshness, MODES, periodLabel, totals } from "../lib/collections.js";
import { card, dataTable, ext, h, statusLabel, tile } from "../lib/dom.js";
import { DASH, fmtAgo, fmtBytes, fmtInt, parseTime } from "../lib/format.js";
import { countWord, groupCollections, joinCollections, parseMarkup, rolesUsed, variableColumns } from "../lib/inventory.js";

export const title = () => "Data inventory";
export const refreshOnLive = true;

const DOC_PAGE = "docs/datasets/inventory/";

/** Dictionary text -> nodes (`code`, _{sub}); never parsed as HTML. */
function rich(text) {
  return parseMarkup(text).map((r) => (r.kind === "code" ? h("code", null, r.text) : r.kind === "sub" ? h("sub", null, r.text) : r.text));
}

function roleChip(role) {
  return h("span", { class: ["role-chip", `role-${role.id}`], title: role.short }, role.label);
}

function chips(roleIds, roles) {
  return h("span", { class: "role-chips" }, (roleIds || []).map((id) => (roles.has(id) ? roleChip(roles.get(id)) : null)));
}

const anchorId = (key) => `inv-${key}`;

/** In-page link that scrolls and keeps the hash shareable without a re-render. */
function jump(key, text) {
  return h(
    "a",
    {
      href: `#/data/${key}`,
      onclick: (e) => {
        const el = document.getElementById(anchorId(key));
        if (!el) return;
        e.preventDefault();
        history.replaceState(null, "", `#/data/${key}`);
        el.scrollIntoView({ block: "start" });
        el.focus({ preventScroll: true });
      },
    },
    text,
  );
}

function section(key, eyebrow, heading, ...children) {
  return h(
    "section",
    { class: "inv-section", id: anchorId(key), tabindex: -1, "aria-labelledby": `${anchorId(key)}-h` },
    h("div", { class: "inv-head" }, eyebrow ? h("div", { class: "inv-eyebrow" }, eyebrow) : null, h("h2", { id: `${anchorId(key)}-h` }, heading)),
    ...children,
  );
}

const nowrap = (text) => h("span", { class: "nowrap" }, text);

// --------------------------------------------------------------------- pieces

function rolesSection(dict) {
  return section(
    "roles",
    "Categories",
    "The six roles data plays",
    h("p", { class: "inv-prose" }, "Every collection does one or more of these jobs. The chips mark the job throughout the page."),
    h(
      "div",
      { class: "inv-roles" },
      dict.roles.map((r) => h("div", { class: "card inv-role" }, roleChip(r), h("p", null, h("strong", null, `${r.short}. `), rich(r.description)))),
    ),
  );
}

function liveCount(live) {
  return live ? nowrap(countLabel(live.count, live.unit)) : h("span", { class: "muted" }, "not on host");
}

function overviewTable(dict, joined, roles, live) {
  const head = ["Collection", "Role", "Where", "Data period (UTC)", "How often", "Held", "Size"];
  const rows = joined.map(({ id, dict: d, live: c }) =>
    h(
      "tr",
      null,
      h("td", null, d ? jump(id, h("strong", null, d.name)) : h("strong", null, c.name), h("div", { class: "small muted" }, d ? rich(d.subtitle) : c.provider)),
      h("td", null, d ? chips(d.roles, roles) : DASH),
      h("td", null, d ? rich(d.where) : DASH),
      h("td", null, c ? periodLabel(c.first, c.last) : DASH),
      h("td", null, d ? d.cadence : MODES[c?.mode] || DASH),
      h("td", { class: "num" }, liveCount(c)),
      h("td", { class: "num" }, c ? nowrap(fmtBytes(c.bytes)) : DASH),
    ),
  );
  const tableRoles = [...new Set(dict.tables.flatMap((t) => t.roles))];
  rows.push(
    h(
      "tr",
      null,
      h("td", null, jump("tables", h("strong", null, "Tables we built")), h("div", { class: "small muted" }, "Pairing, audits, benchmark results, ledgers, registries")),
      h("td", null, chips(tableRoles, roles)),
      h("td", { colspan: 4 }, `${fmtInt(dict.tables.length)} tables and record sets, built from the collections above`),
      h("td", { class: "num" }, DASH),
    ),
  );
  return h(
    "div",
    { class: "table-wrap" },
    h(
      "table",
      { class: "inv-overview" },
      h("caption", { class: "sr-only" }, "All data collections"),
      h("thead", null, h("tr", null, head.map((t, i) => h("th", { scope: "col", class: i >= 5 ? "num" : null }, t)))),
      h("tbody", null, rows),
    ),
  );
}

function offsiteNotice(st) {
  const t = parseTime(st?.offsite?.last_sync);
  const cam = (st?.collections || []).find((c) => c.id === "ndbc_buoycam");
  const lead = h("strong", null, "One collection cannot be downloaded again. ");
  const body = "NDBC deletes buoy-camera images after about 72 hours, so the project's copy is the only one.";
  if (!st) return h("p", { class: "notice" }, lead, body);
  if (t && Date.now() - t < 30 * 3600 * 1000)
    return h("p", { class: "notice info" }, lead, body, ` Two hosts collect it, and the last offsite copy ran ${fmtAgo(t)}.`);
  return h(
    "p",
    { class: "notice critical" },
    lead,
    body,
    t ? ` The last offsite copy ran ${fmtAgo(t)}, more than 30 hours ago.` : " No offsite copy is recorded.",
    cam ? ` ${fmtInt(cam.count)} images are at risk.` : "",
  );
}

function liveStrip(c) {
  if (!c) return h("p", { class: "inv-live muted" }, "No live totals: the collection is not on the collector host, or the live data could not be loaded.");
  const f = freshness(c);
  const chk = checkStatus(c);
  const item = (label, value, title) => h("div", { title }, h("span", { class: "inv-live-k" }, label), h("span", { class: "inv-live-v" }, value));
  return h(
    "div",
    { class: "inv-live", "aria-label": "Live totals" },
    item("Held", countLabel(c.count, c.unit)),
    item("Size", fmtBytes(c.bytes)),
    item("Data period (UTC)", periodLabel(c.first, c.last)),
    item("Collecting", f ? statusLabel(f.level, f.label) : MODES[c.mode] || c.mode, c.updated ? `Last ledger activity ${c.updated}` : null),
    item("Files vs ledger", chk.level ? statusLabel(chk.level, chk.label) : chk.label, chk.detail),
  );
}

function facts(list) {
  if (!list?.length) return null;
  return h(
    "dl",
    { class: "inv-facts" },
    list.map((f) => h("div", null, h("dt", null, f.label), h("dd", null, rich(f.value)))),
  );
}

function variablesTable(vars, roles, caption) {
  if (!vars?.length) return null;
  const cols = variableColumns(vars);
  const nCols = 2 + cols.unit + cols.role + cols.source;
  const body = [];
  for (const v of vars) {
    if (v.group) body.push(h("tr", { class: "inv-group" }, h("td", { colspan: nCols }, rich(v.group))));
    body.push(
      h(
        "tr",
        null,
        h("td", { class: "inv-var" }, v.name.split(", ").map((n) => h("div", null, n))),
        h(
          "td",
          null,
          rich(v.meaning),
          v.canonical ? h("div", { class: "inv-canon", title: "Name in the project's standardized tables (schemas/variables.py)" }, "→ ", h("code", null, v.canonical)) : null,
        ),
        cols.unit ? h("td", { class: "inv-unit" }, v.unit || "") : null,
        cols.role ? h("td", null, v.role && roles.has(v.role) ? roleChip(roles.get(v.role)) : "") : null,
        cols.source ? h("td", { class: "inv-unit" }, v.source || "") : null,
      ),
    );
  }
  const heads = ["Variable", "Meaning", cols.unit && "Unit", cols.role && "Role", cols.source && "Source"].filter(Boolean);
  return h(
    "div",
    { class: "table-wrap" },
    h(
      "table",
      { class: "inv-vars" },
      h("caption", { class: "sr-only" }, caption),
      h("thead", null, h("tr", null, heads.map((t) => h("th", { scope: "col" }, t)))),
      h("tbody", null, body),
    ),
  );
}

function partsTable(c) {
  const parts = c?.parts || [];
  if (parts.length < 2) return null;
  const label = { external: "dataset", ndbc_history: "product", cdip: "station", ndbc_realtime: "station" }[c.id] ||
    { images: "camera", "site-months": "site" }[c.unit] || "part";
  return h(
    "details",
    { class: "table-view" },
    h("summary", null, `Per ${label} (${fmtInt(parts.length)})`),
    dataTable(
      [
        { key: "id", label: label[0].toUpperCase() + label.slice(1), render: (p) => h("code", null, p.id) },
        { key: "count", label: "Held", num: true, render: (p) => fmtInt(p.count) },
        { key: "bytes", label: "Size", num: true, render: (p) => nowrap(fmtBytes(p.bytes)) },
        { key: "first", label: "Data period (UTC)", render: (p) => periodLabel(p.first, p.last) },
      ],
      parts,
    ),
  );
}

function collectionBlock({ id, dict: d, live: c }, roles) {
  if (!d) {
    return h(
      "article",
      { class: "card inv-coll", id: anchorId(id), tabindex: -1 },
      h("h3", null, c.name),
      h("p", { class: "muted" }, "This collection is on the collector host but has no entry in the data dictionary yet."),
      liveStrip(c),
    );
  }
  const notes = [...d.notes.map((n) => rich(n)), ...(c ? collectionNotes(c) : [])];
  return h(
    "article",
    { class: "card inv-coll", id: anchorId(id), tabindex: -1, "aria-labelledby": `${anchorId(id)}-h` },
    h(
      "div",
      { class: "inv-coll-head" },
      h("h3", { id: `${anchorId(id)}-h` }, d.name),
      chips(d.roles, roles),
    ),
    h(
      "p",
      { class: "inv-coll-sub" },
      d.provider,
      " · ",
      h("a", { href: d.doc_url }, "datasheet"),
      d.datasets.map((ds) => [" · ", h("a", { href: `#/sources/${ds}` }, d.datasets.length > 1 ? ds : "registry entry")]),
    ),
    liveStrip(c),
    facts(d.facts),
    variablesTable(d.variables, roles, `Variables of ${d.name}`),
    notes.length ? h("ul", { class: "inv-notes" }, notes.map((n) => h("li", null, n))) : null,
    partsTable(c),
  );
}

function groupSection(g, roles, dict) {
  const used = rolesUsed(
    dict,
    g.items.map((r) => r.dict),
  );
  return section(
    g.id,
    h("span", { class: "role-chips" }, used.map(roleChip)),
    g.title,
    h("p", { class: "inv-prose" }, rich(g.intro)),
    h("div", { class: "stack" }, g.items.map((r) => collectionBlock(r, roles))),
  );
}

function tablesSection(dict, roles) {
  return section(
    "tables",
    h("span", { class: "role-chips" }, rolesUsed(dict, dict.tables).map(roleChip)),
    "Tables we built",
    h(
      "p",
      { class: "inv-prose" },
      "Derived from the collections above by scripts in the repository. Their provenance (inputs, checksums, code version) is recorded next to them; summaries are committed under ",
      h("code", null, "data/manifests/processed/"),
      ".",
    ),
    h(
      "div",
      { class: "table-wrap" },
      h(
        "table",
        { class: "inv-vars" },
        h("caption", { class: "sr-only" }, "Tables built by the project"),
        h("thead", null, h("tr", null, ["Table", "What it holds", "Size", "Where"].map((t) => h("th", { scope: "col" }, t)))),
        h(
          "tbody",
          null,
          dict.tables.map((t) =>
            h(
              "tr",
              null,
              h("td", null, h("strong", null, t.name), h("div", null, chips(t.roles, roles))),
              h("td", null, rich(t.holds), " ", h("a", { href: t.doc_url, class: "small" }, "More")),
              h("td", { class: "inv-unit" }, t.shape),
              h("td", null, h("code", { class: "inv-path" }, t.path)),
            ),
          ),
        ),
      ),
    ),
  );
}

function crosswalkSection(dict) {
  const cw = dict.crosswalk;
  const cell = (c) => {
    if (!c || (!c.vars.length && !c.note)) return h("span", { class: "muted" }, DASH);
    return h(
      "span",
      null,
      c.vars.map((v, i) => [i ? " " : null, h("code", null, v)]),
      c.note ? h("span", { class: "muted small" }, c.vars.length ? ` (${c.note})` : c.note) : null,
    );
  };
  return section(
    "crosswalk",
    "Cross-reference",
    "Same quantity, different names",
    h("p", { class: "inv-prose" }, rich(cw.intro)),
    h(
      "div",
      { class: "table-wrap" },
      h(
        "table",
        { class: "inv-cross" },
        h("caption", { class: "sr-only" }, "Variable names by source"),
        h("thead", null, h("tr", null, h("th", { scope: "col" }, "Quantity"), cw.columns.map((c) => h("th", { scope: "col" }, c.label)))),
        h(
          "tbody",
          null,
          cw.rows.map((r) =>
            h(
              "tr",
              null,
              h("th", { scope: "row" }, rich(r.quantity), r.note ? h("div", { class: "small muted" }, rich(r.note)) : null),
              cw.columns.map((c) => h("td", null, cell(r.cells[c.id]))),
            ),
          ),
        ),
      ),
    ),
  );
}

function stage(cls, heading, where, items, ordered = true) {
  return h(
    "div",
    { class: ["inv-stage", cls] },
    h("h3", null, heading),
    h("div", { class: "inv-where" }, where),
    Array.isArray(items) ? h(ordered ? "ol" : "ul", null, items.map((x) => h("li", null, x))) : h("p", null, items),
  );
}

function processingSection() {
  const arrow = () => h("div", { class: "inv-arrow", "aria-hidden": "true" }, "→");
  return section(
    "buoy-processing",
    "NDBC buoys",
    "Where buoy wave numbers are made",
    h(
      "p",
      { class: "inv-prose" },
      "Both places do part of the work. The buoy turns its motion into a spectrum and sends only that. The shore unpacks it, checks it, computes the wave numbers and publishes them. The motion record itself stays on the buoy.",
    ),
    h(
      "div",
      { class: "inv-pipe", role: "img", "aria-label": "Pipeline: sensors and wave processor on the buoy, satellite link, processing at NDBC on shore, public files" },
      stage("on-buoy", "On the buoy", "sensors + wave processor", [
        "Records its own motion for 20 min (40 on one system): vertical acceleration, pitch, roll, heading. 1.5 to 2.56 readings a second, by system.",
        "Turns pitch, roll and heading into east-west and north-south slopes.",
        "Fourier transform: motion over time becomes energy per frequency band.",
        "Averages into bands and computes the direction values (α₁, α₂, r₁, r₂); newer systems also correct for the hull and sensors on board.",
        "Packs it into one short message, with summary statistics of pitch, roll and heading for quality checks.",
      ]),
      arrow(),
      stage("link", "Satellite", "short coded message", "The message is too short for the raw motion record, which is why it is never sent."),
      arrow(),
      stage("shore", "On shore", "NDBC, Stennis Space Center, Mississippi", [
        "Decodes and checks the message: complete, values in range, hull behaving.",
        "Unpacks it into a wave-height spectrum.",
        "Computes WVHT, DPD, APD, MWD, the swell and wind-wave split, steepness.",
        "Further automated and manual quality control.",
      ]),
      arrow(),
      stage("files", "Public files", "what the project downloads", [h("code", null, ".txt"), h("code", null, ".spec"), "five spectral files", "45-day realtime + yearly archive"], false),
    ),
    h(
      "p",
      { class: "notice inv-dropped" },
      h("strong", null, "Never sent to shore: "),
      "the moment-by-moment motion record. NDBC's FAQ says the raw acceleration or displacement measurements are “not transmitted shore-side.”",
    ),
    h(
      "div",
      { class: "grid grid-2" },
      card(
        "Raw vs processed",
        h(
          "p",
          null,
          h("strong", null, "Raw"),
          " is every reading in time order: a 20-minute record at 1.7066 readings a second is 2,048 readings each of acceleration, pitch, roll and heading. You can look up what the buoy was doing at any second.",
        ),
        h(
          "p",
          null,
          h("strong", null, "Processed"),
          " is a summary of the whole record: energy and direction in 46 frequency bands (46 × 5 = 230 numbers) plus WVHT, DPD, APD, MWD. It says how much energy there was at each wave period over those 20 minutes, but no longer ",
          h("em", null, "when"),
          " anything happened.",
        ),
        h(
          "pre",
          { class: "bib", "aria-label": "Example processed record" },
          "WVHT 0.9 m   DPD 9 s   APD 5.3 s   MWD 51°\n0.078 Hz  S 0.342 m²/Hz\n0.083 Hz  S 0.488 m²/Hz\n…  43 more bands",
        ),
        h("p", { class: "card-foot" }, "Real values: buoy 41002, record of 1 Oct 2026 11:50 UTC. Like a song's equalizer display against the song itself: the display cannot tell you which note played at 1:23."),
      ),
      card(
        "What this means for pairing with a camera",
        h(
          "p",
          null,
          statusLabel("good", "Processed is enough"),
          " for ",
          h("em", null, "how big were the waves around this photo?"),
          " Wave height, period and direction are statistics of 20–27 minutes of sea anyway. Each photo gets the buoy records nearest to it; the Waimea labels average the records within ±30 min. What limits this is time and place, not rawness.",
        ),
        h(
          "p",
          null,
          statusLabel("critical", "Needs raw data"),
          " for ",
          h("em", null, "how was the camera tilted at the instant of this photo?"),
          " A buoy camera rides on the hull: a 10° roll tilts the horizon 10°. NDBC never sends the motion at the shutter's second. Ways around it: measure the tilt from the horizon in each view (Q-M5); the Schwendeman & Thomson ship stereo record, which logs the camera's own motion; or a camera and motion sensor on one clock (gap G-9 in the collection plan).",
        ),
      ),
    ),
    h(
      "p",
      { class: "small muted" },
      "Long form, with how the on-board share grew across NDBC's wave systems and the sources: ",
      h("a", { href: `${DOC_PAGE}#where-buoy-wave-numbers-are-made` }, "Data inventory in the documentation"),
      ".",
    ),
  );
}

// ---------------------------------------------------------------------- view

export async function render(root, ctx, [target]) {
  const dict = ctx.catalog.dictionary;
  const st = ctx.live?.status || null;
  const roles = new Map(dict.roles.map((r) => [r.id, r]));
  const joined = joinCollections(dict, st?.collections);
  const groups = groupCollections(dict, joined);
  const liveCols = st?.collections || [];
  const tot = totals(liveCols);
  const pictures = joined.filter((r) => r.dict?.group === "pictures").length;
  const measures = joined.filter((r) => r.dict?.group === "measurements").length;

  const toc = [
    ["roles", "The six roles"],
    ["collections", "All collections"],
    ...groups.map((g) => [g.id, g.title]),
    ["tables", "Tables we built"],
    ["crosswalk", "Same quantity, different names"],
    ["buoy-processing", "Where buoy numbers are made"],
  ];

  root.append(
    h(
      "header",
      { class: "inv-intro" },
      h("div", { class: "inv-eyebrow" }, st ? `Counted from disk ${fmtAgo(parseTime(st.generated_at))} · descriptions checked ${dict.checked}` : `Descriptions checked ${dict.checked}`),
      h("h1", null, "Data inventory"),
      h(
        "p",
        { class: "lede" },
        st
          ? `The project holds ${fmtInt(tot.collections)} collections, ${fmtBytes(tot.bytes)} in all: ${countWord(pictures)} sets of sea pictures, ${countWord(measures)} sets of wave and weather measurements, a wave model, and published research datasets. On top of them sit the tables the project built. This page lists every collection and every variable in it, and explains where buoy wave numbers come from.`
          : "Every collection the project holds and every variable in it, and where buoy wave numbers come from. Live counts are unavailable right now.",
      ),
      st
        ? h(
            "div",
            { class: "tiles" },
            tile("Images held", fmtInt(tot.images), `in ${fmtInt(tot.imageCollections)} image collections`),
            tile("Total size", fmtBytes(tot.bytes), `${fmtInt(tot.collections)} collections`),
            tile("Variables described", fmtInt(dict.collections.reduce((n, c) => n + c.variables.length, 0)), `${fmtInt(dict.crosswalk.rows.length)} quantities cross-referenced`),
          )
        : null,
      h("nav", { class: "inv-toc", "aria-label": "On this page" }, toc.map(([k, t]) => jump(k, t))),
    ),
    rolesSection(dict),
    section("collections", "At a glance", "All collections", overviewTable(dict, joined, roles, st), offsiteNotice(st)),
    ...groups.map((g) => groupSection(g, roles, dict)),
    tablesSection(dict, roles),
    crosswalkSection(dict),
    processingSection(),
    h(
      "footer",
      { class: "inv-sources small muted" },
      h("p", null, "Sources: counts, sizes and periods from the files on the collector host, checked against each collection's ledger. Descriptions from ", ext(`https://github.com/${ctx.catalog.build.repository}/blob/main/data/registry/data_dictionary.yaml`, "data_dictionary.yaml"), " (checked on the files ", dict.checked, ") and each collection's datasheet. NDBC processing: Earle (2003), NDBC Technical Document 03-01, and the NDBC wave FAQ."),
    ),
  );

  if (target) {
    // After the router's scroll-to-top.
    requestAnimationFrame(() => {
      const el = document.getElementById(anchorId(target));
      if (el) {
        el.scrollIntoView({ block: "start" });
        el.focus({ preventScroll: true });
      }
    });
  }
}
