import { columnChart } from "../lib/charts.js";
import { card, copyButton, ext, h, tableView } from "../lib/dom.js";
import { fmtInt, matchesQuery } from "../lib/format.js";

export const title = ([key]) => (key ? `Reference ${key}` : "Literature");

const LEVELS = {
  V1: "Checked against the primary full text.",
  V2: "Checked against the publisher abstract, Crossref/Semantic Scholar metadata, or the dataset record.",
  V3: "Taken from project notes; must be re-checked before being quoted.",
  "V1/V2": "Read for the landscape review; the log does not record which of V1 or V2.",
  "V2/V3": "Abstract and metadata checked (V2); details from project notes (V3).",
};

const levelClass = (lv) => (lv ? `lv-${lv.split("/")[0]}` : null);

function docLink(ctx, path) {
  const title = ctx.catalog.docs.titles[path] || path;
  // docs/<path>.md -> docs/<path>/ (MkDocs directory URLs; README/index -> directory)
  const rel = path.replace(/^docs\//, "").replace(/\.md$/, "");
  const url = /(^|\/)(README|index)$/.test(rel) ? `docs/${rel.replace(/(^|\/)(README|index)$/, "$1")}` : `docs/${rel}/`;
  return h("a", { href: url }, title);
}

function citeLine(r) {
  const parts = [r.venue, r.volume ? `${r.volume}` : null, r.pages].filter(Boolean);
  return parts.join(", ");
}

function detail(root, ctx, r) {
  root.append(
    h("div", { class: "crumbs" }, h("a", { href: "#/literature" }, "Literature"), " / ", r.key),
    h("h1", null, r.title),
    h("p", { class: "lede" }, r.authors.join("; "), r.year ? ` (${r.year})` : "", ". ", citeLine(r)),
  );
  const links = [
    r.doi ? ext(`https://doi.org/${r.doi}`, `doi:${r.doi}`) : null,
    r.arxiv ? ext(`https://arxiv.org/abs/${r.arxiv}`, `arXiv:${r.arxiv}`) : null,
    !r.doi && !r.arxiv && r.url ? ext(r.url) : null,
  ].filter(Boolean);
  const v = r.verification;
  root.append(
    h(
      "div",
      { class: "grid grid-2" },
      card(
        "Verification",
        v
          ? [
              h("p", null, h("span", { class: ["badge", levelClass(v.level)] }, v.level), " ", LEVELS[v.level] || ""),
              h("p", { class: "small muted" }, `Basis: ${v.basis}`),
            ]
          : h("p", { class: "muted" }, "No check logged for this entry yet (textbooks, standards and architecture papers are cited for background)."),
        h("p", { class: "card-foot" }, "Log: ", docLink(ctx, "docs/literature/source_verification.md")),
      ),
      card(
        "Links and use in the project",
        links.length ? h("p", null, links.flatMap((l, i) => (i ? [" · ", l] : [l]))) : null,
        r.notes ? h("p", null, "Paper notes: ", docLink(ctx, r.notes)) : null,
        r.mentions.length
          ? [h("h3", null, `Mentioned in ${r.mentions.length} document${r.mentions.length > 1 ? "s" : ""}`), h("ul", null, r.mentions.map((p) => h("li", null, docLink(ctx, p))))]
          : h("p", { class: "muted small" }, "Not mentioned by key or author-year in the docs."),
        h("p", { class: "card-foot" }, `Bibliography section: ${r.section || "–"}`),
      ),
    ),
    card("BibTeX", h("pre", { class: "bib" }, r.bibtex), h("p", { style: { marginTop: "8px" } }, copyButton(r.bibtex, "Copy BibTeX"))),
  );
}

export async function render(root, ctx, [key], params) {
  const refs = ctx.catalog.references;
  if (key) {
    const r = refs.find((x) => x.key === key);
    if (!r) {
      root.append(h("h1", null, "Unknown reference"), h("p", null, h("a", { href: "#/literature" }, "All references")));
      return undefined;
    }
    detail(root, ctx, r);
    return undefined;
  }
  root.append(
    h("h1", null, "Literature"),
    h(
      "p",
      { class: "lede" },
      `${refs.length} references from the project bibliography, with the verification level recorded for each claim the project relies on. Screened items (titles seen, not yet read) are listed separately so they are never mistaken for evidence.`,
    ),
  );
  const q = h("input", { type: "search", placeholder: "Search title, author, key, venue", "aria-label": "Search references", value: params.get("q") || "" });
  const sections = [...new Set(refs.map((r) => r.section).filter(Boolean))];
  const sec = h("select", { "aria-label": "Section" }, h("option", { value: "" }, "All sections"), sections.map((s) => h("option", { value: s }, s)));
  const levels = ["V1", "V1/V2", "V2", "V2/V3", "none"];
  const lvl = h(
    "select",
    { "aria-label": "Verification level" },
    h("option", { value: "" }, "Any verification"),
    levels.map((l) => h("option", { value: l }, l === "none" ? "not logged" : l)),
  );
  root.append(h("div", { class: "filters" }, q, sec, lvl));

  const years = refs.map((r) => r.year).filter(Boolean);
  const yearCard = card("References by publication year", h("p", { class: "card-sub" }, "All bibliography entries."));
  const lo = Math.min(...years);
  const hi = Math.max(...years);
  const counts = new Map();
  for (const y of years) counts.set(y, (counts.get(y) || 0) + 1);
  const yearRows = [];
  for (let y = lo; y <= hi; y += 1) yearRows.push({ t: Date.UTC(y, 0, 1), year: y, values: { n: counts.get(y) || 0 } });
  const chart = columnChart(yearCard, {
    rows: yearRows,
    keys: [{ key: "n", label: "References", color: "var(--s1)" }],
    xMin: Date.UTC(lo, 0, 1),
    xMax: Date.UTC(hi + 1, 0, 1),
    bandMs: 365.25 * 86400000,
    height: 140,
    title: (r) => String(r.year),
    ariaLabel: "References per publication year",
  });
  yearCard.append(
    tableView(
      [
        { key: "year", label: "Year" },
        { key: "n", label: "References", num: true, sort: (r) => r.values.n, render: (r) => fmtInt(r.values.n) },
      ],
      yearRows.filter((r) => r.values.n),
    ),
  );

  const listHost = h("div");
  const listCard = card(null, listHost);
  root.append(listCard);

  function draw() {
    const rows = refs.filter(
      (r) =>
        (!sec.value || r.section === sec.value) &&
        (!lvl.value || (lvl.value === "none" ? !r.verification : r.verification?.level === lvl.value)) &&
        matchesQuery(`${r.key} ${r.title} ${r.authors.join(" ")} ${r.venue || ""} ${r.year || ""}`, q.value),
    );
    const bySection = new Map();
    for (const r of rows) {
      const k = r.section || "Other";
      if (!bySection.has(k)) bySection.set(k, []);
      bySection.get(k).push(r);
    }
    listHost.replaceChildren(
      ...[...bySection.entries()].map(([s, items]) =>
        h(
          "div",
          { style: { marginBottom: "12px" } },
          h("h2", null, s, h("span", { class: "muted small", style: { fontWeight: 400 } }, ` · ${items.length}`)),
          h(
            "ul",
            { class: "ref-list" },
            items
              .sort((a, b) => (b.year || 0) - (a.year || 0) || a.key.localeCompare(b.key))
              .map((r) =>
                h(
                  "li",
                  null,
                  h("div", { class: "ref-title" }, h("a", { href: `#/literature/${r.key}` }, r.title)),
                  h(
                    "div",
                    { class: "ref-meta" },
                    `${r.author_short} (${r.year ?? "n.d."}). ${citeLine(r)} `,
                    r.verification ? h("span", { class: ["badge", levelClass(r.verification.level)], title: LEVELS[r.verification.level] || "" }, r.verification.level) : null,
                    r.mentions.length ? h("span", { class: "muted" }, ` · cited in ${r.mentions.length} doc${r.mentions.length > 1 ? "s" : ""}`) : null,
                    r.doi ? [" · ", ext(`https://doi.org/${r.doi}`, "DOI")] : r.arxiv ? [" · ", ext(`https://arxiv.org/abs/${r.arxiv}`, "arXiv")] : null,
                  ),
                ),
              ),
          ),
        ),
      ),
      rows.length ? "" : h("p", { class: "muted" }, "No references match."),
      h("p", { class: "card-foot" }, `${rows.length} of ${refs.length} references.`),
    );
  }
  q.addEventListener("input", draw);
  sec.addEventListener("change", draw);
  lvl.addEventListener("change", draw);
  draw();

  root.append(
    h(
      "div",
      { class: "grid grid-2" },
      yearCard,
      card(
        "Verification levels",
        h(
          "dl",
          { class: "kv" },
          Object.entries(LEVELS).map(([k, v]) => [h("dt", null, h("span", { class: ["badge", levelClass(k)] }, k)), h("dd", null, v)]),
          h("dt", null, h("span", { class: "badge" }, "Q")),
          h("dd", null, "A data service queried programmatically (used for datasets, not papers)."),
          h("dt", null, h("span", { class: "badge" }, "S")),
          h("dd", null, "Screened by title only; not evidence (list below)."),
        ),
      ),
    ),
    card(
      `Screened, not yet read (${ctx.catalog.screened.length})`,
      h("p", { class: "card-sub" }, "From the landscape review. Read these before citing them."),
      h(
        "ul",
        null,
        ctx.catalog.screened.map((sItem) =>
          h(
            "li",
            null,
            sItem.text.replace(/\*/g, ""),
            ...sItem.dois.map((d) => [" · ", ext(`https://doi.org/${d}`, "DOI")]),
            ...sItem.arxiv.map((a) => [" · ", ext(`https://arxiv.org/abs/${a}`, `arXiv:${a}`)]),
          ),
        ),
      ),
    ),
  );
  return () => chart.destroy();
}
