import { card, copyButton, dataTable, ext, h } from "../lib/dom.js";
import { DASH, fmtNum, matchesQuery } from "../lib/format.js";

export const title = ([id], ctx) => {
  if (!id) return "Data sources";
  const d = ctx.catalog.datasets.find((x) => x.dataset_id === id);
  return d ? d.name : "Data source";
};

const STATUS_HELP = {
  verified: "accessed programmatically; formats confirmed on real files",
  documented: "confirmed from provider documentation only",
  candidate: "plausible and relevant, not yet checked",
};

const pretty = (s) => String(s).replace(/_/g, " ");

function isUrl(s) {
  return /^https?:\/\//.test(s);
}

function linkOrText(s) {
  const m = String(s).match(/^(https?:\/\/\S+)(.*)$/);
  return m ? h("span", null, ext(m[1]), m[2]) : String(s);
}

function kvFromObject(obj) {
  const dl = h("dl", { class: "kv" });
  for (const [k, v] of Object.entries(obj || {})) {
    if (v == null || v === "") continue;
    dl.append(h("dt", null, pretty(k)), h("dd", null, Array.isArray(v) ? v.map(pretty).join(", ") : typeof v === "object" ? JSON.stringify(v) : String(v)));
  }
  return dl.childNodes.length ? dl : h("p", { class: "muted" }, DASH);
}

function detail(root, ctx, d) {
  const v = d.verification;
  root.append(
    h("div", { class: "crumbs" }, h("a", { href: "#/sources" }, "Sources"), " / ", d.dataset_id),
    h("h1", null, d.name),
    h(
      "p",
      { class: "lede" },
      d.provider,
      " · ",
      h("span", { class: ["badge", v.status], title: STATUS_HELP[v.status] }, v.status),
      v.date ? ` checked ${v.date}` : "",
      ` · phase ${d.phase}`,
      d.doc_url ? [" · ", h("a", { href: d.doc_url }, "project notes")] : null,
    ),
  );
  const access = d.access;
  root.append(
    h(
      "div",
      { class: "grid grid-2" },
      card(
        "Access",
        h(
          "dl",
          { class: "kv" },
          h("dt", null, "Public"),
          h("dd", null, access.public ? "yes" : "no"),
          h("dt", null, "Authentication"),
          h("dd", null, access.authentication_required ? "required" : "not required"),
          h("dt", null, "Protocols"),
          h("dd", null, access.protocols.join(", ")),
          access.rate_limit_guidance ? [h("dt", null, "Rate limits"), h("dd", null, access.rate_limit_guidance)] : null,
          h("dt", null, "Formats"),
          h("dd", null, d.formats.join(", ") || DASH),
        ),
        access.endpoints.length
          ? [h("h3", { style: { marginTop: "12px" } }, "Endpoints"), h("ul", null, access.endpoints.map((e) => h("li", null, isUrl(e) ? ext(e) : e)))]
          : null,
      ),
      card(
        "Content",
        h(
          "dl",
          { class: "kv" },
          h("dt", null, "Modalities"),
          h("dd", null, d.modalities.map(pretty).join(", ")),
          h("dt", null, "Intended roles"),
          h("dd", null, d.intended_roles.map(pretty).join(", ")),
          h("dt", null, "Data types"),
          h("dd", null, d.data_types.map(pretty).join(", ")),
        ),
        h("h3", { style: { marginTop: "12px" } }, "Time"),
        kvFromObject(d.temporal),
        h("h3", { style: { marginTop: "12px" } }, "Space"),
        kvFromObject(d.spatial),
      ),
      card(
        "Verification",
        h("p", null, h("span", { class: ["badge", v.status] }, v.status), " ", STATUS_HELP[v.status], v.date ? ` (${v.date}).` : "."),
        v.notes ? h("p", null, v.notes) : null,
        v.evidence.length ? [h("h3", null, "Evidence"), h("ul", null, v.evidence.map((e) => h("li", null, linkOrText(e))))] : null,
      ),
      card(
        "Licence and citation",
        h("dl", { class: "kv" }, h("dt", null, "Licence"), h("dd", null, pretty(d.license.status), d.license.url ? [" · ", ext(d.license.url, "terms")] : null)),
        d.license.notes ? h("p", { class: "small muted", style: { marginTop: "8px" } }, d.license.notes) : null,
        d.citation ? [h("h3", { style: { marginTop: "12px" } }, "Citation"), h("p", { class: "small" }, d.citation), copyButton(d.citation, "Copy citation")] : null,
        d.documentation.length ? [h("h3", { style: { marginTop: "12px" } }, "Documentation"), h("ul", null, d.documentation.map((u) => h("li", null, isUrl(u) ? ext(u) : u)))] : null,
      ),
    ),
  );
  if (d.notes) root.append(card("Notes", h("p", null, d.notes)));
  const sites = ctx.catalog.sites.filter((s) => d.sites.includes(s.site_id));
  if (sites.length)
    root.append(
      card(
        "Camera sites from this source",
        dataTable(
          [
            { key: "name", label: "Site" },
            { key: "period", label: "Period" },
            {
              key: "ref",
              label: "Primary reference",
              render: (r) => {
                const p = r.references.find((x) => x.role === "primary");
                return p ? `${p.network} ${p.cdip_id || p.station_id} · ${fmtNum(p.distance_km, 1)} km` : DASH;
              },
            },
          ],
          sites,
          { onRow: () => ctx.navigate("cameras") },
        ),
      ),
    );
}

export async function render(root, ctx, [id], params) {
  const all = ctx.catalog.datasets;
  if (id) {
    const d = all.find((x) => x.dataset_id === id);
    if (!d) {
      root.append(h("h1", null, "Unknown source"), h("p", null, h("a", { href: "#/sources" }, "All sources")));
      return;
    }
    detail(root, ctx, d);
    return;
  }
  root.append(
    h("h1", null, "Data sources"),
    h(
      "p",
      { class: "lede" },
      `The project's registry of ${all.length} public sources (data/registry/datasets.yaml), each with a verification status and evidence. Availability is recorded per station, product and period, never per provider.`,
    ),
  );
  const q = h("input", { type: "search", placeholder: "Search name, provider, notes", "aria-label": "Search sources", value: params.get("q") || "" });
  const statuses = ["verified", "documented", "candidate"];
  const active = new Set(statuses);
  const chips = statuses.map((s) =>
    h("button", {
      class: "chip",
      type: "button",
      "aria-pressed": "true",
      title: STATUS_HELP[s],
      text: s,
      onclick: (e) => {
        if (active.has(s)) active.delete(s);
        else active.add(s);
        e.currentTarget.setAttribute("aria-pressed", String(active.has(s)));
        draw();
      },
    }),
  );
  const modalities = [...new Set(all.flatMap((d) => d.modalities))].sort();
  const mod = h("select", { "aria-label": "Modality" }, h("option", { value: "" }, "Any modality"), modalities.map((m) => h("option", { value: m }, pretty(m))));
  const roles = [...new Set(all.flatMap((d) => d.intended_roles))].sort();
  const role = h("select", { "aria-label": "Role" }, h("option", { value: "" }, "Any role"), roles.map((m) => h("option", { value: m }, pretty(m))));
  root.append(h("div", { class: "filters" }, q, h("div", { class: "chips" }, chips), mod, role));
  const host = h("div");
  root.append(card(null, host));

  function draw() {
    const rows = all.filter(
      (d) =>
        active.has(d.verification.status) &&
        (!mod.value || d.modalities.includes(mod.value)) &&
        (!role.value || d.intended_roles.includes(role.value)) &&
        matchesQuery(`${d.dataset_id} ${d.name} ${d.provider} ${d.notes || ""} ${d.verification.notes || ""}`, q.value),
    );
    host.replaceChildren(
      dataTable(
        [
          { key: "name", label: "Source", render: (d) => h("a", { href: `#/sources/${d.dataset_id}` }, d.name) },
          { key: "provider", label: "Provider", render: (d) => h("span", { class: "small" }, d.provider) },
          {
            key: "status",
            label: "Status",
            sort: (d) => statuses.indexOf(d.verification.status),
            render: (d) => h("span", { class: ["badge", d.verification.status], title: STATUS_HELP[d.verification.status] }, d.verification.status),
          },
          { key: "modalities", label: "Modalities", sort: (d) => d.modalities.join(), render: (d) => h("span", { class: "small" }, d.modalities.map(pretty).join(", ")) },
          { key: "roles", label: "Roles", sort: (d) => d.intended_roles.join(), render: (d) => h("span", { class: "small" }, d.intended_roles.map(pretty).join(", ")) },
          { key: "phase", label: "Phase" },
          { key: "license", label: "Licence", sort: (d) => d.license.status, render: (d) => h("span", { class: "small" }, pretty(d.license.status)) },
          {
            key: "auth",
            label: "Access",
            sort: (d) => d.access.authentication_required,
            render: (d) => h("span", { class: "small" }, d.access.public ? (d.access.authentication_required ? "public, account" : "public, open") : "restricted"),
          },
        ],
        rows,
        { onRow: (d) => ctx.navigate(`sources/${d.dataset_id}`), initialSort: { key: "status", dir: 1 }, caption: "Data sources" },
      ),
      h("p", { class: "card-foot" }, `${rows.length} of ${all.length} sources.`),
    );
  }
  q.addEventListener("input", draw);
  mod.addEventListener("change", draw);
  role.addEventListener("change", draw);
  draw();
}
