// Dashboard bootstrap: load data, route between views, theme, freshness pill.
import { collectorHealth } from "./lib/coverage.js";
import { loadCatalog, loadLive, clearSeaStateCache } from "./lib/data.js";
import { clear, ext, h } from "./lib/dom.js";
import { fmtAgo, fmtUtc, parseTime } from "./lib/format.js";
import { hideTooltip } from "./lib/tooltip.js";
import * as about from "./views/about.js";
import * as archive from "./views/archive.js";
import * as cameras from "./views/cameras.js";
import * as literature from "./views/literature.js";
import * as overview from "./views/overview.js";
import * as sources from "./views/sources.js";
import * as station from "./views/station.js";

const ROUTES = [
  { re: /^$/, view: overview, nav: "" },
  { re: /^cameras$/, view: cameras, nav: "cameras" },
  { re: /^station\/([A-Za-z0-9_]+)(?:\/(\d+))?$/, view: station, nav: "cameras" },
  { re: /^archive$/, view: archive, nav: "archive" },
  { re: /^sources(?:\/([a-z0-9_]+))?$/, view: sources, nav: "sources" },
  { re: /^literature(?:\/([A-Za-z0-9_]+))?$/, view: literature, nav: "literature" },
  { re: /^about$/, view: about, nav: "about" },
];

const LIVE_REFRESH_MS = 10 * 60 * 1000;
const ctx = { catalog: null, live: null, navigate };
let cleanup = null;
let lastRoute = null;
let currentView = null;

function navigate(path) {
  location.hash = `#/${path}`;
}

function parseRoute() {
  const raw = decodeURIComponent(location.hash.replace(/^#\/?/, ""));
  const [path, query = ""] = raw.split("?");
  return { path: path.replace(/\/+$/, ""), params: new URLSearchParams(query) };
}

function setNav(nav) {
  for (const a of document.querySelectorAll(".nav a[data-route]")) {
    if (a.dataset.route === nav) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  }
}

async function render({ focus = true } = {}) {
  const root = document.getElementById("view");
  const { path, params } = parseRoute();
  const route = ROUTES.find((r) => r.re.test(path));
  hideTooltip();
  if (cleanup) {
    try {
      cleanup();
    } catch (e) {
      console.warn("view cleanup failed", e);
    }
    cleanup = null;
  }
  clear(root);
  if (!route) {
    setNav(null);
    document.title = "Not found · Wave_Analysis";
    root.append(h("h1", null, "Page not found"), h("p", null, h("a", { href: "#/" }, "Back to the overview")));
    return;
  }
  setNav(route.nav);
  currentView = route.view;
  const match = path.match(route.re);
  try {
    const out = await route.view.render(root, ctx, match.slice(1), params);
    cleanup = typeof out === "function" ? out : null;
    document.title = `${route.view.title?.(match.slice(1), ctx) ?? "Dashboard"} · Wave_Analysis`;
  } catch (e) {
    console.error(e);
    clear(root);
    root.append(
      h("h1", null, "Something went wrong"),
      h("p", { class: "error" }, String(e?.message || e)),
      h("p", null, "Reload the page, or report the problem on ", ext(`https://github.com/${ctx.catalog?.build?.repository || "StevenFAU/Wave_Analysis"}/issues`, "GitHub"), "."),
    );
  }
  if (focus && lastRoute !== null && lastRoute !== path) {
    window.scrollTo(0, 0);
    root.focus({ preventScroll: true });
  }
  lastRoute = path;
}

function renderFreshness() {
  const pill = document.getElementById("freshness");
  clear(pill);
  const live = ctx.live;
  if (!live || !live.status) {
    pill.append(h("span", { class: "status critical" }, "Live data unavailable"));
    pill.title = live?.error ? String(live.error.message || live.error) : "";
    return;
  }
  const st = live.status;
  const lastRun = parseTime(st.collector?.last_run);
  const health = collectorHealth(lastRun);
  const gen = parseTime(st.generated_at);
  pill.append(h("span", { class: ["status", health.level] }, health.short));
  pill.append(h("span", { class: "muted" }, `· data ${fmtAgo(gen)}${live.source === "bundled" ? " (cached copy)" : ""}`));
  pill.title = `Last collector run ${fmtUtc(lastRun)}; data generated ${fmtUtc(gen)}; source: ${
    live.source === "live" ? "dashboard-data branch" : "copy bundled with the site"
  }`;
}

function renderFooter() {
  const f = document.getElementById("footer");
  clear(f);
  const b = ctx.catalog.build;
  const repo = `https://github.com/${b.repository}`;
  f.append(
    h(
      "p",
      null,
      "Built from ",
      b.commit ? ext(`${repo}/commit/${b.commit}`, b.commit.slice(0, 7)) : "an uncommitted tree",
      ` on ${fmtUtc(parseTime(b.built_at))} · wave-analysis ${b.software_version} · `,
      ext(repo, "Source on GitHub"),
      " · ",
      h("a", { href: "docs/" }, "Documentation"),
    ),
    h(
      "p",
      null,
      "Buoy-camera images and observations: NOAA National Data Buoy Center (U.S. Government work, public domain); sea-state values are provisional realtime data. ",
      "Map data © OpenStreetMap contributors; bathymetry © GEBCO Compilation Group. Code: MIT licence.",
    ),
  );
}

function setupTheme() {
  const btn = document.getElementById("theme-toggle");
  const order = ["auto", "light", "dark"];
  const current = () => document.documentElement.dataset.theme || "auto";
  const label = () => {
    const t = current();
    btn.textContent = { auto: "Theme: auto", light: "Theme: light", dark: "Theme: dark" }[t];
    btn.setAttribute("aria-label", `Colour theme: ${t}. Click to change.`);
  };
  label();
  btn.addEventListener("click", () => {
    const next = order[(order.indexOf(current()) + 1) % order.length];
    if (next === "auto") delete document.documentElement.dataset.theme;
    else document.documentElement.dataset.theme = next;
    try {
      if (next === "auto") localStorage.removeItem("wa-theme");
      else localStorage.setItem("wa-theme", next);
    } catch {
      /* storage unavailable */
    }
    label();
    render({ focus: false }); // canvases resolve colours at draw time
  });
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    if (current() === "auto") render({ focus: false });
  });
}

async function refreshLive() {
  const next = await loadLive(ctx.catalog);
  if (next.status && (!ctx.live?.status || next.status.generated_at !== ctx.live.status.generated_at)) {
    ctx.live = next;
    clearSeaStateCache();
    renderFreshness();
    if (currentView?.refreshOnLive) render({ focus: false });
  } else if (!ctx.live?.status) {
    ctx.live = next;
    renderFreshness();
  }
}

async function main() {
  setupTheme();
  const root = document.getElementById("view");
  try {
    ctx.catalog = await loadCatalog();
  } catch (e) {
    clear(root);
    root.append(h("h1", null, "Could not load the catalog"), h("p", { class: "error" }, String(e.message || e)));
    return;
  }
  renderFooter();
  ctx.live = await loadLive(ctx.catalog);
  renderFreshness();
  window.addEventListener("hashchange", () => render());
  await render({ focus: false });
  setInterval(refreshLive, LIVE_REFRESH_MS);
  setInterval(renderFreshness, 60 * 1000);
}

main();
