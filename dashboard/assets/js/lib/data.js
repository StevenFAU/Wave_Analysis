// Data loading. The catalog ships with the site. Live data is read from the
// dashboard-data branch (raw.githubusercontent.com, CORS-enabled, ~5 min CDN
// cache); if that fails, the copy bundled at the last deploy is used.

const TIMEOUT_MS = 10000;

export async function getJSON(url, { timeout = TIMEOUT_MS } = {}) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeout);
  try {
    const res = await fetch(url, { signal: ctrl.signal, cache: "no-cache" });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText} for ${url}`);
    return await res.json();
  } finally {
    clearTimeout(timer);
  }
}

export async function loadCatalog() {
  const catalog = await getJSON("data/catalog.json");
  if (!String(catalog.schema || "").startsWith("wave-analysis/dashboard-catalog@1")) {
    throw new Error(`unsupported catalog schema ${catalog.schema}`);
  }
  return catalog;
}

const BUNDLED = "data/live/";

function checkStatus(status) {
  if (!String(status.schema || "").startsWith("wave-analysis/dashboard-status@1")) {
    throw new Error(`unsupported status schema ${status.schema}`);
  }
  return status;
}

/** -> {status, base, source: 'live'|'bundled'|'none', error} */
export async function loadLive(catalog) {
  const remote = catalog.build?.live_data_url;
  // Five-minute buckets: the CDN caches ~5 min anyway, and this avoids stale
  // browser copies across reloads.
  const bucket = Math.floor(Date.now() / 300000);
  let error = null;
  if (remote) {
    try {
      const status = checkStatus(await getJSON(`${remote}status.json?v=${bucket}`));
      return { status, base: remote, bucket, source: "live", error: null };
    } catch (e) {
      error = e;
    }
  }
  try {
    const status = checkStatus(await getJSON(`${BUNDLED}status.json`));
    return { status, base: BUNDLED, bucket, source: "bundled", error };
  } catch (e) {
    return { status: null, base: null, bucket, source: "none", error: error || e };
  }
}

const seaCache = new Map();

/** Recent realtime observations for one station, or null if not published. */
export async function loadSeaState(live, stationId) {
  if (!live?.base || !live.status?.seastate?.stations?.includes(stationId)) return null;
  const key = `${live.base}${stationId}`;
  if (!seaCache.has(key)) {
    const q = live.source === "live" ? `?v=${live.bucket}` : "";
    const p = getJSON(`${live.base}seastate/${stationId}.json${q}`).catch(async (e) => {
      if (live.source === "live") {
        try {
          return await getJSON(`${BUNDLED}seastate/${stationId}.json`);
        } catch {
          /* fall through */
        }
      }
      seaCache.delete(key);
      throw e;
    });
    seaCache.set(key, p);
  }
  return seaCache.get(key);
}

export function clearSeaStateCache() {
  seaCache.clear();
}

/** Column block -> [{t(ms), ...values}] rows. */
export function blockRows(block) {
  if (!block) return [];
  const keys = Object.keys(block).filter((k) => k !== "t");
  return block.t.map((t, i) => {
    const row = { t: t * 1000 };
    for (const k of keys) row[k] = block[k][i];
    return row;
  });
}

/** Nearest row to time t within maxMs where `key` is present. */
export function nearestRow(rows, t, key, maxMs = 40 * 60 * 1000) {
  let best = null;
  let bestD = Infinity;
  for (const r of rows) {
    if (r[key] == null) continue;
    const d = Math.abs(r.t - t);
    if (d < bestD) {
      bestD = d;
      best = r;
    }
  }
  return bestD <= maxMs ? best : null;
}
