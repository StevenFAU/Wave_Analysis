// The data inventory: the catalog's data dictionary (data/registry/data_dictionary.yaml)
// joined with the live collection totals from status.json. Pure; unit-tested in
// tests/js/inventory.test.mjs.

/**
 * Split dictionary text into runs: `code` and _{subscript} (the only markup the
 * dictionary uses). Returns [{kind: "text"|"code"|"sub", text}].
 */
export function parseMarkup(text) {
  const out = [];
  const re = /`([^`]+)`|_\{([^}]+)\}/g;
  let last = 0;
  let m;
  const s = String(text ?? "");
  while ((m = re.exec(s))) {
    if (m.index > last) out.push({ kind: "text", text: s.slice(last, m.index) });
    out.push(m[1] != null ? { kind: "code", text: m[1] } : { kind: "sub", text: m[2] });
    last = re.lastIndex;
  }
  if (last < s.length) out.push({ kind: "text", text: s.slice(last) });
  return out;
}

/**
 * One entry per collection, in dictionary order: {id, dict, live}. Live
 * collections the dictionary does not describe are appended (dict null), so a
 * new collection shows up before its description is written.
 */
export function joinCollections(dictionary, liveCollections) {
  const live = new Map((liveCollections || []).map((c) => [c.id, c]));
  const rows = (dictionary?.collections || []).map((d) => ({ id: d.id, dict: d, live: live.get(d.id) || null }));
  const known = new Set(rows.map((r) => r.id));
  for (const c of liveCollections || []) if (!known.has(c.id)) rows.push({ id: c.id, dict: null, live: c });
  return rows;
}

/** Collections grouped by the dictionary's groups, in order; empty groups dropped. */
export function groupCollections(dictionary, joined) {
  return (dictionary?.groups || [])
    .map((g) => ({ ...g, items: joined.filter((r) => r.dict?.group === g.id) }))
    .filter((g) => g.items.length);
}

/** Which optional variable-table columns have any value. */
export function variableColumns(variables) {
  const has = (k) => (variables || []).some((v) => v[k] != null && v[k] !== "");
  return { unit: has("unit"), role: has("role"), source: has("source") };
}

/** The roles used in a list of collections and tables, in dictionary order. */
export function rolesUsed(dictionary, items) {
  const used = new Set(items.flatMap((x) => x.roles || []));
  return (dictionary?.roles || []).filter((r) => used.has(r.id));
}

/** Plain-language phrase for a group count: "three sets of sea pictures". */
const WORDS = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"];
export function countWord(n) {
  return n >= 0 && n < WORDS.length ? WORDS[n] : String(n);
}
