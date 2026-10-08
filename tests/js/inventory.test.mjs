// Data inventory helpers: the catalog's data dictionary joined with status.json collections.
import assert from "node:assert/strict";
import { test } from "node:test";

import {
  countWord,
  groupCollections,
  joinCollections,
  parseMarkup,
  rolesUsed,
  variableColumns,
} from "../../dashboard/assets/js/lib/inventory.js";

const dictionary = {
  roles: [
    { id: "pictures", label: "Pictures" },
    { id: "labels", label: "Labels" },
    { id: "model", label: "Model" },
  ],
  groups: [
    { id: "pictures", title: "Pictures of the sea" },
    { id: "measurements", title: "Measurements" },
    { id: "model", title: "Wave model" },
  ],
  collections: [
    { id: "ndbc_buoycam", group: "pictures", roles: ["pictures"] },
    { id: "cdip", group: "measurements", roles: ["labels"] },
  ],
};

test("markup: code and subscripts, never HTML", () => {
  assert.deepEqual(parseMarkup("H_{m0} = 4√m_{0}, see `WVHT`."), [
    { kind: "text", text: "H" },
    { kind: "sub", text: "m0" },
    { kind: "text", text: " = 4√m" },
    { kind: "sub", text: "0" },
    { kind: "text", text: ", see " },
    { kind: "code", text: "WVHT" },
    { kind: "text", text: "." },
  ]);
  assert.deepEqual(parseMarkup("<b>x</b>"), [{ kind: "text", text: "<b>x</b>" }]);
  assert.deepEqual(parseMarkup(null), []);
});

test("join keeps dictionary order and appends undescribed live collections", () => {
  const live = [
    { id: "cdip", count: 1 },
    { id: "brand_new", count: 3 },
  ];
  const rows = joinCollections(dictionary, live);
  assert.deepEqual(
    rows.map((r) => [r.id, Boolean(r.dict), r.live?.count ?? null]),
    [
      ["ndbc_buoycam", true, null], // described but not on the host
      ["cdip", true, 1],
      ["brand_new", false, 3],
    ],
  );
  assert.equal(joinCollections(dictionary, undefined).length, 2); // live data unavailable
});

test("groups keep their order and drop empty ones", () => {
  const groups = groupCollections(dictionary, joinCollections(dictionary, []));
  assert.deepEqual(
    groups.map((g) => [g.id, g.items.map((r) => r.id)]),
    [
      ["pictures", ["ndbc_buoycam"]],
      ["measurements", ["cdip"]],
    ],
  );
});

test("variable table columns and roles used", () => {
  assert.deepEqual(variableColumns([{ name: "a", unit: "m" }, { name: "b", source: "" }]), { unit: true, role: false, source: false });
  assert.deepEqual(
    rolesUsed(dictionary, [{ roles: ["labels"] }, { roles: ["pictures", "labels"] }]).map((r) => r.id),
    ["pictures", "labels"], // dictionary order, not first-seen order
  );
  assert.equal(countWord(3), "three");
  assert.equal(countWord(12), "12");
});
