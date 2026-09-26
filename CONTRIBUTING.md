# Contributing

Thank you for helping build Wave_Analysis. This is research software, so
correctness, provenance, and documentation matter as much as features.

## Setup

```bash
git clone git@github.com:StevenFAU/Wave_Analysis.git && cd Wave_Analysis
uv sync --group dev          # add --extra ml for the torch reference models
uv run pre-commit install
uv run pytest
```

## Workflow

1. Open or reference an issue: bug, dataset-source proposal, or research question.
2. Branch from `main` (`feature/…`, `fix/…`, `docs/…`, `data/…`).
3. Make focused commits. Update `CHANGELOG.md` under *Unreleased*.
4. Before opening a PR, `uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest` must pass.
5. The PR template checklist is part of review.

## Standards for contributions

- **Science claims need sources.** Every factual statement in docs or docstrings
  is backed by `docs/literature/bibliography.bib`, provider documentation, or a
  test. If it comes from a paywalled paper you have not re-checked, mark it V3
  in `docs/literature/source_verification.md`.
- **No silent data removal.** QC returns flags; filters live in task configs
  (ADR 0001).
- **Units and names.** New variables go through `schemas/variables.py` with CF
  standard names where they exist.
- **Tests with real data.** Parsers are tested on real upstream files trimmed
  to a few KB, stored in `tests/fixtures/<source>/` with a README giving URL,
  retrieval date, and trimming method. Physics code is tested against analytic
  results.
- **Numbers in docs are reproducible.** Put the script that produced a number
  in `scripts/` and cite it next to the number.
- **Docstrings:** numpy style, with units and references for physical formulas.
- **Typing:** `mypy --strict` for `src/`.

## Adding a data source

Follow the checklist in `docs/datasets/README.md` (registry entry, config,
adapter, datasheet, fixtures, tests, changelog).

## Code of conduct

See `CODE_OF_CONDUCT.md`.
