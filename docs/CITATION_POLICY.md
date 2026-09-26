# Citation Policy

## Citing this software

Use `CITATION.cff` (GitHub's "Cite this repository" button reads it). Cite
the specific **release version** you used. After the first Zenodo-archived
release, cite its **version DOI**.

## Release → DOI flow

```text
tag vX.Y.Z → GitHub Release → Zenodo (GitHub integration) archives it → version DOI
      → CITATION.cff updated with the DOI (concept DOI + version DOI) in the next commit
```

## Datasets are cited separately from software

A dataset built with Wave_Analysis (e.g. a synchronised camera + wave-reference
set) is a separate research output:

- its own version (`<name>_vX.Y.Z`), its own Zenodo record and **DataCite DOI**;
- DataCite metadata listing the upstream sources with `relatedIdentifiers`
  (`IsDerivedFrom` for each provider dataset or DOI, `IsCompiledBy` for this
  software's version DOI);
- an RO-Crate (`docs/PROVENANCE.md` §6) and a datasheet (`docs/datasets/`
  template) in the deposit.

Never use the software DOI as a dataset citation, and never cite the reverse.

## Citing upstream providers

Any publication using data acquired by this project must cite the upstream
providers as their terms require. Each datasheet has an "Original citation"
section, and each `datasets.yaml` entry has a `citation` string. Redistribution
of derived data must respect the upstream licence, for example the
**CC BY-NC-ND 4.0** licence of Yin et al. (2025) prohibits redistributing
derivatives.

## Citing literature

Claims in documentation must be backed by an entry in
`docs/literature/bibliography.bib` or a provider-documentation URL. Where a
claim comes from a paywalled full text not re-checked, the verification level
in `docs/literature/source_verification.md` must say so.
