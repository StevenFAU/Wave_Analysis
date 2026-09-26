# ADR 0007: Deep learning optional; statistical baselines mandatory

**Status:** Accepted (2026-09-25)

## Context
The project's first contribution is the data foundation. PyTorch is a large
dependency that most data-pipeline users do not need. Complex models without
baselines cannot be interpreted.

## Decision
`torch` and `torchvision` are the optional `ml` extra. Torch modules are not
imported by the package root. Every reported model result must include the
statistical baselines (mean, climatology, ridge) on the same split. Literature
architectures are reference re-implementations with unverifiable details exposed
as parameters.

## Consequences
- The core installs quickly. CI tests torch modules in a separate job.
- Paper replications are parameterised, not hard-coded claims.
