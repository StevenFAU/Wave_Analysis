# ADR 0001: Preserve raw bytes; annotate difficult data, never delete at ingestion

**Status:** Accepted (2026-09-25)

## Context
The reviewed studies removed dark, over-exposed, foggy, rainy, or obstructed
imagery at collection time. Their reported performance therefore applies to
benign conditions, and the excluded data cannot be recovered. Upstream
providers also revise files, so derived results must be traceable to exact bytes.

## Decision
1. Raw upstream bytes are stored unchanged and checksummed. Every request
   (including 404s) is logged.
2. No ingestion or preprocessing step removes an observation for quality
   reasons. QC produces flags ("suspect **or of high interest**", QARTOD).
3. Exclusion happens only at ML-manifest construction, via declarative filters,
   and each excluded sample is written to an exclusion ledger with the rule that
   excluded it.

## Consequences
- More storage and more rows to carry; flags must be propagated.
- Robustness studies (night, glare, storms) become possible without
  re-collecting data.
- The population behind every metric is explicit.
