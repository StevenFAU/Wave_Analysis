# ADR 0010: Labels from time-averaged references use the window mean, with the other rules kept for sensitivity

**Status:** Accepted (2026-10-01)

## Context
A camera image is an instant. A wave label is a statistic of a 20–27 min
record: NDBC 1200 s, CDIP 1600 s every 30 min. An image rarely falls at a
record's centre, and some capture clocks are known only to minutes (PacIOOS
Waimea Bay: −26 to +4 min). The Waimea Bay–CDIP 106 pairing
(`docs/datasets/pacioos_beachcam.md`) measured the consequences:

- consecutive records differ by a median 4 % of H_s, so one record carries
  about 4 % sampling noise (NDBC gives 90 % intervals of about −10 % to +15 %
  for one record; Earle 2003);
- the nearest record is a median 10.6 min from the image, and 73 % of images
  fall inside the nearest record's sample;
- taking a single record inflates extremes: the largest-wave image has records
  of 5.46, 6.44, 6.05 and 5.41 m within the hour, and the nearest gives 6.44 m;
- the buoy-to-bay travel time (median 12 min) is smaller than the clock
  uncertainty and the record length;
- peak period and peak direction are not averages over the spectrum: each
  record reports the band with the most energy, and that band jumps when two
  swell systems carry similar energy. Within the ±30 min window the two
  records' T_p differ by 3 s or more for 808 of 16,404 images (4.9 %), and
  their D_p by 60° or more for 401 (2.4 %). The mean of 8 s and 16 s is 12 s,
  a period neither record saw.

`processing.synchronize.pair_window` keeps every record near each image, so
the rule can be chosen and changed afterwards.

## Decision
1. The primary label is the **window mean**: the mean of every record whose
   centre lies within ±30 min of the image (`label_from_candidates(...,
   rule="window_mean", max_offset="30min")`), normally two records. The
   **spread** of those records is stored with the label as its uncertainty.
2. The window mean applies to statistics of the whole spectrum (H_s, mean
   periods, mean direction). **Peak period and peak direction come from the
   nearest record** (`rule="nearest"`). Their range over the window is stored
   beside them, and a flag marks images where it exceeds 3 s or 60°. Mean
   directions use the circular mean (`circular=True`).
3. No propagation lag is applied while the lag is smaller than the clock
   uncertainty and the record length. A lag is applied (`lag_s`) only where it
   is larger than both, and its effect is reported.
4. Every manifest also records the `nearest` and `interpolate` labels. Each
   benchmark reports its headline metric under the window mean and under
   nearest; a material difference means the label, not the model, limits the
   result.
5. The `in_record` subset (images inside a record's sample) is reported as a
   robustness stratum, not used as a filter.

## Consequences
- The window mean halves the noise variance of a single record (two
  independent records) and covers the clock uncertainty without choosing a
  side.
- It smooths a swell arriving within the hour. Those cases show a large
  spread and can be stratified.
- T_p and D_p keep the single-record noise that the window mean removes from
  H_s; they are never a blend of two swell systems. The flagged images
  (929 at Waimea, 5.7 %) are a stratum for any period or direction result.
- Labels are no longer one buoy number per image. Every label names its rule,
  its window and its record count, so results built on different rules
  are not compared as if they were the same.
