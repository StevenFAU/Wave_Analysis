# Test 8 · Seismic observations at Waimea

Freeze this first comparison before inspecting predictions. Use the existing Waimea benchmark manifest and day, week and year partitions. Pair each image timestamp with the most recent completed UTC hour of IU.KIP vertical ground motion. One hour of waveform may contribute to both cameras. Use only complete hours with a unique, epoch-matched response. No camera pixels, buoy history, wave model or future seismic data enter the seismic regression.

Compute ground-velocity spectra using 256-second detrended Welch windows. Inputs are log integrated power in 0.05–0.10 and 0.10–0.30 Hz, both peak frequencies and the period proxy of twice the reciprocal of the secondary peak. Standardize on training rows, then fit ridge with penalty 10 to raw CDIP significant height and peak period separately. The fit is deterministic across seeds 0, 1 and 2. Compare with the training-month mean on the same finite, positive height and period rows. Report complete-hour coverage, eligible rows and unique HST days, unweighted sample RMSE, and a sensitivity restricted to records with agreeing peak assignments. The primary metric uses all eligible records.

Require the upper end of the 95% paired whole-HST-day bootstrap interval for RMSE(seismic) minus RMSE(monthly climatology) below zero for both targets on the registered year split. Use 2,000 resamples, generator seed 0; day/week splits are context. If either target misses, record the miss and stop before adding a camera. If both pass, the next experiment adds frozen camera features to the same seismic baseline, with wave-model performance as a separate comparator. The archive covers February 2009 through October 2013.

## Result · first gate missed

The held 2013 observations cover 1,926 images across 91 Hawaii dates (January 4 through October 14). Lower RMSE is better. The intervals resample entire dates; both cameras on the same date stay together.

| Instrument target | Monthly climatology | Seismic only | 95% interval for seismic minus climatology |
|---|---:|---:|---:|
| Significant height | 0.4550 m | 0.5345 m | +0.0240 to +0.1326 m |
| Peak period | 2.5312 s | 2.5248 s | −0.2533 to +0.2456 s |

The height result favored climatology. The period difference was unresolved. Restricting to the 1,784 rows with agreeing peak assignments left both decisions unchanged. Of 16,404 rows across the full benchmark manifest, 16,358 matched calibrated complete hours, 44 matched incomplete hours, and two lacked a unique response epoch. These figures apply to this single-station linear seismic recipe. Stop the camera-plus-seismic and Oak Island extensions under the registered rule. All three split results and row predictions are in `runs/seismic-waimea-v1`.
