# Wave Parameters

Code: `physics/timeseries.py`, `physics/dispersion.py`, `processing/spectra.py`.

## 1. Two families of estimators

A single image captures one instant. A sea state is a statistical description of
a time-varying surface over a record, usually 20–30 minutes. Parameters come in
two families that are **not interchangeable**.

### Time domain (wave-by-wave)

Individual waves are delimited by successive zero crossings of the de-meaned
elevation η(t). Crossing times are linearly interpolated between samples, which
removes the ±1/f_s quantisation in the period. For N waves with heights sorted in
descending order $H_1^\downarrow \ge H_2^\downarrow \ge \dots$:

$$
H_{1/3} = \frac{1}{N_s}\sum_{i=1}^{N_s} H_i^{\downarrow}, \qquad N_s = \lfloor N/3 \rfloor
$$

as used by Yang et al. (2026). The floor-versus-round choice for $N_s$ is recorded
(`n_third_rule`). Also computed: $H_{1/10}$, $H_{\max}$, $H_{\text{mean}}$,
$H_{\text{rms}}$, $T_z$ (mean zero-crossing period), $T_{1/3}$, $T_{H\max}$.
Up-crossing is the default (NDBC, Yang et al.). Down-crossing, which IAHR (1989)
recommends for height definitions, is available.

### Frequency domain (spectral)

With variance density $S(f)$ and moments $m_n = \int f^n S(f)\,df$:

| Parameter | Definition | Note |
|---|---|---|
| $H_{m0}$ | $4\sqrt{m_0}$ | spectral significant height |
| $T_p$ | $1/f_p$ at the discrete maximum | quantised by the grid (NDBC DPD) |
| $T_p^{Y}$ | $\left(\int f S^4 df / \int S^4 df\right)^{-1}$ | Young (1995); far less grid-quantised, **recommended ML target** |
| $T_{m01}$ | $m_0/m_1$ | mean period |
| $T_{m02}$ | $\sqrt{m_0/m_2}$ | spectral estimate of $T_z$ |
| $T_{m-10}$ | $m_{-1}/m_0$ | energy period (wave power) |
| $\nu$ | $\sqrt{m_0 m_2/m_1^2 - 1}$ | spectral width (Longuet-Higgins 1975) |
| $Q_p$ | $2 m_0^{-2}\int f S^2 df$ | Goda peakedness |
| $s_p$ | $2\pi H_{m0}/(g T_p^2)$ | deep-water peak steepness |

For any non-negative spectrum, $T_{m02} \le T_{m01} \le T_{m-10}$ (moment
inequality); the property test in `tests/unit/test_spectra.py` checks it.

### They differ, systematically

For a Gaussian sea, $H_{1/3} \approx 0.95\,H_{m0}$ in deep water (Goda 2010).
NDBC's processing document puts $H_{m0}$ "typically" 5–10 % above $H_{1/3}$
(Earle 2003, citing Longuet-Higgins 1980). `tests/unit/test_physics.py`
reproduces the deep-water ratio on simulated records. In shallow water, where
waves are no longer Gaussian, the ratio changes and can approach or pass 1, so
it must be measured rather than assumed near a surf zone. Labels mixing the
two estimators carry a 5–10 % systematic bias. **Every $H_s$ in this project
records its estimator** (`method` / `hs_estimator`).

NDBC's published `WVHT` equals $H_{m0}$ of its own spectrum to reporting
precision at four stations (31,944 records, |mean Δ| ≤ 0.0014 m). It is tagged
as spectral even though NDBC's glossary describes a highest-third average
(`docs/datasets/ndbc.md`).

## 2. Linear dispersion

$$
\omega^2 = g k \tanh(kh), \qquad \omega = 2\pi f.
$$

Solved for $k$ with Guo's (2002) explicit approximation followed by Newton
iterations (residual < 10⁻⁹ ω² for f ∈ [0.02, 1] Hz and h ∈ [0.2, 5000] m;
property-tested). Derived: wavelength $L = 2\pi/k$, phase speed $c = \omega/k$,
group velocity $c_g = nc$ with $n = \tfrac12(1 + 2kh/\sinh 2kh)$, deep-water
wavelength $L_0 = gT^2/2\pi \approx 1.56\,T^2$, steepness, Iribarren number
$\xi_0 = \tan\beta/\sqrt{H_0/L_0}$ (breaker type), and Ursell number.

The **group velocity** sets how fast a sea state propagates from an offshore
buoy to a coastal camera (`processing.synchronize.propagation_lag`). For
example, 10 km at $T_p = 10$ s in deep water takes about 21 minutes.

## 3. Directional parameters

NDBC publishes $\alpha_1(f), \alpha_2(f), r_1(f), r_2(f)$, which define

$$
D(f,\theta) = \frac{1}{\pi}\left[\tfrac12 + r_1\cos(\theta-\alpha_1) + r_2\cos 2(\theta-\alpha_2)\right].
$$

Bulk: peak direction $D_p = \alpha_1(f_p)$ (NDBC MWD); energy-weighted mean
direction $D_m = \operatorname{atan2}(\bar b_1, \bar a_1)$; circular spread
$\sigma = \sqrt{2(1-\bar r_1)}$ (Kuik et al. 1988). Direction arithmetic always
uses circular statistics (`evaluation/circular.py`).

## 4. Which parameter as an ML target?

| Target | Recommendation |
|---|---|
| $H_s$ | $H_{m0}$ from the reference spectrum; state the estimator |
| Peak period | $T_p^{Y}$ (smooth) rather than discrete $T_p$ |
| Mean period | $T_{m02}$ (comparable to $T_z$) and/or $T_{m01}$ |
| $H_{\max}$ | Only from a displacement time series (CDIP, virtual buoys); it has large sampling variability for a 20–30-min record |
| Direction | Predict $(\sin, \cos)$ and evaluate with circular metrics |
| Spectrum | $S(f)$ on a fixed grid after energy-conserving regridding (see `spectral_analysis.md`) |

## 5. One common baseline for labels

Instruments, models and papers all report "significant wave height", but not
the same quantity. Before labels from different sources enter one dataset they
are put on a common baseline:

| Source of difference | Size | Policy |
|---|---|---|
| Estimator: $H_{m0}$ (spectral) vs $H_{1/3}$ (wave by wave) | 5–10 % (deep water); varies near breaking | **$H_{m0}$ is the canonical label.** NDBC WVHT and CDIP `waveHs` are both verified to be $H_{m0}$; ERA5 `swh` and Spotter $H_s$ are $H_{m0}$ by definition. $H_{1/3}$, $H_{\max}$ and $T_z$ are secondary labels, computed only from a displacement record (CDIP deployment files), which also measures the real ratio per site |
| Frequency band | Small for swell, larger for short wind seas | NDBC spectra stop at 0.35–0.50 Hz depending on the payload (Earle 2003); CDIP's reach higher. When labels from several instruments are combined, recompute $H_{m0}$ over a shared band (e.g. 0.03–0.40 Hz) from the preserved spectra and record the band in `method` |
| Sampling variability of one record | 90 % interval about −10 % to +15 % (Earle 2003); consecutive CDIP 106 records differ by a median 4 % | Average records around the image time (ADR 0010) and carry the spread as label uncertainty |
| Point vs area | ERA5 is a grid-cell mean (0.5°) | Model output is a baseline or prior, never a label |
| Unknown estimator in papers | Kamagata 2026 unstated; Kim 2023 (AWAC) not verified | Flag comparisons; do not mix with project labels |

Results quoted against another study must state whether that study used
$H_{m0}$ or $H_{1/3}$, and convert or caveat the comparison.
