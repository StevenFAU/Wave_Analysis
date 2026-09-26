# Spectral Analysis

Code: `processing/spectra.py`; tests: `tests/unit/test_spectra.py`.

## Preservation rule

> Preserve the highest-information upstream representation and derive
> lower-dimensional products later (ADR 0003).

`FrequencySpectrum` holds $S(f)$ plus $(\alpha_1, \alpha_2, r_1, r_2)$ and the
separation frequency when provided. Bulk parameters are derived, tagged, and
never stored in place of the spectrum.

## Bandwidths and moments

Moments use bin bandwidths: $m_n = \sum_i f_i^n S_i \Delta f_i$.

- If the provider publishes bandwidths (CDIP `waveBandwidth`), they are used.
- Otherwise bin edges are the midpoints between centres, with outer edges
  extrapolated symmetrically. On NDBC's non-uniform 47-band grid this rule
  reproduces `WVHT` to rounding at four stations in three ocean basins
  (`docs/datasets/ndbc.md`), which validates it empirically.
- A record with any missing in-band bin returns **NaN** moments, so partial
  spectra never appear as low-energy seas.

## Energy-conserving regridding

Providers use different frequency grids (NDBC: 0.03–0.40 Hz uniform before
~2005, then a 47-band non-uniform grid; CDIP: 64 bands). ML targets need one
grid. Point interpolation of a density does **not** conserve $m_0$ when
resolutions differ. `regrid_energy_conserving` treats the density as
piecewise-constant within source bins (the representation implied by
band-averaged estimates) and redistributes variance by bin overlap:

$$
S^{dst}_j = \frac{1}{\Delta f^{dst}_j}\sum_i S^{src}_i\,\bigl|[f^{src}_{i-}, f^{src}_{i+}] \cap [f^{dst}_{j-}, f^{dst}_{j+}]\bigr|.
$$

Total variance is conserved exactly over the common range (tested to 10⁻¹²).
Destination bins outside the source range, or overlapping a missing bin, are
NaN. They are never extrapolated.

Directional moments must **not** be regridded as densities. Regrid the
energy-weighted coefficients ($S a_1$, $S b_1$, …) and divide by the regridded
$S$. (This helper is planned; `assemble_spectrum` currently refuses mismatched
grids rather than guessing.)

## Directional distribution estimators

From the same four coefficients:

| Method | Properties |
|---|---|
| Truncated Fourier (NDBC formula) | Linear and simple; broad; **can go negative** for narrow spreads |
| **Maximum entropy** (Lygre & Krogstad 1986), default | Non-negative, resolves narrow and bimodal seas much better, **reproduces the input moments exactly** |

Both are verified to reproduce the input $r_1, r_2, \alpha_1$ on cos-2s test
distributions (s = 2, 8, 30). MEM is closer to the true cos-2s shape at s = 40.

With complex moments $c_n = a_n + i b_n$ (nautical angle), the MEM estimate is

$$
\phi_1 = \frac{c_1 - c_2 c_1^*}{1-|c_1|^2},\quad \phi_2 = c_2 - c_1\phi_1,\quad
D(\theta) = \frac{1}{2\pi}\frac{1 - \phi_1 c_1^* - \phi_2 c_2^*}{\left|1 - \phi_1 e^{-i\theta} - \phi_2 e^{-2i\theta}\right|^2}.
$$

## Convention conversions

NDBC $(\alpha, r)$ ↔ nautical Fourier $(a, b)$:
$a_1 = r_1\cos\alpha_1,\ b_1 = r_1\sin\alpha_1,\ a_2 = r_2\cos 2\alpha_2,\ b_2 = r_2\sin 2\alpha_2$.
Round-trip tested. CDIP publishes $(a, b)$ in the nautical convention directly.

## Realisability

The four moments must come from a non-negative distribution:
$|c_2 - c_1^2| \le 1 - |c_1|^2$. Low-energy bins often violate this. They are
flagged SUSPECT (`qc.waves.directional_moment_test`), not trusted.

## Spectra as learning targets (Levels 5–6)

A model predicting $\hat S(f)$ allows every bulk parameter to be derived
consistently, and allows physics-consistency losses
(`models.losses.spectral_consistency_loss`: $|\hat H_s - 4\sqrt{\hat m_0}|$).
Suggested loss for $\hat S$: an error on $\log S$ over the energetic band, plus an
$m_0$ term, since plain MSE is dominated by the peak bins.
