"""A label-free brightness spectrum of the sea in a shore-camera clip (bar M5's physics input).

Recipe v1, fixed 2026-11-18 before any clip was scored (``docs/experiments/webcoos_video.md``,
"Pixel spectrum recipe"). Input: the clip's frames at 4 fps, 256 px high, masks applied, as
cut for the whole-clip windows (frames ``0 .. 64 K - 1``, the K whole 16-s windows).

1. **Grey and blocks.** Each frame to 8-bit grey (PIL ``"L"``, ITU-R 601 luma), cropped to a
   multiple of 4 px wide, and averaged over 4 x 4 px blocks: a 64-row grid of "pixels" that
   averages out JPEG and sensor noise.
2. **Spectra.** For every block's brightness series: Welch's power spectral density, 4 Hz,
   Hann window, segments of 512 samples (128 s; the whole series if shorter), half
   overlapping, each segment linearly detrended.
3. **Water.** The wave band is 0.04-0.33 Hz (periods 3-25 s). Blocks are ranked by how
   much of their brightness *change* lies in the band: the band's share of the rate
   spectrum ``(2 pi f)^2 P(f)`` (the spectrum of the brightness's time derivative) between
   0.04 Hz and the Nyquist frequency (2 Hz). Weighting by f^2 makes slow drift (clouds,
   exposure; a red spectrum) and pixel noise (a white one) both score low, while wave
   motion, which peaks inside the band and falls steeply above it, scores high. Blocks
   with no power above 0.04 Hz (masked or saturated) are dropped; the 20 % with the
   highest share are the water. No labels are used.
4. **Average.** Each water block's brightness spectrum within the band is divided by its
   band power (so bright foam does not outweigh dark water) and the results are averaged:
   S(f), which integrates to 1 over the band.
5. **Features.** The peak of S above its red background: a power law fitted to S over the
   band by least squares in log-log, and the frequency where S most exceeds it (the
   highest bin of log(S / fit), refined by a parabola through it and its two
   neighbours), as a period ``T_pix``; ``T_m01 = m0 / m1`` and ``T_m02 = sqrt(m0 / m2)``
   from the moments of S over the band; the share of S in five period bands (25-16,
   16-12, 12-9, 9-6, 6-3 s); the median band power of the water blocks (grey levels
   squared, log10); and their mean rate share (how clearly the water stands out).
   ``FEATURES`` lists the ten inputs of the ridge heads; ``T_pix`` alone, with no
   training, is the physics floor.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import welch

FPS = 4.0
BLOCK = 4
NPERSEG = 512
BAND = (0.04, 0.33)
SUB_BANDS = ((0.04, 1 / 16), (1 / 16, 1 / 12), (1 / 12, 1 / 9), (1 / 9, 1 / 6), (1 / 6, 0.33))
SUB_BAND_NAMES = ("frac_25_16s", "frac_16_12s", "frac_12_9s", "frac_9_6s", "frac_6_3s")
WATER_FRACTION = 0.2
RECIPE = 1
FEATURES = ("log_T_pix", "log_T_m01", "log_T_m02", *SUB_BAND_NAMES, "log10_band_power",
            "rate_share")  # fmt: skip


def block_means(grey: np.ndarray, block: int = BLOCK) -> np.ndarray:
    """``[T, H, W]`` grey frames to ``[T, H // block, W // block]`` block means (float32)."""
    t, h, w = grey.shape
    h, w = h - h % block, w - w % block
    g = grey[:, :h, :w].astype(np.float32)
    return g.reshape(t, h // block, block, w // block, block).mean(axis=(2, 4))


def background_peak(f: np.ndarray, s: np.ndarray) -> tuple[float, np.ndarray]:
    """The frequency where ``s`` most exceeds a power law fitted to it (log-log least squares).

    Returns the peak (refined by a parabola through the highest bin of ``log(s / fit)``
    and its neighbours) and the fitted background.
    """
    lf, ls = np.log(f), np.log(np.maximum(s, 1e-300))
    slope, icpt = np.polyfit(lf, ls, 1)
    bg = np.exp(icpt + slope * lf)
    r = ls - np.log(bg)
    k = int(np.argmax(r))
    if 0 < k < len(r) - 1:
        den = r[k - 1] - 2 * r[k] + r[k + 1]
        if den < 0:
            delta = float(np.clip(0.5 * (r[k - 1] - r[k + 1]) / den, -0.5, 0.5))
            return float(f[k] + delta * (f[1] - f[0])), bg
    return float(f[k]), bg


def pixel_spectrum(series: np.ndarray, fs: float = FPS) -> dict[str, Any]:
    """The recipe's spectrum and features of block brightness series ``[T, P]`` (time first).

    Returns ``features`` (the ten ridge inputs, plus ``T_pix_s``, ``T_m01_s``, ``T_m02_s``
    in seconds), the band's frequencies ``f`` and averaged spectrum ``S``, and the block
    counts. Features are NaN when no block varies above 0.04 Hz.
    """
    x = np.asarray(series, dtype=np.float64)
    n = x.shape[0]
    nper = min(NPERSEG, n)
    f, p = welch(x, fs=fs, window="hann", nperseg=nper, noverlap=nper // 2,
                 detrend="linear", scaling="density", axis=0)  # fmt: skip
    df = f[1] - f[0]
    band = (f >= BAND[0]) & (f <= BAND[1])
    above = f >= BAND[0]
    p_band = p[band].sum(axis=0) * df
    p_above = p[above].sum(axis=0) * df
    rate = p * (2 * np.pi * f[:, None]) ** 2
    r_band, r_above = rate[band].sum(axis=0), rate[above].sum(axis=0)
    live = p_above > 1e-6
    out: dict[str, Any] = {"n_blocks": int(x.shape[1]), "n_live": int(live.sum()), "n_water": 0,
                           "f": f[band].tolist(), "S": [float("nan")] * int(band.sum()),
                           "background": [float("nan")] * int(band.sum()), "water": []}  # fmt: skip
    nan = dict.fromkeys((*FEATURES, "T_pix_s", "T_m01_s", "T_m02_s"), float("nan"))
    if not live.any() or band.sum() < 2:
        return {**out, "features": nan}
    q = np.where(live, r_band / np.where(live, r_above, 1.0), -np.inf)
    n_water = max(1, round(WATER_FRACTION * int(live.sum())))
    water = np.argsort(-q, kind="stable")[:n_water]
    s = (p[band][:, water] / p_band[water]).mean(axis=1)
    fb = f[band]
    m0, m1, m2 = (float(np.sum(s * fb**k) * df) for k in (0, 1, 2))
    f_peak, bg = background_peak(fb, s)
    feats = {"T_pix_s": 1.0 / f_peak, "T_m01_s": m0 / m1, "T_m02_s": float(np.sqrt(m0 / m2))}
    feats |= {"log_T_pix": np.log(feats["T_pix_s"]), "log_T_m01": np.log(feats["T_m01_s"]),
              "log_T_m02": np.log(feats["T_m02_s"])}  # fmt: skip
    for name, (lo, hi) in zip(SUB_BAND_NAMES, SUB_BANDS, strict=True):
        inside = (fb >= lo) & ((fb < hi) if hi < BAND[1] else (fb <= hi))
        feats[name] = float(np.sum(s[inside]) * df)
    feats["log10_band_power"] = float(np.log10(np.median(p_band[water])))
    feats["rate_share"] = float(np.mean(q[water]))
    return {**out, "n_water": n_water, "S": s.tolist(), "background": bg.tolist(), "water": water.tolist(),
            "features": {k: float(v) for k, v in feats.items()}}  # fmt: skip
