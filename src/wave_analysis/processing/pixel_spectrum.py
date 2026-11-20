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


# --- Recipe v2 (bars M6-M7; docs/experiments/webcoos_video.md, "Physics features v2") ---

RECIPE_V2 = 2
COH_BLOCKS = 500  # at most this many water blocks in the cross-spectral matrix
COH_HALF_BINS = 1  # neighbouring Welch bins pooled on each side (cBathy-style band averaging)
COH_SEED = 0
FEATURES_V2 = ("log_T_on", "log_T_on_m01", "peak_prominence", "coherence")  # added to v1's ten
BREAKING_FEATURES = ("foam_frac", "foam_separability", "onset_rate_per_min", "foam_band_width",
                     "foam_persistence_s")  # fmt: skip


def _welch(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    nper = min(NPERSEG, x.shape[0])
    f, p = welch(x, fs=fs, window="hann", nperseg=nper, noverlap=nper // 2,
                 detrend="linear", scaling="density", axis=0)  # fmt: skip
    return np.asarray(f), np.asarray(p)


def _water(f: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """v1's water rule: (water block indices, live mask, rate share) of spectra ``p`` [f, P]."""
    band, above = (f >= BAND[0]) & (f <= BAND[1]), f >= BAND[0]
    df = f[1] - f[0]
    live = p[above].sum(axis=0) * df > 1e-6
    rate = p * (2 * np.pi * f[:, None]) ** 2
    q = np.where(
        live, rate[band].sum(axis=0) / np.where(live, rate[above].sum(axis=0), 1.0), -np.inf
    )
    n_water = max(1, round(WATER_FRACTION * int(live.sum())))
    return np.argsort(-q, kind="stable")[:n_water], live, q


def otsu(values: np.ndarray, bins: int = 256) -> tuple[float, float]:
    """Otsu's (1979) threshold and its separability eta (between-class over total variance).

    The threshold is the split of the histogram that maximises the between-class variance
    (the lowest such split when several tie); eta is near 1 for two well-separated modes and
    about 0.64 (2 / pi) for one Gaussian mode.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    h, edges = np.histogram(v, bins=bins, range=(float(v.min()), float(v.max()) + 1e-9))
    c = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(h)
    w1 = w0[-1] - w0
    s0 = np.cumsum(h * c)
    m0 = s0 / np.maximum(w0, 1)
    m1 = (s0[-1] - s0) / np.maximum(w1, 1)
    between = w0 * w1 * (m0 - m1) ** 2
    k = int(np.argmax(between[:-1]))
    total = float(np.sum(h * (c - s0[-1] / w0[-1]) ** 2))
    return float(edges[k + 1]), float(between[k] / w0[-1] ** 2 / max(total / w0[-1], 1e-12))


def coherence_spectrum(x: np.ndarray, f_band: np.ndarray, fs: float = FPS) -> np.ndarray:
    """``lambda_1 / sum(lambda)`` of the cross-spectral matrix of ``x`` [T, J] at each band frequency.

    Each block's series is cut into Welch's segments (Hann, half overlap, linear detrend) and
    scaled to unit band power; at each frequency, the Fourier coefficients of all segments
    in that bin and its ``COH_HALF_BINS`` neighbours on each side form the cross-spectral
    matrix, whose eigenvalues come from the small Gram matrix of those coefficient vectors.
    A wave train propagating across the blocks is one coherent mode (share near 1); foam
    flicker and noise spread over many (share near 1 / number of vectors).
    """
    from scipy.signal import detrend

    n, j = x.shape
    nper = min(NPERSEG, n)
    step = nper // 2
    starts = range(0, n - nper + 1, step)
    win = np.hanning(nper + 2)[1:-1] if nper > 2 else np.ones(nper)
    seg = np.stack(
        [detrend(x[s : s + nper], axis=0, type="linear") for s in starts]
    )  # [S, nper, J]
    fft = np.fft.rfft(seg * win[None, :, None], axis=1)  # [S, F, J]
    f = np.fft.rfftfreq(nper, 1 / fs)
    band = (f >= BAND[0]) & (f <= BAND[1])
    pw = (np.abs(fft[:, band]) ** 2).sum(axis=(0, 1))
    fft = fft / np.sqrt(np.maximum(pw, 1e-12))[None, None, :]
    out = np.empty(len(f_band))
    for i, fb in enumerate(f_band):
        k = int(np.argmin(np.abs(f - fb)))
        lo, hi = max(0, k - COH_HALF_BINS), min(len(f), k + COH_HALF_BINS + 1)
        v = fft[:, lo:hi].reshape(-1, j)  # [vectors, J]
        lam = np.clip(np.linalg.eigvalsh(v @ v.conj().T), 0, None)
        out[i] = lam[-1] / max(lam.sum(), 1e-300)
    return out


def _period_features(fb: np.ndarray, s: np.ndarray, df: float) -> dict[str, float]:
    m0, m1, m2 = (float(np.sum(s * fb**k) * df) for k in (0, 1, 2))
    f_peak, bg = background_peak(fb, s)
    resid = np.log(np.maximum(s, 1e-300)) - np.log(bg)
    return {"T_pix_s": 1.0 / f_peak, "T_m01_s": m0 / m1, "T_m02_s": float(np.sqrt(m0 / m2)),
            "peak_prominence": float(resid.max())}  # fmt: skip


def physics_v2(blocks: np.ndarray, fs: float = FPS) -> dict[str, Any]:
    """Spectrum recipe v2 and the breaking features of block brightness ``[T, rows, cols]``.

    1. Water by v1's rule (on the raw series). The common mode, each frame's median
       brightness over the water blocks (auto-exposure, cloud shade), is taken out of every
       water block with that block's own gain (least squares): exposure scales brightness,
       so a plain subtraction would leave the mode in bright and dark blocks and put it
       into blocks that never had it.
    2. S(f): v1's band-normalised mean spectrum of the cleaned water blocks; S_on(f): the
       same for the rectified rate max(dI/dt, 0) (each breaker's sharp rise as an impulse,
       without the slow foam decay that reddens S).
    3. Coherence (cBathy-style): lambda_1 / sum(lambda) of the cross-spectral matrix of the
       cleaned water blocks (at most ``COH_BLOCKS``, a fixed random subsample) at each band
       frequency. Used as a feature, not as a weight on S: weighting S and S_on by it
       lowered the peak period's correlation with the buoy's T_p on the development dates.
    4. The four features v2 adds to v1's ten (``FEATURES_V2``): ``log_T_on`` (the peak of
       S_on above a power law, as v1's T_pix) and ``log_T_on_m01`` (its m0 / m1);
       ``peak_prominence`` (the natural log of how far S stands above the power law at its
       peak: low when the clip has no clear wave peak); ``coherence`` (the S-weighted mean
       coherence over the band). The untrained period ``T_pix_s`` is the peak of the
       cleaned S, as v1's T_pix.
    5. Breaking (for H_s), all relative to the water mask: ``foam_frac``, the time-mean
       share of water blocks brighter than the clip's Otsu threshold over water-block
       brightness; ``onset_rate_per_min``, foam arrivals per water block per minute: a
       block's brightness rising through the threshold (above it by the frame noise,
       after last being below it by the same; the noise is 1.4826 x the median absolute
       frame-to-frame change over the water blocks);
       ``foam_separability``, that threshold's eta (Otsu's between-class over total
       variance: high when foam is a bright second mode, lower when the split only cuts one
       mode, as on a calm sea, so the head can discount the other foam features);
       ``foam_band_width``, per image column, the share of its water blocks whose time-mean
       brightness passes the threshold (the vertical, roughly cross-shore, extent of the
       bright band), averaged over columns with at least 3 water blocks;
       ``foam_persistence_s``, over the water blocks whose time-mean brightness passes the
       threshold, the median lag at which a block's cleaned brightness autocorrelation
       first falls below 1/e.
    """
    t, nr, nc = blocks.shape
    x = np.asarray(blocks, dtype=np.float64).reshape(t, nr * nc)
    nan = dict.fromkeys((*FEATURES_V2, "T_pix_s", "T_on_s", *BREAKING_FEATURES), float("nan"))
    f, p = _welch(x, fs)
    band = (f >= BAND[0]) & (f <= BAND[1])
    if band.sum() < 2:
        return {"n_blocks": nr * nc, "n_live": 0, "n_water": 0, "features": nan}
    water, live, _ = _water(f, p)
    if not live.any():
        return {"n_blocks": nr * nc, "n_live": 0, "n_water": 0, "features": nan}
    xw = x[:, water]
    cm = np.median(xw, axis=1)
    cm = cm - cm.mean()
    gain = ((xw - xw.mean(axis=0)) * cm[:, None]).sum(axis=0) / max(float((cm**2).sum()), 1e-12)
    xc = xw - cm[:, None] * gain  # [T, water]
    f, pc = _welch(xc, fs)
    df = f[1] - f[0]
    fb = f[band]
    p_band = pc[band].sum(axis=0) * df
    s = (pc[band] / p_band).mean(axis=1)
    rate = np.maximum(np.diff(xc, axis=0, prepend=xc[:1]) * fs, 0.0)  # length T: S's grid
    _, p_on = _welch(rate, fs)
    on_band = p_on[band].sum(axis=0) * df
    ok = on_band > 0
    s_on = (p_on[band][:, ok] / on_band[ok]).mean(axis=1) if ok.any() else np.full(len(fb), np.nan)
    rng = np.random.default_rng(COH_SEED)
    sub = np.sort(rng.choice(len(water), size=min(COH_BLOCKS, len(water)), replace=False))
    coh = coherence_spectrum(xc[:, sub], fb, fs)
    feats = _period_features(fb, s, df)
    feats["coherence"] = float(np.sum(s * coh) * df)
    if ok.any():
        s_on = s_on / (s_on.sum() * df)
        pf = _period_features(fb, s_on, df)
        feats["T_on_s"], feats["T_on_m01_s"] = pf["T_pix_s"], pf["T_m01_s"]
    else:
        feats["T_on_s"] = feats["T_on_m01_s"] = float("nan")
    for k in ("T_pix", "T_on", "T_on_m01"):
        feats[f"log_{k}"] = float(np.log(feats[f"{k}_s"]))
    # breaking: raw brightness for the foam threshold, cleaned series for jumps and decay
    thr, feats["foam_separability"] = otsu(xw[:: max(1, t // 400)])
    feats["foam_frac"] = float(np.mean(xw > thr))
    dx = np.diff(xw, axis=0)
    noise = 1.4826 * float(np.median(np.abs(dx - np.median(dx))))
    state = np.where(xw > thr + noise, 1, np.where(xw < thr - noise, -1, 0))
    last = np.maximum.accumulate(np.where(state != 0, np.arange(t)[:, None], 0), axis=0)
    state = np.take_along_axis(state, last, axis=0)  # the last side the block was clearly on
    arrivals = ((state[1:] == 1) & (state[:-1] == -1)).sum(axis=0)
    feats["onset_rate_per_min"] = float(arrivals.mean() / (t / fs / 60.0))
    timex = x.mean(axis=0)
    wmask = np.zeros(nr * nc, bool)
    wmask[water] = True
    bright = (timex > thr) & wmask
    wcol, bcol = wmask.reshape(nr, nc).sum(axis=0), bright.reshape(nr, nc).sum(axis=0)
    cols = wcol >= 3
    feats["foam_band_width"] = (
        float(np.mean(bcol[cols] / wcol[cols])) if cols.any() else float("nan")
    )
    foam = np.flatnonzero(bright[water])  # positions within the water blocks
    if len(foam):
        fsub = np.sort(rng.choice(foam, size=min(COH_BLOCKS, len(foam)), replace=False))
        y = xc[:, fsub] - xc[:, fsub].mean(axis=0)
        nfft = 1 << int(np.ceil(np.log2(2 * t)))
        spec = np.fft.rfft(y, n=nfft, axis=0)
        acf = np.fft.irfft(np.abs(spec) ** 2, n=nfft, axis=0)[:t]
        acf = acf / np.maximum(acf[0], 1e-12)
        below = acf < np.exp(-1)
        first = np.where(below.any(axis=0), below.argmax(axis=0), t)
        feats["foam_persistence_s"] = float(np.median(first) / fs)
    else:
        feats["foam_persistence_s"] = float("nan")
    return {"n_blocks": nr * nc, "n_live": int(live.sum()), "n_water": len(water), "f": fb.tolist(),
            "S": s.tolist(), "S_on": s_on.tolist(), "coherence": coh.tolist(),
            "threshold": thr, "features": {k: float(v) for k, v in feats.items()}}  # fmt: skip
