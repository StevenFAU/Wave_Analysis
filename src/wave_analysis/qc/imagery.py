"""Image-quality annotation (visual QC layer).

Following the project's preservation principle, these functions **measure and
annotate**; they never discard. Choi et al. (2020) and Kim et al. (2023)
excluded dark, over-exposed, foggy, or obstructed frames at collection time;
here those conditions become explicit columns so that robustness can be
studied (e.g. "error vs. glare fraction") and task-specific filters can be
applied transparently at manifest-construction time.

All metrics are computed with NumPy only, so they are deterministic and do not
depend on optional imaging libraries.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

# ITU-R BT.709 luma coefficients.
_LUMA = np.array([0.2126, 0.7152, 0.0722])


def to_luma(image: ArrayLike) -> FloatArray:
    """Convert an ``HxW`` or ``HxWx3`` image (uint8 or float in [0, 1]) to luma in [0, 1]."""
    img = np.asarray(image)
    x = img.astype(np.float64) / 255.0 if img.dtype == np.uint8 else img.astype(np.float64)
    if x.ndim == 3:
        x = x[..., :3] @ _LUMA
    if x.ndim != 2:
        raise ValueError("image must be HxW or HxWxC")
    return np.clip(x, 0.0, 1.0)


def laplacian_variance(luma: FloatArray) -> float:
    """Variance of the 4-neighbour Laplacian - a standard focus/blur measure (Pech-Pacheco et al. 2000)."""
    lap = (
        -4 * luma[1:-1, 1:-1] + luma[:-2, 1:-1] + luma[2:, 1:-1] + luma[1:-1, :-2] + luma[1:-1, 2:]
    )
    return float(lap.var())


@dataclass(frozen=True)
class ImageQuality:
    """Per-image quality metrics. All fractions in [0, 1]."""

    mean_luma: float
    std_luma: float
    dark_fraction: float
    saturation_fraction: float
    glare_fraction: float
    sharpness: float
    entropy_bits: float

    def as_dict(self) -> dict[str, float]:
        """Plain dict for tabular storage."""
        return asdict(self)


def image_quality(
    image: ArrayLike,
    *,
    dark_threshold: float = 0.08,
    saturation_threshold: float = 0.98,
    glare_threshold: float = 0.92,
    glare_max_chroma: float = 0.08,
) -> ImageQuality:
    """Compute image-quality metrics for one frame.

    ``glare_fraction`` counts pixels that are both very bright and nearly
    achromatic (specular sun glint / whitened sky), which is distinct from
    bright white-water, which retains texture; it is a proxy, validated only
    qualitatively so far (see ``docs/methodology/imagery_processing.md``).
    """
    img = np.asarray(image)
    y = to_luma(img)
    if img.ndim == 3 and img.shape[2] >= 3:
        rgb = img[..., :3].astype(np.float64) / (255.0 if img.dtype == np.uint8 else 1.0)
        chroma = rgb.max(axis=2) - rgb.min(axis=2)
    else:
        chroma = np.zeros_like(y)
    hist, _ = np.histogram(y, bins=256, range=(0.0, 1.0))
    p = hist[hist > 0] / y.size
    return ImageQuality(
        mean_luma=float(y.mean()),
        std_luma=float(y.std()),
        dark_fraction=float((y < dark_threshold).mean()),
        saturation_fraction=float((y >= saturation_threshold).mean()),
        glare_fraction=float(((y >= glare_threshold) & (chroma <= glare_max_chroma)).mean()),
        sharpness=laplacian_variance(y),
        entropy_bits=float(-(p * np.log2(p)).sum()),
    )


def difference_hash(image: ArrayLike, hash_size: int = 8) -> int:
    """64-bit difference hash (dHash) for near-duplicate frame detection.

    Near-duplicate frames split across train and test are a leakage source in
    video datasets; :func:`wave_analysis.datasets.splits.check_leakage` uses
    these hashes to detect them. Resizing is by block averaging (no external
    dependency), which is adequate for duplicate detection.
    """
    y = to_luma(image)
    h, w = hash_size, hash_size + 1
    rows = np.array_split(np.arange(y.shape[0]), h)
    cols = np.array_split(np.arange(y.shape[1]), w)
    small = np.array([[y[np.ix_(r, c)].mean() for c in cols] for r in rows])
    bits = (small[:, 1:] > small[:, :-1]).ravel()
    return int(sum(1 << i for i, b in enumerate(bits) if b))


def hamming_distance(a: int, b: int) -> int:
    """Number of differing bits between two hashes."""
    return int(bin(a ^ b).count("1"))
