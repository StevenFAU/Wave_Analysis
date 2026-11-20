"""DINOv3 ViT weights between timm's format and DINOv3's own.

The training scripts build DINOv3 backbones with timm
(``timm:vit_base_patch16_dinov3.lvd1689m``, an ``Eva`` model); DINOv3's trainer
(``facebookresearch/dinov3``) uses its ``DinoVisionTransformer``. The tensors are
the same; only some names differ:

=====================  ===================
timm                   DINOv3
=====================  ===================
``reg_token``          ``storage_tokens``
``blocks.N.gamma_1``   ``blocks.N.ls1.gamma``
``blocks.N.gamma_2``   ``blocks.N.ls2.gamma``
=====================  ===================

DINOv3 also holds ``mask_token`` (iBOT, training only) and ``rope_embed.periods``
(computed, not learned), which timm leaves out. timm's ViT-B/16 has no
query/key/value bias, so a DINOv3 checkpoint with a non-zero one cannot load into
it exactly: :func:`dinov3_to_timm` refuses it. A model built from either format
gives the same features (``scripts/dinov3_weights.py check``).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

_RENAMES = (("reg_token", "storage_tokens"), (".gamma_1", ".ls1.gamma"), (".gamma_2", ".ls2.gamma"))
_DINOV3_ONLY = ("mask_token", "rope_embed.periods", "qkv.bias_mask")

#: DinoVisionTransformer settings of the released DINOv3 ViT-B/16, as timm builds it (no qkv bias).
VITB16_KWARGS: dict[str, Any] = {
    "patch_size": 16,
    "in_chans": 3,
    "pos_embed_rope_base": 100.0,
    "pos_embed_rope_normalize_coords": "separate",
    "pos_embed_rope_rescale_coords": 2.0,
    "pos_embed_rope_dtype": "fp32",
    "qkv_bias": False,
    "layerscale_init": 1.0e-5,
    "norm_layer": "layernormbf16",
    "ffn_layer": "mlp",
    "ffn_bias": True,
    "proj_bias": True,
    "n_storage_tokens": 4,
    "mask_k_bias": False,
}


def timm_to_dinov3(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """A timm DINOv3 ViT state dict under DINOv3's names."""
    out = {}
    for k, v in state.items():
        for a, b in _RENAMES:
            k = k.replace(a, b)
        out[k] = v
    return out


def dinov3_to_timm(
    state: Mapping[str, torch.Tensor], *, prefix: str = "backbone."
) -> dict[str, torch.Tensor]:
    """The backbone of a DINOv3 state dict (keys under ``prefix``) under timm's names.

    Drops DINOv3-only entries (mask token, RoPE periods, the k-bias mask) and the
    heads. Raises if a query/key/value bias is not all zero.
    """
    out = {}
    for k, v in state.items():
        if not k.startswith(prefix):
            continue
        k = k.removeprefix(prefix)
        if any(k == x or k.endswith("." + x) for x in _DINOV3_ONLY):
            continue
        if k.endswith("attn.qkv.bias"):
            if torch.count_nonzero(v):
                raise ValueError(f"{k} is not zero; timm's ViT-B/16 has no qkv bias")
            continue
        for a, b in _RENAMES:
            k = k.replace(b, a)
        out[k] = v.detach().to("cpu", torch.float32).contiguous()
    if not out:
        raise ValueError(f"no keys under {prefix!r}")
    return out


def teacher_checkpoint(state: Mapping[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    """A DINOv3 trainer checkpoint (``student.resume_from_teacher_chkpt``) holding only a backbone."""
    return {"teacher": {"backbone." + k: v for k, v in timm_to_dinov3(state).items()}}


def load_teacher_backbone(path: Path | str) -> dict[str, torch.Tensor]:
    """The teacher backbone of a DINOv3 ``teacher_checkpoint.pth``, under timm's names."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    return dinov3_to_timm(ckpt.get("teacher", ckpt))


__all__ = [
    "VITB16_KWARGS",
    "dinov3_to_timm",
    "load_teacher_backbone",
    "teacher_checkpoint",
    "timm_to_dinov3",
]
