#!/usr/bin/env python
"""Decode Yin et al.'s VWBT_9000 clips once and cache frozen per-frame features.

Design and bar: ``docs/experiments/breaker_types.md``. The heads are trained on this
cache by ``scripts/train_breaker_heads.py``; nothing here is trained.

Inputs (figshare doi:10.6084/m9.figshare.28814993.v5, CC BY 4.0, checked against the
record's MD5s by ``deploy/dgx/breakers.sh fetch``):

* ``Set_A.zip``: 9,000 clips, MPEG-4, 224 x 224, 5 fps, 150 frames (30 s);
* ``Set_B.zip``: the same clips as 60 JPEG frames at 2 Hz (Yin et al.'s own frames);
* ``loso_split_by_database_filename.csv``: the official leave-one-site-out split.

Outputs in ``--out`` (fp16 memmaps, clip order of ``clips.csv``):

* ``a_dinov3.npy`` (N, 150, 1536): DINOv3 ViT-B/16, every Set A frame, [CLS | mean patch];
* ``a_resnet18.npy`` (N, 150, 512): ImageNet ResNet-18 (Yin et al.'s encoder), every Set A frame;
* ``b_resnet18.npy`` (N, 60, 512): the same ResNet-18 on Set B's 60 frames;
* ``clips.csv`` (per clip: frames decoded, fps, Set B frame names) and ``meta.json``.

With ``--encoders vjepa2`` (bar Y-B3) the pass writes instead, for each of three versions
of every Set A clip (``video``: the frames in order; ``shuffled``: the same 150 frames in
a fixed random order per clip; ``still``: each window's middle frame repeated):

* ``a_vjepa2_<version>.npy`` (N, 75, 1024): frozen V-JEPA 2 ViT-L on three 64-frame
  windows (starting at frames 0, 42, 86) resized to 256 x 256, the encoder output averaged
  over its 16 x 16 spatial positions (one token per 2-frame tubelet, 0.4 s), each of the
  clip's 75 tubelets taken from the window whose centre is nearest.

Each clip is read from its zip once; nothing is written to disk but the features.

    python scripts/breaker_features.py --source <dir with the three files> --out <dir> \\
        --device cuda [--encoders vjepa2] [--limit 16]
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

CLASSES = ["Spilling", "Plunging", "Surging"]
A_FRAMES = 150  # 30 s at 5 fps
B_FRAMES = 60  # 30 s at 2 Hz
SIDE = 224
MEAN = (0.485, 0.456, 0.406)  # ImageNet; also DINOv3's (timm pretrained_cfg)
STD = (0.229, 0.224, 0.225)
DINOV3 = "vit_base_patch16_dinov3.lvd1689m"
VJEPA2 = "facebook/vjepa2-vitl-fpc64-256"
VJ_WINDOW = 64  # frames per window: the model's clip length (12.8 s at 5 fps)
VJ_STARTS = (0, 42, 86)  # three windows covering frames 0-149; even, on tubelet bounds
VJ_SIDE = 256  # the model's input size; HF's rotary positions assume its 16 x 16 grid
VJ_VERSIONS = ("video", "shuffled", "still")
CLIP_RE = re.compile(r"^(Spilling|Plunging|Surging)-(\d+)-([A-Za-z0-9_]+)-(\d{14}Z)$")


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def parse_name(name: str) -> tuple[str, str, str, str]:
    """(label, index, camera, timestamp) of a Set A file or Set B folder name."""
    stem = Path(name).name.removesuffix(".mp4")
    m = CLIP_RE.match(stem)
    if not m:
        raise ValueError(f"not a VWBT_9000 clip name: {name}")
    return m.groups()  # type: ignore[return-value]


def official_split(path: Path) -> pd.DataFrame:
    """The official split, checked: six folds, one held-out site each, every clip once per fold."""
    split = pd.read_csv(path, encoding="utf-8-sig")
    need = {
        "fold",
        "held_out_site",
        "split",
        "filename",
        "site",
        "tensor_camera",
        "true_label",
        "database_camera",
        "timestamp_utc",
    }
    if missing := need - set(split.columns):
        raise ValueError(f"split file lacks {sorted(missing)}")
    if split.fold.nunique() != 6 or not split.split.isin(["train", "val", "test"]).all():
        raise ValueError("expected six folds of train / val / test rows")
    for fold, g in split.groupby("fold"):
        if g.filename.duplicated().any():
            raise ValueError(f"fold {fold}: a clip appears twice")
        test_sites = set(g.site[g.split.eq("test")])
        if test_sites != set(g.held_out_site) or len(test_sites) != 1:
            raise ValueError(f"fold {fold}: the test rows are not exactly one held-out site")
        if test_sites & set(g.site[g.split.ne("test")]):
            raise ValueError(f"fold {fold}: the held-out site is also in train / val")
    clips = split[split.fold.eq(split.fold.min())]
    if set(clips.filename) != set(split.filename) or len(clips) != split.filename.nunique():
        raise ValueError("the folds do not cover the same clips")
    for name, label in zip(clips.filename, clips.true_label, strict=True):
        if parse_name(name)[0] != label:
            raise ValueError(f"label of {name} differs from its name")
    return split


def clip_table(split: pd.DataFrame) -> pd.DataFrame:
    clips = split[split.fold.eq(split.fold.min())].sort_values("filename").reset_index(drop=True)
    parts = clips.filename.map(parse_name)
    clips = clips.assign(
        label=clips.true_label,
        serial=[p[1] for p in parts],
        member_a="Set_A/" + clips.filename,
        key_b=[
            f"{p[0]}-{p[1]}-{cam}-{p[3]}" for p, cam in zip(parts, clips.tensor_camera, strict=True)
        ],
    )
    if clips.duplicated(["label", "serial"]).any():
        raise ValueError("label and serial number do not identify a clip")
    return clips[
        [
            "filename",
            "site",
            "database_camera",
            "tensor_camera",
            "label",
            "timestamp_utc",
            "member_a",
            "key_b",
        ]
    ]


def set_b_members(names: list[str], clips: pd.DataFrame) -> tuple[list[list[str]], list[int]]:
    """Set B's first 60 JPEG members of each clip in frame order, and how many it has.

    The folder is named like the Set A file (v5) or with the tensor camera. Two folders
    of v5 hold a 61st frame; the first 60 are kept, as BreakTypeNet's loader does.
    """
    by_folder: dict[str, list[str]] = {}
    for n in names:
        if n.lower().endswith((".jpg", ".jpeg")):
            by_folder.setdefault(Path(n).parent.name, []).append(n)
    out, found = [], []
    for key, fname in zip(clips.key_b, clips.filename, strict=True):
        frames = by_folder.get(fname.removesuffix(".mp4")) or by_folder.get(key)
        if not frames:
            raise ValueError(f"no Set B folder for {fname} (looked for {key})")
        frames = sorted(frames, key=lambda n: (frame_number(n), n))
        if len(frames) < B_FRAMES:
            raise ValueError(f"{key}: {len(frames)} Set B frames, expected {B_FRAMES}")
        out.append(frames[:B_FRAMES])
        found.append(len(frames))
    return out, found


def frame_number(name: str) -> int:
    digits = re.findall(r"\d+", Path(name).stem)
    if not digits:
        raise ValueError(f"no frame number in {name}")
    return int(digits[-1])


def decode_mp4(data: bytes, tmpdir: str) -> tuple[np.ndarray, float]:
    """All frames of an MP4 (RGB uint8, T x H x W x 3) and its frame rate, via OpenCV's FFmpeg."""
    import cv2

    fd, path = tempfile.mkstemp(suffix=".mp4", dir=tmpdir)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        cap = cv2.VideoCapture(path)
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
    finally:
        os.unlink(path)
    if not frames:
        raise ValueError("no decodable frame")
    return np.stack(frames), fps


def fit_length(frames: np.ndarray, n: int) -> np.ndarray:
    """Exactly n frames: cut the tail or repeat the last frame (counted in clips.csv)."""
    if len(frames) >= n:
        return frames[:n]
    return np.concatenate([frames, np.repeat(frames[-1:], n - len(frames), 0)])


class SetA(torch.utils.data.Dataset):
    def __init__(self, zip_path: Path, members: list[str], tmpdir: str) -> None:
        self.zip_path, self.members, self.tmpdir = zip_path, members, tmpdir
        self.zip: zipfile.ZipFile | None = None

    def __len__(self) -> int:
        return len(self.members)

    def __getitem__(self, i: int):
        if self.zip is None:  # one handle per worker
            self.zip = zipfile.ZipFile(self.zip_path)
        frames, fps = decode_mp4(self.zip.read(self.members[i]), self.tmpdir)
        if frames.shape[1:] != (SIDE, SIDE, 3):
            raise ValueError(f"{self.members[i]}: frames {frames.shape[1:]}, expected 224 x 224")
        n = len(frames)
        x = torch.from_numpy(fit_length(frames, A_FRAMES)).permute(0, 3, 1, 2)
        return x, i, n, fps


class SetB(torch.utils.data.Dataset):
    def __init__(self, zip_path: Path, members: list[list[str]]) -> None:
        self.zip_path, self.members = zip_path, members
        self.zip: zipfile.ZipFile | None = None

    def __len__(self) -> int:
        return len(self.members)

    def __getitem__(self, i: int):
        from io import BytesIO

        from PIL import Image

        if self.zip is None:
            self.zip = zipfile.ZipFile(self.zip_path)
        frames = []
        for m in self.members[i]:
            with Image.open(BytesIO(self.zip.read(m))) as im:
                im = im.convert("RGB")
                if im.size != (SIDE, SIDE):  # Yin et al. resize to 224 x 224
                    im = im.resize((SIDE, SIDE), Image.BILINEAR)
                frames.append(np.asarray(im))
        return torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2), i, len(frames), 2.0


def collate(batch):
    x, i, n, fps = zip(*batch, strict=True)
    return torch.stack(x), torch.tensor(i), torch.tensor(n), torch.tensor(fps)


class Encoders:
    """Frozen ResNet-18 (ImageNet, torchvision V1 weights: Yin et al.'s) and DINOv3 ViT-B/16."""

    def __init__(self, device: str, dinov3: bool = True) -> None:
        from torchvision.models import ResNet18_Weights, resnet18

        self.device = device
        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.fc = torch.nn.Identity()
        self.resnet = self.resnet.to(device).eval().requires_grad_(False)
        self.dinov3 = None
        if dinov3:
            import timm

            self.dinov3 = timm.create_model(DINOV3, pretrained=True, num_classes=0)
            self.dinov3 = self.dinov3.to(device).eval().requires_grad_(False)
            self.prefix = int(self.dinov3.num_prefix_tokens)  # CLS + 4 registers
        self.mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device=device).view(1, 3, 1, 1)

    def normalise(self, x: torch.Tensor) -> torch.Tensor:
        return (x.to(self.device, non_blocking=True).float() / 255 - self.mean) / self.std

    @torch.inference_mode()
    def resnet18(self, frames: torch.Tensor, chunk: int = 600) -> torch.Tensor:
        # fp32, as Yin et al. ran it; eval-mode batch norm (ImageNet statistics)
        return torch.cat([self.resnet(self.normalise(c)) for c in frames.split(chunk)])

    @torch.inference_mode()
    def dino(self, frames: torch.Tensor, chunk: int = 300) -> torch.Tensor:
        out = []
        for c in frames.split(chunk):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device != "cpu"):
                tokens = self.dinov3.forward_features(self.normalise(c)).float()
            out.append(torch.cat([tokens[:, 0], tokens[:, self.prefix :].mean(1)], 1))
        return torch.cat(out)


def stitch_plan(
    starts: tuple[int, ...] = VJ_STARTS,
    window: int = VJ_WINDOW,
    tubelet: int = 2,
    n_frames: int = A_FRAMES,
) -> tuple[np.ndarray, np.ndarray]:
    """For each of the clip's n_frames / tubelet tubelets: the window it is read from (the
    one whose centre is nearest, ties to the earlier) and its token position there."""
    if n_frames % tubelet or window % tubelet or any(s % tubelet for s in starts):
        raise ValueError("windows must start and end on tubelet bounds")
    if starts[0] != 0 or starts[-1] + window != n_frames:
        raise ValueError(f"windows at {starts} do not span frames 0-{n_frames - 1}")
    if any(b - a > window for a, b in itertools.pairwise(starts)):
        raise ValueError(f"windows at {starts} leave a gap")
    centre_t = np.arange(0, n_frames, tubelet) + (tubelet - 1) / 2
    centre_w = np.array(starts) + (window - 1) / 2
    win = np.abs(centre_t[:, None] - centre_w[None]).argmin(1)  # argmin: ties to the first
    pos = (np.arange(0, n_frames, tubelet) - np.array(starts)[win]) // tubelet
    if (pos < 0).any() or (pos >= window // tubelet).any():
        raise ValueError("a tubelet falls outside its window")
    return win, pos


def version_frames(
    version: str, clip: str, starts: tuple[int, ...] = VJ_STARTS, window: int = VJ_WINDOW
) -> np.ndarray:
    """Frame indices (windows x window) into the decoded clip for one version: ``video`` in
    order, ``shuffled`` the clip's frames permuted once (seeded by its name), ``still`` the
    window's middle frame repeated."""
    from wave_analysis.models.video_probe import arm_order

    n = starts[-1] + window
    if version == "video":
        frames = np.arange(n)
    elif version == "shuffled":
        frames = arm_order("shuffled", n, clip)
    elif version == "still":
        return np.stack([np.full(window, s + window // 2) for s in starts])
    else:
        raise ValueError(f"unknown version {version!r}")
    return np.stack([frames[s : s + window] for s in starts])


class VJepa2:
    """Frozen V-JEPA 2 ViT-L (Hugging Face ``facebook/vjepa2-vitl-fpc64-256``): the encoder
    output after its final norm, averaged over the spatial grid of each tubelet."""

    def __init__(self, device: str, model: torch.nn.Module | None = None) -> None:
        self.device = device
        dtype = torch.bfloat16 if device != "cpu" else torch.float32
        if model is None:  # a model may be given in tests
            from transformers import VJEPA2Model

            model = VJEPA2Model.from_pretrained(VJEPA2, torch_dtype=dtype)
        self.model = model.to(device=device, dtype=dtype).eval().requires_grad_(False)
        cfg = self.model.config
        self.tubelet, self.patch = int(cfg.tubelet_size), int(cfg.patch_size)
        if cfg.crop_size != VJ_SIDE or cfg.frames_per_clip != VJ_WINDOW:
            raise ValueError(f"{VJEPA2}: crop {cfg.crop_size}, {cfg.frames_per_clip} frames")
        self.dim, self.dtype = int(cfg.hidden_size), dtype
        self.mean = torch.tensor(MEAN, device=device).view(1, 1, 3, 1, 1)
        self.std = torch.tensor(STD, device=device).view(1, 1, 3, 1, 1)

    @torch.inference_mode()
    def tokens(self, x: torch.Tensor) -> torch.Tensor:
        """uint8 windows (W, T, 3, H, W) -> spatially averaged tokens (W, T / tubelet, D)."""
        from wave_analysis.models.video_probe import pool_tokens

        w, t = x.shape[:2]
        v = x.to(self.device, non_blocking=True).flatten(0, 1).float()
        v = F.interpolate(v, size=(VJ_SIDE, VJ_SIDE), mode="bilinear", align_corners=False)
        v = ((v.view(w, t, 3, VJ_SIDE, VJ_SIDE) / 255 - self.mean) / self.std).to(self.dtype)
        out = self.model(pixel_values_videos=v, skip_predictor=True).last_hidden_state
        grid = (t // self.tubelet, VJ_SIDE // self.patch, VJ_SIDE // self.patch)
        return pool_tokens(out, grid, (grid[0], 1, 1))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def weight_files(encoders: str = "base") -> dict[str, str]:
    """SHA-256 of the cached weights actually loaded (torch hub and Hugging Face caches)."""
    out = {}
    hf = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    if encoders == "vjepa2":
        repo = "models--" + VJEPA2.replace("/", "--")
        for p in sorted(hf.glob(f"{repo}/snapshots/*/*.safetensors")):
            out[f"{VJEPA2}/{p.parent.name[:12]}/{p.name}"] = sha256(p)
        return out
    hub = Path(torch.hub.get_dir()) / "checkpoints"
    for p in sorted(hub.glob("resnet18-*.pth")):
        out[p.name] = sha256(p)
    for p in sorted(hf.glob(f"models--timm--{DINOV3}/snapshots/*/*.safetensors")):
        out[f"{DINOV3}/{p.parent.name[:12]}/{p.name}"] = sha256(p)
    return out


def run(args: argparse.Namespace) -> None:
    t0 = time.time()
    split = official_split(args.source / "loso_split_by_database_filename.csv")
    clips = clip_table(split)
    if args.limit:
        clips = clips.groupby("site").head(max(1, args.limit // 6)).reset_index(drop=True)
    n = len(clips)
    args.out.mkdir(parents=True, exist_ok=False)
    if args.encoders == "vjepa2":
        run_vjepa2(args, clips, t0)
        return
    with zipfile.ZipFile(args.source / "Set_B.zip") as z:
        b_members, b_found = set_b_members(z.namelist(), clips)
    clips["b_frames"] = [" ".join(Path(m).name for m in ms) for ms in b_members]
    clips["b_frames_in_folder"] = b_found
    log(
        f"{n:,} clips; Set B frames found for all; {torch.cuda.get_device_name(0) if args.device == 'cuda' else 'cpu'}"
    )

    enc = Encoders(args.device, dinov3=True)
    mm = {
        "a_dinov3": np.lib.format.open_memmap(
            args.out / "a_dinov3.npy", "w+", np.float16, (n, A_FRAMES, 1536)
        ),
        "a_resnet18": np.lib.format.open_memmap(
            args.out / "a_resnet18.npy", "w+", np.float16, (n, A_FRAMES, 512)
        ),
        "b_resnet18": np.lib.format.open_memmap(
            args.out / "b_resnet18.npy", "w+", np.float16, (n, B_FRAMES, 512)
        ),
    }
    kw = {
        "batch_size": args.batch,
        "num_workers": args.workers,
        "collate_fn": collate,
        "pin_memory": True,
        "persistent_workers": False,
        "prefetch_factor": 4 if args.workers else None,
    }
    a_frames = np.zeros(n, int)
    a_fps = np.zeros(n)
    with tempfile.TemporaryDirectory(dir=args.tmp) as tmpdir:
        loader = torch.utils.data.DataLoader(
            SetA(args.source / "Set_A.zip", list(clips.member_a), tmpdir), **kw
        )
        for b, (x, idx, nf, fps) in enumerate(loader):
            flat = x.flatten(0, 1)
            k = idx.numpy()
            mm["a_resnet18"][k] = enc.resnet18(flat).view(len(k), A_FRAMES, -1).half().cpu().numpy()
            mm["a_dinov3"][k] = enc.dino(flat).view(len(k), A_FRAMES, -1).half().cpu().numpy()
            a_frames[k], a_fps[k] = nf.numpy(), fps.numpy()
            if b % 100 == 0:
                log(
                    f"Set A: {min((b + 1) * args.batch, n):,}/{n:,} clips, {time.time() - t0:.0f} s"
                )
    loader = torch.utils.data.DataLoader(SetB(args.source / "Set_B.zip", b_members), **kw)
    b_frames = np.zeros(n, int)
    for b, (x, idx, nf, _) in enumerate(loader):
        k = idx.numpy()
        mm["b_resnet18"][k] = (
            enc.resnet18(x.flatten(0, 1)).view(len(k), B_FRAMES, -1).half().cpu().numpy()
        )
        b_frames[k] = nf.numpy()
        if b % 200 == 0:
            log(f"Set B: {min((b + 1) * args.batch, n):,}/{n:,} clips, {time.time() - t0:.0f} s")
    for m in mm.values():
        m.flush()
    clips["a_frames_decoded"], clips["a_fps"], clips["b_frames_read"] = a_frames, a_fps, b_frames
    clips.to_csv(args.out / "clips.csv", index=False)
    finite = {
        k: bool(np.isfinite(np.asarray(v[:: max(1, n // 200)], np.float32)).all())
        for k, v in mm.items()
    }
    meta = {
        "clips": n,
        "set_a": {
            "frames": A_FRAMES,
            "fps_values": sorted(set(np.round(a_fps, 3).tolist())),
            "frames_decoded_values": sorted(set(a_frames.tolist())),
        },
        "set_b": {
            "frames": B_FRAMES,
            "frames_read_values": sorted(set(b_frames.tolist())),
            "folders_with_more_frames": clips.filename[
                clips.b_frames_in_folder > B_FRAMES
            ].tolist(),
        },
        "encoders": {
            "a_dinov3": f"timm {DINOV3}, frozen, bf16, input 224 x 224, [CLS | mean of patch tokens] after the final norm",
            "a_resnet18": "torchvision resnet18 IMAGENET1K_V1, frozen, fp32, eval-mode batch norm, global average pool",
            "b_resnet18": "the same ResNet-18 on Set B's 60 JPEG frames",
        },
        "weights_sha256": weight_files("base"),
        "inputs": {
            p: (args.source / p).stat().st_size
            for p in ("Set_A.zip", "Set_B.zip", "loso_split_by_database_filename.csv")
        },
        "finite_sample": finite,
        "seconds": round(time.time() - t0),
        "commit": os.environ.get("COMMIT", ""),
        "limit": args.limit,
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    log(
        f"features written to {args.out}: {json.dumps(meta['set_a'])}, Set B {meta['set_b']['frames_read_values']}"
    )


def run_vjepa2(args: argparse.Namespace, clips: pd.DataFrame, t0: float) -> None:
    """Bar Y-B3's pass: every Set A clip decoded once, three versions through V-JEPA 2."""
    n = len(clips)
    log(f"{n:,} clips; {torch.cuda.get_device_name(0) if args.device == 'cuda' else 'cpu'}")
    enc = VJepa2(args.device)
    win, pos = stitch_plan(tubelet=enc.tubelet)
    n_tok = len(win)
    mm = {
        v: np.lib.format.open_memmap(
            args.out / f"a_vjepa2_{v}.npy", "w+", np.float16, (n, n_tok, enc.dim)
        )
        for v in VJ_VERSIONS
    }
    a_frames, a_fps = np.zeros(n, int), np.zeros(n)
    win_t, pos_t = torch.as_tensor(win), torch.as_tensor(pos)
    with tempfile.TemporaryDirectory(dir=args.tmp) as tmpdir:
        loader = torch.utils.data.DataLoader(
            SetA(args.source / "Set_A.zip", list(clips.member_a), tmpdir),
            batch_size=args.batch,
            num_workers=args.workers,
            collate_fn=collate,
            pin_memory=True,
            prefetch_factor=4 if args.workers else None,
        )
        for b, (x, idx, nf, fps) in enumerate(loader):
            k = idx.numpy()
            x = x.to(args.device, non_blocking=True)  # (B, 150, 3, 224, 224) uint8
            for v in VJ_VERSIONS:
                frames = torch.as_tensor(
                    np.stack([version_frames(v, clips.filename.iloc[i]) for i in k]),
                    device=x.device,
                )  # (B, windows, 64)
                xw = torch.stack([x[j][frames[j].flatten()] for j in range(len(k))])
                xw = xw.view(len(k) * len(VJ_STARTS), VJ_WINDOW, *x.shape[2:])
                tok = enc.tokens(xw).view(len(k), len(VJ_STARTS), -1, enc.dim)
                mm[v][k] = tok[:, win_t, pos_t].half().cpu().numpy()
            a_frames[k], a_fps[k] = nf.numpy(), fps.numpy()
            if b % 100 == 0:
                log(
                    f"V-JEPA 2: {min((b + 1) * args.batch, n):,}/{n:,} clips, {time.time() - t0:.0f} s"
                )
    for m in mm.values():
        m.flush()
    clips["a_frames_decoded"], clips["a_fps"] = a_frames, a_fps
    clips.to_csv(args.out / "clips.csv", index=False)
    finite = {
        k: bool(np.isfinite(np.asarray(v[:: max(1, n // 200)], np.float32)).all())
        for k, v in mm.items()
    }
    meta = {
        "clips": n,
        "set_a": {
            "frames": A_FRAMES,
            "fps_values": sorted(set(np.round(a_fps, 3).tolist())),
            "frames_decoded_values": sorted(set(a_frames.tolist())),
        },
        "encoders": {
            f"a_vjepa2_{v}": f"HF {VJEPA2}, frozen, bf16, {VJ_SIDE} x {VJ_SIDE} (bilinear from 224), "
            "ImageNet mean/std, last_hidden_state averaged over the spatial grid"
            for v in VJ_VERSIONS
        },
        "versions": {
            "video": "the 150 frames in order",
            "shuffled": "the 150 frames in a fixed random order per clip "
            "(wave_analysis.models.video_probe.arm_order, seeded by the file name)",
            "still": "each window's middle frame repeated",
        },
        "windows": {"starts": list(VJ_STARTS), "frames": VJ_WINDOW, "tubelet": enc.tubelet},
        "stitch": {
            "window_of_tubelet": win.tolist(),
            "position_in_window": pos.tolist(),
            "tubelets_per_window": np.bincount(win).tolist(),
        },
        "weights_sha256": weight_files("vjepa2"),
        "inputs": {
            p: (args.source / p).stat().st_size
            for p in ("Set_A.zip", "loso_split_by_database_filename.csv")
        },
        "finite_sample": finite,
        "seconds": round(time.time() - t0),
        "commit": os.environ.get("COMMIT", ""),
        "limit": args.limit,
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    log(f"V-JEPA 2 features written to {args.out}: {json.dumps(meta['set_a'])}, finite {finite}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--source",
        type=Path,
        required=True,
        help="folder with Set_A.zip, Set_B.zip and the split CSV",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=4, help="clips per batch (150 frames each)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument(
        "--tmp", default=None, help="folder for the per-clip MP4 written for the decoder"
    )
    ap.add_argument("--limit", type=int, default=0, help="a smoke test on about this many clips")
    ap.add_argument(
        "--encoders",
        choices=["base", "vjepa2"],
        default="base",
        help="base: ResNet-18 and DINOv3 (Y-B1, Y-B2); vjepa2: V-JEPA 2 in three versions (Y-B3)",
    )
    run(ap.parse_args())


if __name__ == "__main__":
    main()
