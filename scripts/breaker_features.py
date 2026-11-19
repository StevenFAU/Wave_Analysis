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

Each clip is read from its zip once; nothing is written to disk but the features.

    python scripts/breaker_features.py --source <dir with the three files> --out <dir> \\
        --device cuda [--limit 16]
"""

from __future__ import annotations

import argparse
import hashlib
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

CLASSES = ["Spilling", "Plunging", "Surging"]
A_FRAMES = 150  # 30 s at 5 fps
B_FRAMES = 60  # 30 s at 2 Hz
SIDE = 224
MEAN = (0.485, 0.456, 0.406)  # ImageNet; also DINOv3's (timm pretrained_cfg)
STD = (0.229, 0.224, 0.225)
DINOV3 = "vit_base_patch16_dinov3.lvd1689m"
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


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def weight_files() -> dict[str, str]:
    """SHA-256 of the cached weights actually loaded (torch hub and Hugging Face caches)."""
    out = {}
    hub = Path(torch.hub.get_dir()) / "checkpoints"
    for p in sorted(hub.glob("resnet18-*.pth")):
        out[p.name] = sha256(p)
    hf = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
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
        "weights_sha256": weight_files(),
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
    run(ap.parse_args())


if __name__ == "__main__":
    main()
