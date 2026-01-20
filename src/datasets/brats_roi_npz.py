from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import Dataset

@dataclass
class AugmentCfg:
    enable: bool = True
    flip_prob: float = 0.5
    rot90_prob: float = 0.25
    intensity_shift: float = 0.10
    intensity_scale: float = 0.10
    noise_std: float = 0.05
    translate_max: int = 0
    translate_prob: float = 0.0

def _rand_flip(img, seg, p=0.5):
    # img: (C,D,H,W), seg: (D,H,W)
    for dim in [1,2,3]:  # flip over D/H/W
        if random.random() < p:
            img = torch.flip(img, dims=[dim])
            seg = torch.flip(seg, dims=[dim-1])  # seg dims [0,1,2] correspond to img dims [1,2,3]
    return img, seg

def _rand_rot90(img, seg, p=0.25):
    # rotate by k*90 in a random plane
    if random.random() >= p:
        return img, seg
    k = random.randint(0, 3)
    plane = random.choice([(1,2), (1,3), (2,3)])  # on img dims
    img = torch.rot90(img, k=k, dims=plane)
    # seg dims are one less: (D,H,W) -> map plane dims minus 1
    seg_plane = (plane[0]-1, plane[1]-1)
    seg = torch.rot90(seg, k=k, dims=seg_plane)
    return img, seg

def _rand_translate(img, seg, max_shift=8, p=0.5):
    # zero-padded translation; img:(C,D,H,W), seg:(D,H,W)
    if max_shift <= 0 or random.random() >= p:
        return img, seg

    C, D, H, W = img.shape
    sd = random.randint(-max_shift, max_shift)
    sh = random.randint(-max_shift, max_shift)
    sw = random.randint(-max_shift, max_shift)

    out_img = torch.zeros_like(img)
    out_seg = torch.zeros_like(seg)

    def _copy_1d(L, s):
        src0 = max(0, -s)
        dst0 = max(0, s)
        ln = L - abs(s)
        return src0, dst0, ln

    d_src, d_dst, d_ln = _copy_1d(D, sd)
    h_src, h_dst, h_ln = _copy_1d(H, sh)
    w_src, w_dst, w_ln = _copy_1d(W, sw)

    if d_ln <= 0 or h_ln <= 0 or w_ln <= 0:
        return img, seg

    out_img[:, d_dst:d_dst+d_ln, h_dst:h_dst+h_ln, w_dst:w_dst+w_ln] = \
        img[:, d_src:d_src+d_ln, h_src:h_src+h_ln, w_src:w_src+w_ln]
    out_seg[d_dst:d_dst+d_ln, h_dst:h_dst+h_ln, w_dst:w_dst+w_ln] = \
        seg[d_src:d_src+d_ln, h_src:h_src+h_ln, w_src:w_src+w_ln]

    return out_img, out_seg

def _intensity_aug(img, shift=0.10, scale=0.10):
    # per-sample, per-channel affine intensity
    if shift > 0:
        s = (torch.rand((img.shape[0],1,1,1), device=img.device) * 2 - 1) * shift
        img = img + s
    if scale > 0:
        a = 1.0 + (torch.rand((img.shape[0],1,1,1), device=img.device) * 2 - 1) * scale
        img = img * a
    return img

def _noise(img, std=0.05):
    if std <= 0:
        return img
    n = torch.randn_like(img) * std
    return img + n

class BratsRoiNpzDataset(Dataset):
    def __init__(self, roi_root: str, split: str, splits_dir: str, augment: AugmentCfg):
        self.roi_root = Path(roi_root)
        self.split = split
        self.augment = augment

        ids_path = Path(splits_dir) / f"{split}.txt"
        ids = [l.strip() for l in ids_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        self.files = [self.roi_root / split / f"{cid}.npz" for cid in ids]

        missing = [p for p in self.files if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} npz files, e.g. {missing[:3]}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        fp = self.files[idx]
        d = np.load(fp)
        img = torch.from_numpy(d["img"]).float()        # (4,128,128,128)
        seg = torch.from_numpy(d["seg"]).long()         # (128,128,128)

        if self.split == "train" and self.augment.enable:
            img, seg = _rand_flip(img, seg, p=self.augment.flip_prob)
            img, seg = _rand_rot90(img, seg, p=self.augment.rot90_prob)
            img, seg = _rand_translate(img, seg, max_shift=self.augment.translate_max, p=self.augment.translate_prob)
            img = _intensity_aug(img, shift=self.augment.intensity_shift, scale=self.augment.intensity_scale)
            img = _noise(img, std=self.augment.noise_std)

        return {"image": img, "label": seg, "case_id": fp.stem}
