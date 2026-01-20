from __future__ import annotations
import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import yaml

from src.utils.seed import set_seed
from src.datasets.brats_roi_npz import BratsRoiNpzDataset, AugmentCfg
from src.models.unet3d_res import ResUNet3D
from src.metrics.brats_composites import compute_all_metrics


def now_tag():
    return time.strftime("%Y%m%d_%H%M%S")


def make_run_dir(stage: str, run_name: str):
    run_dir = Path("runs") / stage / run_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def soft_dice_loss(logits, target, num_classes: int, class_weights=None, eps=1e-6):
    """
    logits: (B,C,D,H,W)
    target: (B,D,H,W) long
    """
    probs = torch.softmax(logits, dim=1)
    target_oh = F.one_hot(target, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()

    # exclude background from dice by default, but keep it in weight list if provided
    dice_per_class = []
    for c in range(1, num_classes):
        p = probs[:, c]
        g = target_oh[:, c]
        inter = (p * g).sum(dim=(1, 2, 3))
        den = p.sum(dim=(1, 2, 3)) + g.sum(dim=(1, 2, 3))
        d = (2 * inter + eps) / (den + eps)  # (B,)
        dice_per_class.append(d)

    dice = torch.stack(dice_per_class, dim=1)  # (B, C-1)
    if class_weights is not None:
        w = torch.tensor(class_weights[1:], device=logits.device).float().view(1, -1)
        loss = (1.0 - dice) * w
        return loss.mean()
    return (1.0 - dice).mean()


def focal_ce_loss(logits, target, gamma=2.0, class_weights=None):
    """
    logits: (B,C,D,H,W), target: (B,D,H,W)
    """
    ce = F.cross_entropy(logits, target, weight=None, reduction="none")  # (B,D,H,W)
    pt = torch.exp(-ce)
    focal = (1 - pt) ** gamma * ce

    if class_weights is not None:
        w = torch.tensor(class_weights, device=logits.device).float()
        w_map = w[target]
        focal = focal * w_map

    return focal.mean()


def center_crop_with_pad(vol: np.ndarray, center_zyx, roi_size: int, pad_val: float = 0.0):
    """
    vol: (C,D,H,W) or (D,H,W)
    returns:
      crop: (C,roi,roi,roi) or (roi,roi,roi)
    """
    is_4d = (vol.ndim == 4)
    if is_4d:
        C, D, H, W = vol.shape
    else:
        D, H, W = vol.shape

    cz, cy, cx = center_zyx
    half = roi_size // 2
    sz, ez = cz - half, cz - half + roi_size
    sy, ey = cy - half, cy - half + roi_size
    sx, ex = cx - half, cx - half + roi_size

    fz0, fz1 = max(0, sz), min(D, ez)
    fy0, fy1 = max(0, sy), min(H, ey)
    fx0, fx1 = max(0, sx), min(W, ex)

    pz0, pz1 = fz0 - sz, (fz0 - sz) + (fz1 - fz0)
    py0, py1 = fy0 - sy, (fy0 - sy) + (fy1 - fy0)
    px0, px1 = fx0 - sx, (fx0 - sx) + (fx1 - fx0)

    if is_4d:
        crop = np.full((C, roi_size, roi_size, roi_size), pad_val, dtype=vol.dtype)
        crop[:, pz0:pz1, py0:py1, px0:px1] = vol[:, fz0:fz1, fy0:fy1, fx0:fx1]
    else:
        crop = np.full((roi_size, roi_size, roi_size), pad_val, dtype=vol.dtype)
        crop[pz0:pz1, py0:py1, px0:px1] = vol[fz0:fz1, fy0:fy1, fx0:fx1]

    return crop


def sample_center_fullbrain(seg_zyx: np.ndarray, fg_ratio: float = 0.5) -> Tuple[int, int, int]:
    """
    seg_zyx: (D,H,W) uint8
    fg_ratio: probability to sample a voxel from foreground (seg>0). fallback to uniform random.
    """
    D, H, W = seg_zyx.shape
    if (np.random.rand() < fg_ratio) and (seg_zyx > 0).any():
        zz, yy, xx = np.where(seg_zyx > 0)
        j = np.random.randint(len(zz))
        return int(zz[j]), int(yy[j]), int(xx[j])
    return int(np.random.randint(D)), int(np.random.randint(H)), int(np.random.randint(W))


def apply_patch_aug(img: np.ndarray, seg: np.ndarray, aug_cfg: AugmentCfg):
    """
    img: (C,D,H,W), seg: (D,H,W)
    use your yaml keys: flip_prob, rot90_prob, intensity_shift, intensity_scale, noise_std
    """
    if not getattr(aug_cfg, "enable", False):
        return img, seg

    # flip
    p = float(getattr(aug_cfg, "flip_prob", 0.5))
    if np.random.rand() < p:
        # flip each axis with 50% prob
        for ax in [1, 2, 3]:
            if np.random.rand() < 0.5:
                img = np.flip(img, axis=ax).copy()
                seg = np.flip(seg, axis=ax - 1).copy()

    # rot90 (rotate in (H,W) plane; you can extend to other planes if you want)
    rp = float(getattr(aug_cfg, "rot90_prob", 0.25))
    if np.random.rand() < rp:
        k = np.random.randint(1, 4)
        # img: (C,D,H,W) rotate H,W
        img = np.rot90(img, k=k, axes=(2, 3)).copy()
        seg = np.rot90(seg, k=k, axes=(1, 2)).copy()

    # intensity
    shift = float(getattr(aug_cfg, "intensity_shift", 0.10))
    scale = float(getattr(aug_cfg, "intensity_scale", 0.10))
    if shift > 0 or scale > 0:
        s = 1.0 + np.random.uniform(-scale, scale)
        b = np.random.uniform(-shift, shift)
        img = img * s + b

    # noise
    ns = float(getattr(aug_cfg, "noise_std", 0.05))
    if ns and ns > 0:
        img = img + np.random.normal(0.0, ns, size=img.shape).astype(img.dtype)

    return img, seg


def apply_simple_aug(img: np.ndarray, seg: np.ndarray, aug_cfg: AugmentCfg):
    """
    Very lightweight augmentation for fullbrain baseline. Uses attributes if present in AugmentCfg.
    img: (C,D,H,W) float32
    seg: (D,H,W) uint8
    """
    if not getattr(aug_cfg, "enable", False):
        return img, seg

    # flips
    p_flip = float(getattr(aug_cfg, "p_flip", 0.5))
    if np.random.rand() < p_flip:
        # flip any subset of axes
        for ax in [1, 2, 3]:  # D,H,W in img
            if np.random.rand() < 0.5:
                img = np.flip(img, axis=ax).copy()
                seg = np.flip(seg, axis=ax - 1).copy()

    # intensity scale/shift (very mild)
    p_int = float(getattr(aug_cfg, "p_intensity", 0.2))
    if np.random.rand() < p_int:
        scale = float(getattr(aug_cfg, "intensity_scale", 0.1))
        shift = float(getattr(aug_cfg, "intensity_shift", 0.1))
        s = 1.0 + np.random.uniform(-scale, scale)
        b = np.random.uniform(-shift, shift)
        img = img * s + b

    # gaussian noise
    p_noise = float(getattr(aug_cfg, "p_noise", 0.15))
    if np.random.rand() < p_noise:
        std = float(getattr(aug_cfg, "noise_std", 0.01))
        img = img + np.random.normal(0.0, std, size=img.shape).astype(img.dtype)

    return img, seg


class BratsFullbrainPatchDataset(Dataset):
    """
    1-stage fullbrain baseline dataset:
      reads full volumes from npy_full_v1/{train,val}/*.npz
      samples random centers (foreground-oversampling) and returns 128^3 patches
    npz format expected:
      img: (4,D,H,W) float32
      seg: (D,H,W) uint8 in {0..4}
    """
    def __init__(
        self,
        full_root: str,
        split: str,
        splits_dir: str,
        roi_size: int = 128,
        cache_cases: int = 8,
        fg_ratio: float = 0.5,
        samples_per_case: int = 4,
        aug_cfg: Optional[AugmentCfg] = None,
    ):
        self.full_root = Path(full_root)
        self.split = split
        self.roi_size = int(roi_size)
        self.fg_ratio = float(fg_ratio)
        self.samples_per_case = int(samples_per_case)
        self.aug_cfg = aug_cfg if aug_cfg is not None else AugmentCfg(enable=False)

        split_file = Path(splits_dir) / f"{split}.txt"
        self.case_ids = [l.strip() for l in split_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        if len(self.case_ids) == 0:
            raise RuntimeError(f"Empty split list: {split_file}")

        # Support either:
        #   full_root/train/*.npz
        # or full_root itself already points to split folder
        candidate = self.full_root / split
        self.split_root = candidate if candidate.is_dir() else self.full_root
        self.cache_cases = int(cache_cases)
        self._cache = {}
        self._cache_order = []

    def __len__(self):
        # more steps per epoch than number of cases (like nnU-Net style sampling)
        return len(self.case_ids) * max(1, self.samples_per_case)

    def _case_id_from_index(self, idx: int) -> str:
        # map idx to a case id
        return self.case_ids[idx % len(self.case_ids)]

    def __getitem__(self, idx: int):
        cid = self._case_id_from_index(idx)
        npz_path = self.split_root / f"{cid}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"missing npz: {npz_path}")

        if cid in self._cache:
            img, seg = self._cache[cid]
        else:
            d = np.load(npz_path)
            img = d["img"].astype(np.float32)   # (4,D,H,W)
            seg = d["seg"].astype(np.uint8)     # (D,H,W)

        if self.cache_cases > 0:
            self._cache[cid] = (img, seg)
            self._cache_order.append(cid)
            if len(self._cache_order) > self.cache_cases:
                old = self._cache_order.pop(0)
                self._cache.pop(old, None)

        
        center = sample_center_fullbrain(seg, self.fg_ratio)
        crop_img = center_crop_with_pad(img, center, self.roi_size, pad_val=0.0)
        crop_seg = center_crop_with_pad(seg, center, self.roi_size, pad_val=0)

        # patch-level aug (fast)
        crop_img, crop_seg = apply_patch_aug(crop_img, crop_seg, self.aug_cfg)

       
        # torch tensors
        crop_img_t = torch.from_numpy(crop_img).float()
        crop_seg_t = torch.from_numpy(crop_seg).long()

        return {"image": crop_img_t, "label": crop_seg_t, "case_id": cid}


def train_one_epoch(model, loader, opt, scaler, cfg, device):
    model.train()
    total = 0.0
    n = 0
    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        seg = batch["label"].to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=cfg["train"]["amp"]):
            logits = model(img)
            dice = soft_dice_loss(
                logits, seg, num_classes=cfg["model"]["out_channels"],
                class_weights=cfg["loss"]["class_weights"]
            ) * cfg["loss"]["dice_weight"]
            focal = focal_ce_loss(
                logits, seg, gamma=cfg["loss"]["focal_gamma"],
                class_weights=cfg["loss"]["class_weights"]
            ) * cfg["loss"]["focal_weight"]
            loss = dice + focal

        scaler.scale(loss).backward()
        if cfg["train"]["grad_clip"] and cfg["train"]["grad_clip"] > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
        scaler.step(opt)
        scaler.update()

        total += float(loss.item())
        n += 1
    return total / max(1, n)


@torch.no_grad()
def validate(model, loader, cfg, device):
    model.eval()
    agg = None
    n = 0
    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        seg = batch["label"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=cfg["train"]["amp"]):
            logits = model(img)

        m = compute_all_metrics(logits, seg)
        if agg is None:
            agg = {k: 0.0 for k in m.keys()}
        for k, v in m.items():
            agg[k] += float(v)
        n += 1

    for k in agg.keys():
        agg[k] /= max(1, n)
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    set_seed(int(cfg["seed"]))

    # mode switch: ROI (default) vs fullbrain baseline
    mode = str(cfg.get("data", {}).get("mode", "roi")).lower()
    is_fullbrain = (mode in ["fullbrain", "full_brain", "full"])

    run_name = cfg["run"]["name"]
    if run_name == "auto":
        tag = "fullbrain" if is_fullbrain else "roi"
        run_name = f"{now_tag()}_{tag}128_bs{cfg['data']['batch_size']}_lr{cfg['train']['lr']}"
    # route runs to separate folder for cleanliness
    stage_folder = "stage2_fullbrain" if is_fullbrain else "stage2"
    run_dir = make_run_dir(stage_folder, run_name)

    # save config snapshot
    (run_dir / "config.yaml").write_text(Path(args.config).read_text(encoding="utf-8"), encoding="utf-8")
    save_json(run_dir / "meta.json", {"run_name": run_name, "created": now_tag(), "config": args.config, "mode": mode})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    aug_cfg = AugmentCfg(**cfg["augment"])

    if not is_fullbrain:
        # ---------- ROI training (original behavior) ----------
        train_ds = BratsRoiNpzDataset(cfg["data"]["roi_root"], "train", cfg["data"]["splits_dir"], aug_cfg)
        val_ds   = BratsRoiNpzDataset(cfg["data"]["roi_root"], "val", cfg["data"]["splits_dir"], AugmentCfg(enable=False))
    else:
        # ---------- 1-stage fullbrain baseline ----------
        # Expect cfg["data"]["full_root"] = "data/preprocessed/npy_full_v1"
        full_root = cfg["data"].get("full_root", "data/preprocessed/npy_full_v1")
        fg_ratio = float(cfg["data"].get("fg_ratio", 0.5))
        spc = int(cfg["data"].get("samples_per_case", 4))

        cache_cases = int(cfg["data"].get("cache_cases", 8))
        train_ds = BratsFullbrainPatchDataset(
            full_root=full_root,
            split="train",
            splits_dir=cfg["data"]["splits_dir"],
            roi_size=int(cfg["data"].get("roi_size", 128)),
            fg_ratio=fg_ratio,
            samples_per_case=spc,
            cache_cases=cache_cases,
            aug_cfg=aug_cfg,
        )
        val_ds = BratsFullbrainPatchDataset(
            full_root=full_root,
            split="val",
            splits_dir=cfg["data"]["splits_dir"],
            roi_size=int(cfg["data"].get("roi_size", 128)),
            fg_ratio=float(cfg["data"].get("val_fg_ratio", fg_ratio)),
            samples_per_case=max(1, int(cfg["data"].get("val_samples_per_case", 1))),
            cache_cases=0,
            aug_cfg=AugmentCfg(enable=False),
        )

    train_loader = DataLoader(
        train_ds, batch_size=cfg["data"]["batch_size"], shuffle=True,
        num_workers=cfg["data"]["num_workers"], pin_memory=cfg["data"]["pin_memory"],
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=max(1, cfg["data"]["num_workers"] // 2), pin_memory=cfg["data"]["pin_memory"]
    )

    model = ResUNet3D(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        base=cfg["model"]["base_channels"],
        dropout=cfg["model"]["dropout"]
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"])
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["train"]["amp"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(cfg["train"]["epochs"]))

    metrics_csv = run_dir / "metrics" / "metrics_epoch.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "lr", "train_loss",
                    "WT", "TC", "ET", "RC", "mean_wt_tc_et",
                    "dice_1_netc", "dice_2_snf_h", "dice_3_et", "dice_4_rc"])

    best = -1.0
    best_epoch = -1
    no_improve = 0

    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        t0 = time.time()
        lr = opt.param_groups[0]["lr"]
        tr_loss = train_one_epoch(model, train_loader, opt, scaler, cfg, device)
        sched.step()

        do_val = (epoch % int(cfg["train"]["val_every"]) == 0)
        if do_val:
            val_m = validate(model, val_loader, cfg, device)
            main_key = cfg["metrics"]["main"]
            score = float(val_m[main_key])

            # save best
            if score > best:
                best = score
                best_epoch = epoch
                no_improve = 0
                ckpt = {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "config": cfg,
                    "best_score": best,
                }
                torch.save(ckpt, run_dir / "checkpoints" / "best.pt")
            else:
                no_improve += 1

            with metrics_csv.open("a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    epoch, lr, tr_loss,
                    val_m["WT"], val_m["TC"], val_m["ET"], val_m["RC"], val_m["mean_wt_tc_et"],
                    val_m["dice_1_netc"], val_m["dice_2_snf_h"], val_m["dice_3_et"], val_m["dice_4_rc"]
                ])

            dt = time.time() - t0
            print(f"[E{epoch:03d}] loss={tr_loss:.4f}  {main_key}={score:.4f}  best={best:.4f}@{best_epoch}  time={dt:.1f}s")

            # early stop
            if cfg["train"]["early_stop_patience"] and no_improve >= int(cfg["train"]["early_stop_patience"]):
                print(f"[EARLY STOP] no improve for {no_improve} evals. best={best:.4f}@{best_epoch}")
                break
        else:
            dt = time.time() - t0
            print(f"[E{epoch:03d}] loss={tr_loss:.4f}  (no val)  time={dt:.1f}s")

    summary = {
        "run_name": run_name,
        "mode": mode,
        "best_score": best,
        "best_epoch": best_epoch,
        "main_metric": cfg["metrics"]["main"],
        "best_ckpt": str(run_dir / "checkpoints" / "best.pt"),
        "metrics_csv": str(metrics_csv),
    }
    save_json(run_dir / "metrics" / "summary.json", summary)

    summary_csv = run_dir / "metrics" / "summary_val.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run_name", "mode", "best_epoch", "best_score", "main_metric", "best_ckpt"])
        w.writerow([run_name, mode, best_epoch, best, cfg["metrics"]["main"], str(run_dir / "checkpoints" / "best.pt")])

    print("[OK] training finished")
    print("run_dir    :", run_dir)
    print("best_ckpt  :", run_dir / "checkpoints" / "best.pt")
    print("metrics_csv:", metrics_csv)
    print("summary    :", summary_csv)


if __name__ == "__main__":
    main()
