import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

from src.datasets.brats_roi_npz import BratsRoiNpzDataset, AugmentCfg
from src.models.unet3d_res import ResUNet3D
from src.metrics.brats_composites import compute_all_metrics

try:
    from scipy.ndimage import label as cc_label
except Exception:
    cc_label = None


def _remove_small_cc(mask: np.ndarray, min_vox: int) -> np.ndarray:
    """mask: bool (D,H,W)"""
    if min_vox <= 0:
        return mask
    if cc_label is None:
        raise RuntimeError("scipy not available; please `pip install scipy` to use RC-only postprocess.")
    lab, n = cc_label(mask.astype(np.uint8))
    if n == 0:
        return mask
    out = np.zeros_like(mask, dtype=bool)
    for i in range(1, n + 1):
        comp = (lab == i)
        if int(comp.sum()) >= int(min_vox):
            out |= comp
    return out


def _postprocess_rc_only(pred_lbl: np.ndarray, rc_min_vox: int) -> np.ndarray:
    """pred_lbl: uint8 labels (D,H,W) in 0..4"""
    if rc_min_vox <= 0:
        return pred_lbl
    out = pred_lbl.copy()
    rc = (out == 4)
    rc2 = _remove_small_cc(rc, rc_min_vox)
    out[rc & (~rc2)] = 0
    out[rc2] = 4
    return out


def _tta_transforms(mode: str):
    """
    Return list of (flip_dims, rot_k) where:
    - flip_dims: tuple of spatial dims in (D,H,W) => mapped to tensor dims (2,3,4)
    - rot_k: rotation k for rot90 around (H,W) plane; k in {0,1,2,3}
    """
    flips = [
        (),            # none
        (2,),          # D
        (3,),          # H
        (4,),          # W
        (2,3),
        (2,4),
        (3,4),
        (2,3,4),
    ]
    if mode == "none":
        return [((), 0)]
    if mode == "flip":
        return [(fd, 0) for fd in flips]
    if mode == "flip_rot90hw":
        out = []
        for k in (0,1,2,3):
            for fd in flips:
                out.append((fd, k))
        return out
    raise ValueError(f"Unknown TTA mode: {mode}")


@torch.no_grad()
def predict_probs_tta(model, x, tta_mode: str):
    """
    x: (B,4,D,H,W)
    returns probs: (B,5,D,H,W)
    """
    B = x.shape[0]
    acc = None
    t_list = _tta_transforms(tta_mode)
    for flip_dims, rot_k in t_list:
        xt = x
        # rot90 around (H,W) => dims (3,4) in x
        if rot_k != 0:
            xt = torch.rot90(xt, k=rot_k, dims=(3,4))
        if flip_dims:
            xt = torch.flip(xt, dims=flip_dims)

        logits = model(xt)
        probs = F.softmax(logits, dim=1)

        # invert transforms on probs (B,C,D,H,W)
        if flip_dims:
            probs = torch.flip(probs, dims=flip_dims)
        if rot_k != 0:
            probs = torch.rot90(probs, k=(4-rot_k) % 4, dims=(3,4))

        if acc is None:
            acc = probs
        else:
            acc = acc + probs

    acc = acc / float(len(t_list))
    return acc


@torch.no_grad()
def eval_tta(cfg, ckpt_path: str, tta_mode: str, rc_min_vox: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = BratsRoiNpzDataset(cfg["data"]["roi_root"], "val", cfg["data"]["splits_dir"], AugmentCfg(enable=False))
    dl = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=max(1, cfg["data"]["num_workers"] // 2),
        pin_memory=cfg["data"]["pin_memory"],
    )

    model = ResUNet3D(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        base=cfg["model"]["base_channels"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location="cpu")  # warning ok (trusted local ckpt)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    agg = None
    n = 0

    for batch in dl:
        x = batch["image"].to(device, non_blocking=True)
        gt = batch["label"].to(device, non_blocking=True)

        probs = predict_probs_tta(model, x, tta_mode=tta_mode)  # (1,5,D,H,W)

        if rc_min_vox > 0:
            pred = torch.argmax(probs, dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)
            pred_pp = _postprocess_rc_only(pred, rc_min_vox=rc_min_vox)

            # rebuild probs-like tensor (one-hot) so compute_all_metrics can be reused
            C = probs.shape[1]
            onehot = torch.zeros((1, C) + gt.shape[1:], device=gt.device, dtype=torch.float32)
            idx = torch.from_numpy(pred_pp).to(gt.device).long().unsqueeze(0)  # (1,D,H,W)
            onehot.scatter_(1, idx.unsqueeze(1), 1.0)
            m = compute_all_metrics(onehot, gt)
        else:
            m = compute_all_metrics(probs, gt)

        if agg is None:
            agg = {k: 0.0 for k in m.keys()}
        for k, v in m.items():
            agg[k] += float(v)
        n += 1

    for k in agg:
        agg[k] /= max(1, n)

    meta = {
        "tta_mode": tta_mode,
        "rc_min_vox": int(rc_min_vox),
        "ckpt": str(ckpt_path),
        "epoch": int(ckpt.get("epoch", -1)),
    }
    return meta, agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tta", default="flip", choices=["none", "flip", "flip_rot90hw"])
    ap.add_argument("--rc_min_vox", type=int, default=0)
    ap.add_argument("--out_dir", default="", type=str)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    meta, m = eval_tta(cfg, args.ckpt, tta_mode=args.tta, rc_min_vox=args.rc_min_vox)

    out_dir = Path(args.out_dir) if args.out_dir else (Path(args.ckpt).resolve().parents[1] / "metrics")
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = f"tta_{args.tta}_rcmin{args.rc_min_vox}"
    json_path = out_dir / f"val_metrics_{tag}.json"
    csv_path  = out_dir / f"val_metrics_{tag}.csv"

    json_path.write_text(json.dumps({"meta": meta, "metrics": m}, indent=2), encoding="utf-8")

    cols = ["WT","TC","ET","RC","mean_wt_tc_et","dice_1_netc","dice_2_snf_h","dice_3_et","dice_4_rc"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tta","rc_min_vox","ckpt","epoch"] + cols)
        w.writerow([args.tta, args.rc_min_vox, args.ckpt, meta["epoch"]] + [m[c] for c in cols])

    print("[OK] TTA eval done")
    print("json:", json_path)
    print("csv :", csv_path)
    print("metrics:", m)


if __name__ == "__main__":
    main()
