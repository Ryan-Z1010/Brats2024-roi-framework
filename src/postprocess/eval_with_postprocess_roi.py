import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml
from scipy.ndimage import label as cc_label

from src.datasets.brats_roi_npz import BratsRoiNpzDataset, AugmentCfg
from src.models.unet3d_res import ResUNet3D
from src.metrics.brats_composites import compute_all_metrics

def keep_largest_cc(mask: np.ndarray):
    lab, n = cc_label(mask.astype(np.uint8))
    if n <= 1:
        return mask
    sizes = [(lab == i).sum() for i in range(1, n+1)]
    k = int(np.argmax(sizes) + 1)
    return (lab == k)

def remove_small_cc(mask: np.ndarray, min_vox: int):
    lab, n = cc_label(mask.astype(np.uint8))
    if n == 0:
        return mask
    out = np.zeros_like(mask, dtype=bool)
    for i in range(1, n+1):
        comp = (lab == i)
        if comp.sum() >= min_vox:
            out |= comp
    return out

def postprocess(pred: np.ndarray, et_lcc: bool, rc_min_vox: int):
    # pred: (D,H,W) uint8 labels 0..4
    out = pred.copy()

    if et_lcc:
        et = (out == 3)
        et2 = keep_largest_cc(et)
        out[et & (~et2)] = 0
        out[et2] = 3

    if rc_min_vox > 0:
        rc = (out == 4)
        rc2 = remove_small_cc(rc, rc_min_vox)
        out[rc & (~rc2)] = 0
        out[rc2] = 4

    return out

@torch.no_grad()
def eval_model(cfg, ckpt_path: str, et_lcc: bool, rc_min_vox: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = BratsRoiNpzDataset(cfg["data"]["roi_root"], "val", cfg["data"]["splits_dir"], AugmentCfg(enable=False))
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=max(1, cfg["data"]["num_workers"] // 2),
                    pin_memory=cfg["data"]["pin_memory"])

    model = ResUNet3D(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        base=cfg["model"]["base_channels"],
        dropout=cfg["model"]["dropout"]
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    agg = None
    n = 0
    for batch in dl:
        img = batch["image"].to(device, non_blocking=True)
        gt  = batch["label"].to(device, non_blocking=True)

        logits = model(img)
        pred = torch.argmax(logits, dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)

        pred_pp = postprocess(pred, et_lcc=et_lcc, rc_min_vox=rc_min_vox)
        pred_pp_t = torch.from_numpy(pred_pp).to(gt.device).long().unsqueeze(0)

        # fake logits from labels for metric: we can compute dice directly by comparing pred labels to gt,
        # but reuse compute_all_metrics by constructing logits is wasteful; simplest: compute_all_metrics expects logits.
        # We'll compute metrics directly using a small wrapper:
        # Here we use pred labels by constructing one-hot logits-like tensor.
        C = cfg["model"]["out_channels"]
        onehot = torch.zeros((1, C) + pred_pp_t.shape[1:], device=gt.device, dtype=torch.float32)
        onehot.scatter_(1, pred_pp_t.unsqueeze(1), 1.0)

        m = compute_all_metrics(onehot, gt)  # argmax(onehot)=pred_pp
        if agg is None:
            agg = {k: 0.0 for k in m.keys()}
        for k, v in m.items():
            agg[k] += float(v)
        n += 1

    for k in agg:
        agg[k] /= max(1, n)
    return agg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--et_lcc", action="store_true")
    ap.add_argument("--rc_min_vox", type=int, default=0)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    m = eval_model(cfg, args.ckpt, et_lcc=args.et_lcc, rc_min_vox=args.rc_min_vox)

    out_path = Path(args.out) if args.out else (Path(args.ckpt).resolve().parents[1] / "metrics" / f"val_metrics_post_etlcc{int(args.et_lcc)}_rcmin{args.rc_min_vox}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"postprocess": {"et_lcc": args.et_lcc, "rc_min_vox": args.rc_min_vox}, "metrics": m}, indent=2), encoding="utf-8")

    print("[OK] postprocessed eval done")
    print("out:", out_path)
    print("metrics:", m)

if __name__ == "__main__":
    main()
