import argparse, json, csv
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

def remove_small_cc(mask: np.ndarray, min_vox: int) -> np.ndarray:
    if min_vox <= 0:
        return mask
    if cc_label is None:
        raise RuntimeError("scipy not available; pip install scipy")
    lab, n = cc_label(mask.astype(np.uint8))
    if n == 0:
        return mask
    out = np.zeros_like(mask, dtype=bool)
    for i in range(1, n+1):
        comp = (lab == i)
        if int(comp.sum()) >= int(min_vox):
            out |= comp
    return out

def postprocess_rc_only(pred_lbl: np.ndarray, rc_min_vox: int) -> np.ndarray:
    if rc_min_vox <= 0:
        return pred_lbl
    out = pred_lbl.copy()
    rc = (out == 4)
    rc2 = remove_small_cc(rc, rc_min_vox)
    out[rc & (~rc2)] = 0
    out[rc2] = 4
    return out

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpts", required=True, help="comma-separated list of best.pt paths")
    ap.add_argument("--rc_min_vox", type=int, default=0)
    ap.add_argument("--out_dir", default="", type=str)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    ckpts = [c.strip() for c in args.ckpts.split(",") if c.strip()]
    assert len(ckpts) >= 2, "Need >=2 checkpoints for ensemble"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = BratsRoiNpzDataset(cfg["data"]["roi_root"], "val", cfg["data"]["splits_dir"], AugmentCfg(enable=False))
    dl = DataLoader(ds, batch_size=1, shuffle=False,
                    num_workers=max(1, cfg["data"]["num_workers"] // 2),
                    pin_memory=cfg["data"]["pin_memory"])

    models = []
    for p in ckpts:
        m = ResUNet3D(
            in_channels=cfg["model"]["in_channels"],
            out_channels=cfg["model"]["out_channels"],
            base=cfg["model"]["base_channels"],
            dropout=cfg["model"]["dropout"],
        ).to(device)
        ck = torch.load(p, map_location="cpu")
        m.load_state_dict(ck["model"], strict=True)
        m.eval()
        models.append(m)

    agg, n = None, 0
    for batch in dl:
        x = batch["image"].to(device, non_blocking=True)
        gt = batch["label"].to(device, non_blocking=True)

        probs_sum = None
        for m in models:
            logits = m(x)
            probs = F.softmax(logits, dim=1)
            probs_sum = probs if probs_sum is None else (probs_sum + probs)
        probs_avg = probs_sum / float(len(models))

        if args.rc_min_vox > 0:
            pred = torch.argmax(probs_avg, dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)
            pred_pp = postprocess_rc_only(pred, args.rc_min_vox)
            C = probs_avg.shape[1]
            onehot = torch.zeros((1, C) + gt.shape[1:], device=gt.device, dtype=torch.float32)
            idx = torch.from_numpy(pred_pp).to(gt.device).long().unsqueeze(0)
            onehot.scatter_(1, idx.unsqueeze(1), 1.0)
            mtr = compute_all_metrics(onehot, gt)
        else:
            mtr = compute_all_metrics(probs_avg, gt)

        if agg is None:
            agg = {k: 0.0 for k in mtr.keys()}
        for k,v in mtr.items():
            agg[k] += float(v)
        n += 1

    for k in agg:
        agg[k] /= max(1, n)

    out_dir = Path(args.out_dir) if args.out_dir else Path("results/ensembles/ens") 
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = f"ens{len(ckpts)}_rcmin{args.rc_min_vox}"
    jpath = out_dir / f"val_metrics_{tag}.json"
    cpath = out_dir / f"val_metrics_{tag}.csv"

    meta = {"ckpts": ckpts, "rc_min_vox": int(args.rc_min_vox)}
    jpath.write_text(json.dumps({"meta": meta, "metrics": agg}, indent=2), encoding="utf-8")

    cols = ["WT","TC","ET","RC","mean_wt_tc_et","dice_1_netc","dice_2_snf_h","dice_3_et","dice_4_rc"]
    with cpath.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n_models","rc_min_vox"] + cols)
        w.writerow([len(ckpts), args.rc_min_vox] + [agg[c] for c in cols])

    print("[OK] ensemble eval done")
    print("json:", jpath)
    print("csv :", cpath)
    print("metrics:", agg)

if __name__ == "__main__":
    main()
