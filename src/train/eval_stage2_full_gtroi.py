import argparse, json, csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    from scipy.ndimage import label as cc_label
except Exception:
    cc_label = None


# ---------------- metrics (voxel-wise, match your "official" definition) ----------------
def dice_bool(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    ia = int(a.sum())
    ib = int(b.sum())
    if ia == 0 and ib == 0:
        return 1.0
    if ia == 0 or ib == 0:
        return 0.0
    inter = int((a & b).sum())
    return (2.0 * inter) / (ia + ib)

def compute_metrics_from_labels(pred: np.ndarray, gt: np.ndarray) -> dict:
    # pred/gt: uint8 (D,H,W) in 0..4
    m = {}
    m["dice_1_netc"] = dice_bool(pred == 1, gt == 1)
    m["dice_2_snf_h"] = dice_bool(pred == 2, gt == 2)
    m["dice_3_et"]   = dice_bool(pred == 3, gt == 3)
    m["dice_4_rc"]   = dice_bool(pred == 4, gt == 4)

    WT = ((pred == 1) | (pred == 2) | (pred == 3))
    WTg = ((gt == 1) | (gt == 2) | (gt == 3))
    TC = ((pred == 1) | (pred == 3))
    TCg = ((gt == 1) | (gt == 3))
    ET = (pred == 3)
    ETg = (gt == 3)
    RC = (pred == 4)
    RCg = (gt == 4)

    m["WT"] = dice_bool(WT, WTg)
    m["TC"] = dice_bool(TC, TCg)
    m["ET"] = dice_bool(ET, ETg)
    m["RC"] = dice_bool(RC, RCg)
    m["mean_wt_tc_et"] = (m["WT"] + m["TC"] + m["ET"]) / 3.0
    return m


# ---------------- crop & paste (oracle ROI from GT) ----------------
def bbox_from_mask(mask: np.ndarray):
    # mask: bool (D,H,W)
    if mask.any() is False:
        return None
    idx = np.where(mask)
    zmin, zmax = int(idx[0].min()), int(idx[0].max())
    ymin, ymax = int(idx[1].min()), int(idx[1].max())
    xmin, xmax = int(idx[2].min()), int(idx[2].max())
    return (zmin, zmax, ymin, ymax, xmin, xmax)

def center_crop_with_pad(vol: np.ndarray, center_zyx, roi_size: int, pad_val: float = 0.0):
    """
    vol: (C,D,H,W) or (D,H,W)
    returns:
      crop, slices_full, slices_crop
    slices_full: tuple of slice in full volume
    slices_crop: tuple of slice in crop volume (where real data resides)
    """
    is_4d = (vol.ndim == 4)
    if is_4d:
        C, D, H, W = vol.shape
    else:
        D, H, W = vol.shape

    cz, cy, cx = center_zyx
    half = roi_size // 2

    sz = cz - half; ez = sz + roi_size
    sy = cy - half; ey = sy + roi_size
    sx = cx - half; ex = sx + roi_size

    fz0 = max(0, sz); fz1 = min(D, ez)
    fy0 = max(0, sy); fy1 = min(H, ey)
    fx0 = max(0, sx); fx1 = min(W, ex)

    # where to paste inside crop
    pz0 = fz0 - sz; pz1 = pz0 + (fz1 - fz0)
    py0 = fy0 - sy; py1 = py0 + (fy1 - fy0)
    px0 = fx0 - sx; px1 = px0 + (fx1 - fx0)

    if is_4d:
        crop = np.full((C, roi_size, roi_size, roi_size), pad_val, dtype=vol.dtype)
        crop[:, pz0:pz1, py0:py1, px0:px1] = vol[:, fz0:fz1, fy0:fy1, fx0:fx1]
    else:
        crop = np.full((roi_size, roi_size, roi_size), pad_val, dtype=vol.dtype)
        crop[pz0:pz1, py0:py1, px0:px1] = vol[fz0:fz1, fy0:fy1, fx0:fx1]

    slices_full = (slice(fz0, fz1), slice(fy0, fy1), slice(fx0, fx1))
    slices_crop = (slice(pz0, pz1), slice(py0, py1), slice(px0, px1))
    return crop, slices_full, slices_crop


# ---------------- postprocess (RC-only) ----------------
def remove_small_cc(mask: np.ndarray, min_vox: int) -> np.ndarray:
    if min_vox <= 0:
        return mask
    if cc_label is None:
        raise RuntimeError("scipy not available; pip install scipy")
    lab, n = cc_label(mask.astype(np.uint8))
    if n == 0:
        return mask
    out = np.zeros_like(mask, dtype=bool)
    for i in range(1, n + 1):
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


# ---------------- model loader ----------------
def build_model(base_channels: int, device):
    # match your ResUNet3D used before
    from src.models.unet3d_res import ResUNet3D
    m = ResUNet3D(in_channels=4, out_channels=5, base=base_channels, dropout=0.0).to(device)
    m.eval()
    return m

@torch.no_grad()
def predict_probs_ensemble(models, x):
    # x: (1,4,D,H,W) -> probs: (1,5,D,H,W)
    ps = None
    for m in models:
        logits = m(x)
        p = F.softmax(logits, dim=1)
        ps = p if ps is None else (ps + p)
    return ps / float(len(models))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full_root", required=True, help="e.g. data/preprocessed/npy_full_v1/val")
    ap.add_argument("--val_list", required=True, help="e.g. data/splits/val.txt")
    ap.add_argument("--ckpts", required=True, help="comma-separated best.pt paths (1=single, >=2=ensemble)")
    ap.add_argument("--base_channels", type=int, default=48)
    ap.add_argument("--roi_size", type=int, default=128)
    ap.add_argument("--rc_min_vox", type=int, default=0)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--save_preds", action="store_true", help="save full-volume preds per case (large)")
    ap.add_argument("--max_cases", type=int, default=-1, help="debug: only eval first N cases")
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = False

    full_root = Path(args.full_root)
    case_ids = [l.strip() for l in Path(args.val_list).read_text().splitlines() if l.strip()]
    if args.max_cases > 0:
        case_ids = case_ids[:args.max_cases]

    ckpts = [c.strip() for c in args.ckpts.split(",") if c.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "preds").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    for p in ckpts:
        m = build_model(args.base_channels, device)
        ck = torch.load(p, map_location="cpu")
        m.load_state_dict(ck["model"], strict=True)
        m.eval()
        models.append(m)

    per_rows = []
    sum_m = None

    for i, cid in enumerate(case_ids, 1):
        f = full_root / f"{cid}.npz"
        d = np.load(f)
        img = d["img"].astype(np.float32)   # (4,182,218,182)
        gt  = d["seg"].astype(np.uint8)     # (182,218,182)

        # oracle ROI center from GT tumor (labels 1..4)
        tumor = (gt > 0)
        bb = bbox_from_mask(tumor)
        if bb is None:
            # fallback: center crop whole volume
            D,H,W = gt.shape
            center = (D//2, H//2, W//2)
        else:
            zmin,zmax,ymin,ymax,xmin,xmax = bb
            center = ((zmin+zmax)//2, (ymin+ymax)//2, (xmin+xmax)//2)

        crop_img, sl_full, sl_crop = center_crop_with_pad(img, center, args.roi_size, pad_val=0.0)

        x = torch.from_numpy(crop_img).unsqueeze(0).to(device)  # (1,4,128,128,128)
        probs = predict_probs_ensemble(models, x)
        pred_roi = torch.argmax(probs, dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)

        # paste back to full
        pred_full = np.zeros_like(gt, dtype=np.uint8)
        pred_full[sl_full] = pred_roi[sl_crop]

        # rc-only postprocess on full volume
        pred_full = postprocess_rc_only(pred_full, args.rc_min_vox)

        m = compute_metrics_from_labels(pred_full, gt)

        row = {"case_id": cid, **m,
               "center_zyx": list(center),
               "full_slices": [sl_full[0].start, sl_full[0].stop, sl_full[1].start, sl_full[1].stop, sl_full[2].start, sl_full[2].stop]}
        per_rows.append(row)

        if sum_m is None:
            sum_m = {k: 0.0 for k in m.keys()}
        for k in sum_m:
            sum_m[k] += float(m[k])

        if args.save_preds:
            np.savez_compressed(out_dir / "preds" / f"{cid}.npz",
                                pred=pred_full.astype(np.uint8),
                                center=np.array(center, dtype=np.int16),
                                full_slices=np.array(row["full_slices"], dtype=np.int16))

        if i % 20 == 0 or i == len(case_ids):
            print(f"[progress] {i}/{len(case_ids)} done")

    # mean metrics
    mean_m = {k: (sum_m[k] / float(len(per_rows))) for k in sum_m}

    meta = {
        "full_root": str(full_root),
        "val_list": str(args.val_list),
        "roi_size": int(args.roi_size),
        "oracle_roi_from_gt": True,
        "rc_min_vox": int(args.rc_min_vox),
        "base_channels": int(args.base_channels),
        "ckpts": ckpts,
        "n_cases": len(per_rows),
    }

    # write outputs
    json_path = out_dir / f"full_gtroi_metrics_rcmin{args.rc_min_vox}.json"
    csv_case  = out_dir / f"full_gtroi_per_case_rcmin{args.rc_min_vox}.csv"
    csv_sum   = out_dir / f"full_gtroi_summary_rcmin{args.rc_min_vox}.csv"

    json_path.write_text(json.dumps({"meta": meta, "mean_metrics": mean_m}, indent=2), encoding="utf-8")

    cols = ["WT","TC","ET","RC","mean_wt_tc_et","dice_1_netc","dice_2_snf_h","dice_3_et","dice_4_rc"]
    with csv_sum.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n_cases"] + cols)
        w.writerow([len(per_rows)] + [mean_m[c] for c in cols])

    with csv_case.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["case_id"] + cols + ["center_zyx","full_slices"])
        w.writeheader()
        for r in per_rows:
            w.writerow({k: r[k] for k in w.fieldnames})

    print("[OK] full-volume GT-ROI eval done")
    print("json:", json_path)
    print("summary_csv:", csv_sum)
    print("per_case_csv:", csv_case)
    print("mean_metrics:", mean_m)


if __name__ == "__main__":
    main()
