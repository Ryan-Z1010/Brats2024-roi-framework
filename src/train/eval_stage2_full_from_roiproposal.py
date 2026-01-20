import argparse, json, csv
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from src.models.unet3d_res import ResUNet3D

try:
    from scipy.ndimage import label as cc_label
except Exception:
    cc_label = None


def dice_bool(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool); b = b.astype(bool)
    ia = int(a.sum()); ib = int(b.sum())
    if ia == 0 and ib == 0: return 1.0
    if ia == 0 or ib == 0: return 0.0
    inter = int((a & b).sum())
    return (2.0 * inter) / (ia + ib)

def compute_metrics_from_labels(pred: np.ndarray, gt: np.ndarray) -> dict:
    m = {}
    m["dice_1_netc"] = dice_bool(pred == 1, gt == 1)
    m["dice_2_snf_h"] = dice_bool(pred == 2, gt == 2)
    m["dice_3_et"]   = dice_bool(pred == 3, gt == 3)
    m["dice_4_rc"]   = dice_bool(pred == 4, gt == 4)

    WT  = ((pred==1)|(pred==2)|(pred==3)); WTg = ((gt==1)|(gt==2)|(gt==3))
    TC  = ((pred==1)|(pred==3));          TCg = ((gt==1)|(gt==3))
    ET  = (pred==3);                       ETg = (gt==3)
    RC  = (pred==4);                       RCg = (gt==4)

    m["WT"] = dice_bool(WT, WTg)
    m["TC"] = dice_bool(TC, TCg)
    m["ET"] = dice_bool(ET, ETg)
    m["RC"] = dice_bool(RC, RCg)
    m["mean_wt_tc_et"] = (m["WT"] + m["TC"] + m["ET"]) / 3.0
    return m


def center_crop_with_pad(vol: np.ndarray, center_zyx, roi_size: int, pad_val: float = 0.0):
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

    slices_full = (slice(fz0, fz1), slice(fy0, fy1), slice(fx0, fx1))
    slices_crop = (slice(pz0, pz1), slice(py0, py1), slice(px0, px1))
    return crop, slices_full, slices_crop


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


@torch.no_grad()
def predict_probs_ensemble(models, x):
    ps = None
    for m in models:
        logits = m(x)
        p = F.softmax(logits, dim=1)
        ps = p if ps is None else (ps + p)
    return ps / float(len(models))


def load_proposals(csv_path: str):
    rows = {}
    import csv as _csv
    with open(csv_path, "r", encoding="utf-8") as f:
        r = _csv.DictReader(f)
        for row in r:
            cid = row["case_id"]
            rows[cid] = (int(row["center_z"]), int(row["center_y"]), int(row["center_x"]))
    return rows


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full_root", required=True, help="data/preprocessed/npy_full_v1/val")
    ap.add_argument("--val_list", required=True, help="data/splits/val.txt")
    ap.add_argument("--proposal_csv", required=True, help="results/.../val_roi128.csv")
    ap.add_argument("--ckpts", required=True, help="comma-separated best.pt paths (>=2 for ensemble)")
    ap.add_argument("--base_channels", type=int, default=48)
    ap.add_argument("--roi_size", type=int, default=128)
    ap.add_argument("--rc_min_vox", type=int, default=0)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--save_nii", action="store_true", help="save pred_full as nii.gz per case")
    ap.add_argument("--nii_affine", default="identity", choices=["identity"], help="affine for saved nii (analysis only)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpts = [c.strip() for c in args.ckpts.split(",") if c.strip()]
    models = []
    for p in ckpts:
        m = ResUNet3D(in_channels=4, out_channels=5, base=args.base_channels, dropout=0.0).to(device)
        ck = torch.load(p, map_location="cpu")
        m.load_state_dict(ck["model"], strict=True)
        m.eval()
        models.append(m)

    prop = load_proposals(args.proposal_csv)

    case_ids = [l.strip() for l in Path(args.val_list).read_text().splitlines() if l.strip()]
    full_root = Path(args.full_root)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    per_rows = []
    sum_m = None

    for i, cid in enumerate(case_ids, 1):
        f = full_root / f"{cid}.npz"
        d = np.load(f)
        img = d["img"].astype(np.float32)
        gt  = d["seg"].astype(np.uint8)

        full_shape = gt.shape
        if cid in prop:
            center = prop[cid]
        else:
            center = (full_shape[0]//2, full_shape[1]//2, full_shape[2]//2)

        crop_img, sl_full, sl_crop = center_crop_with_pad(img, center, args.roi_size, pad_val=0.0)

        x = torch.from_numpy(crop_img).unsqueeze(0).to(device)
        probs = predict_probs_ensemble(models, x)
        pred_roi = torch.argmax(probs, dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)

        pred_full = np.zeros_like(gt, dtype=np.uint8)
        pred_full[sl_full] = pred_roi[sl_crop]

        pred_full = postprocess_rc_only(pred_full, args.rc_min_vox)

        if args.save_nii:
            import nibabel as nib
            nii_dir = out_dir / "nii"
            nii_dir.mkdir(exist_ok=True)
            aff = np.eye(4, dtype=np.float32)
            nib.save(nib.Nifti1Image(pred_full.astype(np.uint8), aff),
                     str(nii_dir / f"{cid}.nii.gz"))

        m = compute_metrics_from_labels(pred_full, gt)

        per_rows.append({"case_id": cid, **m})
        if sum_m is None:
            sum_m = {k: 0.0 for k in m.keys()}
        for k in sum_m:
            sum_m[k] += float(m[k])

        if i % 20 == 0 or i == len(case_ids):
            print(f"[progress] {i}/{len(case_ids)} done")

    mean_m = {k: (sum_m[k] / float(len(per_rows))) for k in sum_m}

    meta = {
        "full_root": str(full_root),
        "val_list": str(args.val_list),
        "proposal_csv": str(args.proposal_csv),
        "roi_size": int(args.roi_size),
        "rc_min_vox": int(args.rc_min_vox),
        "base_channels": int(args.base_channels),
        "ckpts": ckpts,
        "n_cases": len(per_rows),
    }

    json_path = out_dir / f"full_stage1roi_metrics_rcmin{args.rc_min_vox}.json"
    sum_csv   = out_dir / f"full_stage1roi_summary_rcmin{args.rc_min_vox}.csv"
    case_csv  = out_dir / f"full_stage1roi_per_case_rcmin{args.rc_min_vox}.csv"

    json_path.write_text(json.dumps({"meta": meta, "mean_metrics": mean_m}, indent=2), encoding="utf-8")

    cols = ["WT","TC","ET","RC","mean_wt_tc_et","dice_1_netc","dice_2_snf_h","dice_3_et","dice_4_rc"]
    with sum_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n_cases"] + cols)
        w.writerow([len(per_rows)] + [mean_m[c] for c in cols])

    with case_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["case_id"] + cols)
        w.writeheader()
        for r in per_rows:
            w.writerow({k: r[k] for k in ["case_id"] + cols})

    print("[OK] full-volume Stage1-ROI eval done")
    print("json:", json_path)
    print("summary_csv:", sum_csv)
    print("per_case_csv:", case_csv)
    print("mean_metrics:", mean_m)

if __name__ == "__main__":
    main()
