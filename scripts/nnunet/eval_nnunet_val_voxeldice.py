import argparse, csv, json
from pathlib import Path
import numpy as np
import nibabel as nib

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

def read_ids(p: Path):
    return [l.strip() for l in p.read_text().splitlines() if l.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True, help="nnUNet predict output folder")
    ap.add_argument("--raw_root", required=True, help=".../training_data1_v2")
    ap.add_argument("--val_list", required=True)
    ap.add_argument("--rc_min_vox", type=int, default=0)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)
    raw_root = Path(args.raw_root)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    case_ids = read_ids(Path(args.val_list))
    cols = ["WT","TC","ET","RC","mean_wt_tc_et","dice_1_netc","dice_2_snf_h","dice_3_et","dice_4_rc"]

    per_rows = []
    sum_m = None

    for i, cid in enumerate(case_ids, 1):
        pred_path = pred_dir / f"{cid}.nii.gz"
        gt_path = raw_root / cid / f"{cid}-seg.nii.gz"
        if not pred_path.exists():
            raise FileNotFoundError(f"missing pred: {pred_path}")
        if not gt_path.exists():
            raise FileNotFoundError(f"missing gt: {gt_path}")

        pred = np.asanyarray(nib.load(pred_path).dataobj).astype(np.uint8)
        gt   = np.asanyarray(nib.load(gt_path).dataobj).astype(np.uint8)

        if pred.shape != gt.shape:
            raise RuntimeError(f"shape mismatch {cid}: pred {pred.shape} vs gt {gt.shape}")

        pred = postprocess_rc_only(pred, args.rc_min_vox)

        m = compute_metrics_from_labels(pred, gt)
        per_rows.append({"case_id": cid, **m})

        if sum_m is None:
            sum_m = {k: 0.0 for k in m.keys()}
        for k in sum_m:
            sum_m[k] += float(m[k])

        if i % 20 == 0 or i == len(case_ids):
            print(f"[progress] {i}/{len(case_ids)}")

    mean_m = {k: (sum_m[k] / float(len(per_rows))) for k in sum_m}

    meta = {
        "pred_dir": str(pred_dir),
        "raw_root": str(raw_root),
        "val_list": str(args.val_list),
        "rc_min_vox": int(args.rc_min_vox),
        "n_cases": len(per_rows),
    }

    (out_dir / f"nnunet_val_metrics_rcmin{args.rc_min_vox}.json").write_text(
        json.dumps({"meta": meta, "mean_metrics": mean_m}, indent=2), encoding="utf-8"
    )

    with (out_dir / f"nnunet_val_summary_rcmin{args.rc_min_vox}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n_cases"] + cols)
        w.writerow([len(per_rows)] + [mean_m[c] for c in cols])

    with (out_dir / f"nnunet_val_per_case_rcmin{args.rc_min_vox}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["case_id"] + cols)
        w.writeheader()
        for r in per_rows:
            w.writerow({k: r[k] for k in ["case_id"] + cols})

    print("[OK] mean_metrics:", mean_m)

if __name__ == "__main__":
    main()
