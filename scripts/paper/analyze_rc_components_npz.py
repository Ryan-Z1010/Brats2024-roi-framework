import os, glob, argparse
import numpy as np
import pandas as pd
from skimage import measure

def rc_filter_min_vox(lbl: np.ndarray, rc_label=4, min_vox=120):
    """Remove RC connected components smaller than min_vox (in voxel count)."""
    out = lbl.copy()
    rc = (out == rc_label).astype(np.uint8)
    cc = measure.label(rc, connectivity=1)
    if cc.max() == 0:
        return out
    for i in range(1, cc.max() + 1):
        m = (cc == i)
        if int(m.sum()) < int(min_vox):
            out[m] = 0
    return out

def rc_cc_stats(gt_lbl: np.ndarray, pred_lbl: np.ndarray, rc_label=4):
    gt_rc = (gt_lbl == rc_label).astype(np.uint8)
    pr_rc = (pred_lbl == rc_label).astype(np.uint8)

    gt_vox = int(gt_rc.sum())
    pr_vox = int(pr_rc.sum())

    cc_gt = measure.label(gt_rc, connectivity=1)
    cc_pr = measure.label(pr_rc, connectivity=1)
    n_gt = int(cc_gt.max())
    n_pr = int(cc_pr.max())

    fp_cc = 0
    fp_vox = 0
    tp_cc = 0
    sizes = []

    for i in range(1, n_pr + 1):
        comp = (cc_pr == i)
        s = int(comp.sum())
        sizes.append(s)
        if (gt_rc & comp).sum() == 0:
            fp_cc += 1
            fp_vox += s
        else:
            tp_cc += 1

    largest_ratio = (max(sizes) / sum(sizes)) if len(sizes) > 0 else 0.0
    return {
        "GT_RC_vox": gt_vox,
        "Pred_RC_vox": pr_vox,
        "n_gt_cc": n_gt,
        "n_pred_cc": n_pr,
        "fp_cc": int(fp_cc),
        "fp_vox": int(fp_vox),
        "tp_cc": int(tp_cc),
        "largest_cc_ratio": float(largest_ratio),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_npz_root", required=True, help="data/preprocessed/npy_full_v1/val")
    ap.add_argument("--pred_npz_root", required=True, help="dir containing per-case pred_full npz (key=pred)")
    ap.add_argument("--rc_min_vox", type=int, default=120)
    ap.add_argument("--out_per_case_csv", required=True)
    ap.add_argument("--out_summary_csv", required=True)
    args = ap.parse_args()

    pred_files = sorted(glob.glob(os.path.join(args.pred_npz_root, "*.npz")))
    if len(pred_files) == 0:
        raise FileNotFoundError(f"No .npz found in pred_npz_root: {args.pred_npz_root}")

    rows = []
    for pf in pred_files:
        case_id = os.path.splitext(os.path.basename(pf))[0]
        gt_path = os.path.join(args.gt_npz_root, f"{case_id}.npz")
        if not os.path.exists(gt_path):
            continue

        gt = np.load(gt_path)["seg"].astype(np.uint8)
        pred0 = np.load(pf)["pred"].astype(np.uint8)
        pred120 = rc_filter_min_vox(pred0, rc_label=4, min_vox=args.rc_min_vox)

        s0 = rc_cc_stats(gt, pred0, rc_label=4)
        s1 = rc_cc_stats(gt, pred120, rc_label=4)

        row = {"case_id": case_id}
        for k, v in s0.items():
            row[f"{k}_rcmin0"] = v
        for k, v in s1.items():
            row[f"{k}_rcmin{args.rc_min_vox}"] = v
        row["delta_fp_cc"] = row[f"fp_cc_rcmin{args.rc_min_vox}"] - row["fp_cc_rcmin0"]
        row["delta_fp_vox"] = row[f"fp_vox_rcmin{args.rc_min_vox}"] - row["fp_vox_rcmin0"]
        row["delta_largest_cc_ratio"] = row[f"largest_cc_ratio_rcmin{args.rc_min_vox}"] - row["largest_cc_ratio_rcmin0"]
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(args.out_per_case_csv, index=False)

    # 高标准：分 GT_RC_vox=0 和 >0 两组统计
    def summarize(sub: pd.DataFrame, name: str):
        return pd.Series({
            "group": name,
            "n_cases": len(sub),
            "mean_fp_cc_rcmin0": sub["fp_cc_rcmin0"].mean(),
            f"mean_fp_cc_rcmin{args.rc_min_vox}": sub[f"fp_cc_rcmin{args.rc_min_vox}"].mean(),
            "mean_delta_fp_cc": sub["delta_fp_cc"].mean(),
            "mean_fp_vox_rcmin0": sub["fp_vox_rcmin0"].mean(),
            f"mean_fp_vox_rcmin{args.rc_min_vox}": sub[f"fp_vox_rcmin{args.rc_min_vox}"].mean(),
            "mean_delta_fp_vox": sub["delta_fp_vox"].mean(),
            "mean_largest_ratio_rcmin0": sub["largest_cc_ratio_rcmin0"].mean(),
            f"mean_largest_ratio_rcmin{args.rc_min_vox}": sub[f"largest_cc_ratio_rcmin{args.rc_min_vox}"].mean(),
            "mean_delta_largest_ratio": sub["delta_largest_cc_ratio"].mean(),
        })

    s_all = summarize(df, "all")
    s_zero = summarize(df[df["GT_RC_vox_rcmin0"] == 0], "GT_RC=0")
    s_pos = summarize(df[df["GT_RC_vox_rcmin0"] > 0], "GT_RC>0")

    summ = pd.DataFrame([s_all, s_zero, s_pos])
    summ.to_csv(args.out_summary_csv, index=False)

    print("[OK] saved per-case:", args.out_per_case_csv)
    print("[OK] saved summary :", args.out_summary_csv)
    print(summ)

if __name__ == "__main__":
    main()
