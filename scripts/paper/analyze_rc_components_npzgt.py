import argparse, csv
from pathlib import Path
import numpy as np

import nibabel as nib
from scipy.ndimage import label as cc_label

def load_gt_rc(npz_path: Path, rc_label: int) -> np.ndarray:
    d = np.load(npz_path)
    gt = d["seg"].astype(np.uint8)
    return (gt == rc_label)

def load_pred_rc(nii_path: Path, rc_label: int) -> np.ndarray:
    pred = nib.load(str(nii_path)).get_fdata().astype(np.uint8)
    return (pred == rc_label)

def cc(mask: np.ndarray):
    # 26-connectivity in 3D
    struct = np.ones((3,3,3), dtype=np.uint8)
    lab, n = cc_label(mask.astype(np.uint8), structure=struct)
    return lab, n

def component_match_counts(pred_rc: np.ndarray, gt_rc: np.ndarray):
    plab, pn = cc(pred_rc)
    glab, gn = cc(gt_rc)

    # build GT component masks indices to speed overlap check
    # For each pred component, see if overlaps ANY gt component => TP else FP
    tp = fp = fn = 0
    pred_vols = []
    fp_vols = []

    # mark which GT comps are hit
    gt_hit = np.zeros(gn + 1, dtype=bool)  # index 1..gn

    for i in range(1, pn + 1):
        pcomp = (plab == i)
        v = int(pcomp.sum())
        pred_vols.append(v)
        if v == 0:
            continue
        overlap_gt_ids = np.unique(glab[pcomp])
        overlap_gt_ids = overlap_gt_ids[overlap_gt_ids > 0]
        if overlap_gt_ids.size > 0:
            tp += 1
            gt_hit[overlap_gt_ids] = True
        else:
            fp += 1
            fp_vols.append(v)

    # FN: GT components not hit by any pred component
    for j in range(1, gn + 1):
        if not gt_hit[j]:
            fn += 1

    return tp, fp, fn, pred_vols, fp_vols

def summarize(vols):
    if len(vols) == 0:
        return dict(mean=0.0, p50=0.0, p90=0.0, max=0.0)
    a = np.array(vols, dtype=np.float32)
    return dict(
        mean=float(a.mean()),
        p50=float(np.percentile(a, 50)),
        p90=float(np.percentile(a, 90)),
        max=float(a.max())
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_npz_root", required=True)
    ap.add_argument("--val_list", required=True)
    ap.add_argument("--pred_root0", required=True)
    ap.add_argument("--pred_root120", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--rc_label", type=int, default=4)
    args = ap.parse_args()

    gt_root = Path(args.gt_npz_root)
    pred0 = Path(args.pred_root0)
    pred120 = Path(args.pred_root120)

    case_ids = [l.strip() for l in Path(args.val_list).read_text().splitlines() if l.strip()]

    def eval_one(pred_root: Path):
        TP=FP=FN=0
        all_pred_cc = 0
        all_gt_cc = 0
        all_fp_vols = []

        for cid in case_ids:
            gt_path = gt_root / f"{cid}.npz"
            pred_path = pred_root / f"{cid}.nii.gz"
            if not gt_path.exists():
                raise FileNotFoundError(f"Missing GT: {gt_path}")
            if not pred_path.exists():
                raise FileNotFoundError(f"Missing pred: {pred_path}")

            gt_rc = load_gt_rc(gt_path, args.rc_label)
            pr_rc = load_pred_rc(pred_path, args.rc_label)

            tp, fp, fn, pred_vols, fp_vols = component_match_counts(pr_rc, gt_rc)

            TP += tp; FP += fp; FN += fn
            all_pred_cc += (tp + fp)
            all_gt_cc += (tp + fn)
            all_fp_vols.extend(fp_vols)

        prec = TP / (TP + FP) if (TP + FP) > 0 else 1.0
        rec  = TP / (TP + FN) if (TP + FN) > 0 else 1.0
        f1   = (2*prec*rec)/(prec+rec) if (prec+rec) > 0 else 0.0
        s = summarize(all_fp_vols)
        return dict(
            TP=TP, FP=FP, FN=FN,
            pred_cc=all_pred_cc, gt_cc=all_gt_cc,
            comp_precision=prec, comp_recall=rec, comp_f1=f1,
            fp_vol_mean=s["mean"], fp_vol_p50=s["p50"], fp_vol_p90=s["p90"], fp_vol_max=s["max"]
        )

    r0 = eval_one(pred0)
    r120 = eval_one(pred120)

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["version","pred_cc","gt_cc","TP","FP","FN","comp_precision","comp_recall","comp_f1",
            "fp_vol_mean","fp_vol_p50","fp_vol_p90","fp_vol_max"]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerow({"version":"rcmin0", **r0})
        w.writerow({"version":"rcmin120", **r120})

    print("[OK] wrote:", out)
    print("rcmin0  :", r0)
    print("rcmin120:", r120)

if __name__ == "__main__":
    main()
