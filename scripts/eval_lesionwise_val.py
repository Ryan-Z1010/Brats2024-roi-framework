import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import label as cc_label

LABELS = {
    1: "NETC",
    2: "SNFH",
    3: "ET",
    4: "RC",
}

def dice(a: np.ndarray, b: np.ndarray, eps=1e-8) -> float:
    inter = np.logical_and(a, b).sum()
    den = a.sum() + b.sum()
    return (2.0 * inter + eps) / (den + eps)

def iou(a: np.ndarray, b: np.ndarray, eps=1e-8) -> float:
    inter = np.logical_and(a, b).sum()
    uni = np.logical_or(a, b).sum()
    return (inter + eps) / (uni + eps)

def components(binmask: np.ndarray):
    # 26-connectivity in 3D
    struct = np.ones((3,3,3), dtype=np.uint8)
    lab, n = cc_label(binmask.astype(np.uint8), structure=struct)
    comps = []
    for k in range(1, n+1):
        comps.append(lab == k)
    return comps

def match_components(gt_comps, pr_comps, iou_thr=1e-6):
    """
    Greedy matching by IoU (descending).
    Return list of matched pairs (gi, pi), plus unmatched indices.
    """
    pairs = []
    for gi, g in enumerate(gt_comps):
        for pi, p in enumerate(pr_comps):
            val = iou(g, p)
            if val > iou_thr:
                pairs.append((val, gi, pi))
    pairs.sort(reverse=True, key=lambda x: x[0])

    used_g = set()
    used_p = set()
    matches = []
    for val, gi, pi in pairs:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        matches.append((gi, pi))

    unmatched_g = [i for i in range(len(gt_comps)) if i not in used_g]
    unmatched_p = [i for i in range(len(pr_comps)) if i not in used_p]
    return matches, unmatched_g, unmatched_p

def load_seg(path: Path) -> np.ndarray:
    # ensure integer labels
    return np.asarray(nib.load(str(path)).get_fdata(), dtype=np.int16)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_npz_root", default="", help="if set, read GT seg from <root>/<cid>.npz instead of raw nii")
    ap.add_argument("--val_list", required=True)
    ap.add_argument("--gt_root", required=True, help=".../training_data1_v2")
    ap.add_argument("--pred_dir", required=True, help="folder containing BraTS-GLI-xxxx.nii.gz")
    ap.add_argument("--out_csv", default="results/lesionwise_val.csv")
    args = ap.parse_args()

    case_ids = [l.strip() for l in Path(args.val_list).read_text().splitlines() if l.strip()]
    gt_root = Path(args.gt_root)
    pred_dir = Path(args.pred_dir)

    rows = []
    agg = {name: {"tp":0, "fp":0, "fn":0, "dice_sum":0.0, "dice_n":0} for name in LABELS.values()}

    for cid in case_ids:
        pr_path = pred_dir / f"{cid}.nii.gz"
        missing_pred = int(not pr_path.exists())

        # ---- load GT ----
        if args.gt_npz_root:
            npz_path = Path(args.gt_npz_root) / f"{cid}.npz"
            missing_gt = int(not npz_path.exists())
            if missing_gt or missing_pred:
                rows.append({"case_id": cid, "missing_gt": missing_gt, "missing_pred": missing_pred})
                continue
            gt = np.load(npz_path)["seg"].astype(np.int16)
        else:
            gt_path = gt_root / cid / f"{cid}-seg.nii.gz"
            missing_gt = int(not gt_path.exists())
            if missing_gt or missing_pred:
                rows.append({"case_id": cid, "missing_gt": missing_gt, "missing_pred": missing_pred})
                continue
            gt = load_seg(gt_path)

        pr = load_seg(pr_path)

        row = {"case_id": cid}
        for lab, name in LABELS.items():
            g = (gt == lab)
            p = (pr == lab)
            gt_comps = components(g)
            pr_comps = components(p)

            matches, ug, up = match_components(gt_comps, pr_comps)
            tp = len(matches)
            fn = len(ug)
            fp = len(up)

            # lesion-wise dice: average dice over matched lesions (gt lesions that found a pred)
            dices = []
            for gi, pi in matches:
                dices.append(dice(gt_comps[gi], pr_comps[pi]))
            lw_dice = float(np.mean(dices)) if dices else 0.0

            prec = tp / (tp + fp + 1e-8)
            rec  = tp / (tp + fn + 1e-8)
            f1   = 2*prec*rec/(prec+rec+1e-8)

            row[f"{name}_tp"] = tp
            row[f"{name}_fp"] = fp
            row[f"{name}_fn"] = fn
            row[f"{name}_prec"] = prec
            row[f"{name}_rec"] = rec
            row[f"{name}_f1"] = f1
            row[f"{name}_lesion_dice"] = lw_dice

            agg[name]["tp"] += tp
            agg[name]["fp"] += fp
            agg[name]["fn"] += fn
            agg[name]["dice_sum"] += lw_dice
            agg[name]["dice_n"] += 1

        rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    # summary
    summary = []
    for name in LABELS.values():
        tp, fp, fn = agg[name]["tp"], agg[name]["fp"], agg[name]["fn"]
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2*prec*rec/(prec+rec+1e-8)
        mean_lw_dice = agg[name]["dice_sum"] / max(1, agg[name]["dice_n"])
        summary.append({
            "class": name,
            "TP": tp, "FP": fp, "FN": fn,
            "Precision": prec, "Recall": rec, "F1": f1,
            "MeanLesionDice": mean_lw_dice
        })
    sdf = pd.DataFrame(summary)
    sum_csv = out_csv.with_name(out_csv.stem + "_summary.csv")
    sdf.to_csv(sum_csv, index=False)

    print("[OK] wrote:")
    print(" -", out_csv)
    print(" -", sum_csv)
    print(sdf.to_string(index=False))

if __name__ == "__main__":
    main()
