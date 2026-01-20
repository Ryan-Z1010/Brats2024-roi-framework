import os, numpy as np, pandas as pd, nibabel as nib
from skimage import measure
from tqdm import tqdm
import argparse

def analyze_case(gt_path, pred_path):
    gt = np.asanyarray(nib.load(gt_path).dataobj)
    pred = np.asanyarray(nib.load(pred_path).dataobj)
    gt_rc = (gt == 4).astype(np.uint8)
    pred_rc = (pred == 4).astype(np.uint8)
    
    # 统计 GT 与预测的连通域
    cc_gt = measure.label(gt_rc, connectivity=1)
    cc_pred = measure.label(pred_rc, connectivity=1)
    n_gt = cc_gt.max()
    n_pred = cc_pred.max()
    
    # 找 FP 连通域：与 GT 无交集
    fp_cc = 0
    fp_vox = 0
    tp_cc = 0
    for i in range(1, n_pred+1):
        comp = (cc_pred == i)
        if (gt_rc & comp).sum() == 0:
            fp_cc += 1
            fp_vox += comp.sum()
        else:
            tp_cc += 1

    # 最大连通域占比
    if n_pred > 0:
        sizes = [(cc_pred == i).sum() for i in range(1, n_pred+1)]
        largest_cc_ratio = max(sizes) / sum(sizes)
    else:
        largest_cc_ratio = 0

    return dict(
        n_gt=n_gt,
        n_pred=n_pred,
        fp_cc=fp_cc,
        fp_vox=fp_vox,
        tp_cc=tp_cc,
        largest_cc_ratio=largest_cc_ratio
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_root", type=str, default="data/preprocessed/npy_full_v1/val_nifti")
    ap.add_argument("--pred_root0", required=True, help="Dir for rcmin0 predictions")
    ap.add_argument("--pred_root120", required=True, help="Dir for rcmin120 predictions")
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    # 列出所有病例
    cases = [f.replace(".nii.gz","") for f in os.listdir(args.pred_root0) if f.endswith(".nii.gz")]
    results = []

    for case in tqdm(cases):
        gt_path = os.path.join(args.gt_root, f"{case}_seg.nii.gz")
        pred0 = os.path.join(args.pred_root0, f"{case}_pred.nii.gz")
        pred120 = os.path.join(args.pred_root120, f"{case}_pred.nii.gz")
        if not (os.path.exists(gt_path) and os.path.exists(pred0) and os.path.exists(pred120)):
            continue

        m0 = analyze_case(gt_path, pred0)
        m120 = analyze_case(gt_path, pred120)
        row = dict(case=case)
        for k in m0:
            row[f"{k}_rcmin0"] = m0[k]
            row[f"{k}_rcmin120"] = m120[k]
            row[f"delta_{k}"] = m120[k] - m0[k]
        results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(args.out_csv, index=False)
    print(f"[OK] saved: {args.out_csv}")
    print(df.describe().round(3))

if __name__ == "__main__":
    main()
