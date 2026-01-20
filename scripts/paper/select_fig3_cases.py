# scripts/paper/select_fig3_cases.py
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def load_gt_rc_vox(roi_root: Path, case_id: str) -> int:
    npz_path = roi_root / f"{case_id}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing ROI npz for case {case_id}: {npz_path}")
    d = np.load(npz_path)
    seg = d["seg"]
    return int((seg == 4).sum())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rcmin0_csv", type=str, required=True)
    ap.add_argument("--rcmin120_csv", type=str, required=True)
    ap.add_argument("--out_txt", type=str, required=True)
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--topk", type=int, default=2)

    # NEW: use GT to filter cases
    ap.add_argument("--roi_root", type=str, default=None,
                    help="ROI val root containing {case_id}.npz with GT seg. "
                         "e.g., data/preprocessed/npy_roi128_v1/val")
    ap.add_argument("--min_gt_rc_vox", type=int, default=1,
                    help="Only keep cases with GT_RC_vox >= this value. "
                         "Set 1 to enforce GT has RC; set e.g. 200 for clearer visualization.")
    ap.add_argument("--prefer_gt_present", action="store_true",
                    help="If set, first try selecting from GT-present cases; "
                         "if not enough, fall back to GT-empty cases.")

    args = ap.parse_args()

    a = pd.read_csv(args.rcmin0_csv)
    b = pd.read_csv(args.rcmin120_csv)

    # --- robustly find the id column ---
    # try common names: case_id / id / case
    def get_id_col(df):
        for c in ["case_id", "id", "case"]:
            if c in df.columns:
                return c
        raise KeyError(f"Cannot find case id column in {df.columns}")

    id_col_a = get_id_col(a)
    id_col_b = get_id_col(b)

    # --- robustly find RC dice column ---
    # in your pipeline it is likely column "RC" or "dice_4_rc"
    def get_rc_col(df):
        for c in ["RC", "dice_4_rc"]:
            if c in df.columns:
                return c
        raise KeyError(f"Cannot find RC metric column in {df.columns}")

    rc_col_a = get_rc_col(a)
    rc_col_b = get_rc_col(b)

    a = a[[id_col_a, rc_col_a]].rename(columns={id_col_a: "case_id", rc_col_a: "RC_rcmin0"})
    b = b[[id_col_b, rc_col_b]].rename(columns={id_col_b: "case_id", rc_col_b: "RC_rcmin120"})

    m = a.merge(b, on="case_id", how="inner")
    m["delta_rc"] = m["RC_rcmin120"] - m["RC_rcmin0"]

    # NEW: compute GT_RC_vox if roi_root provided
    if args.roi_root is not None:
        roi_root = Path(args.roi_root)
        gt_list = []
        for cid in m["case_id"].tolist():
            try:
                gt_vox = load_gt_rc_vox(roi_root, cid)
            except Exception as e:
                gt_vox = -1  # mark missing
            gt_list.append(gt_vox)
        m["GT_RC_vox"] = gt_list

        # drop missing
        m = m[m["GT_RC_vox"] >= 0].copy()

        # filtering logic
        gt_present = m[m["GT_RC_vox"] >= args.min_gt_rc_vox].copy()
        gt_empty   = m[m["GT_RC_vox"] <  args.min_gt_rc_vox].copy()

        if args.prefer_gt_present and len(gt_present) >= args.topk:
            cand = gt_present
        elif args.prefer_gt_present and len(gt_present) < args.topk:
            # not enough GT-present cases: take all GT-present plus best GT-empty
            cand = pd.concat([gt_present, gt_empty], axis=0, ignore_index=True)
        else:
            # default: strictly filter
            cand = gt_present

        m_sel = cand.sort_values("delta_rc", ascending=False).head(args.topk).copy()
    else:
        # original behavior (no GT filtering)
        m_sel = m.sort_values("delta_rc", ascending=False).head(args.topk).copy()

    out_txt = Path(args.out_txt)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt, "w") as f:
        for cid in m_sel["case_id"].tolist():
            f.write(cid + "\n")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    m_sel.to_csv(out_csv, index=False)

    print("[OK] selected cases:")
    print(m_sel[["case_id", "RC_rcmin0", "RC_rcmin120", "delta_rc"] + (["GT_RC_vox"] if "GT_RC_vox" in m_sel.columns else [])])

if __name__ == "__main__":
    main()
