import argparse
import json
import re
from pathlib import Path

import numpy as np

try:
    import nibabel as nib
except Exception as e:
    raise RuntimeError(
        "nibabel is required for raw verification. Install with: pip install nibabel"
    ) from e


MODS = ["t1c", "t1n", "t2f", "t2w", "seg"]
PAT = re.compile(r"^(?P<cid>.+)_(?P<mod>t1c|t1n|t2f|t2w|seg)\.nii(\.gz)?$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", type=str, default="data/raw/brats2024_gli")
    ap.add_argument("--out_json", type=str, default="data/cache/raw_verify_report.json")
    ap.add_argument("--out_csv", type=str, default="data/cache/raw_cases.csv")
    ap.add_argument("--max_cases", type=int, default=0, help="0 means all")
    args = ap.parse_args()

    raw_root = Path(args.raw_root)
    if not raw_root.exists():
        raise FileNotFoundError(f"raw_root not found: {raw_root}")

    files = list(raw_root.rglob("*.nii")) + list(raw_root.rglob("*.nii.gz"))
    by_case = {}

    for f in files:
        m = PAT.match(f.name)
        if not m:
            continue
        cid = m.group("cid")
        mod = m.group("mod")
        by_case.setdefault(cid, {})[mod] = str(f)

    case_ids = sorted(by_case.keys())
    if args.max_cases and args.max_cases > 0:
        case_ids = case_ids[: args.max_cases]

    report = {
        "raw_root": str(raw_root),
        "n_cases_indexed": len(by_case),
        "n_cases_checked": len(case_ids),
        "missing_modalities": [],
        "label_out_of_range": [],
        "shapes_mismatch": [],
        "summary": {},
    }

    rows = []
    allowed_labels = set([0, 1, 2, 3, 4])

    for cid in case_ids:
        mods = by_case[cid]
        missing = [m for m in MODS if m not in mods]
        if missing:
            report["missing_modalities"].append({"case_id": cid, "missing": missing})

        shapes = {}
        labels = None

        for m in MODS:
            if m not in mods:
                continue
            img = nib.load(mods[m])
            arr = img.get_fdata(dtype=np.float32)
            shapes[m] = list(arr.shape)

            if m == "seg":
                seg = nib.load(mods[m]).get_fdata(dtype=np.float32)
                uniq = set(np.unique(seg).astype(np.int32).tolist())
                labels = sorted(list(uniq))
                if not set(labels).issubset(allowed_labels):
                    report["label_out_of_range"].append({"case_id": cid, "labels": labels})

        # shape consistency check among image modalities
        img_shapes = [tuple(shapes[m]) for m in ["t1c", "t1n", "t2f", "t2w"] if m in shapes]
        if len(set(img_shapes)) > 1:
            report["shapes_mismatch"].append({"case_id": cid, "shapes": shapes})

        row = {"case_id": cid}
        for m in MODS:
            row[f"has_{m}"] = int(m in mods)
            if m in shapes:
                row[f"shape_{m}"] = "x".join(map(str, shapes[m]))
        row["seg_labels"] = "" if labels is None else ",".join(map(str, labels))
        rows.append(row)

    # summary
    report["summary"]["n_missing_any"] = len(report["missing_modalities"])
    report["summary"]["n_label_out_of_range"] = len(report["label_out_of_range"])
    report["summary"]["n_shapes_mismatch"] = len(report["shapes_mismatch"])

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # write csv (no pandas dependency)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = sorted({k for r in rows for k in r.keys()})
    with out_csv.open("w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")

    print("[OK] raw verification done")
    print("report:", out_json)
    print("cases :", out_csv)
    print("summary:", report["summary"])


if __name__ == "__main__":
    main()
