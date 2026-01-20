import argparse
import json
import re
from pathlib import Path

MODS = ["t1c", "t1n", "t2f", "t2w", "seg"]

def guess_patient_id(case_id: str) -> str:
    # BraTS-GLI-03050-101 -> BraTS-GLI-03050
    parts = case_id.split("-")
    if len(parts) >= 2 and parts[-1].isdigit():
        return "-".join(parts[:-1])
    return case_id

def find_mod_file(case_dir: Path, mod: str) -> Path | None:
    # robust patterns (handles hyphen/underscore naming)
    patterns = [
        f"*{mod}*.nii.gz",
        f"*{mod}*.nii",
        f"*_{mod}.nii.gz",
        f"*_{mod}.nii",
        f"*-{mod}.nii.gz",
        f"*-{mod}.nii",
    ]
    for p in patterns:
        hits = sorted(case_dir.glob(p))
        if hits:
            return hits[0]
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", type=str, default="data/raw/brats2024_gli/training_data1_v2")
    ap.add_argument("--out_csv", type=str, default="data/cache/cases.csv")
    ap.add_argument("--out_json", type=str, default="data/cache/index_report.json")
    args = ap.parse_args()

    raw_root = Path(args.raw_root)
    if not raw_root.exists():
        raise FileNotFoundError(f"raw_root not found: {raw_root}")

    case_dirs = sorted([p for p in raw_root.iterdir() if p.is_dir()])
    rows = []
    missing = []

    for d in case_dirs:
        case_id = d.name
        patient_id = guess_patient_id(case_id)

        rec = {"case_id": case_id, "patient_id": patient_id, "case_dir": str(d)}
        ok = True
        for m in MODS:
            f = find_mod_file(d, m)
            rec[m] = "" if f is None else str(f)
            if f is None:
                ok = False

        if not ok:
            missing.append(case_id)
        rows.append(rec)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    cols = ["case_id", "patient_id", "case_dir"] + MODS
    with out_csv.open("w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(r.get(c, "") for c in cols) + "\n")

    report = {
        "raw_root": str(raw_root),
        "n_case_dirs": len(case_dirs),
        "n_rows": len(rows),
        "n_missing_any_mod": len(missing),
        "missing_case_ids": missing[:50],  # avoid huge json
        "mods": MODS,
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("[OK] indexed cases")
    print("cases_csv:", out_csv)
    print("report   :", out_json)
    print("summary  :", {k: report[k] for k in ["n_case_dirs","n_missing_any_mod"]})

if __name__ == "__main__":
    main()
