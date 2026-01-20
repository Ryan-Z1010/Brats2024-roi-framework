import argparse
import random
from pathlib import Path

def read_cases_csv(path: Path):
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    header = lines[0].split(",")
    idx_case = header.index("case_id")
    idx_patient = header.index("patient_id")
    rows = []
    for ln in lines[1:]:
        parts = ln.split(",")
        if len(parts) < max(idx_case, idx_patient) + 1:
            continue
        rows.append((parts[idx_case], parts[idx_patient]))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases_csv", type=str, default="data/cache/cases.csv")
    ap.add_argument("--out_dir", type=str, default="data/splits")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cases_csv = Path(args.cases_csv)
    if not cases_csv.exists():
        raise FileNotFoundError(f"cases_csv not found: {cases_csv}")

    rows = read_cases_csv(cases_csv)

    # group by patient
    by_patient = {}
    for case_id, patient_id in rows:
        by_patient.setdefault(patient_id, []).append(case_id)

    patient_ids = sorted(by_patient.keys())
    rng = random.Random(args.seed)
    rng.shuffle(patient_ids)

    n_val_pat = max(1, int(round(len(patient_ids) * args.val_frac)))
    val_pat = set(patient_ids[:n_val_pat])

    train_cases, val_cases = [], []
    for pid, case_list in by_patient.items():
        case_list = sorted(case_list)
        if pid in val_pat:
            val_cases.extend(case_list)
        else:
            train_cases.extend(case_list)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_txt = out_dir / "train.txt"
    val_txt = out_dir / "val.txt"

    if (train_txt.exists() or val_txt.exists()) and not args.overwrite:
        raise RuntimeError("split files exist. Use --overwrite to overwrite.")

    train_txt.write_text("\n".join(train_cases) + "\n", encoding="utf-8")
    val_txt.write_text("\n".join(val_cases) + "\n", encoding="utf-8")

    print("[OK] splits generated")
    print("seed:", args.seed, "val_frac:", args.val_frac)
    print("n_patients:", len(patient_ids), "n_val_patients:", n_val_pat)
    print("train cases:", len(train_cases), "val cases:", len(val_cases))
    print("train_txt:", train_txt)
    print("val_txt  :", val_txt)

if __name__ == "__main__":
    main()
