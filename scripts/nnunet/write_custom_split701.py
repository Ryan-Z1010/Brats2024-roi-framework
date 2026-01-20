import argparse, json
from pathlib import Path

def read_ids(p: Path):
    return [l.strip() for l in p.read_text().splitlines() if l.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nnunet_preprocessed", required=True)
    ap.add_argument("--dataset_id", type=int, default=701)
    ap.add_argument("--dataset_name", default="BraTS2024GLI")
    ap.add_argument("--train_list", required=True)
    ap.add_argument("--val_list", required=True)
    args = ap.parse_args()

    pp_dir = Path(args.nnunet_preprocessed) / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    pp_dir.mkdir(parents=True, exist_ok=True)

    train_ids = read_ids(Path(args.train_list))
    val_ids   = read_ids(Path(args.val_list))

    splits = [{"train": train_ids, "val": val_ids}]  # 只有 1 个 split -> fold0
    out = pp_dir / "splits_final.json"
    out.write_text(json.dumps(splits, indent=2), encoding="utf-8")
    print("[OK] wrote:", out)
    print(" fold0: train =", len(train_ids), "val =", len(val_ids))

if __name__ == "__main__":
    main()
