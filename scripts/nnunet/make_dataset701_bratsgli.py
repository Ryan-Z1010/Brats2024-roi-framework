import argparse, json, os
from pathlib import Path

MODS = [
    ("t1n", 0),
    ("t1c", 1),
    ("t2w", 2),
    ("t2f", 3),
]

def ln_sf(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    os.symlink(src.resolve(), dst)

def read_ids(p: Path):
    return [l.strip() for l in p.read_text().splitlines() if l.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", required=True, help=".../training_data1_v2 (contains case folders)")
    ap.add_argument("--train_list", required=True)
    ap.add_argument("--val_list", required=True)
    ap.add_argument("--nnunet_raw", required=True, help="env nnUNet_raw")
    ap.add_argument("--dataset_id", type=int, default=701)
    ap.add_argument("--dataset_name", default="BraTS2024GLI")
    args = ap.parse_args()

    raw_root = Path(args.raw_root)
    train_ids = read_ids(Path(args.train_list))
    val_ids   = read_ids(Path(args.val_list))

    ds_dir = Path(args.nnunet_raw) / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    img_tr = ds_dir / "imagesTr"
    lab_tr = ds_dir / "labelsTr"
    img_ts = ds_dir / "imagesTs"
    img_tr.mkdir(parents=True, exist_ok=True)
    lab_tr.mkdir(parents=True, exist_ok=True)
    img_ts.mkdir(parents=True, exist_ok=True)

    missing = []

    def link_case(cid: str, is_val: bool):
        cdir = raw_root / cid
        if not cdir.is_dir():
            missing.append((cid, "case_dir_missing"))
            return

    # 1) 所有 case 都进 imagesTr（用于训练/验证与预处理）
        out_img_tr = img_tr
        for mod, ch in MODS:
            src = cdir / f"{cid}-{mod}.nii.gz"
            dst = out_img_tr / f"{cid}_{ch:04d}.nii.gz"
            if not src.exists():
                missing.append((cid, f"missing_{mod}"))
                continue
            ln_sf(src, dst)

    # 2) 所有 case 都必须有 label（labelsTr），否则不能用于 val split
        src = cdir / f"{cid}-seg.nii.gz"
        dst = lab_tr / f"{cid}.nii.gz"
        if not src.exists():
            missing.append((cid, "missing_seg"))
        else:
            ln_sf(src, dst)

    # 3) val case 另外再软链一份到 imagesTs，方便 nnUNetv2_predict
        if is_val:
            for mod, ch in MODS:
                src = cdir / f"{cid}-{mod}.nii.gz"
                dst = img_ts / f"{cid}_{ch:04d}.nii.gz"
                if src.exists():
                    ln_sf(src, dst)

    for cid in train_ids:
        link_case(cid, is_val=False)
    for cid in val_ids:
        link_case(cid, is_val=True)

    # dataset.json (nnU-Net v2 needs channel_names + labels + file_ending at minimum)
    dataset_json = {
        "name": args.dataset_name,
        "description": "BraTS 2024 GLI (local split)",
        "reference": "BraTS 2024",
        "licence": "research_only",
        "release": "0.0",
        "tensorImageSize": "3D",
        "channel_names": {str(i): m for m, i in MODS},
        "labels": {
            "background": 0,
            "NETC": 1,
            "SNFH": 2,
            "ET": 3,
            "RC": 4
        },
        "numTraining": len(train_ids) + len(val_ids),
        "file_ending": ".nii.gz",
    }
    (ds_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2), encoding="utf-8")

    print("[OK] Dataset prepared:", ds_dir)
    if missing:
        print("[WARN] Missing items (first 20):")
        for x in missing[:20]:
            print(" ", x)
        print(f"[WARN] Total missing: {len(missing)}")

if __name__ == "__main__":
    main()
