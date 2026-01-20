import argparse, json, csv
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

def downsample_img(img4d: np.ndarray, out_size: int) -> np.ndarray:
    # img4d: (4,D,H,W) float32
    x = torch.from_numpy(img4d).unsqueeze(0)  # (1,4,D,H,W)
    x = F.interpolate(x, size=(out_size,out_size,out_size), mode="trilinear", align_corners=False)
    return x.squeeze(0).numpy().astype(np.float32)

def downsample_mask(mask3d: np.ndarray, out_size: int) -> np.ndarray:
    # mask3d: (D,H,W) uint8 -> binary
    x = torch.from_numpy((mask3d > 0).astype(np.float32)).unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
    x = F.interpolate(x, size=(out_size,out_size,out_size), mode="nearest")
    return x.squeeze(0).squeeze(0).numpy().astype(np.uint8)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full_root", required=True, help="data/preprocessed/npy_full_v1")
    ap.add_argument("--split", required=True, choices=["train","val"])
    ap.add_argument("--split_list", required=True, help="data/splits/train.txt or val.txt")
    ap.add_argument("--out_root", required=True, help="data/preprocessed/npy_coarse96_v1")
    ap.add_argument("--size", type=int, default=96)
    args = ap.parse_args()

    full_root = Path(args.full_root) / args.split
    out_root = Path(args.out_root) / args.split
    out_root.mkdir(parents=True, exist_ok=True)

    case_ids = [l.strip() for l in Path(args.split_list).read_text().splitlines() if l.strip()]
    rows = []
    for i, cid in enumerate(case_ids, 1):
        f = full_root / f"{cid}.npz"
        d = np.load(f)
        img = d["img"].astype(np.float32)   # (4,182,218,182)
        seg = d["seg"].astype(np.uint8)     # (182,218,182)

        img_ds = downsample_img(img, args.size)
        m_ds   = downsample_mask(seg, args.size)

        out_f = out_root / f"{cid}.npz"
        np.savez_compressed(out_f, img=img_ds, mask=m_ds)
        rows.append([cid, str(out_f), list(img.shape[1:]), list(img_ds.shape[1:])])

        if i % 50 == 0 or i == len(case_ids):
            print(f"[progress] {i}/{len(case_ids)} done")

    manifest = out_root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case_id","path","full_shape","coarse_shape"])
        w.writerows(rows)

    meta = {"full_root": str(full_root), "split": args.split, "size": args.size, "n": len(rows)}
    (out_root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("[OK] coarse preprocessing done")
    print("out_root :", out_root)
    print("manifest :", manifest)

if __name__ == "__main__":
    main()
