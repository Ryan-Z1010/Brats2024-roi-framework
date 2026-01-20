import argparse, json, csv
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from src.models.unet3d_res import ResUNet3D

def bbox_from_mask(mask: np.ndarray):
    if not mask.any():
        return None
    z, y, x = np.where(mask)
    return (int(z.min()), int(z.max()), int(y.min()), int(y.max()), int(x.min()), int(x.max()))

def map_center_coarse_to_full(center_zyx, full_shape, coarse_size):
    cz, cy, cx = center_zyx
    D, H, W = full_shape
    s = float(coarse_size - 1)
    zf = int(round(cz * (D - 1) / s))
    yf = int(round(cy * (H - 1) / s))
    xf = int(round(cx * (W - 1) / s))
    zf = max(0, min(D - 1, zf))
    yf = max(0, min(H - 1, yf))
    xf = max(0, min(W - 1, xf))
    return (zf, yf, xf)

def roi_slices_from_center(center_zyx, full_shape, roi_size):
    cz, cy, cx = center_zyx
    D, H, W = full_shape
    half = roi_size // 2
    sz, ez = cz - half, cz - half + roi_size
    sy, ey = cy - half, cy - half + roi_size
    sx, ex = cx - half, cx - half + roi_size
    fz0, fz1 = max(0, sz), min(D, ez)
    fy0, fy1 = max(0, sy), min(H, ey)
    fx0, fx1 = max(0, sx), min(W, ex)
    return (fz0, fz1, fy0, fy1, fx0, fx1)

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1_ckpt", required=True)
    ap.add_argument("--full_root", required=True, help="data/preprocessed/npy_full_v1/(train|val)")
    ap.add_argument("--split_list", required=True, help="data/splits/(train|val).txt")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--coarse_size", type=int, default=96)
    ap.add_argument("--roi_size", type=int, default=128)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--base_channels", type=int, default=16)
    ap.add_argument("--eval_coverage", action="store_true", help="if GT seg exists in full npz, compute coverage stats (val)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # stage1 model: ResUNet3D out_channels=2, we use channel0 as logit
    model = ResUNet3D(in_channels=4, out_channels=2, base=args.base_channels, dropout=0.0).to(device)
    ck = torch.load(args.stage1_ckpt, map_location="cpu")
    model.load_state_dict(ck["model"], strict=True)
    model.eval()

    case_ids = [l.strip() for l in Path(args.split_list).read_text().splitlines() if l.strip()]
    full_root = Path(args.full_root)

    out_csv = Path(args.out_csv); out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json = Path(args.out_json); out_json.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    n_fallback = 0
    cov_list = []
    empty_gt = 0

    for i, cid in enumerate(case_ids, 1):
        f = full_root / f"{cid}.npz"
        d = np.load(f)
        img = d["img"].astype(np.float32)  # (4,D,H,W)
        full_shape = tuple(img.shape[1:])  # (D,H,W)

        # downsample to coarse_size^3
        x = torch.from_numpy(img).unsqueeze(0).to(device)  # (1,4,D,H,W)
        x_ds = F.interpolate(x, size=(args.coarse_size, args.coarse_size, args.coarse_size),
                             mode="trilinear", align_corners=False)

        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            logits = model(x_ds)[:, :1]  # (1,1,96,96,96)
            prob = torch.sigmoid(logits).squeeze(0).squeeze(0).float().cpu().numpy()

        mask = (prob >= args.thr)
        bb = bbox_from_mask(mask)

        if bb is None:
            # fallback: center of volume
            center_full = (full_shape[0]//2, full_shape[1]//2, full_shape[2]//2)
            n_fallback += 1
            center_coarse = (args.coarse_size//2, args.coarse_size//2, args.coarse_size//2)
        else:
            zmin,zmax,ymin,ymax,xmin,xmax = bb
            center_coarse = ((zmin+zmax)//2, (ymin+ymax)//2, (xmin+xmax)//2)
            center_full = map_center_coarse_to_full(center_coarse, full_shape, args.coarse_size)

        sl = roi_slices_from_center(center_full, full_shape, args.roi_size)  # (z0,z1,y0,y1,x0,x1)

        cov = None
        if args.eval_coverage and ("seg" in d):
            gt = d["seg"].astype(np.uint8)
            tumor = (gt > 0)
            denom = int(tumor.sum())
            if denom == 0:
                empty_gt += 1
                cov = 1.0
            else:
                z0,z1,y0,y1,x0,x1 = sl
                inside = int(tumor[z0:z1, y0:y1, x0:x1].sum())
                cov = inside / float(denom)
            cov_list.append(cov)

        rows.append({
            "case_id": cid,
            "center_z": int(center_full[0]),
            "center_y": int(center_full[1]),
            "center_x": int(center_full[2]),
            "z0": int(sl[0]), "z1": int(sl[1]),
            "y0": int(sl[2]), "y1": int(sl[3]),
            "x0": int(sl[4]), "x1": int(sl[5]),
            "fallback": int(bb is None),
            "coverage": ("" if cov is None else f"{cov:.6f}"),
        })

        if i % 50 == 0 or i == len(case_ids):
            print(f"[progress] {i}/{len(case_ids)} done")

    # write CSV
    fields = ["case_id","center_z","center_y","center_x","z0","z1","y0","y1","x0","x1","fallback","coverage"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # stats
    stats = {
        "n_cases": len(rows),
        "n_fallback_center": int(n_fallback),
        "coarse_size": int(args.coarse_size),
        "roi_size": int(args.roi_size),
        "thr": float(args.thr),
        "stage1_ckpt": str(args.stage1_ckpt),
        "full_root": str(full_root),
        "split_list": str(args.split_list),
        "eval_coverage": bool(args.eval_coverage),
        "n_empty_gt": int(empty_gt),
    }
    if cov_list:
        stats["coverage_mean"] = float(np.mean(cov_list))
        stats["coverage_p05"]  = float(np.quantile(cov_list, 0.05))
        stats["coverage_p10"]  = float(np.quantile(cov_list, 0.10))
        stats["coverage_min"]  = float(np.min(cov_list))

    out_json.write_text(json.dumps({"stats": stats}, indent=2), encoding="utf-8")

    print("[OK] ROI proposals generated")
    print("csv :", out_csv)
    print("json:", out_json)
    print("stats:", stats)

if __name__ == "__main__":
    main()
