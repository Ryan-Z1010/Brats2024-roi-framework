import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from src.models.unet3d_res import ResUNet3D


def crop_bounds_from_center(center_zyx, full_shape, roi_size):
    cz, cy, cx = [int(v) for v in center_zyx]
    D, H, W = [int(v) for v in full_shape]
    half = roi_size // 2

    sz, ez = cz - half, cz - half + roi_size
    sy, ey = cy - half, cy - half + roi_size
    sx, ex = cx - half, cx - half + roi_size

    z0, z1 = max(0, sz), min(D, ez)
    y0, y1 = max(0, sy), min(H, ey)
    x0, x1 = max(0, sx), min(W, ex)
    return z0, z1, y0, y1, x0, x1


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1_ckpt", required=True)
    ap.add_argument("--stage1_config", required=True)
    ap.add_argument("--coarse_root", required=True, help="e.g. data/preprocessed/npy_coarse96_split2_v1")
    ap.add_argument("--full_root", required=True, help="e.g. data/preprocessed/npy_full_split2_v1")
    ap.add_argument("--split_list", required=True, help="e.g. data/splits_split2/val.txt")
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--thr", type=float, default=0.35)
    ap.add_argument("--roi_size", type=int, default=128)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.stage1_config).read_text(encoding="utf-8"))
    base_channels = int(cfg["model"]["base_channels"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResUNet3D(in_channels=4, out_channels=2, base=base_channels, dropout=0.0).to(device)
    ck = torch.load(args.stage1_ckpt, map_location="cpu")
    model.load_state_dict(ck["model"], strict=True)
    model.eval()

    coarse_dir = Path(args.coarse_root) / args.split
    full_dir = Path(args.full_root) / args.split
    case_ids = [x.strip() for x in Path(args.split_list).read_text(encoding="utf-8").splitlines() if x.strip()]

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    coverages = []

    for i, cid in enumerate(case_ids, 1):
        coarse_npz = coarse_dir / f"{cid}.npz"
        full_npz = full_dir / f"{cid}.npz"

        d_coarse = np.load(coarse_npz)
        d_full = np.load(full_npz)

        img96 = d_coarse["img"].astype(np.float32)      # (4,96,96,96)
        gt_full = d_full["seg"].astype(np.uint8)        # (182,218,182) typically
        full_shape = gt_full.shape

        x = torch.from_numpy(img96).unsqueeze(0).to(device)
        logits = model(x)[:, :1]                        # (1,1,96,96,96)
        prob96 = torch.sigmoid(logits)                  # (1,1,96,96,96)

        prob_full = F.interpolate(
            prob96,
            size=full_shape,
            mode="trilinear",
            align_corners=False
        )[0, 0].detach().cpu().numpy().astype(np.float32)

        fg = prob_full >= float(args.thr)
        fallback = 0

        if fg.any():
            coords = np.argwhere(fg)  # (N,3)
            weights = prob_full[fg].astype(np.float64)
            denom = float(weights.sum()) if float(weights.sum()) > 0 else float(len(weights))
            center = np.round((coords.astype(np.float64) * weights[:, None]).sum(axis=0) / denom).astype(int)
        else:
            fallback = 1
            center = np.array([full_shape[0] // 2, full_shape[1] // 2, full_shape[2] // 2], dtype=int)

        center[0] = int(np.clip(center[0], 0, full_shape[0] - 1))
        center[1] = int(np.clip(center[1], 0, full_shape[1] - 1))
        center[2] = int(np.clip(center[2], 0, full_shape[2] - 1))

        z0, z1, y0, y1, x0, x1 = crop_bounds_from_center(center, full_shape, int(args.roi_size))

        gt_fg = (gt_full > 0)
        gt_total = int(gt_fg.sum())
        if gt_total > 0:
            gt_in = int(gt_fg[z0:z1, y0:y1, x0:x1].sum())
            coverage = gt_in / gt_total
        else:
            coverage = 1.0

        rows.append({
            "case_id": cid,
            "center_z": int(center[0]),
            "center_y": int(center[1]),
            "center_x": int(center[2]),
            "z0": int(z0),
            "z1": int(z1),
            "y0": int(y0),
            "y1": int(y1),
            "x0": int(x0),
            "x1": int(x1),
            "fallback": int(fallback),
            "coverage": f"{coverage:.6f}",
        })
        coverages.append(float(coverage))

        if i % 20 == 0 or i == len(case_ids):
            print(f"[progress] {i}/{len(case_ids)} done")

    fieldnames = [
        "case_id", "center_z", "center_y", "center_x",
        "z0", "z1", "y0", "y1", "x0", "x1",
        "fallback", "coverage"
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    summary = {
        "stage1_ckpt": str(args.stage1_ckpt),
        "stage1_config": str(args.stage1_config),
        "coarse_root": str(coarse_dir),
        "full_root": str(full_dir),
        "split_list": str(args.split_list),
        "split": args.split,
        "thr": float(args.thr),
        "roi_size": int(args.roi_size),
        "n_cases": len(rows),
        "coverage_mean": float(np.mean(coverages)) if coverages else None,
        "coverage_min": float(np.min(coverages)) if coverages else None,
        "n_fallback": int(sum(int(r["fallback"]) for r in rows)),
        "out_csv": str(out_csv),
    }
    summary_path = out_csv.with_name("summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[OK] proposal export done")
    print("csv     :", out_csv)
    print("summary :", summary_path)
    print("mean coverage:", summary["coverage_mean"])
    print("min  coverage:", summary["coverage_min"])
    print("n_fallback   :", summary["n_fallback"])


if __name__ == "__main__":
    main()
