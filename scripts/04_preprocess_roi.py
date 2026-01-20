import argparse
import json
import os
import zlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np


def stable_rng(seed: int, case_id: str):
    h = zlib.crc32(case_id.encode("utf-8")) & 0xFFFFFFFF
    return np.random.default_rng(seed + h)


def bbox_from_mask(mask: np.ndarray):
    # mask: (D,H,W) bool
    coords = np.array(np.nonzero(mask))
    if coords.size == 0:
        return None
    mins = coords.min(axis=1)
    maxs = coords.max(axis=1)
    return mins, maxs


def compute_center(seg: np.ndarray):
    m = seg > 0
    bb = bbox_from_mask(m)
    if bb is None:
        # fallback to volume center
        D, H, W = seg.shape
        return np.array([D // 2, H // 2, W // 2], dtype=np.int64), None
    mins, maxs = bb
    center = (mins + maxs) // 2
    return center.astype(np.int64), (mins.astype(np.int64), maxs.astype(np.int64))


def pad_to_fit(arr: np.ndarray, start: np.ndarray, end: np.ndarray, pad_value: float):
    # arr: (C,D,H,W) or (D,H,W)
    is_img = (arr.ndim == 4)
    if is_img:
        _, D, H, W = arr.shape
    else:
        D, H, W = arr.shape

    pads = []
    for i, dim in enumerate([D, H, W]):
        s = int(start[i])
        e = int(end[i])
        pad_l = max(0, -s)
        pad_r = max(0, e - dim)
        pads.append((pad_l, pad_r))

    if is_img:
        pad_width = [(0, 0), pads[0], pads[1], pads[2]]
    else:
        pad_width = [pads[0], pads[1], pads[2]]

    if any(p[0] > 0 or p[1] > 0 for p in pads):
        arr = np.pad(arr, pad_width, mode="constant", constant_values=pad_value)
        start = start + np.array([pads[0][0], pads[1][0], pads[2][0]], dtype=np.int64)
        end = end + np.array([pads[0][0], pads[1][0], pads[2][0]], dtype=np.int64)

    return arr, start, end, pads


def crop_patch(img4: np.ndarray, seg: np.ndarray, center: np.ndarray, size: int):
    # img4: (4,D,H,W), seg: (D,H,W)
    half = size // 2
    start = center - half
    end = start + size

    img4p, start2, end2, pads = pad_to_fit(img4, start.copy(), end.copy(), pad_value=0.0)
    segp, start3, end3, pads2 = pad_to_fit(seg, start.copy(), end.copy(), pad_value=0)

    # after padding, starts should match
    start = start2
    end = end2

    d0, d1 = int(start[0]), int(end[0])
    h0, h1 = int(start[1]), int(end[1])
    w0, w1 = int(start[2]), int(end[2])

    img_patch = img4p[:, d0:d1, h0:h1, w0:w1]
    seg_patch = segp[d0:d1, h0:h1, w0:w1].astype(np.uint8)

    assert img_patch.shape[1:] == (size, size, size), img_patch.shape
    assert seg_patch.shape == (size, size, size), seg_patch.shape

    return img_patch.astype(np.float32), seg_patch, pads


def process_one(in_path: str, out_dir: str, size: int, jitter: int, seed: int, overwrite: bool):
    in_path = Path(in_path)
    case_id = in_path.stem
    out_dir = Path(out_dir)
    out_path = out_dir / f"{case_id}.npz"

    if out_path.exists() and not overwrite:
        return {"case_id": case_id, "status": "skip_exists", "out": str(out_path)}

    data = np.load(in_path)
    img = data["img"].astype(np.float32)   # (4,D,H,W)
    seg = data["seg"].astype(np.uint8)     # (D,H,W)

    center, bbox = compute_center(seg)
    orig_center = center.copy()

    shift = np.array([0, 0, 0], dtype=np.int64)
    if jitter > 0:
        rng = stable_rng(seed, case_id)
        shift = rng.integers(-jitter, jitter + 1, size=3, dtype=np.int64)
        center = center + shift

    img_patch, seg_patch, pads = crop_patch(img, seg, center, size=size)
    uniq = np.unique(seg_patch).astype(np.int32).tolist()

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = str(out_path) + ".tmp.npz"
    np.savez_compressed(tmp, img=img_patch, seg=seg_patch)
    os.replace(tmp, out_path)

    rec = {
        "case_id": case_id,
        "status": "ok",
        "out": str(out_path),
        "in": str(in_path),
        "roi_size": size,
        "orig_shape": list(seg.shape),
        "patch_shape": list(seg_patch.shape),
        "center": [int(x) for x in center],
        "center_nojitter": [int(x) for x in orig_center],
        "jitter_shift": [int(x) for x in shift],
        "pads": pads,
        "seg_labels": uniq,
    }
    if bbox is not None:
        mins, maxs = bbox
        rec["bbox_min"] = [int(x) for x in mins]
        rec["bbox_max"] = [int(x) for x in maxs]
    else:
        rec["bbox_min"] = None
        rec["bbox_max"] = None

    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=str, required=True, choices=["train", "val"])
    ap.add_argument("--full_root", type=str, default="data/preprocessed/npy_full_v1")
    ap.add_argument("--out_root", type=str, default="data/preprocessed/npy_roi128_v1")
    ap.add_argument("--roi_size", type=int, default=128)
    ap.add_argument("--jitter", type=int, default=0, help="voxels, baseline=0; later for robustness")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--max_cases", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    in_dir = Path(args.full_root) / args.split
    out_dir = Path(args.out_root) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*.npz"))
    if args.max_cases and args.max_cases > 0:
        files = files[: args.max_cases]
    if not files:
        raise RuntimeError(f"No npz found in: {in_dir}")

    meta = {
        "split": args.split,
        "full_root": str(in_dir),
        "out_root": str(out_dir),
        "roi_size": args.roi_size,
        "jitter": args.jitter,
        "seed": args.seed,
        "num_workers": args.num_workers,
    }
    (out_dir / "preprocess_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    stats_csv = out_dir / "stats.csv"
    header = ["case_id","status","out","center","center_nojitter","jitter_shift","bbox_min","bbox_max","pads","seg_labels"]
    stats_csv.write_text(",".join(header) + "\n", encoding="utf-8")

    results = []
    with ProcessPoolExecutor(max_workers=max(1, args.num_workers)) as ex:
        futs = [ex.submit(process_one, str(p), str(out_dir), args.roi_size, args.jitter, args.seed, args.overwrite) for p in files]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results.append(r)
            row = [
                r.get("case_id",""),
                r.get("status",""),
                r.get("out",""),
                '"' + json.dumps(r.get("center", None)) + '"',
                '"' + json.dumps(r.get("center_nojitter", None)) + '"',
                '"' + json.dumps(r.get("jitter_shift", None)) + '"',
                '"' + json.dumps(r.get("bbox_min", None)) + '"',
                '"' + json.dumps(r.get("bbox_max", None)) + '"',
                '"' + json.dumps(r.get("pads", None)) + '"',
                '"' + json.dumps(r.get("seg_labels", None)) + '"',
            ]
            with stats_csv.open("a", encoding="utf-8") as f:
                f.write(",".join(row) + "\n")

            if i % 50 == 0 or i == len(files):
                print(f"[progress] {i}/{len(files)} done")

    manifest = {
        "meta": meta,
        "n_total": len(files),
        "n_ok": sum(1 for r in results if r["status"] == "ok"),
        "n_skip": sum(1 for r in results if r["status"] == "skip_exists"),
        "outputs_dir": str(out_dir),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("[OK] ROI preprocess done")
    print("outputs_dir:", out_dir)
    print("manifest   :", out_dir / "manifest.json")
    print("stats_csv  :", stats_csv)
    print("summary    :", manifest)


if __name__ == "__main__":
    main()
