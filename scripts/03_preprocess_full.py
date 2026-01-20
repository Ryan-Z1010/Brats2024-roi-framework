import argparse
import json
import math
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import nibabel as nib

MODS = ["t1c", "t1n", "t2f", "t2w"]

def json_safe(x):
    """Convert numpy scalars/arrays to pure Python types for json.dumps."""
    if x is None:
        return None
    # numpy scalar -> python scalar
    if hasattr(x, "item") and callable(x.item):
        try:
            return x.item()
        except Exception:
            pass
    # dict
    if isinstance(x, dict):
        return {k: json_safe(v) for k, v in x.items()}
    # list/tuple
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    return x

def load_nifti(path: str):
    img = nib.load(path)
    arr = img.get_fdata(dtype=np.float32)
    zooms = img.header.get_zooms()[:3]
    return arr, zooms

def robust_norm(x: np.ndarray, mask: np.ndarray, pmin: float, pmax: float, eps: float = 1e-8):
    """clip by percentiles (within mask) then z-score (within mask)."""
    if mask.sum() == 0:
        return x.astype(np.float32), {"mean": 0.0, "std": 1.0, "pmin": 0.0, "pmax": 0.0}

    vals = x[mask]
    lo = np.percentile(vals, pmin)
    hi = np.percentile(vals, pmax)
    xc = np.clip(x, lo, hi)

    vals2 = xc[mask]
    mu = float(vals2.mean())
    sd = float(vals2.std())
    xn = (xc - mu) / (sd + eps)
    return xn.astype(np.float32), {"mean": mu, "std": sd, "pmin": float(lo), "pmax": float(hi)}

def read_cases_csv(path: Path):
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    header = lines[0].split(",")
    idx = {k: header.index(k) for k in (["case_id","patient_id"] + MODS + ["seg"])}
    rows = []
    for ln in lines[1:]:
        parts = ln.split(",")
        if len(parts) < max(idx.values()) + 1:
            continue
        rec = {k: parts[i] for k,i in idx.items()}
        rows.append(rec)
    return rows

def load_split_ids(split_txt: Path):
    return set([l.strip() for l in split_txt.read_text(encoding="utf-8").splitlines() if l.strip()])

def process_one(rec: dict, out_dir: Path, pmin: float, pmax: float, overwrite: bool):
    case_id = rec["case_id"]
    out_path = out_dir / f"{case_id}.npz"
    if out_path.exists() and not overwrite:
        return {"case_id": case_id, "status": "skip_exists", "out": str(out_path)}

    # load modalities
    imgs = []
    zooms_ref = None
    shapes = {}
    for m in MODS:
        arr, zooms = load_nifti(rec[m])
        imgs.append(arr)
        shapes[m] = list(arr.shape)
        if zooms_ref is None:
            zooms_ref = zooms

    # brain-ish mask: union of nonzero across modalities
    stack = np.stack(imgs, axis=0)  # (4,D,H,W)
    mask = np.any(stack != 0, axis=0)

    normed = []
    norm_stats = {}
    for i, m in enumerate(MODS):
        xn, st = robust_norm(stack[i], mask, pmin, pmax)
        normed.append(xn)
        norm_stats[m] = st

    img4 = np.stack(normed, axis=0).astype(np.float32)

    # load seg as uint8
    seg, zooms_seg = load_nifti(rec["seg"])
    seg_u = seg.astype(np.uint8)
    uniq = np.unique(seg_u).astype(np.int32).tolist()

    # quick safety check
    if not set(uniq).issubset({0,1,2,3,4}):
        raise ValueError(f"[{case_id}] seg labels out of range: {uniq}")

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = str(out_path) + ".tmp.npz"
    np.savez_compressed(tmp, img=img4, seg=seg_u)

    os.replace(tmp, out_path)

    return {
        "case_id": case_id,
        "status": "ok",
        "out": str(out_path),
        "shape": list(img4.shape),
        "zooms": list(zooms_ref) if zooms_ref is not None else None,
        "seg_labels": uniq,
        "norm": norm_stats,
        "raw_shapes": shapes,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases_csv", type=str, default="data/cache/cases.csv")
    ap.add_argument("--split", type=str, default="all", choices=["all","train","val"])
    ap.add_argument("--splits_dir", type=str, default="data/splits")
    ap.add_argument("--out_dir", type=str, default="data/preprocessed/npy_full_v1")
    ap.add_argument("--pmin", type=float, default=0.5)
    ap.add_argument("--pmax", type=float, default=99.5)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--max_cases", type=int, default=0, help="0 means all")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cases = read_cases_csv(Path(args.cases_csv))

    if args.split != "all":
        ids = load_split_ids(Path(args.splits_dir) / f"{args.split}.txt")
        cases = [r for r in cases if r["case_id"] in ids]

    if args.max_cases and args.max_cases > 0:
        cases = cases[: args.max_cases]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "cases_csv": args.cases_csv,
        "split": args.split,
        "pmin": args.pmin,
        "pmax": args.pmax,
        "num_workers": args.num_workers,
    }
    (out_dir / "preprocess_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    results = []
    stats_csv = out_dir / "stats.csv"
    header = ["case_id","status","out","shape","zooms","seg_labels"]
    stats_csv.write_text(",".join(header) + "\n", encoding="utf-8")

    with ProcessPoolExecutor(max_workers=max(1, args.num_workers)) as ex:
        futs = [ex.submit(process_one, r, out_dir, args.pmin, args.pmax, args.overwrite) for r in cases]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results.append(r)
            row = [
                r.get("case_id",""),
                r.get("status",""),
                r.get("out",""),
                '"' + json.dumps(json_safe(r.get("shape", None))) + '"',
                '"' + json.dumps(json_safe(r.get("zooms", None))) + '"',
                '"' + json.dumps(json_safe(r.get("seg_labels", None))) + '"',
            ]
            with stats_csv.open("a", encoding="utf-8") as f:
                f.write(",".join(row) + "\n")
            if i % 50 == 0 or i == len(cases):
                print(f"[progress] {i}/{len(cases)} done")

    manifest = {
        "meta": meta,
        "n_total": len(cases),
        "n_ok": sum(1 for r in results if r["status"] == "ok"),
        "n_skip": sum(1 for r in results if r["status"] == "skip_exists"),
        "outputs_dir": str(out_dir),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("[OK] preprocess full done")
    print("outputs_dir:", out_dir)
    print("manifest   :", out_dir / "manifest.json")
    print("stats_csv  :", stats_csv)
    print("summary    :", manifest)

if __name__ == "__main__":
    main()
