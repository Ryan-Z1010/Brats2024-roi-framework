import argparse, json, csv, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from src.models.unet3d_res import ResUNet3D

try:
    from scipy.ndimage import label as cc_label
except Exception:
    cc_label = None

def dice_bool(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool); b = b.astype(bool)
    ia = int(a.sum()); ib = int(b.sum())
    if ia == 0 and ib == 0: return 1.0
    if ia == 0 or ib == 0: return 0.0
    inter = int((a & b).sum())
    return (2.0 * inter) / (ia + ib)

def compute_metrics_from_labels(pred: np.ndarray, gt: np.ndarray) -> dict:
    m = {}
    m["dice_1_netc"] = dice_bool(pred == 1, gt == 1)
    m["dice_2_snf_h"] = dice_bool(pred == 2, gt == 2)
    m["dice_3_et"]   = dice_bool(pred == 3, gt == 3)
    m["dice_4_rc"]   = dice_bool(pred == 4, gt == 4)

    WT  = ((pred==1)|(pred==2)|(pred==3)); WTg = ((gt==1)|(gt==2)|(gt==3))
    TC  = ((pred==1)|(pred==3));          TCg = ((gt==1)|(gt==3))
    ET  = (pred==3);                       ETg = (gt==3)
    RC  = (pred==4);                       RCg = (gt==4)

    m["WT"] = dice_bool(WT, WTg)
    m["TC"] = dice_bool(TC, TCg)
    m["ET"] = dice_bool(ET, ETg)
    m["RC"] = dice_bool(RC, RCg)
    m["mean_wt_tc_et"] = (m["WT"] + m["TC"] + m["ET"]) / 3.0
    return m

def remove_small_cc(mask: np.ndarray, min_vox: int) -> np.ndarray:
    if min_vox <= 0:
        return mask
    if cc_label is None:
        raise RuntimeError("scipy not available; pip install scipy")
    lab, n = cc_label(mask.astype(np.uint8))
    if n == 0:
        return mask
    out = np.zeros_like(mask, dtype=bool)
    for i in range(1, n + 1):
        comp = (lab == i)
        if int(comp.sum()) >= int(min_vox):
            out |= comp
    return out

def postprocess_rc_only(pred_lbl: np.ndarray, rc_min_vox: int) -> np.ndarray:
    if rc_min_vox <= 0:
        return pred_lbl
    out = pred_lbl.copy()
    rc = (out == 4)
    rc2 = remove_small_cc(rc, rc_min_vox)
    out[rc & (~rc2)] = 0
    out[rc2] = 4
    return out

@torch.no_grad()
def predict_probs_ensemble(models, x):
    ps = None
    for m in models:
        logits = m(x)
        p = F.softmax(logits, dim=1)
        ps = p if ps is None else (ps + p)
    return ps / float(len(models))

def gen_starts(L, win, stride):
    if L <= win:
        return [0]
    starts = list(range(0, L - win + 1, stride))
    if starts[-1] != L - win:
        starts.append(L - win)
    return starts

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full_root", required=True, help="data/preprocessed/npy_full_v1/val")
    ap.add_argument("--val_list", required=True, help="data/splits/val.txt")
    ap.add_argument("--ckpts", required=True, help="comma-separated best.pt paths (>=1)")
    ap.add_argument("--base_channels", type=int, default=48)
    ap.add_argument("--roi_size", type=int, default=128)
    ap.add_argument("--stride", type=int, default=64, help="sliding window stride (e.g., 64 for 50% overlap)")
    ap.add_argument("--rc_min_vox", type=int, default=0)
    ap.add_argument("--save_nii", action="store_true")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpts = [c.strip() for c in args.ckpts.split(",") if c.strip()]
    models = []
    for p in ckpts:
        m = ResUNet3D(in_channels=4, out_channels=5, base=args.base_channels, dropout=0.0).to(device)
        ck = torch.load(p, map_location="cpu")
        m.load_state_dict(ck["model"], strict=True)
        m.eval()
        models.append(m)

    case_ids = [l.strip() for l in Path(args.val_list).read_text().splitlines() if l.strip()]
    full_root = Path(args.full_root)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_nii:
        (out_dir / "nii").mkdir(exist_ok=True)

    per_rows = []
    sum_m = None
    times = []
    peak_mems = []

    for i, cid in enumerate(case_ids, 1):
        f = full_root / f"{cid}.npz"
        d = np.load(f)
        img = d["img"].astype(np.float32)          # (C,D,H,W)
        gt  = d["seg"].astype(np.uint8)            # (D,H,W)

        C, D, H, W = img.shape
        win = args.roi_size
        stride = args.stride

        # accum probs in float32
        prob_sum = np.zeros((5, D, H, W), dtype=np.float32)
        cnt      = np.zeros((D, H, W), dtype=np.float32)

        zs = gen_starts(D, win, stride)
        ys = gen_starts(H, win, stride)
        xs = gen_starts(W, win, stride)

        torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
        t0 = time.time()

        for z0 in zs:
            for y0 in ys:
                for x0 in xs:
                    z1, y1, x1 = z0 + win, y0 + win, x0 + win
                    crop = img[:, z0:z1, y0:y1, x0:x1]
                    x = torch.from_numpy(crop).unsqueeze(0).to(device)  # (1,C,win,win,win)
                    probs = predict_probs_ensemble(models, x).squeeze(0).detach().cpu().numpy()  # (5,win,win,win)

                    prob_sum[:, z0:z1, y0:y1, x0:x1] += probs.astype(np.float32)
                    cnt[z0:z1, y0:y1, x0:x1] += 1.0

        prob_sum /= np.maximum(cnt[None, ...], 1e-6)
        pred_full = np.argmax(prob_sum, axis=0).astype(np.uint8)

        pred_full = postprocess_rc_only(pred_full, args.rc_min_vox)

        dt = time.time() - t0
        times.append(dt)
        if device.type == "cuda":
            peak_mems.append(torch.cuda.max_memory_allocated(device) / (1024**3))
        else:
            peak_mems.append(0.0)

        if args.save_nii:
            import nibabel as nib
            nib.save(nib.Nifti1Image(pred_full.astype(np.uint8), np.eye(4)),
                     str(out_dir / "nii" / f"{cid}.nii.gz"))

        m = compute_metrics_from_labels(pred_full, gt)
        per_rows.append({"case_id": cid, **m})

        if sum_m is None:
            sum_m = {k: 0.0 for k in m.keys()}
        for k in sum_m:
            sum_m[k] += float(m[k])

        if i % 10 == 0 or i == len(case_ids):
            print(f"[progress] {i}/{len(case_ids)}")

    mean_m = {k: (sum_m[k] / float(len(per_rows))) for k in sum_m}
    meta = {
        "full_root": str(full_root),
        "val_list": str(args.val_list),
        "roi_size": int(args.roi_size),
        "stride": int(args.stride),
        "rc_min_vox": int(args.rc_min_vox),
        "base_channels": int(args.base_channels),
        "ckpts": ckpts,
        "n_cases": len(per_rows),
        "sec_per_case_mean": float(np.mean(times)),
        "sec_per_case_p90": float(np.percentile(times, 90)),
        "peak_gpu_gb_mean": float(np.mean(peak_mems)),
        "peak_gpu_gb_max": float(np.max(peak_mems)),
    }

    json_path = out_dir / f"fullbrain_metrics_rcmin{args.rc_min_vox}.json"
    sum_csv   = out_dir / f"fullbrain_summary_rcmin{args.rc_min_vox}.csv"
    case_csv  = out_dir / f"fullbrain_per_case_rcmin{args.rc_min_vox}.csv"

    json_path.write_text(json.dumps({"meta": meta, "mean_metrics": mean_m}, indent=2), encoding="utf-8")

    cols = ["WT","TC","ET","RC","mean_wt_tc_et","dice_1_netc","dice_2_snf_h","dice_3_et","dice_4_rc"]
    with sum_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n_cases"] + cols + ["sec_per_case_mean","peak_gpu_gb_max"])
        w.writerow([len(per_rows)] + [mean_m[c] for c in cols] + [meta["sec_per_case_mean"], meta["peak_gpu_gb_max"]])

    with case_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["case_id"] + cols)
        w.writeheader()
        for r in per_rows:
            w.writerow({k: r[k] for k in ["case_id"] + cols})

    print("[OK] fullbrain sliding eval done")
    print("mean_metrics:", mean_m)
    print("meta:", {k: meta[k] for k in ["sec_per_case_mean","sec_per_case_p90","peak_gpu_gb_max"]})

if __name__ == "__main__":
    main()
