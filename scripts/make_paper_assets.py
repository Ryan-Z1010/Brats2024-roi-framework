import os, json, csv
from pathlib import Path
import pandas as pd
import numpy as np

def read_lines(p: Path):
    return [l.strip() for l in p.read_text().splitlines() if l.strip()]

def case_to_patient(case_id: str) -> str:
    # BraTS-GLI-03050-101 -> BraTS-GLI-03050
    parts = case_id.split("-")
    return "-".join(parts[:-1]) if len(parts) >= 4 else case_id

def write_csv(path: Path, rows, header):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

def fmt(x, nd=6):
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.{nd}f}"
    return x

def main():
    root = Path(".")
    out_tables = root / "results" / "paper_assets" / "tables"
    out_figs   = root / "results" / "paper_assets" / "figures"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    # ---------- Table 1: label/region definition ----------
    table1 = [
        [0, "Background", "", "", "", ""],
        [1, "NETC", "WT", "TC", "", ""],
        [2, "SNFH", "WT", "", "", ""],
        [3, "ET",   "WT", "TC", "ET", ""],
        [4, "RC",   "",   "",   "",   "RC"],
    ]
    write_csv(out_tables/"table1_label_regions.csv",
              table1,
              ["Label ID","Name","Included in WT","Included in TC","ET","RC"])

    # ---------- Table 2: split summary ----------
    train_txt = root/"data/splits/train.txt"
    val_txt   = root/"data/splits/val.txt"
    tr = read_lines(train_txt)
    va = read_lines(val_txt)
    tr_pat = sorted({case_to_patient(x) for x in tr})
    va_pat = sorted({case_to_patient(x) for x in va})
    all_pat = sorted(set(tr_pat) | set(va_pat))
    table2 = [
        ["train", len(tr_pat), len(tr)],
        ["val",   len(va_pat), len(va)],
        ["total", len(all_pat), len(tr)+len(va)]
    ]
    write_csv(out_tables/"table2_split_summary.csv",
              table2,
              ["Split","#Patients","#Cases"])

    # ---------- Table 3: preprocessing summary (fixed text fields, easy to edit later) ----------
    table3 = [
        ["Modalities","T1n, T1c, T2w, T2-FLAIR"],
        ["Intensity clipping","0.5–99.5 percentile (per modality)"],
        ["Normalization","z-score (per modality)"],
        ["Storage format",".npz with keys: img (float32), seg (uint8)"],
        ["Full-volume root","data/preprocessed/npy_full_v1/{train,val}/"],
        ["ROI128 root (non-jitter)","data/preprocessed/npy_roi128_v1/{train,val}/"],
        ["ROI128 root (jitter train)","data/preprocessed/npy_roi128_jit8_v1/train/"],
        ["ROI size","128^3"],
        ["Coarse size (Stage1)","96^3"],
    ]
    write_csv(out_tables/"table3_preprocess_summary.csv",
              table3,
              ["Item","Value"])

    # ---------- Table 4: ROI coverage stats (robust: compute from val_roi128.csv) ----------
    roi_csv = root/"results/roi_proposals/stage1_20260102/val_roi128.csv"
    out_t4 = out_tables/"table4_roi_coverage_stats.csv"

    if roi_csv.exists():
        df = pd.read_csv(roi_csv)
        # 自动找 coverage 列
        cov_col = None
        for c in df.columns:
            if c.lower() in ("coverage","roi_coverage","gt_coverage"):
                cov_col = c
                break
        if cov_col is None:
            raise RuntimeError(f"No coverage column found in {roi_csv}. columns={list(df.columns)}")

        cov = df[cov_col].astype(float).to_numpy()
        rows = [
            ["n_cases", len(cov)],
            ["coverage_mean", float(np.mean(cov))],
            ["coverage_p05", float(np.quantile(cov, 0.05))],
            ["coverage_p10", float(np.quantile(cov, 0.10))],
            ["coverage_min", float(np.min(cov))],
            ["coverage_max", float(np.max(cov))],
        ]
        write_csv(out_t4, [[k, fmt(v, 6)] for k,v in rows], ["Metric","Value"])
    else:
        print(f"[WARN] missing: {roi_csv}")

    # ---------- Figure 2: ROI coverage distribution ----------
    # Prefer reading coverage column from val_roi128.csv if present.
    roi_csv = root/"results/roi_proposals/stage1_20260102/val_roi128.csv"
    fig2_path = out_figs/"fig2_roi_coverage_hist.png"
    if roi_csv.exists():
        df = pd.read_csv(roi_csv)
        cov_col = None
        for c in df.columns:
            if c.lower() in ("coverage","roi_coverage","gt_coverage"):
                cov_col = c
                break
        if cov_col is not None:
            cov = df[cov_col].astype(float).to_numpy()
            import matplotlib.pyplot as plt
            plt.figure()
            plt.hist(cov, bins=30)
            plt.xlabel("ROI coverage (fraction of GT foreground inside ROI)")
            plt.ylabel("Number of cases")
            plt.title("Validation ROI coverage distribution")
            plt.tight_layout()
            plt.savefig(fig2_path, dpi=300)
            plt.savefig("results/paper_assets/figures/fig2_roi_coverage_hist.pdf")
            plt.close()
        else:
            print(f"[WARN] {roi_csv} has no coverage column. Available columns: {list(df.columns)}")
            print("[HINT] If coverage not saved, we can compute it from ROI coords + GT, tell me the ROI coord columns in the CSV.")
    else:
        print(f"[WARN] missing: {roi_csv}")

    # ---------- Table 5: training settings for Stage2 (robust extraction + fallback from run name) ----------
    stage2_cfg = root/"runs/stage2/20260101_053221_roi128_bs2_lr0.00015/config.yaml"
    if stage2_cfg.exists():
        import yaml, re
        cfg = yaml.safe_load(stage2_cfg.read_text())
        run_dir = stage2_cfg.parent
        run_name = run_dir.name

        def get_path(d, path):
            cur = d
            for k in path:
                if isinstance(cur, dict) and k in cur:
                    cur = cur[k]
                else:
                    return None
            return cur

        def get_any(d, paths):
            for p in paths:
                v = get_path(d, p)
                if v is not None:
                    return v
            return None

        # batch size: try multiple common keys, fallback parse from run_name "bs2"
        bs = get_any(cfg, [
            ["train","batch_size"],
            ["train","bs"],
            ["data","batch_size"],
            ["dataloader","batch_size"],
        ])
        if bs is None:
            m = re.search(r"bs(\d+)", run_name)
            if m: bs = int(m.group(1))

        # lr: try config, fallback parse from run_name "lr0.00015"
        lr = get_any(cfg, [
            ["train","lr"],
            ["optimizer","lr"],
            ["train","learning_rate"],
        ])
        if lr is None:
            m = re.search(r"lr([0-9.]+)", run_name)
            if m: lr = float(m.group(1))

        # optimizer name / weight decay (best effort)
        opt_name = get_any(cfg, [
            ["train","optimizer"],
            ["optimizer","name"],
            ["train","optim"],
        ])
        wd = get_any(cfg, [
            ["train","weight_decay"],
            ["optimizer","weight_decay"],
        ])

        epochs = get_any(cfg, [["train","epochs"], ["train","max_epochs"]])
        amp = get_any(cfg, [["train","amp"], ["train","use_amp"]])
        seed = get_any(cfg, [["seed"], ["train","seed"]])
        num_workers = get_any(cfg, [["data","num_workers"], ["train","num_workers"], ["dataloader","num_workers"]])
        patience = get_any(cfg, [["train","early_stop_patience"], ["train","patience"]])

        rows = []
        rows.append(["Model base_channels", get_any(cfg, [["model","base_channels"]])])
        rows.append(["ROI size", "128^3"])
        rows.append(["Batch size", bs])
        rows.append(["Learning rate", lr])
        rows.append(["Optimizer", opt_name if opt_name is not None else "see config.yaml"])
        rows.append(["Weight decay", wd if wd is not None else "see config.yaml"])
        rows.append(["Epochs (max)", epochs if epochs is not None else "see config.yaml"])
        rows.append(["Scheduler", "CosineAnnealingLR"])
        rows.append(["AMP", amp])
        rows.append(["Early stopping patience", patience if patience is not None else "40 evals (default in training log)"])
        rows.append(["Seed", seed if seed is not None else "see run meta"])
        rows.append(["Num workers", num_workers if num_workers is not None else "see config.yaml"])

        write_csv(out_tables/"table5_stage2_training_settings.csv", rows, ["Setting","Value"])
    else:
        print(f"[WARN] missing: {stage2_cfg} (edit path inside script to your representative config)")

    # ---------- Table 6: ablation (read summary CSVs; you can edit paths here) ----------
    # These are the three experiments you just listed.
    exp_map = {
        "baseline_single_seed1337_thr0p35_rcmin120":
            root/"results/full_stage1roi/baseline_single_seed1337_thr0p35_rcmin120/full_stage1roi_summary_rcmin120.csv",
        "jit8_single_seed2025_thr0p35_rcmin120":
            root/"results/full_stage1roi/jit8_single_seed2025_thr0p35_rcmin120/full_stage1roi_summary_rcmin120.csv",
        "mixed_ens4_nonjitEns3_plus_jit8seed2025_thr0p35_rcmin120":
            root/"results/full_stage1roi/mixed_ens4_nonjitEns3_plus_jit8seed2025_thr0p35_rcmin120/full_stage1roi_summary_rcmin120.csv",
    }
    rows = []
    for name, p in exp_map.items():
        if not p.exists():
            print(f"[WARN] missing: {p}")
            continue
        df = pd.read_csv(p)
        # your summary csv seems to have columns like WT,TC,ET,RC,mean_wt_tc_et
        # handle both single-row and key-value formats
        if "mean_wt_tc_et" in df.columns:
            r = df.iloc[0].to_dict()
            rows.append([name,
                         float(r.get("mean_wt_tc_et")),
                         float(r.get("WT")),
                         float(r.get("TC")),
                         float(r.get("ET")),
                         float(r.get("RC"))])
        else:
            # try key-value format: Metric,Value
            if set(df.columns) >= {"Metric","Value"}:
                kv = dict(zip(df["Metric"], df["Value"]))
                rows.append([name,
                             float(kv["mean_wt_tc_et"]),
                             float(kv["WT"]), float(kv["TC"]), float(kv["ET"]), float(kv["RC"])])
            else:
                raise RuntimeError(f"Unrecognized summary format in {p}: columns={df.columns}")

    rows.sort(key=lambda x: -x[1])
    write_csv(out_tables/"table6_ablation_end2end.csv",
              [[r[0], fmt(r[1],6), fmt(r[2],6), fmt(r[3],6), fmt(r[4],6), fmt(r[5],6)] for r in rows],
              ["exp_dir","mean_wt_tc_et","WT","TC","ET","RC"])

    print("[OK] paper assets written to:")
    print("  tables :", out_tables)
    print("  figures:", out_figs)

if __name__ == "__main__":
    main()
