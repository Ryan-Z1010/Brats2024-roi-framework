import os, json, argparse
import pandas as pd

KEYS = [
    "mean_wt_tc_et","WT","TC","ET","RC",
    "dice_1_netc","dice_2_snf_h","dice_3_et","dice_4_rc"
]

def load_metrics_json(p):
    with open(p, "r") as f:
        d = json.load(f)
    # 支持不同结构：有的直接就是 mean_metrics，有的包在 dict 里
    if isinstance(d, dict) and "mean_metrics" in d:
        m = d["mean_metrics"]
    elif isinstance(d, dict) and "metrics" in d and isinstance(d["metrics"], dict):
        m = d["metrics"]
    else:
        # 尝试把顶层当 metrics
        m = d
    row = {}
    for k in KEYS:
        if k in m:
            row[k] = float(m[k])
    return row

def find_artifacts(root, rcmin=120):
    target = f"full_stage1roi_metrics_rcmin{rcmin}.json"
    rows = []
    for dirpath, _, filenames in os.walk(root):
        if target in filenames:
            json_path = os.path.join(dirpath, target)
            exp_dir = os.path.relpath(dirpath, root).replace("\\","/")
            summary_csv = os.path.join(dirpath, f"full_stage1roi_summary_rcmin{rcmin}.csv")
            per_case_csv = os.path.join(dirpath, f"full_stage1roi_per_case_rcmin{rcmin}.csv")
            row = {
                "exp_dir": exp_dir,
                "json": json_path,
                "summary_csv": summary_csv if os.path.exists(summary_csv) else "",
                "per_case_csv": per_case_csv if os.path.exists(per_case_csv) else "",
            }
            row.update(load_metrics_json(json_path))
            rows.append(row)
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/full_stage1roi")
    ap.add_argument("--rcmin", type=int, default=120)
    ap.add_argument("--out_dir", default="results/paper_tables")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    rows = find_artifacts(args.root, rcmin=args.rcmin)
    if not rows:
        raise SystemExit(f"[ERR] no metrics json found under {args.root} for rcmin={args.rcmin}")

    df = pd.DataFrame(rows)
    # 保证列顺序稳定
    cols = ["exp_dir"] + [k for k in KEYS if k in df.columns] + ["json","summary_csv","per_case_csv"]
    df = df[cols]

    # 排序：主指标降序
    if "mean_wt_tc_et" in df.columns:
        df = df.sort_values("mean_wt_tc_et", ascending=False).reset_index(drop=True)

    all_csv = os.path.join(args.out_dir, f"all_end2end_rcmin{args.rcmin}.csv")
    df.to_csv(all_csv, index=False)

    # 论文主表：我们关心的几条（能找到就收录）
    wanted = [
        "baseline_single_seed1337_thr0p35_rcmin120",
        "jit8_single_seed2025_thr0p35_rcmin120",
        "mixed_ens4_nonjitEns3_plus_jit8seed2025_thr0p35_rcmin120",
        # non-jitter ens3 thr0p35 如果你有保存到某个目录，也会被自动匹配进来
        "ens3",
    ]
    keep = []
    for w in wanted:
        sub = df[df["exp_dir"].str.contains(w, case=False, na=False)]
        if len(sub) > 0:
            # w="ens3" 可能匹配多条，取其中 mean_wt_tc_et 最大的一条
            keep.append(sub.iloc[[0]])
    if keep:
        main_df = pd.concat(keep, axis=0).drop_duplicates(subset=["exp_dir"])
        main_csv = os.path.join(args.out_dir, "table_main_end2end.csv")
        main_df.to_csv(main_csv, index=False)

        # 同步导出 Markdown 方便粘进论文
        md_path = os.path.join(args.out_dir, "table_main_end2end.md")
        with open(md_path, "w") as f:
            show_cols = ["exp_dir","mean_wt_tc_et","WT","TC","ET","RC"]
            show_cols = [c for c in show_cols if c in main_df.columns]
            f.write(main_df[show_cols].to_markdown(index=False))
            f.write("\n")
    else:
        print("[WARN] did not match any wanted experiments for table_main_end2end.csv")

    print("[OK] wrote:")
    print(" -", all_csv)
    if keep:
        print(" -", os.path.join(args.out_dir, "table_main_end2end.csv"))
        print(" -", os.path.join(args.out_dir, "table_main_end2end.md"))

if __name__ == "__main__":
    main()
