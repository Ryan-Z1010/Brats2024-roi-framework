import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import yaml

from src.datasets.brats_roi_npz import BratsRoiNpzDataset, AugmentCfg
from src.models.unet3d_res import ResUNet3D
from src.metrics.brats_composites import compute_all_metrics

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    agg = None
    n = 0
    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        gt  = batch["label"].to(device, non_blocking=True)
        logits = model(img)
        m = compute_all_metrics(logits, gt)
        if agg is None:
            agg = {k: 0.0 for k in m.keys()}
        for k, v in m.items():
            agg[k] += float(v)
        n += 1
    for k in agg:
        agg[k] /= max(1, n)
    return agg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str, help="run config.yaml or training yaml")
    ap.add_argument("--ckpt", required=True, type=str, help="best.pt path")
    ap.add_argument("--out_dir", default="", type=str, help="where to write results (default: ckpt's run_dir/metrics)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = BratsRoiNpzDataset(cfg["data"]["roi_root"], "val", cfg["data"]["splits_dir"], AugmentCfg(enable=False))
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=max(1, cfg["data"]["num_workers"] // 2), pin_memory=cfg["data"]["pin_memory"])

    model = ResUNet3D(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        base=cfg["model"]["base_channels"],
        dropout=cfg["model"]["dropout"]
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)

    m = validate(model, dl, device)

    out_dir = Path(args.out_dir) if args.out_dir else (Path(args.ckpt).resolve().parents[1] / "metrics")
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "val_metrics_official.json"
    json_path.write_text(json.dumps(m, indent=2), encoding="utf-8")

    csv_path = out_dir / "val_metrics_official.csv"
    cols = ["WT","TC","ET","RC","mean_wt_tc_et","dice_1_netc","dice_2_snf_h","dice_3_et","dice_4_rc"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ckpt","epoch"] + cols)
        w.writerow([args.ckpt, ckpt.get("epoch", -1)] + [m[c] for c in cols])

    print("[OK] official val metrics written")
    print("json:", json_path)
    print("csv :", csv_path)
    print("metrics:", m)

if __name__ == "__main__":
    main()
