import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

from src.datasets.brats_coarse_npz import BratsCoarseNpzDataset, AugmentCfg
from src.models.unet3d_res import ResUNet3D

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def dice_from_logits(logits, target, eps=1e-6):
    # logits: (B,1,D,H,W), target: (B,D,H,W) float 0/1
    prob = torch.sigmoid(logits)
    pred = (prob > 0.5).float()                  # (B,1,D,H,W)
    tgt  = (target > 0.5).float().unsqueeze(1)   # (B,1,D,H,W)

    inter = (pred * tgt).sum(dim=(1,2,3,4))
    a = pred.sum(dim=(1,2,3,4))
    b = tgt.sum(dim=(1,2,3,4))
    d = (2*inter + eps) / (a + b + eps)
    return d.mean()

def dice_loss_from_logits(logits, target, eps=1e-6):
    prob = torch.sigmoid(logits)
    tgt  = target.unsqueeze(1)  # (B,1,D,H,W)
    inter = (prob * tgt).sum(dim=(2,3,4))
    a = prob.sum(dim=(2,3,4))
    b = tgt.sum(dim=(2,3,4))
    d = (2*inter + eps) / (a + b + eps)
    return 1.0 - d.mean()

def run_epoch(model, loader, opt, scaler, device, train=True, bce_w=1.0, dice_w=1.0):
    if train:
        model.train()
    else:
        model.eval()

    loss_sum, n = 0.0, 0
    dice_sum = 0.0
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)            # (B,4,96,96,96)
        y = batch["mask"].to(device, non_blocking=True)             # (B,96,96,96)
        y = y.float()

        with torch.set_grad_enabled(train):
            with torch.amp.autocast("cuda", enabled=scaler is not None):
                logits = model(x)[:, :1]  # (B,1,...)
                bce = F.binary_cross_entropy_with_logits(logits, y.unsqueeze(1))
                dl  = dice_loss_from_logits(logits, y)
                loss = bce_w*bce + dice_w*dl

            if train:
                opt.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    opt.step()

        loss_sum += float(loss.detach().cpu())
        dice_sum += float(dice_from_logits(logits.detach(), y).cpu())
        n += 1

    return loss_sum/max(1,n), dice_sum/max(1,n)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    set_seed(int(cfg["seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name = time.strftime("%Y%m%d_%H%M%S") + f"_coarse96_bs{cfg['data']['batch_size']}_lr{cfg['train']['lr']}"
    run_dir = Path("runs/stage1") / run_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)

    (run_dir / "config.yaml").write_text(Path(args.config).read_text(encoding="utf-8"), encoding="utf-8")
    (run_dir / "meta.json").write_text(json.dumps({"run_name": run_name}, indent=2), encoding="utf-8")

    tr_ds = BratsCoarseNpzDataset(cfg["data"]["root"], "train", cfg["data"]["splits_dir"],
                                 AugmentCfg(enable=cfg["augment"]["enable"], flip_prob=cfg["augment"]["flip_prob"]))
    va_ds = BratsCoarseNpzDataset(cfg["data"]["root"], "val", cfg["data"]["splits_dir"],
                                 AugmentCfg(enable=False))

    tr_loader = DataLoader(tr_ds, batch_size=cfg["data"]["batch_size"], shuffle=True,
                           num_workers=cfg["data"]["num_workers"], pin_memory=True)
    va_loader = DataLoader(va_ds, batch_size=1, shuffle=False,
                           num_workers=max(1, cfg["data"]["num_workers"]//2), pin_memory=True)

    model = ResUNet3D(in_channels=4, out_channels=2, base=cfg["model"]["base_channels"], dropout=0.0).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(cfg["train"]["epochs"]))

    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg["train"]["amp"])) if torch.cuda.is_available() else None

    best, best_ep = -1.0, -1
    bad = 0

    metrics_csv = run_dir / "metrics" / "metrics_epoch.csv"
    with metrics_csv.open("w", encoding="utf-8") as f:
        f.write("epoch,train_loss,train_dice,val_loss,val_dice,lr\n")

    for ep in range(1, int(cfg["train"]["epochs"])+1):
        t0 = time.time()
        tr_loss, tr_dice = run_epoch(model, tr_loader, opt, scaler, device, train=True,
                                     bce_w=cfg["loss"]["bce_weight"], dice_w=cfg["loss"]["dice_weight"])
        va_loss, va_dice = run_epoch(model, va_loader, opt, scaler, device, train=False,
                                     bce_w=cfg["loss"]["bce_weight"], dice_w=cfg["loss"]["dice_weight"])
        sched.step()

        lr = opt.param_groups[0]["lr"]
        with metrics_csv.open("a", encoding="utf-8") as f:
            f.write(f"{ep},{tr_loss:.6f},{tr_dice:.6f},{va_loss:.6f},{va_dice:.6f},{lr:.8f}\n")

        if va_dice > best + 1e-5:
            best, best_ep = va_dice, ep
            bad = 0
            torch.save({"model": model.state_dict(), "epoch": ep, "val_dice": float(va_dice)},
                       run_dir / "checkpoints" / "best.pt")
        else:
            bad += 1

        dt = time.time() - t0
        print(f"[E{ep:03d}] tr_dice={tr_dice:.4f} va_dice={va_dice:.4f} best={best:.4f}@{best_ep} time={dt:.1f}s")

        if bad >= int(cfg["train"]["early_stop_patience"]):
            print(f"[EARLY STOP] no improve for {bad} evals. best={best:.4f}@{best_ep}")
            break

    (run_dir / "metrics" / "summary_val.json").write_text(
        json.dumps({"best_val_dice": float(best), "best_epoch": int(best_ep), "run_dir": str(run_dir)}, indent=2),
        encoding="utf-8"
    )

    print("[OK] stage1 training finished")
    print("run_dir:", run_dir)
    print("best_ckpt:", run_dir / "checkpoints" / "best.pt")
    print("metrics_csv:", metrics_csv)

if __name__ == "__main__":
    main()
