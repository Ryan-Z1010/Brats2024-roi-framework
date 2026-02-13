import argparse, csv
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from src.models.unet3d_res import ResUNet3D
from src.train.eval_stage2_full_from_roiproposal import center_crop_with_pad, load_proposals, predict_probs_ensemble
from src.postprocess.rc_lcf import cc_26, extract_features, MLP
from scipy.ndimage import distance_transform_edt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full_root", required=True)
    ap.add_argument("--list", required=True, help="train.txt (recommended)")
    ap.add_argument("--proposal_csv", required=True)
    ap.add_argument("--ckpts", required=True)
    ap.add_argument("--base_channels", type=int, default=48)
    ap.add_argument("--roi_size", type=int, default=128)
    ap.add_argument("--out_pt", required=True)
    ap.add_argument("--max_cases", type=int, default=0)
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

    prop = load_proposals(args.proposal_csv)
    case_ids = [l.strip() for l in Path(args.list).read_text().splitlines() if l.strip()]
    if args.max_cases > 0:
        case_ids = case_ids[:args.max_cases]

    X, Y = [], []
    full_root = Path(args.full_root)

    for i, cid in enumerate(case_ids, 1):
        d = np.load(full_root / f"{cid}.npz")
        img = d["img"].astype(np.float32)
        gt  = d["seg"].astype(np.uint8)
        full_shape = gt.shape

        center = prop.get(cid, (full_shape[0]//2, full_shape[1]//2, full_shape[2]//2))
        crop_img, sl_full, sl_crop = center_crop_with_pad(img, center, args.roi_size, pad_val=0.0)

        x = torch.from_numpy(crop_img).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = predict_probs_ensemble(models, x)  # (1,5,roi,roi,roi)

        pred_roi = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        rc_prob_roi = probs.squeeze(0)[4].detach().cpu().numpy().astype(np.float32)

        pred_full = np.zeros_like(gt, dtype=np.uint8)
        pred_full[sl_full] = pred_roi[sl_crop]

        rc_prob_full = np.zeros_like(gt, dtype=np.float32)
        rc_prob_full[sl_full] = rc_prob_roi[sl_crop]

        rc_mask = (pred_full == 4)
        if rc_mask.sum() == 0:
            continue

        tc_mask = (pred_full == 1) | (pred_full == 3)
        wt_mask = (pred_full == 1) | (pred_full == 2) | (pred_full == 3)
        dist_to_tc = distance_transform_edt(~tc_mask)
        dist_to_wt = distance_transform_edt(~wt_mask)

        gt_rc = (gt == 4)

        lab, n = cc_26(rc_mask)
        for k in range(1, n+1):
            comp = (lab == k)
            if comp.sum() == 0:
                continue
            feat = extract_features(comp, rc_prob_full, tc_mask, wt_mask, dist_to_tc, dist_to_wt)
            y = 1.0 if (gt_rc & comp).any() else 0.0  # overlap>0 => positive
            X.append(feat)
            Y.append(y)

        if i % 50 == 0:
            print(f"[progress] {i}/{len(case_ids)} cases, comps={len(Y)}")

    X = np.stack(X, 0).astype(np.float32)
    Y = np.array(Y, dtype=np.float32)

    feat_mean = X.mean(axis=0)
    feat_std  = X.std(axis=0) + 1e-6
    Xn = (X - feat_mean) / feat_std

    # train tiny MLP
    model = MLP(Xn.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # imbalance handling
    pos = Y.sum()
    neg = len(Y) - pos
    pos_weight = torch.tensor([neg / (pos + 1e-8)], dtype=torch.float32)

    Xt = torch.from_numpy(Xn).float()
    Yt = torch.from_numpy(Y).float()

    for epoch in range(30):
        model.train()
        logits = model(Xt).float()
        loss = F.binary_cross_entropy_with_logits(logits, Yt, pos_weight=pos_weight)
        opt.zero_grad(); loss.backward(); opt.step()
        if (epoch+1) % 5 == 0:
            with torch.no_grad():
                prob = torch.sigmoid(logits)
                pred = (prob >= 0.5).float()
                acc = (pred == Yt).float().mean().item()
            print(f"epoch {epoch+1:02d} loss={loss.item():.4f} acc={acc:.3f}")

    out = {
        "state_dict": model.state_dict(),
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "thr": 0.5
    }
    Path(args.out_pt).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out_pt)
    print("[OK] saved:", args.out_pt, "samples:", len(Y), "pos:", int(pos), "neg:", int(neg))

if __name__ == "__main__":
    main()
