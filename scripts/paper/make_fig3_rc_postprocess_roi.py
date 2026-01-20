import os, json, argparse
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
import inspect

from src.models.unet3d_res import ResUNet3D

def load_model(run_dir: str, device: str):
    cfg_path = os.path.join(run_dir, "config.yaml")
    ckpt_path = os.path.join(run_dir, "checkpoints", "best.pt")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    base = int(cfg.get("model", {}).get("base_channels", 48))

    # --- robust init: match actual __init__ signature ---
    sig = inspect.signature(ResUNet3D)
    kwargs = {}
    # common names
    if "in_channels" in sig.parameters: kwargs["in_channels"] = 4
    if "out_channels" in sig.parameters: kwargs["out_channels"] = 5
    # base width name variants
    for k in ["base_channels", "base_ch", "base", "width", "channels", "feat0", "c1"]:
        if k in sig.parameters:
            kwargs[k] = base
            break
    # optional dropout
    if "dropout" in sig.parameters:
        kwargs["dropout"] = 0.0

    model = ResUNet3D(**kwargs)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model

def rc_filter(pred_lbl: np.ndarray, rc_min_vox: int):
    # pred_lbl: (Z,Y,X) or (H,W,D) label map uint8
    # 这里只做 RC(=4) 的连通域过滤。为了简化复现，我们用一个轻量实现：3D 6-connectivity BFS
    if rc_min_vox <= 0:
        return pred_lbl

    rc = (pred_lbl == 4).astype(np.uint8)
    if rc.sum() == 0:
        return pred_lbl

    Z, Y, X = rc.shape
    visited = np.zeros_like(rc, dtype=np.uint8)
    out = pred_lbl.copy()

    nbrs = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]

    for z in range(Z):
        for y in range(Y):
            for x in range(X):
                if rc[z,y,x] == 0 or visited[z,y,x]:
                    continue
                # BFS
                stack=[(z,y,x)]
                visited[z,y,x]=1
                comp=[(z,y,x)]
                while stack:
                    cz,cy,cx = stack.pop()
                    for dz,dy,dx in nbrs:
                        nz,ny,nx = cz+dz, cy+dy, cx+dx
                        if 0 <= nz < Z and 0 <= ny < Y and 0 <= nx < X:
                            if rc[nz,ny,nx] and not visited[nz,ny,nx]:
                                visited[nz,ny,nx]=1
                                stack.append((nz,ny,nx))
                                comp.append((nz,ny,nx))
                if len(comp) < rc_min_vox:
                    for (cz,cy,cx) in comp:
                        out[cz,cy,cx] = 0  # 去掉小 RC
    return out

@torch.no_grad()
def infer_ensemble(models, img_np: np.ndarray, device: str):
    # img_np: (4,128,128,128) float32
    x = torch.from_numpy(img_np[None]).to(device)  # (1,4,*,*,*)
    probs = None
    for m in models:
        logits = m(x)
        p = torch.softmax(logits, dim=1)
        probs = p if probs is None else (probs + p)
    probs = probs / len(models)
    pred = torch.argmax(probs, dim=1)[0].cpu().numpy().astype(np.uint8)  # (128,128,128)
    return pred

def pick_slice(flair: np.ndarray, gt: np.ndarray, pred0: np.ndarray):
    # 选一个“RC最明显”的轴向切片：优先 GT_RC，否则用 pred_RC
    rc_gt = (gt==4).sum(axis=(0,1))
    if rc_gt.max() > 0:
        return int(rc_gt.argmax())
    rc_pr = (pred0==4).sum(axis=(0,1))
    if rc_pr.max() > 0:
        return int(rc_pr.argmax())
    # fallback：取中间
    return flair.shape[-1]//2

def norm01(x):
    x = x.astype(np.float32)
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    return x

def overlay(ax, bg, mask, title):
    ax.imshow(bg, cmap="gray")
    # 红色 RC overlay
    m = np.ma.masked_where(mask==0, mask)
    ax.imshow(m, alpha=0.6)  # 默认 colormap 会给颜色；投稿图你后期也可改成轮廓线
    ax.set_title(title, fontsize=10)
    ax.axis("off")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases_txt", type=str, required=True)
    ap.add_argument("--roi_root", type=str, default="data/preprocessed/npy_roi128_v1/val")
    ap.add_argument("--run_dirs", nargs="+", required=True)
    ap.add_argument("--rc_min_vox", type=int, default=120)
    ap.add_argument("--out_png", type=str, default="results/paper_assets/figures/fig3_rc_postprocess_examples.png")
    ap.add_argument("--out_pdf", type=str, default="results/paper_assets/figures/fig3_rc_postprocess_examples.pdf")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_png), exist_ok=True)

    with open(args.cases_txt) as f:
        cases=[l.strip() for l in f if l.strip()]

    device = args.device if (args.device=="cpu" or torch.cuda.is_available()) else "cpu"
    models=[load_model(r, device) for r in args.run_dirs]

    # 2 rows × 3 cols
    fig, axes = plt.subplots(len(cases), 3, figsize=(9, 3*len(cases)), dpi=200)

    if len(cases) == 1:
        axes = np.array([axes])

    for i, cid in enumerate(cases):
        npz_path = os.path.join(args.roi_root, f"{cid}.npz")
        d = np.load(npz_path)
        img = d["img"]              # (4,128,128,128)
        gt  = d["seg"].astype(np.uint8)  # (128,128,128)

        # 背景用 FLAIR：默认 index=2（如果你实际顺序不同，就改这里）
        flair = img[2]
        pred0 = infer_ensemble(models, img, device)
        pred1 = rc_filter(pred0, args.rc_min_vox)

        z = pick_slice(flair, gt, pred0)
        bg = norm01(flair[:,:,z])

        overlay(axes[i,0], bg, (gt[:,:,z]==4).astype(np.uint8),  f"{cid}\nGT RC")
        overlay(axes[i,1], bg, (pred0[:,:,z]==4).astype(np.uint8), "Pred RC (before)")
        overlay(axes[i,2], bg, (pred1[:,:,z]==4).astype(np.uint8), f"Pred RC (after, rc_min_vox={args.rc_min_vox})")

        print(cid, "GT_RC_vox", int((gt==4).sum()),
              "Pred_RC_vox(before)", int((pred0==4).sum()),
              "Pred_RC_vox(after)",  int((pred1==4).sum()),
              "slice_z", z)

    plt.tight_layout()
    fig.savefig(args.out_png)
    fig.savefig(args.out_pdf)
    print("[OK] Fig3 saved:")
    print("png:", args.out_png)
    print("pdf:", args.out_pdf)

if __name__ == "__main__":
    main()
