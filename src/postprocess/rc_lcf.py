from dataclasses import dataclass
import numpy as np
import torch
from scipy.ndimage import label as cc_label, distance_transform_edt

STRUCT26 = np.ones((3,3,3), dtype=np.uint8)

def cc_26(mask: np.ndarray):
    lab, n = cc_label(mask.astype(np.uint8), structure=STRUCT26)
    return lab, n

def bbox_extent(idx):
    z, y, x = idx
    z0, z1 = int(z.min()), int(z.max())
    y0, y1 = int(y.min()), int(y.max())
    x0, x1 = int(x.min()), int(x.max())
    ez, ey, ex = (z1-z0+1), (y1-y0+1), (x1-x0+1)
    return ez, ey, ex

def extract_features(comp_mask: np.ndarray,
                     rc_prob: np.ndarray,
                     tc_mask: np.ndarray,
                     wt_mask: np.ndarray,
                     dist_to_tc: np.ndarray,
                     dist_to_wt: np.ndarray) -> np.ndarray:
    idx = np.where(comp_mask)
    vol = float(len(idx[0]))
    ez, ey, ex = bbox_extent(idx)

    pr = rc_prob[idx]
    p_mean = float(pr.mean()) if pr.size else 0.0
    p_max  = float(pr.max())  if pr.size else 0.0
    p90    = float(np.percentile(pr, 90)) if pr.size else 0.0

    tc_overlap = float((tc_mask & comp_mask).sum()) / (vol + 1e-8)
    wt_overlap = float((wt_mask & comp_mask).sum()) / (vol + 1e-8)

    d_tc = float(dist_to_tc[idx].min()) if pr.size else 999.0
    d_wt = float(dist_to_wt[idx].min()) if pr.size else 999.0

    # 10~12 个特征足够写论文、也足够有效
    feats = np.array([
        np.log1p(vol),
        np.log1p(ez), np.log1p(ey), np.log1p(ex),
        p_mean, p_max, p90,
        tc_overlap, wt_overlap,
        d_tc, d_wt
    ], dtype=np.float32)
    return feats

class MLP(torch.nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
        )
    def forward(self, x):  # (N,d)
        return self.net(x).squeeze(1)

@dataclass
class LCFModelPack:
    state_dict: dict
    feat_mean: np.ndarray
    feat_std: np.ndarray
    thr: float = 0.5

def load_lcf(path: str, device="cpu") -> LCFModelPack:
    ck = torch.load(path, map_location=device)
    return LCFModelPack(
        state_dict=ck["state_dict"],
        feat_mean=ck["feat_mean"],
        feat_std=ck["feat_std"],
        thr=float(ck.get("thr", 0.5)),
    )

def apply_lcf(pred_lbl: np.ndarray,
              rc_prob: np.ndarray,
              lcf_pack: LCFModelPack,
              thr: float = None,
              device="cpu") -> np.ndarray:
    thr = lcf_pack.thr if thr is None else float(thr)

    out = pred_lbl.copy()
    rc_mask = (out == 4)
    if rc_mask.sum() == 0:
        return out

    tc_mask = (out == 1) | (out == 3)
    wt_mask = (out == 1) | (out == 2) | (out == 3)

    dist_to_tc = distance_transform_edt(~tc_mask)  # distance to nearest TC voxel
    dist_to_wt = distance_transform_edt(~wt_mask)

    lab, n = cc_26(rc_mask)
    feats = []
    comps = []
    for i in range(1, n+1):
        comp = (lab == i)
        if comp.sum() == 0:
            continue
        f = extract_features(comp, rc_prob, tc_mask, wt_mask, dist_to_tc, dist_to_wt)
        feats.append(f)
        comps.append(comp)

    if not feats:
        return out

    X = np.stack(feats, 0)
    Xn = (X - lcf_pack.feat_mean) / (lcf_pack.feat_std + 1e-8)

    model = MLP(Xn.shape[1]).to(device)
    model.load_state_dict(lcf_pack.state_dict, strict=True)
    model.eval()

    with torch.no_grad():
        logits = model(torch.from_numpy(Xn).to(device))
        prob = torch.sigmoid(logits).cpu().numpy()

    keep = prob >= thr

    # drop components predicted as FP
    for comp, k in zip(comps, keep):
        if not k:
            out[comp] = 0
    out[out == 4] = 4
    return out
