import torch

def _dice_binary(pred: torch.Tensor, gt: torch.Tensor, eps=1e-6):
    pred = pred.float()
    gt = gt.float()
    inter = (pred * gt).sum(dim=(1,2,3))
    den = pred.sum(dim=(1,2,3)) + gt.sum(dim=(1,2,3))
    dice = (2 * inter + eps) / (den + eps)
    return dice.mean().item()

def _dice_label(pred_lbl: torch.Tensor, gt_lbl: torch.Tensor, label: int, eps=1e-6):
    return _dice_binary(pred_lbl == label, gt_lbl == label, eps=eps)

@torch.no_grad()
def compute_all_metrics(logits: torch.Tensor, gt: torch.Tensor):
    """
    BraTS 2024 GLI official tissue evaluations:
      WT = (1,2,3)   # RC(4) is NOT in WT
      TC = (1,3)
      ET = (3)
      RC = (4)
    logits: (B,C,D,H,W), gt: (B,D,H,W)
    """
    pred = torch.argmax(logits, dim=1)

    d1 = _dice_label(pred, gt, 1)  # NETC
    d2 = _dice_label(pred, gt, 2)  # SNFH
    d3 = _dice_label(pred, gt, 3)  # ET
    d4 = _dice_label(pred, gt, 4)  # RC

    WT_pred = (pred == 1) | (pred == 2) | (pred == 3)
    WT_gt   = (gt   == 1) | (gt   == 2) | (gt   == 3)
    wt = _dice_binary(WT_pred, WT_gt)

    TC_pred = (pred == 1) | (pred == 3)
    TC_gt   = (gt   == 1) | (gt   == 3)
    tc = _dice_binary(TC_pred, TC_gt)

    et = _dice_binary(pred == 3, gt == 3)
    rc = _dice_binary(pred == 4, gt == 4)

    mean_wt_tc_et = (wt + tc + et) / 3.0

    return {
        "dice_1_netc": d1,
        "dice_2_snf_h": d2,
        "dice_3_et": d3,
        "dice_4_rc": d4,
        "WT": wt,
        "TC": tc,
        "ET": et,
        "RC": rc,
        "mean_wt_tc_et": mean_wt_tc_et,
    }
