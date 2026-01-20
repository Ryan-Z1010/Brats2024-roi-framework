from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

@dataclass
class AugmentCfg:
    enable: bool = False
    flip_prob: float = 0.5

def _rand_flip(x, y, p=0.5):
    # x: (C,D,H,W), y: (D,H,W)
    if torch.rand(()) > p:
        return x, y
    # flip along one random spatial dim
    dim = int(torch.randint(0, 3, (1,)).item())
    x = torch.flip(x, dims=[dim+1])
    y = torch.flip(y, dims=[dim])
    return x, y

class BratsCoarseNpzDataset(Dataset):
    def __init__(self, root: str, split: str, splits_dir: str, augment: AugmentCfg):
        self.root = Path(root) / split
        self.case_ids = [l.strip() for l in (Path(splits_dir) / f"{split}.txt").read_text().splitlines() if l.strip()]
        self.augment = augment

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx: int):
        cid = self.case_ids[idx]
        f = self.root / f"{cid}.npz"
        d = np.load(f)
        img = torch.from_numpy(d["img"].astype(np.float32))  # (4,96,96,96)
        m   = torch.from_numpy(d["mask"].astype(np.float32)) # (96,96,96) 0/1

        if self.augment.enable:
            img, m = _rand_flip(img, m, p=self.augment.flip_prob)

        return {"case_id": cid, "image": img, "mask": m}
