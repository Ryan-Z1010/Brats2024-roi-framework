import torch
from src.models.unet3d_res import ResUNet3D

# 按你的实验：in_channels=4 (T1n,T1c,T2w,FLAIR)，out_channels=5 (bg+4类)，base_channels=48
model = ResUNet3D(in_channels=4, out_channels=5, base=48)

n_total = sum(p.numel() for p in model.parameters())
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Model: ResUNet3D")
print("Total params:", n_total)
print("Trainable params:", n_train)
print("Params (M):", n_train/1e6)
