## A Two-Stage ROI-Guided Framework with Mixed Ensembling and RC-Specific Post-processing for Post-Treatment Glioma Subregion Segmentation on BraTS 2024 GLI

It includes the training/evaluation code, configuration files, split lists, ROI proposal CSVs, and table-generation scripts used to produce the results reported in the paper (internal patient-level split: 1206 train / 144 val).

Important: Due to BraTS data licensing and storage size, this repository does not include raw MRI data, preprocessed tensors, checkpoints, or predictions.

## 1) What is included

### Code：
Training/evaluation scripts: src/train/
Datasets: src/datasets/
Model: src/models/
Metrics: src/metrics/
Utilities: src/utils/
Configs (all runs are driven by config files): configs/
Split lists (patient-level): data/splits/train.txt, data/splits/val.txt
Stage 1 ROI proposals (CSV, end-to-end evaluation uses thr=0.35): results/roi_proposals/
Paper tables (CSV): results/paper_assets/tables/
nnU-Net helper scripts (dataset conversion + split + voxel-Dice eval): scripts/nnunet/
Paper asset scripts: scripts/make_paper_assets.py, scripts/90_make_paper_tables.py, scripts/paper/

## 2) What is NOT included

BraTS 2024 GLI raw data (*.nii.gz)
Preprocessed NPZ tensors (*.npz)
Model checkpoints (*.pt, *.pth)
nnU-Net preprocessed arrays (*.b2nd)
Predictions (*.nii.gz)

## 3) Expected directory structure on your machine

### 3.1 Raw BraTS GLI data (not included)
Place the official BraTS 2024 GLI training data under:
data/raw/brats2024_gli/training_data1_v2/

Each case folder is expected to contain:
*-t1n.nii.gz, *-t1c.nii.gz, *-t2w.nii.gz, *-t2f.nii.gz, *-seg.nii.gz

### 3.2 Preprocessed tensors (not included)
This project uses NPZ tensors for fast I/O:
Full-volume tensors: data/preprocessed/npy_full_v1/
Validation set: data/preprocessed/npy_full_v1/val/
ROI128 tensors (Stage 2 ROI training): data/preprocessed/npy_roi128_v1/
Coarse96 tensors (Stage 1 training): data/preprocessed/npy_coarse96_v1/

### 3.3 Patient-level split lists (included)
Train split (1206): data/splits/train.txt
Val split (144): data/splits/val.txt

## 4) Stage 1 ROI proposals (included)

ROI proposals used for end-to-end evaluation:
Main setting (thr = 0.35)
results/roi_proposals/stage1_20260102_thr0p35/val_roi128.csv

Additional thresholds:
results/roi_proposals/stage1_20260102/val_roi128.csv
results/roi_proposals/stage1_20260102_thr0p25/val_roi128.csv

## 5) Baselines and checkpoints (paths from our experiments)

Checkpoints are not shipped in this repository. The paths below document the exact runs used in the manuscript.

### 5.1 Two-stage ROI mixed ensemble (Stage 2 ROI checkpoints)
Mixed ensemble = 3 non-jitter seeds + 1 jitter model:
1.runs/stage2/20251231_023529_roi128_bs2_lr0.00015/checkpoints/best.pt
2.runs/stage2/20260101_000333_roi128_bs2_lr0.00015/checkpoints/best.pt
3.runs/stage2/20260101_053221_roi128_bs2_lr0.00015/checkpoints/best.pt
4.runs/stage2/20260102_074900_roi128_bs2_lr0.00015/checkpoints/best.pt

### 5.2 One-stage full-brain baseline (sliding window)
runs/stage2_fullbrain/20260108_202410_fullbrain128_bs2_lr0.00015/checkpoints/best.pt

### 5.3 nnU-Net v2 baseline (3d_fullres, fold 0)
Training artifacts:
nnUNet_results/Dataset701_BraTS2024GLI/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth
nnUNet_results/Dataset701_BraTS2024GLI/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/training_log_2026_1_8_15_46_07.txt

Voxel-wise Dice evaluation summary:
results/nnunetv2_701_val_metrics.csv

## 6) Paper tables (included)

All manuscript tables are generated under:
results/paper_assets/tables/

Key examples:
table6_ablation_end2end.csv
table7_rc_component_stats.csv

## 7) Environment

Recommended (conda):
conda env create -f environment.yml
conda activate base

Minimal (pip):
pip install -r requirements.txt

## 8) Reproducing results

1. Download BraTS 2024 GLI dataset (per organizer rules) and place it in:
data/raw/brats2024_gli/training_data1_v2/

2. Generate / verify split lists (already included in data/splits/).

3. Preprocess
Create full-volume NPZ tensors (data/preprocessed/npy_full_v1/)
Create coarse96 NPZ tensors (data/preprocessed/npy_coarse96_v1/)
Create ROI128 NPZ tensors (data/preprocessed/npy_roi128_v1/)

4. Train Stage 1 (coarse localization) and export ROI proposals (val_roi128.csv).

5. Train Stage 2 (ROI segmentation) and perform:
single models
jitter models
mixed ensembling evaluation

6. Run strong baselines
one-stage full-brain sliding window baseline
nnU-Net v2 3d_fullres baseline

7. Generate paper tables/assets
scripts/make_paper_assets.py
scripts/90_make_paper_tables.py

## 9) Notes on evaluation

Primary development metric: mean(WT, TC, ET) (voxel-wise Dice).
RC is reported separately; an RC-specific connected-component filtering rule is applied with rc_min_vox = 120.
To better characterize RC fragmentation and false positives, we also report component-level (connected-component) statistics (see table7_rc_component_stats.csv).

## 10) Data availability

This work uses the BraTS 2024 GLI dataset released by the challenge organizers under their data access and usage agreement. No additional patient data were collected in this study.

## 11) Code availability

This repository contains the full pipeline code (preprocessing, ROI proposal generation, Stage 2 training, ensembling, post-processing, and evaluation) and the scripts used to generate all tables/figures reported in the manuscript.
