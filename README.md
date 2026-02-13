## A Two-Stage ROI-Guided Framework with Mixed Ensembling and RC-Specific Post-processing for Post-Treatment Glioma Subregion Segmentation on BraTS 2024 GLI

It includes the training/evaluation code, configuration files, split lists, ROI proposal CSVs, and table-generation scripts used to produce the results reported in the paper (internal patient-level split: 1206 train / 144 val).

Important: Due to BraTS data licensing and storage size, this repository does not include raw MRI data, preprocessed tensors, Stage-1/Stage-2 segmentation checkpoints, or full prediction volumes. We include the lightweight RC learned component filter (LCF) model used in post-processing (results/lcf/lcf_rc.pt) and its training script (scripts/rc_lcf/train_rc_lcf.py).

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
RC learned component filter (LCF) post-processing: src/postprocess/rc_lcf.py, results/lcf/lcf_rc.pt, scripts/rc_lcf/train_rc_lcf.py
Lesion-wise evaluation: scripts/eval_lesionwise_val.py
Paper tables (CSV): results/paper_assets/tables/
nnU-Net helper scripts (dataset conversion + split + voxel-Dice eval): scripts/nnunet/
Paper asset scripts: scripts/make_paper_assets.py, scripts/90_make_paper_tables.py, scripts/paper/

## 2) What is NOT included

BraTS 2024 GLI raw data (*.nii.gz)
Preprocessed NPZ tensors (*.npz)
Segmentation model checkpoints (*.pt, *.pth)
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

### 8.1 RC post-processing ablation (Table 6B)
Note: segmentation checkpoints and full predictions are not shipped in this repository. Provide ckpt paths (or train them) to run the following commands.

Example (mixed Stage-2 ROI ensemble used in the manuscript):
CKPTS="runs/stage2/20251231_023529_roi128_bs2_lr0.00015/checkpoints/best.pt,runs/stage2/20260101_000333_roi128_bs2_lr0.00015/checkpoints/best.pt,runs/stage2/20260101_053221_roi128_bs2_lr0.00015/checkpoints/best.pt,runs/stage2/20260102_074900_roi128_bs2_lr0.00015/checkpoints/best.pt"
PROP_VAL="results/roi_proposals/stage1_20260102_thr0p35/val_roi128.csv"

# none
python -u src/train/eval_stage2_full_from_roiproposal.py --rc_filter none \
  --full_root data/preprocessed/npy_full_v1/val --val_list data/splits/val.txt \
  --proposal_csv "$PROP_VAL" --ckpts "$CKPTS" --roi_size 128 --base_channels 48 \
  --out_dir results/full_stage1roi/mixed_rc_none --save_nii

# size-only ablation
python -u src/train/eval_stage2_full_from_roiproposal.py --rc_filter minvox --rc_min_vox 120 \
  --full_root data/preprocessed/npy_full_v1/val --val_list data/splits/val.txt \
  --proposal_csv "$PROP_VAL" --ckpts "$CKPTS" --roi_size 128 --base_channels 48 \
  --out_dir results/full_stage1roi/mixed_rcmin120 --save_nii

# default hybrid
python -u src/train/eval_stage2_full_from_roiproposal.py --rc_filter hybrid --rc_min_vox 90 \
  --lcf_model results/lcf/lcf_rc.pt --lcf_thr 0.46 \
  --full_root data/preprocessed/npy_full_v1/val --val_list data/splits/val.txt \
  --proposal_csv "$PROP_VAL" --ckpts "$CKPTS" --roi_size 128 --base_channels 48 \
  --out_dir results/full_stage1roi/final_hybrid_rcmin90_thr0.46 --save_nii

### 8.2 Lesion-wise evaluation (Table 9)
python scripts/eval_lesionwise_val.py \
  --val_list data/splits/val.txt \
  --gt_npz_root data/preprocessed/npy_full_v1/val --gt_root . \
  --pred_dir results/full_stage1roi/final_hybrid_rcmin90_thr0.46/nii \
  --out_csv results/lesionwise/lesionwise_final_hybrid_rcmin90_thr0.46.csv


## 9) Notes on evaluation
Primary development metric: mean(WT, TC, ET) (voxel-wise Dice).
RC is reported separately. Our default RC suppression uses a hybrid post-processing: first applying the learned component filter (LCF, lcf_thr=0.46) on predicted RC components, then removing residual small RC components with rc_min_vox=90. Size-only filtering with rc_min_vox=120 is reported as an ablation/analysis setting.
To better characterize RC fragmentation and false positives, we also report component-level statistics (see table7_rc_component_stats.csv).
LCF training (RC false-positive filtering). We run the Stage-2 ROI ensemble on the training split to obtain predicted RC connected components (26-connectivity). Each predicted RC component is labeled positive if it has any voxel overlap with the ground-truth RC, and negative otherwise. Component features are extracted from the RC probability map and the component’s spatial relation to the predicted TC/WT regions (distance transforms). A tiny MLP is trained with class-imbalance weighting; the inference threshold lcf_thr=0.46 is selected on the validation split.

## 10) Data availability

This work uses the BraTS 2024 GLI dataset released by the challenge organizers under their data access and usage agreement. No additional patient data were collected in this study.

## 11) Code availability

This repository contains the full pipeline code (preprocessing, ROI proposal generation, Stage 2 training, ensembling, post-processing, and evaluation) and the scripts used to generate all tables/figures reported in the manuscript.
