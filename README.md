# A Two-Stage ROI-Guided Framework for Post-treatment Glioma Subregion Segmentation with Component-level Resection Cavity False-positive Suppression

This repository contains the training, inference, evaluation, and result-generation code for our post-treatment glioma MRI segmentation framework.

The method is an explicit two-stage ROI-guided pipeline:
- **Stage 1** performs coarse localization on downsampled volumes and generates fixed-size ROI proposals.
- **Stage 2** performs high-resolution four-class segmentation within the proposed ROI.
- An **RC-specific hybrid refinement strategy** combines learned component filtering (LCF) with size-based connected-component filtering to suppress fragmented resection cavity (RC) false positives.

The repository includes:
- training and evaluation code
- configuration files
- split lists
- ROI proposal CSVs
- lightweight result summaries used in the manuscript

Due to BraTS data licensing and storage constraints, this repository does **not** include raw MRI data, preprocessed tensors, Stage-1/Stage-2 segmentation checkpoints, or full NIfTI prediction volumes.

---

## 1. What is included

### Code
- `src/train/` – training and evaluation scripts
- `src/datasets/` – dataset loaders
- `src/models/` – model definitions
- `src/metrics/` – evaluation metrics
- `src/utils/` – utility functions
- `src/postprocess/` – post-processing modules, including RC learned component filtering

### Configs
- `configs/` – original experiment configs
- `configs_split2/` – configs for the additional patient-level split analysis

### Splits
- `data/splits/` – primary internal patient-level split
- `data/splits_split2/` – additional patient-level split used for stability analysis

### Lightweight results and evaluation artifacts
- `results/roi_proposals/` – ROI proposal CSVs and summaries
- `results/full_stage1roi_split2/` – split-2 end-to-end evaluation summaries
- `results/fullbrain_split2_eval/` – split-2 one-stage full-brain evaluation summaries
- `results/lesionwise_split2/` – split-2 lesion-wise evaluation summaries

### Additional scripts
- `scripts/eval_lesionwise_val.py` – lesion-wise evaluation
- `scripts/export_stage1_roi_proposals_split2.py` – export Stage-1 ROI proposals for split-2
- `scripts/rc_lcf/train_rc_lcf.py` – training script for the RC learned component filter (LCF)

---

## 2. What is NOT included

This repository does **not** include:
- BraTS 2024 GLI raw MRI data (`.nii.gz`)
- Preprocessed NPZ tensors (`.npz`)
- Stage-1 or Stage-2 model checkpoints (`.pt`, `.pth`)
- Full prediction volumes (`.nii.gz`)
- nnU-Net preprocessed arrays and large training artifacts

These files are omitted because of dataset licensing and storage constraints.

---

## 3. Expected directory structure on your machine

### 3.1 Raw BraTS GLI data (not included)

Place the official BraTS 2024 GLI training data under:

```text
data/raw/brats2024_gli/training_data1_v2/
```

Each case folder is expected to contain:

```text
*-t1n.nii.gz
*-t1c.nii.gz
*-t2w.nii.gz
*-t2f.nii.gz
*-seg.nii.gz
```

### 3.2 Preprocessed tensors (not included)

This project uses NPZ tensors for fast I/O. Typical local paths are:

```text
data/preprocessed/npy_full_v1/
data/preprocessed/npy_full_v1/val/

data/preprocessed/npy_roi128_v1/
data/preprocessed/npy_coarse96_v1/
```

For the additional split analysis, corresponding split-2 roots can be created locally, e.g.:

```text
data/preprocessed/npy_full_split2_v1/
data/preprocessed/npy_roi128_split2_v1/
data/preprocessed/npy_roi128_jit8_split2_v1/
data/preprocessed/npy_coarse96_split2_v1/
```

---

## 4. Main reported results

### 4.1 Primary internal split

Final two-stage mixed ensemble + hybrid RC refinement:

```text
mean(WT,TC,ET) = 0.8305
RC Dice = 0.7809
```

### 4.2 Additional patient-level split (split-2)

One-stage full-brain baseline:

```text
mean(WT,TC,ET) = 0.7477
RC Dice = 0.6440
```

Two-stage mixed ensemble + size-only RC filtering:

```text
mean(WT,TC,ET) = 0.7865
RC Dice = 0.7112
```

Two-stage mixed ensemble + hybrid RC refinement:

```text
mean(WT,TC,ET) = 0.7865
RC Dice = 0.7326
```

Stage-1 ROI proposal quality on split-2:

```text
mean coverage = 0.9995
minimum coverage = 0.9816
fallback cases = 0
```

---

## 5. Main included result files

### 5.1 Stage-1 ROI proposals

Primary split:

```text
results/roi_proposals/stage1_20260102_thr0p35/val_roi128.csv
```

Additional split:

```text
results/roi_proposals/stage1_split2_thr0p35/val_roi128.csv
results/roi_proposals/stage1_split2_thr0p35/summary.json
```

### 5.2 Split-2 end-to-end evaluation summaries

Two-stage mixed ensemble + size-only:

```text
results/full_stage1roi_split2/mixed_rcmin120/full_stage1roi_metrics_rcmin120.json
results/full_stage1roi_split2/mixed_rcmin120/full_stage1roi_summary_rcmin120.csv
results/full_stage1roi_split2/mixed_rcmin120/full_stage1roi_per_case_rcmin120.csv
```

Two-stage mixed ensemble + hybrid:

```text
results/full_stage1roi_split2/final_hybrid_rcmin90_thr0p46/full_stage1roi_metrics_rcmin90.json
results/full_stage1roi_split2/final_hybrid_rcmin90_thr0p46/full_stage1roi_summary_rcmin90.csv
results/full_stage1roi_split2/final_hybrid_rcmin90_thr0p46/full_stage1roi_per_case_rcmin90.csv
```

### 5.3 Split-2 one-stage full-brain evaluation summary

```text
results/fullbrain_split2_eval/rcmin0/fullbrain_metrics_rcmin0.json
```

### 5.4 Split-2 lesion-wise evaluation summaries

```text
results/lesionwise_split2/lesionwise_mixed_rcmin120.csv
results/lesionwise_split2/lesionwise_final_hybrid_rcmin90_thr0p46.csv
```

---

## 6. Reproducibility note

This repository provides the code and lightweight result files needed to reproduce the reported pipeline structure and evaluation workflow. Users must prepare the BraTS 2024 GLI data locally in accordance with the dataset license.

A typical workflow is:

1. Prepare the BraTS 2024 GLI dataset locally
2. Generate or verify patient-level split lists
3. Preprocess full-volume, coarse-localization, and ROI tensors
4. Train Stage 1 and export ROI proposals
5. Train Stage 2 models and run mixed ensembling
6. Run one-stage and two-stage evaluation
7. Apply RC-specific post-processing and lesion-wise analysis

---

## 7. RC-specific post-processing

The default RC false-positive suppression used in the manuscript is a hybrid strategy:

- apply the learned component filter (LCF) to predicted RC connected components
- remove remaining small RC components with size-based filtering

The repository includes:

- RC learned component filter code: `src/postprocess/`
- LCF training script: `scripts/rc_lcf/train_rc_lcf.py`

The lightweight LCF model file may also be included when available in the repository results directory.

---

## 8. Data availability

This work uses the BraTS 2024 GLI dataset released by the challenge organizers under their data access and usage agreement. No additional patient data were collected in this study.

---

## 9. Code availability

This repository contains the pipeline code used for preprocessing, ROI proposal generation, Stage-2 training, ensembling, RC-specific post-processing, and evaluation, together with lightweight result files used to document the main findings reported in the manuscript.
