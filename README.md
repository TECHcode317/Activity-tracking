# HAR-15: leakage-controlled image-based human activity classification

Code accompanying the manuscript on 15-class human activity recognition with a fine-tuned CLIP ViT-B/16
(proposed) and an EfficientNet-B0 baseline. Both models classify single RGB images. There is no pose
estimation, detection or temporal modelling anywhere in this repository.

## Repository contents

| File | Purpose |
|---|---|
| `make_splits.py` | Near-duplicate audit (64-bit dHash, Hamming <= 4, connected components) and group-stratified splits. Writes the fixed manifests. |
| `train.py` | Trains, evaluates and benchmarks one model (`--model vit` or `--model effnet`). |
| `har_common.py` | Data loading, grouping, split functions, augmentation, prediction. |
| `metrics.py` | Metric computation (no GPU/torch needed). |
| `verify_metrics.py` | Recomputes the reported metrics from the saved prediction files (CPU only). |
| `manifests/` | **Created by `make_splits.py`; commit it.** `groups.csv`, `split_main.csv`, `split_cv.csv`. |
| `results/` | **Created by `train.py`; commit CSV/JSON/PNG, not `.pt`.** |

## 1. Environment

Python 3.10+, one NVIDIA GPU (the paper used a Tesla T4 on Kaggle).

    pip install -r requirements.txt

`requirements.txt` gives minimum versions only. Before publishing, replace it with the exact versions of your
run (`pip freeze | grep -i -E "torch|timm|numpy|pandas|scipy|scikit-learn|pillow|matplotlib"`); the versions are
also saved in `results/<model>/all_results.json`. Splits can depend on the scikit-learn version, which is why
the manifests are committed and used as the source of truth.

## 2. Data

Download the public Kaggle dataset `meetnagadia/human-action-recognition-har-dataset`
(12,600 labelled images, 15 classes; `Training_set.csv` and a `train/` folder).

    pip install kaggle
    kaggle datasets download -d meetnagadia/human-action-recognition-har-dataset -p data --unzip

Point the scripts at the folder containing `Training_set.csv` with `--root` or `export HAR_ROOT=...`
(subfolders are searched). On Kaggle the folder is under `/kaggle/input/...`. The images are third-party data
and are **not** redistributed here; the manifests contain file names only.

## 3. Run

    python make_splits.py --root "data/Human Action Recognition"
    python train.py --model vit --cv
    python train.py --model effnet
    python verify_metrics.py --main results/vit/predictions_main.csv --cv results/vit/predictions_cv.csv

Expected from `make_splits.py` (manuscript Section 3.2): 12,438 groups, split sizes 9,005 / 1,796 / 1,799,
no group shared between splits. If your numbers differ, stop and check the scikit-learn version and the dataset copy.

`train.py` reads the manifests, so training always uses the committed splits. `--cv` (ViT only, 4 epochs per
fold) runs the 5-fold group-stratified cross-validation. `--no-benchmark` skips the latency, FLOPs and power
measurement.

Kaggle notebook: enable GPU and Internet (pretrained weights are downloaded by timm), upload the files, then run
the commands above with `!` in front. Output goes to `/kaggle/working`.

## 4. Mapping to the manuscript

| Manuscript item | File |
|---|---|
| Group assignments, train/val/test split (Sections 3.2, 5.1) | `manifests/groups.csv`, `manifests/split_main.csv` |
| Cross-validation folds (Table 7, Figure 6) | `manifests/split_cv.csv`, `results/vit/cv_folds.csv` |
| Test metrics (Tables 5, 6, 9) | `results/<model>/results_summary.csv`, `classification_report.txt` |
| Per-image predictions of the fixed test split and of every fold | `predictions_main.csv`, `predictions_cv.csv` |
| Confusion matrix, per-class F1 (Figures 3, 4) | `confusion_matrix*.png`, `per_class_f1.png` |
| Training curves (Figure 5) | `training_curves.png`, `training_history.csv` |
| Parameters, GFLOPs, latency, power, energy (Tables 8, 9, 11, Figure 7) | `all_results.json`, `complexity_comparison.csv`, `gflops_comparison.png` |
| Hyperparameters, environment (Table 4) | `all_results.json` (`hyperparameters`) |

`verify_metrics.py` needs only `predictions_*.csv`: a reviewer can reproduce the reported accuracy, macro and
weighted precision/recall/F1, balanced accuracy, AUC, per-class report, confidence interval and CV mean and SD
without a GPU or the images. `--expect-accuracy--`.



* **Loss.** Training loss is label-smoothed cross-entropy (epsilon = 0.1, floor 0.556 for 15 classes).
  Validation loss is plain negative log-likelihood of the (EMA) model's SoftMax output, without label smoothing
  and without flip averaging, so the two are not comparable.
* **Optimiser.** `torch.optim.AdamW` (decoupled weight decay). Weight decay is applied to weight matrices only;
  bias, normalisation, `cls_token` and `pos_embed` use 0.
* **Learning rate.** ViT parameter group learning rate = 1e-4 x 0.75^(13 - l) (l = 0 for embeddings, l = 1..12 for
  blocks), head 5 x 1e-4. The top transformer block therefore trains at 0.75e-4 and the final norm at 1e-4; 1e-4 is
  the base rate eta_0 of Eq. (9). Warm-up and cosine decay are per step, with a floor of 1% of each group's rate.
* **Energy.** Power is read from `nvidia-smi` every 100 ms in a shared cloud notebook and the elapsed time
  includes 10 warm-up passes, so `energy_per_image_J` is slightly high. `energy_per_image_steady_state_J`
  (power / throughput) is saved next to it. Treat absolute values as indicative.
* **Run-to-run variation.** GPU kernels are not bit-deterministic; repeated runs with the same seed differ by about
  one accuracy point. Use `verify_metrics.py` on the
  committed predictions to reproduce the *reported* numbers exactly; a fresh training run will be close, not identical.


