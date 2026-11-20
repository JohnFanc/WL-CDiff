# WL-CDiff data and code

Companion materials for **Hydraulic conductivity inversion from sparse monitoring-well observations using a conditional diffusion model**.

This package provides synthetic input/reference datasets and source code for preprocessing, model training, inference and evaluation. Trained model weights, historical prediction arrays, experiment metrics/logs and finished paper figures are not distributed. Train a model before running inference. A new training run is not guaranteed to reproduce each published numerical value exactly.

## Data download

The datasets are distributed separately from this code repository. A companion Zenodo record has been reserved as DOI `10.5281/zenodo.24384634`; publication is pending. After the record is published, download the complete archive and extract it. Either run the included code directly from the extracted archive, or copy its `dataset/` directory and `SHA256SUMS.csv` into this repository root. The manifest verifies the complete archive; it is not a manifest of this code-only repository.

## Contents

- `src/`: original model, loss, dataset, inference, evaluation and plotting implementations.
- `configs/`: model/training configuration and 4/9/18/36-well coordinate files; `_abl_s04_*` are existing ablation configurations.
- `dataset/v1/train/`: combined 1500-sample training/validation pool, IDs 1–1500.
- `dataset/v1/test/`: 100 held-out samples, IDs 1501–1600.
- `dataset/processed/v1/`: supplied 36-well train/validation/test HDF5 files (1350/150/100 samples).
- `dataset/processed/v1_s18/`: supplied 18-well HDF5 files with the same split.
- `dataset/split_indices.csv`: exact sample assignments and their baseline HDF5 row order.
- `DATA_DICTIONARY.md`: array definitions, units, coordinates and normalization.
- `run.py`: portable entry point for the existing workflow.
- `verify_package.py`, `SHA256SUMS.csv`: integrity and data checks.
- `smoke_check.py`: small untrained-model forward/backward check, without saving weights or predictions.

## Environment

The locally checked environment uses Python 3.9.25 and PyTorch 2.8.0+cu128. See `environment_reference.json`. These are observed installed versions, not a historical lockfile. Use an isolated environment; install the appropriate PyTorch 2.8.0 build for your CPU/GPU, then install the other pinned dependencies:

```sh
python -m venv .venv
# Activate .venv using your shell's normal activation command.
python -m pip install -r requirements.txt
python verify_package.py
python smoke_check.py
```

GPU training is recommended. The source falls back to CPU when CUDA is unavailable. CPU smoke checks are short; full 2000-epoch training is substantially more expensive. No binaries, external services or credentials are bundled.

## Training and inference

Run commands from the extracted package root. Baseline:

```sh
python run.py prepare --wells 36
python run.py train --wells 36 --device cuda
python run.py infer --wells 36 --device cuda
python run.py evaluate --wells 36
```

Training automatically builds its cache from the 1500 raw observation/reference pairs, using a 10% validation split with seed 42. The separate `prepare` command is optional. Normalization statistics are calculated from the 1350 training samples only. Training writes new checkpoints/logs under `outputs/public_s36/train/`; inference uses the newly trained `best.pt` and writes new results under `outputs/public_s36/infer/clean/`. These output folders do not exist in the release archive.

Select another density using `--wells 4`, `9` or `18` consistently for preparation, training, inference and evaluation. `--config configs/_abl_s04_E2.yaml --tag s04_E2` selects an existing ablation configuration and isolates its generated outputs. Use the same tag for inference/evaluation; checkpoint metadata contains the model configuration. This is a convenience interface, not evidence that every run matches a historical published experiment.

For the supplied 0.03 m noisy test observations, train the clean 36-well model first, then:

```sh
python run.py infer --wells 36 --noise 0.03 --device cuda
python run.py evaluate --wells 36 --noise 0.03
```

For a short local execution check, `python run.py train --wells 36 --epochs 1 --batch-size 1 --num-workers 0 --tag execution_check` is supported, but still traverses the full training dataset. Prefer `smoke_check.py` for a fast model/array check. The smoke check does not assess model accuracy.

Run `python run.py --help` for options. `--dry-run` prints resolved paths and checks required inputs without running training/inference. Inference requires a locally trained checkpoint, or an explicit checkpoint you are authorized to use. No checkpoint is supplied here.

## Configuration and reproducibility boundaries

`default.yaml` currently specifies 300 diffusion training timesteps, 100 DDIM inference steps, 8 generated samples per input and guidance scale 2.0. The inference script calculates metrics from the mean generated lnK field and exponentiates that mean for K-space metrics. These current source defaults are preserved; they must not be presented as independently verified settings for every historical table/figure. In particular, distinguish diffusion training timesteps from DDIM sampling steps when describing a run.

Some original scripts retain edited, experiment-specific defaults (4-well train/inference, 18-well registry/evaluation). The wrapper resolves these explicitly; do not assume `src/run_config.py` controls every script. No original weights/outputs are provided, so the archive cannot directly re-render all historical result figures. The plotting code remains available after suitable new output paths and sample selections are configured.

SSIM in `src/utils/metrics.py` is computed once over all grid cells per field pair, using population variances/covariance, then averaged across test samples. Its data range is calculated separately from the joint minimum and maximum of each pair; it is not a sliding-window average. Do not substitute a different library's default SSIM implementation when comparing results.

The original field-generation and MODFLOW project files were not present in the selected source tree. Included simulated heads are dataset inputs, not withheld model-inversion outputs. See `PACKAGING_CHANGES.md` and `VALIDATION_REPORT.json` for the inspected scope.

## Repository publication

Code repository: https://github.com/JohnFanc/WL-CDiff. The companion Zenodo record is being prepared; the reserved DOI becomes active only upon publication. Author metadata and reuse licenses are pending confirmation. No model weights or historical predictions are distributed or promised on request.
