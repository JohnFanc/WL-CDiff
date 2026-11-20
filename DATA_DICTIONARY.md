# Data dictionary

## Sample organization

There are 1600 synthetic realizations. The directory `dataset/v1/train` contains the 1500-sample pool used to form 1350 training and 150 validation samples, not 1500 final training samples. Held-out IDs 1501–1600 are in `test`. Exact assignments and row order are in `dataset/split_indices.csv`; the supplied HDF5 caches are authoritative for those assignments. All clean well-density observation variants refer to the same realization IDs.

## Raw/reference files

- `fields/lnKfield_NNNN.csv`: 50 × 50 reference lnK field. Values are the natural logarithm of conductivity numerically expressed in m/day.
- `fields/Kfield_NNNN.csv` and `.dat`: corresponding K field, m/day. CSVs are read by the included model pipeline; `.dat` files are retained as original dataset companions.
- `train/fields/eig_decomp.npz`: existing eigendecomposition data; retained as a dataset companion, not needed for the documented training command.
- `heads/HeadNNNN.h5`: simulated input heads. `Datasets/Head/Values` has shape (51, 2500) and is reshaped by the loader to (time, row, column) = (51, 50, 50); `Datasets/Head/Times` supplies time values. Heads are in metres and physical time is in days. GUID/Active/Min/Max datasets are original exporter metadata.
- `obData/HeadNNNN.npz`: 36-well observations. `obData_s04`, `_s09`, `_s18` use 4, 9, 18 wells. Test folders with `noise_s003`/`noise_s005` contain existing Gaussian-noise observations at 0.03/0.05 m; metadata specifies seed and settings. The extra 0.05 m and 18-well noisy variants are included as available dataset variants, without claiming they are all reported in the manuscript.

## Observation NPZ schema

- `sample_id`: scalar integer realization ID.
- `obs_abs`: (S, 51), head in metres.
- `obs_feat`: (S, 51, 3), channels are head h(t), h(t)-h(t0), and time derivative from `numpy.gradient(h, times, axis=1)`. First two channels are in metres; derivative is in metres/day. The subtraction uses the first stored time point.
- `sensor_xy`: (S, 2), zero-based integer (column x, row y) indices; the loader samples `head[:, y, x]`. These are grid indices, not metre coordinates. Model coordinates are divided by 49.
- `times`: (51,), physical simulation times, not diffusion-step indices.
- `head_file`: original HDF5 basename.

## Processed HDF5 schema

`train.h5`, `val.h5`, `test.h5` contain `obs_feat` (n, S, 51, 3), `lnk` (n, 50, 50), `sample_id` (n,), `sensor_xy` (S, 2), and `times` (51,). Stored field/features are unstandardized float32 values. Sample IDs are integers. Standardization is applied by `SensorDataset` using the companion `stats.json`.

For each feature, z=(value-training mean)/training standard deviation; lnK is standardized similarly. The standard deviations include 1e-6 in the preprocessing code. K-space predictions are obtained after undoing lnK standardization and exponentiating. Statistics must come from the corresponding training density, never the test set.

## Not distributed

No learned weights, checkpoints, model-predicted fields, published metric tables, training logs, or finished paper figures are included. The raw simulated heads and known K/lnK fields are inputs/reference labels for training/testing and are included intentionally.
