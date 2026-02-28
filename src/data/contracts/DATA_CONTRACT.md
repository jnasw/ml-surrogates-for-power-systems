# Dataset Data Contract

This document defines the required I/O formats for the two-stage pipeline.

## 1. Raw Simulation Output (Stage 1)

Path pattern:
- `data/<model_flag>/dataset_vN/raw/file*.pkl`
- `data/<model_flag>/dataset_vN/info.txt`

Raw pickle format:
- Each file contains a Python list of trajectories.
- Each trajectory is a list of rows:
  - row `0`: time vector `t`
  - rows `1..`: channels in model-defined order (`keys + keys_ext`)

Constraints:
- All rows in a trajectory have the same length.
- At least 2 rows per trajectory (time + one channel).

`info.txt` must include at least:
- `Number of files`
- `Initial conditions file`
- `Number of different simulated trajectories`
- `Time horizon of the simulations`
- `Number of points in the each simulation`

## 2. Preprocessed Output (Stage 2)

Path pattern:
- `data/<model_flag>/dataset_vN/train/*.h5`
- `data/<model_flag>/dataset_vN/val/*.h5`
- `data/<model_flag>/dataset_vN/test/*.h5`

Supervised files:
- `train_data*.h5` with datasets: `x_train`, `y_train`
- `val_data*.h5` with datasets: `x_val`, `y_val`
- `test_data*.h5` with datasets: `x_test`, `y_test`

PINN support files (train split only):
- `train_data_col*.h5`: collocation points, only `x_train`
- `train_data_init*.h5`: initial points at `t=0`, datasets `x_train`, `y_train`

Tensor semantics:
- `x_*`: `[time, initial-condition/features...]`
- `y_*`: target dynamic states only

## 3. Naming Rules

- Raw files: `file{index}.pkl`
- H5 suffixes:
  - supervised: `_data`
  - collocation: `_data_col`
  - initial points: `_data_init`

## 4. IC Generation Methods

- `full_factorial`:
  - sample values per variable, then take Cartesian product across variables
  - dataset size is multiplicative (`prod(iterations)`)
- `joint_lhs`:
  - sample full IC vectors directly with D-dimensional LHS
  - dataset size is `model.ic_num_samples` (or `prod(iterations)` if `null`)
- `adaptive_iterative`:
  - reserved for future active/adaptive IC refinement workflows
