# Requirements Map

## Purpose
This document translates the experiment specifications into a cross-experiment requirement map for the refactor.

Its purpose is to identify:
- shared capabilities required across experiments
- experiment dimensions that must be swappable
- common data, logging, and output requirements
- optional extensions that should be supported without shaping the core architecture

This document is the bridge between:
- `docs/experiments/*.md`
- `docs/refactor/plan.md`
- the target code structure

---

## Experiment Inventory

### Core experiments
- `dataset_generation.md`
  - varies trajectory/data sampling strategy and dataset budget

- `optimizer_comparison.md`
  - varies optimizer and optimizer schedule / multi-phase setup

- `multi_stage_training.md`
  - varies training structure via stage-wise residual correction

- `loss_balancing.md`
  - varies loss formulation and weighting strategy

- `collocation_sampling.md`
  - varies collocation/residual point placement strategy and collocation budget

### Optional / secondary experiments
- `vrba.md`
  - varies adaptive physics-supervision pipeline (weighting / sampling / potential / smoothing)

- `curriculum_learning.md`
  - varies supervised-row exposure schedule based on trajectory difficulty

---

## Shared Requirements Across Experiments

The following requirements appear across most or all experiments:

### Common run structure
- configuration-driven execution
- fixed random seed support
- reproducible run metadata
- consistent per-run artifact structure
- support for repeated runs across seeds
- aggregation across seeds

### Common evaluation structure
- in-distribution (ID) evaluation
- out-of-distribution (OOD) evaluation
- comparable metric definitions across experiments
- comparable timing / cost reporting across experiments

### Common reporting structure
- saved effective config
- metrics
- timings
- run metadata
- summarized results for comparison

### Common execution structure
- local execution via Python scripts
- HPC execution via `lsf.sh` wrappers that call Python scripts
- experiment-specific Python files are allowed
- shared orchestration logic should be reused underneath

---

## Swappable Experiment Dimensions

The refactor should make the following dimensions configurable and composable without code duplication.

### Data / dataset dimensions
- trajectory sampling strategy
- dataset size / simulator budget
- random seed
- difficulty metadata availability

### Training dimensions
- optimizer
- optimizer schedule / phase transitions
- single-stage vs multi-stage training
- curriculum enabled / disabled
- curriculum unlock schedule
- optional trainer modes such as vRBA

### Loss dimensions
- data-only vs PINN loss
- static vs adaptive weighting
- warmup or staged loss activation
- optional advanced weighting schemes

### Physics supervision dimensions
- collocation sampling strategy
- collocation budget / density
- collocation budget schedule
- pool allocation strategy
- optional residual-based adaptive weighting / sampling via vRBA

### Reporting dimensions
- local artifact logging
- optional W&B reporting
- diagnostics enabled / disabled
- summary aggregation

---

## Shared Metric Requirements

A shared metric schema should exist across experiments.

### Core trajectory-performance metrics
These should be available wherever applicable:
- MSE
- RMSE
- MAE
- percentile errors (e.g. p90, p95)
- worst-case trajectory error

### Generalization metrics
- ID test error
- OOD test error
- ID–OOD gap

### Convergence metrics
- loss vs epoch
- loss vs walltime
- convergence speed
- convergence success/failure

### Cost metrics
- training walltime
- preprocessing time where relevant
- dataset generation time where relevant
- number of simulator calls where relevant
- number of residual evaluations where relevant
- memory usage where relevant

### Experiment-specific diagnostics
These vary by experiment but must be supported where required:
- sampled IC distributions
- collocation point distributions
- loss-term evolution
- weight evolution
- stage-wise contribution
- curriculum active bin vs epoch
- residual variance
- optional gradient SNR

---

## Logging and Reporting Requirements

Logging/reporting must be separated into distinct responsibilities.

### Metric computation
The codebase should define metrics in one shared place or shared layer.
Metric definitions must not depend on the reporting backend.

### Artifact logging
Per-run local artifacts in `outputs/` are the source of truth.

### Reporting backends
- console/file logging
- JSON/CSV summaries
- optional W&B integration

### W&B policy
W&B should act as a thin reporting backend over already computed metrics and diagnostics.
A run must remain valid without W&B.

### Current logging implications
The refactor must address:
- missing core metrics such as MAE
- experiment-specific logging gaps
- bloated and tightly coupled metric/reporting logic

---

## Data and Preprocessing Requirements

Several experiments depend on preprocessing artifacts and metadata contracts.

### Shared preprocessing requirements
- deterministic preprocessing
- preserved float64 precision for PINN Training
- stable train/validation/test semantics
- stable normalization metadata
- stable HDF5 / dataset contracts where relevant

### Experiment-specific metadata requirements
- curriculum learning requires:
  - `difficulty_score`
  - `difficulty_bin`

- dataset generation and analysis may require:
  - marker features
  - trajectory statistics
  - sampling metadata

- collocation and PINN-related experiments may require:
  - collocation/init export support
  - residual evaluation metadata

### Architectural implication
Preprocessing must make experiment dependencies explicit.
Training-time failures caused by missing preprocessing metadata should be validated earlier and surfaced clearly.

---

## Execution and Script Requirements

### Core rule
Python scripts remain the core execution path.
HPC scripts wrap them.

### Current known entrypoints
- `00_create_dataset.py`
- `01_preprocess_dataset.py`
- `10_run_baseline`
- `20_run_pinn.py`

### Script organization requirement
The refactor should allow one Python file per experiment where useful, but should avoid duplicated orchestration logic across scripts.

### Shared orchestration responsibilities
These should be reusable rather than copied:
- config resolution
- run naming
- manifest construction
- seed expansion
- output directory creation
- logging setup
- result bookkeeping

---

## Cross-Experiment Architectural Implications

The experiment specs imply the need for a small number of shared building blocks.

### 1. Data / preprocessing contract
A clear contract is needed between:
- dataset generation
- preprocessing
- runtime dataset loading

This contract must include:
- core supervised data
- optional metadata such as difficulty bins
- split semantics
- normalization metadata

### 2. Trainer core
A smaller shared training core should handle:
- common loop structure
- optimizer stepping
- checkpointing when enabled
- metric emission
- evaluation triggers

This should be separated from:
- multi-stage logic
- curriculum logic
- vRBA logic
- collocation adaptation
- experiment-specific diagnostics

### 3. Trainer modes / feature hooks
Optional and experiment-specific behaviour should be added through modular modes or hooks, not by overloading one giant trainer surface.

Likely categories:
- single-stage vs multi-stage
- curriculum mode
- collocation adaptation mode
- vRBA mode
- weighting mode

### 4. Sampling abstractions
At least two distinct sampling concepts exist and should not be conflated:
- trajectory/data sampling
- collocation/residual sampling

These may share utilities, but they are different experiment dimensions and should remain conceptually separate.

### 5. Loss / weighting layer
Loss formulation and weighting strategy should be configurable without duplicating the training loop.

### 6. Metrics / logging layer
Metric computation, artifact writing, and reporting backends should be separated.

### 7. Evaluation layer
Evaluation should be shared across experiments and consistently handle:
- ID / OOD datasets
- common metrics
- aggregation-ready outputs

### 8. Orchestration layer
Experiment scripts should remain thin and rely on shared orchestration/bookkeeping helpers.

---

## Optional Methods Policy

The following methods are optional / secondary and should be supported without shaping the core architecture:
- vRBA
- curriculum learning

The following methods require extra caution due to instability or sensitivity:
- stochastic second-order methods
- vRBA
- ReLoBRaLo
- LRA

These should be structurally integrated, but their functionality should not be changed unintentionally during refactor.

---

## Immediate Refactor Priorities Derived From This Map

1. Map and reduce overload in `trainer.py`
2. Define a shared metric schema, including MAE
3. Separate metric computation from reporting backends, including W&B
4. Standardize `outputs/` vs `results/`
5. Make preprocessing dependencies explicit, especially for curriculum metadata
6. Extract shared orchestration logic from duplicated experiment runner scripts
7. Introduce clearer boundaries between trainer core and optional trainer modes
8. Keep trajectory sampling and collocation sampling as distinct concepts in the architecture