# Experiment: Adaptive Data and Collocation Augmentation

## Objective
Evaluate whether adding supervised trajectories and collocation points during PINN training improves sample efficiency, convergence behaviour, and final surrogate performance compared to fixed-budget training.

This experiment addresses:
- Whether model-error-based supervised acquisition improves over random trajectory growth
- Whether residual-based collocation growth improves physics enforcement under a staged budget
- Whether combining supervised and collocation growth is better than either mechanism alone
- The trade-off between acquisition overhead, residual evaluations, labelled data budget, and accuracy

---

## Research Questions
- Does adding supervised trajectories during training improve final accuracy compared to a fixed low-data baseline?
- Does MAE-guided trajectory acquisition outperform random trajectory acquisition at the same final labelled budget?
- Does residual-based collocation growth improve performance when supervised data is also growing?
- Is the combined adaptive strategy more sample-efficient than using only supervised growth or only collocation growth?
- Does adaptive augmentation improve OOD generalization or mainly reduce ID/test error?
- How sensitive are results to acquisition cadence and per-refresh budget increments?

---

## Independent Variables

- **Supervised acquisition strategy**
  - fixed_low_data
  - random_growth
  - mae_nearest_growth

- **Supervised trajectory budget**
  - initial active trajectory count
  - trajectories added per acquisition refresh
  - final active trajectory count

- **Collocation growth strategy**
  - static_low_budget
  - rar_d_growth

- **Collocation budget**
  - initial active collocation point count
  - collocation points appended per refresh
  - final active collocation point count

- **Acquisition cadence**
  - refresh interval in epochs

- **Random seed**
  - multiple seeds per configuration (e.g. 3-5)

---

## Controlled Variables
- Fixed preprocessed dataset containing the supervised candidate pool
- Fixed ID and OOD evaluation datasets
- Fixed model architecture, activation, initialization, and optimizer schedule
- Fixed loss formulation and static/dynamic loss weighting configuration unless explicitly varied
- Fixed preprocessing semantics, splits, and PINN runtime precision
- Fixed final supervised and collocation budgets within fair comparisons

---

## Required Inputs

The pipeline must support:

- Configurable supervised acquisition strategy
- Configurable initial and final supervised trajectory budgets
- Configurable trajectories added per refresh
- Configurable collocation growth strategy
- Configurable initial and final collocation budgets
- Configurable collocation points appended per refresh
- Configurable acquisition cadence
- Configurable random seed

The dataset must provide:

- Supervised training rows with `trajectory_id_*` metadata
- A labelled supervised candidate pool contained in the training split
- Preprocessed initial-condition rows for PINN IC constraints
- Optional preprocessed collocation rows for static collocation baselines

Fixed evaluation datasets:

- ID dataset
- OOD dataset

---

## Required Outputs

Each run must produce:

### Metrics (per run)
- ID test error
- OOD test error
- Mean trajectory error
- Percentile errors (e.g. p90, p95)
- Worst-case trajectory error
- Final train/validation/test MSE, RMSE, and MAE

### Supervised acquisition diagnostics
- Active supervised trajectory count over time
- Active supervised row count over time
- Candidate supervised trajectory count over time
- Newly acquired trajectory IDs per refresh
- Acquisition score statistics for selected and candidate trajectories
- Acquisition overhead / scoring time

### Collocation diagnostics
- Active collocation point count over time
- Collocation append count per refresh
- Residual score statistics for candidate points
- Number of residual evaluations used for acquisition
- Collocation update overhead

### Convergence metrics
- Total loss vs epoch
- Total loss vs walltime
- Data, physics, dt, and IC losses over time
- Stability indicators (e.g. plateaus, oscillations, failures)

### Cost metrics
- Total training walltime
- Number of active supervised trajectories
- Number of acquired supervised trajectories
- Number of residual candidate evaluations
- Acquisition overhead split by supervised and collocation acquisition

### Aggregation (across seeds)
- Mean and standard deviation of all final metrics
- Mean and standard deviation of acquisition overhead and final budgets

---

## Evaluation Setup

- **Training data**
  - Pool-based active learning setup.
  - A larger labelled training pool is preprocessed before the PINN run.
  - Only an initial subset of trajectory IDs is visible at epoch 1.
  - Acquisition reveals additional whole trajectories from the candidate pool.
  - Supervised acquisition budgets are counted after splitting, in active
    train trajectories. They are not total generated-trajectory budget labels.

- **Supervised acquisition**
  - Acquisition is trajectory-level, not row-level.
  - `mae_nearest_growth` scores active trajectories by model MAE, selects the hardest active trajectories as anchors, and activates hidden trajectories nearest to those anchors in normalized initial-condition space.
  - `random_growth` activates candidate trajectories uniformly at random.
  - Candidate trajectory labels must not be used for the `mae_nearest_growth` selection decision.
  - Validation, test, ID evaluation, and OOD evaluation rows must never be used for acquisition.

- **Training-time regression metrics**
  - Log periodic MAE/RMSE/MSE for the active supervised train set, full train pool, validation split, and test split.
  - Analyze these alongside weighted PINN losses because adaptive growth can make the optimization loss increase when harder data is appended.
  - The full-train and fixed validation/test metrics provide stable curves across acquisition events.

- **Collocation acquisition**
  - Collocation growth uses residual-based append behaviour.
  - The default growth strategy is RAR-D: candidate collocation points are scored by residual and appended from a residual-weighted distribution.
  - This is the append/refinement counterpart to fixed-budget RAD and should be reported clearly as collocation budget growth.

- **Training budget**
  - Runs must track epochs, walltime, supervised acquisition events, and collocation acquisition events.
  - Fair comparisons should keep final supervised and collocation budgets equal unless budget scaling is the explicit independent variable.
  - The intended thesis run uses a `b4096` total reference pool, yielding about
    3276 train trajectories with the standard 0.8 split. The supervised schedule
    is 256 initial active train trajectories, 32 added per refresh, and 512 final
    active train trajectories.

---

## Metrics

### Performance
- Mean trajectory error (e.g. RMSE / MSE)
- MAE, RMSE, and MSE on train, validation, test, ID evaluation, and OOD evaluation sets
- Percentile errors (p90, p95)
- Worst-case trajectory error

---

### Generalization
- ID test error
- OOD test error
- ID-OOD gap
- Test-OOD gap

---

### Supervised Acquisition Behaviour
- Selected trajectory IDs
- Acquisition scores per refresh
- Active/candidate trajectory counts
- Error distribution of acquired vs non-acquired candidate trajectories

---

### Physics / Residual Behaviour
- Training residual loss over time
- Residual score statistics for collocation candidates
- Active collocation point count over time
- Residual evaluations spent on acquisition

---

### Convergence Behaviour
- Loss vs epochs
- Loss vs walltime
- Accuracy vs active supervised budget
- Accuracy vs active collocation budget

---

### Computational Cost
- Total training walltime
- Supervised acquisition scoring time
- Collocation acquisition scoring time
- Number of supervised candidate rows evaluated
- Number of collocation candidate points evaluated

---

## Run Matrix

Each experiment run is defined by:

```text
supervised_acquisition_strategy x supervised_budget_schedule
x collocation_growth_strategy x collocation_budget_schedule
x cadence x seed
```

Feasible screening subsets are expected before full runs.

---

## Key Comparisons

- Fixed low-data baseline vs random supervised growth
- Random supervised growth vs MAE-guided supervised growth
- Static collocation vs residual-based collocation growth
- Supervised-only growth vs collocation-only growth
- Combined supervised + collocation growth vs each single-growth baseline
- Accuracy vs labelled trajectory budget
- Accuracy vs residual evaluation budget
- ID vs OOD generalization under adaptive growth

---

## Required Outputs / Plots

The pipeline must enable generation of:

- Error vs active supervised trajectory count
- Error vs active collocation point count
- Error vs walltime
- Acquisition score distributions over refreshes
- Active budget schedules over epochs
- Selected trajectory diagnostics in IC space
- Static vs random-growth vs MAE-growth comparison tables
- Collocation static vs RAR-D-growth comparison tables
- Combined-growth ablation plots

---

## Success Criteria (for Refactor)

The codebase supports this experiment if:

- Supervised acquisition can be enabled/disabled via configuration
- Supervised acquisition operates on trajectory IDs and does not require trainer forks
- Random and MAE-guided trajectory acquisition use the same training loop
- Collocation growth reuses existing residual-based collocation infrastructure
- Runs save the effective config, metrics, timings, and acquisition diagnostics
- Results can be aggregated across seeds automatically
- Existing fixed-dataset PINN, collocation, loss-weighting, and multistage experiments remain runnable without behaviour changes

---

## Notes / Assumptions

- MAE-guided acquisition requires labels for candidate trajectories. The first implementation should therefore use a precomputed labelled candidate pool rather than simulator-in-the-loop generation.
- The experiment is pool-based active learning / active reveal, not fully online dataset generation.
- Acquiring whole trajectories preserves the meaning of supervised data budget and avoids row-level leakage.
- Newly preprocessed supervised HDF5 files should always include `trajectory_id_*`
  provenance. Difficulty score/bin metadata is optional legacy curriculum
  metadata and is not part of the adaptive augmentation contract.
- RAR-D is the appropriate existing mechanism for append-style collocation growth. Fixed-budget RAD should remain available as the replacement-style adaptive collocation baseline.
- Acquisition overhead must be tracked separately from normal training walltime where possible.
- Precision behaviour must follow the existing PINN runtime expectations; adaptive scoring should not accidentally downgrade PINN tensors from float64.
