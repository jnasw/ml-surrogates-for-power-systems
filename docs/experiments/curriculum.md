# Experiment: Curriculum Learning

## Objective
Evaluate whether curriculum learning—progressive exposure of supervised training data based on trajectory difficulty—improves convergence behaviour, training stability, and final surrogate model performance compared to standard full-data training.

This experiment addresses:
- Whether staged exposure to increasing trajectory difficulty improves optimization  
- Whether curriculum learning improves convergence speed and stability  
- Whether curriculum learning improves ID and OOD generalization  
- Whether balancing samples across difficulty bins improves training effectiveness  
- The trade-off between added training complexity and performance gain  

---

## Research Questions
- Does curriculum learning improve final accuracy compared to standard training?  
- Does curriculum learning improve convergence speed or stability?  
- Is curriculum learning especially beneficial for difficult trajectories or OOD regions?  
- How does performance evolve as more difficult bins are unlocked?  
- Does balancing samples across active bins improve performance compared to naive sampling?  
- Is the added complexity of curriculum learning justified by performance gains?  

---

## Independent Variables

- **Training strategy**
  - no_curriculum (baseline: full supervised dataset available from start)
  - curriculum_enabled

- **Curriculum configuration**
  - num_bins (e.g. 3, 5, 10)
  - unlock schedule:
    - explicit `unlock_epochs`
    - or fixed `stage_length_epochs`

- **Sampling strategy within active bins**
  - standard sampling
  - balanced_active_bins (on/off)

- **Random seed**
  - multiple seeds per configuration (e.g. 3–5)

---

## Controlled Variables
- Dataset (same dataset and preprocessing across runs)  
- Difficulty scoring method (fixed marker-based definition)  
- Surrogate model architecture  
- Optimizer / training schedule  
- Loss formulation  
- Collocation strategy  
- Data preprocessing (except curriculum metadata)  
- Evaluation datasets (must be fixed across all runs)  

---

## Required Inputs

The pipeline must support:

- Configurable curriculum toggle (enabled / disabled)
- Configurable number of difficulty bins
- Configurable unlock schedule:
  - explicit epoch list
  - or fixed stage length
- Configurable sampling strategy within active bins (balanced vs unbalanced)
- Configurable random seed

- Preprocessed dataset with:
  - `difficulty_score`
  - `difficulty_bin`

- Fixed evaluation datasets:
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

### Convergence metrics
- Loss vs epoch (logged)
- Loss vs walltime (logged)
- Training stability indicators (oscillations, plateaus)

### Curriculum diagnostics
- Active maximum difficulty bin vs epoch  
- Fraction of dataset visible vs epoch  
- Distribution of sampled bins per epoch  
- Performance metrics per difficulty bin  

### Cost metrics
- Total training walltime  
- Epoch count  

### Stability metrics
- Convergence success/failure flag  
- Variance across seeds  

### Aggregation (across seeds)
- Mean and standard deviation of all metrics  

---

## Evaluation Setup

- **Training data**
  - Same dataset across all runs  
  - Curriculum controls which rows are visible at each epoch  

- **Difficulty definition**
  - Marker-based trajectory difficulty score  
  - Quantile-based binning computed from training data  

- **Test datasets**
  - ID: same IC distribution as training  
  - OOD: extended or shifted IC distribution  

- **Training budget**
  - Must be tracked consistently across runs:
    - epochs  
    - walltime  

---

## Metrics

### Performance
- Mean trajectory error (e.g. RMSE / MSE)  
- Percentile errors (p90, p95)  
- Worst-case trajectory error  

---

### Generalization
- ID test error  
- OOD test error  
- ID–OOD gap  

---

### Convergence Behaviour
- Loss vs epochs  
- Loss vs walltime  
- Convergence speed  
- Training stability  

---

### Curriculum Behaviour
- Active bin vs epoch  
- Dataset coverage vs epoch  
- Performance vs difficulty bin  
- Effect of unlocking harder bins  

---

### Computational Cost
- Total training walltime  
- Epoch count  

---

### Stability / Robustness
- Variance across seeds  
- Convergence failures  

---

## Run Matrix

Each experiment run is defined by: training_strategy × num_bins × unlock_schedule × sampling_strategy × seed
All combinations must be runnable (with feasible subsets if necessary).

---

## Key Comparisons

- Curriculum learning vs standard training  
- Performance vs number of bins  
- Different unlock schedules  
- Balanced vs unbalanced sampling within active bins  
- Accuracy vs computational cost trade-off  
- ID vs OOD generalization under curriculum learning  

---

## Required Outputs / Plots

The pipeline must enable generation of:

- ID / OOD error comparison (curriculum vs baseline)  
- Error vs walltime  
- Loss vs epoch curves  
- Active difficulty bin vs epoch  
- Performance vs difficulty bin  
- Dataset coverage vs epoch  
- Effect of balanced vs unbalanced sampling  
- Example trajectories grouped by difficulty  

---

## Success Criteria (for Refactor)

The codebase supports this experiment if:

- Curriculum learning can be enabled/disabled via configuration  
- Difficulty bins are integrated into dataset loading  
- Unlock schedules are configurable without code changes  
- Sampling within active bins is configurable  
- Curriculum diagnostics are logged consistently  
- Results can be aggregated across seeds automatically  
- No duplication of training logic is required  

---

## Notes / Assumptions

- Difficulty is defined via marker-based trajectory scoring  
- Curriculum applies only to supervised data rows  
- Collocation points and loss formulation remain unchanged  
- Curriculum learning is optional and secondary to core experiments  
- Performance improvements must be evaluated against added training complexity  