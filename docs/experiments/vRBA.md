# Experiment: vRBA

## Objective
Evaluate whether variational residual-based attention (vRBA) improves PINN training accuracy, convergence behaviour, and robustness compared to simpler baseline physics-informed training pipelines.

This experiment addresses:
- Whether a variational residual-based adaptive pipeline improves surrogate accuracy compared to standard PINN training
- Whether vRBA improves training dynamics through adaptive weighting or sampling of high-residual regions
- Whether vRBA reduces discretization error by lowering residual variance
- Whether vRBA improves gradient quality and convergence behaviour sufficiently to justify its additional complexity
- Whether vRBA provides meaningful gains over simpler adaptive baselines such as residual-based weighting or adaptive collocation

---

## Research Questions
- Does vRBA improve final accuracy compared to a standard PINN baseline?
- Does vRBA improve ID and OOD generalization, or mainly improve fit in difficult residual regions?
- Does vRBA improve convergence speed or stability?
- Does vRBA reduce residual variance and thereby improve the quality of the training signal?
- Does vRBA improve gradient signal-to-noise ratio or other learning-dynamics indicators?
- Is vRBA more effective as importance weighting, importance sampling, or both?
- Do the exponential and quadratic potential variants behave differently in practice?
- Is the added algorithmic complexity and computational overhead justified by the performance gains?

---

## Independent Variables

- **Training pipeline**
  - baseline_pinn
  - vrba_weighting
  - vrba_sampling
  - (optional) vrba_hybrid

- **Potential / tilt type**
  - quadratic_potential
  - exponential_potential

- **Smoothing / adaptivity settings**
  - ema_enabled
  - smoothing_to_uniform_enabled
  - annealing_enabled

- **Random seed**
  - multiple seeds per configuration (e.g. 3–5)

---

## Controlled Variables
- Dataset (fixed dataset generation strategy and size)
- Surrogate model architecture
- Optimizer / optimizer schedule
- Loss formulation outside the vRBA mechanism
- Collocation candidate set or base collocation generation process
- Data preprocessing
- Evaluation datasets (must be fixed across all runs)

---

## Required Inputs

The pipeline must support:

- Configurable training pipeline (baseline vs vRBA variants)
- Configurable vRBA mode:
  - importance weighting
  - importance sampling
  - optional hybrid mode
- Configurable potential type
- Configurable EMA / smoothing parameters
- Configurable annealing schedule
- Configurable random seed

- Fixed dataset input
- Fixed evaluation datasets:
  - ID dataset
  - OOD dataset

- Optional:
  - Configurable collocation refresh frequency
  - Configurable candidate pool size
  - Configurable residual statistic logging frequency

---

## Required Outputs

Each run must produce:

### Metrics (per run)
- ID test error
- OOD test error
- Mean trajectory error
- Percentile errors (e.g. p90, p95)
- Worst-case trajectory error

### Residual / adaptivity metrics
- Training residual loss over time
- Residual on unseen collocation points
- Residual variance over time
- Importance weights or sampling probabilities over time
- Distribution statistics of adaptive weights / selected points

### Learning-dynamics metrics
- Loss vs epoch (logged)
- Loss vs walltime (logged)
- Gradient signal-to-noise ratio (if implemented)
- Stage / phase indicators for training dynamics (if implemented)

### Cost metrics
- Total training walltime
- Overhead of vRBA updates
- Number of residual evaluations
- Memory usage (if relevant)

### Stability metrics
- Convergence success/failure flag
- Variance across seeds

### Aggregation (across seeds)
- Mean and standard deviation of all metrics

---

## Evaluation Setup

- **Training data**
  - Fixed supervised dataset across all configurations

- **Physics supervision**
  - Baseline PINN uses standard static collocation / weighting
  - vRBA variants adapt weighting and/or sampling based on current residuals

- **Test datasets**
  - ID: same IC distribution as training
  - OOD: extended or shifted IC distribution

- **Training budget**
  - Must be tracked consistently across runs:
    - epochs
    - walltime
    - total residual evaluations

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

### Residual / Discretization Behaviour
- Training residual loss vs time
- Residual on unseen collocation points
- Residual variance vs time
- Distribution of residual magnitudes across collocation points

---

### Learning Dynamics
- Loss vs epochs
- Loss vs walltime
- Gradient signal-to-noise ratio (if available)
- Transition timing between training phases (if available)

---

### Adaptive Mechanism Behaviour
- Importance weight distribution over time
- Sampling distribution over time
- Concentration on high-residual regions
- Degree of collapse vs diversity of selected / emphasized points

---

### Computational Cost
- Total training walltime
- Overhead from vRBA updates
- Number of residual evaluations
- Memory usage (if relevant)

---

### Stability / Robustness
- Variance across seeds
- Convergence failures
- Sensitivity to vRBA hyperparameters

---

## Run Matrix

Each experiment run is defined by: training_pipeline × potential_type × smoothing_setting × seed
All combinations must be runnable (with feasible subsets if necessary).

---

## Key Comparisons

- Baseline PINN vs vRBA pipeline
- Importance weighting vs importance sampling
- Quadratic vs exponential potential
- Accuracy vs computational overhead
- Residual variance reduction vs true generalization improvement
- Learning-dynamics improvements vs final performance gains

---

## Required Outputs / Plots

The pipeline must enable generation of:

- ID / OOD error comparison across pipelines
- Error vs walltime
- Residual variance vs training progress
- Residual on unseen collocation points vs true generalization error
- Weight / sampling distribution evolution over time
- Gradient SNR or related learning-dynamics plots (if available)
- Example trajectory or residual maps showing where vRBA focuses attention
- Accuracy vs overhead comparison plots

---

## Success Criteria (for Refactor)

The codebase supports this experiment if:

- vRBA can be enabled as a separate configurable training pipeline
- Weighting and sampling variants are modular and selectable without code changes
- Potential type, smoothing, and annealing are configurable
- Residual statistics and adaptive weights / sampling distributions are logged consistently
- vRBA-specific overhead is tracked separately from base training cost
- Results can be aggregated across seeds automatically
- vRBA can be added without duplicating the core training and evaluation logic

---

## Notes / Assumptions

- vRBA is optional and lower priority than the core experiments
- The framework should support vRBA, even if the full experiment is deferred
- Baseline comparisons must use the same dataset, model, optimizer, and evaluation setup
- vRBA should be treated as a pipeline-level adaptive physics-supervision method, not merely a scalar loss-weight tweak
- Claimed gains in residual variance or SNR must always be compared against true surrogate accuracy and generalization

