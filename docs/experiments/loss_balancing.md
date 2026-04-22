# Experiment: Loss Balancing

## Objective
Evaluate how different loss formulations and weighting strategies affect training stability, convergence behaviour, and final surrogate model performance in physics-informed learning.

This experiment addresses:
- Whether physics-informed training improves accuracy compared to data-only approaches  
- How different loss terms (data, physics, IC) interact during training  
- Which loss weighting strategies stabilize PINN training for stiff dynamics  
- The trade-off between training complexity and performance improvement  

---

## Research Questions
- Does physics-informed training improve generalization compared to data-only models?  
- At what data regime does physics supervision become beneficial?  
- Which loss weighting scheme produces the most stable training?  
- Do data and physics losses compete or support each other during training?  
- Are complex adaptive weighting schemes necessary, or are simpler approaches sufficient?  

---

## Independent Variables

- **Loss formulation**
  - data_only  
  - pinn_uniform  
  - pinn_static_tuned  
  - pinn_data_warmup_then_static  

- **Weighting scheme**
  - static (baseline)  
  - ma  
  - id  
  - dn  
  - ntk  
  - lra  
  - relobralo  

- **Random seed**
  - multiple seeds per configuration (e.g. 3–5)

---

## Controlled Variables
- Dataset (fixed dataset generation strategy and size)  
- Surrogate model architecture  
- Optimizer (fixed baseline)  
- Collocation strategy  
- Data preprocessing  
- Evaluation datasets (must be fixed across all runs)  

---

## Required Inputs

The pipeline must support:

- Configurable loss formulation (data-only vs PINN variants)
- Configurable weighting scheme
- Configurable random seed
- Fixed dataset input
- Fixed evaluation datasets:
  - ID dataset
  - OOD dataset

- Optional:
  - Configurable warmup phase (for staged loss application)
  - Configurable initial/static weights

---

## Required Outputs

Each run must produce:

### Metrics (per run)
- ID test error  
- OOD test error  
- Mean trajectory error  
- Percentile errors (e.g. p90, p95)  
- Worst-case trajectory error  

### Loss tracking (critical)
- Data loss over time  
- Physics loss over time  
- IC loss (if present)  
- Relative scale of loss terms  
- Weight evolution over time (for adaptive schemes)  

### Convergence metrics
- Total loss vs epoch (logged)
- Total loss vs walltime (logged)
- Stability indicators (e.g. oscillations, plateaus)

### Cost metrics
- Total training walltime  
- Overhead of weighting scheme  

### Stability metrics
- Convergence success/failure flag  
- Variance across seeds  

### Aggregation (across seeds)
- Mean and standard deviation of all metrics  

---

## Evaluation Setup

- **Training data**
  - Fixed dataset used across all configurations  

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

### Loss Behaviour (critical)
- Data loss vs time  
- Physics loss vs time  
- IC loss (if present)  
- Relative magnitude of each loss term  
- Weight evolution (for adaptive schemes)  

---

### Convergence Behaviour
- Total loss vs epochs  
- Total loss vs walltime  
- Training stability (oscillations, plateaus)  

---

### Computational Cost
- Total training walltime  
- Overhead of weighting scheme  

---

### Stability / Robustness
- Variance across seeds  
- Convergence failures  

---

## Run Matrix

Each experiment run is defined by: loss_formulation × weighting_scheme × seed
All combinations must be runnable.

---

## Key Comparisons

- Data-only vs physics-informed training  
- Static vs adaptive loss weighting  
- Accuracy vs computational cost trade-off  
- Stability of different weighting strategies  

---

## Required Outputs / Plots

The pipeline must enable generation of:

- ID / OOD error comparison across methods  
- Error vs walltime  
- Data loss vs time  
- Physics loss vs time  
- Weight evolution (adaptive methods)  
- Loss ratio (data vs physics)  
- Correlation between physics loss and generalization error  
- Example trajectory comparisons  

---

## Success Criteria (for Refactor)

The codebase supports this experiment if:

- Loss formulations are modular and configurable  
- Weighting schemes are interchangeable without code changes  
- All loss terms are logged consistently across runs  
- Weight evolution is tracked for adaptive schemes  
- Results can be aggregated across seeds automatically  
- No duplication of training logic is required  

---

## Notes / Assumptions

- All methods are compared on the same dataset and model  
- Optimizer is fixed to isolate loss effects  
- Loss scales must be tracked and interpreted carefully  
- Adaptive weighting overhead must be accounted for