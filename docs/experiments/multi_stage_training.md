# Experiment: Multi-Stage Training

## Objective
Evaluate whether multi-stage training improves surrogate model accuracy, convergence behaviour, and computational efficiency compared to a standard single-stage baseline.

This experiment addresses:
- Whether decomposing learning into sequential stages improves optimization  
- Whether residual correction stages improve final surrogate accuracy  
- Whether multi-stage training improves ID and OOD generalization  
- The trade-off between additional training complexity and performance gain  

---

## Research Questions
- Does multi-stage training improve final accuracy compared to the single-stage baseline?  
- Does multi-stage training improve convergence speed or stability?  
- Do later stages provide meaningful residual correction, or do gains saturate quickly?  
- Is multi-stage training especially beneficial for difficult trajectories or OOD regions?  
- Is the added complexity and computational overhead justified by the performance improvement?  

---

## Independent Variables

- **Training strategy**
  - single_stage (baseline)
  - multi_stage

- **Number of stages**
  - 2-stage
  - 3-stage
  - (optional) 4-stage

- **Random seed**
  - multiple seeds per configuration (e.g. 3–5)

---

## Controlled Variables
- Dataset (fixed dataset generation strategy and size)  
- Surrogate model architecture / architecture family  
- Loss formulation  
- Optimizer setup (unless explicitly varied within stages)  
- Data preprocessing  
- Evaluation datasets (must be fixed across all runs)  

---

## Required Inputs

The pipeline must support:

- Configurable training strategy (single vs multi-stage)
- Configurable number of stages
- Configurable random seed
- Fixed dataset input
- Fixed evaluation datasets:
  - ID dataset
  - OOD dataset

- Optional:
  - Configurable stage-wise training budget
  - Configurable optimizer per stage

---

## Required Outputs

Each run must produce:

### Metrics (per run)
- ID test error  
- OOD test error  
- Mean trajectory error  
- Percentile errors (e.g. p90, p95)  
- Worst-case trajectory error  
- Final training loss  

### Convergence metrics
- Loss vs epoch (logged)
- Loss vs walltime (logged)
- Improvement after each stage
- Stage-wise performance contribution

### Cost metrics
- Total training walltime  
- Walltime per stage  
- Number of epochs  
- Memory usage  

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
    - total epochs  
    - total walltime  
    - stage-wise allocation  

---

## Metrics

### Performance
- Mean trajectory error (e.g. RMSE / MSE)  
- Percentile errors (p90, p95)  
- Worst-case trajectory error  
- Final training loss  

---

### Generalization
- ID test error  
- OOD test error  
- ID–OOD gap  

---

### Convergence Behaviour
- Loss vs epochs  
- Loss vs walltime  
- Improvement after each stage  
- Stage-wise contribution to final performance  

---

### Computational Cost
- Total training walltime  
- Walltime per stage  
- Number of epochs  
- Memory usage  

---

### Stability / Robustness
- Variance across seeds  
- Convergence failures  
- Sensitivity to initialization  

---

## Run Matrix

Each experiment run is defined by: training_strategy × num_stages × seed
All combinations must be runnable.

---

## Key Comparisons

- Single-stage vs multi-stage training  
- Performance vs number of stages  
- Accuracy vs computational cost trade-off  
- ID vs OOD generalization for stage-wise training  

---

## Required Outputs / Plots

The pipeline must enable generation of:

- Final error comparison: single-stage vs multi-stage  
- Error vs walltime  
- Loss vs epochs / walltime  
- Performance vs number of stages  
- Stage-wise improvement curves  
- Example trajectory reconstructions showing contribution of later stages  

---

## Success Criteria (for Refactor)

The codebase supports this experiment if:

- Training strategies (single vs multi-stage) are modular and interchangeable  
- Number of stages is configurable without code changes  
- Stage-wise training can be executed sequentially within one experiment run  
- All runs produce consistent, structured outputs  
- Stage-wise metrics are logged uniformly  
- Results can be aggregated across seeds automatically  
- No duplication of training logic is required  

---

## Notes / Assumptions

- All methods are compared on the same dataset and model setting  
- Training configuration is fixed to isolate the effect of stage-wise learning  
- Total computational cost must be tracked explicitly  
- Comparison fairness should account for differences in total training budget and effective model complexity  