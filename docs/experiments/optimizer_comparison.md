# Experiment: Optimizer Comparison

## Objective
Evaluate how different optimization strategies affect convergence behaviour, computational efficiency, and final surrogate model accuracy.

This experiment addresses:
- Whether 2nd order methods improve convergence compared to 1st order methods  
- The trade-off between convergence speed, computational cost, and final accuracy  
- Whether stochastic 2nd order methods provide a practical alternative to full-batch methods  
- Whether combining optimizers in multi-phase training improves performance  

---

## Research Questions
- Do 2nd order methods outperform Adam in convergence speed and/or final accuracy?  
- Which optimizer provides the best performance for a given computational budget?  
- Are stochastic 2nd order methods viable alternatives to full-batch methods?  
- Does multi-phase training improve convergence or final performance?  
- Is there a clear trade-off between computational cost (time/memory) and accuracy?  
- Which optimizer is most robust across different random seeds?  

---

## Independent Variables

- **Optimizer type**
  - adam (baseline)

  - 2nd order (full-batch):
    - bfgs
    - lbfgs
    - ssbfgs / ssbroyden

  - 2nd order (stochastic / mini-batch):
    - soap
    - sssbfgs
    - sssbroyden  

- **Training strategy**
  - single_optimizer (baseline)
  - multi_phase (e.g. adam → lbfgs)

- **Random seed**
  - multiple seeds per configuration (e.g. 3–5)

---

## Controlled Variables
- Dataset (fixed dataset generation strategy and size)  
- Surrogate model architecture  
- Loss formulation  
- Collocation strategy (if PINN)  
- Data preprocessing  
- Evaluation datasets (must be fixed across all runs)  

---

## Required Inputs

The pipeline must support:

- Configurable optimizer (including 1st and 2nd order methods)
- Configurable training strategy (single vs multi-phase)
- Configurable random seed
- Fixed dataset input
- Fixed evaluation datasets:
  - ID dataset
  - OOD dataset

- Optional:
  - Configurable phase schedule (for multi-phase runs)

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
- Time to convergence (if threshold defined)

### Cost metrics
- Total training walltime  
- Number of epochs  
- (Optional) number of gradient evaluations  
- Memory usage  

### Stability metrics
- Convergence success/failure flag  
- Variance across seeds  

### Aggregation (across seeds)
- Mean and standard deviation of all metrics  

---

## Evaluation Setup

- **Training data**
  - Fixed dataset used across all optimizer configurations  

- **Test datasets**
  - ID: same IC distribution as training  
  - OOD: extended or shifted IC distribution  

- **Training budget**
  - Must be tracked consistently across runs:
    - epochs  
    - walltime  
    - (optionally) gradient evaluations  

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
- Convergence speed (epochs and time)  

---

### Computational Cost
- Total training walltime  
- Number of epochs  
- (Optional) gradient evaluations  
- Memory usage  

---

### Stability / Robustness
- Variance across seeds  
- Convergence failures  
- Sensitivity to initialization  

---

## Run Matrix

Each experiment run is defined by: optimizer × training_strategy × seed
All combinations must be runnable.

---

## Key Comparisons

- 1st order vs 2nd order optimization  
- Full-batch vs stochastic 2nd order methods  
- Single optimizer vs multi-phase training  
- Accuracy vs computational cost trade-off  

---

## Required Outputs / Plots

The pipeline must enable generation of:

- Loss vs epochs (per optimizer)  
- Loss vs walltime  
- Test error vs walltime  
- Final performance vs computational cost  
- Memory usage vs performance  
- Convergence curves for multi-phase training  
- Error bars across seeds  

---

## Success Criteria (for Refactor)

The codebase supports this experiment if:

- Optimizers are modular and interchangeable  
- Training strategy (single vs multi-phase) is configurable without code changes  
- All runs produce consistent, structured outputs  
- Convergence metrics are logged uniformly across optimizers  
- Results can be aggregated across seeds automatically  
- No duplication of training loops is required  

---

## Notes / Assumptions

- All methods are compared on the same dataset and model  
- Training configuration is fixed to isolate optimizer effects  
- Walltime is the primary comparison metric for efficiency  
- Multi-phase training must use clearly defined phase transitions  
- Memory limitations of 2nd order methods must be tracked explicitly  