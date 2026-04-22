# Experiment: Collocation Point Sampling

## Objective
Evaluate how different collocation point sampling strategies affect PINN training accuracy, convergence behaviour, and computational efficiency.

This experiment addresses:
- Whether adaptive collocation improves performance compared to static uniform/LHS sampling  
- How collocation point placement influences physics enforcement and surrogate generalization  
- How surrogate performance scales with collocation point density  
- Whether adaptive methods achieve the same accuracy with lower effective density  
- The trade-off between improved residual targeting and additional sampling overhead  
- Whether residuals on unseen collocation points can serve as a validation signal  

---

## Research Questions
- Do adaptive collocation strategies improve final accuracy compared to static sampling?  
- Which collocation method provides the best performance for a fixed residual budget?  
- How does surrogate performance scale with collocation point density?  
- At what collocation density do performance gains begin to saturate?  
- Do adaptive methods achieve the same performance with fewer collocation points than static methods?  
- Do adaptive methods improve OOD generalization or mainly reduce local residual errors?  
- Do adaptive methods discover genuinely informative residual regions, or collapse to narrow high-residual areas?  
- Can residuals evaluated on unseen collocation points reliably indicate generalization error?  
- Does multi-pool allocation improve the effectiveness of collocation point usage?  

---

## Independent Variables

- **Collocation sampling strategy**
  - uniform_lhs (baseline)
  - static_random
  - rad
  - rar_d
  - rar_g
  - (optional) additional implemented variants

- **Collocation budget / density**
  - defined as:
    - number of collocation points
    - or total residual evaluations
    - optionally normalized (e.g. per trajectory or time interval)

- **Budget schedule**
  - fixed collocation budget
  - increasing / adaptive budget over training

- **Pool allocation strategy**
  - single_pool (baseline)
  - multi_pool_allocation

- **Random seed**
  - multiple seeds per configuration (e.g. 3–5)

---

## Controlled Variables
- Dataset (fixed dataset generation strategy and size)  
- Surrogate model architecture  
- Optimizer  
- Loss formulation / weighting scheme  
- Data preprocessing  
- Evaluation datasets (must be fixed across all runs)  

---

## Required Inputs

The pipeline must support:

- Configurable collocation sampling strategy  
- Configurable collocation budget / density  
- Configurable budget schedule (fixed vs increasing)  
- Configurable pool allocation strategy (single vs multi-pool)  
- Configurable random seed  

- Fixed dataset input  
- Fixed evaluation datasets:
  - ID dataset
  - OOD dataset

- Optional:
  - Configurable candidate pool size for adaptive methods  

---

## Required Outputs

Each run must produce:

### Metrics (per run)
- ID test error  
- OOD test error  
- Mean trajectory error  
- Percentile errors (e.g. p90, p95)  
- Worst-case trajectory error  

### Residual / physics metrics
- Training residual loss over time  
- Residual on unseen collocation points  
- Residual distribution statistics  
- Correlation between residual metrics and generalization error  

### Convergence metrics
- Loss vs epoch (logged)
- Loss vs walltime (logged)

### Cost metrics
- Total training walltime  
- Collocation update overhead  
- Number of collocation / residual evaluations  

### Sampling diagnostics
- Locations of collocation points (saved)  
- Distribution of collocation points in time/state space  
- Acquisition scores (for adaptive methods)  
- Pool allocation statistics (if multi-pool is used)  

### Stability metrics
- Convergence success/failure flag  
- Variance across seeds  

### Aggregation (across seeds)
- Mean and standard deviation of all metrics  

---

## Evaluation Setup

- **Training data**
  - Fixed supervised dataset across all configurations  

- **Collocation points**
  - Sampled using the specified strategy, budget, and schedule  

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

### Physics / Residual Behaviour
- Training residual loss vs time  
- Residual on unseen collocation points  
- Residual distribution statistics  
- Correlation between residual and true error  

---

### Convergence Behaviour
- Loss vs epochs  
- Loss vs walltime  

---

### Computational Cost
- Total training walltime  
- Collocation update overhead  
- Number of residual evaluations  

---

### Coverage / Sampling Behaviour
- Distribution of collocation points in time/state space  
- Diversity of sampled points  
- Concentration in difficult regions  
- Pool allocation distribution (if applicable)  

---

### Stability / Robustness
- Variance across seeds  
- Convergence failures  

---

## Run Matrix

Each experiment run is defined by: sampling_strategy × collocation_budget × budget_schedule × pool_strategy × seed
All combinations must be runnable (with feasible subsets if necessary).

---

## Key Comparisons

- Static vs adaptive collocation strategies  
- Performance scaling with collocation density  
- Static vs adaptive efficiency at equal budgets  
- Fixed vs increasing collocation budgets  
- Single-pool vs multi-pool allocation  
- Residual-based validation vs true generalization error  

---

## Required Outputs / Plots

The pipeline must enable generation of:

- ID / OOD error vs collocation budget  
- Performance scaling with collocation density (per strategy)  
- Comparison of static vs adaptive efficiency  
- Residual loss vs walltime  
- Residual vs generalization error correlation plots  
- Distribution of collocation points in time/state space  
- Acquisition score distributions  
- Example collocation selections for different strategies  

---

## Success Criteria (for Refactor)

The codebase supports this experiment if:

- Collocation strategies are modular and interchangeable  
- Collocation budget/density is configurable without code changes  
- Adaptive and static strategies share a common interface  
- Residual evaluation and logging is consistent across methods  
- Sampling diagnostics are saved and accessible  
- Results can be aggregated across seeds automatically  
- No duplication of training or sampling logic is required  

---

## Notes / Assumptions

- All methods are compared on the same dataset and model  
- Total collocation budget must be controlled fairly  
- Candidate evaluation overhead must be tracked explicitly  
- Residual loss must not be interpreted as a direct proxy for true accuracy without comparison  
- Adaptive methods may change local density even at equal global budgets  