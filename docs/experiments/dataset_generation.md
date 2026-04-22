# Experiment: Dataset Generation Strategy

## Objective
Evaluate how different trajectory sampling strategies affect surrogate model accuracy, generalization capability, and sample efficiency under a fixed simulation budget.

This experiment addresses:
- How training data should be sampled in IC/state space
- Whether static sampling saturates with increasing dataset size
- Whether adaptive methods identify more informative (difficult) trajectories
- The trade-off between simulator cost and generalization performance

---

## Research Questions
- Does static LHS sampling saturate with increasing dataset size?
- Which sampling strategy achieves the lowest error for a fixed simulator budget?
- Are adaptive methods particularly beneficial in low-data regimes?
- Do adaptive methods improve OOD generalization or mainly ID performance?
- Do QBC and marker-based methods select fundamentally different types of trajectories?
- Does the hybrid approach improve both diversity and difficulty coverage?

---

## Independent Variables
- **Sampling strategy**
  - static_lhs (baseline)
  - qbc
  - marker
  - hybrid_qbc_marker

- **Training dataset size (simulator budget)**
  - defined as number of trajectories: `N_trajectories`

- **Random seed**
  - multiple seeds per configuration (e.g. 3–5)

---

## Controlled Variables
- Surrogate model architecture
- Training procedure (optimizer, epochs, stopping criteria)
- Loss formulation
- Data preprocessing
- Evaluation datasets (must be fixed across all runs)

---

## Required Inputs

The pipeline must support:

- Configurable sampling strategy
- Configurable trajectory budget (`N_trajectories`)
- Configurable random seed
- Fixed training configuration
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

### Cost metrics
- Number of simulator calls
- Dataset generation time
- Training time

### Sampling diagnostics
- Sampled ICs (saved)
- Sampling scores (e.g. QBC uncertainty, marker score if applicable)
- Metadata per trajectory (e.g. difficulty indicators if available)

### Aggregation (across seeds)
- Mean and standard deviation of all metrics

---

## Evaluation Setup

- **Training data**
  - Generated using specified sampling strategy and budget

- **Test datasets**
  - ID: same IC distribution as training
  - OOD: extended or shifted IC distribution

---

## Metrics

### Performance
- Mean trajectory error (e.g. RMSE / MSE)
- Percentile errors (p90, p95)
- Worst-case trajectory error

### Generalization
- ID test error
- OOD test error
- ID–OOD gap

### Computational Cost
- Number of simulator calls
- Dataset generation time
- Training time

### Coverage / Sampling Behaviour
- Distribution of sampled initial conditions
- Diversity of sampled trajectories
- Difficulty indicators (if available)

---

## Run Matrix

Each experiment run is defined by: sampling_strategy × N_trajectories × seed

All combinations must be runnable.

---

## Key Comparisons

- Error vs. simulator budget (per sampling strategy)
- Sample efficiency (error vs. simulator calls)
- ID vs. OOD generalization
- Coverage differences in IC space between strategies

---

## Required Outputs / Plots

The pipeline must enable generation of:

- Generalization error vs. simulator budget
- OOD error vs. simulator budget
- Sample efficiency curves
- IC-space coverage visualizations
- Distribution of selected trajectory difficulty
- Example trajectories per strategy

---

## Success Criteria (for Refactor)

The codebase supports this experiment if:

- Sampling strategies are modular and interchangeable
- Dataset size and seeds are configurable without code changes
- All runs produce consistent, structured outputs
- Results can be aggregated across seeds automatically
- Evaluation uses fixed and reusable test datasets
- No duplication of training or evaluation logic is required

---

## Notes / Assumptions

- All methods must use equal simulator budgets
- Adaptive selection overhead must be tracked separately
- Training configuration is fixed to isolate dataset effects