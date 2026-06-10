# HPC Launchers

This folder contains optional LSF wrappers for running the thesis experiment
entrypoints on a cluster.

The canonical local entrypoints are the Python modules under
`src/experiments/pipeline/`. The scripts in this folder only translate cluster
environment variables into those Python commands and add LSF resource
directives.

Submit wrappers from the repository root:

```bash
bsub -env "MODE=screening,DRY_RUN=true" < hpc/run_optimizer_comparison.lsf.sh
```

Use `bsub -env` to pass environment variables to the batch job. Inline
`VAR=value bsub < script` does not reliably propagate variables on the target
cluster. For comma-separated values, prefer exporting variables first and
submitting with `bsub -env "all"`:

```bash
(export MODE=screening STRATEGIES=adam,lbfgs SEED_LABELS=s01 && \
  bsub -env "all" < hpc/run_optimizer_comparison.lsf.sh)
```

Wrappers create LSF log directories under `hpc/logs/` at runtime. The log files
are generated cluster artifacts and are not part of the curated thesis results.

Shared wrapper defaults and virtual-environment activation live in
`hpc/common/lsf_defaults.sh`.
