# Refactor Plan

## Goal
Reduce the codebase to the minimum structure needed to run the thesis experiments cleanly, reproducibly, and comparably.

The refactor should:
- simplify and restructure the current repository
- preserve current functionality unless changes are explicitly intended
- improve modularity so experiments can be adjusted or extended without major rewrites
- reduce redundancy and script sprawl
- support both local execution and HPC execution

The refactor is not a redesign for its own sake. It should prioritize clarity, reuse, and experiment support over framework complexity.

---

## Experiment Support Requirements

The refactored repository must support the following experiment contracts:

### Core experiments
- `docs/experiments/dataset_generation.md`
- `docs/experiments/optimizer_comparison.md`
- `docs/experiments/multi_stage_training.md`
- `docs/experiments/loss_balancing.md`
- `docs/experiments/collocation_sampling.md`

### Optional / secondary experiments
- `docs/experiments/vrba.md`
- `docs/experiments/curriculum_learning.md`

Optional experiments should be supported structurally, but they must not drive the core architecture.

---

## Non-Negotiables

The refactor must preserve the following unless explicitly approved otherwise:

- float64 throughout the pipeline
- deterministic preprocessing behaviour
- local execution via Python entrypoints
- HPC execution via `lsf.sh` wrappers that call Python scripts
- config-driven experiment behaviour
- current core functionality of implemented methods
- current behaviour of unstable methods unless explicitly being worked on

The following areas require extra caution:
- stochastic second-order methods
- vRBA
- ReLoBRaLo
- LRA

These should be integrated into the new structure without unintentionally changing behaviour.

---

## Current Pain Points

The current repository has grown incrementally through implementation of methods from multiple papers and experiments. As a result, it currently suffers from:

- bloated structure
- duplicated logic
- experiment-specific paths and scripts
- unclear reuse boundaries between methods
- messy output organization
- inconsistent smoke-test coverage across methods
- diagnostics and plots partly living only in notebooks
- optional / advanced methods being harder to integrate cleanly

More specifically, the main structural pain points are:

- `trainer.py` is overloaded and currently acts as the main integration surface for many training features, including single-stage training, multistage training, curriculum gating, vRBA updates, weighting logic, optimizer transitions, scheduler handling, checkpointing, and diagnostics. This makes changes high-risk.
- experiment orchestration is duplicated across several large scripts rather than sharing thin reusable infrastructure
- some features have mismatches between configuration surface, runtime behaviour, and stated support level
- preprocessing currently mixes generic data preparation with experiment-specific metadata generation, increasing coupling
- some experiment-required diagnostics are not yet surfaced consistently in runtime logging
- logging and metric reporting are currently bloated and not fully aligned with experiment contracts
- metric computation and reporting destinations are too tightly coupled
- some core comparison metrics, such as MAE, are missing or inconsistently surfaced
- W&B integration should be treated as a reporting backend rather than the source of experiment logic

The main refactor objective is to reduce this complexity while preserving working functionality.

---

## Target Design Principles

The target structure should follow these principles:

- minimal core pipeline
- lightweight functional modules
- fewer files where possible
- reuse shared definitions and shared logic across experiments
- make experiment dimensions configurable instead of hardcoded
- support optional methods as separate trainer modes
- preserve the ability to run experiments both locally and on HPC
- avoid a single giant canonical experiment runner
- allow one Python file per experiment where useful, but avoid redundant implementations
- separate full run artifacts from summarized results
- separate metric computation from logging/reporting backends
- make local run artifacts the source of truth, with W&B as an optional thin reporting adapter

The target architecture should make it easy to:
- run existing experiments
- tweak experiment dimensions
- compare methods consistently
- add small experiment variations without major rewrites

---

## Outputs and Results Policy

The refactor should standardize the distinction between:

- `outputs/` = full per-run artifacts for reproducibility and debugging
- `results/` = summarized, aggregated, thesis-oriented outputs

### `outputs/` should contain
- saved effective config
- metrics
- timings
- metadata / logs
- optional checkpoints or snapshots when explicitly enabled

### `results/` should contain
- aggregated metrics
- comparison tables
- plot-ready summaries
- selected final result files

Plots and richer diagnostics may remain notebook-driven for now and do not need to be forced into the runtime pipeline during this refactor.

---

## Execution Model

The refactor should preserve and clarify the following execution model:

### Local execution
Python scripts should remain the core execution path.

Current key entrypoints include:
- `00_create_dataset.py`
- `01_preprocess_dataset.py`
- `10_run_baseline`
- `20_run_pinn.py`

### HPC execution
`lsf.sh` scripts should wrap the Python entrypoints rather than implement separate experiment logic.

The refactor may reorganize scripts, but the final structure must make it clear:
- how to run experiments locally
- how to submit them to the HPC cluster
- how experiment scripts relate to the core pipeline

---

## Refactor Strategy

The refactor should happen in phases, starting with low-risk/high-leverage work.

### Phase 1 — Repository Mapping and Invariants
Goals:
- map the current repository against the experiment contracts
- identify shared components, duplicated logic, fragile areas, and experiment-specific forks
- identify candidate merge points without merging silently
- define one very small smoke run
- later define one trusted baseline experiment config as a regression target
- map trainer responsibilities and feature integration points
- build a feature truth table for implemented methods vs config surface vs experiment docs
- identify required preprocessing artifacts for each experiment
- identify missing diagnostics/logging relative to experiment contracts
- identify current metric definitions, missing core metrics, and where metrics are computed vs only reported

Outputs:
- current architecture map
- duplication map
- list of fragile areas
- feature truth table
- experiment-to-artifact dependency map
- logging gap list
- metric gap list
- first smoke-run verification target

---

### Phase 2 — Standardize Run Artifacts and Metrics
Goals:
- define consistent per-run artifact structure
- standardize config snapshots, metrics, timings, and run metadata
- establish the `outputs/` vs `results/` distinction clearly
- define a shared metric schema across experiments
- include missing core metrics such as MAE
- separate metric computation from reporting backends
- make local artifact logging the source of truth
- make W&B a thin optional adapter over standardized metrics and logs
- avoid changing notebook-driven plotting unless necessary

Outputs:
- consistent output schema
- clear artifact expectations for all experiment paths
- shared metric schema
- defined logging/reporting responsibilities

---

### Phase 3 — Stabilize Data and Preprocessing Contracts
Goals:
- clarify separation between dataset generation and preprocessing
- preserve deterministic preprocessing semantics
- preserve numeric precision and metadata integrity
- document and stabilize dataset contracts used by training
- ensure curriculum metadata remains supported cleanly
- make preprocessing dependencies for experiment features explicit
- separate generic preprocessing responsibilities from experiment-specific metadata generation where possible

Outputs:
- stable data/preprocessing interfaces
- documented preprocessing assumptions
- preserved HDF5/data schema expectations where relevant
- explicit mapping of which training features depend on which preprocessing artifacts

---

### Phase 4 — Simplify and Centralize Training Core
Goals:
- identify the common training loop structure
- separate shared trainer logic from experiment-specific behaviour
- reduce duplicated trainer paths
- keep optional methods decoupled from the core path
- separate trainer core from trainer modes, feature hooks, and diagnostics
- reduce the number of responsibilities owned directly by `trainer.py`
- ensure diagnostics expected by experiment specs can be emitted consistently
- separate training logic from logging/reporting integrations

Outputs:
- simpler training core
- cleaner boundaries for optional trainer modes
- reduced duplication in training logic
- clearer diagnostics/logging boundaries

---

### Phase 5 — Modularize Experiment Dimensions
Goals:
- make core experiment dimensions easier to swap and combine:
  - optimizer / optimizer schedule
  - loss formulation / weighting
  - collocation strategy
  - multi-stage training
  - curriculum learning
  - optional vRBA mode
- preserve functionality while simplifying control flow

Outputs:
- modular experiment dimensions
- clearer composition of methods
- reduced hardcoded experiment branches

---

### Phase 6 — Reduce Script Redundancy
Goals:
- keep the ability to have one Python file per experiment where useful
- reduce redundant scripts and duplicated paths
- avoid further growth of ad hoc script sprawl
- preserve local and HPC execution clarity
- extract shared orchestration and bookkeeping logic from large experiment runner scripts
- keep experiment-specific entry scripts thin

Outputs:
- cleaner script organization
- less redundancy
- clearer experiment entry structure
- thinner experiment scripts with shared orchestration underneath

---

### Phase 7 — Cleanup and Deletion
Goals:
- remove obsolete paths
- remove redundant utilities
- simplify config surface where possible
- document any migrations clearly if compatibility changes are introduced

Outputs:
- smaller codebase
- less dead code
- cleaner final structure

---

## Verification Strategy

Verification should be progressive and lightweight.

### Early verification
- first define one very small smoke run
- use it as the default verification step during early phases

### Regression target
- later define one trusted baseline experiment config
- preserve it as a regression target for the rest of the refactor

### Verification rules
- do not claim a refactor phase succeeded without verifying something
- be explicit about what was checked
- if duplicate paths are candidates for merging, identify a verification path before merging

Smoke tests are currently not uniform across methods, so verification should remain explicit and method-aware.

---

## Migration Policy

Backward compatibility with current configs is desirable but not mandatory.

Config or structure migrations are acceptable when:
- they simplify the repository
- they reduce redundancy
- they are clearly documented
- they preserve the ability to run the intended experiments

Do not introduce undocumented breaking changes.

---

## Out of Scope

This refactor should not:
- redesign algorithms unnecessarily
- optimize unstable methods beyond structural integration
- move all notebook-based diagnostics into runtime scripts
- create a large generic framework for hypothetical future needs
- force all experiments into one giant canonical runner

The focus is a minimal, practical, thesis-oriented experimental codebase.

---

## Immediate Next Steps

1. Map the current repository against the experiment contracts.
2. Identify shared components, duplicated logic, fragile areas, overloaded integration surfaces, and current logging/metric bottlenecks.
3. Build a feature truth table for implemented methods, config surface, logging support, preprocessing dependencies, and reporting support.
4. Define one very small smoke run.
5. Define the initial run artifact structure (`outputs/` vs `results/`).
6. Define the shared metric schema, including MAE.
7. Propose the first low-risk code changes before editing core logic.