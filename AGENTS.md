# AGENTS.md

## Repository Purpose
This repository supports a master’s thesis on surrogate modeling of power system dynamics using machine learning. The primary goal of the codebase is to enable clean, reproducible, and comparable experiments across data-driven and physics-informed methods.

The code should optimize for:
- controlled experiment execution
- reproducibility
- modular comparison of methodologies
- minimal and understandable structure

Do not optimize for framework complexity or speculative extensibility.

---

## Core Working Principles
- Prefer deletion over new abstraction.
- Prefer lightweight functional modules over heavy class hierarchies.
- Prefer fewer files when possible.
- Preserve numerical behavior unless a change is explicitly intended and documented.
- Preserve float64 throughout the pipeline unless explicitly approved otherwise.
- Avoid duplicate training, evaluation, and logging paths.
- Keep experiment behavior config-driven whenever possible.
- Reuse existing definitions and components wherever possible.
- Optional methods must remain decoupled from the core pipeline.
- Do not introduce new infrastructure unless it clearly supports one or more defined experiment contracts.

---

## Source of Truth
Before making significant changes, read:
- docs/experiments/*.md
- docs/refactor/plan.md
- docs/refactor/requirements_map.md

These documents define what the repository must support.  
If implementation details conflict with an experiment contract, surface the conflict explicitly.

---

## Execution Model
Experiments must be runnable in both of the following ways:
- locally via direct Python entrypoints
- on the HPC cluster via lsf.sh wrapper scripts that call the Python entrypoints

Prefer Python entrypoints as the core execution path.  
HPC scripts should wrap existing Python scripts rather than introduce separate logic.

Current core entrypoints include:
- 00_create_dataset.py for dataset generation
- 01_preprocess_dataset.py for preprocessing
- 10_run_baseline.py for the data-driven NN baseline
- 20_run_pinn.py for PINN training

There are also experiment-specific scripts in tools/pinn/, but avoid expanding ad hoc script sprawl further.

---

## Refactor Expectations
- Do not perform broad rewrites without a phased plan.
- Start by mapping the current implementation against experiment requirements before editing.
- Refactor in small, verifiable slices.
- Do not mix multiple refactor concerns in one change (e.g. restructuring + logic changes).
- Keep at least one trusted baseline experiment path runnable throughout the refactor.
- Standardize outputs, logging, and config snapshots before introducing deeper abstractions.
- Prefer shared training/evaluation paths with modular hooks over experiment-specific forks.
- Optional or unstable methods should be refactored to fit the minimal architecture, but their functionality should not be changed unless explicitly requested.

---

## Handling Duplication and Fragile Areas
When duplicate logic is found:
- identify the overlapping paths
- map how each path is used
- summarize whether they appear equivalent or meaningfully different
- do not merge them silently
- surface the merge candidate explicitly before consolidating

If configuration, runtime behavior, and documentation disagree:
- identify and surface the mismatch before making changes

Be especially careful around methods that are currently less stable or more experimental, including:
- stochastic second-order methods
- vRBA
- ReLoBRaLo
- LRA

These should be structurally integrated into the new architecture, but not behaviorally changed without explicit intent.

---

## Experiment Design Expectations
- Every experiment should be runnable from configuration without code edits.
- Shared components should be reused across experiments.
- Swappable methods should be implemented behind clear functional interfaces or modular hooks.
- Optional features should be implemented as separate trainer modes rather than deeply entangled logic.
- All runs must produce consistent, structured outputs.
- Save the effective config for every run.
- Log metrics and diagnostics needed by the corresponding experiment specification.

---

## Reproducibility Expectations
- Use fixed seeds where experiments require them.
- Preserve deterministic preprocessing behavior.
- Save the effective config used for each run.
- Do not silently change preprocessing semantics, splits, or numeric precision.

---

## Outputs and Results
Distinguish clearly between:
- outputs/ = full run artifacts for reproducibility and debugging
- results/ = summarized artifacts for comparison, aggregation, and thesis-ready reporting

### outputs/
Use for per-run artifacts such as:
- saved config
- metrics
- timings
- logs / metadata
- checkpoints or snapshots when explicitly enabled

### results/
Use only for summarized or aggregated outputs such as:
- comparison tables
- aggregated metrics
- plot-ready summaries
- selected final result files

Do not treat results/ as the default dump location for normal run artifacts.

---

## Checkpointing / Snapshots
- Checkpoints or snapshots should only be saved when explicitly configured.
- Do not make checkpoint saving mandatory for every run.

---

## Verification Expectations
- After making changes, run the smallest relevant verification step available.
- Prefer a very small smoke run as the default verification step.
- Do not claim success without checking outputs, logs, or tests.
- If no verification was run, state that explicitly.
- Prefer smoke tests and trusted baseline runs before broader experiment execution.

Note: existing smoke tests are not yet uniform across methods, so be explicit about what was actually verified.

---

## Scope Control
- Do not let optional methods drive the architecture.
- Support optional experiments such as vRBA and curriculum learning through extension points or separate trainer modes.
- Keep the baseline path simple and robust.

---

## Coding Style and Documentation

### General Principles
- Prefer clarity over cleverness.
- Prefer simple, explicit code over abstraction-heavy designs.
- Keep functions small and focused on a single responsibility.
- Avoid hidden side effects and implicit state changes.

---

### Structure and Modularity
- Reuse existing functions/modules instead of duplicating logic.
- Do not introduce new abstraction layers unless they clearly reduce duplication or complexity.
- Keep optional features (e.g. vRBA, curriculum) isolated from the core training logic.
- Prefer lightweight functional modules over deep class hierarchies.

---

### Naming
- Use descriptive, explicit names (avoid abbreviations unless standard).
- Be consistent with naming across modules (e.g. collocation, dataset, optimizer).
- Config keys, variable names, and logging names should align where possible.

---

### Comments and Documentation
- Write comments that explain why, not just what.
- Avoid obvious comments that restate the code.
- Add short docstrings to:
  - non-trivial functions
  - core modules (trainer, preprocessing, sampling)
- When implementing experiment logic, reference the corresponding experiment concept where helpful.

Example:
    # Implements curriculum gating: only samples from bins <= active_max_bin

---

### Logging and Metrics
- Do not compute metrics inside logging code.
- Keep metric computation and reporting separate.
- Use consistent metric names across the codebase.
- Ensure core metrics required by experiment specs (e.g. MAE, RMSE, percentiles) are computed and logged consistently.
- Treat W&B as an optional reporting backend; all runs must remain valid without it.

---

### Config Usage
- Do not hardcode experiment parameters.
- All experiment-relevant behavior must be configurable.
- Avoid hidden fallback logic unless explicitly documented.

---

### Refactor Safety
- Do not change numerical behavior unintentionally.
- If logic is unclear or duplicated, map it first before modifying it.
- Prefer incremental changes over large rewrites.

---

### File Organization
- Prefer fewer files, but avoid overly large “god files”.
- Keep related functionality together.
- Avoid scattering related logic across multiple modules.

---

### When in Doubt
- Choose the simplest solution that satisfies the experiment requirements.
- If multiple implementations exist, identify and compare them before merging.

---

## Documentation Expectations
- When changing behavior, update the relevant experiment or refactor documentation if needed.
- If the same confusion or mistake appears repeatedly, update this file to prevent it from recurring.