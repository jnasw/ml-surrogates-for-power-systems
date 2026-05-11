# Chapter 4 Dynamical System Analysis

This folder contains exploratory analysis for Chapter 4, "Dynamical System Analysis Framework".

The notebook uses the canonical surrogate ODE implementation in `src/sim/ode/model_definitions.py` and canonical configuration files under `src/config/`.

The older `tools/analysis/dysys` folder is historical/reference material and is not used directly here, because it contains local helper code with different network assumptions.

Run the notebook from either the repository root or this folder:

```bash
jupyter notebook tools/analysis/chapter4/chapter4_dynamical_system_analysis.ipynb
```

For now, the notebook only performs a canonical SM4 RHS smoke test.
