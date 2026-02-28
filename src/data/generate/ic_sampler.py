"""Initial-condition sampling utilities for active learning."""

from __future__ import annotations

import numpy as np
from scipy.stats import qmc


def sample_initial_ics(
    method: str,
    n: int,
    bounds: np.ndarray,
    seed: int | None = None,
) -> np.ndarray:
    """Sample IC vectors in the provided bounds.

    Args:
        method: One of ``sobol``, ``lhs``, or ``random``.
        n: Number of samples.
        bounds: Array of shape (D, 2), [min, max] per dimension.
        seed: Optional random seed.
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    bounds = np.asarray(bounds, dtype=np.float32)
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError("bounds must have shape (D, 2).")

    lower = bounds[:, 0]
    upper = bounds[:, 1]
    if np.any(upper < lower):
        raise ValueError("Each bound must satisfy max >= min.")

    variable_mask = upper > lower
    d_var = int(variable_mask.sum())
    out = np.zeros((n, bounds.shape[0]), dtype=np.float32)

    if d_var > 0:
        method = method.lower()
        if method == "sobol":
            sampler = qmc.Sobol(d=d_var, scramble=True, seed=seed)
            m = int(np.ceil(np.log2(n)))
            unit = sampler.random_base2(m=m)[:n]
        elif method == "lhs":
            sampler = qmc.LatinHypercube(d=d_var, seed=seed)
            unit = sampler.random(n=n)
        elif method == "random":
            rng = np.random.default_rng(seed)
            unit = rng.uniform(0.0, 1.0, size=(n, d_var))
        else:
            raise ValueError(f"Unknown sampling method: {method}")

        scaled = qmc.scale(unit, l_bounds=lower[variable_mask], u_bounds=upper[variable_mask])
        out[:, variable_mask] = scaled.astype(np.float32)

    if np.any(~variable_mask):
        out[:, ~variable_mask] = lower[~variable_mask]

    return out
