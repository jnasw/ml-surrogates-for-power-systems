"""Domain abstractions and generators for train-time collocation points."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc
import torch


@dataclass(frozen=True)
class CollocationDomain:
    """Continuous sampling domain for collocation points.

    Phase 1 only introduces the abstraction. Concrete generators for this domain
    will be added in later phases once adaptive train-time sampling is enabled.
    """

    time_min: float
    time_max: float
    input_dim: int
    feature_bounds: torch.Tensor | None = None

    def input_bounds(self) -> torch.Tensor:
        """Return box bounds for the full model input `(t, features...)`."""
        lower = torch.full((self.input_dim,), float(self.time_min), dtype=torch.float32)
        upper = torch.full((self.input_dim,), float(self.time_max), dtype=torch.float32)
        if self.feature_bounds is not None:
            if self.feature_bounds.ndim != 2 or self.feature_bounds.shape[1] != 2:
                raise ValueError("feature_bounds must have shape (D, 2).")
            if int(self.feature_bounds.shape[0]) != self.input_dim - 1:
                raise ValueError(
                    "feature_bounds row count must match input_dim - 1. "
                    f"Got {self.feature_bounds.shape[0]} rows for input_dim={self.input_dim}."
                )
            lower[1:] = self.feature_bounds[:, 0]
            upper[1:] = self.feature_bounds[:, 1]
        return torch.stack((lower, upper), dim=1)


def sample_collocation_points(
    *,
    domain: CollocationDomain,
    n: int,
    method: str,
    seed: int | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Sample collocation points uniformly inside a box domain."""
    if n <= 0:
        raise ValueError("n must be positive.")

    bounds = domain.input_bounds().cpu().numpy()
    lower = bounds[:, 0]
    upper = bounds[:, 1]
    variable_mask = upper > lower
    d_var = int(variable_mask.sum())

    out = np.zeros((n, domain.input_dim), dtype=np.float32)
    if d_var > 0:
        method_name = str(method).strip().lower()
        if method_name == "sobol":
            sampler = qmc.Sobol(d=d_var, scramble=True, seed=seed)
            m = int(np.ceil(np.log2(n)))
            unit = sampler.random_base2(m=m)[:n]
        elif method_name == "lhs":
            sampler = qmc.LatinHypercube(d=d_var, seed=seed)
            unit = sampler.random(n=n)
        elif method_name == "random":
            rng = np.random.default_rng(seed)
            unit = rng.uniform(0.0, 1.0, size=(n, d_var))
        else:
            raise ValueError(f"Unknown collocation sampler: {method}")

        scaled = qmc.scale(unit, l_bounds=lower[variable_mask], u_bounds=upper[variable_mask])
        out[:, variable_mask] = scaled.astype(np.float32)

    if np.any(~variable_mask):
        out[:, ~variable_mask] = lower[~variable_mask]

    return torch.as_tensor(out, dtype=dtype, device=device)
