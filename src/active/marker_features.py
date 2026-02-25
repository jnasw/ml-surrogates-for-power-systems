"""Compact trajectory marker extraction for smooth-dynamics sampling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _trapz_compat(y: np.ndarray, t: np.ndarray, axis: int) -> np.ndarray:
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x=t, axis=axis)
    return np.trapz(y, x=t, axis=axis)


@dataclass(frozen=True)
class MarkerFeatureConfig:
    """Configuration for marker extraction."""

    settling_fraction: float = 0.05
    include_anchor_state_markers: bool = True
    anchor_state_names: tuple[str, ...] = ("theta", "omega")


def _statewise_settling_time(x: np.ndarray, t: np.ndarray, frac: float) -> np.ndarray:
    """Approximate settling time per state using a suffix-in-threshold criterion."""
    final = x[:, -1:, :]
    amp = np.max(x, axis=1) - np.min(x, axis=1)
    threshold = np.maximum(frac * amp[:, None, :], 1e-8)
    err = np.abs(x - final)
    within = err <= threshold

    n, _, s = within.shape
    out = np.empty((n, s), dtype=np.float32)
    for i in range(n):
        for j in range(s):
            ok = within[i, :, j]
            suffix_ok = np.flip(np.cumprod(np.flip(ok.astype(np.int32)))) > 0
            idx = np.where(suffix_ok)[0]
            out[i, j] = t[idx[0]] if idx.size > 0 else t[-1]
    return out


def compute_marker_matrix(
    trajs: np.ndarray,
    time_grid: np.ndarray,
    state_names: list[str] | None = None,
    cfg: MarkerFeatureConfig | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Return marker matrix and marker names.

    Marker families:
    - Global amplitude/activity markers (order-agnostic).
    - Optional anchor-state markers for interpretable electromechanical dynamics.
    """
    cfg = MarkerFeatureConfig() if cfg is None else cfg
    x = np.asarray(trajs, dtype=np.float32)
    t = np.asarray(time_grid, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"trajs must be 3D (N,T,S), got {x.shape}")
    if t.ndim != 1:
        raise ValueError(f"time_grid must be 1D, got {t.shape}")
    if x.shape[1] != t.shape[0]:
        raise ValueError("Trajectory time dimension must match time_grid length.")

    n, _, s = x.shape
    if state_names is None:
        state_names = [f"x{i}" for i in range(s)]
    if len(state_names) != s:
        raise ValueError("state_names length must match trajectory state dimension.")

    dxdt = np.gradient(x, t, axis=1)
    x_norm_sq = np.sum(x**2, axis=2)
    dx_norm_sq = np.sum(dxdt**2, axis=2)
    dx_norm = np.linalg.norm(dxdt, axis=2)

    columns: list[np.ndarray] = []
    names: list[str] = []

    # Global markers.
    columns.append(_trapz_compat(x_norm_sq, t=t, axis=1))
    names.append("traj_energy_l2")

    columns.append(_trapz_compat(dx_norm_sq, t=t, axis=1))
    names.append("derivative_energy_l2")

    columns.append(np.max(dx_norm, axis=1))
    names.append("max_derivative_mag")

    # Aggregate per-state descriptors for model-order scalability.
    state_range = np.max(x, axis=1) - np.min(x, axis=1)
    state_tv = np.sum(np.abs(np.diff(x, axis=1)), axis=1)
    settling = _statewise_settling_time(x, t, frac=float(cfg.settling_fraction))
    final = x[:, -1:, :]
    iae = _trapz_compat(np.abs(x - final), t=t, axis=1)

    for arr, base in (
        (state_range, "range"),
        (state_tv, "total_variation"),
        (settling, "settling_time"),
        (iae, "iae"),
    ):
        columns.append(np.mean(arr, axis=1))
        names.append(f"{base}__state_mean")
        columns.append(np.max(arr, axis=1))
        names.append(f"{base}__state_max")

    if cfg.include_anchor_state_markers:
        index_by_name = {name: idx for idx, name in enumerate(state_names)}
        for anchor in cfg.anchor_state_names:
            if anchor not in index_by_name:
                continue
            j = index_by_name[anchor]
            columns.append(state_range[:, j])
            names.append(f"range__{anchor}")
            columns.append(settling[:, j])
            names.append(f"settling_time__{anchor}")
            columns.append(iae[:, j])
            names.append(f"iae__{anchor}")
            columns.append(state_tv[:, j])
            names.append(f"total_variation__{anchor}")

    matrix = np.column_stack(columns).astype(np.float32)
    if matrix.shape != (n, len(names)):
        raise ValueError("Internal marker matrix shape mismatch.")
    return matrix, names
