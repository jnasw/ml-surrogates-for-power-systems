"""Marker feature extraction and marker-space selection utilities."""

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
    """Return marker matrix and marker names."""
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

    columns.append(_trapz_compat(x_norm_sq, t=t, axis=1))
    names.append("traj_energy_l2")

    columns.append(_trapz_compat(dx_norm_sq, t=t, axis=1))
    names.append("derivative_energy_l2")

    columns.append(np.max(dx_norm, axis=1))
    names.append("max_derivative_mag")

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


def standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def fit_pca(x_std: np.ndarray, explained_var_ratio: float) -> tuple[np.ndarray, np.ndarray, int]:
    x_centered = x_std - x_std.mean(axis=0, keepdims=True)
    _, svals, vt = np.linalg.svd(x_centered, full_matrices=False)
    var = (svals**2) / max(1, x_centered.shape[0] - 1)
    total = float(np.sum(var))
    if total <= 0.0:
        n_comp = 1
    else:
        cum = np.cumsum(var / total)
        n_comp = int(np.searchsorted(cum, float(explained_var_ratio)) + 1)
        n_comp = max(1, min(n_comp, vt.shape[0]))
    components = vt[:n_comp].T
    return x_centered.mean(axis=0), components, n_comp


def pca_transform(x_std: np.ndarray, center: np.ndarray, components: np.ndarray) -> np.ndarray:
    return (x_std - center) @ components


def normalize01(x: np.ndarray) -> np.ndarray:
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if x_max - x_min < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - x_min) / (x_max - x_min)).astype(np.float32)


def greedy_select(
    *,
    embedding: np.ndarray,
    scores: np.ndarray,
    k: int,
    preselect_factor: int,
    alpha_score: float,
) -> np.ndarray:
    n = embedding.shape[0]
    if k <= 0 or k > n:
        raise ValueError("k must be in [1, n_candidates].")
    pre_n = min(n, max(k, int(k * preselect_factor)))
    pool_idx = np.argsort(scores, kind="mergesort")[-pre_n:][::-1]
    pool = embedding[pool_idx]
    pool_scores = scores[pool_idx]

    selected_local = [int(np.argmax(pool_scores))]
    while len(selected_local) < k:
        remain = [i for i in range(pre_n) if i not in selected_local]
        sel_pts = pool[selected_local]
        best_i = None
        best_val = -np.inf
        for i in remain:
            d = np.linalg.norm(pool[i][None, :] - sel_pts, axis=1)
            min_d = float(np.min(d))
            value = float(alpha_score) * float(pool_scores[i]) + (1.0 - float(alpha_score)) * min_d
            if value > best_val:
                best_val = value
                best_i = i
        selected_local.append(int(best_i))
    return pool_idx[np.array(selected_local, dtype=int)]
