#!/usr/bin/env python3
"""Shared matrix-building utilities for the workflow HPO pipeline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def to_dict(cfg: Any, default: dict[str, Any] | None = None) -> dict[str, Any]:
    out = OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_config(cfg) else cfg
    if out is None:
        return {} if default is None else dict(default)
    if not isinstance(out, dict):
        raise ValueError(f"Expected mapping, got: {type(out)}")
    return out


def to_list(cfg: Any) -> list[Any]:
    out = OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_config(cfg) else cfg
    if out is None:
        return []
    if not isinstance(out, list):
        raise ValueError(f"Expected list, got: {type(out)}")
    return out


def parse_override(override: str) -> tuple[str, str]:
    if "=" not in override:
        raise ValueError(f"Invalid override '{override}'. Expected key=value.")
    k, v = override.split("=", 1)
    key = k.strip()
    val = v.strip()
    if not key:
        raise ValueError(f"Invalid override '{override}'. Empty key.")
    return key, val


def parse_overrides(overrides: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for ov in overrides:
        k, v = parse_override(str(ov))
        out[k] = v
    return out


def format_override_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def merge_stage_overrides(
    fixed_list: list[str],
    axis_keys: list[str],
    axis_values: tuple[Any, ...],
) -> dict[str, str]:
    merged = parse_overrides(fixed_list)
    for key, value in zip(axis_keys, axis_values):
        merged[str(key)] = format_override_value(value)
    return merged


def derive_stage1_complement(cfg: dict[str, Any], stage1: dict[str, str]) -> None:
    derive_cfg = cfg.get("matrix", {}).get("derive", {}).get("stage1", {})
    if not derive_cfg:
        return
    enabled = bool(derive_cfg.get("uncertainty_distance_complement", False))
    if not enabled:
        return
    uncertainty_key = str(derive_cfg.get("uncertainty_key", "active.diversity.uncertainty_weight"))
    distance_key = str(derive_cfg.get("distance_key", "active.diversity.distance_weight"))
    if uncertainty_key not in stage1 or distance_key in stage1:
        return
    u = float(stage1[uncertainty_key])
    d = 1.0 - u
    stage1[distance_key] = f"{d:.6f}".rstrip("0").rstrip(".")


def load_budget_registry(repo_root: Path) -> dict[str, dict[str, int]]:
    budgets_path = repo_root / "src" / "config" / "registry" / "budgets.yaml"
    if not budgets_path.exists():
        raise FileNotFoundError(f"Budget registry not found: {budgets_path}")
    raw = OmegaConf.load(str(budgets_path))
    out = to_dict(raw)
    parsed: dict[str, dict[str, int]] = {}
    for label, entry in out.items():
        if not isinstance(entry, dict):
            continue
        parsed[str(label)] = {k: int(v) for k, v in entry.items()}
    return parsed


def enforce_budget_constraint(
    cfg: dict[str, Any],
    repo_root: Path,
    budget_label: str,
    stage1_overrides: dict[str, str],
) -> None:
    constraints = cfg.get("matrix", {}).get("constraints", {})
    if not bool(constraints.get("enforce_final_n_from_budget", False)):
        return
    budgets = load_budget_registry(repo_root)
    if budget_label not in budgets:
        raise ValueError(f"Unknown budget label '{budget_label}' in budget registry.")
    budget = budgets[budget_label]
    n0_key = str(constraints.get("n0_key", "qbc_n0"))
    k_key = str(constraints.get("k_key", "qbc_K"))
    t_key = str(constraints.get("t_key", "qbc_T"))
    for key in (n0_key, k_key, t_key):
        if key not in stage1_overrides:
            raise ValueError(
                "Missing required stage1 override for final_n constraint: "
                f"'{key}'. Present keys: {sorted(stage1_overrides.keys())}"
            )
    n0 = int(float(stage1_overrides[n0_key]))
    k = int(float(stage1_overrides[k_key]))
    t = int(float(stage1_overrides[t_key]))
    final_n = n0 + k * t
    if final_n != int(budget["final_n"]):
        raise ValueError(
            f"Budget constraint failed for {budget_label}: "
            f"{n0_key}+{k_key}*{t_key}={final_n}, expected final_n={budget['final_n']}."
        )


def serialize_overrides(override_map: dict[str, str]) -> str:
    if not override_map:
        return ""
    parts = [f"{k}={override_map[k]}" for k in sorted(override_map.keys())]
    return ";;".join(parts)


def cfg_hash(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(data).hexdigest()[:10]
