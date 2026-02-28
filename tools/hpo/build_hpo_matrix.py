#!/usr/bin/env python3
"""Build a deterministic HPO matrix TSV from a method-scoped YAML config."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def _to_dict(cfg: Any, default: dict[str, Any] | None = None) -> dict[str, Any]:
    out = OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_config(cfg) else cfg
    if out is None:
        return {} if default is None else dict(default)
    if not isinstance(out, dict):
        raise ValueError(f"Expected mapping, got: {type(out)}")
    return out


def _to_list(cfg: Any) -> list[Any]:
    out = OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_config(cfg) else cfg
    if out is None:
        return []
    if not isinstance(out, list):
        raise ValueError(f"Expected list, got: {type(out)}")
    return out


def _parse_override(override: str) -> tuple[str, str]:
    if "=" not in override:
        raise ValueError(f"Invalid override '{override}'. Expected key=value.")
    k, v = override.split("=", 1)
    key = k.strip()
    val = v.strip()
    if not key:
        raise ValueError(f"Invalid override '{override}'. Empty key.")
    return key, val


def _parse_overrides(overrides: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for ov in overrides:
        k, v = _parse_override(str(ov))
        out[k] = v
    return out


def _format_override_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _merge_stage_overrides(
    fixed_list: list[str],
    axis_keys: list[str],
    axis_values: tuple[Any, ...],
) -> dict[str, str]:
    merged = _parse_overrides(fixed_list)
    for key, value in zip(axis_keys, axis_values):
        merged[str(key)] = _format_override_value(value)
    return merged


def _derive_stage1_complement(cfg: dict[str, Any], stage1: dict[str, str]) -> None:
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


def _load_budget_registry(repo_root: Path) -> dict[str, dict[str, int]]:
    budgets_path = repo_root / "src" / "config" / "registry" / "budgets.yaml"
    if not budgets_path.exists():
        raise FileNotFoundError(f"Budget registry not found: {budgets_path}")
    raw = OmegaConf.load(str(budgets_path))
    out = _to_dict(raw)
    parsed: dict[str, dict[str, int]] = {}
    for label, entry in out.items():
        if not isinstance(entry, dict):
            continue
        parsed[str(label)] = {k: int(v) for k, v in entry.items()}
    return parsed


def _enforce_budget_constraint(
    cfg: dict[str, Any],
    repo_root: Path,
    budget_label: str,
    stage1_overrides: dict[str, str],
) -> None:
    constraints = cfg.get("matrix", {}).get("constraints", {})
    if not bool(constraints.get("enforce_final_n_from_budget", False)):
        return
    budgets = _load_budget_registry(repo_root)
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


def _serialize_overrides(override_map: dict[str, str]) -> str:
    if not override_map:
        return ""
    parts = [f"{k}={override_map[k]}" for k in sorted(override_map.keys())]
    return ";;".join(parts)


def _cfg_hash(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(data).hexdigest()[:10]


def build_matrix(config_path: Path, output_root_override: str | None, with_timestamp: bool) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    cfg = _to_dict(OmegaConf.load(str(config_path)))
    hpo = cfg.get("hpo", {})
    matrix = cfg.get("matrix", {})
    axes = matrix.get("axes", {})
    stage_axes = _to_dict(axes.get("stage1", {}))
    stage2_axes = _to_dict(axes.get("stage2", {}))
    stage3_axes = _to_dict(axes.get("stage3", {}))

    budgets = [str(x) for x in _to_list(axes.get("budget"))]
    seeds = [str(x) for x in _to_list(axes.get("seed"))]
    if not budgets or not seeds:
        raise ValueError("matrix.axes.budget and matrix.axes.seed must both be non-empty lists.")

    hpo_id_base = str(hpo.get("id", config_path.stem)).strip()
    if not hpo_id_base:
        raise ValueError("hpo.id must be a non-empty string.")
    stage_name = str(hpo.get("stage", "stage_unspecified"))
    method_name = str(hpo.get("method", "")).strip()
    if not method_name:
        raise ValueError("hpo.method must be set.")
    preset = str(hpo.get("preset", "default"))
    experiment_id = str(hpo.get("experiment_id", f"hpo_{method_name}"))
    model_flag = str(hpo.get("model_flag", "SM4"))
    skip_preprocess = bool(hpo.get("skip_preprocess", False))
    skip_baseline = bool(hpo.get("skip_baseline", False))
    baseline_epochs = hpo.get("baseline_epochs", None)
    output_root = str(output_root_override or hpo.get("output_root", "outputs/hpo"))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    hpo_id = f"{hpo_id_base}_{stamp}" if with_timestamp else hpo_id_base
    hpo_root = repo_root / output_root / method_name / stage_name / hpo_id
    hpo_root.mkdir(parents=True, exist_ok=True)
    runs_root = hpo_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    fixed_overrides = _to_dict(matrix.get("fixed_overrides", {}))
    stage1_fixed = [str(x) for x in _to_list(fixed_overrides.get("stage1"))]
    stage2_fixed = [str(x) for x in _to_list(fixed_overrides.get("stage2"))]
    stage3_fixed = [str(x) for x in _to_list(fixed_overrides.get("stage3"))]

    stage1_keys = list(stage_axes.keys())
    stage2_keys = list(stage2_axes.keys())
    stage3_keys = list(stage3_axes.keys())
    stage1_values = [list(stage_axes[k]) for k in stage1_keys] if stage1_keys else [[]]
    stage2_values = [list(stage2_axes[k]) for k in stage2_keys] if stage2_keys else [[]]
    stage3_values = [list(stage3_axes[k]) for k in stage3_keys] if stage3_keys else [[]]

    rows: list[dict[str, Any]] = []
    row_idx = 0
    for budget, seed in product(budgets, seeds):
        for combo1 in (product(*stage1_values) if stage1_keys else [tuple()]):
            for combo2 in (product(*stage2_values) if stage2_keys else [tuple()]):
                for combo3 in (product(*stage3_values) if stage3_keys else [tuple()]):
                    stage1 = _merge_stage_overrides(stage1_fixed, stage1_keys, combo1)
                    stage2 = _merge_stage_overrides(stage2_fixed, stage2_keys, combo2)
                    stage3 = _merge_stage_overrides(stage3_fixed, stage3_keys, combo3)

                    _derive_stage1_complement(cfg, stage1)
                    _enforce_budget_constraint(cfg, repo_root, budget, stage1)

                    payload = {
                        "method": method_name,
                        "preset": preset,
                        "budget": budget,
                        "seed": seed,
                        "stage1": stage1,
                        "stage2": stage2,
                        "stage3": stage3,
                    }
                    cfg_id = f"cfg_{row_idx:05d}_{_cfg_hash(payload)}"
                    run_root = runs_root / cfg_id

                    rows.append(
                        {
                            "row_idx": row_idx,
                            "cfg_id": cfg_id,
                            "method": method_name,
                            "preset": preset,
                            "budget": budget,
                            "seed": seed,
                            "experiment_id": experiment_id,
                            "model_flag": model_flag,
                            "skip_preprocess": str(skip_preprocess).lower(),
                            "skip_baseline": str(skip_baseline).lower(),
                            "baseline_epochs": "" if baseline_epochs is None else str(int(baseline_epochs)),
                            "run_root": str(run_root),
                            "stage1_overrides": _serialize_overrides(stage1),
                            "stage2_overrides": _serialize_overrides(stage2),
                            "stage3_overrides": _serialize_overrides(stage3),
                        }
                    )
                    row_idx += 1

    matrix_path = hpo_root / "matrix.tsv"
    with matrix_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "row_idx",
                "cfg_id",
                "method",
                "preset",
                "budget",
                "seed",
                "experiment_id",
                "model_flag",
                "skip_preprocess",
                "skip_baseline",
                "baseline_epochs",
                "run_root",
                "stage1_overrides",
                "stage2_overrides",
                "stage3_overrides",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)

    meta = {
        "created_at_utc": datetime.utcnow().isoformat() + "Z",
        "config_path": str(config_path.resolve()),
        "hpo_root": str(hpo_root),
        "matrix_path": str(matrix_path),
        "total_rows": len(rows),
        "method": method_name,
        "stage": stage_name,
        "hpo_id": hpo_id,
    }
    meta_path = hpo_root / "matrix_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {
        "hpo_root": str(hpo_root),
        "matrix_path": str(matrix_path),
        "meta_path": str(meta_path),
        "total_rows": len(rows),
        "hpo_id": hpo_id,
        "method": method_name,
        "stage": stage_name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build HPO matrix TSV from YAML.")
    parser.add_argument("--config", required=True, help="Path to HPO YAML config.")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional override of hpo.output_root (repo-relative or absolute).",
    )
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help="Do not append timestamp to hpo.id.",
    )
    parser.add_argument(
        "--env-out",
        default=None,
        help="Optional output path for shell env file with HPO_ROOT/MATRIX_PATH/META_PATH/TOTAL_ROWS.",
    )
    args = parser.parse_args()

    result = build_matrix(
        config_path=Path(args.config),
        output_root_override=args.output_root,
        with_timestamp=not args.no_timestamp,
    )

    if args.env_out:
        env_path = Path(args.env_out)
        env_path.parent.mkdir(parents=True, exist_ok=True)
        with env_path.open("w", encoding="utf-8") as f:
            f.write(f"HPO_ROOT={result['hpo_root']}\n")
            f.write(f"MATRIX_PATH={result['matrix_path']}\n")
            f.write(f"META_PATH={result['meta_path']}\n")
            f.write(f"TOTAL_ROWS={result['total_rows']}\n")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

