"""Run a declarative experiment campaign from YAML config."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
import subprocess
import sys
import csv
from datetime import datetime, timezone
from time import monotonic
from typing import Any

from omegaconf import OmegaConf


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


PRUNED_DATA_NOTE = (
    "This path records the dataset location used during the run. Campaign cleanup may prune the on-disk "
    "dataset afterwards to control disk usage for large benchmark sweeps."
)


def _to_container_if_config(value: Any, default: Any) -> Any:
    if OmegaConf.is_config(value):
        out = OmegaConf.to_container(value, resolve=True)
        return default if out is None else out
    if value is None:
        return default
    return value


def _matches(combo: dict[str, str], rule: dict[str, Any]) -> bool:
    for k, v in rule.items():
        if str(combo.get(k)) != str(v):
            return False
    return True


def _build_matrix(axes: dict[str, list[Any]]) -> list[dict[str, str]]:
    keys = list(axes.keys())
    vals = [axes[k] for k in keys]
    out: list[dict[str, str]] = []
    for prod in itertools.product(*vals):
        out.append({k: str(v) for k, v in zip(keys, prod)})
    return out


def _normalize_dataset_axes(raw_axes: dict[str, list[Any]]) -> dict[str, list[Any]]:
    axes = dict(raw_axes)
    if "baseline_seed" in axes:
        raise ValueError("Use top-level 'baseline_seeds', not 'axes.baseline_seed'.")
    if "dataset_seed" not in axes:
        raise ValueError("campaign config must define 'axes.dataset_seed'.")
    return axes


def _apply_exclusions(
    combos: list[dict[str, str]],
    excludes: list[dict[str, Any]],
) -> list[dict[str, str]]:
    if not excludes:
        return combos
    keep: list[dict[str, str]] = []
    for combo in combos:
        if any(_matches(combo, rule) for rule in excludes):
            continue
        keep.append(combo)
    return keep


def _read_csv_rows(path: str) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _resolve_hpo_run_overrides(cfg: Any, repo_root: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    hpo_cfg = dict(_to_container_if_config(getattr(cfg, "hpo_integration", {}), {}))
    if not bool(hpo_cfg.get("enabled", False)):
        return [], {"enabled": False, "resolved": []}

    winners_csv = str(hpo_cfg.get("winners_csv", "results/hpo/hpo_winners.csv"))
    winners_path = winners_csv if os.path.isabs(winners_csv) else os.path.join(repo_root, winners_csv)
    if not os.path.exists(winners_path):
        raise FileNotFoundError(f"hpo_integration winners_csv not found: {winners_path}")

    winner_rows = _read_csv_rows(winners_path)
    policies = list(_to_container_if_config(hpo_cfg.get("policies", []), []))
    resolved_rules: list[dict[str, Any]] = []
    resolved_meta: list[dict[str, Any]] = []

    for policy in policies:
        method = str(policy.get("method", "")).strip()
        workflow_id = str(policy.get("workflow_id", "")).strip()
        preferred_stages = [str(x) for x in policy.get("preferred_stages", [])]
        stage1_keys = [str(x) for x in policy.get("stage1_keys", [])]
        allow_missing = bool(policy.get("allow_missing", False))
        if not method or not workflow_id or not preferred_stages or not stage1_keys:
            raise ValueError("Each hpo_integration policy requires method, workflow_id, preferred_stages, and stage1_keys.")

        selected_row: dict[str, str] | None = None
        selected_stage = ""
        for stage_name in preferred_stages:
            matches = [row for row in winner_rows if row.get("workflow_id") == workflow_id and row.get("stage") == stage_name]
            if matches:
                selected_row = matches[-1]
                selected_stage = stage_name
                break

        if selected_row is None:
            if allow_missing:
                resolved_meta.append(
                    {
                        "method": method,
                        "workflow_id": workflow_id,
                        "status": "missing_allowed",
                        "preferred_stages": preferred_stages,
                    }
                )
                continue
            raise ValueError(
                f"No HPO winner found for workflow_id='{workflow_id}' in stages {preferred_stages}. "
                "Either run HPO first or set allow_missing=true for this policy."
            )

        stage1 = []
        for key in stage1_keys:
            value = str(selected_row.get(key, "")).strip()
            if value:
                stage1.append(f"{key}={value}")
        if not stage1:
            if allow_missing:
                resolved_meta.append(
                    {
                        "method": method,
                        "workflow_id": workflow_id,
                        "status": "empty_allowed",
                        "selected_stage": selected_stage,
                    }
                )
                continue
            raise ValueError(
                f"HPO winner for workflow_id='{workflow_id}' stage='{selected_stage}' did not provide any configured stage1 keys."
            )

        resolved_rules.append(
            {
                "match": {"method": method},
                "stage1": stage1,
            }
        )
        resolved_meta.append(
            {
                "method": method,
                "workflow_id": workflow_id,
                "status": "resolved",
                "selected_stage": selected_stage,
                "stage1": stage1,
            }
        )

    return resolved_rules, {"enabled": True, "winners_path": winners_path, "resolved": resolved_meta}


def _expected_run_root(
    *,
    repo_root: str,
    experiment_id: str,
    preset: str,
    method: str,
    budget: str,
    dataset_seed: str,
) -> str:
    return os.path.join(
        repo_root,
        "outputs",
        "experiments",
        experiment_id,
        preset,
        method,
        budget,
        dataset_seed,
    )


def _is_manifest_completed(dataset_manifest_path: str) -> bool:
    if not os.path.exists(dataset_manifest_path):
        return False
    try:
        with open(dataset_manifest_path, "r", encoding="utf-8") as f:
            m = json.load(f)
    except Exception:
        return False
    stages_enabled = dict(m.get("stages_enabled", {}))
    stages = dict(m.get("stages", {}))
    for stage_name, enabled in stages_enabled.items():
        if not bool(enabled):
            continue
        if stages.get(stage_name, {}).get("status") != "completed":
            return False
    return True


def _annotate_pruned_dataset_path(dataset_manifest_path: str) -> None:
    if not os.path.exists(dataset_manifest_path):
        return
    try:
        with open(dataset_manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception:
        return
    artifacts = manifest.setdefault("artifacts", {})
    artifacts["dataset_root_note"] = PRUNED_DATA_NOTE
    if "preprocessed_root" in artifacts:
        artifacts["preprocessed_root_note"] = PRUNED_DATA_NOTE
    with open(dataset_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run campaign from YAML matrix.")
    p.add_argument("--config", required=True, help="Campaign YAML path.")
    p.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = OmegaConf.load(args.config)

    name = str(getattr(cfg.campaign, "name", "campaign"))
    experiment_id = str(getattr(cfg.campaign, "experiment_id", "thesis_sm4_campaign_v1"))
    preset = str(getattr(cfg.campaign, "preset", "default"))
    model_flag = str(getattr(cfg.campaign, "model_flag", "SM4"))
    skip_preprocess = bool(getattr(cfg.campaign, "skip_preprocess", False))
    skip_baseline = bool(getattr(cfg.campaign, "skip_baseline", False))
    baseline_epochs = getattr(cfg.campaign, "baseline_epochs", None)
    continue_on_error = bool(getattr(cfg.campaign, "continue_on_error", True))
    skip_completed_runs = bool(getattr(cfg.campaign, "skip_completed_runs", True))
    cleanup_cfg = dict(_to_container_if_config(getattr(cfg, "cleanup", {}), {}))
    prune_run_raw_data = bool(cleanup_cfg.get("prune_run_raw_data", False))
    prune_run_qbc_artifacts = bool(cleanup_cfg.get("prune_run_qbc_artifacts", False))
    prune_shared_test_raw_data = bool(cleanup_cfg.get("prune_shared_test_raw_data", False))

    axes = _to_container_if_config(cfg.axes, {})
    if not isinstance(axes, dict) or not axes:
        raise ValueError("campaign config must define non-empty 'axes'.")
    axes = _normalize_dataset_axes({k: list(v) for k, v in axes.items()})
    combos = _build_matrix(axes=axes)
    baseline_seeds = [str(x) for x in _to_container_if_config(getattr(cfg, "baseline_seeds", []), [])]
    if not skip_baseline and not baseline_seeds:
        raise ValueError("campaign config must define non-empty 'baseline_seeds' unless skip_baseline=true.")

    excludes = list(_to_container_if_config(getattr(cfg, "exclude", []), []))
    combos = _apply_exclusions(combos, excludes)
    total_dataset_runs = len(combos)
    total_baseline_subruns = total_dataset_runs * len(baseline_seeds) if not skip_baseline else 0
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    stage_overrides = dict(_to_container_if_config(getattr(cfg, "stage_overrides", {}), {}))
    stage1_overrides = [str(x) for x in stage_overrides.get("stage1", [])]
    stage2_overrides = [str(x) for x in stage_overrides.get("stage2", [])]
    stage3_overrides = [str(x) for x in stage_overrides.get("stage3", [])]
    run_override_rules = list(_to_container_if_config(getattr(cfg, "run_overrides", []), []))
    hpo_run_override_rules, hpo_integration_meta = _resolve_hpo_run_overrides(cfg, repo_root)
    run_override_rules.extend(hpo_run_override_rules)
    campaign_root = os.path.join(
        repo_root,
        "outputs",
        "campaigns",
        f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    os.makedirs(campaign_root, exist_ok=True)
    campaign_manifest_path = os.path.join(campaign_root, "campaign_manifest.json")

    manifest: dict[str, Any] = {
        "campaign": {
            "name": name,
            "config_path": os.path.abspath(args.config),
            "experiment_id": experiment_id,
            "preset": preset,
            "model_flag": model_flag,
            "axes": axes,
            "baseline_seeds": baseline_seeds,
            "skip_preprocess": skip_preprocess,
            "skip_baseline": skip_baseline,
            "baseline_epochs": baseline_epochs,
            "continue_on_error": continue_on_error,
            "skip_completed_runs": skip_completed_runs,
            "cleanup": {
                "prune_run_raw_data": prune_run_raw_data,
                "prune_run_qbc_artifacts": prune_run_qbc_artifacts,
                "prune_shared_test_raw_data": prune_shared_test_raw_data,
            },
            "hpo_integration": hpo_integration_meta,
            "created_at_utc": _utc_now_iso(),
        },
        "dataset_runs": [],
    }

    # Optional shared-test bootstrap: generate one dataset once, then use it for
    # all campaign preprocess stages via test_split_mode=shared_dataset.
    shared_test_cfg = dict(_to_container_if_config(getattr(cfg, "shared_test", {}), {}))
    shared_test_enabled = bool(shared_test_cfg.get("enabled", False))
    if shared_test_enabled:
        st_method = str(shared_test_cfg.get("method", "lhs_static"))
        st_budget = str(shared_test_cfg.get("budget", "b4096"))
        st_dataset_seed = str(shared_test_cfg.get("dataset_seed", "")).strip()
        if not st_dataset_seed:
            raise ValueError("shared_test.dataset_seed must be defined when shared_test.enabled=true.")
        st_max = shared_test_cfg.get("max_trajectories", None)
        st_stage1_overrides = [str(x) for x in shared_test_cfg.get("stage1_overrides", [])]
        st_run_root = str(
            shared_test_cfg.get(
                "run_root",
                os.path.join(campaign_root, "shared_test_source"),
            )
        )
        shared_test_reuse_if_exists = bool(shared_test_cfg.get("reuse_if_exists", True))

        shared_cmd = [
            sys.executable,
            "tools/benchmark/run_experiment.py",
            "--method",
            st_method,
            "--budget",
            st_budget,
            "--dataset-seed",
            st_dataset_seed,
            "--preset",
            preset,
            "--experiment-id",
            f"{experiment_id}_shared_test",
            "--model-flag",
            model_flag,
            "--run-root",
            st_run_root,
            "--skip-preprocess",
            "--skip-baseline",
        ]
        for ov in st_stage1_overrides:
            shared_cmd.extend(["--stage1-override", ov])

        manifest["shared_test"] = {
            "enabled": True,
            "command": shared_cmd,
            "status": "pending",
            "run_root": st_run_root,
            "reuse_if_exists": shared_test_reuse_if_exists,
            "dataset_seed": st_dataset_seed,
        }
        with open(campaign_manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        print("[campaign] preparing shared test source dataset...", flush=True)
        if args.dry_run:
            manifest["shared_test"]["status"] = "dry_run"
        else:
            shared_manifest_path = os.path.join(st_run_root, "dataset_manifest.json")
            if shared_test_reuse_if_exists and _is_manifest_completed(shared_manifest_path):
                manifest["shared_test"]["status"] = "reused_existing"
            else:
                proc = subprocess.run(shared_cmd, cwd=repo_root, text=True, check=False)
                manifest["shared_test"]["return_code"] = int(proc.returncode)
                if proc.returncode != 0:
                    manifest["shared_test"]["status"] = "failed"
                    with open(campaign_manifest_path, "w", encoding="utf-8") as f:
                        json.dump(manifest, f, indent=2)
                    raise RuntimeError("Shared-test source generation failed.")
                manifest["shared_test"]["status"] = "completed"
            with open(shared_manifest_path, "r", encoding="utf-8") as f:
                shared_manifest = json.load(f)
            shared_dataset_root = str(shared_manifest["artifacts"]["dataset_root"])
            manifest["shared_test"]["dataset_root"] = shared_dataset_root

            # Ensure all campaign runs use the same external test split.
            stage2_overrides = [
                ov
                for ov in stage2_overrides
                if not ov.startswith("dataset.shared_test_dataset_root=")
                and not ov.startswith("dataset.shared_test_dataset_number=")
            ]
            stage2_overrides = [
                ov
                for ov in stage2_overrides
                if not ov.startswith("dataset.test_split_mode=")
            ]
            stage2_overrides.append("dataset.test_split_mode=shared_dataset")
            stage2_overrides.append(f"dataset.shared_test_dataset_root={shared_dataset_root}")
            if st_max is not None:
                stage2_overrides = [
                    ov for ov in stage2_overrides if not ov.startswith("dataset.shared_test_max_trajectories=")
                ]
                stage2_overrides.append(f"dataset.shared_test_max_trajectories={int(st_max)}")

        with open(campaign_manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    campaign_started = monotonic()
    completed_dataset_durations: list[float] = []
    for idx, combo in enumerate(combos):
        method = combo.get("method")
        budget = combo.get("budget")
        dataset_seed = combo.get("dataset_seed")
        if not method or not budget or not dataset_seed:
            raise ValueError("Each matrix row must define method, budget, dataset_seed.")

        combo_stage1 = list(stage1_overrides)
        combo_stage2 = list(stage2_overrides)
        combo_stage3 = list(stage3_overrides)
        for rule in run_override_rules:
            match = dict(rule.get("match", {}))
            if _matches(combo, match):
                combo_stage1.extend([str(x) for x in rule.get("stage1", [])])
                combo_stage2.extend([str(x) for x in rule.get("stage2", [])])
                combo_stage3.extend([str(x) for x in rule.get("stage3", [])])

        cmd = [
            sys.executable,
            "tools/benchmark/run_experiment.py",
            "--method",
            method,
            "--budget",
            budget,
            "--dataset-seed",
            dataset_seed,
            "--preset",
            preset,
            "--experiment-id",
            experiment_id,
            "--model-flag",
            model_flag,
            "--dataset-run-index",
            str(idx + 1),
            "--dataset-run-total",
            str(total_dataset_runs),
        ]
        if not skip_baseline:
            cmd.extend(["--baseline-run-offset", str(idx * len(baseline_seeds))])
            cmd.extend(["--baseline-run-total", str(total_baseline_subruns)])
        for baseline_seed in baseline_seeds:
            cmd.extend(["--baseline-seed", baseline_seed])
        if skip_preprocess:
            cmd.append("--skip-preprocess")
        if skip_baseline:
            cmd.append("--skip-baseline")
        if baseline_epochs is not None:
            cmd.extend(["--baseline-epochs", str(int(baseline_epochs))])
        for ov in combo_stage1:
            cmd.extend(["--stage1-override", ov])
        for ov in combo_stage2:
            cmd.extend(["--stage2-override", ov])
        for ov in combo_stage3:
            cmd.extend(["--stage3-override", ov])

        run_item: dict[str, Any] = {
            "index": idx,
            "method": method,
            "budget": budget,
            "dataset_seed": dataset_seed,
            "baseline_seeds": list(baseline_seeds),
            "combo": combo,
            "command": cmd,
            "started_at_utc": _utc_now_iso(),
        }
        combo_text = " ".join(
            [f"{k}={v}" for k, v in combo.items() if k not in {"method", "budget", "dataset_seed"}]
        )
        combo_suffix = f" {combo_text}" if combo_text else ""
        baseline_span_text = ""
        if not skip_baseline and baseline_seeds:
            first_baseline_idx = idx * len(baseline_seeds) + 1
            last_baseline_idx = first_baseline_idx + len(baseline_seeds) - 1
            baseline_span_text = f" baseline_subruns={first_baseline_idx}-{last_baseline_idx}/{total_baseline_subruns}"
        print(
            f"[campaign] dataset_run={idx + 1}/{total_dataset_runs} "
            f"method={method} budget={budget} dataset_seed={dataset_seed} "
            f"baseline_seeds={','.join(baseline_seeds)}{baseline_span_text}{combo_suffix}",
            flush=True,
        )
        if args.dry_run:
            run_item["status"] = "dry_run"
            run_item["completed_at_utc"] = _utc_now_iso()
            manifest["dataset_runs"].append(run_item)
            continue

        run_root = _expected_run_root(
            repo_root=repo_root,
            experiment_id=experiment_id,
            preset=preset,
            method=method,
            budget=budget,
            dataset_seed=dataset_seed,
        )
        dataset_manifest_path = os.path.join(run_root, "dataset_manifest.json")
        if skip_completed_runs and _is_manifest_completed(dataset_manifest_path):
            run_item["status"] = "skipped_existing"
            run_item["completed_at_utc"] = _utc_now_iso()
            run_item["existing_run_root"] = run_root
            run_item["existing_dataset_manifest"] = dataset_manifest_path
            manifest["dataset_runs"].append(run_item)
            with open(campaign_manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
            print(
                f"[campaign] skipped existing completed run: {run_root}",
                flush=True,
            )
            continue

        run_started = monotonic()
        proc = subprocess.run(cmd, cwd=repo_root, text=True, check=False)
        elapsed = monotonic() - run_started
        run_item["return_code"] = int(proc.returncode)
        run_item["status"] = "completed" if proc.returncode == 0 else "failed"
        run_item["completed_at_utc"] = _utc_now_iso()
        run_item["duration_seconds"] = elapsed
        manifest["dataset_runs"].append(run_item)
        completed_dataset_durations.append(elapsed)

        with open(campaign_manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        elapsed_campaign = monotonic() - campaign_started
        avg_dataset_duration = sum(completed_dataset_durations) / len(completed_dataset_durations)
        remaining_dataset_runs = total_dataset_runs - (idx + 1)
        eta_seconds = avg_dataset_duration * remaining_dataset_runs
        print(
            f"[campaign] dataset_run={idx + 1}/{total_dataset_runs} "
            f"status={run_item['status']} duration={elapsed:.1f}s "
            f"elapsed_total={elapsed_campaign / 60.0:.1f}m eta_dataset_only={eta_seconds / 60.0:.1f}m",
            flush=True,
        )

        if proc.returncode != 0 and not continue_on_error:
            break

        if proc.returncode == 0:
            if prune_run_raw_data:
                data_dir = os.path.join(run_root, "data")
                if os.path.exists(data_dir):
                    shutil.rmtree(data_dir, ignore_errors=True)
                    _annotate_pruned_dataset_path(dataset_manifest_path)
                    print(f"[campaign] pruned raw data: {data_dir}", flush=True)
            if prune_run_qbc_artifacts:
                qbc_rounds_dir = os.path.join(run_root, "qbc", "rounds")
                qbc_ckpt_dir = os.path.join(run_root, "qbc", "checkpoints")
                removed_any = False
                for path in (qbc_rounds_dir, qbc_ckpt_dir):
                    if os.path.exists(path):
                        shutil.rmtree(path, ignore_errors=True)
                        removed_any = True
                if removed_any:
                    print(f"[campaign] pruned qbc artifacts: {run_root}/qbc/{{rounds,checkpoints}}", flush=True)

    if shared_test_enabled and prune_shared_test_raw_data and not args.dry_run:
        shared_test_run_root = str(manifest.get("shared_test", {}).get("run_root", "")).strip()
        if shared_test_run_root:
            shared_data_dir = os.path.join(shared_test_run_root, "data")
            shared_dataset_manifest_path = os.path.join(shared_test_run_root, "dataset_manifest.json")
            if os.path.exists(shared_data_dir):
                shutil.rmtree(shared_data_dir, ignore_errors=True)
                _annotate_pruned_dataset_path(shared_dataset_manifest_path)
                print(f"[campaign] pruned shared-test raw data: {shared_data_dir}", flush=True)

    with open(campaign_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[campaign] manifest: {campaign_manifest_path}", flush=True)


if __name__ == "__main__":
    main()
