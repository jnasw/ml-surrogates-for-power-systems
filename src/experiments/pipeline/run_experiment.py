"""Run dataset-centric experiments: dataset generation -> preprocess -> baseline subruns."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from time import monotonic
from typing import Any

from omegaconf import OmegaConf

from src.experiments.pipeline.helpers.manifest import init_dataset_manifest, save_manifest, set_stage_status, utc_now_iso
from src.experiments.pipeline.helpers.seeds import load_seed_registry
from src.experiments.pipeline.helpers.telemetry import (
    build_round_telemetry_rows,
    write_rows_csv,
    write_rows_parquet_best_effort,
)

METHOD_TO_CONFIG: dict[str, str] = {
    "lhs_static": "lhs_static",
    "qbc_deep_ensemble": "qbc_deep_ensemble",
    "marker_directed": "marker_directed",
    "qbc_marker_hybrid": "qbc_marker_hybrid",
}


def _resolve_path(path_like: str, cwd: str) -> str:
    if os.path.isabs(path_like):
        return path_like
    return os.path.abspath(os.path.join(cwd, path_like))


def _load_seed_value(repo_root: str, seed_label: str) -> int:
    seed_registry = os.path.join(repo_root, "src", "config", "registry", "seeds.yaml")
    registry = load_seed_registry(Path(seed_registry))
    if seed_label not in registry:
        raise ValueError(f"Unknown seed label '{seed_label}' in {seed_registry}")
    return int(registry[seed_label])


def _load_budget(repo_root: str, budget_label: str) -> dict[str, int]:
    budget_registry = os.path.join(repo_root, "src", "config", "registry", "budgets.yaml")
    if not os.path.exists(budget_registry):
        raise FileNotFoundError(f"Budget registry not found: {budget_registry}")
    cfg = OmegaConf.load(budget_registry)
    if budget_label not in cfg:
        raise ValueError(f"Unknown budget label '{budget_label}' in {budget_registry}")
    entry = cfg[budget_label]
    required = ("final_n", "n0", "k", "t", "p")
    for key in required:
        if key not in entry:
            raise ValueError(f"Budget '{budget_label}' missing '{key}' in {budget_registry}")
    return {k: int(entry[k]) for k in required}


def _git_commit(repo_root: str) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return None


def _run_stage(
    *,
    name: str,
    command: list[str],
    cwd: str,
    log_path: str,
    manifest: dict[str, Any],
    manifest_path: str,
) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    stage_started = monotonic()
    print(f"[experiment] starting {name}", flush=True)
    set_stage_status(
        manifest,
        stage=name,
        status="running",
        command=command,
        log_file=log_path,
        started_at_utc=utc_now_iso(),
    )
    save_manifest(manifest_path, manifest)

    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(command, cwd=cwd, stdout=logf, stderr=subprocess.STDOUT, text=True, check=False)

    if proc.returncode != 0:
        elapsed = monotonic() - stage_started
        set_stage_status(
            manifest,
            stage=name,
            status="failed",
            completed_at_utc=utc_now_iso(),
            return_code=proc.returncode,
            error=f"Stage '{name}' failed. See log: {log_path}",
        )
        save_manifest(manifest_path, manifest)
        raise RuntimeError(
            f"Stage '{name}' failed with code {proc.returncode} after {elapsed:.1f}s. See {log_path}"
        )

    elapsed = monotonic() - stage_started
    set_stage_status(
        manifest,
        stage=name,
        status="completed",
        completed_at_utc=utc_now_iso(),
        return_code=0,
    )
    save_manifest(manifest_path, manifest)
    print(f"[experiment] completed {name} in {elapsed:.1f}s", flush=True)


def _set_baseline_run_status(
    manifest: dict[str, Any],
    *,
    baseline_seed_label: str,
    baseline_seed_value: int,
    baseline_root: str,
    status: str,
    command: list[str] | None = None,
    log_file: str | None = None,
    started_at_utc: str | None = None,
    completed_at_utc: str | None = None,
    return_code: int | None = None,
    error: str | None = None,
    metrics_path: str | None = None,
    metrics_payload: dict[str, Any] | None = None,
) -> None:
    entry = manifest["baseline_runs"].get(baseline_seed_label, {})
    entry["baseline_seed_label"] = baseline_seed_label
    entry["baseline_seed_value"] = int(baseline_seed_value)
    entry["baseline_root"] = baseline_root
    entry["status"] = status
    if command is not None:
        entry["command"] = command
    if log_file is not None:
        entry["log_file"] = log_file
    if started_at_utc is not None:
        entry["started_at_utc"] = started_at_utc
    if completed_at_utc is not None:
        entry["completed_at_utc"] = completed_at_utc
    if return_code is not None:
        entry["return_code"] = int(return_code)
    if error is not None:
        entry["error"] = error
    if metrics_path is not None:
        entry["metrics_path"] = metrics_path
    if metrics_payload is not None:
        entry["metrics_payload"] = metrics_payload
    manifest["baseline_runs"][baseline_seed_label] = entry


def _write_baseline_summary(manifest: dict[str, Any], baseline_root: str) -> str | None:
    completed = [
        run for run in manifest.get("baseline_runs", {}).values() if run.get("status") == "completed"
    ]
    if not completed:
        manifest["baseline_summary"] = {}
        return None

    def _metric_values(metric_name: str) -> list[float]:
        out: list[float] = []
        for run in completed:
            payload = dict(run.get("metrics_payload", {}))
            final_test_metrics = dict(payload.get("final_test_metrics", {}) or {})
            final_train_metrics = dict(payload.get("final_train_metrics", {}) or {})
            value = final_test_metrics.get(metric_name)
            if value is None:
                value = final_train_metrics.get(metric_name)
            if value is not None:
                out.append(float(value))
        return out

    mse_vals = _metric_values("mse")
    rmse_vals = _metric_values("rmse")
    n_train = next(
        (
            int(run.get("metrics_payload", {}).get("n_train"))
            for run in completed
            if run.get("metrics_payload", {}).get("n_train") is not None
        ),
        None,
    )
    n_test = next(
        (
            int(run.get("metrics_payload", {}).get("n_test"))
            for run in completed
            if run.get("metrics_payload", {}).get("n_test") is not None
        ),
        None,
    )
    summary = {
        "n_runs": len(completed),
        "baseline_seed_labels": [str(run["baseline_seed_label"]) for run in completed],
        "mse_mean": statistics.mean(mse_vals) if mse_vals else None,
        "mse_std": statistics.stdev(mse_vals) if len(mse_vals) > 1 else 0.0 if mse_vals else None,
        "mse_min": min(mse_vals) if mse_vals else None,
        "mse_max": max(mse_vals) if mse_vals else None,
        "rmse_mean": statistics.mean(rmse_vals) if rmse_vals else None,
        "rmse_std": statistics.stdev(rmse_vals) if len(rmse_vals) > 1 else 0.0 if rmse_vals else None,
        "rmse_min": min(rmse_vals) if rmse_vals else None,
        "rmse_max": max(rmse_vals) if rmse_vals else None,
        "n_train": n_train,
        "n_test": n_test,
    }
    manifest["baseline_summary"] = summary
    os.makedirs(baseline_root, exist_ok=True)
    out_path = os.path.join(baseline_root, "summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return out_path


def _find_dataset_root(dataset_base_dir: str, model_flag: str) -> str:
    model_dir = os.path.join(dataset_base_dir, model_flag)
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model dataset directory not found: {model_dir}")

    dataset_dirs = [
        d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d)) and d.startswith("dataset_v")
    ]
    if not dataset_dirs:
        raise FileNotFoundError(f"No dataset_vN directories found in {model_dir}")
    dataset_dirs.sort(key=lambda x: int(x.replace("dataset_v", "")))
    return os.path.join(model_dir, dataset_dirs[-1])


def _read_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _dedupe_hydra_command(command: list[str]) -> list[str]:
    """Keep only the last occurrence for duplicate Hydra override keys."""
    if len(command) <= 2:
        return list(command)

    prefix = list(command[:2])
    overrides = list(command[2:])
    last_index: dict[str, int] = {}
    for idx, item in enumerate(overrides):
        key = item.split("=", 1)[0] if "=" in item else item
        last_index[key] = idx

    deduped = [
        item
        for idx, item in enumerate(overrides)
        if last_index[item.split("=", 1)[0] if "=" in item else item] == idx
    ]
    return prefix + deduped


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run full experiment pipeline in one command.")
    parser.add_argument("--method", required=True, choices=sorted(METHOD_TO_CONFIG.keys()))
    parser.add_argument("--budget", required=True, help="Budget label from src/config/registry/budgets.yaml")
    parser.add_argument(
        "--dataset-seed",
        required=True,
        help="Dataset generation seed label from src/config/registry/seeds.yaml.",
    )
    parser.add_argument(
        "--baseline-seed",
        action="append",
        default=[],
        help="Baseline training seed label from src/config/registry/seeds.yaml (repeatable).",
    )
    parser.add_argument("--preset", default="default", help="Preset config name under src/config/preset.")
    parser.add_argument("--experiment-id", default="thesis_sm4_pipeline_v1")
    parser.add_argument("--model-flag", default="SM4")
    parser.add_argument(
        "--run-root",
        default=None,
        help="Optional explicit run root. Default: outputs/experiments/<id>/<preset>/<method>/<budget>/<dataset-seed>",
    )
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--baseline-epochs", type=int, default=None)
    parser.add_argument("--baseline-save-name", default="baseline.pt")
    parser.add_argument("--dataset-run-index", type=int, default=None)
    parser.add_argument("--dataset-run-total", type=int, default=None)
    parser.add_argument("--baseline-run-offset", type=int, default=0)
    parser.add_argument("--baseline-run-total", type=int, default=None)
    parser.add_argument(
        "--stage1-override",
        action="append",
        default=[],
        help="Extra Hydra override for create_dataset.py (repeatable). Example: --stage1-override qbc_M=5",
    )
    parser.add_argument(
        "--stage2-override",
        action="append",
        default=[],
        help="Extra Hydra override for preprocess_dataset.py (repeatable).",
    )
    parser.add_argument(
        "--stage3-override",
        action="append",
        default=[],
        help="Extra Hydra override for run_baseline.py (repeatable).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    method_config_name = METHOD_TO_CONFIG[args.method]
    dataset_seed_label = str(args.dataset_seed).strip()
    dataset_seed_value = _load_seed_value(repo_root=repo_root, seed_label=dataset_seed_label)
    baseline_seed_labels = [str(x).strip() for x in list(args.baseline_seed) if str(x).strip()]
    if not args.skip_baseline and not baseline_seed_labels:
        baseline_seed_labels = [dataset_seed_label]
    baseline_seed_values = {
        seed_label: _load_seed_value(repo_root=repo_root, seed_label=seed_label) for seed_label in baseline_seed_labels
    }
    budget_cfg = _load_budget(repo_root=repo_root, budget_label=args.budget)
    dataset_progress = None
    if args.dataset_run_index is not None and args.dataset_run_total is not None:
        dataset_progress = f"{int(args.dataset_run_index)}/{int(args.dataset_run_total)}"

    if args.run_root:
        run_root = _resolve_path(args.run_root, repo_root)
    else:
        run_root = os.path.join(
            repo_root,
            "outputs",
            "experiments",
            args.experiment_id,
            args.preset,
            args.method,
            args.budget,
            dataset_seed_label,
        )

    data_root = os.path.join(run_root, "data")
    qbc_root = os.path.join(run_root, "qbc")
    hydra_root = os.path.join(run_root, "hydra")
    logs_root = os.path.join(run_root, "logs")
    baseline_root = os.path.join(run_root, "baseline")
    os.makedirs(run_root, exist_ok=True)

    dataset_id = f"{args.method}_{args.budget}_{dataset_seed_label}"
    manifest_path = os.path.join(run_root, "dataset_manifest.json")
    manifest = init_dataset_manifest(
        dataset_id=dataset_id,
        run_root=run_root,
        method=args.method,
        budget=args.budget,
        dataset_seed_label=dataset_seed_label,
        dataset_seed_value=dataset_seed_value,
        preset=args.preset,
        experiment_id=args.experiment_id,
        model_flag=args.model_flag,
        git_commit=_git_commit(repo_root),
        stages_enabled={
            "stage1_create_dataset": True,
            "stage2_preprocess": not args.skip_preprocess,
            "stage3_baseline": not args.skip_baseline,
        },
    )
    save_manifest(manifest_path, manifest)
    print(
        "[experiment] dataset run"
        + (f" {dataset_progress}" if dataset_progress else "")
        + f": method={args.method} budget={args.budget} dataset_seed={dataset_seed_label}",
        flush=True,
    )
    if not args.skip_baseline:
        if args.baseline_run_total:
            first_global = args.baseline_run_offset + 1
            last_global = args.baseline_run_offset + len(baseline_seed_labels)
            print(
                f"[experiment] baseline subruns {first_global}-{last_global}/{int(args.baseline_run_total)}: "
                + ",".join(baseline_seed_labels),
                flush=True,
            )
        else:
            print(
                f"[experiment] baseline subruns 1-{len(baseline_seed_labels)}/{len(baseline_seed_labels)}: "
                + ",".join(baseline_seed_labels),
                flush=True,
            )

    stage1_cmd = [
        sys.executable,
        "00_create_dataset.py",
        "+base=base",
        f"preset={args.preset}",
        f"+method={method_config_name}",
        f"experiment.id={args.experiment_id}",
        f"experiment.preset={args.preset}",
        f"experiment.budget={args.budget}",
        f"experiment.seed={dataset_seed_label}",
        f"model.seed={dataset_seed_value}",
        f"model.model_flag={args.model_flag}",
        f"dirs.dataset_dir={data_root}",
        f"qbc_run_dir={qbc_root}",
        f"hydra.run.dir={os.path.join(hydra_root, 'stage1_' + datetime.now().strftime('%Y%m%d_%H%M%S'))}",
        "hydra.job.chdir=false",
    ]
    if args.method == "lhs_static":
        stage1_cmd.append("qbc_enable_logging=false")
        stage1_cmd.append(f"model.ic_num_samples={budget_cfg['final_n']}")
    else:
        stage1_cmd.append("qbc_enable_logging=true")
        stage1_cmd.append(f"qbc_n0={budget_cfg['n0']}")
        stage1_cmd.append(f"qbc_K={budget_cfg['k']}")
        stage1_cmd.append(f"qbc_T={budget_cfg['t']}")
        stage1_cmd.append(f"qbc_P={budget_cfg['p']}")
    stage1_cmd.extend(list(args.stage1_override))
    stage1_cmd = _dedupe_hydra_command(stage1_cmd)
    _run_stage(
        name="stage1_create_dataset",
        command=stage1_cmd,
        cwd=repo_root,
        log_path=os.path.join(logs_root, "stage1_create_dataset.log"),
        manifest=manifest,
        manifest_path=manifest_path,
    )

    dataset_root = _find_dataset_root(dataset_base_dir=data_root, model_flag=args.model_flag)
    manifest["artifacts"]["dataset_root"] = dataset_root
    manifest["artifacts"]["stage1_data_root"] = data_root
    if os.path.exists(os.path.join(qbc_root, "history.jsonl")):
        history_path = os.path.join(qbc_root, "history.jsonl")
        rounds_dir = os.path.join(qbc_root, "rounds")
        manifest["artifacts"]["qbc_history"] = history_path
        telemetry_rows = build_round_telemetry_rows(
            history_jsonl_path=history_path,
            rounds_dir=rounds_dir,
            dataset_id=dataset_id,
            method=args.method,
            budget=args.budget,
            dataset_seed_label=dataset_seed_label,
        )
        telemetry_csv = os.path.join(run_root, "telemetry", "round_telemetry.csv")
        write_rows_csv(telemetry_rows, telemetry_csv)
        manifest["artifacts"]["round_telemetry_csv"] = telemetry_csv
        telemetry_parquet = os.path.join(run_root, "telemetry", "round_telemetry.parquet")
        if write_rows_parquet_best_effort(telemetry_rows, telemetry_parquet):
            manifest["artifacts"]["round_telemetry_parquet"] = telemetry_parquet
    save_manifest(manifest_path, manifest)

    if not args.skip_preprocess:
        stage2_cmd = [
            sys.executable,
            "01_preprocess_dataset.py",
            f"model.model_flag={args.model_flag}",
            f"dataset.root={dataset_root}",
            f"dirs.dataset_dir={data_root}",
            f"hydra.run.dir={os.path.join(hydra_root, 'stage2_' + datetime.now().strftime('%Y%m%d_%H%M%S'))}",
            "hydra.job.chdir=false",
        ]
        stage2_cmd.extend(list(args.stage2_override))
        stage2_cmd = _dedupe_hydra_command(stage2_cmd)
        _run_stage(
            name="stage2_preprocess",
            command=stage2_cmd,
            cwd=repo_root,
            log_path=os.path.join(logs_root, "stage2_preprocess.log"),
            manifest=manifest,
            manifest_path=manifest_path,
        )
        manifest["artifacts"]["preprocessed_root"] = dataset_root
        save_manifest(manifest_path, manifest)

    if not args.skip_baseline:
        set_stage_status(
            manifest,
            stage="stage3_baseline",
            status="running",
            started_at_utc=utc_now_iso(),
            command=[
                "baseline_subruns",
                f"dataset_seed={dataset_seed_label}",
                f"baseline_seeds={','.join(baseline_seed_labels)}",
            ],
            log_file=os.path.join(logs_root, "stage3_baseline.log"),
            extra={"baseline_seed_labels": baseline_seed_labels},
        )
        save_manifest(manifest_path, manifest)

        try:
            total_local_baselines = len(baseline_seed_labels)
            for local_idx, baseline_seed_label in enumerate(baseline_seed_labels, start=1):
                baseline_seed_value = baseline_seed_values[baseline_seed_label]
                baseline_run_root = os.path.join(baseline_root, baseline_seed_label)
                baseline_log_path = os.path.join(logs_root, f"stage3_baseline_{baseline_seed_label}.log")
                baseline_started = monotonic()
                global_idx = args.baseline_run_offset + local_idx
                global_text = (
                    f"{global_idx}/{int(args.baseline_run_total)}"
                    if args.baseline_run_total
                    else f"{local_idx}/{total_local_baselines}"
                )
                print(
                    f"[experiment] baseline subrun {global_text} "
                    f"(dataset {local_idx}/{total_local_baselines}) seed={baseline_seed_label} starting",
                    flush=True,
                )
                stage3_cmd = [
                    sys.executable,
                    "10_run_baseline.py",
                    f"model.model_flag={args.model_flag}",
                    f"model.seed={baseline_seed_value}",
                    f"dataset.root={dataset_root}",
                    f"baseline.run_dir={baseline_run_root}",
                    f"baseline.checkpoint_name={args.baseline_save_name}",
                    (
                        "hydra.run.dir="
                        + os.path.join(
                            hydra_root,
                            "stage3_" + baseline_seed_label + "_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
                        )
                    ),
                    "hydra.job.chdir=false",
                ]
                if args.baseline_epochs is not None:
                    stage3_cmd.append(f"baseline.epochs={int(args.baseline_epochs)}")
                stage3_cmd.extend(list(args.stage3_override))
                stage3_cmd = _dedupe_hydra_command(stage3_cmd)

                _set_baseline_run_status(
                    manifest,
                    baseline_seed_label=baseline_seed_label,
                    baseline_seed_value=baseline_seed_value,
                    baseline_root=baseline_run_root,
                    status="running",
                    command=stage3_cmd,
                    log_file=baseline_log_path,
                    started_at_utc=utc_now_iso(),
                )
                save_manifest(manifest_path, manifest)

                os.makedirs(os.path.dirname(baseline_log_path), exist_ok=True)
                with open(baseline_log_path, "w", encoding="utf-8") as logf:
                    proc = subprocess.run(
                        stage3_cmd,
                        cwd=repo_root,
                        stdout=logf,
                        stderr=subprocess.STDOUT,
                        text=True,
                        check=False,
                    )

                if proc.returncode != 0:
                    elapsed = monotonic() - baseline_started
                    _set_baseline_run_status(
                        manifest,
                        baseline_seed_label=baseline_seed_label,
                        baseline_seed_value=baseline_seed_value,
                        baseline_root=baseline_run_root,
                        status="failed",
                        completed_at_utc=utc_now_iso(),
                        return_code=proc.returncode,
                        error=f"Baseline subrun failed. See log: {baseline_log_path}",
                    )
                    save_manifest(manifest_path, manifest)
                    raise RuntimeError(
                        f"Baseline subrun '{baseline_seed_label}' failed with code {proc.returncode} "
                        f"after {elapsed:.1f}s. "
                        f"See {baseline_log_path}"
                    )

                metrics_path = os.path.join(baseline_run_root, "metrics.json")
                metrics_payload = _read_json(metrics_path) if os.path.exists(metrics_path) else {}
                elapsed = monotonic() - baseline_started
                _set_baseline_run_status(
                    manifest,
                    baseline_seed_label=baseline_seed_label,
                    baseline_seed_value=baseline_seed_value,
                    baseline_root=baseline_run_root,
                    status="completed",
                    completed_at_utc=utc_now_iso(),
                    return_code=0,
                    metrics_path=metrics_path if os.path.exists(metrics_path) else None,
                    metrics_payload=metrics_payload,
                )
                save_manifest(manifest_path, manifest)
                final_test_metrics = metrics_payload.get("final_test_metrics") or {}
                rmse = final_test_metrics.get("rmse")
                rmse_text = f" rmse={float(rmse):.6f}" if rmse is not None else ""
                print(
                    f"[experiment] baseline subrun {global_text} "
                    f"(dataset {local_idx}/{total_local_baselines}) seed={baseline_seed_label} "
                    f"completed in {elapsed:.1f}s{rmse_text}",
                    flush=True,
                )
        except Exception as exc:
            set_stage_status(
                manifest,
                stage="stage3_baseline",
                status="failed",
                completed_at_utc=utc_now_iso(),
                error=str(exc),
                extra={"baseline_seed_labels": baseline_seed_labels},
            )
            save_manifest(manifest_path, manifest)
            raise

        baseline_summary_path = _write_baseline_summary(manifest, baseline_root)
        if baseline_summary_path is not None:
            manifest["artifacts"]["baseline_summary"] = baseline_summary_path
        save_manifest(manifest_path, manifest)
        set_stage_status(
            manifest,
            stage="stage3_baseline",
            status="completed",
            completed_at_utc=utc_now_iso(),
            return_code=0,
            extra={"baseline_seed_labels": baseline_seed_labels},
        )
        save_manifest(manifest_path, manifest)

    print(f"[experiment] run completed: {dataset_id}", flush=True)
    print(f"[experiment] manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
