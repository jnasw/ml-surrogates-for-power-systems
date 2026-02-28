"""Run end-to-end experiments: dataset generation -> preprocess -> baseline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from typing import Any

from omegaconf import OmegaConf

from src.pipeline.manifest import init_manifest, save_manifest, set_stage_status, utc_now_iso
from src.pipeline.telemetry import (
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
    if not os.path.exists(seed_registry):
        raise FileNotFoundError(f"Seed registry not found: {seed_registry}")
    cfg = OmegaConf.load(seed_registry)
    if seed_label not in cfg:
        raise ValueError(f"Unknown seed label '{seed_label}' in {seed_registry}")
    return int(cfg[seed_label])


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
        set_stage_status(
            manifest,
            stage=name,
            status="failed",
            completed_at_utc=utc_now_iso(),
            return_code=proc.returncode,
            error=f"Stage '{name}' failed. See log: {log_path}",
        )
        save_manifest(manifest_path, manifest)
        raise RuntimeError(f"Stage '{name}' failed with code {proc.returncode}. See {log_path}")

    set_stage_status(
        manifest,
        stage=name,
        status="completed",
        completed_at_utc=utc_now_iso(),
        return_code=0,
    )
    save_manifest(manifest_path, manifest)


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run full experiment pipeline in one command.")
    parser.add_argument("--method", required=True, choices=sorted(METHOD_TO_CONFIG.keys()))
    parser.add_argument("--budget", required=True, help="Budget label from src/config/registry/budgets.yaml")
    parser.add_argument("--seed", required=True, help="Seed label from src/config/registry/seeds.yaml")
    parser.add_argument("--preset", default="default", help="Preset config name under src/config/preset.")
    parser.add_argument("--experiment-id", default="thesis_sm4_pipeline_v1")
    parser.add_argument("--model-flag", default="SM4")
    parser.add_argument(
        "--run-root",
        default=None,
        help="Optional explicit run root. Default: outputs/experiments/<id>/<preset>/<method>/<budget>/<seed>",
    )
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--baseline-epochs", type=int, default=None)
    parser.add_argument("--baseline-save-name", default="baseline.pt")
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

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    method_config_name = METHOD_TO_CONFIG[args.method]
    seed_value = _load_seed_value(repo_root=repo_root, seed_label=args.seed)
    budget_cfg = _load_budget(repo_root=repo_root, budget_label=args.budget)

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
            args.seed,
        )

    data_root = os.path.join(run_root, "data")
    qbc_root = os.path.join(run_root, "qbc")
    hydra_root = os.path.join(run_root, "hydra")
    logs_root = os.path.join(run_root, "logs")
    baseline_root = os.path.join(run_root, "baseline")
    os.makedirs(run_root, exist_ok=True)

    run_id = f"{args.method}_{args.budget}_{args.seed}"
    manifest_path = os.path.join(run_root, "run_manifest.json")
    manifest = init_manifest(
        run_id=run_id,
        run_root=run_root,
        method=args.method,
        budget=args.budget,
        seed_label=args.seed,
        seed_value=seed_value,
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

    stage1_cmd = [
        sys.executable,
        "00_create_dataset.py",
        "+base=base",
        f"preset={args.preset}",
        f"+method={method_config_name}",
        f"experiment.id={args.experiment_id}",
        f"experiment.preset={args.preset}",
        f"experiment.budget={args.budget}",
        f"experiment.seed={args.seed}",
        f"model.seed={seed_value}",
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
            run_id=run_id,
            method=args.method,
            budget=args.budget,
            seed_label=args.seed,
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
        stage3_cmd = [
            sys.executable,
            "10_run_baseline.py",
            f"model.model_flag={args.model_flag}",
            f"model.seed={seed_value}",
            f"dataset.root={dataset_root}",
            f"baseline.save_dir={baseline_root}",
            f"baseline.save_name={args.baseline_save_name}",
            f"hydra.run.dir={os.path.join(hydra_root, 'stage3_' + datetime.now().strftime('%Y%m%d_%H%M%S'))}",
            "hydra.job.chdir=false",
        ]
        if args.baseline_epochs is not None:
            stage3_cmd.append(f"baseline.epochs={int(args.baseline_epochs)}")
        stage3_cmd.extend(list(args.stage3_override))
        _run_stage(
            name="stage3_baseline",
            command=stage3_cmd,
            cwd=repo_root,
            log_path=os.path.join(logs_root, "stage3_baseline.log"),
            manifest=manifest,
            manifest_path=manifest_path,
        )
        metrics_path = os.path.join(baseline_root, "metrics.json")
        if os.path.exists(metrics_path):
            manifest["artifacts"]["baseline_metrics"] = metrics_path
            manifest["artifacts"]["baseline_metrics_payload"] = _read_json(metrics_path)
            save_manifest(manifest_path, manifest)

    print(f"Run completed: {run_id}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
