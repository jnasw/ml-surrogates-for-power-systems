"""Post-training checkpoint evaluation runner.

This runner currently performs discovery and output bookkeeping for posthoc
checkpoint evaluation. The checkpoint inference step plugs into the same loop.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from src.data.contracts.data_contract import SPLITS
from src.experiments.pipeline.helpers.posthoc_evaluation import (
    PosthocRunCandidate,
    base_posthoc_run_result,
    discover_dataset_generation_posthoc_runs,
    discover_posthoc_runs,
    evaluate_posthoc_prediction,
    posthoc_output_dir,
    posthoc_run_result_path,
    predict_pinn_model_on_evaluation_split,
    read_posthoc_run_result,
    read_posthoc_run_results,
    write_posthoc_batch_outputs,
    write_posthoc_run_result,
)
from src.training.trainer import PinnModel


OOD_MODES = ("exclusive", "wide")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover post-training checkpoint evaluation candidates and write posthoc output scaffolds."
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Experiment root, timestamp batch root, or single run directory to post-evaluate.",
    )
    parser.add_argument("--checkpoint-tag", default="best", help="Checkpoint tag under checkpoints/, default: best.")
    parser.add_argument("--eval-split", default="train", choices=SPLITS, help="External evaluation split to use.")
    parser.add_argument("--ood-mode", default="exclusive", choices=OOD_MODES, help="OOD handling mode.")
    parser.add_argument("--batch-size", type=int, default=4096, help="Inference batch size for the future evaluation step.")
    parser.add_argument("--repo-root", default=".", help="Repository root used for resolving copied HPC paths.")
    parser.add_argument("--id-eval-id", default=None, help="Explicit ID evaluation dataset ID from data/evaluation/index.json.")
    parser.add_argument("--ood-eval-id", default=None, help="Explicit OOD evaluation dataset ID from data/evaluation/index.json.")
    parser.add_argument("--id-eval-root", default=None, help="Explicit ID evaluation dataset root.")
    parser.add_argument("--ood-eval-root", default=None, help="Explicit OOD evaluation dataset root.")
    parser.add_argument(
        "--dataset-generation-layout",
        action="store_true",
        help="Discover runs from outputs/dataset_generation_analysis nested summary layout.",
    )
    parser.add_argument(
        "--model-flag",
        default=None,
        help="Fallback model flag used when a run config.yaml is missing, e.g. SM4.",
    )
    parser.add_argument(
        "--init-condition-bounds",
        type=int,
        default=None,
        help="Fallback training IC bounds index used when a run config.yaml is missing.",
    )
    parser.add_argument("--no-id-eval", action="store_true", help="Do not require or discover ID evaluation inputs.")
    parser.add_argument("--no-ood-eval", action="store_true", help="Do not require or discover OOD evaluation inputs.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing per-run posthoc JSON files.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print discovery summary without writing posthoc files.",
    )
    return parser.parse_args()


def _settings(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "checkpoint_tag": str(args.checkpoint_tag),
        "eval_split": str(args.eval_split),
        "ood_mode": str(args.ood_mode),
        "batch_size": int(args.batch_size),
        "root": str(args.root),
        "repo_root": str(args.repo_root),
        "id_eval_id": args.id_eval_id,
        "ood_eval_id": args.ood_eval_id,
        "id_eval_root": args.id_eval_root,
        "ood_eval_root": args.ood_eval_root,
        "require_id_eval": not bool(args.no_id_eval),
        "require_ood_eval": not bool(args.no_ood_eval),
        "dataset_generation_layout": bool(args.dataset_generation_layout),
        "model_flag": args.model_flag,
        "init_condition_bounds": args.init_condition_bounds,
    }


def _candidate_model_info(candidate: PosthocRunCandidate, args: argparse.Namespace) -> tuple[str, int, str]:
    if candidate.config_path.exists():
        config = OmegaConf.load(candidate.config_path)
        model_flag = str(config.model.model_flag)
        init_condition_bounds = int(getattr(config.model, "init_condition_bounds", 1))
        return model_flag, init_condition_bounds, "config"
    if args.model_flag in (None, ""):
        raise FileNotFoundError(f"Config not found and --model-flag was not provided: {candidate.config_path}")
    init_condition_bounds = 1 if args.init_condition_bounds is None else int(args.init_condition_bounds)
    return str(args.model_flag), init_condition_bounds, "cli_fallback"


def _missing_candidate_result(
    *,
    candidate: PosthocRunCandidate,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return base_posthoc_run_result(
        candidate=candidate,
        checkpoint_tag=str(args.checkpoint_tag),
        eval_split=str(args.eval_split),
        ood_mode=str(args.ood_mode),
        status=candidate.status,
        missing=candidate.missing,
        extra={"batch_size": int(args.batch_size)},
    )


def _evaluate_candidate_result(
    *,
    candidate: PosthocRunCandidate,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    result_path = posthoc_run_result_path(
        output_dir=output_dir,
        run_name=candidate.run_name,
        checkpoint_tag=str(args.checkpoint_tag),
    )
    if result_path.exists() and not bool(args.force):
        existing = read_posthoc_run_result(result_path)
        if existing and existing.get("status") == "completed":
            return existing

    if not candidate.ready:
        result = _missing_candidate_result(candidate=candidate, args=args)
        write_posthoc_run_result(output_dir=output_dir, result=result)
        return result

    try:
        model_flag, init_condition_bounds, model_info_source = _candidate_model_info(candidate, args)
        pinn_model = PinnModel.load_checkpoint(str(candidate.checkpoint_path), device_preference="cpu")
        extra: dict[str, Any] = {
            "batch_size": int(args.batch_size),
            "model_flag": model_flag,
            "init_condition_bounds": init_condition_bounds,
            "model_info_source": model_info_source,
        }
        if not bool(args.no_id_eval):
            if candidate.id_eval_root is None:
                raise ValueError("ID evaluation root is required but missing.")
            id_prediction = predict_pinn_model_on_evaluation_split(
                pinn_model=pinn_model,
                checkpoint_path=candidate.checkpoint_path,
                dataset_root=candidate.id_eval_root,
                split=str(args.eval_split),
                batch_size=int(args.batch_size),
            )
            extra["id"] = evaluate_posthoc_prediction(
                prediction=id_prediction,
                kind="id",
                ood_mode=str(args.ood_mode),
                model_flag=model_flag,
                init_condition_bounds=init_condition_bounds,
            )
        if not bool(args.no_ood_eval):
            if candidate.ood_eval_root is None:
                raise ValueError("OOD evaluation root is required but missing.")
            ood_prediction = predict_pinn_model_on_evaluation_split(
                pinn_model=pinn_model,
                checkpoint_path=candidate.checkpoint_path,
                dataset_root=candidate.ood_eval_root,
                split=str(args.eval_split),
                batch_size=int(args.batch_size),
            )
            extra["ood"] = evaluate_posthoc_prediction(
                prediction=ood_prediction,
                kind="ood",
                ood_mode=str(args.ood_mode),
                model_flag=model_flag,
                init_condition_bounds=init_condition_bounds,
            )
        result = base_posthoc_run_result(
            candidate=candidate,
            checkpoint_tag=str(args.checkpoint_tag),
            eval_split=str(args.eval_split),
            ood_mode=str(args.ood_mode),
            status="completed",
            extra=extra,
        )
    except Exception as exc:
        result = base_posthoc_run_result(
            candidate=candidate,
            checkpoint_tag=str(args.checkpoint_tag),
            eval_split=str(args.eval_split),
            ood_mode=str(args.ood_mode),
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            extra={"batch_size": int(args.batch_size)},
        )
    write_posthoc_run_result(output_dir=output_dir, result=result)
    return result


def main() -> None:
    args = _parse_args()
    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.id_eval_id and args.id_eval_root:
        raise ValueError("--id-eval-id and --id-eval-root are mutually exclusive.")
    if args.ood_eval_id and args.ood_eval_root:
        raise ValueError("--ood-eval-id and --ood-eval-root are mutually exclusive.")
    allow_missing_config = args.model_flag not in (None, "")

    discover_kwargs = {
        "root": args.root,
        "checkpoint_tag": str(args.checkpoint_tag),
        "repo_root": args.repo_root,
        "require_id_eval": not bool(args.no_id_eval),
        "require_ood_eval": not bool(args.no_ood_eval),
        "id_eval_id": args.id_eval_id,
        "ood_eval_id": args.ood_eval_id,
        "id_eval_root": args.id_eval_root,
        "ood_eval_root": args.ood_eval_root,
        "require_config": not allow_missing_config,
    }
    if bool(args.dataset_generation_layout):
        candidates = discover_dataset_generation_posthoc_runs(**discover_kwargs)
    else:
        candidates = discover_posthoc_runs(**discover_kwargs)

    by_batch: dict[Path, list[PosthocRunCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_batch[candidate.batch_root].append(candidate)

    ready_count = sum(1 for candidate in candidates if candidate.ready)
    missing_count = len(candidates) - ready_count
    print(
        "[posthoc] discovered "
        f"runs={len(candidates)} ready={ready_count} missing_inputs={missing_count} batches={len(by_batch)}"
    )

    if args.dry_run:
        for batch_root, batch_candidates in sorted(by_batch.items(), key=lambda item: str(item[0])):
            batch_ready = sum(1 for candidate in batch_candidates if candidate.ready)
            print(f"[posthoc] dry_run batch={batch_root} runs={len(batch_candidates)} ready={batch_ready}")
        return

    settings = _settings(args)
    for batch_root, batch_candidates in sorted(by_batch.items(), key=lambda item: str(item[0])):
        output_dir = posthoc_output_dir(
            batch_root=batch_root,
            checkpoint_tag=str(args.checkpoint_tag),
            eval_split=str(args.eval_split),
            ood_mode=str(args.ood_mode),
            repo_root=args.repo_root,
        )
        current_results = [
            _evaluate_candidate_result(candidate=candidate, output_dir=output_dir, args=args)
            for candidate in batch_candidates
        ]
        run_results = read_posthoc_run_results(output_dir)
        if not run_results:
            run_results = current_results
        outputs = write_posthoc_batch_outputs(
            output_dir=output_dir,
            run_results=run_results,
            settings={**settings, "batch_root": str(batch_root)},
        )
        completed = sum(1 for result in run_results if result.get("status") == "completed")
        failed = sum(1 for result in run_results if result.get("status") == "failed")
        missing = sum(1 for result in run_results if result.get("status") == "missing_inputs")
        print(
            "[posthoc] wrote "
            f"batch={batch_root} completed={completed} failed={failed} missing_inputs={missing} "
            f"summary={outputs['summary_csv']}"
        )


if __name__ == "__main__":
    main()
