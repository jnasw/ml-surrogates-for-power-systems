"""External evaluation dataset resolution helpers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_EVALUATION_INDEX_PATH = REPO_ROOT / "data" / "evaluation" / "index.json"


def evaluation_generation_message(
    evaluation_id: str,
    *,
    evaluation_index_path: Path = DEFAULT_EVALUATION_INDEX_PATH,
) -> str:
    return (
        f"Evaluation dataset '{evaluation_id}' was not found in {evaluation_index_path}. "
        "Generate it with:\n"
        f"python3 -m src.experiments.pipeline.run_evaluation_datasets --evaluation-id {evaluation_id}"
    )


def resolve_evaluation_dataset(
    evaluation_id: str,
    *,
    evaluation_index_path: Path = DEFAULT_EVALUATION_INDEX_PATH,
) -> dict[str, Any]:
    if not evaluation_index_path.exists():
        raise FileNotFoundError(
            evaluation_generation_message(evaluation_id, evaluation_index_path=evaluation_index_path)
        )
    with evaluation_index_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    for entry in payload.get("evaluations", []):
        if not isinstance(entry, dict):
            continue
        if str(entry.get("evaluation_id")) == evaluation_id:
            dataset_root_raw = entry.get("dataset_root")
            if not dataset_root_raw:
                raise ValueError(f"Evaluation dataset '{evaluation_id}' has no dataset_root in {evaluation_index_path}.")
            dataset_root = Path(str(dataset_root_raw)).resolve()
            if not dataset_root.exists():
                raise FileNotFoundError(
                    f"Evaluation dataset '{evaluation_id}' points to a missing dataset_root: {dataset_root}"
                )
            return {**entry, "dataset_root": str(dataset_root)}
    raise ValueError(evaluation_generation_message(evaluation_id, evaluation_index_path=evaluation_index_path))


def resolve_eval_inputs(
    args: argparse.Namespace,
    *,
    default_ood_eval_id: str | None,
    evaluation_index_path: Path = DEFAULT_EVALUATION_INDEX_PATH,
) -> dict[str, str | None]:
    resolved: dict[str, str | None] = {
        "id_eval_id": None,
        "id_eval_root": None,
        "ood_eval_id": None,
        "ood_eval_root": None,
    }
    for kind in ("id", "ood"):
        eval_id = getattr(args, f"{kind}_eval_id")
        eval_root = getattr(args, f"{kind}_eval_root")
        if kind == "ood" and bool(getattr(args, "no_ood_eval", False)):
            if eval_id or eval_root:
                raise ValueError("--no-ood-eval is mutually exclusive with --ood-eval-id and --ood-eval-root.")
            continue
        if kind == "ood" and not eval_id and not eval_root and default_ood_eval_id:
            eval_id = default_ood_eval_id
        if eval_id and eval_root:
            raise ValueError(f"--{kind}-eval-id and --{kind}-eval-root are mutually exclusive.")
        if eval_id:
            entry = resolve_evaluation_dataset(str(eval_id), evaluation_index_path=evaluation_index_path)
            resolved[f"{kind}_eval_id"] = str(eval_id)
            resolved[f"{kind}_eval_root"] = str(entry["dataset_root"])
        elif eval_root:
            resolved[f"{kind}_eval_root"] = str(Path(str(eval_root)).resolve())
    return resolved
