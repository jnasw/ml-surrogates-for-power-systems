"""Reference dataset lookup helpers for experiment launchers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_REFERENCE_INDEX_PATH = REPO_ROOT / "data" / "reference" / "index.json"


def reference_generation_message(
    reference_id: str,
    *,
    reference_index_path: Path = DEFAULT_REFERENCE_INDEX_PATH,
) -> str:
    return (
        f"Reference dataset '{reference_id}' was not found in {reference_index_path}. "
        "Generate it with:\n"
        f"python3 -m src.experiments.pipeline.run_reference_datasets --reference-id {reference_id}"
    )


def resolve_reference_dataset(
    reference_id: str,
    *,
    reference_index_path: Path = DEFAULT_REFERENCE_INDEX_PATH,
) -> dict[str, Any]:
    if not reference_index_path.exists():
        raise FileNotFoundError(
            reference_generation_message(reference_id, reference_index_path=reference_index_path)
        )
    with reference_index_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    for entry in payload.get("references", []):
        if not isinstance(entry, dict):
            continue
        if str(entry.get("reference_id")) == reference_id:
            preprocessed_root = entry.get("preprocessed_root")
            if not preprocessed_root:
                raise ValueError(f"Reference dataset '{reference_id}' has no preprocessed_root in {reference_index_path}.")
            dataset_root = Path(str(preprocessed_root)).resolve()
            if not dataset_root.exists():
                raise FileNotFoundError(
                    f"Reference dataset '{reference_id}' points to a missing preprocessed_root: {dataset_root}"
                )
            return {**entry, "preprocessed_root": str(dataset_root)}
    raise ValueError(reference_generation_message(reference_id, reference_index_path=reference_index_path))
