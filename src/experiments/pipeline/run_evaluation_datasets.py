"""Generate persistent supervised ID/OOD evaluation datasets.

This runner intentionally reuses the existing dataset-generation and
preprocessing pipeline. It only adds a small index/verification layer for
evaluation datasets under ``data/evaluation``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.data.contracts.data_contract import H5_COLLOCATION_SUFFIX, H5_DATA_SUFFIX, H5_FILE_SUFFIX, H5_INIT_SUFFIX, TEST_SPLIT
from src.experiments.pipeline.helpers.manifest import utc_now_iso


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "evaluation"
MODEL_FLAGS = ("SM4", "SM6", "SM_AVR_GOV")


@dataclass(frozen=True)
class EvaluationDatasetSpec:
    evaluation_id: str
    kind: str
    model_flag: str
    method: str
    budget: str
    dataset_seed: str
    distribution: str
    init_condition_bounds: int = 1
    preset: str = "main"
    stage1_overrides: tuple[str, ...] = ()
    stage2_overrides: tuple[str, ...] = ()
    notes: str | None = None


def _evaluation_specs() -> list[EvaluationDatasetSpec]:
    specs: list[EvaluationDatasetSpec] = []
    for model_flag in MODEL_FLAGS:
        specs.append(
            EvaluationDatasetSpec(
                evaluation_id=f"id_{model_flag}_lhs_b512_ds01",
                kind="id",
                model_flag=model_flag,
                method="lhs_static",
                budget="b512",
                dataset_seed="ds01",
                distribution="training_distribution",
            )
        )
        specs.append(
            EvaluationDatasetSpec(
                evaluation_id=f"ood_{model_flag}_wide_ic_b512_ds01",
                kind="ood",
                model_flag=model_flag,
                method="lhs_static",
                budget="b512",
                dataset_seed="ds01",
                distribution="wide_ic",
                init_condition_bounds=2,
                stage1_overrides=("model.init_condition_bounds=2",),
            )
        )
        specs.append(
            EvaluationDatasetSpec(
                evaluation_id=f"id_{model_flag}_lhs_b4096_eval01",
                kind="id",
                model_flag=model_flag,
                method="lhs_static",
                budget="b4096",
                dataset_seed="eval01",
                distribution="training_distribution",
                notes="Larger independent evaluation set for dataset-generation comparisons.",
            )
        )
        specs.append(
            EvaluationDatasetSpec(
                evaluation_id=f"ood_{model_flag}_wide_ic_b4096_eval01",
                kind="ood",
                model_flag=model_flag,
                method="lhs_static",
                budget="b4096",
                dataset_seed="eval01",
                distribution="wide_ic",
                init_condition_bounds=2,
                stage1_overrides=("model.init_condition_bounds=2",),
                notes="Larger independent OOD evaluation set for dataset-generation comparisons.",
            )
        )
    return specs


def _iter_h5_files(split_dir: Path, suffix: str) -> list[Path]:
    if not split_dir.exists():
        return []
    paths: list[Path] = []
    for path in sorted(split_dir.iterdir()):
        if not path.is_file() or path.suffix != H5_FILE_SUFFIX:
            continue
        name = path.name
        if suffix == H5_DATA_SUFFIX:
            if H5_DATA_SUFFIX not in name or H5_COLLOCATION_SUFFIX in name or H5_INIT_SUFFIX in name:
                continue
        elif suffix not in name:
            continue
        paths.append(path)
    return paths


def _run_root_for_spec(spec: EvaluationDatasetSpec, output_root: Path) -> Path:
    return output_root / spec.kind / spec.model_flag / spec.evaluation_id


def _build_command(*, python_bin: str, spec: EvaluationDatasetSpec, run_root: Path) -> list[str]:
    command = [
        python_bin,
        "-m",
        "src.experiments.pipeline.run_dataset_pipeline",
        "--method",
        spec.method,
        "--budget",
        spec.budget,
        "--dataset-seed",
        spec.dataset_seed,
        "--preset",
        spec.preset,
        "--experiment-id",
        f"evaluation_{spec.kind}_{spec.model_flag.lower()}",
        "--model-flag",
        spec.model_flag,
        "--run-root",
        str(run_root),
        "--skip-baseline",
    ]
    for override in spec.stage1_overrides:
        command.extend(["--stage1-override", override])
    for override in spec.stage2_overrides:
        command.extend(["--stage2-override", override])
    return command


def _verify_evaluation_run(*, run_root: Path, evaluation_id: str) -> dict[str, Any]:
    manifest_path = run_root / "dataset_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"[{evaluation_id}] Missing dataset manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = dict(manifest.get("artifacts", {}) or {})
    preprocessed_root_raw = artifacts.get("preprocessed_root") or artifacts.get("dataset_root")
    if not preprocessed_root_raw:
        raise ValueError(f"[{evaluation_id}] Manifest does not contain artifacts.preprocessed_root or artifacts.dataset_root.")

    dataset_root = Path(str(preprocessed_root_raw)).resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"[{evaluation_id}] Preprocessed dataset root not found: {dataset_root}")

    test_root = dataset_root / TEST_SPLIT
    if not test_root.exists():
        raise FileNotFoundError(f"[{evaluation_id}] Missing test split directory: {test_root}")
    test_supervised = _iter_h5_files(test_root, H5_DATA_SUFFIX)
    if not test_supervised:
        raise ValueError(f"[{evaluation_id}] No supervised test HDF5 files found in: {test_root}")

    return {
        "manifest": manifest,
        "manifest_path": str(manifest_path.resolve()),
        "dataset_root": str(dataset_root),
        "test_root": str(test_root.resolve()),
        "status": "verified",
    }


def _read_index(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"updated_at_utc": None, "evaluations": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {"updated_at_utc": None, "evaluations": []}
    evaluations = payload.get("evaluations")
    if not isinstance(evaluations, list):
        evaluations = []
    return {"updated_at_utc": payload.get("updated_at_utc"), "evaluations": evaluations}


def _write_index(path: Path, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at_utc": utc_now_iso(),
        "evaluations": sorted(entries, key=lambda item: str(item.get("evaluation_id", ""))),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _index_entry(*, spec: EvaluationDatasetSpec, run_root: Path, verified: dict[str, Any]) -> dict[str, Any]:
    manifest = dict(verified["manifest"])
    entry = {
        "evaluation_id": spec.evaluation_id,
        "kind": spec.kind,
        "model_flag": spec.model_flag,
        "method": spec.method,
        "budget": spec.budget,
        "dataset_seed": spec.dataset_seed,
        "distribution": spec.distribution,
        "init_condition_bounds": int(spec.init_condition_bounds),
        "run_root": str(run_root.resolve()),
        "manifest_path": str(verified["manifest_path"]),
        "dataset_root": str(verified["dataset_root"]),
        "test_root": str(verified["test_root"]),
        "status": str(verified["status"]),
        "created_at_utc": manifest.get("created_at_utc"),
        "verified_at_utc": utc_now_iso(),
    }
    if spec.notes:
        entry["notes"] = spec.notes
    return entry


def _selected_specs(args: argparse.Namespace) -> list[EvaluationDatasetSpec]:
    all_specs = _evaluation_specs()
    specs_by_id = {spec.evaluation_id: spec for spec in all_specs}

    if args.evaluation_id:
        unknown = [evaluation_id for evaluation_id in args.evaluation_id if evaluation_id not in specs_by_id]
        if unknown:
            known = ", ".join(sorted(specs_by_id))
            raise ValueError(f"Unknown evaluation id(s): {', '.join(unknown)}. Known evaluation ids: {known}")
        specs = [specs_by_id[evaluation_id] for evaluation_id in args.evaluation_id]
    else:
        specs = list(all_specs)

    if args.kind:
        specs = [spec for spec in specs if spec.kind == args.kind]
    if args.model_flag and args.model_flag != "all":
        specs = [spec for spec in specs if spec.model_flag == args.model_flag]

    if not specs:
        raise ValueError("No evaluation datasets match the current filters.")
    return specs


def _print_command(evaluation_id: str, command: list[str], notes: str | None) -> None:
    print(f"[evaluation-datasets] {evaluation_id}")
    if notes:
        print(f"[evaluation-datasets] note: {notes}")
    print(" ".join(command))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate reusable supervised ID/OOD evaluation datasets via the dataset pipeline."
    )
    parser.add_argument(
        "--evaluation-id",
        action="append",
        default=[],
        help="Optional repeatable evaluation-id selector, e.g. id_SM4_lhs_b512_ds01.",
    )
    parser.add_argument("--kind", choices=("id", "ood"), default=None, help="Optional evaluation kind filter.")
    parser.add_argument("--model-flag", choices=(*MODEL_FLAGS, "all"), default="all")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-rebuild", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.force_rebuild and args.skip_existing:
        args.skip_existing = False

    output_root = Path(args.output_root).resolve()
    index_path = output_root / "index.json"
    index_payload = _read_index(index_path)
    entries_by_id = {
        str(entry.get("evaluation_id")): dict(entry)
        for entry in index_payload.get("evaluations", [])
        if isinstance(entry, dict) and entry.get("evaluation_id")
    }

    try:
        specs = _selected_specs(args)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"[evaluation-datasets] selected={len(specs)} output_root={output_root}")

    for spec in specs:
        run_root = _run_root_for_spec(spec, output_root)
        command = _build_command(python_bin=str(args.python_bin), spec=spec, run_root=run_root)

        if args.dry_run:
            _print_command(spec.evaluation_id, command, spec.notes)
            continue

        if run_root.exists():
            if args.force_rebuild:
                print(f"[evaluation-datasets] removing existing run root for force rebuild: {run_root}")
                shutil.rmtree(run_root)
            else:
                try:
                    verified = _verify_evaluation_run(run_root=run_root, evaluation_id=spec.evaluation_id)
                except Exception as exc:
                    raise RuntimeError(
                        f"[{spec.evaluation_id}] Target run root exists but verification failed: {run_root}. "
                        "Use --force-rebuild to replace it."
                    ) from exc
                if verified is not None and args.skip_existing:
                    print(f"[evaluation-datasets] skipping existing verified dataset: {spec.evaluation_id}")
                    entries_by_id[spec.evaluation_id] = _index_entry(spec=spec, run_root=run_root, verified=verified)
                    continue
                raise RuntimeError(
                    f"[{spec.evaluation_id}] Target run root already exists: {run_root}. "
                    "Use --skip-existing to accept verified datasets or --force-rebuild to replace them."
                )

        _print_command(spec.evaluation_id, command, spec.notes)
        proc = subprocess.run(command, cwd=REPO_ROOT, check=False)
        if proc.returncode != 0:
            raise SystemExit(proc.returncode)

        verified = _verify_evaluation_run(run_root=run_root, evaluation_id=spec.evaluation_id)
        entries_by_id[spec.evaluation_id] = _index_entry(spec=spec, run_root=run_root, verified=verified)
        _write_index(index_path, list(entries_by_id.values()))
        print(f"[evaluation-datasets] verified {spec.evaluation_id}")

    if not args.dry_run:
        _write_index(index_path, list(entries_by_id.values()))
        print(f"[evaluation-datasets] index={index_path}")


if __name__ == "__main__":
    main()
