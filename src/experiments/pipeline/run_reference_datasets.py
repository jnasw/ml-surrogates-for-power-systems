"""Generate persistent reusable reference datasets via the existing dataset pipeline."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.data.contracts.data_contract import (
    H5_COLLOCATION_SUFFIX,
    H5_DATA_SUFFIX,
    H5_FILE_SUFFIX,
    H5_INIT_SUFFIX,
    TEST_SPLIT,
    TRAIN_SPLIT,
)
from src.experiments.pipeline.helpers.manifest import utc_now_iso


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "reference"
DEFAULT_INDEX_PATH = DEFAULT_OUTPUT_ROOT / "index.json"
MODEL_FLAGS = ("SM4", "SM6", "SM_AVR_GOV")
METHODS = ("lhs_static", "qbc_deep_ensemble", "marker_directed", "qbc_marker_hybrid")
METHOD_ID_ALIASES = {
    "lhs_static": "lhs",
    "qbc_deep_ensemble": "qbc",
    "marker_directed": "marker",
    "qbc_marker_hybrid": "hybrid",
}


@dataclass(frozen=True)
class ReferenceDatasetSpec:
    suite: str
    model_flag: str
    method: str
    budget: str
    dataset_seed: str
    preset: str
    stage1_overrides: tuple[str, ...] = ()
    stage2_overrides: tuple[str, ...] = ()
    default_for_training_experiments: bool = False

    @property
    def reference_id(self) -> str:
        method_id = METHOD_ID_ALIASES.get(self.method, self.method)
        return f"{self.suite}_{self.model_flag}_{method_id}_{self.budget}_{self.dataset_seed}"

    @property
    def legacy_reference_id(self) -> str:
        return f"{self.suite}_{self.model_flag}_{self.method}_{self.budget}_{self.dataset_seed}"


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


def _verify_reference_run(*, run_root: Path, reference_id: str) -> dict[str, Any]:
    manifest_path = run_root / "dataset_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"[{reference_id}] Missing dataset manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = dict(manifest.get("artifacts", {}))
    preprocessed_root_raw = artifacts.get("preprocessed_root")
    dataset_root_raw = artifacts.get("dataset_root")
    dataset_root = Path(str(preprocessed_root_raw or dataset_root_raw or "")).resolve()
    if not str(dataset_root):
        raise ValueError(f"[{reference_id}] dataset_manifest.json does not contain artifacts.preprocessed_root or artifacts.dataset_root.")
    if not dataset_root.exists():
        raise FileNotFoundError(f"[{reference_id}] Preprocessed dataset root not found: {dataset_root}")

    train_dir = dataset_root / TRAIN_SPLIT
    test_dir = dataset_root / TEST_SPLIT
    if not train_dir.exists():
        raise FileNotFoundError(f"[{reference_id}] Missing train split directory: {train_dir}")
    if not test_dir.exists():
        raise FileNotFoundError(f"[{reference_id}] Missing test split directory: {test_dir}")

    train_supervised = _iter_h5_files(train_dir, H5_DATA_SUFFIX)
    test_supervised = _iter_h5_files(test_dir, H5_DATA_SUFFIX)
    train_collocation = _iter_h5_files(train_dir, H5_COLLOCATION_SUFFIX)
    train_init = _iter_h5_files(train_dir, H5_INIT_SUFFIX)
    if not train_supervised:
        raise ValueError(f"[{reference_id}] No supervised train HDF5 files found in: {train_dir}")
    if not test_supervised:
        raise ValueError(f"[{reference_id}] No supervised test HDF5 files found in: {test_dir}")
    if not train_collocation:
        raise ValueError(f"[{reference_id}] No collocation HDF5 files found in: {train_dir}")
    if not train_init:
        raise ValueError(f"[{reference_id}] No init HDF5 files found in: {train_dir}")

    return {
        "manifest": manifest,
        "manifest_path": str(manifest_path.resolve()),
        "dataset_root": str(dataset_root),
        "raw_root": str((dataset_root / "raw").resolve()),
        "preprocessed_root": str(dataset_root),
        "status": "verified",
    }


def _reference_specs() -> list[ReferenceDatasetSpec]:
    specs: list[ReferenceDatasetSpec] = []
    for model_flag in MODEL_FLAGS:
        specs.append(
            ReferenceDatasetSpec(
                suite="smoke",
                model_flag=model_flag,
                method="lhs_static",
                budget="b256",
                dataset_seed="ds01",
                preset="smoke",
                stage1_overrides=("model.ic_num_samples=32",),
                stage2_overrides=("dataset.validation_flag=false", "time=0.05", "num_of_points=20"),
            )
        )
        specs.append(
            ReferenceDatasetSpec(
                suite="dev",
                model_flag=model_flag,
                method="lhs_static",
                budget="b512",
                dataset_seed="ds01",
                preset="main",
            )
        )
        for method in METHODS:
            specs.append(
                ReferenceDatasetSpec(
                    suite="main",
                    model_flag=model_flag,
                    method=method,
                    budget="b512",
                    dataset_seed="ds01",
                    preset="main",
                    default_for_training_experiments=(method == "qbc_deep_ensemble"),
                )
            )
        specs.append(
            ReferenceDatasetSpec(
                suite="main",
                model_flag=model_flag,
                method="qbc_deep_ensemble",
                budget="b1024",
                dataset_seed="ds01",
                preset="main",
            )
        )
        if model_flag == "SM4":
            specs.append(
                ReferenceDatasetSpec(
                    suite="main",
                    model_flag=model_flag,
                    method="lhs_static",
                    budget="b4096",
                    dataset_seed="ds01",
                    preset="main",
                )
            )
    return specs


def _selected_model_flags(raw_values: list[str]) -> set[str]:
    if not raw_values or "all" in raw_values:
        return set(MODEL_FLAGS)
    return {value for value in raw_values if value in MODEL_FLAGS}


def _selected_specs(args: argparse.Namespace) -> list[ReferenceDatasetSpec]:
    model_flags = _selected_model_flags(list(args.model_flag or []))
    selected_suites = {"smoke", "dev", "main"} if args.suite == "all" else {args.suite}
    selected_reference_ids = set(args.reference_id or [])

    specs = [
        spec
        for spec in _reference_specs()
        if spec.suite in selected_suites and spec.model_flag in model_flags
    ]
    if selected_reference_ids:
        specs = [
            spec
            for spec in specs
            if spec.reference_id in selected_reference_ids or spec.legacy_reference_id in selected_reference_ids
        ]
    if not specs:
        raise ValueError("No reference datasets match the current filters.")
    return specs


def _run_root_for_spec(spec: ReferenceDatasetSpec, output_root: Path) -> Path:
    return output_root / spec.suite / spec.model_flag / spec.method / spec.budget / spec.dataset_seed


def _build_command(*, python_bin: str, spec: ReferenceDatasetSpec, run_root: Path) -> list[str]:
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
        f"reference_{spec.suite}_{spec.model_flag.lower()}",
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


def _read_index(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"updated_at_utc": None, "references": []}
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        return {"updated_at_utc": None, "references": []}
    references = payload.get("references")
    if not isinstance(references, list):
        references = []
    return {"updated_at_utc": payload.get("updated_at_utc"), "references": references}


def _write_index(path: Path, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at_utc": utc_now_iso(),
        "references": sorted(entries, key=lambda item: str(item.get("reference_id", ""))),
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _index_entry(*, spec: ReferenceDatasetSpec, run_root: Path, verified: dict[str, Any]) -> dict[str, Any]:
    manifest = dict(verified["manifest"])
    return {
        "reference_id": spec.reference_id,
        "suite": spec.suite,
        "model_flag": spec.model_flag,
        "method": spec.method,
        "budget": spec.budget,
        "dataset_seed": spec.dataset_seed,
        "run_root": str(run_root.resolve()),
        "manifest_path": str(verified["manifest_path"]),
        "dataset_root": str(verified["dataset_root"]),
        "raw_root": str(verified["raw_root"]),
        "preprocessed_root": str(verified["preprocessed_root"]),
        "created_at_utc": manifest.get("created_at_utc"),
        "updated_at_utc": manifest.get("updated_at_utc"),
        "status": str(verified["status"]),
        "default_for_training_experiments": bool(spec.default_for_training_experiments),
    }


def _print_command(reference_id: str, command: list[str]) -> None:
    print(f"[reference-datasets] {reference_id}")
    print(" ".join(command))


def _maybe_existing_verified(*, run_root: Path, reference_id: str) -> dict[str, Any] | None:
    if not run_root.exists():
        return None
    return _verify_reference_run(run_root=run_root, reference_id=reference_id)


def _remove_stale_index_entries(
    entries_by_id: dict[str, dict[str, Any]],
    *,
    spec: ReferenceDatasetSpec,
    run_root: Path,
) -> None:
    resolved_run_root = str(run_root.resolve())
    stale_ids = {
        entry_id
        for entry_id, entry in entries_by_id.items()
        if entry_id in {spec.reference_id, spec.legacy_reference_id}
        or str(entry.get("run_root", "")) == resolved_run_root
    }
    for entry_id in stale_ids:
        entries_by_id.pop(entry_id, None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate persistent reusable reference datasets via the dataset pipeline.")
    parser.add_argument("--suite", choices=("smoke", "dev", "main", "all"), default="all")
    parser.add_argument(
        "--model-flag",
        action="append",
        default=[],
        choices=(*MODEL_FLAGS, "all"),
        help="Repeatable model selector. Default: all.",
    )
    parser.add_argument(
        "--reference-id",
        action="append",
        default=[],
        help="Optional repeatable stable reference-id selector.",
    )
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
        str(entry.get("reference_id")): dict(entry)
        for entry in index_payload.get("references", [])
        if isinstance(entry, dict) and entry.get("reference_id")
    }

    specs = _selected_specs(args)
    print(f"[reference-datasets] selected={len(specs)} output_root={output_root}")

    for spec in specs:
        run_root = _run_root_for_spec(spec, output_root)
        command = _build_command(python_bin=str(args.python_bin), spec=spec, run_root=run_root)

        if args.dry_run:
            _print_command(spec.reference_id, command)
            continue

        if run_root.exists():
            if args.force_rebuild:
                print(f"[reference-datasets] removing existing run root for force rebuild: {run_root}")
                shutil.rmtree(run_root)
            else:
                try:
                    verified = _maybe_existing_verified(run_root=run_root, reference_id=spec.reference_id)
                except Exception as exc:
                    raise RuntimeError(
                        f"[{spec.reference_id}] Target run root exists but verification failed: {run_root}. "
                        "Use --force-rebuild to replace it."
                    ) from exc
                if verified is not None and args.skip_existing:
                    print(f"[reference-datasets] skipping existing verified dataset: {spec.reference_id}")
                    _remove_stale_index_entries(entries_by_id, spec=spec, run_root=run_root)
                    entries_by_id[spec.reference_id] = _index_entry(spec=spec, run_root=run_root, verified=verified)
                    continue
                raise RuntimeError(
                    f"[{spec.reference_id}] Target run root already exists: {run_root}. "
                    "Use --skip-existing to accept verified datasets or --force-rebuild to replace them."
                )

        _print_command(spec.reference_id, command)
        proc = subprocess.run(command, cwd=REPO_ROOT, check=False)
        if proc.returncode != 0:
            raise SystemExit(proc.returncode)

        verified = _verify_reference_run(run_root=run_root, reference_id=spec.reference_id)
        _remove_stale_index_entries(entries_by_id, spec=spec, run_root=run_root)
        entries_by_id[spec.reference_id] = _index_entry(spec=spec, run_root=run_root, verified=verified)
        _write_index(index_path, list(entries_by_id.values()))
        print(f"[reference-datasets] verified {spec.reference_id}")

    if not args.dry_run:
        _write_index(index_path, list(entries_by_id.values()))
        print(f"[reference-datasets] index={index_path}")


if __name__ == "__main__":
    main()
