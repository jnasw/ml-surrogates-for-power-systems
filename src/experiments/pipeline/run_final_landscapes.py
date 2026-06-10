"""Run loss-landscape post-processing for final experiment checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from src.experiments.pipeline.helpers.launch_utils import parse_csv_list, run_logged_command, tag_stamp
from src.experiments.pipeline.helpers.manifest import save_manifest, utc_now_iso


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT_TAGS = ("best", "epoch_100pct")
DEFAULT_DN_EXTRA_CHECKPOINT_TAGS = ("epoch_060pct",)


def _parse_optional_csv(raw: str | None) -> list[str] | None:
    if raw in (None, ""):
        return None
    values = [item.strip() for item in parse_csv_list(str(raw)) if item.strip()]
    return values or None


def _parse_csv_or_default(raw: str | None, default: tuple[str, ...]) -> list[str]:
    values = _parse_optional_csv(raw)
    return list(default if values is None else values)


def _read_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected manifest object in {path}")
    return payload


def _parse_manifest_paths(raw_values: list[str]) -> list[Path]:
    paths: list[Path] = []
    for raw in raw_values:
        for item in parse_csv_list(str(raw)):
            value = item.strip()
            if value:
                paths.append(Path(value).resolve())
    if not paths:
        raise ValueError("At least one --run-manifest path is required.")
    return paths


def _checkpoint_path(run_dir: Path, tag: str) -> Path:
    return run_dir / "checkpoints" / f"{tag}.pt"


def _selected_runs(
    runs: list[dict[str, Any]],
    *,
    models: list[str] | None,
    strategies: list[str] | None,
    seed_labels: list[str] | None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for run in runs:
        if str(run.get("status")) not in {"completed", "dry_run"}:
            continue
        if models is not None and str(run.get("model_flag")) not in models:
            continue
        if strategies is not None and str(run.get("strategy")) not in strategies:
            continue
        if seed_labels is not None and str(run.get("seed_label")) not in seed_labels:
            continue
        selected.append(run)
    return selected


def _weight_source_for_run(strategy: str, checkpoint_tag: str, requested: str) -> str:
    if requested != "auto":
        return requested
    if checkpoint_tag == "init":
        return "config"
    return "checkpoint" if strategy == "dn_adam3000_ssbroyden2000" else "config"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-manifest",
        action="append",
        default=[],
        help=(
            "Final experiment run_manifest.json. Repeat this option or pass "
            "comma-separated paths to combine fragmented retry batches."
        ),
    )
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--experiment-tag", default=None)
    parser.add_argument("--models", default=None)
    parser.add_argument("--strategies", default=None)
    parser.add_argument("--seed-labels", default=None)
    parser.add_argument("--checkpoint-tags", default=",".join(DEFAULT_CHECKPOINT_TAGS))
    parser.add_argument(
        "--include-dn-boundary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also evaluate epoch_060pct for DN runs when present.",
    )
    parser.add_argument("--grid", choices=["1d", "2d"], default="2d")
    parser.add_argument("--resolution", type=int, default=21)
    parser.add_argument("--alpha-min", type=float, default=-1.0)
    parser.add_argument("--alpha-max", type=float, default=1.0)
    parser.add_argument("--beta-min", type=float, default=-1.0)
    parser.add_argument("--beta-max", type=float, default=1.0)
    parser.add_argument("--split", default="train")
    parser.add_argument("--supervised-rows", type=int, default=512)
    parser.add_argument("--collocation-rows", type=int, default=4096)
    parser.add_argument("--init-rows", type=int, default=512)
    parser.add_argument("--analysis-seed", type=int, default=0)
    parser.add_argument("--direction-seed", type=int, default=0)
    parser.add_argument("--normalization", default="filter")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--weight-source",
        choices=["auto", "config", "checkpoint"],
        default="auto",
        help="auto uses checkpoint weights for DN runs and config weights otherwise.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-on-missing", action=argparse.BooleanOptionalAction, default=False)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    source_manifest_paths = _parse_manifest_paths(list(args.run_manifest))
    source_manifests = [(path, _read_manifest(path)) for path in source_manifest_paths]
    source_run_roots = [
        Path(str(manifest.get("run_root", path.parent))).resolve()
        for path, manifest in source_manifests
    ]
    source_run_root = source_run_roots[0]
    stamp = args.experiment_tag or tag_stamp()
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else (source_run_root / "loss_landscapes" / stamp).resolve()
    )
    output_root.mkdir(parents=True, exist_ok=True)

    models = _parse_optional_csv(args.models)
    strategies = _parse_optional_csv(args.strategies)
    seed_labels = _parse_optional_csv(args.seed_labels)
    checkpoint_tags = _parse_csv_or_default(args.checkpoint_tags, DEFAULT_CHECKPOINT_TAGS)
    runs: list[dict[str, Any]] = []
    for source_manifest_path, source_manifest in source_manifests:
        selected = _selected_runs(
            [dict(run) for run in source_manifest.get("runs", []) if isinstance(run, dict)],
            models=models,
            strategies=strategies,
            seed_labels=seed_labels,
        )
        for run in selected:
            run["source_run_manifest"] = str(source_manifest_path)
            runs.append(run)

    landscape_manifest: dict[str, Any] = {
        "created_at_utc": utc_now_iso(),
        "source_run_manifests": [str(path) for path in source_manifest_paths],
        "source_run_roots": [str(path) for path in source_run_roots],
        "output_root": str(output_root),
        "filters": {
            "models": models,
            "strategies": strategies,
            "seed_labels": seed_labels,
            "checkpoint_tags": checkpoint_tags,
            "include_dn_boundary": bool(args.include_dn_boundary),
            "weight_source": str(args.weight_source),
        },
        "settings": {
            "grid": args.grid,
            "resolution": int(args.resolution),
            "alpha_min": float(args.alpha_min),
            "alpha_max": float(args.alpha_max),
            "beta_min": float(args.beta_min),
            "beta_max": float(args.beta_max),
            "split": args.split,
            "supervised_rows": int(args.supervised_rows),
            "collocation_rows": int(args.collocation_rows),
            "init_rows": int(args.init_rows),
            "analysis_seed": int(args.analysis_seed),
            "direction_seed": int(args.direction_seed),
            "normalization": args.normalization,
            "device": args.device,
        },
        "runs": [],
    }
    landscape_manifest_path = output_root / "landscape_manifest.json"
    save_manifest(str(landscape_manifest_path), landscape_manifest)

    for run in runs:
        run_dir = Path(str(run.get("run_dir", ""))).resolve()
        strategy = str(run.get("strategy"))
        tags = list(checkpoint_tags)
        if bool(args.include_dn_boundary) and strategy == "dn_adam3000_ssbroyden2000":
            for tag in DEFAULT_DN_EXTRA_CHECKPOINT_TAGS:
                if tag not in tags:
                    tags.append(tag)
        for checkpoint_tag in tags:
            checkpoint = _checkpoint_path(run_dir, checkpoint_tag)
            landscape_dir = output_root / str(run.get("model_flag")) / strategy / str(run.get("seed_label")) / checkpoint_tag
            weight_source = _weight_source_for_run(strategy, checkpoint_tag, str(args.weight_source))
            entry = {
                "source_run_name": run.get("run_name"),
                "source_run_manifest": run.get("source_run_manifest"),
                "model_flag": run.get("model_flag"),
                "strategy": strategy,
                "seed_label": run.get("seed_label"),
                "checkpoint_tag": checkpoint_tag,
                "checkpoint": str(checkpoint),
                "output_dir": str(landscape_dir),
                "weight_source": weight_source,
            }
            if not checkpoint.exists() and not args.dry_run:
                entry["status"] = "missing"
                landscape_manifest["runs"].append(entry)
                save_manifest(str(landscape_manifest_path), landscape_manifest)
                message = f"[final-landscapes] missing checkpoint: {checkpoint}"
                if args.fail_on_missing:
                    raise SystemExit(message)
                print(message)
                continue

            command = [
                args.python_bin,
                "-m",
                "src.experiments.pipeline.run_loss_landscape",
                "--checkpoint",
                str(checkpoint),
                "--output-dir",
                str(landscape_dir),
                "--grid",
                str(args.grid),
                "--resolution",
                str(int(args.resolution)),
                "--alpha-min",
                str(float(args.alpha_min)),
                "--alpha-max",
                str(float(args.alpha_max)),
                "--beta-min",
                str(float(args.beta_min)),
                "--beta-max",
                str(float(args.beta_max)),
                "--split",
                str(args.split),
                "--supervised-rows",
                str(int(args.supervised_rows)),
                "--collocation-rows",
                str(int(args.collocation_rows)),
                "--init-rows",
                str(int(args.init_rows)),
                "--analysis-seed",
                str(int(args.analysis_seed)),
                "--direction-seed",
                str(int(args.direction_seed)),
                "--normalization",
                str(args.normalization),
                "--device",
                str(args.device),
                "--weight-source",
                weight_source,
            ]
            if args.dry_run:
                command.append("--dry-run")
            entry["command"] = command
            entry["status"] = "running" if not args.dry_run else "dry_run"
            landscape_manifest["runs"].append(entry)
            save_manifest(str(landscape_manifest_path), landscape_manifest)
            result = run_logged_command(
                label="final-landscapes",
                command=command,
                log_path=output_root / "logs" / f"{run.get('run_name')}_{checkpoint_tag}.log",
                dry_run=bool(args.dry_run),
                cwd=REPO_ROOT,
            )
            entry.update({
                "status": str(result["status"]),
                "return_code": int(result["return_code"]),
                "completed_at_utc": str(result["completed_at_utc"]),
                "elapsed_seconds": float(result["elapsed_seconds"]),
            })
            save_manifest(str(landscape_manifest_path), landscape_manifest)
            if int(result["return_code"]) != 0:
                raise SystemExit(int(result["return_code"]))

    print(f"[final-landscapes] landscape_manifest={landscape_manifest_path}")


if __name__ == "__main__":
    main()
