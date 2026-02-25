"""Run a declarative experiment campaign from YAML config."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

from omegaconf import OmegaConf


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    phase = str(getattr(cfg.campaign, "phase", "main"))
    model_flag = str(getattr(cfg.campaign, "model_flag", "SM4"))
    skip_preprocess = bool(getattr(cfg.campaign, "skip_preprocess", False))
    skip_baseline = bool(getattr(cfg.campaign, "skip_baseline", False))
    baseline_epochs = getattr(cfg.campaign, "baseline_epochs", None)
    continue_on_error = bool(getattr(cfg.campaign, "continue_on_error", True))

    axes = OmegaConf.to_container(cfg.axes, resolve=True)
    if not isinstance(axes, dict) or not axes:
        raise ValueError("campaign config must define non-empty 'axes'.")
    combos = _build_matrix(axes={k: list(v) for k, v in axes.items()})

    excludes = OmegaConf.to_container(getattr(cfg, "exclude", []), resolve=True)
    excludes = [] if excludes is None else list(excludes)
    combos = _apply_exclusions(combos, excludes)

    stage_overrides = OmegaConf.to_container(getattr(cfg, "stage_overrides", {}), resolve=True) or {}
    stage1_overrides = [str(x) for x in stage_overrides.get("stage1", [])]
    stage2_overrides = [str(x) for x in stage_overrides.get("stage2", [])]
    stage3_overrides = [str(x) for x in stage_overrides.get("stage3", [])]
    run_override_rules = list(OmegaConf.to_container(getattr(cfg, "run_overrides", []), resolve=True) or [])

    repo_root = os.path.abspath(os.path.dirname(__file__))
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
            "phase": phase,
            "model_flag": model_flag,
            "skip_preprocess": skip_preprocess,
            "skip_baseline": skip_baseline,
            "baseline_epochs": baseline_epochs,
            "continue_on_error": continue_on_error,
            "created_at_utc": _utc_now_iso(),
        },
        "runs": [],
    }

    # Optional shared-test bootstrap: generate one dataset once, then use it for
    # all campaign preprocess stages via test_split_mode=shared_dataset.
    shared_test_cfg = OmegaConf.to_container(getattr(cfg, "shared_test", {}), resolve=True) or {}
    shared_test_enabled = bool(shared_test_cfg.get("enabled", False))
    if shared_test_enabled:
        st_method = str(shared_test_cfg.get("method", "lhs_static"))
        st_budget = str(shared_test_cfg.get("budget", "b4096"))
        st_seed = str(shared_test_cfg.get("seed", "s01"))
        st_max = shared_test_cfg.get("max_trajectories", None)
        st_stage1_overrides = [str(x) for x in shared_test_cfg.get("stage1_overrides", [])]
        st_run_root = str(
            shared_test_cfg.get(
                "run_root",
                os.path.join(campaign_root, "shared_test_source"),
            )
        )

        shared_cmd = [
            sys.executable,
            "run_experiment.py",
            "--method",
            st_method,
            "--budget",
            st_budget,
            "--seed",
            st_seed,
            "--phase",
            phase,
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
        }
        with open(campaign_manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        print("[campaign] Preparing shared test source dataset...")
        if args.dry_run:
            manifest["shared_test"]["status"] = "dry_run"
        else:
            proc = subprocess.run(shared_cmd, cwd=repo_root, text=True, check=False)
            manifest["shared_test"]["return_code"] = int(proc.returncode)
            if proc.returncode != 0:
                manifest["shared_test"]["status"] = "failed"
                with open(campaign_manifest_path, "w", encoding="utf-8") as f:
                    json.dump(manifest, f, indent=2)
                raise RuntimeError("Shared-test source generation failed.")
            manifest["shared_test"]["status"] = "completed"
            shared_manifest_path = os.path.join(st_run_root, "run_manifest.json")
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

    for idx, combo in enumerate(combos):
        method = combo.get("method")
        budget = combo.get("budget")
        seed = combo.get("seed")
        if not method or not budget or not seed:
            raise ValueError("Each matrix row must define method, budget, seed.")

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
            "run_experiment.py",
            "--method",
            method,
            "--budget",
            budget,
            "--seed",
            seed,
            "--phase",
            phase,
            "--experiment-id",
            experiment_id,
            "--model-flag",
            model_flag,
        ]
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
            "seed": seed,
            "combo": combo,
            "command": cmd,
            "started_at_utc": _utc_now_iso(),
        }
        combo_text = " ".join([f"{k}={v}" for k, v in combo.items() if k not in {"method", "budget", "seed"}])
        combo_suffix = f" {combo_text}" if combo_text else ""
        print(f"[campaign] ({idx + 1}/{len(combos)}) method={method} budget={budget} seed={seed}{combo_suffix}")
        if args.dry_run:
            run_item["status"] = "dry_run"
            run_item["completed_at_utc"] = _utc_now_iso()
            manifest["runs"].append(run_item)
            continue

        proc = subprocess.run(cmd, cwd=repo_root, text=True, check=False)
        run_item["return_code"] = int(proc.returncode)
        run_item["status"] = "completed" if proc.returncode == 0 else "failed"
        run_item["completed_at_utc"] = _utc_now_iso()
        manifest["runs"].append(run_item)

        with open(campaign_manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        if proc.returncode != 0 and not continue_on_error:
            break

    with open(campaign_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[campaign] Manifest: {campaign_manifest_path}")


if __name__ == "__main__":
    main()
