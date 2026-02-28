#!/usr/bin/env python3
"""Run one row from an HPO matrix by delegating to run_experiment.py."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def _parse_overrides(serialized: str) -> list[str]:
    text = (serialized or "").strip()
    if not text:
        return []
    return [chunk for chunk in text.split(";;") if chunk]


def _load_rows(matrix_path: Path) -> list[dict[str, str]]:
    with matrix_path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _select_row(rows: list[dict[str, str]], row_index: int | None, cfg_id: str | None) -> dict[str, str]:
    if row_index is None and not cfg_id:
        raise ValueError("Provide either --row-index or --cfg-id.")
    if row_index is not None and cfg_id:
        raise ValueError("Use only one of --row-index or --cfg-id.")
    if row_index is not None:
        if row_index < 0 or row_index >= len(rows):
            raise IndexError(f"row_index {row_index} out of range (0..{len(rows)-1}).")
        return rows[row_index]
    assert cfg_id is not None
    for row in rows:
        if row.get("cfg_id") == cfg_id:
            return row
    raise KeyError(f"cfg_id '{cfg_id}' not found in matrix.")


def _bool_from_str(text: str) -> bool:
    return str(text).strip().lower() in {"1", "true", "yes", "on"}


def build_command(row: dict[str, str], python_bin: str) -> list[str]:
    cmd = [
        python_bin,
        "tools/pipeline/run_experiment.py",
        "--method",
        row["method"],
        "--budget",
        row["budget"],
        "--seed",
        row["seed"],
        "--preset",
        row["preset"],
        "--experiment-id",
        row["experiment_id"],
        "--model-flag",
        row["model_flag"],
        "--run-root",
        row["run_root"],
    ]
    if _bool_from_str(row.get("skip_preprocess", "false")):
        cmd.append("--skip-preprocess")
    if _bool_from_str(row.get("skip_baseline", "false")):
        cmd.append("--skip-baseline")
    baseline_epochs = (row.get("baseline_epochs") or "").strip()
    if baseline_epochs:
        cmd.extend(["--baseline-epochs", str(int(float(baseline_epochs)))])
    for ov in _parse_overrides(row.get("stage1_overrides", "")):
        cmd.extend(["--stage1-override", ov])
    for ov in _parse_overrides(row.get("stage2_overrides", "")):
        cmd.extend(["--stage2-override", ov])
    for ov in _parse_overrides(row.get("stage3_overrides", "")):
        cmd.extend(["--stage3-override", ov])
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one matrix row for HPO.")
    parser.add_argument("--matrix", required=True, help="Path to matrix.tsv")
    parser.add_argument("--row-index", type=int, default=None, help="0-based row index in matrix.tsv")
    parser.add_argument("--cfg-id", default=None, help="Optional cfg_id instead of row index")
    parser.add_argument("--python-bin", default=sys.executable, help="Python executable for run_experiment.py")
    parser.add_argument("--dry-run", action="store_true", help="Print command and exit.")
    args = parser.parse_args()

    matrix_path = Path(args.matrix).resolve()
    rows = _load_rows(matrix_path)
    row = _select_row(rows, args.row_index, args.cfg_id)
    cmd = build_command(row, python_bin=args.python_bin)

    run_root = Path(row["run_root"]).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    with (run_root / "hpo_row.json").open("w", encoding="utf-8") as f:
        json.dump(row, f, indent=2)
    with (run_root / "hpo_command.sh").open("w", encoding="utf-8") as f:
        f.write(" ".join(cmd) + "\n")

    print(f"[hpo] cfg_id={row['cfg_id']} row_idx={row['row_idx']} run_root={run_root}")
    print("[hpo] command:")
    print(" ".join(cmd))

    if args.dry_run:
        return

    proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[2], text=True, check=False)
    status = {
        "cfg_id": row["cfg_id"],
        "row_idx": int(row["row_idx"]),
        "return_code": int(proc.returncode),
    }
    with (run_root / "hpo_status.json").open("w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()

