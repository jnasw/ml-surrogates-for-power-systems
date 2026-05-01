"""Export dataset-, baseline-, and round-level tables from experiment manifests."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.pipeline.helpers.experiment_manifest_tables import (
    load_baseline_table,
    load_dataset_table,
    load_round_table,
)


def _write_csv(rows: list[dict[str, Any]], out_path: str) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)
    p = Path(out_path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export experiment CSV tables.")
    parser.add_argument("--root", default="outputs/experiments", help="Manifest root.")
    parser.add_argument("--out-dir", default="outputs/dashboard", help="Output directory for csv files.")
    args = parser.parse_args()

    dataset_rows = load_dataset_table(args.root)
    baseline_rows = load_baseline_table(args.root)
    round_rows = load_round_table(args.root)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_csv = str(out_dir / "dataset_table.csv")
    baseline_csv = str(out_dir / "baseline_table.csv")
    round_csv = str(out_dir / "round_table.csv")
    _write_csv(dataset_rows, dataset_csv)
    _write_csv(baseline_rows, baseline_csv)
    _write_csv(round_rows, round_csv)
    print(f"Wrote dataset table ({len(dataset_rows)} rows): {dataset_csv}")
    print(f"Wrote baseline table ({len(baseline_rows)} rows): {baseline_csv}")
    print(f"Wrote round table ({len(round_rows)} rows): {round_csv}")


if __name__ == "__main__":
    main()
