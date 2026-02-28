"""Export dashboard-ready run/round tables from manifests."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.dashboard_data import load_round_table, load_run_table


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
    parser = argparse.ArgumentParser(description="Export dashboard-ready CSV tables.")
    parser.add_argument("--root", default="outputs/experiments", help="Manifest root.")
    parser.add_argument("--out-dir", default="outputs/dashboard", help="Output directory for csv files.")
    args = parser.parse_args()

    run_rows = load_run_table(args.root)
    round_rows = load_round_table(args.root)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_csv = str(out_dir / "run_table.csv")
    round_csv = str(out_dir / "round_table.csv")
    _write_csv(run_rows, run_csv)
    _write_csv(round_rows, round_csv)
    print(f"Wrote run table ({len(run_rows)} rows): {run_csv}")
    print(f"Wrote round table ({len(round_rows)} rows): {round_csv}")


if __name__ == "__main__":
    main()
