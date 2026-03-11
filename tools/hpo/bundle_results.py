"""Copy compact HPO analysis artifacts from outputs/ into a tracked results/ tree."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULT_PATTERNS = (
    "*/stage*/*/matrix.tsv",
    "*/stage*/*/runs/*/run_manifest.json",
    "*/stage*/*/runs/*/hpo_status.json",
    "*/stage*/*/runs/*/telemetry/round_telemetry.csv",
    "*/stage*/*/runs/*/qbc/history.jsonl",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        default="outputs/hpo",
        help="Source HPO outputs root.",
    )
    parser.add_argument(
        "--dst",
        default="results/hpo",
        help="Destination tracked results root.",
    )
    return parser


def _copy_matches(src_root: Path, dst_root: Path) -> tuple[int, int]:
    copied = 0
    total_bytes = 0
    seen: set[Path] = set()
    for pattern in DEFAULT_PATTERNS:
        for src_path in sorted(src_root.glob(pattern)):
            if not src_path.is_file() or src_path in seen:
                continue
            seen.add(src_path)
            rel_path = src_path.relative_to(src_root)
            dst_path = dst_root / rel_path
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_path)
            copied += 1
            total_bytes += src_path.stat().st_size
    return copied, total_bytes


def main() -> None:
    args = build_parser().parse_args()
    src_root = Path(args.src).resolve()
    dst_root = Path(args.dst).resolve()
    if not src_root.exists():
        raise FileNotFoundError(f"Source HPO root does not exist: {src_root}")

    copied, total_bytes = _copy_matches(src_root=src_root, dst_root=dst_root)
    print(f"[hpo-bundle] source: {src_root}")
    print(f"[hpo-bundle] destination: {dst_root}")
    print(f"[hpo-bundle] copied files: {copied}")
    print(f"[hpo-bundle] copied size: {total_bytes / (1024 * 1024):.1f} MiB")


if __name__ == "__main__":
    main()
