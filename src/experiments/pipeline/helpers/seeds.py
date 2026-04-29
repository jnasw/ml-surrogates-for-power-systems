"""Shared seed-registry helpers for experiment launchers."""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf


DEFAULT_SEED_REGISTRY_PATH = Path(__file__).resolve().parents[3] / "config" / "registry" / "seeds.yaml"


def parse_int_list(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def parse_label_list(raw: str) -> list[str]:
    labels = [item.strip() for item in raw.split(",") if item.strip()]
    if not labels:
        raise ValueError("Expected at least one seed label.")
    return labels


def load_seed_registry(seed_registry_path: Path = DEFAULT_SEED_REGISTRY_PATH) -> dict[str, int]:
    if not seed_registry_path.exists():
        raise FileNotFoundError(f"Seed registry not found: {seed_registry_path}")
    cfg = OmegaConf.load(seed_registry_path)
    return {str(key): int(value) for key, value in cfg.items()}


def seed_pairs_from_labels(
    labels: list[str],
    *,
    seed_registry_path: Path = DEFAULT_SEED_REGISTRY_PATH,
) -> list[tuple[str, int]]:
    registry = load_seed_registry(seed_registry_path)
    missing = [label for label in labels if label not in registry]
    if missing:
        raise ValueError(
            f"Unknown seed label(s): {', '.join(missing)}. "
            f"Check {seed_registry_path}."
        )
    return [(label, registry[label]) for label in labels]


def raw_seed_pairs(seeds: list[int]) -> list[tuple[str, int]]:
    return [(f"raw{int(seed)}", int(seed)) for seed in seeds]
