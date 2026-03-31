"""Lightweight logging and checkpointing for PINN runs."""

from __future__ import annotations

import csv
import importlib
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import torch
from omegaconf import OmegaConf

from src.pinn.losses import LOSS_COMPONENTS


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    global_epoch: int
    stage_name: str
    optimizer: str
    train_total_loss: float
    train_component_losses: dict[str, float]
    train_total_grad_norm: float | None = None
    train_component_grad_norms: dict[str, float | None] | None = None
    train_weighted_component_grad_norms: dict[str, float | None] | None = None
    val_total_loss: float | None = None
    val_component_losses: dict[str, float | None] | None = None
    test_metrics: dict[str, float | None] | None = None

    def _component_loss(self, split: str, name: str) -> float | None:
        if split == "train":
            return self.train_component_losses.get(name)
        if split == "val":
            return None if self.val_component_losses is None else self.val_component_losses.get(name)
        raise ValueError(f"Unsupported split: {split}")

    def _component_grad_norm(self, weighted: bool, name: str) -> float | None:
        source = self.train_weighted_component_grad_norms if weighted else self.train_component_grad_norms
        if source is None:
            return None
        return source.get(name)

    @property
    def train_data_loss(self) -> float | None:
        return self._component_loss("train", "data")

    @property
    def train_dt_loss(self) -> float | None:
        return self._component_loss("train", "dt")

    @property
    def train_physics_loss(self) -> float | None:
        return self._component_loss("train", "physics")

    @property
    def train_ic_loss(self) -> float | None:
        return self._component_loss("train", "ic")

    @property
    def train_data_grad_norm(self) -> float | None:
        return self._component_grad_norm(False, "data")

    @property
    def train_dt_grad_norm(self) -> float | None:
        return self._component_grad_norm(False, "dt")

    @property
    def train_physics_grad_norm(self) -> float | None:
        return self._component_grad_norm(False, "physics")

    @property
    def train_ic_grad_norm(self) -> float | None:
        return self._component_grad_norm(False, "ic")

    @property
    def val_data_loss(self) -> float | None:
        return self._component_loss("val", "data")

    @property
    def val_dt_loss(self) -> float | None:
        return self._component_loss("val", "dt")

    @property
    def val_physics_loss(self) -> float | None:
        return self._component_loss("val", "physics")

    @property
    def test_data_loss(self) -> float | None:
        if self.test_metrics is None:
            return None
        return self.test_metrics.get("data_loss")

    def as_flat_dict(self) -> dict[str, Any]:
        flat = {
            "epoch": int(self.epoch),
            "global_epoch": int(self.global_epoch),
            "stage_name": str(self.stage_name),
            "optimizer": str(self.optimizer),
            "train_total_loss": float(self.train_total_loss),
            "train_total_grad_norm": self.train_total_grad_norm,
            "val_total_loss": self.val_total_loss,
        }
        for name, value in self.train_component_losses.items():
            flat[f"train_{name}_loss"] = value
        for name in LOSS_COMPONENTS:
            flat.setdefault(f"train_{name}_loss", self.train_component_losses.get(name))
        source_maps = [
            ("train", "grad_norm", self.train_component_grad_norms),
            ("train_weighted", "grad_norm", self.train_weighted_component_grad_norms),
            ("val", "loss", self.val_component_losses),
        ]
        for prefix, suffix, values in source_maps:
            values = {} if values is None else values
            ordered_names = list(LOSS_COMPONENTS) + sorted(name for name in values.keys() if name not in LOSS_COMPONENTS)
            for name in ordered_names:
                flat[f"{prefix}_{name}_{suffix}"] = values.get(name)
        test_values = {} if self.test_metrics is None else self.test_metrics
        for key, value in test_values.items():
            flat[f"test_{key}"] = value
        flat.setdefault("test_data_loss", test_values.get("data_loss"))
        return flat


class PinnLogger:
    def __init__(self, run_dir: str, config: Any | None = None):
        self.run_dir = run_dir
        self.ckpt_dir = os.path.join(run_dir, "checkpoints")
        self.metrics_path = os.path.join(run_dir, "metrics.csv")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self._wandb_run = None
        self._wandb_enabled = False
        self._config_logging = None
        if config is not None:
            self._config_logging = getattr(config, "logging", None)
            self._configure_wandb(config)

    @staticmethod
    def default_run_dir(base_dir: str = "outputs/pinn") -> str:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(base_dir, f"run_{stamp}")

    def _configure_wandb(self, config: Any) -> None:
        wandb_cfg = getattr(config, "wandb", None)
        if wandb_cfg is None or not bool(getattr(wandb_cfg, "use", False)):
            return
        try:
            wandb = importlib.import_module("wandb")
        except ImportError as exc:
            raise RuntimeError("wandb.use=true but the wandb package is not available.") from exc

        os.makedirs(self.run_dir, exist_ok=True)
        api_key = getattr(wandb_cfg, "api_key", None)
        if api_key not in (None, ""):
            wandb.login(key=str(api_key))
        config_dict = OmegaConf.to_container(config, resolve=True)
        init_kwargs = {
            "project": str(getattr(wandb_cfg, "project", "sm-surrogates-pinn")),
            "config": config_dict,
            "name": str(getattr(wandb_cfg, "name", os.path.basename(self.run_dir))),
            "dir": self.run_dir,
        }
        if getattr(wandb_cfg, "entity", None) not in (None, ""):
            init_kwargs["entity"] = str(wandb_cfg.entity)
        if getattr(wandb_cfg, "group", None) not in (None, ""):
            init_kwargs["group"] = str(wandb_cfg.group)
        if getattr(wandb_cfg, "tags", None) not in (None, ""):
            init_kwargs["tags"] = list(wandb_cfg.tags)

        self._wandb_run = wandb.init(**init_kwargs)
        self._wandb_enabled = True

    def save_config(self, config: Any) -> None:
        os.makedirs(self.run_dir, exist_ok=True)
        OmegaConf.save(
            config=OmegaConf.create(OmegaConf.to_container(config, resolve=True)),
            f=os.path.join(self.run_dir, "config.yaml"),
        )

    def write_metrics(self, rows: list[EpochMetrics]) -> None:
        if not rows:
            return
        row = rows[-1]
        row_dict = row.as_flat_dict()
        fieldnames = list(row_dict.keys())
        write_header = not os.path.exists(self.metrics_path) or os.path.getsize(self.metrics_path) == 0
        with open(self.metrics_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row_dict)

    def _should_log_epoch(self, row: EpochMetrics) -> bool:
        log_every_epoch = int(getattr(getattr(self, "_config_logging", None), "log_every_epoch", 1))
        if log_every_epoch < 1:
            log_every_epoch = 1
        return (int(row.global_epoch) % log_every_epoch) == 0

    def print_epoch_metrics(self, row: EpochMetrics) -> None:
        if not self._should_log_epoch(row):
            return
        parts = [
            "[pinn]",
            f"global_epoch={int(row.global_epoch)}",
            f"stage={row.stage_name}",
            f"optimizer={row.optimizer}",
            f"train_total={float(row.train_total_loss):.6e}",
        ]
        for name, value in row.train_component_losses.items():
            parts.append(f"train_{name}={float(value):.6e}")
        if row.train_total_grad_norm is not None:
            parts.append(f"grad_total={float(row.train_total_grad_norm):.6e}")
        for name, value in (row.train_component_grad_norms or {}).items():
            if value is not None:
                parts.append(f"grad_{name}={float(value):.6e}")
        for name, value in (row.train_weighted_component_grad_norms or {}).items():
            if value is not None:
                parts.append(f"grad_weighted_{name}={float(value):.6e}")
        for name, value in (row.val_component_losses or {}).items():
            if value is not None:
                parts.append(f"val_{name}={float(value):.6e}")
        if row.val_total_loss is not None:
            parts.append(f"val_total={float(row.val_total_loss):.6e}")
        for key, value in (row.test_metrics or {}).items():
            if value is not None:
                parts.append(f"test_{key.replace('_loss', '')}={float(value):.6e}")
        print(" ".join(parts), flush=True)

    def log_epoch_metrics(self, row: EpochMetrics) -> None:
        if not self._wandb_enabled or self._wandb_run is None:
            return
        if not self._should_log_epoch(row):
            return
        payload = {
            "stage/epoch": int(row.epoch),
            "stage/global_epoch": int(row.global_epoch),
            "stage/name": str(row.stage_name),
            "stage/optimizer": str(row.optimizer),
            "train/total_loss": float(row.train_total_loss),
        }
        for name, value in row.train_component_losses.items():
            payload[f"train/{name}_loss"] = float(value)
        if row.train_total_grad_norm is not None:
            payload["train/grad_total_norm"] = float(row.train_total_grad_norm)
        for name, value in (row.train_component_grad_norms or {}).items():
            if value is not None:
                payload[f"train/grad_{name}_norm"] = float(value)
        for name, value in (row.train_weighted_component_grad_norms or {}).items():
            if value is not None:
                payload[f"train/grad_weighted_{name}_norm"] = float(value)
        for name, value in (row.val_component_losses or {}).items():
            if value is not None:
                payload[f"val/{name}_loss"] = float(value)
        if row.val_total_loss is not None:
            payload["val/total_loss"] = float(row.val_total_loss)
        for key, value in (row.test_metrics or {}).items():
            if value is not None:
                payload[f"test/{key}"] = float(value)
        self._wandb_run.log(payload, step=int(row.global_epoch))

    def save_checkpoint(self, payload: dict[str, Any], tag: str) -> str:
        path = os.path.join(self.ckpt_dir, f"{tag}.pt")
        torch.save(payload, path)
        return path

    def finish(self) -> None:
        if self._wandb_run is None:
            return
        self._wandb_run.finish()
        self._wandb_run = None
        self._wandb_enabled = False
