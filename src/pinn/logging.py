"""Lightweight logging and checkpointing for PINN runs."""

from __future__ import annotations

import csv
import importlib
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import torch
from omegaconf import OmegaConf


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    global_epoch: int
    stage_name: str
    optimizer: str
    train_total_loss: float
    train_data_loss: float
    train_physics_loss: float
    train_ic_loss: float
    val_data_loss: float | None = None
    val_physics_loss: float | None = None
    test_data_loss: float | None = None


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
        fieldnames = list(asdict(row).keys())
        write_header = not os.path.exists(self.metrics_path) or os.path.getsize(self.metrics_path) == 0
        with open(self.metrics_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(asdict(row))

    def log_epoch_metrics(self, row: EpochMetrics) -> None:
        if not self._wandb_enabled or self._wandb_run is None:
            return
        log_every_epoch = int(getattr(getattr(self, "_config_logging", None), "log_every_epoch", 1))
        if log_every_epoch < 1:
            log_every_epoch = 1
        if (int(row.global_epoch) % log_every_epoch) != 0:
            return
        payload = {
            "stage/epoch": int(row.epoch),
            "stage/global_epoch": int(row.global_epoch),
            "stage/name": str(row.stage_name),
            "stage/optimizer": str(row.optimizer),
            "train/total_loss": float(row.train_total_loss),
            "train/data_loss": float(row.train_data_loss),
            "train/physics_loss": float(row.train_physics_loss),
            "train/ic_loss": float(row.train_ic_loss),
        }
        if row.val_data_loss is not None:
            payload["val/data_loss"] = float(row.val_data_loss)
        if row.val_physics_loss is not None:
            payload["val/physics_loss"] = float(row.val_physics_loss)
        if row.test_data_loss is not None:
            payload["test/data_loss"] = float(row.test_data_loss)
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
