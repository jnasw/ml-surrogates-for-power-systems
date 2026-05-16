"""Lightweight logging and checkpointing for PINN runs."""

from __future__ import annotations

import csv
import importlib
import json
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
    phase_name: str
    optimizer: str
    train_total_loss: float
    train_component_losses: dict[str, float]
    train_total_grad_norm: float | None = None
    train_component_grad_norms: dict[str, float | None] | None = None
    train_weighted_component_grad_norms: dict[str, float | None] | None = None
    val_total_loss: float | None = None
    val_component_losses: dict[str, float | None] | None = None
    test_metrics: dict[str, float | None] | None = None
    regression_metrics: dict[str, dict[str, float]] | None = None
    weighting_scheme: str | None = None
    weighting_updated: bool = False
    train_loss_weights: dict[str, float] | None = None
    weighting_raw_candidate_weights: dict[str, float] | None = None
    weighting_probe_grad_l2_norms: dict[str, float] | None = None
    weighting_probe_grad_mean_abs: dict[str, float] | None = None
    weighting_probe_grad_max_abs: dict[str, float] | None = None
    weighting_probe_grad_std: dict[str, float] | None = None
    weighting_probe_ntk_mean_trace: dict[str, float] | None = None
    weighting_probe_ntk_batch_sizes: dict[str, int] | None = None
    weighting_anchor: str | None = None
    vrba_enabled: bool = False
    vrba_sampling_enabled: bool = False
    vrba_weighting_enabled: bool = False
    vrba_potential: str | None = None
    vrba_target_sets: tuple[str, ...] | None = None
    vrba_update_count: int | None = None
    vrba_sampling_frozen: bool | None = None
    vrba_weighting_frozen: bool | None = None
    vrba_physics_lambda_mean: float | None = None
    vrba_physics_lambda_max: float | None = None
    vrba_ic_lambda_mean: float | None = None
    vrba_ic_lambda_max: float | None = None
    vrba_dt_lambda_mean: float | None = None
    vrba_dt_lambda_max: float | None = None
    epoch_wall_seconds: float | None = None
    cumulative_wall_seconds: float | None = None
    num_batches: int | None = None
    num_train_steps: int | None = None
    num_supervised_rows: int | None = None
    num_collocation_rows: int | None = None
    num_init_rows: int | None = None
    supervised_acquisition_enabled: bool = False
    supervised_acquisition_strategy: str | None = None
    supervised_active_trajectories: int | None = None
    supervised_candidate_trajectories: int | None = None
    supervised_active_rows: int | None = None
    supervised_acquired_trajectories: int | None = None
    supervised_acquisition_count: int | None = None
    supervised_last_acquisition_epoch: int | None = None
    supervised_last_acquired_trajectory_ids: tuple[int, ...] | None = None
    supervised_last_anchor_score_mean: float | None = None
    supervised_last_anchor_score_max: float | None = None
    supervised_last_selected_distance_mean: float | None = None
    supervised_last_selected_distance_max: float | None = None
    peak_gpu_memory_allocated_bytes: int | None = None
    peak_gpu_memory_reserved_bytes: int | None = None
    optimizer_diagnostics: dict[str, float | int | str | bool | None] | None = None

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
            "phase_name": str(self.phase_name),
            "optimizer": str(self.optimizer),
            "train_total_loss": float(self.train_total_loss),
            "train_total_grad_norm": self.train_total_grad_norm,
            "val_total_loss": self.val_total_loss,
            "weighting_scheme": self.weighting_scheme,
            "weighting_updated": bool(self.weighting_updated),
            "weighting_anchor": self.weighting_anchor,
            "epoch_wall_seconds": self.epoch_wall_seconds,
            "cumulative_wall_seconds": self.cumulative_wall_seconds,
            "num_batches": self.num_batches,
            "num_train_steps": self.num_train_steps,
            "num_supervised_rows": self.num_supervised_rows,
            "num_collocation_rows": self.num_collocation_rows,
            "num_init_rows": self.num_init_rows,
            "peak_gpu_memory_allocated_bytes": self.peak_gpu_memory_allocated_bytes,
            "peak_gpu_memory_reserved_bytes": self.peak_gpu_memory_reserved_bytes,
        }
        if self.vrba_enabled or self.vrba_potential is not None or self.vrba_update_count is not None:
            flat["vrba_enabled"] = bool(self.vrba_enabled)
            flat["vrba_sampling_enabled"] = bool(self.vrba_sampling_enabled)
            flat["vrba_weighting_enabled"] = bool(self.vrba_weighting_enabled)
            flat["vrba_potential"] = self.vrba_potential
            flat["vrba_target_sets"] = None if self.vrba_target_sets is None else ",".join(self.vrba_target_sets)
            flat["vrba_update_count"] = self.vrba_update_count
            flat["vrba_sampling_frozen"] = self.vrba_sampling_frozen
            flat["vrba_weighting_frozen"] = self.vrba_weighting_frozen
            flat["vrba_physics_lambda_mean"] = self.vrba_physics_lambda_mean
            flat["vrba_physics_lambda_max"] = self.vrba_physics_lambda_max
            flat["vrba_ic_lambda_mean"] = self.vrba_ic_lambda_mean
            flat["vrba_ic_lambda_max"] = self.vrba_ic_lambda_max
            flat["vrba_dt_lambda_mean"] = self.vrba_dt_lambda_mean
            flat["vrba_dt_lambda_max"] = self.vrba_dt_lambda_max
        if self.supervised_acquisition_enabled or self.supervised_acquisition_strategy is not None:
            flat["supervised_acquisition_enabled"] = bool(self.supervised_acquisition_enabled)
            flat["supervised_acquisition_strategy"] = self.supervised_acquisition_strategy
            flat["supervised_active_trajectories"] = self.supervised_active_trajectories
            flat["supervised_candidate_trajectories"] = self.supervised_candidate_trajectories
            flat["supervised_active_rows"] = self.supervised_active_rows
            flat["supervised_acquired_trajectories"] = self.supervised_acquired_trajectories
            flat["supervised_acquisition_count"] = self.supervised_acquisition_count
            flat["supervised_last_acquisition_epoch"] = self.supervised_last_acquisition_epoch
            flat["supervised_last_acquired_trajectory_ids"] = (
                None
                if self.supervised_last_acquired_trajectory_ids is None
                else ",".join(str(value) for value in self.supervised_last_acquired_trajectory_ids)
            )
            flat["supervised_last_anchor_score_mean"] = self.supervised_last_anchor_score_mean
            flat["supervised_last_anchor_score_max"] = self.supervised_last_anchor_score_max
            flat["supervised_last_selected_distance_mean"] = self.supervised_last_selected_distance_mean
            flat["supervised_last_selected_distance_max"] = self.supervised_last_selected_distance_max
        flat["stage_name"] = str(self.phase_name)
        for name, value in self.train_component_losses.items():
            flat[f"train_{name}_loss"] = value
        for name in LOSS_COMPONENTS:
            flat.setdefault(f"train_{name}_loss", self.train_component_losses.get(name))
        weight_values = {} if self.train_loss_weights is None else self.train_loss_weights
        candidate_values = {} if self.weighting_raw_candidate_weights is None else self.weighting_raw_candidate_weights
        for name in LOSS_COMPONENTS:
            flat[f"train_weight_{name}"] = weight_values.get(name)
            flat[f"weighting_candidate_{name}"] = candidate_values.get(name)
        source_maps = [
            ("train", "grad_norm", self.train_component_grad_norms),
            ("train_weighted", "grad_norm", self.train_weighted_component_grad_norms),
            ("val", "loss", self.val_component_losses),
            ("weight_probe", "l2_norm", self.weighting_probe_grad_l2_norms),
            ("weight_probe", "mean_abs", self.weighting_probe_grad_mean_abs),
            ("weight_probe", "max_abs", self.weighting_probe_grad_max_abs),
            ("weight_probe", "std", self.weighting_probe_grad_std),
            ("weight_probe", "ntk_mean_trace", self.weighting_probe_ntk_mean_trace),
            ("weight_probe", "ntk_batch_size", self.weighting_probe_ntk_batch_sizes),
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
        for split_name, values in (self.regression_metrics or {}).items():
            for key, value in values.items():
                flat[f"eval_{split_name}_{key}"] = value
        optimizer_values = {} if self.optimizer_diagnostics is None else self.optimizer_diagnostics
        for key, value in optimizer_values.items():
            flat[f"optimizer_{key}"] = value
        return flat


class PinnLogger:
    def __init__(self, run_dir: str, config: Any | None = None):
        self.run_dir = run_dir
        self.ckpt_dir = os.path.join(run_dir, "checkpoints")
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

    def write_metrics_json(
        self,
        rows: list[EpochMetrics],
        *,
        run_summary: dict[str, Any] | None = None,
        final_train_metrics: dict[str, Any] | None = None,
        final_val_metrics: dict[str, Any] | None = None,
        final_test_metrics: dict[str, Any] | None = None,
        evaluation_sets: dict[str, Any] | None = None,
        best_checkpoint_metrics: dict[str, Any] | None = None,
    ) -> str | None:
        if not rows:
            return None
        path = os.path.join(self.run_dir, "metrics.json")
        epoch_metrics_csv = self.write_epoch_metrics_csv(rows)
        final_row_obj = rows[-1]
        final_row = final_row_obj.as_flat_dict()
        final_train_losses = {
            "total_loss": float(final_row_obj.train_total_loss),
            "component_losses": dict(final_row_obj.train_component_losses),
        }
        if final_row_obj.train_loss_weights is not None:
            final_train_losses["loss_weights"] = dict(final_row_obj.train_loss_weights)
        final_val_losses = None
        if final_row_obj.val_total_loss is not None or final_row_obj.val_component_losses is not None:
            final_val_losses = {
                "total_loss": final_row_obj.val_total_loss,
                "component_losses": None if final_row_obj.val_component_losses is None else dict(final_row_obj.val_component_losses),
            }
        # final_*_metrics and final_*_losses are the canonical semantic summaries.
        # final_epoch is retained as the last logged epoch telemetry/debug snapshot.
        payload = {
            "schema_version": "run_metrics_v1",
            "run_summary": {} if run_summary is None else dict(run_summary),
            "final_train_metrics": final_train_metrics,
            "final_val_metrics": final_val_metrics,
            "final_test_metrics": final_test_metrics,
            "best_checkpoint_metrics": best_checkpoint_metrics,
            "final_train_losses": final_train_losses,
            "final_val_losses": final_val_losses,
            "final_test_losses": None,
            "evaluation_sets": evaluation_sets,
            "epochs_recorded": int(len(rows)),
            "epoch_metrics_csv": epoch_metrics_csv,
            "final_epoch": final_row,
        }
        os.makedirs(self.run_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return path

    def write_epoch_metrics_csv(self, rows: list[EpochMetrics]) -> str | None:
        """Persist per-epoch telemetry for local convergence/time analysis."""
        if not rows:
            return None
        path = os.path.join(self.run_dir, "epoch_metrics.csv")
        flat_rows = [row.as_flat_dict() for row in rows]
        fieldnames: list[str] = []
        for row in flat_rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        os.makedirs(self.run_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flat_rows)
        return path

    def _should_log_epoch(self, row: EpochMetrics) -> bool:
        if int(row.global_epoch) in (0, 1):
            return True
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
            f"phase={row.phase_name}",
            f"optimizer={row.optimizer}",
            f"train_total={float(row.train_total_loss):.6e}",
        ]
        current_lr = None if row.optimizer_diagnostics is None else row.optimizer_diagnostics.get("current_lr")
        if current_lr is not None:
            parts.append(f"lr={float(current_lr):.6e}")
        for name in LOSS_COMPONENTS:
            value = row.train_component_losses.get(name)
            if value is not None:
                parts.append(f"train_{name}={float(value):.6e}")
        weight_values = row.train_loss_weights or {}
        for name in ("data", "physics", "dt", "ic"):
            value = weight_values.get(name)
            if value is not None:
                parts.append(f"weight_{name}={float(value):.6e}")
        if row.val_total_loss is not None:
            parts.append(f"val_total={float(row.val_total_loss):.6e}")
        test_data_loss = None if row.test_metrics is None else row.test_metrics.get("data_loss")
        if test_data_loss is not None:
            parts.append(f"test_data={float(test_data_loss):.6e}")
        if row.epoch_wall_seconds is not None:
            parts.append(f"epoch_s={float(row.epoch_wall_seconds):.3f}")
        print(" ".join(parts), flush=True)

    def log_epoch_metrics(self, row: EpochMetrics) -> None:
        if not self._wandb_enabled or self._wandb_run is None:
            return
        if not self._should_log_epoch(row):
            return
        payload = {
            "phase/name": str(row.phase_name),
            "phase/optimizer": str(row.optimizer),
            "train/loss/total": float(row.train_total_loss),
            "weights/updated": bool(row.weighting_updated),
        }
        if row.weighting_scheme is not None:
            payload["weights/scheme"] = str(row.weighting_scheme)
        if row.weighting_anchor is not None:
            payload["weights/anchor"] = str(row.weighting_anchor)
        if row.vrba_enabled:
            payload["vrba/enabled"] = True
            payload["vrba/adaptive_sampling"] = bool(row.vrba_sampling_enabled)
            payload["vrba/adaptive_weighting"] = bool(row.vrba_weighting_enabled)
            if row.vrba_potential is not None:
                payload["vrba/potential"] = str(row.vrba_potential)
            if row.vrba_target_sets is not None:
                payload["vrba/target_sets"] = list(row.vrba_target_sets)
            if row.vrba_update_count is not None:
                payload["vrba/update_count"] = int(row.vrba_update_count)
        for name, value in row.train_component_losses.items():
            payload[f"train/loss/{name}"] = float(value)
        for name, value in (row.train_loss_weights or {}).items():
            payload[f"weights/active/{name}"] = float(value)
        for name, value in (row.weighting_raw_candidate_weights or {}).items():
            payload[f"debug/weights/candidate/{name}"] = float(value)
        if row.train_total_grad_norm is not None:
            payload["gradients/raw/total"] = float(row.train_total_grad_norm)
        if row.epoch_wall_seconds is not None:
            payload["runtime/epoch_seconds"] = float(row.epoch_wall_seconds)
        if row.cumulative_wall_seconds is not None:
            payload["runtime/cumulative_seconds"] = float(row.cumulative_wall_seconds)
        if row.num_batches is not None:
            payload["runtime/num_batches"] = int(row.num_batches)
        if row.num_train_steps is not None:
            payload["runtime/num_train_steps"] = int(row.num_train_steps)
        if row.num_supervised_rows is not None:
            payload["runtime/num_supervised_rows"] = int(row.num_supervised_rows)
        if row.num_collocation_rows is not None:
            payload["runtime/num_collocation_rows"] = int(row.num_collocation_rows)
        if row.num_init_rows is not None:
            payload["runtime/num_init_rows"] = int(row.num_init_rows)
        if row.supervised_acquisition_enabled:
            payload["supervised_acquisition/enabled"] = True
            if row.supervised_acquisition_strategy is not None:
                payload["supervised_acquisition/strategy"] = str(row.supervised_acquisition_strategy)
            if row.supervised_active_trajectories is not None:
                payload["supervised_acquisition/active_trajectories"] = int(row.supervised_active_trajectories)
            if row.supervised_candidate_trajectories is not None:
                payload["supervised_acquisition/candidate_trajectories"] = int(row.supervised_candidate_trajectories)
            if row.supervised_active_rows is not None:
                payload["supervised_acquisition/active_rows"] = int(row.supervised_active_rows)
            if row.supervised_acquired_trajectories is not None:
                payload["supervised_acquisition/acquired_trajectories"] = int(row.supervised_acquired_trajectories)
            if row.supervised_acquisition_count is not None:
                payload["supervised_acquisition/acquisition_count"] = int(row.supervised_acquisition_count)
            if row.supervised_last_acquisition_epoch is not None:
                payload["supervised_acquisition/last_acquisition_epoch"] = int(row.supervised_last_acquisition_epoch)
            if row.supervised_last_anchor_score_mean is not None:
                payload["supervised_acquisition/last_anchor_score_mean"] = float(row.supervised_last_anchor_score_mean)
            if row.supervised_last_anchor_score_max is not None:
                payload["supervised_acquisition/last_anchor_score_max"] = float(row.supervised_last_anchor_score_max)
            if row.supervised_last_selected_distance_mean is not None:
                payload["supervised_acquisition/last_selected_distance_mean"] = float(row.supervised_last_selected_distance_mean)
            if row.supervised_last_selected_distance_max is not None:
                payload["supervised_acquisition/last_selected_distance_max"] = float(row.supervised_last_selected_distance_max)
        if row.peak_gpu_memory_allocated_bytes is not None:
            payload["runtime/peak_gpu_memory_allocated_bytes"] = int(row.peak_gpu_memory_allocated_bytes)
            payload["runtime/peak_gpu_memory_allocated_mb"] = float(row.peak_gpu_memory_allocated_bytes) / (1024.0 * 1024.0)
        if row.peak_gpu_memory_reserved_bytes is not None:
            payload["runtime/peak_gpu_memory_reserved_bytes"] = int(row.peak_gpu_memory_reserved_bytes)
            payload["runtime/peak_gpu_memory_reserved_mb"] = float(row.peak_gpu_memory_reserved_bytes) / (1024.0 * 1024.0)
        for name, value in (row.train_component_grad_norms or {}).items():
            if value is not None:
                payload[f"gradients/raw/{name}"] = float(value)
        for name, value in (row.train_weighted_component_grad_norms or {}).items():
            if value is not None:
                payload[f"gradients/weighted/{name}"] = float(value)
        for name, value in (row.weighting_probe_grad_l2_norms or {}).items():
            payload[f"debug/weights/probe/l2_norm/{name}"] = float(value)
        for name, value in (row.weighting_probe_grad_mean_abs or {}).items():
            payload[f"debug/weights/probe/mean_abs/{name}"] = float(value)
        for name, value in (row.weighting_probe_grad_max_abs or {}).items():
            payload[f"debug/weights/probe/max_abs/{name}"] = float(value)
        for name, value in (row.weighting_probe_grad_std or {}).items():
            payload[f"debug/weights/probe/std/{name}"] = float(value)
        for name, value in (row.weighting_probe_ntk_mean_trace or {}).items():
            payload[f"debug/weights/probe/ntk_mean_trace/{name}"] = float(value)
        for name, value in (row.weighting_probe_ntk_batch_sizes or {}).items():
            payload[f"debug/weights/probe/ntk_batch_size/{name}"] = int(value)
        for name, value in (row.val_component_losses or {}).items():
            if value is not None:
                payload[f"val/loss/{name}"] = float(value)
        if row.val_total_loss is not None:
            payload["val/loss/total"] = float(row.val_total_loss)
        for key, value in (row.test_metrics or {}).items():
            if value is not None:
                if key == "data_loss":
                    payload["test/loss/data"] = float(value)
                else:
                    payload[f"test/metric/{key}"] = float(value)
        for split_name, values in (row.regression_metrics or {}).items():
            for key, value in values.items():
                if value is not None:
                    payload[f"eval/{split_name}/{key}"] = float(value)
        for key, value in (row.optimizer_diagnostics or {}).items():
            if value is not None:
                payload[f"optimizer/{key}"] = value
        self._wandb_run.log(payload, step=int(row.global_epoch))

    def save_checkpoint(self, payload: dict[str, Any], tag: str) -> str:
        os.makedirs(self.ckpt_dir, exist_ok=True)
        path = os.path.join(self.ckpt_dir, f"{tag}.pt")
        torch.save(payload, path)
        return path

    def finish(self) -> None:
        if self._wandb_run is None:
            return
        self._wandb_run.finish()
        self._wandb_run = None
        self._wandb_enabled = False
