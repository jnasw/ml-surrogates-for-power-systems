"""Experiment logging and checkpoint helpers for QBC runs."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from src.methods.ensemble import DeepEnsemble, load_ensemble_checkpoint, save_ensemble_checkpoint
from src.data.loaders.trajectory_dataset import TrajectoryDataset


class ExperimentLogger:
    """Lightweight file-based logger for iterative QBC experiments."""

    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.rounds_dir = os.path.join(run_dir, "rounds")
        self.ckpt_dir = os.path.join(run_dir, "checkpoints")
        os.makedirs(self.rounds_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.history_path = os.path.join(run_dir, "history.jsonl")

    @staticmethod
    def default_run_dir(base_dir: str = "outputs/qbc") -> str:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(base_dir, f"run_{stamp}")

    def save_config(self, config: Any) -> None:
        os.makedirs(self.run_dir, exist_ok=True)
        OmegaConf.save(config=OmegaConf.create(OmegaConf.to_container(config, resolve=True)), f=os.path.join(self.run_dir, "config.yaml"))

    def log_round(
        self,
        summary: Any,
        x_cand: np.ndarray,
        scores: np.ndarray,
        selected_idx: np.ndarray,
        selected_ics: np.ndarray,
    ) -> None:
        round_dir = os.path.join(self.rounds_dir, f"round_{summary.round_idx:03d}")
        os.makedirs(round_dir, exist_ok=True)
        np.save(os.path.join(round_dir, "candidate_ics.npy"), x_cand)
        np.save(os.path.join(round_dir, "scores.npy"), scores)
        np.save(os.path.join(round_dir, "selected_indices.npy"), selected_idx)
        np.save(os.path.join(round_dir, "selected_ics.npy"), selected_ics)

        payload = asdict(summary)
        payload["selected_indices"] = summary.selected_indices.tolist()
        with open(self.history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

    def save_round_arrays(self, round_idx: int, **arrays: Any) -> None:
        """Save additional round-level arrays for custom acquisition analyses."""
        round_dir = os.path.join(self.rounds_dir, f"round_{int(round_idx):03d}")
        os.makedirs(round_dir, exist_ok=True)
        for key, value in arrays.items():
            if value is None:
                continue
            np.save(os.path.join(round_dir, f"{key}.npy"), np.asarray(value))

    def save_dataset_checkpoint(self, dataset: TrajectoryDataset, tag: str) -> str:
        path = os.path.join(self.ckpt_dir, f"dataset_{tag}.npz")
        test_ics, test_trajs = dataset.test_view()
        np.savez_compressed(
            path,
            train_ics=dataset.train_ics,
            train_trajs=dataset.train_trajs,
            test_ics=test_ics if test_ics is not None else np.array([], dtype=np.float32),
            test_trajs=test_trajs if test_trajs is not None else np.array([], dtype=np.float32),
            has_test=np.array([test_ics is not None], dtype=bool),
            time_grid=dataset.time_grid if dataset.time_grid is not None else np.array([], dtype=np.float32),
        )
        return path

    def load_dataset_checkpoint(self, tag: str) -> TrajectoryDataset:
        path = os.path.join(self.ckpt_dir, f"dataset_{tag}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        d = np.load(path, allow_pickle=False)
        has_test = bool(d["has_test"][0])
        test_ics = d["test_ics"] if has_test else None
        test_trajs = d["test_trajs"] if has_test else None
        time_grid = d["time_grid"] if d["time_grid"].size > 0 else None
        return TrajectoryDataset(
            train_ics=d["train_ics"],
            train_trajs=d["train_trajs"],
            test_ics=test_ics,
            test_trajs=test_trajs,
            time_grid=time_grid,
        )

    def save_ensemble_checkpoint(self, ensemble: DeepEnsemble, tag: str) -> str:
        return save_ensemble_checkpoint(ensemble=ensemble, folder=self.ckpt_dir, tag=tag)

    def load_ensemble_checkpoint(self, tag: str, device_preference: str = "auto") -> DeepEnsemble:
        return load_ensemble_checkpoint(folder=self.ckpt_dir, tag=tag, device_preference=device_preference)
