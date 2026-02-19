"""Stage-2 preprocessing: raw trajectories -> train/val/test + collocation HDF5."""

from __future__ import annotations

import os
import pickle
from typing import Any

import h5py
import numpy as np
from omegaconf import OmegaConf

from src.dataset.create_dataset_functions import ODEModelling
from src.dataset.data_contract import (
    H5_COLLOCATION_SUFFIX,
    H5_DATA_SUFFIX,
    H5_FILE_SUFFIX,
    H5_INIT_SUFFIX,
    H5_X_KEYS,
    H5_Y_KEYS,
    INFO_FILE_NAME,
    TRAIN_SPLIT,
    VAL_SPLIT,
    TEST_SPLIT,
    validate_info_lines,
)


class Datapreprocessor:
    """Preprocess raw ODE trajectories and persist training datasets."""

    def __init__(self, cfg: Any):
        self.cfg = cfg
        self.dataset_cfg = cfg.dataset
        self.model_flag = str(cfg.model.model_flag)
        self.save_freq = 500_000
        self.info_attributes: dict[str, Any] = {}

        self.scrap_info()
        self.create_train_val_test_folder()
        self.load_keys()

        # Running stats used for optional normalization and metadata.
        self.n_samples = 0
        self.mean = None
        self.m2 = None
        self.min = None
        self.max = None

    def scrap_info(self) -> None:
        number = int(self.dataset_cfg.number)
        self.folder_path = os.path.join(self.cfg.dirs.dataset_dir, self.model_flag, f"dataset_v{number}")
        self.info_path = os.path.join(self.folder_path, INFO_FILE_NAME)

        if not os.path.exists(self.info_path):
            raise FileNotFoundError(f"info.txt not found: {self.info_path}")

        with open(self.info_path, "r", encoding="utf-8") as text_file:
            lines = text_file.readlines()
        validate_info_lines(lines)

        self.num_of_raw_files = int(lines[0].split(":", 1)[1].strip())
        self.total_init_conditions = int(lines[2].split(":", 1)[1].strip())
        self.time_sim = float(lines[3].split(":", 1)[1].strip())
        self.num_of_points_sim = int(lines[4].split(":", 1)[1].strip())

    def create_train_val_test_folder(self) -> None:
        if not os.path.exists(self.folder_path):
            raise FileNotFoundError(f"Dataset folder not found: {self.folder_path}")

        self.raw_data_path = os.path.join(self.folder_path, "raw")
        if not os.path.exists(self.raw_data_path):
            raise FileNotFoundError(f"Raw data folder not found: {self.raw_data_path}")

        raw_files = [f for f in os.listdir(self.raw_data_path) if f.endswith(".pkl")]
        if len(raw_files) != self.num_of_raw_files:
            raise ValueError(
                f"Raw file count mismatch: info={self.num_of_raw_files}, actual={len(raw_files)}"
            )

        self.train_folder = os.path.join(self.folder_path, TRAIN_SPLIT)
        self.val_folder = os.path.join(self.folder_path, VAL_SPLIT)
        self.test_folder = os.path.join(self.folder_path, TEST_SPLIT)
        os.makedirs(self.train_folder, exist_ok=True)
        os.makedirs(self.val_folder, exist_ok=True)
        os.makedirs(self.test_folder, exist_ok=True)

        self.train_file_count = self._existing_split_count(self.train_folder, H5_DATA_SUFFIX)
        self.val_file_count = self._existing_split_count(self.val_folder, H5_DATA_SUFFIX)
        self.test_file_count = self._existing_split_count(self.test_folder, H5_DATA_SUFFIX)
        self.col_file_count = self._existing_split_count(self.train_folder, H5_COLLOCATION_SUFFIX)
        self.init_file_count = self._existing_split_count(self.train_folder, H5_INIT_SUFFIX)

    def load_keys(self) -> None:
        modeling_guide_path = os.path.join(self.cfg.dirs.init_conditions_dir, "modellings_guide.yaml")
        if not os.path.exists(modeling_guide_path):
            raise FileNotFoundError(f"Modeling guide not found: {modeling_guide_path}")

        modeling_guide = OmegaConf.load(modeling_guide_path)
        keys = None
        keys_ext = None
        for model in modeling_guide:
            if model.get("name") == self.model_flag:
                keys = model.get("keys")
                keys_ext = model.get("keys_ext")
                break
        if not keys:
            raise ValueError(f"Model '{self.model_flag}' not found in modeling guide.")

        self.input_dim = len(keys) + (len(keys_ext) if keys_ext else 0) + 1
        self.output_dim = len(keys)

    def _existing_split_count(self, folder_path: str, suffix: str) -> int:
        return len(
            [
                f
                for f in os.listdir(folder_path)
                if f.endswith(H5_FILE_SUFFIX) and suffix in f
            ]
        )

    def _save_dataset(
        self,
        split: str,
        x_data: np.ndarray,
        y_data: np.ndarray | None,
        cnt: int,
        suffix: str = H5_DATA_SUFFIX,
    ) -> None:
        folder = {
            TRAIN_SPLIT: self.train_folder,
            VAL_SPLIT: self.val_folder,
            TEST_SPLIT: self.test_folder,
        }[split]
        file_name = f"{split}{suffix}{cnt}{H5_FILE_SUFFIX}"
        file_path = os.path.join(folder, file_name)
        with h5py.File(file_path, "w") as h5f:
            h5f.create_dataset(H5_X_KEYS[split], data=x_data, compression="gzip", compression_opts=9)
            if y_data is not None:
                h5f.create_dataset(H5_Y_KEYS[split], data=y_data, compression="gzip", compression_opts=9)

    def _update_train_stats(self, x_chunk: np.ndarray) -> None:
        if x_chunk.size == 0:
            return
        batch_n = x_chunk.shape[0]
        batch_mean = x_chunk.mean(axis=0)
        batch_var = x_chunk.var(axis=0)
        batch_min = x_chunk.min(axis=0)
        batch_max = x_chunk.max(axis=0)

        if self.mean is None:
            self.n_samples = batch_n
            self.mean = batch_mean
            self.m2 = batch_var * batch_n
            self.min = batch_min
            self.max = batch_max
            return

        n_total = self.n_samples + batch_n
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * (batch_n / n_total)
        self.m2 = self.m2 + (batch_var * batch_n) + (delta**2) * (self.n_samples * batch_n / n_total)
        self.n_samples = n_total
        self.min = np.minimum(self.min, batch_min)
        self.max = np.maximum(self.max, batch_max)

    def _finalize_stats(self) -> None:
        if self.n_samples == 0:
            return
        var = self.m2 / self.n_samples
        std = np.sqrt(var)
        std = np.where(std < 1e-7, 1.0, std)
        data_range = self.max - self.min
        data_range = np.where(data_range < 1e-7, 1.0, data_range)

        self.set_info_attributes(
            min=self.min.tolist(),
            max=self.max.tolist(),
            range=data_range.tolist(),
            mean=self.mean.tolist(),
            std=std.tolist(),
            n_samples=int(self.n_samples),
        )

    def set_info_attributes(self, **kwargs: Any) -> None:
        self.info_attributes.update(kwargs)

    def update_info_file(self) -> None:
        if not os.path.exists(self.info_path):
            raise FileNotFoundError(f"Info file not found: {self.info_path}")
        with open(self.info_path, "r", encoding="utf-8") as text_file:
            lines = text_file.readlines()

        ordered_keys: list[str] = []
        info_map: dict[str, str] = {}
        for line in lines:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if key not in info_map:
                ordered_keys.append(key)
            info_map[key] = value

        for key, value in self.info_attributes.items():
            out_key = key.replace("_", " ")
            out_value = str(value)
            if out_key not in info_map:
                ordered_keys.append(out_key)
            info_map[out_key] = out_value

        with open(self.info_path, "w", encoding="utf-8") as text_file:
            for key in ordered_keys:
                text_file.write(f"{key}: {info_map[key]}\n")

    def _load_raw_file(self, file_path: str, shuffle_flag: bool) -> list[Any]:
        with open(file_path, "rb") as handle:
            data = pickle.load(handle)
        if shuffle_flag:
            rng = np.random.default_rng(getattr(self.cfg.model, "seed", None))
            rng.shuffle(data)
        return data

    def _trajectory_to_xy(self, trajectory: Any, time_limit: float) -> tuple[np.ndarray, np.ndarray]:
        arr = np.array(trajectory, dtype=np.float32)
        if time_limit > self.time_sim:
            raise ValueError("Configured preprocessing time exceeds simulation time in info.txt.")
        if time_limit != 0:
            arr = arr[:, arr[0] <= time_limit]

        y = arr[1 : self.output_dim + 1].T.copy()
        x = arr.T.copy()
        x[:, 1:] = x[0, 1:]
        return x, y

    def _split_from_index(self, traj_idx: int, train_cutoff: int, val_cutoff: int, has_val: bool) -> str:
        if traj_idx < train_cutoff:
            return TRAIN_SPLIT
        if has_val and traj_idx < val_cutoff:
            return VAL_SPLIT
        return TEST_SPLIT

    def get_preprocess_save_data(self) -> None:
        train_traj = int(self.total_init_conditions * float(self.dataset_cfg.split_ratio))
        if bool(self.dataset_cfg.validation_flag):
            remaining = self.total_init_conditions - train_traj
            val_traj = remaining // 2
            val_cutoff = train_traj + val_traj
        else:
            val_cutoff = train_traj

        buffers_x = {TRAIN_SPLIT: [], VAL_SPLIT: [], TEST_SPLIT: []}
        buffers_y = {TRAIN_SPLIT: [], VAL_SPLIT: [], TEST_SPLIT: []}
        trajectories_by_split = {TRAIN_SPLIT: 0, VAL_SPLIT: 0, TEST_SPLIT: 0}

        traj_idx_global = 0
        raw_files = sorted([f for f in os.listdir(self.raw_data_path) if f.endswith(".pkl")])
        for raw_file in raw_files:
            file_path = os.path.join(self.raw_data_path, raw_file)
            trajectories = self._load_raw_file(file_path, bool(self.dataset_cfg.shuffle))
            for trajectory in trajectories:
                split = self._split_from_index(
                    traj_idx=traj_idx_global,
                    train_cutoff=train_traj,
                    val_cutoff=val_cutoff,
                    has_val=bool(self.dataset_cfg.validation_flag),
                )
                x, y = self._trajectory_to_xy(trajectory, float(self.cfg.time))
                buffers_x[split].append(x)
                buffers_y[split].append(y)
                trajectories_by_split[split] += 1
                traj_idx_global += 1

                # Flush split buffers by sample count.
                for split_name in (TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT):
                    sample_count = sum(chunk.shape[0] for chunk in buffers_x[split_name])
                    if sample_count >= self.save_freq:
                        self._flush_split_buffers(split_name, buffers_x, buffers_y)

        for split_name in (TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT):
            self._flush_split_buffers(split_name, buffers_x, buffers_y, flush_all=True)

        self._finalize_stats()
        self.set_info_attributes(
            num_of_train_files=self.train_file_count,
            num_of_training_trajectories=trajectories_by_split[TRAIN_SPLIT],
            num_of_val_files=self.val_file_count,
            num_of_validation_trajectories=trajectories_by_split[VAL_SPLIT],
            num_of_test_files=self.test_file_count,
            num_of_testing_trajectories=trajectories_by_split[TEST_SPLIT],
        )

    def _flush_split_buffers(
        self,
        split: str,
        buffers_x: dict[str, list[np.ndarray]],
        buffers_y: dict[str, list[np.ndarray]],
        flush_all: bool = False,
    ) -> None:
        if not buffers_x[split]:
            return
        if (not flush_all) and (sum(chunk.shape[0] for chunk in buffers_x[split]) < self.save_freq):
            return

        x_chunk = np.concatenate(buffers_x[split], axis=0)
        y_chunk = np.concatenate(buffers_y[split], axis=0)
        buffers_x[split].clear()
        buffers_y[split].clear()

        if split == TRAIN_SPLIT:
            self.train_file_count += 1
            self._update_train_stats(x_chunk)
            self._save_dataset(split, x_chunk, y_chunk, self.train_file_count, suffix=H5_DATA_SUFFIX)
        elif split == VAL_SPLIT:
            self.val_file_count += 1
            self._save_dataset(split, x_chunk, y_chunk, self.val_file_count, suffix=H5_DATA_SUFFIX)
        else:
            self.test_file_count += 1
            self._save_dataset(split, x_chunk, y_chunk, self.test_file_count, suffix=H5_DATA_SUFFIX)

    def create_col_points(self, init_condition_table: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        init_condition_table = np.asarray(init_condition_table, dtype=np.float32)
        if init_condition_table.ndim != 2:
            raise ValueError("init_condition_table must be a 2D array")

        col_points_zero = np.concatenate(
            (np.zeros((init_condition_table.shape[0], 1), dtype=np.float32), init_condition_table), axis=1
        )
        time = np.linspace(0.0, float(self.cfg.time), int(self.cfg.num_of_points) + 1, dtype=np.float32)
        time = np.tile(time[None, :, None], (init_condition_table.shape[0], 1, 1))
        expanded_ic = np.tile(init_condition_table[:, None, :], (1, time.shape[1], 1))
        col_points = np.concatenate((time, expanded_ic), axis=2).reshape(-1, expanded_ic.shape[2] + 1)
        return col_points, col_points_zero

    def create_save_col_data(self, save_flag: bool = True):
        if not bool(self.dataset_cfg.new_coll_points_flag):
            return None

        ode_model = ODEModelling(self.cfg)
        init_conditions = np.array(ode_model.build_initial_conditions(), dtype=np.float32)

        points_per_traj = int(self.cfg.num_of_points) + 1
        ic_chunk_size = max(1, self.save_freq // points_per_traj)
        saved_col_files = 0

        for start in range(0, len(init_conditions), ic_chunk_size):
            chunk = init_conditions[start : start + ic_chunk_size]
            x_col, x_init = self.create_col_points(chunk)
            y_init = x_init[:, 1 : self.output_dim + 1]
            if not save_flag:
                return x_col, x_init, y_init

            self._update_train_stats(x_col)
            self.col_file_count += 1
            self.init_file_count += 1
            self._save_dataset(TRAIN_SPLIT, x_col, None, self.col_file_count, suffix=H5_COLLOCATION_SUFFIX)
            self._save_dataset(TRAIN_SPLIT, x_init, y_init, self.init_file_count, suffix=H5_INIT_SUFFIX)
            saved_col_files += 1

        self.set_info_attributes(
            num_of_train_col_files=self.col_file_count,
            num_of_training_col_trajectories=int(len(init_conditions)),
            num_of_train_init_files=self.init_file_count,
        )
        return None
