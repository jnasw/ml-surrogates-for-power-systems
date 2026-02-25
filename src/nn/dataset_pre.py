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
        self.test_split_mode = str(getattr(self.dataset_cfg, "test_split_mode", "internal")).lower()
        self.shared_test_dataset_number = getattr(self.dataset_cfg, "shared_test_dataset_number", None)
        self.shared_test_dataset_root = getattr(self.dataset_cfg, "shared_test_dataset_root", None)
        self.shared_test_max_trajectories = getattr(self.dataset_cfg, "shared_test_max_trajectories", None)
        self.paired_other_dataset_number = getattr(self.dataset_cfg, "paired_other_dataset_number", None)
        self.ic_key_decimals = int(getattr(self.dataset_cfg, "ic_key_decimals", 8))

        self.scrap_info()
        self.create_train_val_test_folder()
        self.load_keys()

        # Running stats used for optional normalization and metadata.
        self.n_samples = 0
        self.mean = None
        self.m2 = None
        self.min = None
        self.max = None
        print(
            "[preprocess] Initialized | "
            f"folder={self.folder_path} raw_files={self.num_of_raw_files} "
            f"trajectories={self.total_init_conditions} split_mode={self.test_split_mode}"
        )

    def scrap_info(self) -> None:
        dataset_root = getattr(self.dataset_cfg, "root", None)
        if dataset_root not in (None, ""):
            candidate = str(dataset_root)
            self.folder_path = (
                candidate
                if os.path.isabs(candidate)
                else os.path.join(str(self.cfg.dirs.dataset_dir), candidate)
            )
        else:
            number = int(self.dataset_cfg.number)
            self.folder_path = os.path.join(self.cfg.dirs.dataset_dir, self.model_flag, f"dataset_v{number}")
        self.info_path = os.path.join(self.folder_path, INFO_FILE_NAME)

        if not os.path.exists(self.info_path):
            raise FileNotFoundError(f"info.txt not found: {self.info_path}")

        with open(self.info_path, "r", encoding="utf-8") as text_file:
            lines = text_file.readlines()
        validate_info_lines(lines)
        info_map = self._parse_info_lines(lines)

        self.num_of_raw_files = int(info_map["Number of files"])
        self.total_init_conditions = int(info_map["Number of different simulated trajectories"])
        self.time_sim = float(info_map["Time horizon of the simulations"])
        self.num_of_points_sim = int(info_map["Number of points in the each simulation"])

    def _parse_info_lines(self, lines: list[str]) -> dict[str, str]:
        out: dict[str, str] = {}
        for line in lines:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            out[key.strip()] = value.strip()
        return out

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
        print(
            "[preprocess] Output folders ready | "
            f"train_files={self.train_file_count} val_files={self.val_file_count} test_files={self.test_file_count}"
        )

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
        print(f"[preprocess] Updated info.txt: {self.info_path}")

    def _load_raw_file(self, file_path: str, shuffle_flag: bool) -> list[Any]:
        with open(file_path, "rb") as handle:
            data = pickle.load(handle)
        if shuffle_flag:
            rng = np.random.default_rng(getattr(self.cfg.model, "seed", None))
            rng.shuffle(data)
        return data

    def _trajectory_to_xy(
        self,
        trajectory: Any,
        time_limit: float,
        simulation_time: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        arr = np.array(trajectory, dtype=np.float32)
        sim_time = self.time_sim if simulation_time is None else float(simulation_time)
        if time_limit > sim_time:
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

    def _ic_key_from_trajectory(self, trajectory: Any) -> tuple[float, ...]:
        arr = np.asarray(trajectory, dtype=np.float32)
        ic = arr[1:, 0]
        return tuple(np.round(ic, self.ic_key_decimals).tolist())

    def _internal_split_bounds(self, total_trajectories: int) -> tuple[int, int]:
        train_cutoff = int(total_trajectories * float(self.dataset_cfg.split_ratio))
        if bool(self.dataset_cfg.validation_flag):
            remaining = total_trajectories - train_cutoff
            val_traj = remaining // 2
            val_cutoff = train_cutoff + val_traj
        else:
            val_cutoff = train_cutoff
        return train_cutoff, val_cutoff

    def _collect_internal_test_trajectories(
        self, dataset_number: int
    ) -> dict[tuple[float, ...], tuple[Any, float]]:
        raw_path, total_trajectories, time_sim = self._load_dataset_info(int(dataset_number))
        train_cutoff, val_cutoff = self._internal_split_bounds(total_trajectories)
        has_val = bool(self.dataset_cfg.validation_flag)

        mapping: dict[tuple[float, ...], tuple[Any, float]] = {}
        idx = 0
        raw_files = sorted([f for f in os.listdir(raw_path) if f.endswith(".pkl")])
        for raw_file in raw_files:
            file_path = os.path.join(raw_path, raw_file)
            trajectories = self._load_raw_file(file_path, shuffle_flag=False)
            for trajectory in trajectories:
                split = self._split_from_index(
                    traj_idx=idx,
                    train_cutoff=train_cutoff,
                    val_cutoff=val_cutoff,
                    has_val=has_val,
                )
                if split == TEST_SPLIT:
                    key = self._ic_key_from_trajectory(trajectory)
                    if key not in mapping:
                        mapping[key] = (trajectory, time_sim)
                idx += 1
        return mapping

    def _load_dataset_info(self, dataset_number: int) -> tuple[str, int, float]:
        folder = os.path.join(self.cfg.dirs.dataset_dir, self.model_flag, f"dataset_v{int(dataset_number)}")
        return self._load_dataset_info_from_folder(folder)

    def _load_dataset_info_from_folder(self, folder: str) -> tuple[str, int, float]:
        info_path = os.path.join(folder, INFO_FILE_NAME)
        if not os.path.exists(info_path):
            raise FileNotFoundError(f"info.txt not found for shared dataset: {info_path}")
        with open(info_path, "r", encoding="utf-8") as text_file:
            lines = text_file.readlines()
        validate_info_lines(lines)
        info_map = self._parse_info_lines(lines)
        total_trajectories = int(info_map["Number of different simulated trajectories"])
        time_sim = float(info_map["Time horizon of the simulations"])
        raw_path = os.path.join(folder, "raw")
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"Raw data folder not found for shared dataset: {raw_path}")
        return raw_path, total_trajectories, time_sim

    def get_preprocess_save_data(self) -> None:
        print(
            "[preprocess] Running supervised preprocessing | "
            f"time={float(self.cfg.time)} split_ratio={float(self.dataset_cfg.split_ratio)} "
            f"validation={bool(self.dataset_cfg.validation_flag)} mode={self.test_split_mode}"
        )
        if self.test_split_mode not in {"internal", "shared_dataset", "paired_common_from_datasets"}:
            raise ValueError(
                "dataset.test_split_mode must be one of: "
                "['internal', 'shared_dataset', 'paired_common_from_datasets']"
            )

        train_traj = int(self.total_init_conditions * float(self.dataset_cfg.split_ratio))
        has_val = bool(self.dataset_cfg.validation_flag)
        if self.test_split_mode in {"internal", "paired_common_from_datasets"}:
            if has_val:
                remaining = self.total_init_conditions - train_traj
                val_traj = remaining // 2
                val_cutoff = train_traj + val_traj
            else:
                val_cutoff = train_traj
        else:
            # In shared-dataset test mode, current dataset contributes only train/(optional)val.
            val_cutoff = self.total_init_conditions if has_val else train_traj

        buffers_x = {TRAIN_SPLIT: [], VAL_SPLIT: [], TEST_SPLIT: []}
        buffers_y = {TRAIN_SPLIT: [], VAL_SPLIT: [], TEST_SPLIT: []}
        trajectories_by_split = {TRAIN_SPLIT: 0, VAL_SPLIT: 0, TEST_SPLIT: 0}

        raw_files = sorted([f for f in os.listdir(self.raw_data_path) if f.endswith(".pkl")])
        if self.test_split_mode != "paired_common_from_datasets":
            traj_idx_global = 0
            for raw_idx, raw_file in enumerate(raw_files):
                if raw_idx == 0 or raw_idx % 10 == 0 or raw_idx == len(raw_files) - 1:
                    print(f"[preprocess] Processing raw file {raw_idx + 1}/{len(raw_files)}: {raw_file}")
                file_path = os.path.join(self.raw_data_path, raw_file)
                trajectories = self._load_raw_file(file_path, bool(self.dataset_cfg.shuffle))
                for trajectory in trajectories:
                    if self.test_split_mode == "internal":
                        split = self._split_from_index(
                            traj_idx=traj_idx_global,
                            train_cutoff=train_traj,
                            val_cutoff=val_cutoff,
                            has_val=has_val,
                        )
                    else:
                        if traj_idx_global < train_traj:
                            split = TRAIN_SPLIT
                        elif has_val:
                            split = VAL_SPLIT
                        else:
                            split = TRAIN_SPLIT

                    x, y = self._trajectory_to_xy(
                        trajectory,
                        float(self.cfg.time),
                        simulation_time=self.time_sim,
                    )
                    buffers_x[split].append(x)
                    buffers_y[split].append(y)
                    trajectories_by_split[split] += 1
                    traj_idx_global += 1

                    # Flush split buffers by sample count.
                    for split_name in (TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT):
                        sample_count = sum(chunk.shape[0] for chunk in buffers_x[split_name])
                        if sample_count >= self.save_freq:
                            self._flush_split_buffers(split_name, buffers_x, buffers_y)

        if self.test_split_mode == "shared_dataset":
            if self.shared_test_dataset_root in (None, "") and self.shared_test_dataset_number is None:
                raise ValueError(
                    "Set dataset.shared_test_dataset_root or dataset.shared_test_dataset_number when "
                    "test_split_mode='shared_dataset'."
                )
            if self.shared_test_dataset_root not in (None, ""):
                shared_root = str(self.shared_test_dataset_root)
                if not os.path.isabs(shared_root):
                    shared_root = os.path.abspath(os.path.join(str(self.cfg.dirs.dataset_dir), shared_root))
                shared_raw_path, _, shared_time_sim = self._load_dataset_info_from_folder(shared_root)
            else:
                shared_raw_path, _, shared_time_sim = self._load_dataset_info(int(self.shared_test_dataset_number))
            max_test = None if self.shared_test_max_trajectories is None else int(self.shared_test_max_trajectories)
            if max_test is not None and max_test <= 0:
                raise ValueError("dataset.shared_test_max_trajectories must be positive if set.")

            test_count = 0
            shared_files = sorted([f for f in os.listdir(shared_raw_path) if f.endswith(".pkl")])
            for raw_idx, raw_file in enumerate(shared_files):
                if raw_idx == 0 or raw_idx % 10 == 0 or raw_idx == len(shared_files) - 1:
                    print(
                        f"[preprocess] Processing shared-test raw file {raw_idx + 1}/{len(shared_files)}: {raw_file}"
                    )
                file_path = os.path.join(shared_raw_path, raw_file)
                trajectories = self._load_raw_file(file_path, bool(self.dataset_cfg.shuffle))
                for trajectory in trajectories:
                    if max_test is not None and test_count >= max_test:
                        break
                    x, y = self._trajectory_to_xy(
                        trajectory,
                        float(self.cfg.time),
                        simulation_time=shared_time_sim,
                    )
                    buffers_x[TEST_SPLIT].append(x)
                    buffers_y[TEST_SPLIT].append(y)
                    trajectories_by_split[TEST_SPLIT] += 1
                    test_count += 1

                    sample_count = sum(chunk.shape[0] for chunk in buffers_x[TEST_SPLIT])
                    if sample_count >= self.save_freq:
                        self._flush_split_buffers(TEST_SPLIT, buffers_x, buffers_y)
                if max_test is not None and test_count >= max_test:
                    break

        if self.test_split_mode == "paired_common_from_datasets":
            if self.paired_other_dataset_number is None:
                raise ValueError(
                    "dataset.paired_other_dataset_number must be set when "
                    "test_split_mode='paired_common_from_datasets'."
                )

            ds_a = int(self.dataset_cfg.number)
            ds_b = int(self.paired_other_dataset_number)
            ordered_pair = sorted({ds_a, ds_b})
            combined_test_map: dict[tuple[float, ...], tuple[Any, float]] = {}
            for ds_num in ordered_pair:
                part = self._collect_internal_test_trajectories(dataset_number=ds_num)
                for key, value in part.items():
                    if key not in combined_test_map:
                        combined_test_map[key] = value

            common_test_keys = set(combined_test_map.keys())

            # Rebuild train/val buffers from current dataset without any common-test ICs.
            buffers_x = {TRAIN_SPLIT: [], VAL_SPLIT: [], TEST_SPLIT: []}
            buffers_y = {TRAIN_SPLIT: [], VAL_SPLIT: [], TEST_SPLIT: []}
            trajectories_by_split = {TRAIN_SPLIT: 0, VAL_SPLIT: 0, TEST_SPLIT: 0}

            non_test_flags: list[bool] = []
            raw_files = sorted([f for f in os.listdir(self.raw_data_path) if f.endswith(".pkl")])
            for raw_idx, raw_file in enumerate(raw_files):
                if raw_idx == 0 or raw_idx % 10 == 0 or raw_idx == len(raw_files) - 1:
                    print(
                        "[preprocess] Building paired common test map | "
                        f"raw file {raw_idx + 1}/{len(raw_files)}: {raw_file}"
                    )
                file_path = os.path.join(self.raw_data_path, raw_file)
                trajectories = self._load_raw_file(file_path, shuffle_flag=False)
                for trajectory in trajectories:
                    key = self._ic_key_from_trajectory(trajectory)
                    non_test_flags.append(key not in common_test_keys)

            non_test_total = int(sum(non_test_flags))
            train_non_test = int(non_test_total * float(self.dataset_cfg.split_ratio))
            if has_val:
                val_non_test = (non_test_total - train_non_test) // 2
                val_non_test_cutoff = train_non_test + val_non_test
            else:
                val_non_test_cutoff = train_non_test

            non_test_idx = 0
            global_idx = 0
            for raw_idx, raw_file in enumerate(raw_files):
                if raw_idx == 0 or raw_idx % 10 == 0 or raw_idx == len(raw_files) - 1:
                    print(
                        "[preprocess] Rebuilding train/val without common-test ICs | "
                        f"raw file {raw_idx + 1}/{len(raw_files)}: {raw_file}"
                    )
                file_path = os.path.join(self.raw_data_path, raw_file)
                trajectories = self._load_raw_file(file_path, shuffle_flag=False)
                for trajectory in trajectories:
                    if non_test_flags[global_idx]:
                        if non_test_idx < train_non_test:
                            split = TRAIN_SPLIT
                        elif has_val and non_test_idx < val_non_test_cutoff:
                            split = VAL_SPLIT
                        else:
                            split = TRAIN_SPLIT if not has_val else VAL_SPLIT
                        x, y = self._trajectory_to_xy(
                            trajectory,
                            float(self.cfg.time),
                            simulation_time=self.time_sim,
                        )
                        buffers_x[split].append(x)
                        buffers_y[split].append(y)
                        trajectories_by_split[split] += 1
                        non_test_idx += 1

                        sample_count = sum(chunk.shape[0] for chunk in buffers_x[split])
                        if sample_count >= self.save_freq:
                            self._flush_split_buffers(split, buffers_x, buffers_y)
                    global_idx += 1

            # Append shared/common test set (identical for both compared datasets).
            ordered_test_keys = sorted(combined_test_map.keys())
            for key in ordered_test_keys:
                trajectory, source_time = combined_test_map[key]
                x, y = self._trajectory_to_xy(
                    trajectory,
                    float(self.cfg.time),
                    simulation_time=source_time,
                )
                buffers_x[TEST_SPLIT].append(x)
                buffers_y[TEST_SPLIT].append(y)
                trajectories_by_split[TEST_SPLIT] += 1
                sample_count = sum(chunk.shape[0] for chunk in buffers_x[TEST_SPLIT])
                if sample_count >= self.save_freq:
                    self._flush_split_buffers(TEST_SPLIT, buffers_x, buffers_y)

        for split_name in (TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT):
            self._flush_split_buffers(split_name, buffers_x, buffers_y, flush_all=True)

        self._finalize_stats()
        self.set_info_attributes(
            dataset_test_split_mode=self.test_split_mode,
            dataset_shared_test_number=self.shared_test_dataset_number,
            dataset_shared_test_root=self.shared_test_dataset_root,
            dataset_shared_test_max_trajectories=self.shared_test_max_trajectories,
            dataset_paired_other_number=self.paired_other_dataset_number,
            dataset_ic_key_decimals=self.ic_key_decimals,
        )
        self.set_info_attributes(
            num_of_train_files=self.train_file_count,
            num_of_training_trajectories=trajectories_by_split[TRAIN_SPLIT],
            num_of_val_files=self.val_file_count,
            num_of_validation_trajectories=trajectories_by_split[VAL_SPLIT],
            num_of_test_files=self.test_file_count,
            num_of_testing_trajectories=trajectories_by_split[TEST_SPLIT],
        )
        print(
            "[preprocess] Supervised split complete | "
            f"train_traj={trajectories_by_split[TRAIN_SPLIT]} "
            f"val_traj={trajectories_by_split[VAL_SPLIT]} "
            f"test_traj={trajectories_by_split[TEST_SPLIT]}"
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
            out_idx = self.train_file_count
        elif split == VAL_SPLIT:
            self.val_file_count += 1
            self._save_dataset(split, x_chunk, y_chunk, self.val_file_count, suffix=H5_DATA_SUFFIX)
            out_idx = self.val_file_count
        else:
            self.test_file_count += 1
            self._save_dataset(split, x_chunk, y_chunk, self.test_file_count, suffix=H5_DATA_SUFFIX)
            out_idx = self.test_file_count
        print(
            f"[preprocess] Wrote {split} chunk #{out_idx} | "
            f"x_shape={tuple(x_chunk.shape)} y_shape={tuple(y_chunk.shape)}"
        )

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
            print("[preprocess] Skipping collocation generation (dataset.new_coll_points_flag=false).")
            return None
        print("[preprocess] Generating collocation/init datasets.")

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
            print(
                "[preprocess] Wrote collocation chunk | "
                f"chunk={saved_col_files} ic_batch={chunk.shape[0]} x_col_shape={tuple(x_col.shape)}"
            )

        self.set_info_attributes(
            num_of_train_col_files=self.col_file_count,
            num_of_training_col_trajectories=int(len(init_conditions)),
            num_of_train_init_files=self.init_file_count,
        )
        print(
            "[preprocess] Collocation generation complete | "
            f"chunks={saved_col_files} trajectories={int(len(init_conditions))}"
        )
        return None
