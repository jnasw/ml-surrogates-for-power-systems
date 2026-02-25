"""Dataset creation utilities for ODE simulation output."""

from __future__ import annotations

import os
import pickle
import re
import time
from typing import Any, Sequence

import numpy as np
from omegaconf import OmegaConf
from scipy.integrate import solve_ivp
from scipy.stats import qmc

from src.dataset.data_contract import (
    INFO_FILE_NAME,
    RAW_FILE_PREFIX,
    RAW_FILE_SUFFIX,
    validate_raw_trajectory,
)


def _set_time(time_horizon: float, num_of_points: int) -> tuple[tuple[float, float], np.ndarray]:
    t_span = (0.0, float(time_horizon))
    t_eval = np.linspace(t_span[0], t_span[1], int(num_of_points) + 1)
    return t_span, t_eval


class ODEModelling:
    """Create initial conditions, solve ODEs, and persist raw trajectory data."""

    def __init__(self, config: Any):
        self.config = config
        self.model_flag = str(config.model.model_flag)
        self.time = float(config.time)
        self.num_of_points = int(config.num_of_points)
        self.init_condition_bounds = int(config.model.init_condition_bounds)
        self.sampling = str(config.model.sampling)
        self.ic_generation_method = str(getattr(config.model, "ic_generation_method", "full_factorial"))
        self.ic_num_samples = getattr(config.model, "ic_num_samples", None)
        self.torch_mode = bool(config.model.torch)
        self.seed = None if not hasattr(config.model, "seed") else int(config.model.seed)
        self.save_freq = int(getattr(config, "save_freq", 1000))

        self._save_flag = True
        self._load_keys()

    def _ensure_dataset_folder(self) -> None:
        dataset_dir = self.config.dirs.dataset_dir
        model_dir = os.path.join(dataset_dir, self.model_flag)
        dataset_version_dir = self._next_dataset_version_folder(model_dir)
        self.dataset_root_path = dataset_version_dir
        self.dataset_folder_path = os.path.join(dataset_version_dir, "raw")
        os.makedirs(self.dataset_folder_path, exist_ok=False)
        print(f"Created dataset folder: {dataset_version_dir}")

    def get_dataset_root_path(self) -> str:
        """Return the dataset root path after the first save operation."""
        if not hasattr(self, "dataset_root_path"):
            raise RuntimeError("Dataset root path is not available before saving data.")
        return str(self.dataset_root_path)

    def _write_info_file(self, num_files: int, total_trajectories: int, extra_info: dict[str, Any] | None = None) -> None:
        info_path = os.path.join(os.path.dirname(self.dataset_folder_path), INFO_FILE_NAME)
        with open(info_path, "w", encoding="utf-8") as info:
            info.write(f"Number of files: {num_files}\n")
            info.write(f"Initial conditions file: {self.init_conditions_path}\n")
            info.write(f"Number of different simulated trajectories: {total_trajectories}\n")
            info.write(f"Time horizon of the simulations: {self.time}\n")
            info.write(f"Number of points in the each simulation: {self.num_of_points}\n")
            info.write(f"IC generation method: {self.ic_generation_method}\n")
            info.write(f"IC per-variable sampling method: {self.sampling}\n")
            info.write(f"IC joint sample count override: {self.ic_num_samples}\n")
            if extra_info:
                for key, value in extra_info.items():
                    info.write(f"{key}: {value}\n")

    def _load_keys(self) -> None:
        guide_path = os.path.join(self.config.dirs.init_conditions_dir, "modellings_guide.yaml")
        if not os.path.exists(guide_path):
            raise FileNotFoundError(f"Modeling guide not found: {guide_path}")
        modeling_guide = OmegaConf.load(guide_path)

        keys = None
        keys_ext = None
        for model in modeling_guide:
            if model.get("name") == self.model_flag:
                keys = model.get("keys")
                keys_ext = model.get("keys_ext")
                self.full_name = model.get("full_name")
                break

        if not keys:
            raise ValueError(f"Model '{self.model_flag}' not found in modeling guide.")

        self.keys_length = len(keys)
        self.keys_ext_length = len(keys_ext) if keys_ext else 0
        self.keys = list(keys) + (list(keys_ext) if keys_ext else [])

    def _check_ic_yaml(self, init_conditions: Sequence[Any]) -> None:
        for i, condition in enumerate(init_conditions):
            name = condition.get("name")
            if name not in self.keys:
                raise ValueError(f"Variable '{name}' not found in modeling guide keys.")
            if name != self.keys[i]:
                raise ValueError(
                    f"Variable order mismatch at index {i}: got '{name}', expected '{self.keys[i]}'."
                )
            iterations = condition.get("iterations")
            if not isinstance(iterations, int) or iterations <= 0:
                raise ValueError(f"Variable '{name}' iterations must be a positive integer.")

    def _sample_points(self, num_samples: int) -> np.ndarray:
        if self.sampling == "Random":
            rng = np.random.default_rng(self.seed)
            return rng.uniform(0.0, 1.0, num_samples)
        if self.sampling == "Linear":
            return np.linspace(0.0, 1.0, num_samples)
        if self.sampling == "Lhs":
            sampler = qmc.LatinHypercube(d=1, seed=self.seed)
            return sampler.random(n=num_samples).flatten()
        raise ValueError(f"Unsupported sampling method: {self.sampling}")

    def build_full_factorial_ic_table(
        self, set_of_values: Sequence[Sequence[float]], iterations: Sequence[int]
    ) -> list[list[float]]:
        sampled_points: list[np.ndarray] = []

        for value_range, num_samples in zip(set_of_values, iterations):
            if len(value_range) == 1 or num_samples == 1:
                sampled_values = np.array([float(value_range[0])], dtype=float)
            else:
                v_min, v_max = float(value_range[0]), float(value_range[1])
                points = self._sample_points(num_samples)
                sampled_values = v_min + points * (v_max - v_min)
            sampled_points.append(sampled_values)

        table = np.array(np.meshgrid(*sampled_points)).T.reshape(-1, len(set_of_values))
        return table.tolist()

    def build_joint_lhs_ic_table(
        self, set_of_values: Sequence[Sequence[float]], iterations: Sequence[int]
    ) -> list[list[float]]:
        """Build initial-condition vectors with true D-dimensional LHS.

        Sample count is taken from ``model.ic_num_samples`` if provided.
        Otherwise it defaults to the full-factorial count (product of iterations),
        which keeps dataset size roughly comparable when switching methods.
        """
        if self.ic_num_samples is None:
            num_samples = int(np.prod(np.array(iterations, dtype=int)))
        else:
            num_samples = int(self.ic_num_samples)
        if num_samples <= 0:
            raise ValueError("model.ic_num_samples must be a positive integer.")

        lower_bounds = []
        upper_bounds = []
        variable_indices = []
        constant_values: dict[int, float] = {}

        for idx, value_range in enumerate(set_of_values):
            if len(value_range) == 1:
                constant_values[idx] = float(value_range[0])
            else:
                variable_indices.append(idx)
                lower_bounds.append(float(value_range[0]))
                upper_bounds.append(float(value_range[1]))

        total_dims = len(set_of_values)
        table = np.zeros((num_samples, total_dims), dtype=float)

        if variable_indices:
            sampler = qmc.LatinHypercube(d=len(variable_indices), seed=self.seed)
            unit_samples = sampler.random(n=num_samples)
            scaled_samples = qmc.scale(unit_samples, l_bounds=lower_bounds, u_bounds=upper_bounds)
            for col, original_idx in enumerate(variable_indices):
                table[:, original_idx] = scaled_samples[:, col]

        for original_idx, value in constant_values.items():
            table[:, original_idx] = value

        return table.tolist()

    def create_init_conditions_info(self) -> tuple[list[str], list[list[float]], list[int]]:
        ic_dir = self.config.dirs.init_conditions_dir
        filename = (
            f"nn_init_cond{self.init_condition_bounds}.yaml"
            if self.torch_mode
            else f"init_cond{self.init_condition_bounds}.yaml"
        )
        init_conditions_path = os.path.join(ic_dir, self.model_flag, filename)
        if not os.path.exists(init_conditions_path):
            raise FileNotFoundError(f"Initial conditions file not found: {init_conditions_path}")

        init_conditions = OmegaConf.load(init_conditions_path)
        self._check_ic_yaml(init_conditions)
        self.init_conditions_path = init_conditions_path

        variables: list[str] = []
        set_of_values: list[list[float]] = []
        iterations: list[int] = []

        for condition in init_conditions:
            value_range = list(condition["range"])
            iteration_count = 1 if len(value_range) == 1 else int(condition["iterations"])
            variables.append(str(condition["name"]))
            set_of_values.append([float(v) for v in value_range])
            iterations.append(iteration_count)

        return variables, set_of_values, iterations

    def build_initial_conditions(self) -> list[list[float]]:
        """Build initial conditions using the configured IC generation method.

        Supported methods:
        - ``full_factorial``: sample each variable independently and take cartesian product.
        - ``joint_lhs``: sample full IC vectors directly with D-dimensional LHS.
        - ``adaptive_iterative``: placeholder for future adaptive refinement workflows.
        """
        _, set_of_values, iterations = self.create_init_conditions_info()

        if self.ic_generation_method == "full_factorial":
            return self.build_full_factorial_ic_table(set_of_values, iterations)
        if self.ic_generation_method == "joint_lhs":
            return self.build_joint_lhs_ic_table(set_of_values, iterations)
        if self.ic_generation_method == "adaptive_iterative":
            raise NotImplementedError(
                "ic_generation_method='adaptive_iterative' is orchestrated in create_dataset.py. "
                "Use create_dataset.py with model.ic_generation_method=adaptive_iterative."
            )
        raise ValueError(
            f"Unsupported ic_generation_method '{self.ic_generation_method}'. "
            "Allowed: ['full_factorial', 'joint_lhs', 'adaptive_iterative']"
        )

    def solve(self, x0: list[float], modelling_full: Any, method: bool = True):
        if method:
            return solve_ivp(modelling_full.odequations, self.t_span, x0, t_eval=self.t_eval)
        x0 = list(x0)
        x0[1] = x0[1] / modelling_full.omega_B
        return solve_ivp(modelling_full.odequations_v2, self.t_span, x0, t_eval=self.t_eval)

    def solve_sm_model(self, init_conditions: list[list[float]], modelling_full: Any, flag_time: bool = False) -> None:
        self.t_span, self.t_eval = _set_time(self.time, self.num_of_points)
        self.total_init_conditions = len(init_conditions)
        self._save_flag = True

        solution_all = []
        time_list = []
        start = time.time()
        start_iter = start

        for i, init_condition in enumerate(init_conditions):
            solution = self.solve(init_condition, modelling_full, method=True)
            solution_all.append(solution)

            should_flush = ((i % self.save_freq == 0 and i > 0) or i == (self.total_init_conditions - 1))
            if should_flush:
                self.save_dataset(solution_all, i)
                solution_all = []

            if flag_time:
                now = time.time()
                time_list.append(now - start_iter)
                start_iter = now

        if flag_time and time_list:
            total = time.time() - start
            print(f"Mean solve time per trajectory: {np.mean(time_list):.6f}s (std={np.std(time_list):.6f}s)")
            print(f"Total solve time for {self.total_init_conditions} trajectories: {total:.6f}s")

    def _next_dataset_version_folder(self, model_dir: str) -> str:
        os.makedirs(model_dir, exist_ok=True)
        versions = []
        for entry in os.listdir(model_dir):
            full = os.path.join(model_dir, entry)
            if os.path.isdir(full):
                match = re.match(r"dataset_v(\d+)$", entry)
                if match:
                    versions.append(int(match.group(1)))
        next_version = (max(versions) + 1) if versions else 1
        return os.path.join(model_dir, f"dataset_v{next_version}")

    def save_dataset(self, solution_batch: list[Any], iteration: int) -> None:
        if self._save_flag:
            self._ensure_dataset_folder()
            self._save_flag = False

        path = os.path.join(self.dataset_folder_path, f"{RAW_FILE_PREFIX}{iteration}{RAW_FILE_SUFFIX}")
        dataset_payload = []
        for solution in solution_batch:
            trajectory = [solution.t]
            for channel in solution.y:
                trajectory.append(channel)
            validate_raw_trajectory(trajectory)
            dataset_payload.append(trajectory)

        with open(path, "wb") as handle:
            pickle.dump(dataset_payload, handle)

        if iteration == self.total_init_conditions - 1:
            raw_files = [
                f
                for f in os.listdir(self.dataset_folder_path)
                if os.path.isfile(os.path.join(self.dataset_folder_path, f))
            ]
            num_files = len(raw_files)
            self._write_info_file(num_files=num_files, total_trajectories=self.total_init_conditions)

            print(f"Saved dataset in {os.path.dirname(self.dataset_folder_path)} with {num_files} raw files.")

    def save_dataset_from_arrays(
        self,
        ics: np.ndarray,
        trajectories: np.ndarray,
        time_grid: np.ndarray,
        extra_info: dict[str, Any] | None = None,
    ) -> None:
        """Persist trajectories from in-memory arrays using the standard raw dataset contract.

        Args:
            ics: Initial conditions with shape (N, D). Used for consistency checks.
            trajectories: Simulated state trajectories with shape (N, T, S).
            time_grid: Shared time grid with shape (T,).
            extra_info: Optional extra metadata written to info.txt as ``key: value`` lines.
        """
        ics = np.asarray(ics, dtype=np.float32)
        trajectories = np.asarray(trajectories, dtype=np.float32)
        time_grid = np.asarray(time_grid, dtype=np.float32)
        if ics.ndim != 2:
            raise ValueError(f"ics must be 2D, got shape {ics.shape}")
        if trajectories.ndim != 3:
            raise ValueError(f"trajectories must be 3D, got shape {trajectories.shape}")
        if time_grid.ndim != 1:
            raise ValueError(f"time_grid must be 1D, got shape {time_grid.shape}")
        if ics.shape[0] != trajectories.shape[0]:
            raise ValueError("ics and trajectories must have the same first dimension.")
        if trajectories.shape[1] != time_grid.shape[0]:
            raise ValueError("trajectory time dimension must match time_grid length.")
        if trajectories.shape[2] not in (self.keys_length, len(self.keys)):
            raise ValueError(
                "Expected trajectory state dimension to match either modeled states "
                f"({self.keys_length}) or full channel count ({len(self.keys)}), "
                f"got {trajectories.shape[2]}."
            )
        if ics.shape[1] < self.keys_length:
            raise ValueError(
                f"ics must contain at least {self.keys_length} state dimensions, got {ics.shape[1]}."
            )
        if trajectories.shape[2] == self.keys_length and self.keys_ext_length > 0 and ics.shape[1] < len(self.keys):
            raise ValueError(
                "To export full raw channel layout, ICs must include extended keys "
                f"({len(self.keys)} total dims), got {ics.shape[1]}."
            )

        self.total_init_conditions = int(trajectories.shape[0])
        self._ensure_dataset_folder()

        batch: list[list[np.ndarray]] = []
        file_counter = 0
        for i in range(self.total_init_conditions):
            traj = trajectories[i]
            if not np.allclose(traj[0], ics[i, : self.keys_length], rtol=1e-5, atol=1e-5):
                raise ValueError(
                    "Trajectory initial state does not match provided IC values at index "
                    f"{i}. Ensure trajectories are generated from the same IC set."
                )

            # Static datasets serialize all channels (keys + keys_ext).
            # If adaptive trajectories only contain modeled states (keys), append keys_ext as constants from ICs.
            if trajectories.shape[2] == self.keys_length and self.keys_ext_length > 0:
                ext_vals = ics[i, self.keys_length : len(self.keys)].astype(np.float32)
                ext_traj = np.tile(ext_vals[None, :], (traj.shape[0], 1))
                full_traj = np.concatenate((traj, ext_traj), axis=1)
            else:
                full_traj = traj

            record: list[np.ndarray] = [time_grid]
            for s in range(full_traj.shape[1]):
                record.append(full_traj[:, s])
            validate_raw_trajectory(record)
            batch.append(record)

            flush = (len(batch) >= self.save_freq) or (i == self.total_init_conditions - 1)
            if flush:
                end_idx = i
                out_path = os.path.join(self.dataset_folder_path, f"{RAW_FILE_PREFIX}{end_idx}{RAW_FILE_SUFFIX}")
                with open(out_path, "wb") as handle:
                    pickle.dump(batch, handle)
                batch = []
                file_counter += 1

        self._write_info_file(
            num_files=file_counter,
            total_trajectories=self.total_init_conditions,
            extra_info=extra_info,
        )
        print(f"Saved dataset in {os.path.dirname(self.dataset_folder_path)} with {file_counter} raw files.")

    def load_dataset(self, relative_path: str):
        path = os.path.join(self.config.dirs.dataset_dir, self.model_flag, relative_path)
        with open(path, "rb") as handle:
            return pickle.load(handle)
