"""Dataset creation utilities for ODE simulation output."""

from __future__ import annotations

import os
import pickle
import re
import time
from typing import Any, Sequence
from dataclasses import dataclass

import numpy as np
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf
from scipy.integrate import solve_ivp
from scipy.stats import qmc


from src.data.generate.adaptive_metadata import build_marker_metadata, build_qbc_metadata
from src.data.generate.bounds import load_ic_bounds
from src.data.generate.ic_sampler import sample_initial_ics
from src.data.loaders.trajectory_dataset import TrajectoryDataset
from src.methods.logger import ExperimentLogger
from src.methods.loop import QBCConfig, run_marker_loop, run_qbc_loop
from src.sim.ode.model_definitions import SynchronousMachineModels
from src.sim.simulator import TrajectorySimulator

from src.data.contracts.data_contract import (
    INFO_FILE_NAME,
    RAW_FILE_PREFIX,
    RAW_FILE_SUFFIX,
    validate_raw_trajectory,
)


def _set_time(time_horizon: float, num_of_points: int) -> tuple[tuple[float, float], np.ndarray]:
    t_span = (0.0, float(time_horizon))
    t_eval = np.linspace(t_span[0], t_span[1], int(num_of_points) + 1)
    return t_span, t_eval


class ODETrajectoryBuilder:
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
                "ic_generation_method='adaptive_iterative' is orchestrated in 00_create_dataset.py. "
                "Use 00_create_dataset.py with model.ic_generation_method=adaptive_iterative."
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
        
METHOD_LHS = "lhs_static"
METHOD_QBC_DEEP_ENSEMBLE = "qbc_deep_ensemble"
METHOD_MARKER_DIRECTED = "marker_directed"
METHOD_QBC_MARKER_HYBRID = "qbc_marker_hybrid"

SUPPORTED_METHODS = {
    METHOD_LHS,
    METHOD_QBC_DEEP_ENSEMBLE,
    METHOD_MARKER_DIRECTED,
    METHOD_QBC_MARKER_HYBRID,
}


@dataclass(frozen=True)
class AdaptiveConfig:
    method_name: str
    base_seed: int
    n0: int
    n_test: int
    m_members: int
    p_candidates: int
    k_select: int
    t_rounds: int
    enable_logging: bool
    run_dir: str
    resume_from_round: int
    resume_stage: str


def resolve_method(config) -> str:
    method_name = str(getattr(getattr(config, "experiment", None), "method", "")).strip()
    if method_name not in SUPPORTED_METHODS:
        raise ValueError(
            "experiment.method must be one of: "
            "['lhs_static', 'qbc_deep_ensemble', 'marker_directed', 'qbc_marker_hybrid']"
        )
    return method_name


def run_method(config, dataset_builder: ODETrajectoryBuilder, method_name: str) -> None:
    runners = {
        METHOD_LHS: run_lhs_static,
        METHOD_QBC_DEEP_ENSEMBLE: run_qbc_deep_ensemble,
        METHOD_MARKER_DIRECTED: run_marker_directed,
        METHOD_QBC_MARKER_HYBRID: run_qbc_marker_hybrid,
    }
    try:
        runner = runners[method_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported method '{method_name}'") from exc
    runner(config, dataset_builder)


def run_lhs_static(config, dataset_builder: ODETrajectoryBuilder) -> None:
    if dataset_builder.ic_generation_method not in {"full_factorial", "joint_lhs"}:
        raise ValueError(
            "lhs_static requires model.ic_generation_method in ['full_factorial', 'joint_lhs']. "
            f"Got '{dataset_builder.ic_generation_method}'."
        )

    print("[stage-1] Running LHS/static dataset generation path.")
    modelling_full = SynchronousMachineModels(config)
    init_conditions = dataset_builder.build_initial_conditions()
    print(f"Generated {len(init_conditions)} initial conditions using '{dataset_builder.ic_generation_method}'.")
    dataset_builder.solve_sm_model(init_conditions, modelling_full, flag_time=False)
    print(f"Dataset root: {dataset_builder.get_dataset_root_path()}")


def run_qbc_deep_ensemble(config, dataset_builder: ODETrajectoryBuilder) -> None:
    adaptive_cfg = _build_adaptive_config(config, METHOD_QBC_DEEP_ENSEMBLE)
    _run_qbc_adaptive(config, dataset_builder, adaptive_cfg)


def run_marker_directed(config, dataset_builder: ODETrajectoryBuilder) -> None:
    adaptive_cfg = _build_adaptive_config(config, METHOD_MARKER_DIRECTED)
    _run_marker_adaptive(config, dataset_builder, adaptive_cfg)


def run_qbc_marker_hybrid(config, dataset_builder: ODETrajectoryBuilder) -> None:
    adaptive_cfg = _build_adaptive_config(config, METHOD_QBC_MARKER_HYBRID)
    _run_qbc_adaptive(config, dataset_builder, adaptive_cfg)


def _build_adaptive_config(config, method_name: str) -> AdaptiveConfig:
    run_dir_raw = getattr(config, "qbc_run_dir", None)
    run_dir_cfg = ExperimentLogger.default_run_dir() if run_dir_raw in (None, "") else str(run_dir_raw)
    run_dir = run_dir_cfg if os.path.isabs(run_dir_cfg) else os.path.join(get_original_cwd(), run_dir_cfg)
    adaptive_cfg = AdaptiveConfig(
        method_name=method_name,
        base_seed=int(getattr(config.model, "seed", 0)),
        n0=int(getattr(config, "qbc_n0", 32)),
        n_test=int(getattr(config, "qbc_n_test", 0)),
        m_members=int(getattr(config, "qbc_M", 3)),
        p_candidates=int(getattr(config, "qbc_P", 256)),
        k_select=int(getattr(config, "qbc_K", 16)),
        t_rounds=int(getattr(config, "qbc_T", 3)),
        enable_logging=bool(getattr(config, "qbc_enable_logging", False)),
        run_dir=run_dir,
        resume_from_round=int(getattr(config, "qbc_resume_from_round", -1)),
        resume_stage=str(getattr(config, "qbc_resume_stage", "next_round")),
    )

    if adaptive_cfg.resume_from_round >= 0 and not adaptive_cfg.enable_logging:
        raise ValueError("qbc_resume_from_round requires qbc_enable_logging=true.")
    if adaptive_cfg.method_name == METHOD_MARKER_DIRECTED and adaptive_cfg.resume_from_round >= 0:
        raise ValueError("marker_directed currently does not support qbc_resume_from_round.")
    return adaptive_cfg


def _prepare_adaptive_run(
    config, dataset_builder: ODETrajectoryBuilder, adaptive_cfg: AdaptiveConfig
) -> tuple[TrajectoryDataset, np.ndarray, TrajectorySimulator, ExperimentLogger | None, int, Any]:
    if dataset_builder.ic_generation_method != "adaptive_iterative":
        raise ValueError(
            f"{adaptive_cfg.method_name} requires model.ic_generation_method='adaptive_iterative'. "
            f"Got '{dataset_builder.ic_generation_method}'."
        )

    print(f"[stage-1] Running adaptive path with method='{adaptive_cfg.method_name}'.")
    bounds = load_ic_bounds(config, use_nn_file=False)
    simulator = TrajectorySimulator(config)

    logger = _setup_logger(config, adaptive_cfg)
    dataset, start_round, initial_ensemble = _init_adaptive_dataset(
        config=config,
        adaptive_cfg=adaptive_cfg,
        bounds=bounds,
        simulator=simulator,
        logger=logger,
    )
    return dataset, bounds, simulator, logger, start_round, initial_ensemble


def _run_qbc_adaptive(config, dataset_builder: ODETrajectoryBuilder, adaptive_cfg: AdaptiveConfig) -> None:
    dataset, bounds, simulator, logger, start_round, initial_ensemble = _prepare_adaptive_run(
        config=config,
        dataset_builder=dataset_builder,
        adaptive_cfg=adaptive_cfg,
    )
    post_train_callback, round_callback = _build_callbacks(logger=logger)
    qbc_config = QBCConfig.from_runtime(
        config=config,
        M=adaptive_cfg.m_members,
        P=adaptive_cfg.p_candidates,
        K=adaptive_cfg.k_select,
        T=adaptive_cfg.t_rounds,
        base_seed=adaptive_cfg.base_seed,
        start_round=start_round,
    )

    print("[stage-1] Entering QBC loop.")
    dataset, _, _ = run_qbc_loop(
        dataset=dataset,
        bounds=bounds,
        simulator=simulator,
        qbc_config=qbc_config,
        config=config,
        initial_ensemble=initial_ensemble,
        post_train_callback=post_train_callback if logger is not None else None,
        round_callback=round_callback if logger is not None else None,
    )

    _persist_adaptive_dataset(
        config=config,
        dataset_builder=dataset_builder,
        adaptive_cfg=adaptive_cfg,
        dataset=dataset,
        logger=logger,
        simulator=simulator,
    )


def _run_marker_adaptive(config, dataset_builder: ODETrajectoryBuilder, adaptive_cfg: AdaptiveConfig) -> None:
    dataset, bounds, simulator, logger, _start_round, _initial_ensemble = _prepare_adaptive_run(
        config=config,
        dataset_builder=dataset_builder,
        adaptive_cfg=adaptive_cfg,
    )
    _, round_callback = _build_callbacks(logger=logger)

    print("[stage-1] Entering marker-directed loop.")
    dataset, _ = run_marker_loop(
        dataset=dataset,
        bounds=bounds,
        simulator=simulator,
        P=adaptive_cfg.p_candidates,
        K=adaptive_cfg.k_select,
        T=adaptive_cfg.t_rounds,
        base_seed=adaptive_cfg.base_seed,
        config=config,
        round_callback=round_callback if logger is not None else None,
    )

    _persist_adaptive_dataset(
        config=config,
        dataset_builder=dataset_builder,
        adaptive_cfg=adaptive_cfg,
        dataset=dataset,
        logger=logger,
        simulator=simulator,
    )


def _setup_logger(config, adaptive_cfg: AdaptiveConfig) -> ExperimentLogger | None:
    if not adaptive_cfg.enable_logging:
        print("[stage-1] Adaptive logger disabled.")
        return None

    logger = ExperimentLogger(run_dir=adaptive_cfg.run_dir)
    logger.save_config(config)
    print(f"[stage-1] Adaptive logger enabled. run_dir={adaptive_cfg.run_dir}")
    return logger


def _init_adaptive_dataset(
    *,
    config,
    adaptive_cfg: AdaptiveConfig,
    bounds: np.ndarray,
    simulator: TrajectorySimulator,
    logger: ExperimentLogger | None,
) -> tuple[TrajectoryDataset, int, Any]:
    if adaptive_cfg.resume_from_round < 0:
        dataset = _create_initial_dataset(adaptive_cfg=adaptive_cfg, bounds=bounds, simulator=simulator)
        if logger is not None:
            logger.save_dataset_checkpoint(dataset, tag="round_-01")
        print(
            "[stage-1] Initialized adaptive dataset | "
            f"n_train={dataset.n_train} n_test={dataset.n_test} "
            f"n0={adaptive_cfg.n0} n_test_cfg={adaptive_cfg.n_test} "
            f"M={adaptive_cfg.m_members} P={adaptive_cfg.p_candidates} "
            f"K={adaptive_cfg.k_select} T={adaptive_cfg.t_rounds}"
        )
        return dataset, 0, None

    if logger is None:
        raise RuntimeError("Resume requested but adaptive logger is not initialized.")

    print(
        "[stage-1] Resuming adaptive run | "
        f"resume_from_round={adaptive_cfg.resume_from_round} resume_stage={adaptive_cfg.resume_stage}"
    )
    if adaptive_cfg.resume_stage == "next_round":
        dataset = logger.load_dataset_checkpoint(tag=f"round_{adaptive_cfg.resume_from_round:03d}")
        return dataset, adaptive_cfg.resume_from_round + 1, None
    if adaptive_cfg.resume_stage == "same_round_post_train":
        prev_tag = f"round_{adaptive_cfg.resume_from_round - 1:03d}" if adaptive_cfg.resume_from_round > 0 else "round_-01"
        dataset = logger.load_dataset_checkpoint(tag=prev_tag)
        initial_ensemble = logger.load_ensemble_checkpoint(
            tag=f"round_{adaptive_cfg.resume_from_round:03d}",
            device_preference=str(getattr(config.surrogate, "device", "auto")),
        )
        return dataset, adaptive_cfg.resume_from_round, initial_ensemble
    raise ValueError("qbc_resume_stage must be one of: ['next_round', 'same_round_post_train']")


def _create_initial_dataset(
    *,
    adaptive_cfg: AdaptiveConfig,
    bounds: np.ndarray,
    simulator: TrajectorySimulator,
) -> TrajectoryDataset:
    x0 = sample_initial_ics(method="lhs", n=adaptive_cfg.n0, bounds=bounds, seed=adaptive_cfg.base_seed)
    _, y0 = simulator.simulate_trajectory(x0)

    if adaptive_cfg.n_test > 0:
        x_test = sample_initial_ics(
            method="sobol",
            n=adaptive_cfg.n_test,
            bounds=bounds,
            seed=adaptive_cfg.base_seed + 999,
        )
        _, y_test = simulator.simulate_trajectory(x_test)
    else:
        x_test = None
        y_test = None

    return TrajectoryDataset(
        train_ics=x0,
        train_trajs=y0,
        test_ics=x_test,
        test_trajs=y_test,
        time_grid=simulator.t_eval.astype(np.float32),
    )


def _build_callbacks(logger: ExperimentLogger | None):
    def _post_train_callback(**kwargs) -> None:
        if logger is None:
            return
        logger.save_ensemble_checkpoint(kwargs["ensemble"], tag=f"round_{kwargs['round_idx']:03d}")
        print(f"[stage-1] Saved ensemble checkpoint for round={kwargs['round_idx']:03d}")

    def _round_callback(**kwargs) -> None:
        summary = kwargs["summary"]
        if logger is None:
            return

        logger.log_round(
            summary=summary,
            x_cand=kwargs["x_cand"],
            scores=kwargs["scores"],
            selected_idx=kwargs["selected_idx"],
            selected_ics=kwargs["selected_ics"],
        )
        marker_arrays = {
            key: value
            for key, value in kwargs.items()
            if key.startswith("marker_") or key.startswith("hybrid_")
        }
        if marker_arrays:
            logger.save_round_arrays(summary.round_idx, **marker_arrays)
        logger.save_dataset_checkpoint(kwargs["dataset"], tag=f"round_{summary.round_idx:03d}")

    return _post_train_callback, _round_callback


def _persist_adaptive_dataset(
    *,
    config,
    dataset_builder: ODETrajectoryBuilder,
    adaptive_cfg: AdaptiveConfig,
    dataset: TrajectoryDataset,
    logger: ExperimentLogger | None,
    simulator: TrajectorySimulator,
) -> None:
    if logger is not None:
        logger.save_dataset_checkpoint(dataset, tag="final")

    if adaptive_cfg.method_name in {METHOD_QBC_DEEP_ENSEMBLE, METHOD_QBC_MARKER_HYBRID}:
        extra_info = build_qbc_metadata(
            config=config,
            n0=adaptive_cfg.n0,
            n_test=adaptive_cfg.n_test,
            M=adaptive_cfg.m_members,
            P=adaptive_cfg.p_candidates,
            K=adaptive_cfg.k_select,
            T=adaptive_cfg.t_rounds,
            final_train_size=int(dataset.n_train),
            source_run_dir=adaptive_cfg.run_dir if logger is not None else "00_create_dataset.py",
        )
    else:
        extra_info = build_marker_metadata(
            config=config,
            n0=adaptive_cfg.n0,
            P=adaptive_cfg.p_candidates,
            K=adaptive_cfg.k_select,
            T=adaptive_cfg.t_rounds,
            final_train_size=int(dataset.n_train),
            source_run_dir=adaptive_cfg.run_dir if logger is not None else "00_create_dataset.py",
        )

    dataset_builder.save_dataset_from_arrays(
        ics=dataset.train_ics,
        trajectories=dataset.train_trajs,
        time_grid=dataset.time_grid if dataset.time_grid is not None else simulator.t_eval.astype(np.float32),
        extra_info=extra_info,
    )
    print(f"Dataset root: {dataset_builder.get_dataset_root_path()}")
    print(
        f"Generated {dataset.n_train} adaptive trajectories using "
        f"'{dataset_builder.ic_generation_method}' + {adaptive_cfg.method_name}."
    )
