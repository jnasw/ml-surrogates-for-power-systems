"""Stage-1 dataset-generation orchestration for static and adaptive methods."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from hydra.utils import get_original_cwd

from src.data.active_learning.logger import ExperimentLogger
from src.data.active_learning.loop import QBCConfig, run_marker_loop, run_qbc_loop
from src.data.generate.adaptive_metadata import build_marker_metadata, build_qbc_metadata
from src.data.generate.bounds import load_ic_bounds
from src.data.generate.ic_sampler import sample_initial_ics
from src.data.loaders.trajectory_dataset import TrajectoryDataset
from src.sim.ode.model_definitions import SynchronousMachineModels
from src.sim.simulator import TrajectorySimulator

if TYPE_CHECKING:
    from src.data.generate.dataset_functions import ODETrajectoryBuilder


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
    save_round_arrays: bool
    save_dataset_checkpoints: bool
    save_ensemble_checkpoints: bool
    run_dir: str
    resume_from_round: int
    resume_stage: str


def resolve_method(config: Any) -> str:
    method_name = str(getattr(getattr(config, "experiment", None), "method", "")).strip()
    if method_name not in SUPPORTED_METHODS:
        raise ValueError(
            "experiment.method must be one of: "
            "['lhs_static', 'qbc_deep_ensemble', 'marker_directed', 'qbc_marker_hybrid']"
        )
    return method_name


def run_method(config: Any, dataset_builder: ODETrajectoryBuilder, method_name: str) -> None:
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


def run_lhs_static(config: Any, dataset_builder: ODETrajectoryBuilder) -> None:
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


def run_qbc_deep_ensemble(config: Any, dataset_builder: ODETrajectoryBuilder) -> None:
    adaptive_cfg = _build_adaptive_config(config, METHOD_QBC_DEEP_ENSEMBLE)
    _run_qbc_adaptive(config, dataset_builder, adaptive_cfg)


def run_marker_directed(config: Any, dataset_builder: ODETrajectoryBuilder) -> None:
    adaptive_cfg = _build_adaptive_config(config, METHOD_MARKER_DIRECTED)
    _run_marker_adaptive(config, dataset_builder, adaptive_cfg)


def run_qbc_marker_hybrid(config: Any, dataset_builder: ODETrajectoryBuilder) -> None:
    adaptive_cfg = _build_adaptive_config(config, METHOD_QBC_MARKER_HYBRID)
    _run_qbc_adaptive(config, dataset_builder, adaptive_cfg)


def _build_adaptive_config(config: Any, method_name: str) -> AdaptiveConfig:
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
        save_round_arrays=bool(getattr(config, "qbc_save_round_arrays", True)),
        save_dataset_checkpoints=bool(getattr(config, "qbc_save_dataset_checkpoints", True)),
        save_ensemble_checkpoints=bool(getattr(config, "qbc_save_ensemble_checkpoints", True)),
        run_dir=run_dir,
        resume_from_round=int(getattr(config, "qbc_resume_from_round", -1)),
        resume_stage=str(getattr(config, "qbc_resume_stage", "next_round")),
    )

    if adaptive_cfg.resume_from_round >= 0 and not adaptive_cfg.enable_logging:
        raise ValueError("qbc_resume_from_round requires qbc_enable_logging=true.")
    if adaptive_cfg.resume_from_round >= 0 and not adaptive_cfg.save_dataset_checkpoints:
        raise ValueError("qbc_resume_from_round requires qbc_save_dataset_checkpoints=true.")
    if (
        adaptive_cfg.method_name in {METHOD_QBC_DEEP_ENSEMBLE, METHOD_QBC_MARKER_HYBRID}
        and adaptive_cfg.resume_from_round >= 0
        and adaptive_cfg.resume_stage == "same_round_post_train"
        and not adaptive_cfg.save_ensemble_checkpoints
    ):
        raise ValueError("qbc_resume_stage=same_round_post_train requires qbc_save_ensemble_checkpoints=true.")
    if adaptive_cfg.method_name == METHOD_MARKER_DIRECTED and adaptive_cfg.resume_from_round >= 0:
        raise ValueError("marker_directed currently does not support qbc_resume_from_round.")
    return adaptive_cfg


def _prepare_adaptive_run(
    config: Any,
    dataset_builder: ODETrajectoryBuilder,
    adaptive_cfg: AdaptiveConfig,
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


def _run_qbc_adaptive(config: Any, dataset_builder: ODETrajectoryBuilder, adaptive_cfg: AdaptiveConfig) -> None:
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


def _run_marker_adaptive(config: Any, dataset_builder: ODETrajectoryBuilder, adaptive_cfg: AdaptiveConfig) -> None:
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


def _setup_logger(config: Any, adaptive_cfg: AdaptiveConfig) -> ExperimentLogger | None:
    if not adaptive_cfg.enable_logging:
        print("[stage-1] Adaptive logger disabled.")
        return None

    logger = ExperimentLogger(
        run_dir=adaptive_cfg.run_dir,
        save_round_arrays=adaptive_cfg.save_round_arrays,
        save_dataset_checkpoints=adaptive_cfg.save_dataset_checkpoints,
        save_ensemble_checkpoints=adaptive_cfg.save_ensemble_checkpoints,
    )
    logger.save_config(config)
    print(
        "[stage-1] Adaptive logger enabled. "
        f"run_dir={adaptive_cfg.run_dir} "
        f"round_arrays={adaptive_cfg.save_round_arrays} "
        f"dataset_checkpoints={adaptive_cfg.save_dataset_checkpoints} "
        f"ensemble_checkpoints={adaptive_cfg.save_ensemble_checkpoints}"
    )
    return logger


def _init_adaptive_dataset(
    *,
    config: Any,
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
    config: Any,
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
