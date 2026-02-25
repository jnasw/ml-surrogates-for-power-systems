"""Stage-1 entrypoint: generate raw ODE simulation datasets."""

from __future__ import annotations

import os

import hydra
import numpy as np
from hydra.utils import get_original_cwd

from src.active.experiment import ExperimentLogger
from src.active.marker_loop import run_marker_loop
from src.active.qbc_loop import run_qbc_loop
from src.dataset.adaptive_metadata import build_marker_metadata, build_qbc_metadata
from src.dataset.create_dataset_functions import ODEModelling
from src.data.trajectory_dataset import TrajectoryDataset
from src.ode.model_definitions import SynchronousMachineModels
from src.sampling.bounds import load_ic_bounds
from src.sampling.ic_sampler import sample_initial_ics
from src.simulation.simulator import TrajectorySimulator


@hydra.main(config_path="src/conf", config_name="setup_dataset", version_base=None)
def main(config) -> None:
    dataset_builder = ODEModelling(config)
    # Ensure init-condition metadata path is loaded for info.txt in all modes.
    dataset_builder.create_init_conditions_info()
    print(
        "[stage-1] Starting dataset generation | "
        f"model={config.model.model_flag} ic_generation_method={dataset_builder.ic_generation_method} "
        f"seed={getattr(config.model, 'seed', None)}"
    )

    if dataset_builder.ic_generation_method != "adaptive_iterative":
        print("[stage-1] Running static dataset generation path.")
        modelling_full = SynchronousMachineModels(config)
        init_conditions = dataset_builder.build_initial_conditions()
        print(f"Generated {len(init_conditions)} initial conditions using '{dataset_builder.ic_generation_method}'.")
        dataset_builder.solve_sm_model(init_conditions, modelling_full, flag_time=False)
        print(f"Dataset root: {dataset_builder.get_dataset_root_path()}")
        return

    # Adaptive iterative (QBC) path with the same final raw dataset contract as static creation.
    method_name = str(getattr(config.experiment, "method", "qbc_deep_ensemble"))
    if method_name not in {"qbc_deep_ensemble", "marker_directed", "qbc_marker_hybrid"}:
        raise ValueError(
            "For model.ic_generation_method=adaptive_iterative, experiment.method must be one of "
            "['qbc_deep_ensemble', 'marker_directed', 'qbc_marker_hybrid']."
        )
    print(f"[stage-1] Running adaptive path with method='{method_name}'.")

    bounds = load_ic_bounds(config, use_nn_file=False)
    simulator = TrajectorySimulator(config)

    base_seed = int(getattr(config.model, "seed", 0))
    n0 = int(getattr(config, "qbc_n0", 32))
    n_test = int(getattr(config, "qbc_n_test", 0))
    M = int(getattr(config, "qbc_M", 3))
    P = int(getattr(config, "qbc_P", 256))
    K = int(getattr(config, "qbc_K", 16))
    T = int(getattr(config, "qbc_T", 3))
    enable_logging = bool(getattr(config, "qbc_enable_logging", False))
    run_dir_raw = getattr(config, "qbc_run_dir", None)
    run_dir_cfg = ExperimentLogger.default_run_dir() if run_dir_raw in (None, "") else str(run_dir_raw)
    run_dir = run_dir_cfg if os.path.isabs(run_dir_cfg) else os.path.join(get_original_cwd(), run_dir_cfg)
    resume_from_round = int(getattr(config, "qbc_resume_from_round", -1))
    resume_stage = str(getattr(config, "qbc_resume_stage", "next_round"))
    if resume_from_round >= 0 and not enable_logging:
        raise ValueError("qbc_resume_from_round requires qbc_enable_logging=true.")

    logger: ExperimentLogger | None = None
    start_round = 0
    initial_ensemble = None

    if enable_logging:
        logger = ExperimentLogger(run_dir=run_dir)
        logger.save_config(config)
        print(f"[stage-1] Adaptive logger enabled. run_dir={run_dir}")
    else:
        print("[stage-1] Adaptive logger disabled.")

    dataset: TrajectoryDataset
    if enable_logging and resume_from_round >= 0:
        print(
            "[stage-1] Resuming adaptive run | "
            f"resume_from_round={resume_from_round} resume_stage={resume_stage}"
        )
        if resume_stage == "next_round":
            dataset = logger.load_dataset_checkpoint(tag=f"round_{resume_from_round:03d}")
            start_round = resume_from_round + 1
        elif resume_stage == "same_round_post_train":
            prev_tag = f"round_{resume_from_round - 1:03d}" if resume_from_round > 0 else "round_-01"
            dataset = logger.load_dataset_checkpoint(tag=prev_tag)
            initial_ensemble = logger.load_ensemble_checkpoint(
                tag=f"round_{resume_from_round:03d}",
                device_preference=str(getattr(config.surrogate, "device", "auto")),
            )
            start_round = resume_from_round
        else:
            raise ValueError("qbc_resume_stage must be one of: ['next_round', 'same_round_post_train']")
    else:
        x0 = sample_initial_ics(method="lhs", n=n0, bounds=bounds, seed=base_seed)
        _, y0 = simulator.simulate_trajectory(x0)
        if n_test > 0:
            x_test = sample_initial_ics(method="sobol", n=n_test, bounds=bounds, seed=base_seed + 999)
            _, y_test = simulator.simulate_trajectory(x_test)
            dataset = TrajectoryDataset(
                train_ics=x0,
                train_trajs=y0,
                test_ics=x_test,
                test_trajs=y_test,
                time_grid=simulator.t_eval.astype(np.float32),
            )
        else:
            dataset = TrajectoryDataset(
                train_ics=x0,
                train_trajs=y0,
                test_ics=None,
                test_trajs=None,
                time_grid=simulator.t_eval.astype(np.float32),
            )
        if logger is not None:
            logger.save_dataset_checkpoint(dataset, tag="round_-01")
        print(
            "[stage-1] Initialized adaptive dataset | "
            f"n_train={dataset.n_train} n_test={dataset.n_test} "
            f"n0={n0} n_test_cfg={n_test} M={M} P={P} K={K} T={T}"
        )

    if method_name == "marker_directed" and resume_from_round >= 0:
        raise ValueError("marker_directed currently does not support qbc_resume_from_round.")

    def _post_train_callback(**kwargs) -> None:
        if logger is not None:
            logger.save_ensemble_checkpoint(kwargs["ensemble"], tag=f"round_{kwargs['round_idx']:03d}")
            print(f"[stage-1] Saved ensemble checkpoint for round={kwargs['round_idx']:03d}")

    def _round_callback(**kwargs) -> None:
        summary = kwargs["summary"]
        timing_parts = []
        for key in (
            "train_seconds",
            "candidate_generation_seconds",
            "candidate_simulation_seconds",
            "acquisition_seconds",
            "selected_simulation_seconds",
            "eval_seconds",
            "round_seconds",
        ):
            val = getattr(summary, key, None)
            if val is not None:
                timing_parts.append(f"{key.replace('_seconds', '')}={float(val):.3f}s")
        timing_text = (" | " + ", ".join(timing_parts)) if timing_parts else ""
        print(
            "[stage-1] Round summary | "
            f"round={summary.round_idx:03d} train_size={summary.train_size} "
            f"selected={len(kwargs['selected_idx'])}{timing_text}"
        )
        if logger is not None:
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

    if method_name in {"qbc_deep_ensemble", "qbc_marker_hybrid"}:
        print("[stage-1] Entering QBC loop.")
        dataset, _, _ = run_qbc_loop(
            dataset=dataset,
            bounds=bounds,
            simulator=simulator,
            M=M,
            P=P,
            K=K,
            T=T,
            base_seed=base_seed,
            config=config,
            start_round=start_round,
            initial_ensemble=initial_ensemble,
            post_train_callback=_post_train_callback if logger is not None else None,
            round_callback=_round_callback if logger is not None else None,
        )
    else:
        print("[stage-1] Entering marker-directed loop.")
        dataset, _ = run_marker_loop(
            dataset=dataset,
            bounds=bounds,
            simulator=simulator,
            P=P,
            K=K,
            T=T,
            base_seed=base_seed,
            config=config,
            round_callback=_round_callback if logger is not None else None,
        )

    if logger is not None:
        logger.save_dataset_checkpoint(dataset, tag="final")

    if method_name in {"qbc_deep_ensemble", "qbc_marker_hybrid"}:
        extra_info = build_qbc_metadata(
            config=config,
            n0=n0,
            n_test=n_test,
            M=M,
            P=P,
            K=K,
            T=T,
            final_train_size=int(dataset.n_train),
            source_run_dir=run_dir if logger is not None else "create_dataset.py",
        )
    else:
        extra_info = build_marker_metadata(
            config=config,
            n0=n0,
            P=P,
            K=K,
            T=T,
            final_train_size=int(dataset.n_train),
            source_run_dir=run_dir if logger is not None else "create_dataset.py",
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
        f"'{dataset_builder.ic_generation_method}' + {method_name}."
    )


if __name__ == "__main__":
    main()
