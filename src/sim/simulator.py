"""Batch trajectory simulation wrapper for active learning."""

from __future__ import annotations

from typing import Any

import numpy as np
from omegaconf import OmegaConf
from scipy.integrate import solve_ivp

from src.sim.ode.model_definitions import SynchronousMachineModels


class TrajectorySimulator:
    """Simulate trajectories for IC batches using the reference ODE model."""

    def __init__(self, config: Any):
        self.config = config
        model_cfg = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
        # Solver integration needs full-state derivatives; PINN residual mode truncates outputs.
        model_cfg.PINN_flag = False
        self.model = SynchronousMachineModels(model_cfg)
        self.t_span = (0.0, float(config.time))
        self.t_eval = np.linspace(self.t_span[0], self.t_span[1], int(config.num_of_points) + 1)
        self.output_dim = self._load_output_dim(config)

    def _load_output_dim(self, config: Any) -> int:
        guide_path = f"{config.dirs.init_conditions_dir}/modellings_guide.yaml"
        guide = OmegaConf.load(guide_path)
        for entry in guide:
            if entry.get("name") == config.model.model_flag:
                return len(entry.get("keys"))
        raise ValueError(f"Model '{config.model.model_flag}' not found in modeling guide.")

    def simulate_trajectory(self, ic_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Simulate a batch of ICs.

        Returns:
            ic_batch (unchanged as float32), trajectories with shape (N, T, S).
            ``S`` contains only modeled state outputs (keys), excluding auxiliary states.
        """
        ic_batch = np.asarray(ic_batch, dtype=np.float32)
        if ic_batch.ndim != 2:
            raise ValueError("ic_batch must be 2D.")

        trajs = []
        for ic in ic_batch:
            sol = solve_ivp(self.model.odequations, self.t_span, ic.tolist(), t_eval=self.t_eval)
            if not sol.success:
                raise RuntimeError(f"ODE solve failed: {sol.message}")
            y = np.asarray(sol.y[: self.output_dim], dtype=np.float32).T
            trajs.append(y)

        return ic_batch, np.stack(trajs, axis=0).astype(np.float32)
