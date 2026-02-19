"""ODE model definitions for supported synchronous machine variants."""

from __future__ import annotations

import os
from typing import Any, Sequence

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.integrate import solve_ivp


SUPPORTED_MODELS = {"SM4", "SM6", "SM_AVR_GOV"}


def _set_time(time_horizon: float, num_of_points: int) -> tuple[tuple[float, float], np.ndarray]:
    t_span = (0.0, float(time_horizon))
    t_eval = np.linspace(t_span[0], t_span[1], int(num_of_points) + 1)
    return t_span, t_eval


class SynchronousMachineModels:
    """Synchronous machine ODEs for SM4, SM6 and SM_AVR_GOV."""

    def __init__(self, config: Any):
        self.time_horizon = float(config.time)
        self.num_of_points = int(config.num_of_points)
        self.params_dir = str(config.dirs.params_dir)
        self.machine_num = int(config.model.machine_num)
        self.model_flag = str(config.model.model_flag)
        self.PINN_flag = bool(config.PINN_flag)

        if self.model_flag not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model_flag '{self.model_flag}'. Allowed values: {sorted(SUPPORTED_MODELS)}"
            )

        self._define_machine_params()
        self._define_system_params()

    def _load_yaml(self, path: str) -> Any:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required parameter file: {path}")
        return OmegaConf.load(path)

    def _set_attrs(self, source: Any, names: Sequence[str]) -> None:
        for name in names:
            if not hasattr(source, name):
                raise KeyError(f"Missing parameter '{name}' in loaded yaml.")
            setattr(self, name, getattr(source, name))

    def _define_machine_params(self) -> None:
        machine_params_path = os.path.join(self.params_dir, f"machine{self.machine_num}.yaml")
        machine_params = self._load_yaml(machine_params_path)

        self._set_attrs(
            machine_params,
            ["X_d_dash", "X_q_dash", "H", "D", "T_d_dash", "X_d", "T_q_dash", "X_q", "E_fd", "P_m"],
        )

        if self.model_flag == "SM6":
            self._set_attrs(
                machine_params,
                ["T_d_dash_dash", "T_q_dash_dash", "X_d_dash_dash", "X_q_dash_dash"],
            )

        if self.model_flag == "SM_AVR_GOV":
            avr_params = self._load_yaml(os.path.join(self.params_dir, "avr.yaml"))
            self._set_attrs(avr_params, ["K_A", "T_A", "K_E", "T_E", "K_F", "T_F", "V_ref"])

            gov_params = self._load_yaml(os.path.join(self.params_dir, "gov.yaml"))
            self._set_attrs(gov_params, ["P_c", "R_d", "T_ch", "T_sv"])

    def _define_system_params(self) -> None:
        system_params = self._load_yaml(os.path.join(self.params_dir, "system_bus.yaml"))
        self._set_attrs(system_params, ["omega_B", "Rs", "Re", "Xep"])

    def calculate_currents(
        self,
        theta: float | torch.Tensor,
        E_d_dash: float | torch.Tensor,
        E_q_dash: float | torch.Tensor,
        Vs: float | torch.Tensor,
        theta_vs: float | torch.Tensor,
    ) -> tuple[float | torch.Tensor, float | torch.Tensor]:
        alpha = [
            [self.Rs + self.Re, -(self.X_q_dash + self.Xep)],
            [self.X_d_dash + self.Xep, self.Rs + self.Re],
        ]
        inv_alpha = np.linalg.inv(alpha)

        if isinstance(theta, torch.Tensor):
            beta = [[E_d_dash - Vs * torch.sin(theta - theta_vs)], [E_q_dash - Vs * torch.cos(theta - theta_vs)]]
        else:
            beta = [[E_d_dash - Vs * np.sin(theta - theta_vs)], [E_q_dash - Vs * np.cos(theta - theta_vs)]]

        I_d = inv_alpha[0][0] * beta[0][0] + inv_alpha[0][1] * beta[1][0]
        I_q = inv_alpha[1][0] * beta[0][0] + inv_alpha[1][1] * beta[1][0]
        return I_d, I_q

    def calculate_voltages(
        self,
        theta: float | torch.Tensor,
        I_d: float | torch.Tensor,
        I_q: float | torch.Tensor,
        Vs: float | torch.Tensor,
        theta_vs: float | torch.Tensor,
    ) -> float | torch.Tensor:
        if isinstance(theta, torch.Tensor):
            V_d = self.Re * I_d - self.Xep * I_q + Vs * torch.sin(theta - theta_vs)
            V_q = self.Re * I_q + self.Xep * I_d + Vs * torch.cos(theta - theta_vs)
            return torch.sqrt(V_d**2 + V_q**2)

        V_d = self.Re * I_d - self.Xep * I_q + Vs * np.sin(theta - theta_vs)
        V_q = self.Re * I_q + self.Xep * I_d + Vs * np.cos(theta - theta_vs)
        return np.sqrt(V_d**2 + V_q**2)

    def odequations(self, _t: float, x: Sequence[float]) -> list[float]:
        if self.model_flag == "SM4":
            theta, omega, E_d_dash, E_q_dash, Vs, theta_vs = x
        elif self.model_flag == "SM6":
            theta, omega, E_d_dash, E_q_dash, E_d_dash_dash, E_q_dash_dash, Vs, theta_vs = x
        else:  # SM_AVR_GOV
            theta, omega, E_d_dash, E_q_dash, R_F, V_r, E_fd, P_sv, P_m, Vs, theta_vs = x

        I_d, I_q = self.calculate_currents(theta, E_d_dash, E_q_dash, Vs, theta_vs)
        if self.model_flag == "SM_AVR_GOV":
            V_t = self.calculate_voltages(theta, I_d, I_q, Vs, theta_vs)

        dtheta_dt = omega
        dVs_dt = 0.0
        dtheta_vs_dt = 0.0

        domega_dt = (self.omega_B / (2 * self.H)) * (
            P_m - E_d_dash * I_d - E_q_dash * I_q - (self.X_q_dash - self.X_d_dash) * I_q * I_d - self.D * omega
        ) if self.model_flag == "SM_AVR_GOV" else (self.omega_B / (2 * self.H)) * (
            self.P_m
            - E_d_dash * I_d
            - E_q_dash * I_q
            - (self.X_q_dash - self.X_d_dash) * I_q * I_d
            - self.D * omega
        )

        dE_q_dash_dt = (1 / self.T_d_dash) * (-E_q_dash - I_d * (self.X_d - self.X_d_dash) + self.E_fd)
        dE_d_dash_dt = (1 / self.T_q_dash) * (-E_d_dash + I_q * (self.X_q - self.X_q_dash))

        if self.model_flag == "SM4":
            if self.PINN_flag:
                return [dtheta_dt, domega_dt, dE_d_dash_dt, dE_q_dash_dt]
            return [dtheta_dt, domega_dt, dE_d_dash_dt, dE_q_dash_dt, dVs_dt, dtheta_vs_dt]

        if self.model_flag == "SM6":
            dE_q_dash_dash_dt = (1 / self.T_d_dash_dash) * (
                E_q_dash - E_q_dash_dash + I_d * (self.X_d_dash - self.X_d_dash_dash)
            )
            dE_d_dash_dash_dt = (1 / self.T_q_dash_dash) * (
                E_d_dash - E_d_dash_dash - I_q * (self.X_q_dash - self.X_q_dash_dash)
            )
            if self.PINN_flag:
                return [dtheta_dt, domega_dt, dE_d_dash_dt, dE_q_dash_dt, dE_d_dash_dash_dt, dE_q_dash_dash_dt]
            return [
                dtheta_dt,
                domega_dt,
                dE_d_dash_dt,
                dE_q_dash_dt,
                dE_d_dash_dash_dt,
                dE_q_dash_dash_dt,
                dVs_dt,
                dtheta_vs_dt,
            ]

        dR_F_dt = (1 / self.T_F) * (-R_F + (self.K_F / self.T_F) * E_fd)
        dV_r_dt = (1 / self.T_A) * (
            -V_r + (self.K_A * R_F) - (self.K_A * self.K_F / self.T_F) * E_fd + self.K_A * (self.V_ref - V_t)
        )
        dE_fd_dt = (1 / self.T_E) * (-(self.K_E + 0.098 * np.e ** (E_fd * 0.55)) * E_fd + V_r)
        dP_m_dt = (1 / self.T_ch) * (-P_m + P_sv)
        dP_sv_dt = (1 / self.T_sv) * (-P_sv + self.P_c - (1 / self.R_d) * (omega / self.omega_B))

        if self.PINN_flag:
            return [dtheta_dt, domega_dt, dE_d_dash_dt, dE_q_dash_dt, dR_F_dt, dV_r_dt, dE_fd_dt, dP_sv_dt, dP_m_dt]
        return [
            dtheta_dt,
            domega_dt,
            dE_d_dash_dt,
            dE_q_dash_dt,
            dR_F_dt,
            dV_r_dt,
            dE_fd_dt,
            dP_sv_dt,
            dP_m_dt,
            dVs_dt,
            dtheta_vs_dt,
        ]

    def odequations_v2(self, _t: float, x: Sequence[float]) -> list[float]:
        if self.model_flag == "SM4":
            theta, omega, E_d_dash, E_q_dash = x
        elif self.model_flag == "SM6":
            theta, omega, E_d_dash, E_q_dash, E_d_dash_dash, E_q_dash_dash = x
        else:
            theta, omega, E_d_dash, E_q_dash, R_F, V_r, E_fd, P_sv, P_m = x

        I_d, I_q = self.calculate_currents(theta, E_d_dash, E_q_dash, 1.0, 0.0)
        if self.model_flag == "SM_AVR_GOV":
            V_t = self.calculate_voltages(theta, I_d, I_q, 1.0, 0.0)

        dtheta_dt = omega * self.omega_B
        dVs_dt = 0.0
        dtheta_vs_dt = 0.0

        domega_dt = (1 / (2 * self.H)) * (
            P_m - E_d_dash * I_d - E_q_dash * I_q - (self.X_q_dash - self.X_d_dash) * I_q * I_d - self.D * omega * self.omega_B
        ) if self.model_flag == "SM_AVR_GOV" else (1 / (2 * self.H)) * (
            self.P_m
            - E_d_dash * I_d
            - E_q_dash * I_q
            - (self.X_q_dash - self.X_d_dash) * I_q * I_d
            - self.D * omega * self.omega_B
        )

        dE_q_dash_dt = (1 / self.T_d_dash) * (-E_q_dash - I_d * (self.X_d - self.X_d_dash) + self.E_fd)
        dE_d_dash_dt = (1 / self.T_q_dash) * (-E_d_dash + I_q * (self.X_q - self.X_q_dash))

        if self.model_flag == "SM4":
            return [dtheta_dt, domega_dt, dE_d_dash_dt, dE_q_dash_dt, dVs_dt, dtheta_vs_dt]

        if self.model_flag == "SM6":
            dE_q_dash_dash_dt = (1 / self.T_d_dash_dash) * (
                E_q_dash - E_q_dash_dash + I_d * (self.X_d_dash - self.X_d_dash_dash)
            )
            dE_d_dash_dash_dt = (1 / self.T_q_dash_dash) * (
                E_d_dash - E_d_dash_dash - I_q * (self.X_q_dash - self.X_q_dash_dash)
            )
            return [
                dtheta_dt,
                domega_dt,
                dE_d_dash_dt,
                dE_q_dash_dt,
                dE_d_dash_dash_dt,
                dE_q_dash_dash_dt,
                dVs_dt,
                dtheta_vs_dt,
            ]

        dR_F_dt = (1 / self.T_F) * (-R_F + (self.K_F / self.T_F) * E_fd)
        dV_r_dt = (1 / self.T_A) * (
            -V_r + (self.K_A * R_F) - (self.K_A * self.K_F / self.T_F) * E_fd + self.K_A * (self.V_ref - V_t)
        )
        dE_fd_dt = (1 / self.T_E) * (-(self.K_E + 0.098 * np.e ** (E_fd * 0.55)) * E_fd + V_r)
        dP_m_dt = (1 / self.T_ch) * (-P_m + P_sv)
        dP_sv_dt = (1 / self.T_sv) * (-P_sv + self.P_c - (1 / self.R_d) * omega)

        return [
            dtheta_dt,
            domega_dt,
            dE_d_dash_dt,
            dE_q_dash_dt,
            dR_F_dt,
            dV_r_dt,
            dE_fd_dt,
            dP_sv_dt,
            dP_m_dt,
            dVs_dt,
            dtheta_vs_dt,
        ]

    def solve(self, x0: list[float], method: bool = True):
        t_span, t_eval = _set_time(self.time_horizon, self.num_of_points)
        if method:
            return solve_ivp(self.odequations, t_span, x0, t_eval=t_eval)

        x0 = list(x0)
        x0[1] = x0[1] / self.omega_B
        return solve_ivp(self.odequations_v2, t_span, x0, t_eval=t_eval)
