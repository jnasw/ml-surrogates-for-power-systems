"""Full-batch self-scaled BFGS optimizer for PINN training."""

from __future__ import annotations

from typing import Any

import torch

from src.pinn.optimizers.bfgs import BFGS


class SSBFGS(BFGS):
    """Self-scaled BFGS using a guarded Al-Baali-style scaling factor."""

    def __init__(
        self,
        params,
        *,
        lr: float = 1.0,
        line_search: dict[str, Any] | None = None,
        curvature_eps: float = 1.0e-12,
        init_hessian_scale: float = 1.0,
        grad_tol: float = 0.0,
        step_tol: float = 0.0,
        history_reset_on_failure: bool = True,
        tau_strategy: str = "al_baali",
        tau_min: float = 1.0e-12,
        tau_max: float = 1.0,
    ) -> None:
        super().__init__(
            params,
            lr=lr,
            line_search=line_search,
            curvature_eps=curvature_eps,
            init_hessian_scale=init_hessian_scale,
            grad_tol=grad_tol,
            step_tol=step_tol,
            history_reset_on_failure=history_reset_on_failure,
        )
        if tau_min <= 0.0:
            raise ValueError("SSBFGS requires tau_min > 0.")
        if tau_max <= 0.0:
            raise ValueError("SSBFGS requires tau_max > 0.")
        if tau_min > tau_max:
            raise ValueError("SSBFGS requires tau_min <= tau_max.")
        self.param_groups[0]["tau_strategy"] = str(tau_strategy).strip().lower()
        self.param_groups[0]["tau_min"] = float(tau_min)
        self.param_groups[0]["tau_max"] = float(tau_max)

    @torch.no_grad()
    def _compute_tau(self, H: torch.Tensor, s: torch.Tensor, y: torch.Tensor, curvature_eps: float) -> tuple[float | None, dict[str, float | int | bool | str | None]]:
        ys = float(torch.dot(y, s).item())
        sHs = float(torch.dot(s, H @ s).item())
        diagnostics: dict[str, float | int | bool | str | None] = {
            "tau_strategy": str(self.param_groups[0]["tau_strategy"]),
            "tau_raw": None,
            "tau_clipped": False,
            "sHs": sHs,
        }
        if ys <= float(curvature_eps) or sHs <= float(curvature_eps):
            diagnostics["tau_reason"] = "invalid_curvature"
            return None, diagnostics

        strategy = str(self.param_groups[0]["tau_strategy"])
        if strategy != "al_baali":
            raise ValueError(f"Unsupported SSBFGS tau_strategy: {strategy}")

        tau = min(1.0, ys / sHs)
        tau = float(tau)
        diagnostics["tau_raw"] = tau

        tau_min = float(self.param_groups[0]["tau_min"])
        tau_max = float(self.param_groups[0]["tau_max"])
        tau_clipped = False
        if tau < tau_min:
            tau = tau_min
            tau_clipped = True
        if tau > tau_max:
            tau = tau_max
            tau_clipped = True
        diagnostics["tau_clipped"] = tau_clipped
        diagnostics["tau"] = tau
        diagnostics["tau_reason"] = None
        return tau, diagnostics

    @torch.no_grad()
    def _apply_inverse_hessian_update(
        self,
        H: torch.Tensor,
        s: torch.Tensor,
        y: torch.Tensor,
        curvature_eps: float,
        context: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, bool, float, dict[str, float | int | bool | str | None]]:
        ys = float(torch.dot(y, s).item())
        if ys <= float(curvature_eps):
            return H, True, ys, {
                "tau": None,
                "tau_raw": None,
                "tau_clipped": False,
                "tau_reason": "invalid_curvature",
                "sHs": None,
            }

        tau, tau_diagnostics = self._compute_tau(H=H, s=s, y=y, curvature_eps=curvature_eps)
        if tau is None:
            return H, True, ys, tau_diagnostics

        rho = 1.0 / ys
        identity = torch.eye(H.shape[0], dtype=H.dtype, device=H.device)
        scaled_H = float(tau) * H
        sy = torch.outer(s, y)
        ys_outer = torch.outer(y, s)
        ss = torch.outer(s, s)
        V = identity - rho * sy
        updated = V @ scaled_H @ (identity - rho * ys_outer) + rho * ss
        return updated, False, ys, tau_diagnostics
