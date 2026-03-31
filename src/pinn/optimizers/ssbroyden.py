"""Full-batch self-scaled Broyden optimizer for PINN training."""

from __future__ import annotations

from typing import Any

import torch

from src.pinn.optimizers.bfgs import BFGS


class SSBroyden(BFGS):
    """Self-scaled Broyden update using the paper-style tau/phi rules."""

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
        tau_strategy: str = "paper_default",
        phi_strategy: str = "paper_default",
        tau_min: float = 1.0e-12,
        tau_max: float = 1.0,
        phi_min: float = -1.0e6,
        phi_max: float = 1.0e6,
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
            raise ValueError("SSBroyden requires tau_min > 0.")
        if tau_max <= 0.0:
            raise ValueError("SSBroyden requires tau_max > 0.")
        if tau_min > tau_max:
            raise ValueError("SSBroyden requires tau_min <= tau_max.")
        if phi_min > phi_max:
            raise ValueError("SSBroyden requires phi_min <= phi_max.")
        group = self.param_groups[0]
        group["tau_strategy"] = str(tau_strategy).strip().lower()
        group["phi_strategy"] = str(phi_strategy).strip().lower()
        group["tau_min"] = float(tau_min)
        group["tau_max"] = float(tau_max)
        group["phi_min"] = float(phi_min)
        group["phi_max"] = float(phi_max)

    @torch.no_grad()
    def _compute_scalars(
        self,
        H: torch.Tensor,
        s: torch.Tensor,
        y: torch.Tensor,
        curvature_eps: float,
        context: dict[str, Any] | None,
    ) -> tuple[dict[str, float | int | bool | str | None] | None, dict[str, float | int | bool | str | None]]:
        diagnostics: dict[str, float | int | bool | str | None] = {
            "tau_strategy": str(self.param_groups[0]["tau_strategy"]),
            "phi_strategy": str(self.param_groups[0]["phi_strategy"]),
            "tau1": None,
            "tau_raw": None,
            "tau_clipped": False,
            "phi_raw": None,
            "phi_clipped": False,
            "theta": None,
            "theta_minus": None,
            "theta_plus": None,
            "sigma": None,
            "sigma_power": None,
            "a_k": None,
            "b_k": None,
            "h_k": None,
            "rho_minus": None,
            "rho_plus": None,
            "tau_reason": None,
            "phi_reason": None,
        }

        ys = float(torch.dot(y, s).item())
        yHy = float(torch.dot(y, H @ y).item())
        diagnostics["yHy"] = yHy
        if ys <= float(curvature_eps) or yHy <= float(curvature_eps):
            diagnostics["tau_reason"] = "invalid_curvature"
            diagnostics["phi_reason"] = "invalid_curvature"
            return None, diagnostics

        if context is None or "gk" not in context or "step_size" not in context:
            diagnostics["tau_reason"] = "missing_context"
            diagnostics["phi_reason"] = "missing_context"
            return None, diagnostics

        gk = context["gk"]
        step_size = float(context["step_size"])
        if step_size <= float(curvature_eps):
            diagnostics["tau_reason"] = "invalid_step"
            diagnostics["phi_reason"] = "invalid_step"
            return None, diagnostics

        s_dot_g = float(torch.dot(s, gk).item())
        b_k = float((-step_size * s_dot_g) / ys)
        h_k = float(yHy / ys)
        diagnostics["b_k"] = b_k
        diagnostics["h_k"] = h_k

        if b_k <= float(curvature_eps) or h_k <= float(curvature_eps):
            diagnostics["tau_reason"] = "invalid_b_or_h"
            diagnostics["phi_reason"] = "invalid_b_or_h"
            return None, diagnostics

        a_k = float(b_k * h_k - 1.0)
        diagnostics["a_k"] = a_k
        if a_k <= float(curvature_eps):
            diagnostics["tau_reason"] = "nonpositive_a"
            diagnostics["phi_reason"] = "nonpositive_a"
            return None, diagnostics

        c_k = float((a_k / (1.0 + a_k)) ** 0.5)
        rho_minus = float(min(1.0, h_k * (1.0 - c_k)))
        rho_plus = float(min(1.0, 1.0 / b_k))
        theta_minus = float((rho_minus - 1.0) / a_k)
        if rho_minus <= float(curvature_eps):
            diagnostics["tau_reason"] = "invalid_rho_minus"
            diagnostics["phi_reason"] = "invalid_rho_minus"
            return None, diagnostics
        theta_plus = float(1.0 / rho_minus)
        theta_mid = float((1.0 - b_k) / b_k)
        theta = float(max(theta_minus, min(theta_plus, theta_mid)))
        sigma = float(1.0 + theta * a_k)
        n = max(2, int(context.get("param_dim", H.shape[0])))

        diagnostics["theta_minus"] = theta_minus
        diagnostics["theta_plus"] = theta_plus
        diagnostics["theta"] = theta
        diagnostics["sigma"] = sigma
        diagnostics["rho_minus"] = rho_minus
        diagnostics["rho_plus"] = rho_plus

        if sigma <= float(curvature_eps):
            diagnostics["tau_reason"] = "invalid_sigma"
            diagnostics["phi_reason"] = "invalid_sigma"
            return None, diagnostics

        sigma_power = float(sigma ** (-1.0 / (n - 1.0)))
        diagnostics["sigma_power"] = sigma_power
        tau1 = float(min(1.0, ys / max(float(torch.dot(s, H @ s).item()), float(curvature_eps))))
        diagnostics["tau1"] = tau1

        strategy = str(self.param_groups[0]["tau_strategy"])
        if strategy != "paper_default":
            raise ValueError(f"Unsupported SSBroyden tau_strategy: {strategy}")
        if theta > 0.0:
            tau = tau1 * min(sigma_power, 1.0 / max(theta, float(curvature_eps)))
        else:
            tau = min(tau1 * sigma_power, sigma)
        tau = float(tau)
        diagnostics["tau_raw"] = tau
        tau_clipped = False
        tau_min = float(self.param_groups[0]["tau_min"])
        tau_max = float(self.param_groups[0]["tau_max"])
        if tau < tau_min:
            tau = tau_min
            tau_clipped = True
        if tau > tau_max:
            tau = tau_max
            tau_clipped = True
        diagnostics["tau"] = tau
        diagnostics["tau_clipped"] = tau_clipped

        phi_strategy = str(self.param_groups[0]["phi_strategy"])
        if phi_strategy != "paper_default":
            raise ValueError(f"Unsupported SSBroyden phi_strategy: {phi_strategy}")
        phi_den = 1.0 + a_k * theta
        if abs(phi_den) <= float(curvature_eps):
            diagnostics["phi_reason"] = "invalid_phi_denominator"
            return None, diagnostics
        phi = float((1.0 - theta) / phi_den)
        diagnostics["phi_raw"] = phi
        phi_clipped = False
        phi_min = float(self.param_groups[0]["phi_min"])
        phi_max = float(self.param_groups[0]["phi_max"])
        if phi < phi_min:
            phi = phi_min
            phi_clipped = True
        if phi > phi_max:
            phi = phi_max
            phi_clipped = True
        diagnostics["phi"] = phi
        diagnostics["phi_clipped"] = phi_clipped

        return {
            "ys": ys,
            "yHy": yHy,
            "b_k": b_k,
            "h_k": h_k,
            "a_k": a_k,
            "theta": theta,
            "tau": tau,
            "phi": phi,
        }, diagnostics

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
                "phi": None,
                "theta": None,
                "tau_reason": "invalid_curvature",
                "phi_reason": "invalid_curvature",
            }

        scalars, diagnostics = self._compute_scalars(
            H=H,
            s=s,
            y=y,
            curvature_eps=curvature_eps,
            context=context,
        )
        if scalars is None:
            return H, True, ys, diagnostics

        yHy = float(scalars["yHy"])
        tau = float(scalars["tau"])
        phi = float(scalars["phi"])
        if yHy <= float(curvature_eps) or tau <= float(curvature_eps):
            diagnostics["tau_reason"] = diagnostics.get("tau_reason") or "invalid_tau_or_yHy"
            diagnostics["phi_reason"] = diagnostics.get("phi_reason") or "invalid_tau_or_yHy"
            return H, True, ys, diagnostics

        Hy = H @ y
        v = (s / ys) - (Hy / yHy)
        rank_one = phi * yHy * torch.outer(v, v)
        ss_term = torch.outer(s, s) / ys
        correction = torch.outer(Hy, Hy) / yHy
        updated = (1.0 / tau) * (H - correction + rank_one + ss_term)
        return updated, False, ys, diagnostics
