"""Shared lazy stochastic quasi-Newton step logic."""

from __future__ import annotations

import torch


class LazyStochasticQuasiNewtonMixin:
    """Mini-batch quasi-Newton updates with lazy curvature refreshes."""

    def _configure_stochastic_quasi_newton(
        self,
        *,
        curvature_threshold: float,
        reset_on_direction_failure: bool,
    ) -> None:
        if curvature_threshold < 0.0:
            raise ValueError("stochastic curvature_threshold must be >= 0.")
        group = self.param_groups[0]
        group["stochastic_curvature_threshold"] = float(curvature_threshold)
        group["reset_on_direction_failure"] = bool(reset_on_direction_failure)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError(f"{self._optimizer_label()}.step does not accept a closure.")

        params = self._params()
        if not params:
            return None

        group = self.param_groups[0]
        state = self._state()
        xk = self._flat_params()
        gk = self._flat_grads()
        grad_norm = float(torch.linalg.vector_norm(gk, ord=2).item()) if gk.numel() > 0 else 0.0

        if gk.numel() == 0:
            self._last_diagnostics = {
                "line_search_success": False,
                "line_search_evals": 0,
                "step_size": 0.0,
                "grad_norm": 0.0,
                "curvature_ys": None,
                "update_skipped": True,
                "stochastic_mode": True,
                "reason": "no_parameters",
            }
            return None

        if grad_norm <= float(group["grad_tol"]):
            self._last_diagnostics = {
                "line_search_success": False,
                "line_search_evals": 0,
                "step_size": 0.0,
                "grad_norm": grad_norm,
                "curvature_ys": None,
                "update_skipped": True,
                "stochastic_mode": True,
                "reason": "grad_tol",
            }
            return None

        H = state.get("H")
        if H is None or tuple(H.shape) != (xk.numel(), xk.numel()):
            H = self._reset_inverse_hessian(
                int(xk.numel()),
                dtype=xk.dtype,
                device=xk.device,
                scale=float(group["init_hessian_scale"]),
            )

        prev_gradient = state.get("g_prev")
        prev_step = state.get("s_prev")
        curvature_ys: float | None = None
        update_skipped = True
        hessian_updated = False
        curvature_condition_met = False
        curvature_rhs: float | None = None
        update_diagnostics: dict[str, float | int | bool | str | None] = {}

        if prev_gradient is not None and prev_step is not None:
            y_prev = gk - prev_gradient
            curvature_ys = float(torch.dot(y_prev, prev_step).item())
            step_sq_norm = float(torch.dot(prev_step, prev_step).item())
            curvature_rhs = float(group["stochastic_curvature_threshold"]) * step_sq_norm
            curvature_condition_met = (
                curvature_ys >= curvature_rhs and curvature_ys > float(group["curvature_eps"])
            )
            if curvature_condition_met:
                updated_H, update_skipped, curvature_ys, update_diagnostics = self._update_inverse_hessian(
                    H=H,
                    s=prev_step,
                    y=y_prev,
                    curvature_eps=float(group["curvature_eps"]),
                    context={
                        "gk": prev_gradient,
                        "alpha_total": float(group["lr"]),
                        "param_dim": int(xk.numel()),
                    },
                )
                if not update_skipped:
                    H = updated_H
                    hessian_updated = True
                elif bool(group["history_reset_on_failure"]):
                    H = self._reset_inverse_hessian(
                        int(xk.numel()),
                        dtype=xk.dtype,
                        device=xk.device,
                        scale=float(group["init_hessian_scale"]),
                    )
        else:
            update_diagnostics["tau_reason"] = None

        pk = -(H @ gk)
        directional = float(torch.dot(gk, pk).item())
        direction_reset = False
        if not torch.isfinite(torch.as_tensor(directional, dtype=xk.dtype, device=xk.device)) or directional >= 0.0:
            direction_reset = True
            if bool(group["reset_on_direction_failure"]):
                H = self._reset_inverse_hessian(
                    int(xk.numel()),
                    dtype=xk.dtype,
                    device=xk.device,
                    scale=float(group["init_hessian_scale"]),
                )
            pk = -gk.clone()
            directional = float(torch.dot(gk, pk).item())

        step = float(group["lr"]) * pk
        step_norm = float(torch.linalg.vector_norm(step, ord=2).item()) if step.numel() > 0 else 0.0
        if step_norm <= float(group["step_tol"]):
            state["H"] = H
            self._last_diagnostics = {
                "line_search_success": False,
                "line_search_evals": 0,
                "step_size": float(group["lr"]),
                "effective_step_norm": step_norm,
                "grad_norm": grad_norm,
                "curvature_ys": curvature_ys,
                "update_skipped": True,
                "direction_reset": bool(direction_reset),
                "directional_derivative": directional,
                "param_norm": float(torch.linalg.vector_norm(xk, ord=2).item()),
                "optimizer": self._optimizer_label(),
                "stochastic_mode": True,
                "stochastic_prev_available": bool(prev_gradient is not None and prev_step is not None),
                "stochastic_hessian_updated": bool(hessian_updated),
                "stochastic_curvature_condition_met": bool(curvature_condition_met),
                "stochastic_curvature_rhs": curvature_rhs,
                "reason": "step_tol",
                **update_diagnostics,
            }
            return None

        x_next = xk + step
        self._set_flat_params(x_next)

        state["H"] = H
        state["g_prev"] = gk.detach().clone()
        state["s_prev"] = step.detach().clone()
        state["n_iter"] = int(state.get("n_iter", 0)) + 1

        self._last_diagnostics = {
            "line_search_success": False,
            "line_search_evals": 0,
            "step_size": float(group["lr"]),
            "effective_step_norm": step_norm,
            "grad_norm": grad_norm,
            "curvature_ys": curvature_ys,
            "update_skipped": bool(update_skipped),
            "direction_reset": bool(direction_reset),
            "directional_derivative": directional,
            "param_norm": float(torch.linalg.vector_norm(xk, ord=2).item()),
            "optimizer": self._optimizer_label(),
            "stochastic_mode": True,
            "stochastic_prev_available": bool(prev_gradient is not None and prev_step is not None),
            "stochastic_hessian_updated": bool(hessian_updated),
            "stochastic_curvature_condition_met": bool(curvature_condition_met),
            "stochastic_curvature_rhs": curvature_rhs,
            "reason": None,
            **update_diagnostics,
        }
        return None
