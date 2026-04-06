"""Full-batch BFGS optimizer for PINN training."""

from __future__ import annotations

from typing import Any

import torch

from src.pinn.line_search import run_line_search


class BFGS(torch.optim.Optimizer):
    """Dense full-batch BFGS optimizer with configurable line search."""

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
    ) -> None:
        if lr <= 0.0:
            raise ValueError("BFGS requires lr > 0.")
        defaults = {
            "lr": float(lr),
            "line_search": None if line_search is None else dict(line_search),
            "curvature_eps": float(curvature_eps),
            "init_hessian_scale": float(init_hessian_scale),
            "grad_tol": float(grad_tol),
            "step_tol": float(step_tol),
            "history_reset_on_failure": bool(history_reset_on_failure),
        }
        super().__init__(params, defaults)
        self._last_diagnostics: dict[str, float | int | bool | str | None] = {}

    def get_last_diagnostics(self) -> dict[str, float | int | bool | str | None]:
        return dict(self._last_diagnostics)

    def _params(self) -> list[torch.nn.Parameter]:
        return [param for group in self.param_groups for param in group["params"] if param.requires_grad]

    def _flat_params(self) -> torch.Tensor:
        params = self._params()
        if not params:
            return torch.empty(0, dtype=torch.float32)
        return torch.cat([param.detach().reshape(-1) for param in params], dim=0)

    def _flat_grads(self) -> torch.Tensor:
        params = self._params()
        if not params:
            return torch.empty(0, dtype=torch.float32)
        vectors = []
        for param in params:
            grad = param.grad
            if grad is None:
                vectors.append(torch.zeros_like(param, memory_format=torch.contiguous_format).reshape(-1))
            else:
                vectors.append(grad.detach().reshape(-1))
        return torch.cat(vectors, dim=0)

    @torch.no_grad()
    def _set_flat_params(self, vec: torch.Tensor) -> None:
        params = self._params()
        offset = 0
        for param in params:
            numel = param.numel()
            reshaped = vec[offset : offset + numel].view_as(param).to(device=param.device, dtype=param.dtype)
            param.copy_(reshaped)
            offset += numel

    def _state(self) -> dict[str, Any]:
        params = self._params()
        if not params:
            raise RuntimeError("BFGS optimizer has no trainable parameters.")
        return self.state[params[0]]

    def _optimizer_label(self) -> str:
        return self.__class__.__name__

    def _reset_inverse_hessian(self, size: int, *, dtype: torch.dtype, device: torch.device, scale: float) -> torch.Tensor:
        return torch.eye(size, dtype=dtype, device=device) * float(scale)

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
            return H, True, ys, {}
        rho = 1.0 / ys
        identity = torch.eye(H.shape[0], dtype=H.dtype, device=H.device)
        sy = torch.outer(s, y)
        ys_outer = torch.outer(y, s)
        ss = torch.outer(s, s)
        V = identity - rho * sy
        updated = V @ H @ (identity - rho * ys_outer) + rho * ss
        return updated, False, ys, {}

    @torch.no_grad()
    def _update_inverse_hessian(
        self,
        H: torch.Tensor,
        s: torch.Tensor,
        y: torch.Tensor,
        curvature_eps: float,
        context: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, bool, float, dict[str, float | int | bool | str | None]]:
        return self._apply_inverse_hessian_update(
            H=H,
            s=s,
            y=y,
            curvature_eps=curvature_eps,
            context=context,
        )

    def step(self, closure):
        if closure is None:
            raise ValueError("BFGS.step requires a closure.")

        closure = torch.enable_grad()(closure)
        params = self._params()
        if not params:
            return None
        group = self.param_groups[0]
        state = self._state()

        loss_tensor = closure()
        fk = float(loss_tensor.item())
        gk = self._flat_grads()
        xk = self._flat_params()
        grad_norm = float(torch.linalg.vector_norm(gk, ord=2).item()) if gk.numel() > 0 else 0.0

        if gk.numel() == 0:
            self._last_diagnostics = {
                "line_search_success": False,
                "line_search_evals": 0,
                "step_size": 0.0,
                "grad_norm": 0.0,
                "curvature_ys": None,
                "update_skipped": True,
                "reason": "no_parameters",
            }
            return loss_tensor

        if grad_norm <= float(group["grad_tol"]):
            self._last_diagnostics = {
                "line_search_success": False,
                "line_search_evals": 0,
                "step_size": 0.0,
                "grad_norm": grad_norm,
                "curvature_ys": None,
                "update_skipped": True,
                "reason": "grad_tol",
            }
            return loss_tensor

        H = state.get("H")
        if H is None or tuple(H.shape) != (xk.numel(), xk.numel()):
            H = self._reset_inverse_hessian(
                int(xk.numel()),
                dtype=xk.dtype,
                device=xk.device,
                scale=float(group["init_hessian_scale"]),
            )

        pk = -(H @ gk)
        directional = float(torch.dot(gk, pk).item())
        direction_reset = False
        if not torch.isfinite(torch.as_tensor(directional, dtype=xk.dtype, device=xk.device)) or directional >= 0.0:
            H = self._reset_inverse_hessian(
                int(xk.numel()),
                dtype=xk.dtype,
                device=xk.device,
                scale=float(group["init_hessian_scale"]),
            )
            pk = -gk.clone()
            directional = float(torch.dot(gk, pk).item())
            direction_reset = True

        search_direction = float(group["lr"]) * pk

        def evaluate_at_step(alpha: float) -> tuple[float, torch.Tensor]:
            x_trial = xk + float(alpha) * search_direction
            self._set_flat_params(x_trial)
            loss_trial = closure()
            grad_trial = self._flat_grads()
            return float(loss_trial.item()), grad_trial

        line_search_result = run_line_search(
            config=group["line_search"],
            xk=xk,
            fk=fk,
            gk=gk,
            pk=search_direction,
            evaluate_at_step=evaluate_at_step,
        )
        line_search_cfg = group.get("line_search") or {}
        min_acceptable_step = max(float(group["step_tol"]), float(line_search_cfg.get("min_step", 0.0)))

        update_skipped = False
        curvature_ys: float | None = None
        update_diagnostics: dict[str, float | int | bool | str | None] = {}

        if line_search_result.success and line_search_result.step_size > min_acceptable_step:
            step_size = float(line_search_result.step_size)
            x_next = xk + step_size * search_direction
            self._set_flat_params(x_next)
            f_next = fk if line_search_result.loss_value is None else float(line_search_result.loss_value)
            g_next = gk if line_search_result.gradient is None else line_search_result.gradient
            s = x_next - xk
            y = g_next - gk
            update_context = {
                "xk": xk,
                "x_next": x_next,
                "gk": gk,
                "g_next": g_next,
                "step_size": step_size,
                "stage_lr": float(group["lr"]),
                "alpha_total": float(step_size) * float(group["lr"]),
                "search_direction": search_direction,
                "param_dim": int(xk.numel()),
            }
            H, update_skipped, curvature_ys, update_diagnostics = self._update_inverse_hessian(
                H=H,
                s=s,
                y=y,
                curvature_eps=float(group["curvature_eps"]),
                context=update_context,
            )
            if update_skipped and bool(group["history_reset_on_failure"]):
                H = self._reset_inverse_hessian(
                    int(xk.numel()),
                    dtype=xk.dtype,
                    device=xk.device,
                    scale=float(group["init_hessian_scale"]),
                )
            state["x_prev"] = x_next.detach().clone()
            state["g_prev"] = g_next.detach().clone()
            accepted_loss = torch.tensor(f_next, dtype=loss_tensor.dtype, device=loss_tensor.device)
        else:
            self._set_flat_params(xk)
            if bool(group["history_reset_on_failure"]):
                H = self._reset_inverse_hessian(
                    int(xk.numel()),
                    dtype=xk.dtype,
                    device=xk.device,
                    scale=float(group["init_hessian_scale"]),
                )
            accepted_loss = loss_tensor
            update_skipped = True

        state["H"] = H
        state["n_iter"] = int(state.get("n_iter", 0)) + 1
        reason = line_search_result.reason
        if line_search_result.success and line_search_result.step_size <= min_acceptable_step:
            reason = "step_below_min_threshold"
        self._last_diagnostics = {
            "line_search_success": bool(line_search_result.success),
            "line_search_evals": int(line_search_result.num_evals),
            "step_size": float(line_search_result.step_size),
            "min_acceptable_step": float(min_acceptable_step),
            "grad_norm": grad_norm,
            "curvature_ys": curvature_ys,
            "update_skipped": bool(update_skipped),
            "direction_reset": bool(direction_reset),
            "directional_derivative": directional,
            "param_norm": float(torch.linalg.vector_norm(xk, ord=2).item()),
            "optimizer": self._optimizer_label(),
            "reason": reason,
            **update_diagnostics,
        }
        return accepted_loss
