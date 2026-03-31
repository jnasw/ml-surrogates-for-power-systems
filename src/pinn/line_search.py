"""Shared line-search configuration helpers for PINN optimizers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch.optim.lbfgs import _strong_wolfe

SUPPORTED_LINE_SEARCHES: frozenset[str] = frozenset({"strong_wolfe", "backtracking"})


@dataclass(frozen=True)
class LineSearchConfig:
    name: str
    c1: float = 1.0e-4
    c2: float = 0.9
    shrink: float = 0.5
    growth: float = 2.0
    max_iters: int = 20
    min_step: float = 1.0e-16
    max_step: float = 1.0


@dataclass(frozen=True)
class LineSearchResult:
    step_size: float
    success: bool
    num_evals: int
    loss_value: float | None = None
    gradient: torch.Tensor | None = None
    reason: str | None = None


def normalize_line_search_config(raw: dict[str, Any] | None) -> dict[str, Any] | None:
    if raw is None:
        return None

    if "name" not in raw:
        raise ValueError("line_search must include a 'name' field.")

    normalized_name = str(raw["name"]).strip().lower()
    if normalized_name not in SUPPORTED_LINE_SEARCHES:
        supported = ", ".join(sorted(SUPPORTED_LINE_SEARCHES))
        raise ValueError(f"Unsupported line_search.name '{raw['name']}'. Use one of: {supported}.")

    config = LineSearchConfig(
        name=normalized_name,
        c1=float(raw.get("c1", 1.0e-4)),
        c2=float(raw.get("c2", 0.9)),
        shrink=float(raw.get("shrink", 0.5)),
        growth=float(raw.get("growth", 2.0)),
        max_iters=int(raw.get("max_iters", 20)),
        min_step=float(raw.get("min_step", 1.0e-16)),
        max_step=float(raw.get("max_step", 1.0)),
    )
    return {
        "name": config.name,
        "c1": config.c1,
        "c2": config.c2,
        "shrink": config.shrink,
        "growth": config.growth,
        "max_iters": config.max_iters,
        "min_step": config.min_step,
        "max_step": config.max_step,
    }


def _run_backtracking(
    *,
    config: dict[str, Any],
    xk: torch.Tensor,
    fk: float,
    gk: torch.Tensor,
    pk: torch.Tensor,
    evaluate_at_step: Callable[[float], tuple[float, torch.Tensor]],
) -> LineSearchResult:
    c1 = float(config["c1"])
    shrink = float(config["shrink"])
    max_iters = int(config["max_iters"])
    min_step = float(config["min_step"])
    max_step = float(config["max_step"])
    alpha = min(float(max_step), max(float(min_step), 1.0))
    directional = float(torch.dot(gk, pk).item())

    evals = 0
    last_loss = fk
    last_grad = gk
    for _ in range(max_iters):
        loss_value, grad_value = evaluate_at_step(alpha)
        evals += 1
        last_loss = loss_value
        last_grad = grad_value
        if loss_value <= fk + c1 * alpha * directional:
            return LineSearchResult(
                step_size=float(alpha),
                success=True,
                num_evals=evals,
                loss_value=float(loss_value),
                gradient=grad_value,
                reason=None,
            )
        alpha *= shrink
        if alpha < min_step:
            break

    return LineSearchResult(
        step_size=0.0,
        success=False,
        num_evals=evals,
        loss_value=float(last_loss),
        gradient=last_grad,
        reason="backtracking_failed",
    )


def _run_strong_wolfe(
    *,
    config: dict[str, Any],
    xk: torch.Tensor,
    fk: float,
    gk: torch.Tensor,
    pk: torch.Tensor,
    evaluate_at_step: Callable[[float], tuple[float, torch.Tensor]],
) -> LineSearchResult:
    gtd = float(torch.dot(gk, pk).item())
    if gtd >= 0.0:
        return LineSearchResult(
            step_size=0.0,
            success=False,
            num_evals=0,
            loss_value=float(fk),
            gradient=gk,
            reason="non_descent_direction",
        )

    def obj_func(x: torch.Tensor, t: float, d: torch.Tensor) -> tuple[float, torch.Tensor]:
        del x, d
        return evaluate_at_step(float(t))

    loss_value, gradient, step_size, evals = _strong_wolfe(
        obj_func=obj_func,
        x=xk,
        t=min(float(config["max_step"]), max(float(config["min_step"]), 1.0)),
        d=pk,
        f=float(fk),
        g=gk,
        gtd=gtd,
        c1=float(config["c1"]),
        c2=float(config["c2"]),
        tolerance_change=float(config["min_step"]),
        max_ls=int(config["max_iters"]),
    )
    success = bool(step_size > 0.0)
    return LineSearchResult(
        step_size=float(step_size),
        success=success,
        num_evals=int(evals),
        loss_value=float(loss_value),
        gradient=gradient,
        reason=None if success else "strong_wolfe_failed",
    )


def run_line_search(
    *,
    config: dict[str, Any] | None,
    xk: torch.Tensor,
    fk: float,
    gk: torch.Tensor,
    pk: torch.Tensor,
    evaluate_at_step: Callable[[float], tuple[float, torch.Tensor]],
) -> LineSearchResult:
    normalized = normalize_line_search_config(config)
    if normalized is None:
        normalized = normalize_line_search_config({"name": "strong_wolfe"})
        assert normalized is not None
    if normalized["name"] == "strong_wolfe":
        return _run_strong_wolfe(
            config=normalized,
            xk=xk,
            fk=fk,
            gk=gk,
            pk=pk,
            evaluate_at_step=evaluate_at_step,
        )
    if normalized["name"] == "backtracking":
        return _run_backtracking(
            config=normalized,
            xk=xk,
            fk=fk,
            gk=gk,
            pk=pk,
            evaluate_at_step=evaluate_at_step,
        )
    raise ValueError(f"Unsupported line-search method: {normalized['name']}")
