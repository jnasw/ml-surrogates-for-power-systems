"""Optimizer construction for PINN training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
try:
    from pytorch_optimizer import SOAP
except ImportError:  # pragma: no cover - optional dependency
    SOAP = None

from src.pinn.optimizers.bfgs import BFGS
from src.pinn.optimizers.ssbroyden import SSBroyden, StochasticSSBroyden
from src.pinn.optimizers.ssbfgs import SSBFGS, StochasticSSBFGS


@dataclass(frozen=True)
class OptimizerSpec:
    optimizer: torch.optim.Optimizer
    requires_closure: bool
    supports_minibatch: bool
    default_full_batch: bool
    line_search_name: str | None = None


def build_optimizer(
    model: nn.Module,
    optimizer_name: str,
    lr: float,
    optimizer_kwargs: dict[str, Any] | None = None,
    line_search: dict[str, Any] | None = None,
) -> OptimizerSpec:
    normalized = str(optimizer_name).strip().lower()
    kwargs = {} if optimizer_kwargs is None else dict(optimizer_kwargs)

    if normalized == "adam":
        if line_search not in (None, {}):
            raise ValueError("Adam phases must not configure line_search.")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, **kwargs)
        return OptimizerSpec(
            optimizer=optimizer,
            requires_closure=False,
            supports_minibatch=True,
            default_full_batch=False,
            line_search_name=None,
        )
    if normalized == "soap":
        if line_search not in (None, {}):
            raise ValueError("SOAP phases must not configure line_search.")
        if SOAP is None:
            raise ImportError(
                "SOAP requires the optional 'pytorch-optimizer' package. "
                "Install it with `pip install pytorch-optimizer`."
            )
        optimizer = SOAP(model.parameters(), lr=lr, **kwargs)
        return OptimizerSpec(
            optimizer=optimizer,
            requires_closure=False,
            supports_minibatch=True,
            default_full_batch=False,
            line_search_name=None,
        )
    if normalized == "lbfgs":
        line_search_name = None if line_search is None else str(line_search["name"]).strip().lower()
        if line_search_name in (None, "", "null"):
            line_search_name = "strong_wolfe"
        if line_search_name not in {"strong_wolfe"}:
            raise ValueError("PyTorch LBFGS only supports line_search.name='strong_wolfe' or null.")
        optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=lr,
            line_search_fn=line_search_name,
            **kwargs,
        )
        return OptimizerSpec(
            optimizer=optimizer,
            requires_closure=True,
            supports_minibatch=False,
            default_full_batch=True,
            line_search_name=line_search_name,
        )
    if normalized == "bfgs":
        line_search_cfg = {"name": "strong_wolfe"} if line_search is None else dict(line_search)
        optimizer = BFGS(
            model.parameters(),
            lr=lr,
            line_search=line_search_cfg,
            **kwargs,
        )
        return OptimizerSpec(
            optimizer=optimizer,
            requires_closure=True,
            supports_minibatch=False,
            default_full_batch=True,
            line_search_name=str(line_search_cfg["name"]).strip().lower(),
        )
    if normalized == "ssbfgs":
        line_search_cfg = {"name": "strong_wolfe"} if line_search is None else dict(line_search)
        optimizer = SSBFGS(
            model.parameters(),
            lr=lr,
            line_search=line_search_cfg,
            **kwargs,
        )
        return OptimizerSpec(
            optimizer=optimizer,
            requires_closure=True,
            supports_minibatch=False,
            default_full_batch=True,
            line_search_name=str(line_search_cfg["name"]).strip().lower(),
        )
    if normalized == "ssbroyden":
        line_search_cfg = {"name": "strong_wolfe"} if line_search is None else dict(line_search)
        optimizer = SSBroyden(
            model.parameters(),
            lr=lr,
            line_search=line_search_cfg,
            **kwargs,
        )
        return OptimizerSpec(
            optimizer=optimizer,
            requires_closure=True,
            supports_minibatch=False,
            default_full_batch=True,
            line_search_name=str(line_search_cfg["name"]).strip().lower(),
        )
    if normalized == "sssbfgs":
        if line_search not in (None, {}):
            raise ValueError("sSSBFGS phases must not configure line_search.")
        optimizer = StochasticSSBFGS(
            model.parameters(),
            lr=lr,
            **kwargs,
        )
        return OptimizerSpec(
            optimizer=optimizer,
            requires_closure=False,
            supports_minibatch=True,
            default_full_batch=False,
            line_search_name=None,
        )
    if normalized == "sssbroyden":
        if line_search not in (None, {}):
            raise ValueError("sSSBroyden phases must not configure line_search.")
        optimizer = StochasticSSBroyden(
            model.parameters(),
            lr=lr,
            **kwargs,
        )
        return OptimizerSpec(
            optimizer=optimizer,
            requires_closure=False,
            supports_minibatch=True,
            default_full_batch=False,
            line_search_name=None,
        )
    raise ValueError(
        "Unsupported optimizer. Use one of: Adam, SOAP, LBFGS, BFGS, SSBFGS, SSBroyden, sSSBFGS, sSSBroyden."
    )
