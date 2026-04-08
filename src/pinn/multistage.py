"""Paper-style multistage PINN model helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def _activation(name: str) -> nn.Module:
    normalized = str(name).strip().lower()
    if normalized == "tanh":
        return nn.Tanh()
    if normalized == "silu":
        return nn.SiLU()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "sin":
        return _SinActivation()
    raise ValueError("Unsupported activation. Use one of: tanh, silu, relu, sin.")


class _SinActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


class MultistageStageMLP(nn.Module):
    """Stage network used inside the multistage PINN ensemble."""

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        first_activation: str = "tanh",
        hidden_activation: str = "tanh",
        time_scale: float = 1.0,
        output_init_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if hidden_layers <= 0:
            raise ValueError("hidden_layers must be >= 1 for multistage PINN stages.")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.hidden_layers = int(hidden_layers)
        self.first_activation_name = str(first_activation)
        self.hidden_activation_name = str(hidden_activation)
        self.time_scale = float(time_scale)
        self.output_init_scale = float(output_init_scale)

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.first_activation = _activation(first_activation)
        hidden_blocks: list[nn.Module] = []
        for _ in range(max(0, hidden_layers - 1)):
            hidden_blocks.append(nn.Linear(hidden_dim, hidden_dim))
            hidden_blocks.append(_activation(hidden_activation))
        self.hidden_stack = nn.Sequential(*hidden_blocks)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        # Residual stages should enter as small corrections rather than a
        # full-amplitude perturbation of the existing ensemble prediction.
        if self.output_init_scale != 1.0:
            with torch.no_grad():
                self.output_layer.weight.mul_(self.output_init_scale)
                self.output_layer.bias.mul_(self.output_init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_scaled = x
        if self.time_scale != 1.0:
            x_scaled = x.clone()
            x_scaled[:, :1] = x_scaled[:, :1] * self.time_scale
        h = self.first_activation(self.input_layer(x_scaled))
        h = self.hidden_stack(h)
        return self.output_layer(h)


class MultistagePinnEnsemble(nn.Module):
    """Additive residual ensemble for multistage PINNs."""

    def __init__(
        self,
        *,
        stages: list[nn.Module],
        epsilons: list[float] | None = None,
    ) -> None:
        super().__init__()
        if not stages:
            raise ValueError("MultistagePinnEnsemble requires at least one stage.")
        self.stages = nn.ModuleList(stages)
        eps = [1.0] * len(stages) if epsilons is None else list(epsilons)
        if len(eps) != len(stages):
            raise ValueError("len(epsilons) must match len(stages).")
        self.register_buffer("epsilons", torch.tensor(eps, dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eps = self.epsilons.to(device=x.device, dtype=x.dtype)
        total = None
        for idx, stage in enumerate(self.stages):
            stage_out = stage(x)
            scaled = stage_out * eps[idx]
            total = scaled if total is None else total + scaled
        if total is None:
            raise RuntimeError("MultistagePinnEnsemble has no stages.")
        return total

    def freeze_stages_up_to(self, stage_idx_exclusive: int) -> None:
        for idx, stage in enumerate(self.stages):
            requires_grad = idx >= int(stage_idx_exclusive)
            for param in stage.parameters():
                param.requires_grad_(requires_grad)

    @property
    def num_stages(self) -> int:
        return len(self.stages)

    def epsilon_values(self) -> list[float]:
        return [float(value) for value in self.epsilons.detach().cpu().tolist()]

    def set_stage_epsilon(self, stage_idx: int, value: float) -> None:
        self.epsilons[int(stage_idx)] = float(value)


@dataclass(frozen=True)
class StageDiagnostics:
    residual_rms: float
    residual_zero_crossings: int
    kappa_suggested: float
    dominant_residual_channel: int = 0


def estimate_stage_diagnostics(
    *,
    x_col: torch.Tensor,
    residual: torch.Tensor,
    min_kappa: float = 1.0,
    max_kappa: float = 100.0,
) -> StageDiagnostics:
    if residual.numel() == 0:
        return StageDiagnostics(
            residual_rms=0.0,
            residual_zero_crossings=0,
            kappa_suggested=float(min_kappa),
            dominant_residual_channel=0,
        )

    residual_rms = float(torch.sqrt(torch.mean(residual.pow(2))).detach().cpu().item())
    if int(x_col.shape[0]) < 3:
        return StageDiagnostics(
            residual_rms=residual_rms,
            residual_zero_crossings=0,
            kappa_suggested=float(min_kappa),
            dominant_residual_channel=0,
        )

    order = torch.argsort(x_col[:, 0])
    residual_sorted = residual.index_select(0, order)
    channel_rms = torch.sqrt(torch.mean(residual_sorted.pow(2), dim=0))
    dominant_channel = int(torch.argmax(channel_rms).detach().cpu().item())
    probe = residual_sorted[:, dominant_channel] - residual_sorted[:, dominant_channel].mean()
    signs = torch.sign(probe)
    sign_products = signs[:-1] * signs[1:]
    zero_crossings = int((sign_products < 0).sum().detach().cpu().item())
    kappa = float(max(min_kappa, min(max_kappa, 3 * zero_crossings + 1)))
    return StageDiagnostics(
        residual_rms=residual_rms,
        residual_zero_crossings=zero_crossings,
        kappa_suggested=kappa,
        dominant_residual_channel=dominant_channel,
    )
