from dataclasses import dataclass, field
from typing import Optional, Dict, Union, Literal

import torch
from gpytorch.constraints import Interval, GreaterThan, Positive
from gpytorch.priors import GammaPrior, Prior


@dataclass
class Priors:
    lengthscale_prior: Prior = field(default_factory=lambda: GammaPrior(3.0, 0.5))
    outputscale_prior: Prior = field(default_factory=lambda: GammaPrior(3.0, 0.5))
    noise_prior: Prior = field(default_factory=lambda: GammaPrior(1.1, 0.05))
    df_prior: Prior = field(default_factory=lambda: GammaPrior(2.0, 0.15))

@dataclass
class Constraints:
    lengthscale_constraint: Interval = field(default_factory=Positive)
    outputscale_constraint: Interval = field(default_factory=Positive)
    noise_constraint: Interval = field(default_factory=Positive)
    df_constraint: Interval = field(default_factory=lambda: GreaterThan(2 + 1e-4))

@dataclass
class Device:
    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float


@dataclass
class ModelConfig:
    ard_num_dims: Optional[int]  # Changed to Optional[int]
    kernel_type: Literal["rbf", "matern32", "matern52"] = "rbf"
    priors: Priors = field(default_factory=Priors)
    constraints: Constraints = field(default_factory=Constraints)
    device: Device = field(default_factory=Device)
    trained_params: Optional[Dict[str, Union[float, torch.Tensor]]] = None