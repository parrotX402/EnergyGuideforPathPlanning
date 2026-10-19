"""Paper architecture, both DSM sign conventions, and frozen reward inference."""

from dataclasses import asdict, dataclass
from pathlib import Path
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class EnergyConfig:
    representation: str = "paper4"
    hidden: tuple = (265, 512, 1024, 512, 256)
    sigmas: tuple = (1.0, 0.1, 0.01)
    # Explicit reconstruction assumption: noise embedding added to first hidden layer.
    conditioned: bool = True
    convention: str = "negative_log_density"

    def __post_init__(self):
        if self.representation not in ("paper4", "legacy56"):
            raise ValueError("representation must be paper4 or legacy56")
        if self.convention not in ("negative_log_density", "paper_literal"):
            raise ValueError("Unknown DSM convention")
        if not self.hidden or any(int(n) <= 0 for n in self.hidden):
            raise ValueError("hidden layers must be positive")
        if not self.sigmas or any(not math.isfinite(s) or s <= 0 for s in self.sigmas):
            raise ValueError("sigmas must be finite and positive")

    @property
    def input_dim(self):
        return 4 if self.representation == "paper4" else 56


class EnergyMLP(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or EnergyConfig()
        dims = (self.config.input_dim, *self.config.hidden, 1)
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims, dims[1:]))
        self.noise_embedding = (
            nn.Embedding(len(self.config.sigmas), dims[1]) if self.config.conditioned else None
        )
        # Identity preprocessing until fitted, entirely from training data.
        self.register_buffer("center", torch.zeros(dims[0]))
        self.register_buffer("scale", torch.ones(dims[0]))

    def forward(self, normalized_x, noise_index):
        if normalized_x.shape[-1] != self.config.input_dim:
            raise ValueError(f"Expected {self.config.input_dim} features")
        h = self.layers[0](normalized_x)
        if self.noise_embedding is not None:
            h = h + self.noise_embedding(noise_index)
        h = F.elu(h)
        for layer in self.layers[1:-1]:
            h = F.elu(layer(h))
        return self.layers[-1](h).squeeze(-1)

    def normalize(self, raw_x):
        return (raw_x - self.center) / self.scale

    def density_energy(self, normalized_x, noise_index):
        raw = self(normalized_x, noise_index)
        # Literal paper objective fits log p rather than -log p.
        return -raw if self.config.convention == "paper_literal" else raw


def dsm_loss(model, clean, noise_index=None, noise=None):
    """Sample levels uniformly; weight the squared error by sigma**2 / 2.

    clean is already normalized. Corrected: grad E - (noisy-clean)/sigma².
    Paper literal: grad E + (noisy-clean)/sigma².
    """
    count = clean.shape[0]
    if noise_index is None:
        noise_index = torch.randint(len(model.config.sigmas), (count,), device=clean.device)
    sigma = clean.new_tensor(model.config.sigmas)[noise_index, None]
    noise = torch.randn_like(clean) if noise is None else noise
    noisy = (clean + sigma * noise).detach().requires_grad_(True)
    potential = model(noisy, noise_index)
    grad = torch.autograd.grad(potential.sum(), noisy, create_graph=True)[0]
    target = noise / sigma
    error = grad + target if model.config.convention == "paper_literal" else grad - target
    return (0.5 * sigma.square() * error.square()).sum(-1).mean()


def save_checkpoint(path, model, optimizer=None, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "config": asdict(model.config),
        "model": model.state_dict(),
        "metadata": metadata,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)


class FrozenReward:
    """A portable scalar/batch shaping module, usable with any policy backbone.

    calibrated: exp(-softplus(E-reference)/beta); this is an explicit assumption.
    literal: exp(-raw_model_output/beta), exactly the displayed mapping, unbounded.
    """

    def __init__(self, checkpoint, beta=1.0, mapping="calibrated", device="cpu"):
        if not math.isfinite(beta) or beta <= 0:
            raise ValueError("beta must be finite and positive")
        if mapping not in ("calibrated", "literal"):
            raise ValueError("mapping must be calibrated or literal")
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
        if payload.get("format_version") != 1:
            raise ValueError("Not an EnergyGuide reconstruction checkpoint")
        self.model = EnergyMLP(EnergyConfig(**payload["config"])).to(device)
        self.model.load_state_dict(payload["model"], strict=True)
        self.model.eval().requires_grad_(False)
        self.metadata = payload["metadata"]
        self.reference = float(self.metadata.get("energy_reference", 0.0))
        self.beta = beta
        self.mapping = mapping

    @torch.no_grad()
    def __call__(self, features):
        x = torch.as_tensor(features, dtype=torch.float32, device=self.model.center.device)
        single = x.ndim == 1
        if single:
            x = x.unsqueeze(0)
        if x.ndim != 2 or x.shape[-1] != self.model.config.input_dim or not torch.isfinite(x).all():
            raise ValueError("features have wrong shape or non-finite values")
        # Evaluate at the smallest training noise level, explicit inference assumption.
        level = int(np.argmin(self.model.config.sigmas))
        indices = torch.full((len(x),), level, dtype=torch.long, device=x.device)
        normalized = self.model.normalize(x)
        if self.mapping == "literal":
            energy = self.model(normalized, indices)
        else:
            energy = F.softplus(self.model.density_energy(normalized, indices) - self.reference)
        reward = torch.exp(-energy / self.beta)
        if not torch.isfinite(reward).all():
            raise FloatingPointError("Reward overflow; literal energy mapping is not bounded")
        result = reward.cpu().numpy()
        return float(result[0]) if single else result
