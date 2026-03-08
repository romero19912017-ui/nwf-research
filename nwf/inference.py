# -*- coding: utf-8 -*-
"""Вывод (z, Sigma) из VAE для NWF.

VAE даёт (mu, log_var). Мы используем MAP-оценку: z = mu, sigma_sq = exp(log_var).
Без сэмплирования — детерминированно. Это «точечная» оценка заряда для индексации.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from .core import Charge
from .vae_encoder import VAEEncoder


def vae_loss(recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """ELBO = BCE(recon, x) + KL(q||p). Стандартная функция потерь VAE."""
    bce = F.binary_cross_entropy(recon, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return bce + kl


def encode_batch(
    model: VAEEncoder,
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """MAP-оценка заряда: (z, sigma_sq) для батча x.

    z = mu (центр), sigma_sq = exp(log_var) (диагональ ковариации).
    Основная функция для кодирования при индексации и поиске.
    """
    with torch.no_grad():
        mu, log_var = model.encode(x)
        sigma_sq = log_var.exp().clamp(min=1e-6)
    return mu, sigma_sq


def encode_to_charges(model: VAEEncoder, x: torch.Tensor) -> list[Charge]:
    """Преобразовать батч x в список зарядов Charge."""
    mu, sigma_sq = encode_batch(model, x)
    return [Charge(z=mu[i], sigma_sq=sigma_sq[i]) for i in range(mu.size(0))]
