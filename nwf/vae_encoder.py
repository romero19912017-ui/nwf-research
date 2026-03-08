# -*- coding: utf-8 -*-
"""VAE-энкодер для получения (z, Sigma) по теории NWF.

VAE (Variational Autoencoder) — основа представления в NWF. Encoder выдаёт
(mu, log_var) для латентного пространства; мы интерпретируем:
  - z = mu — центр заряда
  - sigma_sq = exp(log_var) — диагональ ковариации (неопределённость)

VAE естественным образом даёт распределение p(z|x), откуда берём и центр, и разброс.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class VAEEncoder(nn.Module):
    """VAE для латентного представления: (mu, log_var) -> Charge(z, sigma_sq).

    Архитектура: encoder -> (fc_mu, fc_logvar), decoder.
    sigma_sq = exp(log_var) — дисперсия по каждой латентной размерности.
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: Tuple[int, ...] = (512, 256),
        latent_dim: int = 64,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)
        # Decoder
        dec_layers = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec_layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        dec_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        return mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        return recon, mu, log_var
