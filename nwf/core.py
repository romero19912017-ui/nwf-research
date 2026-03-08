# -*- coding: utf-8 -*-
"""Charge, потенциал phi_i, расстояние Махаланобиса (Аксиома А3).

Нейровесовые поля представляют семантику в виде ЗАРЯДОВ в латентном пространстве.
Каждый заряд — это не точка, а точка + мера неопределённости (ковариация).
Это позволяет учитывать, насколько модель «уверена» в положении объекта.

Математика:
  - Потенциал заряда i в точке r: phi_i(r) = exp(-(r-z_i)^T Sigma_i^-1 (r-z_i) / 2)
  - Расстояние Махаланобиса: ||r-z||_Sigma = sqrt((r-z)^T Sigma^-1 (r-z))

Для диагональной Sigma (как в VAE): d^2 = sum_d (r_d - z_d)^2 / sigma_d^2
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class Charge:
    """Заряд — базовый объект NWF: центр z и диагональ ковариации sigma_sq.

    В теории NWF каждый семантический объект (изображение, текст и т.д.)
    представляется зарядом. z — «где» объект в латентном пространстве,
    sigma_sq — «насколько размыто» его положение (неопределённость).

    Attributes:
        z: Центр заряда, shape (latent_dim,)
        sigma_sq: Диагональ ковариации Sigma (храним дисперсии), shape (latent_dim,)
    """
    z: torch.Tensor  # (latent_dim,)
    sigma_sq: torch.Tensor  # (latent_dim,) диагональ Sigma

    def to(self, device: torch.device) -> "Charge":
        return Charge(z=self.z.to(device), sigma_sq=self.sigma_sq.to(device))

    @property
    def Sigma_inv(self) -> torch.Tensor:
        """Обратная ковариация (диагональ). Нужна для расстояния Махаланобиса."""
        eps = 1e-8
        return 1.0 / (self.sigma_sq + eps)


def mahalanobis_dist(
    r: torch.Tensor,
    z: torch.Tensor,
    sigma_sq: torch.Tensor,
) -> torch.Tensor:
    """Расстояние Махаланобиса: ||r - z||_Sigma для диагональной Sigma.

    В отличие от евклидова, масштабирует каждую координату на 1/sigma_d,
    т.е. в направлениях с большей неопределённостью «разрешает» большие отклонения.
    """
    diff = r - z
    sigma_sq_safe = sigma_sq + 1e-8
    return torch.sqrt(((diff ** 2) / sigma_sq_safe).sum())


def mahalanobis_dist_batch(
    r: torch.Tensor,  # (N, D) или (D,)
    z: torch.Tensor,  # (M, D) или (D,)
    sigma_sq: torch.Tensor,  # (M, D) или (D,)
) -> torch.Tensor:
    """Батч-расстояние Махаланобиса: r (N,D) к зарядам z (M,D).

    Использует Sigma индексируемых зарядов (запрос r сравнивается с каждым z_i
    по Sigma_i). Выход: матрица (N, M) расстояний.
    """
    if r.dim() == 1:
        r = r.unsqueeze(0)
    if z.dim() == 1:
        z = z.unsqueeze(0)
        sigma_sq = sigma_sq.unsqueeze(0)
    diff = r.unsqueeze(1) - z.unsqueeze(0)  # (N,M,D)
    sigma_sq_safe = sigma_sq.unsqueeze(0) + 1e-8
    d_sq = ((diff ** 2) / sigma_sq_safe).sum(dim=-1)
    return torch.sqrt(d_sq + 1e-8)


def symmetric_mahalanobis_batch(
    r: torch.Tensor,  # (N, D)
    z: torch.Tensor,  # (M, D)
    sigma_r: torch.Tensor,  # (N, D) - Sigma query
    sigma_z: torch.Tensor,  # (M, D) - Sigma stored
) -> torch.Tensor:
    """Симметричная метрика Махаланобиса: (Sigma_q + Sigma_i)^{-1}.

    Учитывает неопределённости и запроса, и индекса. Более устойчива к шуму,
    чем обычный Махаланобис. Формула: d^2 = (r-z)^T (Sigma_r + Sigma_z)^{-1} (r-z)
    """
    if r.dim() == 1:
        r = r.unsqueeze(0)
        sigma_r = sigma_r.unsqueeze(0)
    if z.dim() == 1:
        z = z.unsqueeze(0)
        sigma_z = sigma_z.unsqueeze(0)
    diff = r.unsqueeze(1) - z.unsqueeze(0)  # (N,M,D)
    sigma_comb = sigma_r.unsqueeze(1) + sigma_z.unsqueeze(0) + 1e-8  # (N,M,D)
    d_sq = ((diff ** 2) / sigma_comb).sum(dim=-1)
    return torch.sqrt(d_sq + 1e-8)


def potential(d_sq: torch.Tensor) -> torch.Tensor:
    """Потенциал phi = exp(-d^2/2). Чем меньше d — тем выше потенциал.

    Интерпретация: «насколько запрос принадлежит» облаку заряда.
    Используется для OOD-детекции и метрик уверенности.
    """
    return torch.exp(-d_sq / 2.0)


def charge_to_potential_at(r: torch.Tensor, charge: Charge) -> torch.Tensor:
    """Потенциал одного заряда в точке r (скаляр)."""
    d_sq = ((r - charge.z) ** 2 / (charge.sigma_sq + 1e-8)).sum()
    return potential(d_sq)
