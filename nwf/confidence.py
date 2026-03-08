# -*- coding: utf-8 -*-
"""Метрики уверенности для NWF: от «насколько мы уверены» до калибровки.

NWF даёт не только предсказание, но и оценку неопределённости. Эти метрики
используются для калибровки (ECE) и доверенного ИИ. До калибровки лучше всего
работает agreement_ratio; после Platt scaling ECE снижается до ~0.03.
"""
from __future__ import annotations

from typing import Tuple

import torch

from .core import mahalanobis_dist_batch, symmetric_mahalanobis_batch
from .storage import NWFStorage


def min_mahalanobis(
    storage: NWFStorage,
    query_z: torch.Tensor,
    query_sigma_sq: torch.Tensor,
    metric: str = "mahalanobis",
) -> torch.Tensor:
    """Минимальное расстояние до ближайшего заряда. Больше d = меньше уверенность."""
    idx, dists = storage.search(query_z, query_sigma_sq, k=1, metric=metric)
    return dists.flatten()[0] if dists.numel() == 1 else dists[:, 0]


def potential_at_query(
    storage: NWFStorage,
    query_z: torch.Tensor,
    query_sigma_sq: torch.Tensor,
    metric: str = "mahalanobis",
    k: int = 100,
) -> torch.Tensor:
    """Потенциал Phi(z_q) = sum exp(-0.5*d_i^2) по k ближайшим зарядам.

    Высокий потенциал = запрос «внутри» облака обученных данных (in-distribution).
    Низкий = OOD. Используется для OOD-детекции.
    """
    idx, dists = storage.search(query_z, query_sigma_sq, k=min(k, len(storage)), metric=metric)
    if dists.dim() == 1:
        return torch.exp(-0.5 * (dists ** 2)).sum()
    return torch.exp(-0.5 * (dists ** 2)).sum(dim=1)


def trace_sigma(sigma_sq: torch.Tensor) -> torch.Tensor:
    """След Sigma = sum(sigma_sq). Больше = выше неопределённость запроса."""
    return sigma_sq.sum(dim=-1)


def agreement_ratio(
    storage: NWFStorage,
    query_z: torch.Tensor,
    query_sigma_sq: torch.Tensor,
    true_label: int,
    k: int = 10,
    metric: str = "mahalanobis",
) -> float:
    """Доля соседей с той же меткой среди k ближайших. Лучшая метрика для калибровки."""
    idx, _ = storage.search(query_z, query_sigma_sq, k=k, metric=metric)
    labels = storage.get_labels(idx)
    if labels.dim() == 0:
        return 1.0 if labels.item() == true_label else 0.0
    if labels.dim() == 1:
        return (labels == true_label).float().mean().item()
    return (labels[:, 0] == true_label).float().mean().item()


def confidence_1_over_1_plus_d(d: torch.Tensor) -> torch.Tensor:
    """Уверенность = 1/(1+d). Малое d -> высокая уверенность. Плохо калибрована."""
    return 1.0 / (1.0 + d)
