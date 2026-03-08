# -*- coding: utf-8 -*-
"""Хранилище зарядов (z_i, Sigma_i). k-NN поиск по Махаланобису.

NWFStorage — центральная структура данных NWF. В отличие от FAISS (хранит только z),
здесь хранятся заряды (z, sigma_sq). Поиск выполняется по расстоянию Махаланобиса
или симметричной метрике. Добавление новых зарядов — O(1), без переобучения.
"""
from __future__ import annotations

from typing import List, Tuple, Optional

import torch

from .core import Charge, mahalanobis_dist_batch, symmetric_mahalanobis_batch


class NWFStorage:
    """Хранилище зарядов с k-NN поиском по Махаланобису.

    Позволяет инкрементально добавлять заряды (add, add_batch) и искать
    ближайших соседей для запроса (z_q, Sigma_q). Ключевое свойство:
    добавление новых классов не влияет на старые (нет катастрофического забывания).
    """

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cpu")
        self._z: List[torch.Tensor] = []
        self._sigma_sq: List[torch.Tensor] = []
        self._labels: List[int] = []

    def add(self, charge: Charge, label: int) -> None:
        z = charge.z.detach().to(self.device)
        s = charge.sigma_sq.detach().to(self.device)
        self._z.append(z)
        self._sigma_sq.append(s)
        self._labels.append(label)

    def add_batch(self, z: torch.Tensor, sigma_sq: torch.Tensor, labels: torch.Tensor) -> None:
        """Добавить батч зарядов. z, sigma_sq: (N,D), labels: (N,)."""
        z = z.detach().to(self.device)
        s = sigma_sq.detach().to(self.device)
        labs = labels.cpu().tolist()
        for i in range(z.size(0)):
            self._z.append(z[i])
            self._sigma_sq.append(s[i])
            self._labels.append(labs[i])

    def search(
        self,
        query_z: torch.Tensor,
        query_sigma_sq: torch.Tensor,
        k: int = 10,
        metric: str = "mahalanobis",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """k ближайших соседей по выбранной метрике.

        Args:
            query_z: Центр запроса (D,) или (N,D)
            query_sigma_sq: Неопределённость запроса
            k: Число соседей
            metric: "mahalanobis" (Sigma индекса) или "symmetric" (Sigma_q + Sigma_i)

        Returns:
            (indices, distances) — индексы и расстояния до k ближайших зарядов
        """
        qz = query_z.to(self.device)
        qs = query_sigma_sq.to(self.device)
        if len(self._z) == 0:
            raise RuntimeError("Storage is empty")
        Z = torch.stack(self._z, dim=0)
        S = torch.stack(self._sigma_sq, dim=0)
        if metric == "symmetric":
            qz_ = qz.unsqueeze(0) if qz.dim() == 1 else qz
            qs_ = qs.unsqueeze(0) if qs.dim() == 1 else qs
            dists = symmetric_mahalanobis_batch(qz_, Z, qs_, S)
            if qz.dim() == 1:
                dists = dists.squeeze(0)
        else:
            dists = mahalanobis_dist_batch(qz, Z, S)
        if dists.dim() == 1:
            vals, idx = torch.topk(dists, min(k, dists.size(0)), largest=False)
            return idx, vals
        else:
            k_act = min(k, dists.size(1))
            vals, idx = torch.topk(dists, k_act, dim=1, largest=False)
            return idx, vals

    def search_euclidean(self, query_z: torch.Tensor, k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """k-NN по евклидову расстоянию (игнорируя Sigma)."""
        qz = query_z.to(self.device)
        if len(self._z) == 0:
            raise RuntimeError("Storage is empty")
        Z = torch.stack(self._z, dim=0)
        diff = qz.unsqueeze(0) - Z
        dists = torch.sqrt((diff ** 2).sum(dim=1) + 1e-8)
        vals, idx = torch.topk(dists, min(k, dists.size(0)), largest=False)
        return idx, vals

    def get_labels(self, indices: torch.Tensor) -> torch.Tensor:
        """Получить метки по индексам."""
        if indices.dim() == 1:
            return torch.tensor([self._labels[i] for i in indices.cpu().tolist()])
        return torch.tensor([[self._labels[i] for i in row] for row in indices.cpu().tolist()])

    def __len__(self) -> int:
        return len(self._z)
