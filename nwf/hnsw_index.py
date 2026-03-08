# -*- coding: utf-8 -*-
"""HNSW индекс для приближённого поиска по Махаланобису.

Ограничение: симметричная метрика (Sigma_q + Sigma_i) не допускает предвычисления
для каждого индекса. Используем аппроксимацию: глобальная Sigma_avg.

White-преобразование: w = z / sqrt(sigma_avg). Тогда L2(w_q, w_i) ~ Mahalanobis.
Позволяет использовать стандартный FAISS HNSW. Ускорение ~36x при малой потере точности.
"""
from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np
import torch

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


def whiten_transform(z: np.ndarray, sigma_sq: np.ndarray, use_global: bool = True,
                     sigma_global: Optional[np.ndarray] = None) -> np.ndarray:
    """White-преобразование: w = z / sqrt(sigma_sq), для диагональной Sigma.

    d_Mahal^2 = (z1 - z2)^T Sigma^{-1} (z1 - z2) = ||w1 - w2||^2,
    где w = z / sqrt(sigma_sq).
    """
    if sigma_global is not None:
        sigma_sq = sigma_global
    sigma_sq = np.asarray(sigma_sq, dtype=np.float32)
    sigma_sq = np.maximum(sigma_sq, 1e-8)
    return z / np.sqrt(sigma_sq)


class HNSWMahalanobisIndex:
    """HNSW индекс для приближённого поиска по Махаланобису (глобальная Sigma)."""

    def __init__(self, d: int, n_links: int = 32, use_faiss: bool = True):
        if not HAS_FAISS and use_faiss:
            raise ImportError("faiss-cpu required: pip install faiss-cpu")
        self.d = d
        self.n_links = n_links
        self.use_faiss = use_faiss and HAS_FAISS
        self._z_raw: list = []
        self._sigma_raw: list = []
        self._labels: list = []
        self._w: Optional[np.ndarray] = None
        self._sigma_global: Optional[np.ndarray] = None
        self._index: Optional["faiss.IndexHNSWFlat"] = None

    def add_batch(self, z: torch.Tensor, sigma_sq: torch.Tensor, labels: torch.Tensor) -> None:
        z_np = z.detach().cpu().numpy().astype(np.float32)
        s_np = sigma_sq.detach().cpu().numpy().astype(np.float32)
        labs = labels.cpu().tolist()
        for i in range(z_np.shape[0]):
            self._z_raw.append(z_np[i])
            self._sigma_raw.append(s_np[i])
            self._labels.append(labs[i])

    def build_index(self) -> None:
        """Строит HNSW индекс после добавления точек."""
        Z = np.stack(self._z_raw, axis=0)
        S = np.stack(self._sigma_raw, axis=0)
        self._sigma_global = np.mean(S, axis=0).astype(np.float32)
        self._sigma_global = np.maximum(self._sigma_global, 1e-8)
        self._w = whiten_transform(Z, S, sigma_global=self._sigma_global)
        if self.use_faiss:
            self._index = faiss.IndexHNSWFlat(self.d, self.n_links, faiss.METRIC_L2)
            self._index.hnsw.efConstruction = 200
            self._index.add(self._w)

    def search(
        self,
        query_z: np.ndarray,
        query_sigma_sq: np.ndarray,
        k: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """k-NN. Использует глобальную Sigma для white-преобразования запроса."""
        if self._index is None:
            self.build_index()
        q = np.asarray(query_z, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        q_white = whiten_transform(q, query_sigma_sq, sigma_global=self._sigma_global)
        distances, indices = self._index.search(q_white, min(k, len(self._labels)))
        if q.shape[0] == 1:
            return indices[0], distances[0]
        return indices, distances

    def get_labels(self, indices: np.ndarray) -> np.ndarray:
        if indices.ndim == 1:
            return np.array([self._labels[i] for i in indices])
        return np.array([[self._labels[i] for i in row] for row in indices])

    def __len__(self) -> int:
        return len(self._z_raw)
