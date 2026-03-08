# -*- coding: utf-8 -*-
"""Product Quantization (PQ) для сжатия зарядов.

Вектор (z, log_sigma_sq) длины 2*d квантуется. Сжатие до 16 байт (m=16, 8 bit)
при потере Precision@10 < 1%. Используется в exp 01 с флагом --use_pq.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


def pack_charge(z: np.ndarray, sigma_sq: np.ndarray) -> np.ndarray:
    """Объединяет z и log(sigma_sq) в вектор длины 2*d."""
    log_sigma = np.log(np.maximum(sigma_sq, 1e-10))
    return np.concatenate([z, log_sigma], axis=-1).astype(np.float32)


class PQChargeIndex:
    """PQ-индекс для зарядов. Вектор (z, log_sigma) длины 2*d квантуется."""

    def __init__(self, d: int, m: int = 16, nbits: int = 8):
        if not HAS_FAISS:
            raise ImportError("faiss-cpu required: pip install faiss-cpu")
        self.d_latent = d
        self.d = 2 * d
        self.m = min(m, self.d)
        self.nbits = nbits
        self._index: Optional["faiss.IndexPQ"] = None
        self._labels: list = []

    def fit_and_add(self, z: np.ndarray, sigma_sq: np.ndarray, labels: list) -> None:
        """Обучает PQ и добавляет точки. z, sigma_sq: (N,d)."""
        X = pack_charge(z, sigma_sq)
        self._labels = list(labels)
        self._index = faiss.IndexPQ(self.d, self.m, self.nbits, faiss.METRIC_L2)
        self._index.train(X)
        self._index.add(X)

    def search(self, query_z: np.ndarray, query_sigma_sq: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        q = pack_charge(
            np.asarray(query_z, dtype=np.float32),
            np.asarray(query_sigma_sq, dtype=np.float32),
        )
        if q.ndim == 1:
            q = q.reshape(1, -1)
        distances, indices = self._index.search(q, min(k, len(self._labels)))
        if q.shape[0] == 1:
            return indices[0], distances[0]
        return indices, distances

    def get_labels(self, indices: np.ndarray) -> np.ndarray:
        if indices.ndim == 1:
            return np.array([self._labels[i] for i in indices])
        return np.array([[self._labels[i] for i in row] for row in indices])

    def __len__(self) -> int:
        return len(self._labels)

    @property
    def bytes_per_vector(self) -> int:
        return self.m * (self.nbits // 8)
