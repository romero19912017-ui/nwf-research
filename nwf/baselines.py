# -*- coding: utf-8 -*-
"""Бейзлайны для сравнения с NWF.

L2Index: аналог FAISS — хранит только z, поиск по евклидову расстоянию.
HDC (Hyperdimensional Computing): случайная проекция 784->D, бинаризация, поиск по Хэммингу.
Используются в экспериментах 01 (сжатие), 02 (шум) для сравнения.
"""
from __future__ import annotations

from typing import List, Tuple, Optional

import numpy as np
import torch


class L2Index:
    """FAISS-подобный индекс: только z, поиск по евклидову расстоянию."""

    def __init__(self):
        self._z: List[np.ndarray] = []
        self._labels: List[int] = []

    def add_batch(self, z: torch.Tensor, labels: torch.Tensor) -> None:
        z_np = z.detach().cpu().numpy()
        labs = labels.cpu().tolist()
        for i in range(z_np.shape[0]):
            self._z.append(z_np[i])
            self._labels.append(labs[i])

    def search(self, query: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """k-NN по L2. query: (D,) или (N,D). Returns (indices, distances)."""
        Z = np.stack(self._z, axis=0)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        dists = np.linalg.norm(query[:, np.newaxis, :] - Z[np.newaxis, :, :], axis=2)
        k_act = min(k, dists.shape[1])
        idx = np.argpartition(dists, k_act - 1, axis=1)[:, :k_act]
        rows = np.arange(idx.shape[0])[:, np.newaxis]
        sorted_order = np.argsort(dists[rows, idx])
        idx = np.take_along_axis(idx, sorted_order, axis=1)
        vals = np.take_along_axis(dists, idx, axis=1)
        if idx.shape[0] == 1:
            return idx[0], vals[0]
        return idx, vals

    def get_labels(self, indices: np.ndarray) -> np.ndarray:
        if indices.ndim == 1:
            return np.array([self._labels[i] for i in indices])
        return np.array([[self._labels[i] for i in row] for row in indices])

    def __len__(self) -> int:
        return len(self._z)


class HDCEncoder:
    """Hyperdimensional Computing: случайная проекция 784 -> D, бинаризация."""

    def __init__(self, dim: int = 10000, seed: int = 42):
        self.dim = dim
        self.rng = np.random.default_rng(seed)
        self.proj = self.rng.choice([-1, 1], size=(784, dim))

    def encode(self, x: np.ndarray) -> np.ndarray:
        """x: (N, 784). Returns (N, dim) binary +1/-1."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        h = np.sign(x @ self.proj)
        h[h == 0] = 1
        return h

    def encode_torch(self, x: torch.Tensor) -> np.ndarray:
        return self.encode(x.detach().cpu().numpy())


class HDCIndex:
    """HDC индекс: поиск по расстоянию Хэмминга."""

    def __init__(self, dim: int = 2000, seed: int = 42):
        self.encoder = HDCEncoder(dim=dim, seed=seed)
        self._codes: List[np.ndarray] = []
        self._labels: List[int] = []

    def add_batch(self, x: torch.Tensor, labels: torch.Tensor) -> None:
        codes = self.encoder.encode_torch(x)
        labs = labels.cpu().tolist()
        for i in range(codes.shape[0]):
            self._codes.append(codes[i])
            self._labels.append(labs[i])

    def search(self, query_code: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """k-NN по Хэммингу (кол-во различающихся бит)."""
        codes = np.stack(self._codes, axis=0)
        if query_code.ndim == 1:
            query_code = query_code.reshape(1, -1)
        hamming = (query_code[:, np.newaxis, :] != codes[np.newaxis, :, :]).sum(axis=2)
        k_act = min(k, hamming.shape[1])
        idx = np.argpartition(hamming, k_act - 1, axis=1)[:, :k_act]
        rows = np.arange(idx.shape[0])[:, np.newaxis]
        sorted_order = np.argsort(hamming[rows, idx])
        idx = np.take_along_axis(idx, sorted_order, axis=1)
        vals = np.take_along_axis(hamming, idx, axis=1)
        if idx.shape[0] == 1:
            return idx[0], vals[0]
        return idx, vals

    def get_labels(self, indices: np.ndarray) -> np.ndarray:
        if indices.ndim == 1:
            return np.array([self._labels[i] for i in indices])
        return np.array([[self._labels[i] for i in row] for row in indices])

    def __len__(self) -> int:
        return len(self._codes)
