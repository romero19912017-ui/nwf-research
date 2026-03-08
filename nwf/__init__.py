# -*- coding: utf-8 -*-
"""NWF Research — Нейровесовые поля.

Реализация теории «Нейровесовые поля: семантический континуум» (препринт).

Модули:
  - core: Charge, расстояние Махаланобиса, потенциал
  - vae_encoder: VAE для (z, Sigma)
  - inference: encode_batch, encode_to_charges
  - storage: NWFStorage — хранилище зарядов, k-NN
  - kalman: KalmanEncoder — итеративное уточнение
  - confidence: метрики уверенности
  - baselines: FAISS (L2), HDC
  - hnsw_index, pq_index: масштабирование
"""
from .core import Charge, mahalanobis_dist, mahalanobis_dist_batch, potential
from .storage import NWFStorage
from .vae_encoder import VAEEncoder
from .inference import vae_loss, encode_batch, encode_to_charges
from .kalman import KalmanEncoder
from .data import get_mnist, get_cifar10

__all__ = [
    "Charge",
    "mahalanobis_dist",
    "mahalanobis_dist_batch",
    "potential",
    "NWFStorage",
    "VAEEncoder",
    "vae_loss",
    "encode_batch",
    "encode_to_charges",
    "KalmanEncoder",
    "get_mnist",
    "get_cifar10",
]
