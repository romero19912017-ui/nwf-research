# -*- coding: utf-8 -*-
"""Заряд NWF: позиция, интенсивность, ковариация.

Заряд — элементарная единица хранения в NWF. Содержит:
- z: позиция в пространстве эмбеддингов
- q: интенсивность (масса заряда)
- sigma: диагональ ковариационной матрицы (ширина гауссова колокола по каждой оси)
"""
import torch


class Charge:
    """Заряд NWF: позиция (z), интенсивность (q), диагональ ковариации (sigma)."""

    def __init__(self, z, q=1.0, sigma=0.5):
        self.z = torch.tensor(z, dtype=torch.float32)
        self.q = q
        self.sigma = torch.full_like(self.z, sigma)


def init_charge(z, q=1.0, sigma_init=0.5):
    """Создать заряд с заданной позицией и опциональной начальной sigma."""
    return Charge(z, q, sigma_init)
