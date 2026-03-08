# -*- coding: utf-8 -*-
"""Нейровесовые Поля (NWF) — реализация на PyTorch.

Метод машинного обучения на основе градиентного поля потенциала гауссовых зарядов.
Данные хранятся как заряды, классификация — трассировка луча + взвешенное голосование.
Без обратного распространения, без катастрофического забывания.

Теория: https://doi.org/10.24108/preprints-3113697
"""
from nwf.charge import Charge, init_charge
from nwf.field import gaussian_potential, field_and_grad
from nwf.trace import trace_ray, trace_ray_memory
from nwf.index import Memory
from nwf.encoder import Encoder
from nwf.voting import classify_weighted, classify_nearest

__all__ = [
    "Charge",
    "init_charge",
    "gaussian_potential",
    "field_and_grad",
    "trace_ray",
    "trace_ray_memory",
    "Memory",
    "Encoder",
    "classify_weighted",
    "classify_nearest",
]
