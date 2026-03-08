# -*- coding: utf-8 -*-
"""Потенциал Гаусса и градиент поля NWF.

Потенциал одного заряда: phi = q * exp(-0.5 * d^2), где d — расстояние Махаланобиса
между точкой r и центром заряда z. Градиент поля направлен к ближайшим зарядам.
"""
import torch


def gaussian_potential(r, z, q, sigma):
    """Скалярный потенциал: q * exp(-0.5 * sum((r - z)^2 / sigma^2))."""
    diff = r - z
    dist_sq = torch.sum((diff**2) / (sigma**2 + 1e-8))
    return q * torch.exp(-0.5 * dist_sq)


def field_and_grad(r, zs, qs, sigmas, indices=None):
    """Суммарный потенциал и его градиент по r. r — точка, zs/qs/sigmas — заряды."""
    if indices is not None:
        zs = zs[indices]
        qs = qs[indices]
        sigmas = sigmas[indices]

    diff = r.unsqueeze(0) - zs  # (N, dim)
    dist_sq = torch.sum((diff**2) / (sigmas**2 + 1e-8), dim=1)  # (N,)
    pots = qs * torch.exp(-0.5 * dist_sq)  # (N,)
    field_val = pots.sum()

    grad = torch.autograd.grad(field_val, r, create_graph=False)[0]
    return field_val, grad
