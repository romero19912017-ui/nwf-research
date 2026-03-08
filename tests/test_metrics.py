# -*- coding: utf-8 -*-
"""Unit-тесты метрик: Махаланобис, симметричная метрика."""
import pytest
import torch
import numpy as np

from nwf.core import (
    mahalanobis_dist,
    mahalanobis_dist_batch,
    symmetric_mahalanobis_batch,
    potential,
)


def test_mahalanobis_simple():
    """Махаланобис для простого случая: d^2 = sum((r-z)^2/sigma)."""
    torch.manual_seed(42)
    r = torch.tensor([1.0, 2.0, 3.0])
    z = torch.tensor([0.0, 0.0, 0.0])
    sigma_sq = torch.ones(3)
    d = mahalanobis_dist(r, z, sigma_sq)
    expected = torch.sqrt(((r - z) ** 2 / sigma_sq).sum())
    assert torch.allclose(d, expected, atol=1e-5)


def test_symmetric_mahalanobis_formula():
    """Симметричная: d^2 = sum((r-z)^2/(sigma_r+sigma_z))."""
    torch.manual_seed(42)
    r = torch.tensor([[1.0, 2.0]])
    z = torch.tensor([[0.0, 0.0]])
    sigma_r = torch.tensor([[1.0, 1.0]])
    sigma_z = torch.tensor([[2.0, 2.0]])
    d = symmetric_mahalanobis_batch(r, z, sigma_r, sigma_z)
    diff_sq = (r - z) ** 2
    sigma_comb = sigma_r + sigma_z
    expected = torch.sqrt((diff_sq / sigma_comb).sum())
    assert torch.allclose(d.squeeze(), expected, atol=1e-5)


def test_symmetric_vs_scipy():
    """Сравнение с scipy.spatial.distance.mahalanobis при Sigma = Sigma_q+Sigma_i."""
    try:
        from scipy.spatial.distance import mahalanobis as scipy_maha
    except ImportError:
        pytest.skip("scipy not installed")
    torch.manual_seed(42)
    r = torch.tensor([1.0, -0.5, 2.0])
    z = torch.tensor([0.0, 0.0, 0.0])
    sigma_r = torch.tensor([1.0, 2.0, 0.5])
    sigma_z = torch.tensor([1.0, 1.0, 1.0])
    sigma_comb = sigma_r + sigma_z
    cov_combined = torch.diag(sigma_comb).numpy()
    cov_inv = np.linalg.inv(cov_combined)
    d_scipy = scipy_maha(r.numpy(), z.numpy(), cov_inv)
    d_ours = symmetric_mahalanobis_batch(
        r.unsqueeze(0), z.unsqueeze(0),
        sigma_r.unsqueeze(0), sigma_z.unsqueeze(0),
    ).item()
    assert abs(d_ours - d_scipy) < 1e-5


def test_potential_range():
    """Потенциал exp(-d^2/2) в (0, 1]."""
    for d_sq in [0.0, 0.1, 1.0, 10.0, 100.0]:
        p = potential(torch.tensor(d_sq))
        assert 0 < p <= 1.001
    assert torch.allclose(potential(torch.tensor(0.0)), torch.tensor(1.0))
