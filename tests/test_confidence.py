# -*- coding: utf-8 -*-
"""Unit-тесты метрик уверенности."""
import pytest
import torch

from nwf.storage import NWFStorage
from nwf.core import Charge
from nwf.confidence import (
    min_mahalanobis,
    potential_at_query,
    trace_sigma,
    agreement_ratio,
    confidence_1_over_1_plus_d,
)


@pytest.fixture
def storage_with_data():
    """Хранилище с 5 зарядами."""
    storage = NWFStorage()
    torch.manual_seed(42)
    for i in range(5):
        z = torch.randn(4)
        s = torch.abs(torch.randn(4)) + 0.1
        storage.add(Charge(z=z, sigma_sq=s), label=i % 3)
    return storage


def test_min_mahalanobis_non_negative(storage_with_data):
    qz = torch.randn(4)
    qs = torch.ones(4) * 0.5
    d = min_mahalanobis(storage_with_data, qz, qs)
    assert d >= 0
    assert d < 100  # разумный верхний предел


def test_potential_at_query_range(storage_with_data):
    """Потенциал sum(exp(-0.5*d^2)) > 0, может быть > 1."""
    qz = torch.randn(4)
    qs = torch.ones(4) * 0.5
    phi = potential_at_query(storage_with_data, qz, qs, k=5)
    assert phi > 0
    assert phi < 1e6  # разумный предел


def test_trace_sigma_positive():
    sigma_sq = torch.tensor([0.1, 0.5, 1.0, 2.0])
    tr = trace_sigma(sigma_sq)
    assert tr > 0
    assert torch.allclose(tr, torch.tensor(3.6))


def test_agreement_ratio_range(storage_with_data):
    qz = torch.randn(4)
    qs = torch.ones(4) * 0.5
    ar = agreement_ratio(storage_with_data, qz, qs, true_label=0, k=5)
    assert 0 <= ar <= 1


def test_confidence_1_over_1_plus_d():
    d = torch.tensor(0.0)
    c = confidence_1_over_1_plus_d(d)
    assert torch.allclose(c, torch.tensor(1.0))
    d = torch.tensor(1e6)
    c = confidence_1_over_1_plus_d(d)
    assert c < 1e-5
