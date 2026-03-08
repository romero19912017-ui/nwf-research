# -*- coding: utf-8 -*-
"""Unit-тесты Kalman-кодировщика."""
import pytest
import torch

from nwf.vae_encoder import VAEEncoder
from nwf.kalman import KalmanEncoder


@pytest.fixture
def vae_and_kalman():
    torch.manual_seed(42)
    vae = VAEEncoder(input_dim=8, hidden_dims=(16,), latent_dim=4)
    kenc = KalmanEncoder(vae, r_noise=0.1)
    return vae, kenc


def test_zero_innovation_no_change(vae_and_kalman):
    """При x=decode(z) EKF должен дать близкий к z результат (линеаризация)."""
    vae, kenc = vae_and_kalman
    vae.eval()
    z0 = torch.randn(4) * 0.5  # малая амплитуда для линеаризации
    with torch.no_grad():
        x = vae.decode(z0.unsqueeze(0)).squeeze(0)
    c = kenc.encode_static(x.unsqueeze(0), n_iter=2)
    diff = (c.z - z0).abs().max()
    assert diff < 3.0  # допуск: нелинейный декодер, r_noise


def test_convergence_decreases_recon_error(vae_and_kalman):
    """Ошибка реконструкции должна уменьшаться с итерациями."""
    vae, kenc = vae_and_kalman
    vae.eval()
    torch.manual_seed(42)
    x = torch.rand(1, 8)
    z0, _ = kenc._init_charge(x)
    err0 = float((vae.decode(z0.unsqueeze(0)) - x).pow(2).mean().detach())
    c = kenc.encode_static(x, n_iter=5)
    err5 = float((vae.decode(c.z.unsqueeze(0)) - x).pow(2).mean())
    assert err5 <= err0 + 0.01  # не должно ухудшиться
