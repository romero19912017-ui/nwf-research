# -*- coding: utf-8 -*-
"""Kalman-кодировщик: рекурсивное уточнение (z, Sigma) по наблюдению.

Аксиома А5 теории NWF: семантика z и неопределённость Sigma эволюционируют
во времени согласно фильтру Калмана. Вместо одного прохода через VAE
мы итеративно обновляем (z, Sigma), минимизируя «расхождение» между
реконструкцией decode(z) и наблюдением x.

Используется Extended Kalman Filter (EKF): линеаризация декодера через якобиан.
Применения: более точное кодирование, онлайн-обновление при дрейфе (новые кадры).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from .core import Charge
from .vae_encoder import VAEEncoder


class KalmanEncoder:
    """Расширенный фильтр Калмана (EKF) для уточнения заряда по наблюдению.

    Инициализация: VAE encoder даёт (z0, P0). Далее итерации EKF:
    innovation = x - decode(z), корректировка z и P через gain K.
    Даёт лучшую реконструкцию, чем один проход VAE, но медленнее (якобиан).
    """

    def __init__(
        self,
        vae: VAEEncoder,
        r_noise: float = 0.1,
        q_noise: float = 0.0,
        device: Optional[torch.device] = None,
    ):
        self.vae = vae
        self.r_noise = r_noise
        self.q_noise = q_noise
        self.device = device or next(vae.parameters()).device
        self.latent_dim = vae.latent_dim

    def _init_charge(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Начальная оценка из VAE: z0, P0 (диагональ)."""
        with torch.no_grad():
            mu, log_var = self.vae.encode(x)
            z = mu.squeeze(0).to(self.device)
            sigma_sq = log_var.exp().clamp(min=1e-6).squeeze(0).to(self.device)
        return z, sigma_sq

    def _ekf_step(
        self,
        z: torch.Tensor,
        P_diag: torch.Tensor,
        x_obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Один шаг EKF: innovation, gain K, обновление z и P.

        H = якобиан decode(z). Innovation = x - decode(z).
        z_new = z + K @ innovation, P_new = (I - K@H) @ P @ (I - K@H)^T
        """
        x_flat = x_obs.flatten().to(self.device)
        z = z.to(self.device)
        P_diag = P_diag.clamp(min=1e-6).to(self.device)

        z_j = z.detach().requires_grad_(True)
        y_pred = self.vae.decode(z_j).flatten()
        H = torch.autograd.functional.jacobian(
            lambda zz: self.vae.decode(zz).flatten(),
            z_j,
            create_graph=False,
        )
        H = H.squeeze(0)  # (784, D)

        innovation = x_flat - y_pred.detach()
        P_mat = torch.diag(P_diag)
        R = self.r_noise * torch.eye(H.size(0), device=self.device)
        S = H @ P_mat @ H.T + R + 1e-2 * torch.eye(H.size(0), device=self.device)

        # K = P @ H.T @ S^{-1} -> solve S.T @ K.T = H @ P, so K.T = solve(S.T, H @ P)
        K = torch.linalg.solve(S.T, H @ P_mat).T  # (D, 784)
        z_new = z + K @ innovation

        I_KH = torch.eye(self.latent_dim, device=self.device) - K @ H
        P_new = I_KH @ P_mat @ I_KH.T
        P_new = (P_new + P_new.T) / 2
        P_new_diag = torch.diag(P_new).clamp(min=1e-6)

        return z_new.detach(), P_new_diag.detach()

    def encode_static(self, x: torch.Tensor, n_iter: int = 5) -> Charge:
        """Статическое кодирование: n_iter шагов EKF от начального (z0, P0) из VAE."""
        x = x.flatten().unsqueeze(0).to(self.device)  # (1, 784)
        z, P_diag = self._init_charge(x)
        x_obs = x
        for _ in range(n_iter - 1):
            z, P_diag = self._ekf_step(z, P_diag, x_obs)
        return Charge(z=z, sigma_sq=P_diag)

    def encode_static_batch(
        self,
        x: torch.Tensor,
        n_iter: int = 5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Батч статического кодирования. x: (N, 784) или (N,1,28,28)."""
        x = x.view(-1, 784).to(self.device)
        z_list, s_list = [], []
        for i in range(x.size(0)):
            c = self.encode_static(x[i], n_iter=n_iter)
            z_list.append(c.z)
            s_list.append(c.sigma_sq)
        return torch.stack(z_list, 0), torch.stack(s_list, 0)

    def update(self, charge: Charge, x: torch.Tensor) -> Charge:
        """Онлайн-обновление: один шаг EKF от текущего заряда.

        Используется при дрейфе (например, вращающаяся цифра — новые кадры
        уточняют представление того же объекта).
        """
        x = x.flatten().unsqueeze(0).to(self.device)
        z_new, P_new = self._ekf_step(charge.z, charge.sigma_sq, x)
        return Charge(z=z_new, sigma_sq=P_new)
