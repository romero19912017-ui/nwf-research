# -*- coding: utf-8 -*-
"""Хранилище зарядов (Memory): заряды с индивидуальными ковариациями.

Хранит позиции (zs), метки классов (labels), интенсивности (qs) и диагонали ковариаций (sigmas).
Поддерживает потенциал, градиент поля, поиск ближайших соседей (L2).
sigma_scale_dim: масштабирование sigma на sqrt(dim) для высокоразмерных пространств.
"""
import torch


class Memory:
    """Хранилище зарядов с индивидуальными диагональными ковариациями."""

    def __init__(self, device=None):
        self.zs = None
        self.qs = None
        self.sigmas = None
        self.labels = None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, z, label, q=1.0, sigma=0.5, sigma_scale_dim=True):
        """Добавить один заряд. sigma_scale_dim: масштабировать sigma на sqrt(dim)."""
        z = torch.as_tensor(z, dtype=torch.float32, device=self.device)
        if z.dim() == 1:
            z = z.unsqueeze(0)
        dim = z.shape[-1]
        sig_val = sigma * (dim**0.5) if sigma_scale_dim else sigma
        q_val = torch.tensor([q], dtype=torch.float32, device=self.device)
        sig = torch.full((1, dim), sig_val, dtype=torch.float32, device=self.device)
        lab = torch.tensor([label], dtype=torch.long, device=self.device)

        if self.zs is None:
            self.zs = z
            self.qs = q_val
            self.sigmas = sig
            self.labels = lab
        else:
            self.zs = torch.cat([self.zs, z], dim=0)
            self.qs = torch.cat([self.qs, q_val], dim=0)
            self.sigmas = torch.cat([self.sigmas, sig], dim=0)
            self.labels = torch.cat([self.labels, lab], dim=0)

    def add_batch(self, zs, labels, q=1.0, sigma=0.5, sigma_scale_dim=True):
        """Добавить пакет зарядов. zs: (N, dim), labels: (N,)."""
        zs = torch.as_tensor(zs, dtype=torch.float32, device=self.device)
        labels = torch.as_tensor(labels, dtype=torch.long, device=self.device)
        qs = torch.full((zs.shape[0],), q, dtype=torch.float32, device=self.device)
        sig_val = sigma * (zs.shape[-1]**0.5) if sigma_scale_dim else sigma
        sigmas = torch.full_like(zs, sig_val)

        if self.zs is None:
            self.zs = zs
            self.qs = qs
            self.sigmas = sigmas
            self.labels = labels
        else:
            self.zs = torch.cat([self.zs, zs], dim=0)
            self.qs = torch.cat([self.qs, qs], dim=0)
            self.sigmas = torch.cat([self.sigmas, sigmas], dim=0)
            self.labels = torch.cat([self.labels, labels], dim=0)

    def potentials(self, r):
        """Потенциалы от всех зарядов в точке r. Возвращает тензор (N,)."""
        if r.dim() == 1:
            r = r.unsqueeze(0)
        r = r.to(self.device)
        diff = self.zs - r  # (N, dim)
        dist_sq = torch.sum((diff**2) / (self.sigmas**2 + 1e-8), dim=1)  # (N,) Mahalanobis
        return self.qs * torch.exp(-0.5 * dist_sq)

    def field(self, r):
        """Суммарный потенциал в точке r."""
        return self.potentials(r).sum()

    def field_grad(self, r):
        """Градиент суммарного потенциала по r."""
        r = r.clone().detach().to(self.device).requires_grad_(True)
        f = self.field(r)
        grad = torch.autograd.grad(f, r, create_graph=False)[0]
        return grad.detach()

    def subset(self, indices):
        """Вернуть новое хранилище с подмножеством зарядов по индексам."""
        m = Memory(device=self.device)
        m.zs = self.zs[indices].clone()
        m.qs = self.qs[indices].clone()
        m.sigmas = self.sigmas[indices].clone()
        m.labels = self.labels[indices].clone()
        return m

    def search(self, query, k=1):
        """Поиск k ближайших соседей (L2). Возвращает (индексы, расстояния)."""
        if query.dim() == 1:
            query = query.unsqueeze(0)
        query = query.to(self.device)
        dists = torch.cdist(query, self.zs).squeeze(0)
        values, indices = torch.topk(dists, min(k, len(self.zs)), largest=False)
        return indices.cpu().numpy(), values.cpu().numpy()

    def get_ntotal(self):
        return 0 if self.zs is None else self.zs.shape[0]
