# -*- coding: utf-8 -*-
"""Трассировка луча в поле потенциала NWF.

Точка r0 «поднимается» по градиенту потенциала к ближайшему максимуму (скоплению зарядов).
Это приближает тестовый пример к «своему» классу перед классификацией.
"""
import torch

from nwf.field import field_and_grad


def trace_ray_memory(r0, memory, num_steps=20, step_size=0.1, k_neighbors=100):
    """Трассировка из r0 по хранилищу Memory. Возвращает (r_final, траектория)."""
    device = memory.device
    r = r0.clone().detach().to(device)
    if r.dim() == 1:
        r = r.unsqueeze(0).squeeze(0)
    trajectory = [r.detach().cpu().numpy().copy()]
    n_total = memory.get_ntotal()
    if n_total == 0:
        return r.detach(), trajectory
    k = min(k_neighbors, n_total)

    for _ in range(num_steps - 1):
        if k < n_total:
            dists = torch.cdist(r.unsqueeze(0), memory.zs).squeeze(0)
            _, idx = torch.topk(dists, k, largest=False)
            sub = memory.subset(idx)
            grad = sub.field_grad(r)
        else:
            grad = memory.field_grad(r)
        grad_norm = torch.norm(grad) + 1e-8
        r = r.detach() + step_size * grad / grad_norm
        trajectory.append(r.cpu().numpy().copy())

    return r.detach(), trajectory


def trace_ray(
    r0,
    zs,
    qs,
    sigmas,
    num_steps=20,
    step_size=0.1,
    k_neighbors=100,
    device=None,
):
    """Трассировка из r0 по градиенту потенциала. Возвращает (r_final, траектория)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(r0, torch.Tensor):
        r0 = torch.tensor(r0, dtype=torch.float32, device=device)
    else:
        r0 = r0.to(device)

    zs = zs.to(device)
    qs = qs.to(device)
    sigmas = sigmas.to(device)

    n_total = zs.shape[0]
    if n_total == 0:
        return r0.detach(), [r0.detach().cpu().numpy()]

    k = min(k_neighbors, n_total)
    trajectory = [r0.detach().cpu().numpy().copy()]

    r = r0.clone().detach().requires_grad_(True)

    for _ in range(num_steps - 1):
        with torch.enable_grad():
            if k < n_total:
                dists = torch.cdist(r.unsqueeze(0), zs).squeeze(0)
                _, idx = torch.topk(dists, k, largest=False)
                z_nearest = zs[idx]
                q_nearest = qs[idx]
                sigma_nearest = sigmas[idx]
            else:
                z_nearest = zs
                q_nearest = qs
                sigma_nearest = sigmas

            _, grad = field_and_grad(r, z_nearest, q_nearest, sigma_nearest)
            grad_norm = torch.norm(grad) + 1e-8
            r_new = (r.detach() + step_size * grad.detach() / grad_norm).requires_grad_(True)
            r = r_new

        trajectory.append(r.detach().cpu().numpy().copy())

    return r.detach(), trajectory
