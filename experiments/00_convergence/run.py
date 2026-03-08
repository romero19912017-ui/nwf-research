# -*- coding: utf-8 -*-
"""Сравнение сходимости: Kalman vs градиентный спуск по z."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from nwf import VAEEncoder, get_mnist
from nwf.kalman import KalmanEncoder
from nwf.inference import encode_batch


def gd_encode(vae, x, n_iter, lr=0.1):
    """Градиентный спуск: минимизация BCE(recon, x) по z."""
    with torch.no_grad():
        z0, _ = encode_batch(vae, x)
    z = z0.clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    errors = []
    for _ in range(n_iter):
        opt.zero_grad()
        recon = vae.decode(z)
        loss = F.binary_cross_entropy(recon, x)
        loss.backward()
        opt.step()
        errors.append(loss.item())
    return z.detach(), errors


def kalman_encode(kenc, x, n_iter):
    """Kalman-кодирование с отслеживанием ошибки реконструкции."""
    c = kenc.encode_static(x, n_iter=n_iter)
    errors = []
    z, P = kenc._init_charge(x)
    x_flat = x.flatten()
    for _ in range(n_iter):
        recon = kenc.vae.decode(z).flatten()
        err = F.binary_cross_entropy(recon, x_flat)
        errors.append(err.item())
        z, P = kenc._ekf_step(z, P, x)
    return c.z, errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/vae_mnist.pt")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--n_iter", type=int, default=10)
    parser.add_argument("--out", type=str, default="results")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    vae = VAEEncoder(input_dim=784, hidden_dims=(512, 256), latent_dim=ckpt["latent_dim"])
    vae.load_state_dict(ckpt["model"])
    vae.to(device)
    vae.eval()

    kenc = KalmanEncoder(vae, r_noise=0.1, device=device)

    _, _, test_loader = get_mnist(batch_size=1, download=True)
    gd_errors_per_iter = [0.0] * args.n_iter
    kalman_errors_per_iter = [0.0] * args.n_iter

    for i, (x, _) in enumerate(test_loader):
        if i >= args.n_samples:
            break
        x = x.view(1, 784).to(device)
        _, gd_err = gd_encode(vae, x, args.n_iter)
        _, kal_err = kalman_encode(kenc, x, args.n_iter)
        for j in range(args.n_iter):
            gd_errors_per_iter[j] += gd_err[j]
            kalman_errors_per_iter[j] += kal_err[j]

    n = min(args.n_samples, i + 1)
    for j in range(args.n_iter):
        gd_errors_per_iter[j] /= n
        kalman_errors_per_iter[j] /= n

    results = {
        "gd_recon_error": gd_errors_per_iter,
        "kalman_recon_error": kalman_errors_per_iter,
        "n_samples": n,
        "n_iter": args.n_iter,
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "00_convergence.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Iter  GD_err  Kalman_err")
    for j in range(args.n_iter):
        print(f"  {j+1}   {gd_errors_per_iter[j]:.4f}   {kalman_errors_per_iter[j]:.4f}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(range(1, args.n_iter + 1), gd_errors_per_iter, "o-", label="GD (Adam)")
            ax.plot(range(1, args.n_iter + 1), kalman_errors_per_iter, "s-", label="Kalman EKF")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Reconstruction BCE")
            ax.set_title("Convergence: Kalman vs Gradient Descent")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_dir / "00_convergence.png", dpi=120)
            plt.close()
            print(f"Plot saved to {out_dir / '00_convergence.png'}")
        except Exception as e:
            print(f"Plot skipped: {e}")

    print(f"Saved {out_dir / '00_convergence.json'}")


if __name__ == "__main__":
    main()
