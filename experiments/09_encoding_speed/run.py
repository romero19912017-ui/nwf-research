# -*- coding: utf-8 -*-
"""Сравнение скорости кодирования: Kalman vs VAE (FAISS) vs HDC."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from nwf import VAEEncoder, get_mnist, encode_batch
from nwf.kalman import KalmanEncoder
from nwf.baselines import HDCEncoder


def recon_mse(model, x):
    with torch.no_grad():
        mu, _ = encode_batch(model, x)
        recon = model.decode(mu)
        return float((recon - x).pow(2).mean().item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/vae_mnist.pt")
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--out", type=str, default="results")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = VAEEncoder(input_dim=784, hidden_dims=(512, 256), latent_dim=ckpt["latent_dim"])
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    _, _, test_loader = get_mnist(batch_size=args.n_samples, download=True)
    x, _ = next(iter(test_loader))
    x = x.view(-1, 784).to(device)

    results = {}

    # VAE (FAISS-style)
    t0 = time.perf_counter()
    for _ in range(3):
        with torch.no_grad():
            encode_batch(model, x)
    t_vae = (time.perf_counter() - t0) / 3
    mse_vae = recon_mse(model, x)
    results["VAE_single_pass"] = {"sec_per_batch": t_vae, "sec_per_sample": t_vae / x.size(0), "recon_mse": mse_vae}

    # Kalman: 1, 3, 5, 10 iterations
    kenc = KalmanEncoder(model, r_noise=0.1, device=device)
    x_sub = x[:50]
    kalman_data = []
    for n_iter in [1, 3, 5, 10]:
        t0 = time.perf_counter()
        z, sigma = kenc.encode_static_batch(x_sub, n_iter=n_iter)
        t_elapsed = time.perf_counter() - t0
        with torch.no_grad():
            recon = model.decode(z)
            mse = float((recon - x_sub).pow(2).mean().item())
        results[f"Kalman_{n_iter}iter"] = {"sec_per_50": t_elapsed, "sec_per_sample": t_elapsed / 50, "recon_mse": mse}
        kalman_data.append((n_iter, t_elapsed / 50, mse))

    # HDC
    hdc = HDCEncoder(dim=2000, seed=42)
    t0 = time.perf_counter()
    for _ in range(3):
        hdc.encode_torch(x)
    t_hdc = (time.perf_counter() - t0) / 3
    results["HDC"] = {"sec_per_batch": t_hdc, "sec_per_sample": t_hdc / x.size(0)}

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "09_encoding_speed.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            iters = [d[0] for d in kalman_data]
            times = [d[1] for d in kalman_data]
            mses = [d[2] for d in kalman_data]
            ax1.plot(iters, times, "o-", label="Kalman")
            ax1.axhline(t_vae / x.size(0), color="green", linestyle="--", label="VAE")
            ax1.axhline(t_hdc / x.size(0), color="orange", linestyle="--", label="HDC")
            ax1.set_xlabel("Kalman iterations")
            ax1.set_ylabel("sec per sample")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax2.plot(iters, mses, "o-", label="Kalman")
            ax2.axhline(mse_vae, color="green", linestyle="--", label="VAE")
            ax2.set_xlabel("Kalman iterations")
            ax2.set_ylabel("Reconstruction MSE")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_dir / "09_encoding_speed.png", dpi=120)
            plt.close()
            print(f"Plot saved to {out_dir / '09_encoding_speed.png'}")
        except Exception as e:
            print(f"Plot skipped: {e}")

    print("Encoding speed (sec per sample):")
    for k, v in results.items():
        sps = v.get("sec_per_sample", v.get("sec_per_50", 0) / 50)
        rm = v.get("recon_mse", "-")
        print(f"  {k}: {sps:.6f} sec/sample" + (f", recon_mse={rm:.6f}" if isinstance(rm, float) else ""))
    print(f"Saved {out_dir / '09_encoding_speed.json'}")


if __name__ == "__main__":
    main()
