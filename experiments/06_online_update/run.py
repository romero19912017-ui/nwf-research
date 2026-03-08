# -*- coding: utf-8 -*-
"""Онлайн-обновление заряда: дрейф концепций (вращающаяся цифра)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF

from nwf import VAEEncoder, get_mnist
from nwf.kalman import KalmanEncoder


def rotate_sequence(x: torch.Tensor, n_frames: int) -> list[torch.Tensor]:
    """Генерация последовательности: вращение от 0 до 360 градусов."""
    seq = []
    for i in range(n_frames):
        angle = 360.0 * i / n_frames
        xi = TF.rotate(x.view(1, 1, 28, 28), angle)
        seq.append(xi.view(1, 784))
    return seq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/vae_mnist.pt")
    parser.add_argument("--n_sequences", type=int, default=50)
    parser.add_argument("--n_frames", type=int, default=10)
    parser.add_argument("--out", type=str, default="results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    vae = VAEEncoder(input_dim=784, hidden_dims=(512, 256), latent_dim=ckpt["latent_dim"])
    vae.load_state_dict(ckpt["model"])
    vae.to(device)
    vae.eval()

    kenc = KalmanEncoder(vae, r_noise=0.1, device=device)

    _, _, test_loader = get_mnist(batch_size=1, download=True)

    recon_with_update = []
    recon_without_update = []

    for idx, (x, _) in enumerate(test_loader):
        if idx >= args.n_sequences:
            break
        x = x.to(device)
        seq = rotate_sequence(x, args.n_frames)

        # С обновлением: инициализируем по первому кадру, обновляем по каждому
        c = kenc.encode_static(seq[0], n_iter=3)
        for i, xi in enumerate(seq):
            xi = xi.to(device)
            recon = vae.decode(c.z).flatten()
            err = F.binary_cross_entropy(recon, xi.flatten())
            recon_with_update.append(err.item())
            if i < args.n_frames - 1:
                c = kenc.update(c, seq[i + 1])

        # Без обновления: только первый кадр, сравниваем реконструкцию с каждым
        c_static = kenc.encode_static(seq[0], n_iter=3)
        for xi in seq:
            xi = xi.to(device)
            recon = vae.decode(c_static.z).flatten()
            err = F.binary_cross_entropy(recon, xi.flatten())
            recon_without_update.append(err.item())

    n_total = len(recon_with_update)
    avg_with = sum(recon_with_update) / n_total
    avg_without = sum(recon_without_update) / n_total

    results = {
        "recon_bce_with_update": avg_with,
        "recon_bce_without_update": avg_without,
        "improvement_ratio": avg_without / max(avg_with, 1e-8),
        "n_sequences": args.n_sequences,
        "n_frames": args.n_frames,
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "06_online_update.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Recon BCE with update:    {avg_with:.4f}")
    print(f"Recon BCE without update: {avg_without:.4f}")
    print(f"Improvement (lower is better): {(avg_without - avg_with):.4f}")
    print(f"Saved {out_dir / '06_online_update.json'}")


if __name__ == "__main__":
    main()
