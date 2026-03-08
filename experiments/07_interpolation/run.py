# -*- coding: utf-8 -*-
"""Семантическая интерполяция: z_alpha = (1-a)*z1 + a*z2, декодирование."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nwf import VAEEncoder, get_mnist, encode_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/vae_mnist.pt")
    parser.add_argument("--n_pairs", type=int, default=5)
    parser.add_argument("--n_alpha", type=int, default=11)
    parser.add_argument("--out", type=str, default="results")
    parser.add_argument("--save_images", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = VAEEncoder(input_dim=784, hidden_dims=(512, 256), latent_dim=ckpt["latent_dim"])
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    _, _, test_loader = get_mnist(batch_size=1, download=True)
    samples_by_class = {i: None for i in range(10)}
    for x, y in test_loader:
        if all(v is not None for v in samples_by_class.values()):
            break
        c = y.item()
        if samples_by_class[c] is None:
            samples_by_class[c] = x.view(784).to(device)

    with torch.no_grad():
        z_by_class = {}
        for c, x in samples_by_class.items():
            if x is not None:
                z, _ = encode_batch(model, x.unsqueeze(0))
                z_by_class[c] = z.squeeze(0)

    # Пары разных классов для интерполяции
    pairs = [(0, 1), (3, 5), (4, 9), (2, 8), (6, 7)]
    results = {"pairs": [], "recon_quality": []}

    for c1, c2 in pairs[: args.n_pairs]:
        if c1 not in z_by_class or c2 not in z_by_class:
            continue
        z1, z2 = z_by_class[c1], z_by_class[c2]
        alphas = torch.linspace(0, 1, args.n_alpha)
        recons = []
        for a in alphas:
            za = (1 - a) * z1 + a * z2
            recon = model.decode(za.unsqueeze(0)).squeeze(0)
            recons.append(recon.cpu())
        results["pairs"].append(f"{c1}-{c2}")

        if args.save_images:
            import torchvision
            out_dir = Path(args.out) / "interpolation"
            out_dir.mkdir(parents=True, exist_ok=True)
            grid = torch.stack(recons).view(-1, 1, 28, 28)
            torchvision.utils.save_image(grid, out_dir / f"interp_{c1}_{c2}.png", nrow=args.n_alpha)
            if (c1, c2) == (3, 5):
                Path(args.out).mkdir(parents=True, exist_ok=True)
                torchvision.utils.save_image(grid, Path(args.out) / "interpolation.png", nrow=args.n_alpha)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "07_interpolation.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Interpolation pairs: {results['pairs']}")
    print(f"Saved {out_dir / '07_interpolation.json'}")


if __name__ == "__main__":
    main()
