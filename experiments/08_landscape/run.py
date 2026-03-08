# -*- coding: utf-8 -*-
"""Визуализация семантического ландшафта: t-SNE/UMAP + контуры потенциала."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from nwf import VAEEncoder, NWFStorage, get_mnist, encode_batch


def potential_on_grid(embeds_2d, grid_res=50):
    """Потенциал на 2D сетке: sum_i exp(-0.5 * ||r - z_i||^2) в 2D проекции."""
    x_min, x_max = embeds_2d[:, 0].min() - 0.5, embeds_2d[:, 0].max() + 0.5
    y_min, y_max = embeds_2d[:, 1].min() - 0.5, embeds_2d[:, 1].max() + 0.5
    xx = np.linspace(x_min, x_max, grid_res)
    yy = np.linspace(y_min, y_max, grid_res)
    X, Y = np.meshgrid(xx, yy)
    grid_pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    phi = np.zeros(len(grid_pts))
    for i, pt in enumerate(grid_pts):
        d_sq = ((embeds_2d - pt) ** 2).sum(axis=1)
        phi[i] = np.exp(-0.5 * d_sq).sum()
    return X, Y, phi.reshape(X.shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/vae_mnist.pt")
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--out", type=str, default="results")
    parser.add_argument("--method", type=str, default="tsne", choices=["tsne", "pca"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = VAEEncoder(input_dim=784, hidden_dims=(512, 256), latent_dim=ckpt["latent_dim"])
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    _, _, test_loader = get_mnist(batch_size=128, download=True)
    z_list, y_list = [], []
    n = 0
    with torch.no_grad():
        for x, y in test_loader:
            if n >= args.max_samples:
                break
            x = x.view(-1, 784).to(device)
            z, _ = encode_batch(model, x)
            z_list.append(z.cpu().numpy())
            y_list.append(y.numpy())
            n += x.size(0)
    Z = np.concatenate(z_list, axis=0)[: args.max_samples]
    Y = np.concatenate(y_list, axis=0)[: args.max_samples]

    if args.method == "pca":
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        embeds_2d = pca.fit_transform(Z)
    else:
        try:
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            embeds_2d = tsne.fit_transform(Z)
        except ImportError:
            from sklearn.decomposition import PCA
            embeds_2d = PCA(n_components=2).fit_transform(Z)

    X, Yg, phi = potential_on_grid(embeds_2d, grid_res=40)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "08_landscape.npz", embeds_2d=embeds_2d, labels=Y, phi=phi, X=X, Y=Yg)

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.contourf(X, Yg, phi, levels=15, cmap="viridis", alpha=0.6)
        scatter = ax.scatter(embeds_2d[:, 0], embeds_2d[:, 1], c=Y, cmap="tab10", s=5, alpha=0.7)
        ax.set_title("Semantic landscape: potential Phi(r)")
        plt.colorbar(scatter, ax=ax, label="Class")
        plt.tight_layout()
        plt.savefig(out_dir / "08_landscape.png", dpi=120)
        plt.close()
        print(f"Plot saved to {out_dir / '08_landscape.png'}")
    except Exception as e:
        print(f"Plot skipped: {e}")

    with open(out_dir / "08_landscape.json", "w", encoding="utf-8") as f:
        json.dump({"n_samples": len(Y), "method": args.method}, f, indent=2)
    print(f"Saved {out_dir / '08_landscape.npz'}")


if __name__ == "__main__":
    main()
