# -*- coding: utf-8 -*-
"""Демонстрация семантического сжатия: 10 примеров цифры образуют один холм потенциала.

Визуализация зарядов в PCA 2D и контур поля. Показывает, что несколько похожих примеров
кластеризуются в пространстве и дают единый максимум потенциала.
"""
import argparse
import pathlib
import sys

path_root = pathlib.Path(__file__).resolve().parent.parent
if str(path_root) not in sys.path:
    sys.path.insert(0, str(path_root))

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from experiments.run_experiment import load_mnist_tfds, MNIST_MEAN, MNIST_STD
from nwf import Memory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--digit", type=int, default=3)
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--use-tfds", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output", type=str, default="results/compression_demo.png")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    print(f"Loading MNIST, digit {args.digit}...")
    X_train, y_train, _, _ = load_mnist_tfds()
    X_train_flat = X_train.reshape(-1, 784).astype(np.float32)
    idx = np.where(y_train == args.digit)[0]
    np.random.shuffle(idx)
    samples = X_train_flat[idx[: args.n_samples]]
    labels = [args.digit] * len(samples)

    device = "cpu"
    memory = Memory(device=device)
    for z, lab in zip(samples, labels):
        memory.add(z, lab, q=1.0, sigma=0.5, sigma_scale_dim=True)

    zs = memory.zs.cpu().numpy()
    pca = PCA(n_components=2)
    zs_2d = pca.fit_transform(zs)

    def field_at_2d(x, y):
        pt_784 = pca.inverse_transform([[x, y]]).ravel()
        pt = torch.tensor(pt_784, dtype=torch.float32, device=device)
        return memory.field(pt).item()

    x_min, x_max = zs_2d[:, 0].min() - 2, zs_2d[:, 0].max() + 2
    y_min, y_max = zs_2d[:, 1].min() - 2, zs_2d[:, 1].max() + 2
    xx = np.linspace(x_min, x_max, 80)
    yy = np.linspace(y_min, y_max, 80)
    Xg, Yg = np.meshgrid(xx, yy)
    F = np.zeros_like(Xg)
    for i in range(Xg.shape[0]):
        for j in range(Xg.shape[1]):
            F[i, j] = field_at_2d(Xg[i, j], Yg[i, j])

    fig, ax = plt.subplots(figsize=(9, 7))
    cont = ax.contourf(Xg, Yg, F, levels=15, cmap="viridis", alpha=0.8)
    ax.contour(Xg, Yg, F, levels=10, colors="white", alpha=0.4, linewidths=0.5)
    ax.scatter(zs_2d[:, 0], zs_2d[:, 1], c="red", s=80, marker="o", edgecolors="white", linewidths=2, label=f"{args.n_samples} charges (digit {args.digit})")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"Semantic compression: {args.n_samples} examples of '{args.digit}' form one hill in the field")
    ax.legend()
    plt.colorbar(cont, ax=ax, label="Potential")
    plt.tight_layout()
    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
