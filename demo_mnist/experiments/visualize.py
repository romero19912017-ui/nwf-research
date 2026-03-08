# -*- coding: utf-8 -*-
"""Визуализация зарядов и траекторий NWF в PCA 2D.

Требует experiment_results.npz (создаётся run_experiment.py при сохранении).
"""
import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def plot_trajectories(results_path: pathlib.Path, output_path: pathlib.Path, n_samples: int = 10):
    """Plot PCA 2D: charges + trajectories for sample test points."""
    data = np.load(results_path)
    X_support = data["X_support_emb"]
    y_support = data["y_support"]
    X_test = data["X_test_emb"]

    # PCA to 2D
    pca = PCA(n_components=2)
    all_emb = np.vstack([X_support, X_test[: min(n_samples * 2, len(X_test))]])
    pca.fit(all_emb)
    support_2d = pca.transform(X_support)
    test_2d = pca.transform(X_test[:n_samples])

    fig, ax = plt.subplots(figsize=(10, 8))
    for c in range(10):
        mask = y_support == c
        ax.scatter(
            support_2d[mask, 0], support_2d[mask, 1],
            label=str(c), alpha=0.6, s=30,
        )
    ax.scatter(test_2d[:, 0], test_2d[:, 1], c="black", marker="x", s=80, label="test (r0)")
    ax.legend(bbox_to_anchor=(1.02, 1))
    ax.set_title("NWF: Charges (support) in PCA 2D")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--n-samples", type=int, default=10)
    args = parser.parse_args()

    results_dir = pathlib.Path(args.results_dir)
    results_path = results_dir / "experiment_results.npz"
    if not results_path.exists():
        print(f"Run experiment first. Expected {results_path}")
        return

    results_dir.mkdir(parents=True, exist_ok=True)
    plot_trajectories(results_path, results_dir / "trajectories_pca.png", args.n_samples)


if __name__ == "__main__":
    main()
