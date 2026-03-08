# -*- coding: utf-8 -*-
"""Поиск по сетке гиперпараметров NWF: sigma, шаг трассировки, число шагов.

Подбор оптимальных значений для заданного K. Результаты в CSV.
"""
import argparse
import csv
import pathlib
import sys

path_root = pathlib.Path(__file__).resolve().parent.parent
if str(path_root) not in sys.path:
    sys.path.insert(0, str(path_root))

import numpy as np
import torch

from experiments.run_experiment import (
    load_mnist_tfds,
    get_support_set_from_arrays,
)
from nwf import Memory, trace_ray


def run_nwf_accuracy(X_support, y_support, X_test, y_test, device, sigma, step_size, num_steps, k_neighbors=100):
    """Run NWF and return accuracy."""
    memory = Memory(device=str(device))
    for emb, label in zip(X_support, y_support):
        memory.add(emb, int(label), q=1.0, sigma=sigma)
    k_neigh = min(k_neighbors, memory.get_ntotal())
    pred = []
    for emb in X_test:
        r0 = torch.tensor(emb, dtype=torch.float32, device=device)
        r_final, _ = trace_ray(r0, memory.zs, memory.qs, memory.sigmas,
                               num_steps=num_steps, step_size=step_size,
                               k_neighbors=k_neigh, device=device)
        ind, _ = memory.search(r_final, k=1)
        pred.append(memory.labels[ind[0]].item())
    return np.mean(np.array(pred) == y_test)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--test-samples", type=int, default=300)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output", type=str, default="results/grid_search.csv")
    args = parser.parse_args()

    if args.cpu:
        device = torch.device("cpu")
    else:
        try:
            _ = torch.zeros(1, device="cuda")
            device = torch.device("cuda")
        except RuntimeError:
            device = torch.device("cpu")
    print(f"Device: {device}")

    print("Loading MNIST...")
    X_train, y_train, X_test, y_test = load_mnist_tfds()
    X_train = X_train.reshape(-1, 784)
    X_test = X_test.reshape(-1, 784)

    train_indices = get_support_set_from_arrays(X_train, y_train, args.k)
    np.random.shuffle(train_indices)
    n_test = min(args.test_samples, len(X_test))
    test_indices = np.random.choice(len(X_test), n_test, replace=False)

    X_support = X_train[train_indices]
    y_support = y_train[train_indices]
    X_te = X_test[test_indices]
    y_te = y_test[test_indices]

    # Reduced grid for quick search; use full grid for thorough tuning
    grid = {
        "sigma": [0.3, 0.5, 0.7, 1.0],
        "step_size": [0.1, 0.2, 0.5],
        "num_steps": [20, 30, 50],
    }

    results = []
    total = len(grid["sigma"]) * len(grid["step_size"]) * len(grid["num_steps"])
    idx = 0
    for sigma in grid["sigma"]:
        for step_size in grid["step_size"]:
            for num_steps in grid["num_steps"]:
                idx += 1
                print(f"[{idx}/{total}] sigma={sigma} step={step_size} steps={num_steps}...", end=" ", flush=True)
                acc = run_nwf_accuracy(X_support, y_support, X_te, y_te, device, sigma, step_size, num_steps)
                results.append({"sigma": sigma, "step_size": step_size, "num_steps": num_steps, "accuracy": acc})
                print(f"acc={acc:.4f}")

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sigma", "step_size", "num_steps", "accuracy"])
        w.writeheader()
        w.writerows(results)

    best = max(results, key=lambda x: x["accuracy"])
    print(f"\nBest: sigma={best['sigma']} step_size={best['step_size']} num_steps={best['num_steps']} -> {best['accuracy']:.4f}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
