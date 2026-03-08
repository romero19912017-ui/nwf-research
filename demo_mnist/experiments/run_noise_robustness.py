# -*- coding: utf-8 -*-
"""Устойчивость к шуму (умеренный): точность vs уровень шума.

Гауссов шум добавляется к тестовым изображениям. Сравнение NWF, 1-NN, LogReg, MLP.
"""
import argparse
import pathlib
import sys

path_root = pathlib.Path(__file__).resolve().parent.parent
if str(path_root) not in sys.path:
    sys.path.insert(0, str(path_root))

import numpy as np
import matplotlib.pyplot as plt
from experiments.run_experiment import main as run_main
import subprocess


def run_with_noise(noise_std, k=10, test_samples=500, **kwargs):
    """Run experiment and return accuracies dict."""
    import torch
    from experiments.run_experiment import (
        load_mnist_tfds, get_support_set_from_arrays,
        MNIST_MEAN, MNIST_STD,
    )
    from nwf import Encoder, Memory, trace_ray_memory, classify_weighted
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.neural_network import MLPClassifier

    device = torch.device("cpu")
    X_train, y_train, X_test, y_test = load_mnist_tfds()
    X_train_flat = X_train.reshape(-1, 784)
    X_test_flat = X_test.reshape(-1, 784)

    train_indices = get_support_set_from_arrays(X_train_flat, y_train, k)
    np.random.shuffle(train_indices)
    n_test = min(test_samples, len(X_test))
    test_indices = np.random.choice(len(X_test), n_test, replace=False)

    encoder_path = path_root / "encoder_mnist.pth"
    encoder = None
    if encoder_path.exists():
        encoder = Encoder(output_dim=64).to(device)
        encoder.load_state_dict(torch.load(encoder_path, map_location=device))
        X_train_4d = torch.tensor(X_train.reshape(-1, 1, 28, 28), dtype=torch.float32)
        X_test_4d = torch.tensor(X_test.reshape(-1, 1, 28, 28), dtype=torch.float32)
        with torch.no_grad():
            train_embs = encoder(X_train_4d[train_indices].to(device)).cpu().numpy()
        X_test_raw = X_test * MNIST_STD + MNIST_MEAN
        X_test_noisy = X_test_raw + np.random.RandomState(42).randn(*X_test.shape).astype(np.float32) * noise_std
        X_test_noisy = np.clip(X_test_noisy, 0, 1)
        X_test_noisy = (X_test_noisy - MNIST_MEAN) / MNIST_STD
        X_test_4d_noisy = torch.tensor(X_test_noisy.reshape(-1, 1, 28, 28), dtype=torch.float32)
        with torch.no_grad():
            test_embs = encoder(X_test_4d_noisy[test_indices].to(device)).cpu().numpy()
        embed_dim = 64
    else:
        X_test_raw = X_test * MNIST_STD + MNIST_MEAN
        X_test_noisy = X_test_raw + np.random.RandomState(42).randn(*X_test.shape).astype(np.float32) * noise_std
        X_test_noisy = np.clip(X_test_noisy, 0, 1)
        X_test_noisy = (X_test_noisy - MNIST_MEAN) / MNIST_STD
        X_test_flat_noisy = X_test_noisy.reshape(-1, 784)
        train_embs = X_train_flat[train_indices]
        test_embs = X_test_flat_noisy[test_indices]
        embed_dim = 784

    train_labels = y_train[train_indices]
    test_labels = y_test[test_indices]

    memory = Memory(device="cpu")
    for emb, label in zip(train_embs, train_labels):
        memory.add(emb, int(label), q=1.0, sigma=0.5, sigma_scale_dim=True)

    k_neigh = min(100, memory.get_ntotal())

    pred_nwf = []
    for emb in test_embs:
        r0 = torch.tensor(emb, dtype=torch.float32, device=device)
        r_final, _ = trace_ray_memory(r0, memory, num_steps=20, step_size=0.1, k_neighbors=k_neigh)
        pred = classify_weighted(r_final, memory, k=10, temperature=1.0)
        pred_nwf.append(pred)
    acc_nwf = accuracy_score(test_labels, pred_nwf)

    pred_1nn = []
    for emb in test_embs:
        r0 = torch.tensor(emb, dtype=torch.float32, device=device)
        ind, _ = memory.search(r0, k=1)
        pred_1nn.append(memory.labels[ind[0]].item())
    acc_1nn = accuracy_score(test_labels, pred_1nn)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(train_embs, train_labels)
    pred_lr = clf.predict(test_embs)
    acc_lr = accuracy_score(test_labels, pred_lr)

    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500)
    mlp.fit(train_embs, train_labels)
    pred_mlp = mlp.predict(test_embs)
    acc_mlp = accuracy_score(test_labels, pred_mlp)

    return {"NWF": acc_nwf, "1-NN": acc_1nn, "LogReg": acc_lr, "MLP": acc_mlp}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--test-samples", type=int, default=500)
    parser.add_argument("--noise-levels", type=str, default="0,0.1,0.2,0.3")
    parser.add_argument("--output", type=str, default="results/noise_robustness.png")
    parser.add_argument("--csv", type=str, default="results/noise_robustness.csv")
    args = parser.parse_args()

    noise_levels = [float(x) for x in args.noise_levels.split(",")]
    results = {"noise": noise_levels, "NWF": [], "1-NN": [], "LogReg": [], "MLP": []}

    for std in noise_levels:
        print(f"Noise std={std}...")
        accs = run_with_noise(std, k=args.k, test_samples=args.test_samples)
        for k in results:
            if k != "noise":
                results[k].append(accs[k])
        print(f"  NWF={accs['NWF']:.4f} 1-NN={accs['1-NN']:.4f} LogReg={accs['LogReg']:.4f} MLP={accs['MLP']:.4f}")

    out_dir = pathlib.Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    for method in ["NWF", "1-NN", "LogReg", "MLP"]:
        plt.plot(noise_levels, results[method], "o-", label=method)
    plt.xlabel("Noise std")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Noise robustness (NWF vs baselines)")
    plt.grid(True, alpha=0.3)
    plt.savefig(args.output, dpi=150)
    plt.close()
    print(f"Saved {args.output}")

    import csv
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["noise_std", "NWF", "1-NN", "LogReg", "MLP"])
        for i, std in enumerate(noise_levels):
            w.writerow([std] + [f"{results[m][i]:.4f}" for m in ["NWF", "1-NN", "LogReg", "MLP"]])
    print(f"Saved {args.csv}")


if __name__ == "__main__":
    main()
