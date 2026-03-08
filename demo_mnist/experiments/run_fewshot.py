# -*- coding: utf-8 -*-
"""Обучение с малым числом примеров: точность vs K для NWF, 1-NN, LogReg, MLP.

Использует эмбеддинги (предобученный энкодер). Несколько запусков с разным seed, усреднение.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

from experiments.run_experiment import load_mnist_tfds, get_support_set_from_arrays, MNIST_MEAN, MNIST_STD
from nwf import Encoder, Memory, trace_ray_memory, classify_weighted


def run_one(K, run_seed, device, encoder_path, test_samples=1000):
    np.random.seed(run_seed)
    X_train, y_train, X_test, y_test = load_mnist_tfds()
    X_train_flat = X_train.reshape(-1, 784)
    X_test_flat = X_test.reshape(-1, 784)

    train_indices = get_support_set_from_arrays(X_train_flat, y_train, K)
    np.random.shuffle(train_indices)
    n_test = min(test_samples, len(X_test))
    test_indices = np.random.choice(len(X_test), n_test, replace=False)

    if encoder_path.exists():
        encoder = Encoder(output_dim=64).to(device)
        encoder.load_state_dict(torch.load(encoder_path, map_location=device))
        X_train_4d = torch.tensor(X_train.reshape(-1, 1, 28, 28), dtype=torch.float32)
        X_test_4d = torch.tensor(X_test.reshape(-1, 1, 28, 28), dtype=torch.float32)
        with torch.no_grad():
            train_embs = encoder(X_train_4d[train_indices].to(device)).cpu().numpy()
            test_embs = encoder(X_test_4d[test_indices].to(device)).cpu().numpy()
    else:
        train_embs = X_train_flat[train_indices]
        test_embs = X_test_flat[test_indices]

    train_labels = y_train[train_indices]
    test_labels = y_test[test_indices]

    memory = Memory(device=str(device))
    for emb, label in zip(train_embs, train_labels):
        memory.add(emb, int(label), q=1.0, sigma=0.5, sigma_scale_dim=True)

    k_neigh = min(100, memory.get_ntotal())

    pred_nwf = []
    for emb in test_embs:
        r0 = torch.tensor(emb, dtype=torch.float32, device=device)
        r_final, _ = trace_ray_memory(r0, memory, num_steps=20, step_size=0.1, k_neighbors=k_neigh)
        pred_nwf.append(classify_weighted(r_final, memory, k=10, temperature=1.0))
    acc_nwf = accuracy_score(test_labels, pred_nwf)

    pred_1nn = []
    for emb in test_embs:
        r0 = torch.tensor(emb, dtype=torch.float32, device=device)
        ind, _ = memory.search(r0, k=1)
        pred_1nn.append(memory.labels[ind[0]].item())
    acc_1nn = accuracy_score(test_labels, pred_1nn)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(train_embs, train_labels)
    acc_lr = accuracy_score(test_labels, clf.predict(test_embs))

    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500)
    mlp.fit(train_embs, train_labels)
    acc_mlp = accuracy_score(test_labels, mlp.predict(test_embs))

    return {"NWF": acc_nwf, "1-NN": acc_1nn, "LogReg": acc_lr, "MLP": acc_mlp}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k-values", type=str, default="1,3,5")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--test-samples", type=int, default=500)
    parser.add_argument("--use-tfds", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output", type=str, default="results/fewshot.csv")
    parser.add_argument("--encoder", type=str, default="encoder_mnist.pth")
    args = parser.parse_args()

    device = torch.device("cpu") if args.cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_path = path_root / args.encoder
    K_values = [int(x) for x in args.k_values.split(",")]

    all_results = {K: {"NWF": [], "1-NN": [], "LogReg": [], "MLP": []} for K in K_values}

    for K in K_values:
        print(f"K={K}...")
        for run in range(args.runs):
            accs = run_one(K, run_seed=run, device=device, encoder_path=encoder_path, test_samples=args.test_samples)
            for m in ["NWF", "1-NN", "LogReg", "MLP"]:
                all_results[K][m].append(accs[m])

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["K", "method", "mean", "std"])
        for K in K_values:
            for m in ["NWF", "1-NN", "LogReg", "MLP"]:
                vals = all_results[K][m]
                w.writerow([K, m, f"{np.mean(vals):.4f}", f"{np.std(vals):.4f}"])
            mean_nwf = np.mean(all_results[K]["NWF"])
            mean_1nn = np.mean(all_results[K]["1-NN"])
            mean_lr = np.mean(all_results[K]["LogReg"])
            mean_mlp = np.mean(all_results[K]["MLP"])
            print(f"  K={K}: NWF={mean_nwf:.4f} 1-NN={mean_1nn:.4f} LogReg={mean_lr:.4f} MLP={mean_mlp:.4f}")

    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
