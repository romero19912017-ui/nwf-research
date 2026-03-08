# -*- coding: utf-8 -*-
"""Перебор числа примеров K: точность vs K, сохранение в CSV.

Запуск базового эксперимента для нескольких значений K, построение графика.
"""
import argparse
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.linear_model import LogisticRegression

path_root = pathlib.Path(__file__).resolve().parent.parent
if str(path_root) not in sys.path:
    sys.path.insert(0, str(path_root))

from nwf import Encoder, Memory, trace_ray


def get_support_indices(dataset, k_per_class):
    targets = np.array(dataset.targets)
    indices = []
    for digit in range(10):
        idx = np.where(targets == digit)[0]
        np.random.shuffle(idx)
        indices.extend(idx[:k_per_class])
    return indices


def get_support_indices_from_arrays(y, k_per_class):
    indices = []
    for digit in range(10):
        idx = np.where(y == digit)[0]
        np.random.shuffle(idx)
        indices.extend(idx[:k_per_class])
    return np.array(indices)


def get_embeddings(encoder, dataset, indices, device, batch_size=64):
    encoder.eval()
    embs, labels = [], []
    with torch.no_grad():
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i : i + batch_size]
            imgs = torch.stack([dataset[j][0] for j in batch_idx]).to(device)
            lbls = [dataset[j][1] for j in batch_idx]
            z = encoder(imgs).cpu().numpy()
            embs.append(z)
            labels.extend(lbls)
    return np.vstack(embs), np.array(labels)


def run_nwf(X_support_emb, y_support, X_test_emb, device, num_steps=20, step_size=0.1, sigma=0.5):
    memory = Memory(device=str(device))
    for emb, label in zip(X_support_emb, y_support):
        memory.add(emb, int(label), q=1.0, sigma=sigma)
    k_neigh = min(100, memory.get_ntotal())
    pred = []
    for emb in X_test_emb:
        r0 = torch.tensor(emb, dtype=torch.float32, device=device)
        r_final, _ = trace_ray(r0, memory.zs, memory.qs, memory.sigmas,
                               num_steps=num_steps, step_size=step_size,
                               k_neighbors=k_neigh, device=device)
        ind, _ = memory.search(r_final, k=1)
        pred.append(memory.labels[ind[0]].item())
    return np.array(pred)


def run_1nn(X_support_emb, y_support, X_test_emb, device):
    memory = Memory(device=str(device))
    for emb, label in zip(X_support_emb, y_support):
        memory.add(emb, int(label), q=1.0, sigma=0.5, sigma_scale_dim=True)
    pred = []
    for emb in X_test_emb:
        r0 = torch.tensor(emb, dtype=torch.float32, device=device)
        ind, _ = memory.search(r0, k=1)
        pred.append(memory.labels[ind[0]].item())
    return np.array(pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k-values", type=str, default="1,3,5,10,20")
    parser.add_argument("--encoder", type=str, default="encoder_mnist.pth")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--test-samples", type=int, default=500)
    parser.add_argument("--no-encoder", action="store_true")
    parser.add_argument("--use-tfds", action="store_true", help="Load MNIST from tensorflow_datasets")
    parser.add_argument("--cpu", action="store_true", help="Force CPU (RTX 5070)")
    parser.add_argument("--output", type=str, default="", help="Save sweep results to CSV")
    args = parser.parse_args()

    if args.cpu:
        device = torch.device("cpu")
    else:
        try:
            _ = torch.zeros(1, device="cuda")
            device = torch.device("cuda")
        except RuntimeError:
            device = torch.device("cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    data_root = path_root / "data"
    use_tfds = args.use_tfds
    if not use_tfds:
        try:
            train_dataset = datasets.MNIST(root=str(data_root), train=True, download=True, transform=transform)
            test_dataset = datasets.MNIST(root=str(data_root), train=False, download=True, transform=transform)
        except Exception:
            use_tfds = True

    if use_tfds:
        import tensorflow_datasets as tfds
        print("Loading MNIST from tensorflow_datasets...")
        train_ds = tfds.load("mnist", split="train")
        test_ds = tfds.load("mnist", split="test")
        X_train = np.stack([np.array(x["image"], dtype=np.float32) / 255.0 for x in train_ds])
        y_train = np.stack([np.array(x["label"], dtype=np.int32) for x in train_ds])
        X_test = np.stack([np.array(x["image"], dtype=np.float32) / 255.0 for x in test_ds])
        y_test = np.stack([np.array(x["label"], dtype=np.int32) for x in test_ds])
        X_train = ((X_train - 0.1307) / 0.3081).reshape(-1, 784)
        X_test = ((X_test - 0.1307) / 0.3081).reshape(-1, 784)
        train_dataset = test_dataset = None
        args.no_encoder = True
    else:
        X_train = X_test = y_train = y_test = None

    n_test = min(args.test_samples, 10000 if use_tfds else len(test_dataset))
    test_indices = np.random.choice(10000 if use_tfds else len(test_dataset), n_test, replace=False)

    encoder = None
    if not args.no_encoder:
        enc_path = path_root / args.encoder if not pathlib.Path(args.encoder).is_absolute() else pathlib.Path(args.encoder)
        if enc_path.exists():
            encoder = Encoder(output_dim=64).to(device)
            encoder.load_state_dict(torch.load(enc_path, map_location=device))
        else:
            args.no_encoder = True

    K_values = [int(x) for x in args.k_values.split(",")]
    results_dir = pathlib.Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    acc_nwf, acc_1nn, acc_lr = [], [], []

    for K in K_values:
        print(f"\n--- K={K} ---")
        if use_tfds:
            train_indices = get_support_indices_from_arrays(y_train, K)
            X_support_emb = X_train[train_indices]
            X_test_emb = X_test[test_indices]
            y_support = y_train[train_indices]
            y_test_arr = y_test[test_indices]
        else:
            train_indices = get_support_indices(train_dataset, K)
            if args.no_encoder:
                X_support_emb = np.stack([train_dataset[i][0].numpy().flatten() for i in train_indices])
                X_test_emb = np.stack([test_dataset[i][0].numpy().flatten() for i in test_indices])
            else:
                X_support_emb, _ = get_embeddings(encoder, train_dataset, train_indices, device)
                X_test_emb, _ = get_embeddings(encoder, test_dataset, test_indices, device)
            y_support = np.array([train_dataset[i][1] for i in train_indices])
            y_test_arr = np.array([test_dataset[i][1] for i in test_indices])

        y_nwf = run_nwf(X_support_emb, y_support, X_test_emb, device)
        y_1nn = run_1nn(X_support_emb, y_support, X_test_emb, device)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_support_emb, y_support)
        y_lr = clf.predict(X_test_emb)

        acc_nwf.append(np.mean(y_nwf == y_test_arr))
        acc_1nn.append(np.mean(y_1nn == y_test_arr))
        acc_lr.append(np.mean(y_lr == y_test_arr))
        print(f"NWF={acc_nwf[-1]:.4f} 1-NN={acc_1nn[-1]:.4f} LogReg={acc_lr[-1]:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(K_values, acc_nwf, "o-", label="NWF")
    plt.plot(K_values, acc_1nn, "s-", label="1-NN")
    plt.plot(K_values, acc_lr, "^-", label="LogReg")
    plt.xlabel("K (examples per class)")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy vs K (PyTorch)")
    plt.grid(True, alpha=0.3)
    plt.savefig(results_dir / "accuracy_vs_k.png", dpi=150)
    plt.close()
    print(f"\nSaved {results_dir / 'accuracy_vs_k.png'}")

    if args.output:
        import csv
        out_path = pathlib.Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["K", "NWF", "1-NN", "LogReg"])
            for k, a, b, c in zip(K_values, acc_nwf, acc_1nn, acc_lr):
                w.writerow([k, f"{a:.4f}", f"{b:.4f}", f"{c:.4f}"])
        print(f"Saved CSV to {out_path}")


if __name__ == "__main__":
    main()
