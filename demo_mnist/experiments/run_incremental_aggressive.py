# -*- coding: utf-8 -*-
"""Агрессивный инкрементальный сценарий: MLP дообучается только на новых классах.

NWF просто добавляет заряды. MLP дообучается только на 5-9 без доступа к 0-4.
Демонстрирует катастрофическое забывание MLP и отсутствие забывания у NWF.
Сырые пиксели (784). Целевой разрыв: NWF ~63%, MLP ~0.4% на старых классах.
"""
import argparse
import json
import pathlib
import sys

path_root = pathlib.Path(__file__).resolve().parent.parent
if str(path_root) not in sys.path:
    sys.path.insert(0, str(path_root))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

from experiments.run_experiment import load_mnist_tfds, MNIST_MEAN, MNIST_STD
from nwf import Memory, trace_ray_memory, classify_weighted


class MLP(nn.Module):
    """MLP 784 -> 128 -> 128 -> 10 (raw pixels, 2 layers of 128)."""

    def __init__(self, input_dim=784, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def get_indices_for_classes(y, classes, k_per_class):
    indices = []
    for c in classes:
        idx = np.where(y == c)[0]
        np.random.shuffle(idx)
        indices.extend(idx[:k_per_class])
    return np.array(indices)


def train_epoch(model, X, y, device, optimizer, criterion):
    model.train()
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.long, device=device)
    optimizer.zero_grad()
    logits = model(X_t)
    loss = criterion(logits, y_t)
    loss.backward()
    optimizer.step()
    return loss.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--test-samples", type=int, default=0)
    parser.add_argument("--use-tfds", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output", type=str, default="results/incremental_aggr.json")
    parser.add_argument("--stage1-epochs", type=int, default=100)
    parser.add_argument("--finetune-epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    OLD_CLASSES = [0, 1, 2, 3, 4]
    NEW_CLASSES = [5, 6, 7, 8, 9]

    print("Loading MNIST (raw pixels)...")
    X_train, y_train, X_test, y_test = load_mnist_tfds()
    X_train_flat = X_train.reshape(-1, 784).astype(np.float32)
    X_test_flat = X_test.reshape(-1, 784).astype(np.float32)

    n_test = len(X_test) if args.test_samples <= 0 else min(args.test_samples, len(X_test))
    test_indices = np.arange(len(X_test))[:n_test]

    old_train_idx = get_indices_for_classes(y_train, OLD_CLASSES, args.k)
    new_train_idx = get_indices_for_classes(y_train, NEW_CLASSES, args.k)

    X_old = X_train_flat[old_train_idx]
    y_old = y_train[old_train_idx]
    X_new = X_train_flat[new_train_idx]
    y_new = y_train[new_train_idx]
    test_X = X_test_flat[test_indices]
    test_labels = y_test[test_indices]
    mask_old_test = np.isin(test_labels, OLD_CLASSES)
    mask_new_test = np.isin(test_labels, NEW_CLASSES)

    # --- NWF ---
    print("NWF: Stage 1 (old only)...")
    memory = Memory(device=str(device))
    for i in range(len(X_old)):
        memory.add(X_old[i], int(y_old[i]), q=1.0, sigma=0.5, sigma_scale_dim=True)

    k_neigh = min(100, memory.get_ntotal())
    def eval_nwf(mem, embs, labels, mask=None):
        if mask is not None:
            embs, labels = embs[mask], labels[mask]
        if len(embs) == 0:
            return 0.0
        pred = []
        for emb in embs:
            r0 = torch.tensor(emb, dtype=torch.float32, device=device)
            r_final, _ = trace_ray_memory(r0, mem, num_steps=20, step_size=0.1, k_neighbors=k_neigh)
            pred.append(classify_weighted(r_final, mem, k=10, temperature=1.0))
        return accuracy_score(labels, pred)

    acc_nwf_old_only = eval_nwf(memory, test_X, test_labels, mask_old_test)
    print(f"  NWF acc on old (stage 1) = {acc_nwf_old_only:.4f}")

    print("NWF: Stage 2 (add new charges)...")
    for i in range(len(X_new)):
        memory.add(X_new[i], int(y_new[i]), q=1.0, sigma=0.5, sigma_scale_dim=True)
    k_neigh = min(100, memory.get_ntotal())

    acc_nwf_old_after = eval_nwf(memory, test_X, test_labels, mask_old_test)
    acc_nwf_new_after = eval_nwf(memory, test_X, test_labels, mask_new_test)
    acc_nwf_total = eval_nwf(memory, test_X, test_labels)
    print(f"  NWF acc on old = {acc_nwf_old_after:.4f}, on new = {acc_nwf_new_after:.4f}, total = {acc_nwf_total:.4f}")

    # --- MLP ---
    print("MLP: Stage 1 (train on old only)...")
    model = MLP(input_dim=784, num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    for ep in range(args.stage1_epochs):
        train_epoch(model, X_old, y_old, device, optimizer, criterion)

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(test_X, dtype=torch.float32, device=device))
        pred = logits.argmax(dim=1).cpu().numpy()
    acc_mlp_old_only = accuracy_score(test_labels[mask_old_test], pred[mask_old_test])
    print(f"  MLP acc on old (stage 1) = {acc_mlp_old_only:.4f}")

    print("MLP: Stage 2 (finetune ONLY on new classes, no old!)...")
    optimizer_ft = torch.optim.Adam(model.parameters(), lr=0.0005)
    for ep in range(args.finetune_epochs):
        train_epoch(model, X_new, y_new, device, optimizer_ft, criterion)

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(test_X, dtype=torch.float32, device=device))
        pred = logits.argmax(dim=1).cpu().numpy()
    acc_mlp_old_after = accuracy_score(test_labels[mask_old_test], pred[mask_old_test])
    acc_mlp_new_after = accuracy_score(test_labels[mask_new_test], pred[mask_new_test])
    acc_mlp_total = accuracy_score(test_labels, pred)
    print(f"  MLP acc on old = {acc_mlp_old_after:.4f}, on new = {acc_mlp_new_after:.4f}, total = {acc_mlp_total:.4f}")

    results = {
        "seed": args.seed,
        "k": args.k,
        "nwf": {
            "acc_old_only": float(acc_nwf_old_only),
            "acc_old_after": float(acc_nwf_old_after),
            "acc_new_after": float(acc_nwf_new_after),
            "acc_total": float(acc_nwf_total),
        },
        "mlp": {
            "acc_old_only": float(acc_mlp_old_only),
            "acc_old_after": float(acc_mlp_old_after),
            "acc_new_after": float(acc_mlp_new_after),
            "acc_total": float(acc_mlp_total),
        },
    }
    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
