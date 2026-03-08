# -*- coding: utf-8 -*-
"""Инкрементальное обучение MLP: демонстрация катастрофического забывания.

Обучаем на старых классах (0-4), затем дообучаем на старых+новых (0-9).
PyTorch MLP. Для агрессивного сценария (только новые) см. run_incremental_aggressive.py.
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
from nwf import Encoder


class MLP(nn.Module):
    """MLP 64 -> 128 -> 64 -> num_classes (same as sklearn in run_experiment)."""

    def __init__(self, input_dim=64, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

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
    parser.add_argument("--output", type=str, default="results/incremental_mlp.json")
    parser.add_argument("--encoder", type=str, default="encoder_mnist.pth")
    parser.add_argument("--finetune-epochs", type=int, default=30)
    parser.add_argument("--stage1-epochs", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cpu")
    torch.manual_seed(42)
    np.random.seed(42)
    OLD_CLASSES = [0, 1, 2, 3, 4]
    NEW_CLASSES = [5, 6, 7, 8, 9]

    print("Loading MNIST...")
    X_train, y_train, X_test, y_test = load_mnist_tfds()
    X_train_flat = X_train.reshape(-1, 784)
    X_test_flat = X_test.reshape(-1, 784)

    encoder_path = path_root / args.encoder
    if encoder_path.exists():
        encoder = Encoder(output_dim=64).to(device)
        encoder.load_state_dict(torch.load(encoder_path, map_location=device))
        X_train_4d = torch.tensor(X_train.reshape(-1, 1, 28, 28), dtype=torch.float32)
        X_test_4d = torch.tensor(X_test.reshape(-1, 1, 28, 28), dtype=torch.float32)
        with torch.no_grad():
            train_embs_all = encoder(X_train_4d.to(device)).cpu().numpy()
            test_embs_all = encoder(X_test_4d.to(device)).cpu().numpy()
    else:
        train_embs_all = X_train_flat
        test_embs_all = X_test_flat

    n_test = len(X_test) if args.test_samples <= 0 else min(args.test_samples, len(X_test))
    test_indices = np.arange(len(X_test))[:n_test]

    old_train_idx = get_indices_for_classes(y_train, OLD_CLASSES, args.k)
    new_train_idx = get_indices_for_classes(y_train, NEW_CLASSES, args.k)

    test_embs = test_embs_all[test_indices]
    test_labels = y_test[test_indices]
    mask_old_test = np.isin(test_labels, OLD_CLASSES)
    mask_new_test = np.isin(test_labels, NEW_CLASSES)

    # Stage 1: Train MLP on old classes only (10 outputs, supervision only for 0-4)
    X_old = train_embs_all[old_train_idx]
    y_old = y_train[old_train_idx]
    model = MLP(input_dim=64, num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    for ep in range(args.stage1_epochs):
        train_epoch(model, X_old, y_old, device, optimizer, criterion)
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(test_embs, dtype=torch.float32, device=device))
        pred = logits.argmax(dim=1).cpu().numpy()
    acc_old_only = accuracy_score(test_labels[mask_old_test], pred[mask_old_test])
    print(f"Stage 1 (old only): accuracy on old classes = {acc_old_only:.4f}")

    # Stage 2: Finetune on old+new (catastrophic forgetting)
    X_all = np.vstack([train_embs_all[old_train_idx], train_embs_all[new_train_idx]])
    y_all = np.concatenate([y_train[old_train_idx], y_train[new_train_idx]])
    optimizer_ft = torch.optim.Adam(model.parameters(), lr=0.001)
    for ep in range(args.finetune_epochs):
        train_epoch(model, X_all, y_all, device, optimizer_ft, criterion)
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(test_embs, dtype=torch.float32, device=device))
        pred = logits.argmax(dim=1).cpu().numpy()
    acc_old_after = accuracy_score(test_labels[mask_old_test], pred[mask_old_test])
    acc_new_after = accuracy_score(test_labels[mask_new_test], pred[mask_new_test])
    acc_total_after = accuracy_score(test_labels, pred)
    print(f"Stage 2 (finetuned on old+new): accuracy on old = {acc_old_after:.4f}, on new = {acc_new_after:.4f}, total = {acc_total_after:.4f}")

    results = {
        "acc_old_only": float(acc_old_only),
        "acc_old_after_add_new": float(acc_old_after),
        "acc_new_after": float(acc_new_after),
        "acc_total_after": float(acc_total_after),
    }
    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
