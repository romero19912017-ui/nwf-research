# -*- coding: utf-8 -*-
"""Инкрементальное обучение NWF: добавление зарядов без забывания.

Классы 0-4 — старые, 5-9 — новые. Сначала обучаем на старых, затем добавляем заряды новых.
Точность на старых классах не падает (в отличие от MLP при дообучении).
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
from sklearn.metrics import accuracy_score

from experiments.run_experiment import load_mnist_tfds, get_support_set_from_arrays, MNIST_MEAN, MNIST_STD
from nwf import Encoder, Memory, trace_ray_memory, classify_weighted


def get_indices_for_classes(y, classes, k_per_class):
    indices = []
    for c in classes:
        idx = np.where(y == c)[0]
        np.random.shuffle(idx)
        indices.extend(idx[:k_per_class])
    return np.array(indices)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--test-samples", type=int, default=0)
    parser.add_argument("--use-tfds", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output", type=str, default="results/incremental_nwf.json")
    parser.add_argument("--encoder", type=str, default="encoder_mnist.pth")
    args = parser.parse_args()

    device = torch.device("cpu") if args.cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    np.random.seed(42)
    old_train_idx = get_indices_for_classes(y_train, OLD_CLASSES, args.k)
    new_train_idx = get_indices_for_classes(y_train, NEW_CLASSES, args.k)

    # Stage 1: NWF with old classes only
    memory = Memory(device=str(device))
    for i in old_train_idx:
        memory.add(train_embs_all[i], int(y_train[i]), q=1.0, sigma=0.5, sigma_scale_dim=True)

    mask_old_test = np.isin(y_test[test_indices], OLD_CLASSES)
    mask_new_test = np.isin(y_test[test_indices], NEW_CLASSES)
    test_embs = test_embs_all[test_indices]
    test_labels = y_test[test_indices]

    def eval_nwf(mem, embs, labels, mask=None):
        if mask is not None:
            embs, labels = embs[mask], labels[mask]
        if len(embs) == 0:
            return 0.0
        pred = []
        k_neigh = min(100, mem.get_ntotal())
        for emb in embs:
            r0 = torch.tensor(emb, dtype=torch.float32, device=device)
            r_final, _ = trace_ray_memory(r0, mem, num_steps=20, step_size=0.1, k_neighbors=k_neigh)
            pred.append(classify_weighted(r_final, mem, k=10, temperature=1.0))
        return accuracy_score(labels, pred)

    acc_old_only = eval_nwf(memory, test_embs, test_labels, mask_old_test)
    print(f"Stage 1 (old only): accuracy on old classes = {acc_old_only:.4f}")

    # Stage 2: Add new classes
    for i in new_train_idx:
        memory.add(train_embs_all[i], int(y_train[i]), q=1.0, sigma=0.5, sigma_scale_dim=True)

    acc_old_after = eval_nwf(memory, test_embs, test_labels, mask_old_test)
    acc_new_after = eval_nwf(memory, test_embs, test_labels, mask_new_test)
    acc_total_after = eval_nwf(memory, test_embs, test_labels)
    print(f"Stage 2 (old+new): accuracy on old = {acc_old_after:.4f}, on new = {acc_new_after:.4f}, total = {acc_total_after:.4f}")

    results = {
        "acc_old_only": acc_old_only,
        "acc_old_after_add_new": acc_old_after,
        "acc_new_after": acc_new_after,
        "acc_total_after": acc_total_after,
    }
    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
