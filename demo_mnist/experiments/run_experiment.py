# -*- coding: utf-8 -*-
"""Базовый эксперимент: NWF, 1-NN, LogReg, MLP на MNIST.

Сравнение методов при заданном числе примеров на класс (K).
Поддерживает эмбеддинги или сырые пиксели, тест на шум.
"""
import argparse
import pathlib
import sys
import time

path_root = pathlib.Path(__file__).resolve().parent.parent
if str(path_root) not in sys.path:
    sys.path.insert(0, str(path_root))

import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

from nwf import Encoder, Memory, trace_ray_memory, classify_weighted

# MNIST normalization (matches torchvision)
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


def load_mnist_tfds():
    """Fallback: load MNIST from tensorflow_datasets (uses cache if available)."""
    import tensorflow_datasets as tfds
    train_ds = tfds.load("mnist", split="train")
    test_ds = tfds.load("mnist", split="test")
    X_train = np.stack([np.array(x["image"], dtype=np.float32) / 255.0 for x in train_ds])
    y_train = np.stack([np.array(x["label"], dtype=np.int32) for x in train_ds])
    X_test = np.stack([np.array(x["image"], dtype=np.float32) / 255.0 for x in test_ds])
    y_test = np.stack([np.array(x["label"], dtype=np.int32) for x in test_ds])
    # Normalize like torchvision
    X_train = (X_train - MNIST_MEAN) / MNIST_STD
    X_test = (X_test - MNIST_MEAN) / MNIST_STD
    return X_train, y_train, X_test, y_test


def get_support_set_from_arrays(X, y, k_per_class):
    """Select k examples per class from numpy arrays."""
    indices = []
    for digit in range(10):
        idx = np.where(y == digit)[0]
        np.random.shuffle(idx)
        indices.extend(idx[:k_per_class])
    return np.array(indices)


def get_support_set(dataset, k_per_class):
    """Select k examples per class from torch Dataset."""
    targets = np.array(dataset.targets)
    indices = []
    for digit in range(10):
        idx = np.where(targets == digit)[0]
        np.random.shuffle(idx)
        indices.extend(idx[:k_per_class])
    return indices


def get_embeddings(encoder, dataset, indices, device, batch_size=64):
    """Extract embeddings for given indices."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5, help="Examples per class")
    parser.add_argument("--encoder", type=str, default="encoder_mnist.pth")
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--step-size", type=float, default=0.1)
    parser.add_argument("--k-neighbors", type=int, default=100)
    parser.add_argument("--sigma", type=float, default=0.5, help="Base sigma (scaled by sqrt(dim) for high-d)")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--test-samples", type=int, default=0, help="0 = all test")
    parser.add_argument("--no-encoder", action="store_true", help="Use raw pixels (784)")
    parser.add_argument("--use-tfds", action="store_true", help="Use tensorflow_datasets (fallback if torchvision download fails)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU (use for RTX 5070 until PyTorch adds sm_120 support)")
    parser.add_argument("--noise-std", type=float, default=0.0, help="Gaussian noise std added to test images")
    parser.add_argument("--weighted-voting", action="store_true", help="Use weighted voting instead of 1-NN after trace")
    parser.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature for weighted voting")
    parser.add_argument("--vote-k", type=int, default=10, help="Top-k charges for weighted vote")
    parser.add_argument("--no-sigma-scale", action="store_true", help="Do not scale sigma by sqrt(dim)")
    args = parser.parse_args()

    if args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        try:
            _ = torch.zeros(1, device="cuda")
            device = torch.device("cuda")
        except RuntimeError:
            device = torch.device("cpu")
            print("CUDA kernel error (e.g. RTX 5070 sm_120), using CPU")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    results_dir = pathlib.Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    use_tfds = args.use_tfds
    if not use_tfds:
        try:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
            ])
            data_root = path_root / "data"
            train_dataset = datasets.MNIST(root=str(data_root), train=True, download=True, transform=transform)
            test_dataset = datasets.MNIST(root=str(data_root), train=False, download=True, transform=transform)
        except Exception as e:
            print(f"Torchvision MNIST download failed: {e}")
            print("Falling back to tensorflow_datasets...")
            use_tfds = True

    if use_tfds:
        print("Loading MNIST from tensorflow_datasets...")
        X_train, y_train, X_test, y_test = load_mnist_tfds()
        X_train_flat = X_train.reshape(-1, 784)
        X_test_flat = X_test.reshape(-1, 784)
        train_indices = get_support_set_from_arrays(X_train_flat, y_train, args.k)
        np.random.shuffle(train_indices)
        n_test = len(X_test) if args.test_samples <= 0 else min(args.test_samples, len(X_test))
        test_indices = np.random.choice(len(X_test), n_test, replace=False) if args.test_samples > 0 else np.arange(len(X_test))

        if args.no_encoder:
            train_embs = X_train_flat[train_indices]
            test_embs = X_test_flat[test_indices]
            embed_dim = 784
        else:
            encoder_path = pathlib.Path(args.encoder)
            if not encoder_path.is_absolute():
                encoder_path = path_root / encoder_path
            if not encoder_path.exists():
                print(f"Encoder not found: {encoder_path}. Using raw pixels.")
                args.no_encoder = True
                train_embs = X_train_flat[train_indices]
                test_embs = X_test_flat[test_indices]
                embed_dim = 784
            else:
                encoder = Encoder(output_dim=64).to(device)
                encoder.load_state_dict(torch.load(encoder_path, map_location=device))
                X_train_4d = torch.tensor(X_train.reshape(-1, 1, 28, 28), dtype=torch.float32)
                X_test_4d = torch.tensor(X_test.reshape(-1, 1, 28, 28), dtype=torch.float32)
                with torch.no_grad():
                    train_embs = encoder(X_train_4d[train_indices].to(device)).cpu().numpy()
                    test_embs = encoder(X_test_4d[test_indices].to(device)).cpu().numpy()
                embed_dim = train_embs.shape[1]
        train_labels = y_train[train_indices]
        test_labels = y_test[test_indices]

        if args.noise_std > 0:
            np.random.seed(42)
            X_test_raw = X_test * MNIST_STD + MNIST_MEAN
            X_test_noisy = X_test_raw + np.random.randn(*X_test.shape).astype(np.float32) * args.noise_std
            X_test_noisy = np.clip(X_test_noisy, 0, 1)
            X_test_noisy = (X_test_noisy - MNIST_MEAN) / MNIST_STD
            X_test_flat_noisy = X_test_noisy.reshape(-1, 784)
            if args.no_encoder:
                test_embs = X_test_flat_noisy[test_indices]
            else:
                X_test_4d_noisy = torch.tensor(X_test_noisy.reshape(-1, 1, 28, 28), dtype=torch.float32)
                with torch.no_grad():
                    test_embs = encoder(X_test_4d_noisy[test_indices].to(device)).cpu().numpy()
    else:
        train_indices = get_support_set(train_dataset, args.k)
        np.random.shuffle(train_indices)

        if args.no_encoder:
            train_embs = np.stack([train_dataset[i][0].numpy().flatten() for i in train_indices])
            train_labels = np.array([train_dataset[i][1] for i in train_indices])
            embed_dim = 784
        else:
            encoder_path = pathlib.Path(args.encoder)
            if not encoder_path.is_absolute():
                encoder_path = path_root / encoder_path
            if not encoder_path.exists():
                print(f"Encoder not found: {encoder_path}. Run: python experiments/train_encoder.py")
                return
            encoder = Encoder(output_dim=64).to(device)
            encoder.load_state_dict(torch.load(encoder_path, map_location=device))
            train_embs, train_labels = get_embeddings(encoder, train_dataset, train_indices, device)
            embed_dim = train_embs.shape[1]

        n_test = len(test_dataset) if args.test_samples <= 0 else min(args.test_samples, len(test_dataset))
        test_indices = np.random.choice(len(test_dataset), n_test, replace=False) if args.test_samples > 0 else np.arange(len(test_dataset))

        if args.no_encoder:
            test_embs = np.stack([test_dataset[i][0].numpy().flatten() for i in test_indices])
        else:
            test_embs, _ = get_embeddings(encoder, test_dataset, test_indices, device)
        test_labels = np.array([test_dataset[i][1] for i in test_indices])

        if args.noise_std > 0:
            np.random.seed(42)
            test_imgs = torch.stack([test_dataset[i][0] for i in test_indices])
            test_imgs = test_imgs + torch.randn_like(test_imgs) * args.noise_std
            test_imgs = torch.clamp(test_imgs, 0, 1)
            if args.no_encoder:
                test_embs = test_imgs.numpy().reshape(len(test_indices), -1)
            else:
                test_embs = encoder(test_imgs.to(device)).detach().cpu().numpy()

    print(f"Support: {len(train_embs)}, Test: {len(test_embs)}, dim: {embed_dim}" + (f", noise_std={args.noise_std}" if args.noise_std > 0 else ""))

    # Build Memory (charges). sigma scaled by sqrt(dim) for high-d spaces
    memory = Memory(device=str(device))
    scale_sigma = not args.no_sigma_scale
    for emb, label in zip(train_embs, train_labels):
        memory.add(emb, int(label), q=1.0, sigma=args.sigma, sigma_scale_dim=scale_sigma)

    log_lines = [
        "=== NWF MNIST Experiment (PyTorch) ===",
        f"K={args.k}, dim={embed_dim}, encoder={not args.no_encoder}",
        f"num_steps={args.num_steps}, step_size={args.step_size}",
        f"k_neighbors={args.k_neighbors}, sigma={args.sigma}",
        f"noise_std={args.noise_std}, weighted_voting={args.weighted_voting}",
        "",
    ]

    # NWF (trace + classify)
    print("Running NWF classification...")
    pred_nwf = []
    t0 = time.perf_counter()
    k_neigh = min(args.k_neighbors, memory.get_ntotal())
    for emb in test_embs:
        r0 = torch.tensor(emb, dtype=torch.float32, device=device)
        r_final, _ = trace_ray_memory(r0, memory, num_steps=args.num_steps, step_size=args.step_size, k_neighbors=k_neigh)
        if args.weighted_voting:
            pred = classify_weighted(r_final, memory, k=args.vote_k, temperature=args.temperature)
        else:
            ind, _ = memory.search(r_final, k=1)
            pred = memory.labels[ind[0]].item()
        pred_nwf.append(pred)
    t_nwf = time.perf_counter() - t0
    acc_nwf = accuracy_score(test_labels, pred_nwf)
    log_lines.append(f"NWF: accuracy={acc_nwf:.4f}, time={t_nwf:.2f}s")
    print(log_lines[-1])

    # 1-NN
    print("Running 1-NN baseline...")
    pred_1nn = []
    t0 = time.perf_counter()
    for emb in test_embs:
        r0 = torch.tensor(emb, dtype=torch.float32, device=device)
        ind, _ = memory.search(r0, k=1)
        pred_1nn.append(memory.labels[ind[0]].item())
    t_1nn = time.perf_counter() - t0
    acc_1nn = accuracy_score(test_labels, pred_1nn)
    log_lines.append(f"1-NN: accuracy={acc_1nn:.4f}, time={t_1nn:.2f}s")
    print(log_lines[-1])

    # LogReg
    print("Running LogReg baseline...")
    t0 = time.perf_counter()
    clf = LogisticRegression(max_iter=1000)
    clf.fit(train_embs, train_labels)
    pred_lr = clf.predict(test_embs)
    t_lr = time.perf_counter() - t0
    acc_lr = accuracy_score(test_labels, pred_lr)
    log_lines.append(f"LogReg: accuracy={acc_lr:.4f}, time={t_lr:.2f}s")
    print(log_lines[-1])

    # MLP
    print("Running MLP baseline...")
    t0 = time.perf_counter()
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500)
    mlp.fit(train_embs, train_labels)
    pred_mlp = mlp.predict(test_embs)
    t_mlp = time.perf_counter() - t0
    acc_mlp = accuracy_score(test_labels, pred_mlp)
    log_lines.append(f"MLP: accuracy={acc_mlp:.4f}, time={t_mlp:.2f}s")
    print(log_lines[-1])

    with open(results_dir / "experiment_log.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    np.savez(
        results_dir / "experiment_results.npz",
        y_test=test_labels, y_nwf=pred_nwf, y_1nn=pred_1nn, y_lr=pred_lr, y_mlp=pred_mlp,
        X_support_emb=train_embs, y_support=train_labels, X_test_emb=test_embs,
    )
    print(f"\nLog saved to {results_dir / 'experiment_log.txt'}")


if __name__ == "__main__":
    main()
