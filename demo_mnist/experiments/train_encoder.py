# -*- coding: utf-8 -*-
"""Обучение CNN-энкодера на MNIST.

Энкодер преобразует изображения в 64-мерные эмбеддинги.
Сохраняет веса в encoder_mnist.pth для использования в экспериментах.
"""
import argparse
import pathlib
import sys

path_root = pathlib.Path(__file__).resolve().parent.parent
if str(path_root) not in sys.path:
    sys.path.insert(0, str(path_root))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from nwf.encoder import Encoder

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


def load_mnist_tfds():
    """Load MNIST from tensorflow_datasets."""
    import tensorflow_datasets as tfds
    import numpy as np

    train_ds = tfds.load("mnist", split="train")
    X = np.stack([np.array(x["image"], dtype=np.float32) / 255.0 for x in train_ds])
    y = np.stack([np.array(x["label"], dtype=np.int64) for x in train_ds])
    X = (X - MNIST_MEAN) / MNIST_STD
    if X.ndim == 3:
        X = X[:, np.newaxis, :, :]  # (N, 28, 28) -> (N, 1, 28, 28)
    else:
        X = np.transpose(X, (0, 3, 1, 2))  # (N, H, W, C) -> (N, C, H, W)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y)
    return TensorDataset(X, y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", type=str, default="encoder_mnist.pth")
    parser.add_argument("--output-dim", type=int, default=64)
    parser.add_argument("--use-tfds", action="store_true", help="Load MNIST from tensorflow_datasets")
    parser.add_argument("--cpu", action="store_true", help="Force CPU (RTX 5070)")
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

    if args.use_tfds:
        print("Loading MNIST from tensorflow_datasets...")
        train_dataset = load_mnist_tfds()
    else:
        from torchvision import datasets, transforms

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ])
        data_root = path_root / "data"
        train_dataset = datasets.MNIST(root=str(data_root), train=True, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    encoder = Encoder(output_dim=args.output_dim).to(device)
    classifier = nn.Linear(args.output_dim, 10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=args.lr)

    encoder.train()
    classifier.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            features = encoder(data)
            output = classifier(features)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {total_loss/len(train_loader):.4f}")

    out_path = pathlib.Path(args.output)
    if not out_path.is_absolute():
        out_path = path_root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), out_path)
    print(f"Encoder saved to {out_path}")


if __name__ == "__main__":
    main()
