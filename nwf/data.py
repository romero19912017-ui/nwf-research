# -*- coding: utf-8 -*-
"""Загрузка датасетов MNIST, CIFAR-10."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


def get_mnist(
    batch_size: int = 64,
    data_dir: Optional[Path] = None,
    download: bool = True,
    normalize: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """MNIST: train, val, test. normalize=False для VAE (BCE требует [0,1])."""
    root = data_dir or DATA_ROOT / "mnist"
    root.mkdir(parents=True, exist_ok=True)
    tf = transforms.Compose(
        [transforms.ToTensor()]
        + ([transforms.Normalize((0.1307,), (0.3081,))] if normalize else [])
    )
    full = datasets.MNIST(root=str(root), train=True, download=download, transform=tf)
    n_val = int(0.1 * len(full))
    n_train = len(full) - n_val
    train_ds, val_ds = random_split(full, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    test_ds = datasets.MNIST(root=str(root), train=False, download=download, transform=tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader


def get_cifar10(
    batch_size: int = 64,
    data_dir: Optional[Path] = None,
    download: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """CIFAR-10: train, val, test."""
    root = data_dir or DATA_ROOT / "cifar10"
    root.mkdir(parents=True, exist_ok=True)
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    full = datasets.CIFAR10(root=str(root), train=True, download=download, transform=tf_train)
    n_val = int(0.1 * len(full))
    n_train = len(full) - n_val
    train_ds, val_ds = random_split(full, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    test_ds = datasets.CIFAR10(root=str(root), train=False, download=download, transform=tf_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader
