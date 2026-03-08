# -*- coding: utf-8 -*-
"""CNN-энкодер для получения эмбеддингов из изображений MNIST.

Преобразует изображение 28x28 в вектор фиксированной размерности (по умолчанию 64).
Архитектура: Conv32 -> Conv64 -> Dense128 -> Dense(output_dim).
"""
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Свёрточная сеть: Conv32 -> Conv64 -> Dense128 -> Dense(output_dim)."""

    def __init__(self, output_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        # x: (batch, 1, 28, 28)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
