# -*- coding: utf-8 -*-
"""Взвешенное голосование для классификации NWF.

Потенциалы в точке r_final служат весами: чем ближе заряд, тем больший вес его метки.
temperature сглаживает распределение (больше — плавнее).
"""
import torch


def classify_weighted(r_final, memory, k=None, temperature=1.0):
    """Классификация взвешенным голосованием по потенциалам. k — верхние k зарядов (None = все)."""
    pots = memory.potentials(r_final).squeeze()
    if pots.dim() == 0:
        pots = pots.unsqueeze(0)
    n = len(pots)
    if k is not None and k < n:
        _, top_idx = torch.topk(pots, k, largest=True)
        pots = pots[top_idx]
        labels = memory.labels[top_idx]
    else:
        labels = memory.labels
    weights = torch.softmax(pots / (temperature + 1e-8), dim=0)
    labels_onehot = torch.nn.functional.one_hot(labels, num_classes=10).float()
    pred_scores = (weights[:, None] * labels_onehot).sum(dim=0)
    return pred_scores.argmax().item()


def classify_nearest(r_final, memory, k=1):
    """Классификация по ближайшему заряду (1-NN в точке r_final)."""
    ind, _ = memory.search(r_final, k=k)
    return memory.labels[ind[0]].item()
