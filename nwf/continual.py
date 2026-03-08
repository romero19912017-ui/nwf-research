# -*- coding: utf-8 -*-
"""Continual learning baselines: EWC, naive fine-tuning, iCaRL."""
from __future__ import annotations

from itertools import cycle
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision.datasets import MNIST
from torchvision import transforms

from .data import DATA_ROOT


class MLPClassifier(nn.Module):
    """MLP для классификации MNIST."""

    def __init__(self, hidden: Tuple[int, ...] = (256, 128), num_classes: int = 10):
        super().__init__()
        layers = [nn.Linear(784, hidden[0]), nn.ReLU()]
        for i in range(len(hidden) - 1):
            layers.extend([nn.Linear(hidden[i], hidden[i + 1]), nn.ReLU()])
        self.backbone = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 784)
        h = self.backbone(x)
        return self.fc(h)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 784)
        return self.backbone(x)


def get_split_mnist_loaders(batch_size: int = 128, max_per_task: Optional[int] = None):
    """Task 0: 0,1,2; Task 1: 3,4,5; Task 2: 6,7,8,9."""
    ranges = [(0, 3), (3, 6), (6, 10)]
    root = DATA_ROOT / "mnist"
    root.mkdir(parents=True, exist_ok=True)
    tf = transforms.ToTensor()
    train_ds = MNIST(root=str(root), train=True, download=True, transform=tf)
    test_ds = MNIST(root=str(root), train=False, download=True, transform=tf)

    loaders = []
    for low, high in ranges:
        idx = [i for i, (_, y) in enumerate(train_ds) if low <= y < high]
        if max_per_task:
            idx = idx[:max_per_task]
        loaders.append(DataLoader(Subset(train_ds, idx), batch_size=batch_size, shuffle=True, num_workers=0))
    return loaders, test_ds


def train_task(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int = 5,
    lr: float = 1e-3,
    ewc_lambda: float = 0,
    fisher: Optional[dict] = None,
    old_params: Optional[dict] = None,
) -> Tuple[dict, dict]:
    """Обучить на одной задаче. Возвращает (fisher, old_params) для EWC."""
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            if ewc_lambda > 0 and fisher is not None and old_params is not None:
                ewc_loss = 0
                for n, p in model.named_parameters():
                    if n in fisher and n in old_params:
                        ewc_loss += (fisher[n] * (p - old_params[n]) ** 2).sum()
                loss += ewc_lambda * ewc_loss / 2
            loss.backward()
            opt.step()

    # Compute Fisher (diagonal) on task data - one sample at a time
    model.eval()
    fisher_new = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
    n_samples = 0
    for x, y in loader:
        if n_samples >= 500:
            break
        for i in range(x.size(0)):
            if n_samples >= 500:
                break
            xi = x[i:i+1].to(device).requires_grad_(True)
            logits = model(xi)
            log_prob = F.log_softmax(logits, dim=1)[0, y[i].item()]
            model.zero_grad()
            log_prob.backward()
            for n, p in model.named_parameters():
                if p.grad is not None and n in fisher_new:
                    fisher_new[n] += p.grad.data ** 2
            n_samples += 1

    for n in fisher_new:
        fisher_new[n] /= max(n_samples, 1)

    old_params_new = {n: p.data.clone() for n, p in model.named_parameters()}
    return fisher_new, old_params_new


def evaluate(model: nn.Module, test_loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0


def evaluate_per_class(model: nn.Module, test_loader: DataLoader, device: torch.device) -> dict:
    """Точность по каждому классу. Возвращает {class_id: acc}."""
    model.eval()
    correct_per_class = {}
    total_per_class = {}
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            for i in range(y.size(0)):
                c = y[i].item()
                total_per_class[c] = total_per_class.get(c, 0) + 1
                correct_per_class[c] = correct_per_class.get(c, 0) + (1 if pred[i] == c else 0)
    return {c: correct_per_class.get(c, 0) / max(total_per_class.get(c, 1), 1)
            for c in range(10)}


# --- iCaRL ---

class ICaRLModel(nn.Module):
    """MLP с расширяемой головой для iCaRL."""

    def __init__(self, hidden: Tuple[int, ...] = (256, 128), num_classes: int = 3):
        super().__init__()
        layers = [nn.Linear(784, hidden[0]), nn.ReLU()]
        for i in range(len(hidden) - 1):
            layers.extend([nn.Linear(hidden[i], hidden[i + 1]), nn.ReLU()])
        self.backbone = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden[-1], num_classes)
        self._num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 784)
        h = self.backbone(x)
        return self.fc(h)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 784)
        return self.backbone(x)

    def expand_head(self, new_num_classes: int, device: torch.device):
        old_fc = self.fc
        self.fc = nn.Linear(old_fc.in_features, new_num_classes).to(device)
        with torch.no_grad():
            self.fc.weight[: self._num_classes] = old_fc.weight
            self.fc.bias[: self._num_classes] = old_fc.bias
        self._num_classes = new_num_classes


def herding_selection(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_id: int,
    k: int,
) -> List[Tuple[torch.Tensor, int]]:
    """Herding: выбрать K экземпляров, чей mean ближе всего к среднему класса."""
    model.eval()
    all_x, all_y = [], []
    with torch.no_grad():
        for x, y in loader:
            for i in range(x.size(0)):
                if y[i].item() == class_id:
                    all_x.append(x[i])
                    all_y.append(y[i].item())
    if len(all_x) == 0:
        return []
    X = torch.stack(all_x).to(device)
    feat = model.features(X)
    mu = feat.mean(dim=0)
    selected = []
    used = set()
    for _ in range(min(k, len(all_x))):
        best_idx, best_dist = -1, float("inf")
        for i in range(len(all_x)):
            if i in used:
                continue
            cand = list(used) + [i]
            mean_sel = feat[cand].mean(dim=0)
            d = (mean_sel - mu).norm().item()
            if d < best_dist:
                best_dist = d
                best_idx = i
        if best_idx >= 0:
            used.add(best_idx)
            selected.append((all_x[best_idx], all_y[best_idx]))
    return selected


def run_icarl(
    loaders: List[DataLoader],
    test_ds,
    device: torch.device,
    exemplars_per_class: int = 20,
    epochs: int = 5,
    lr: float = 1e-3,
) -> Tuple[List[float], List[dict]]:
    """iCaRL: exemplar replay + distillation."""
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)
    ranges = [(0, 3), (3, 6), (6, 10)]
    model = ICaRLModel(hidden=(256, 128), num_classes=3).to(device)
    exemplars: Dict[int, List[Tuple[torch.Tensor, int]]] = {}
    old_model = None
    accs_after = []
    forgetting_matrix = []

    for task_id, loader in enumerate(loaders):
        low, high = ranges[task_id]
        new_classes = list(range(low, high))
        num_old = sum(len(exemplars.get(c, [])) for c in range(high))
        num_new = len(new_classes)

        if task_id == 0:
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            for _ in range(epochs):
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    opt.zero_grad()
                    loss = F.cross_entropy(model(x), y)
                    loss.backward()
                    opt.step()
        else:
            old_model = type(model)(hidden=(256, 128), num_classes=len(exemplars)).to(device)
            old_model.load_state_dict(model.state_dict())
            model.expand_head(len(exemplars) + num_new, device)
            opt = torch.optim.Adam(model.parameters(), lr=lr)

            exemplar_list = []
            for c, exs in exemplars.items():
                exemplar_list.extend(exs)
            ex_x = torch.stack([e[0] for e in exemplar_list])
            ex_y = torch.tensor([e[1] for e in exemplar_list])
            ex_loader_iter = cycle(DataLoader(TensorDataset(ex_x, ex_y), batch_size=32, shuffle=True))
            for ep in range(epochs):
                model.train()
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    opt.zero_grad()
                    logits = model(x)
                    loss_ce = F.cross_entropy(logits, y)

                    ex_x_b, ex_y_b = next(ex_loader_iter)
                    ex_x_b, ex_y_b = ex_x_b.to(device), ex_y_b.to(device)
                    with torch.no_grad():
                        old_logits = old_model(ex_x_b)
                        old_probs = F.softmax(old_logits, dim=1)
                    new_logits = model(ex_x_b)[:, : old_model._num_classes]
                    log_new = F.log_softmax(new_logits, dim=1)
                    loss_distill = F.kl_div(log_new, old_probs, reduction="batchmean")
                    loss = loss_ce + loss_distill
                    loss.backward()
                    opt.step()

        for c in new_classes:
            indices = [i for i in range(len(loader.dataset)) if loader.dataset[i][1] == c]
            if not indices:
                exemplars[c] = []
                continue
            class_subset = Subset(loader.dataset, indices)
            class_loader = DataLoader(class_subset, batch_size=32)
            exs = herding_selection(model, class_loader, device, c, exemplars_per_class)
            exemplars[c] = exs

        per_class = evaluate_per_class_icarl(model, exemplars, test_loader, device, ranges, task_id)
        forgetting_matrix.append(per_class)
        acc = sum(per_class.values()) / 10
        accs_after.append(acc)
    return accs_after, forgetting_matrix


def evaluate_per_class_icarl(
    model: nn.Module,
    exemplars: Dict[int, List[Tuple[torch.Tensor, int]]],
    test_loader: DataLoader,
    device: torch.device,
    ranges: List[Tuple[int, int]],
    task_id: int,
) -> dict:
    """Точность по классам для iCaRL (nearest-mean)."""
    model.eval()
    means = {}
    for c, exs in exemplars.items():
        if not exs:
            continue
        xs = torch.stack([e[0] for e in exs]).to(device)
        with torch.no_grad():
            means[c] = model.features(xs).mean(dim=0)
    classes = list(means.keys())
    if not classes:
        return {c: 0.0 for c in range(10)}

    correct_per_class = {c: 0 for c in range(10)}
    total_per_class = {c: 0 for c in range(10)}

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            feat = model.features(x)
            for i in range(x.size(0)):
                c_true = y[i].item()
                total_per_class[c_true] += 1
                best_c, best_d = -1, float("inf")
                for c in classes:
                    d = (feat[i] - means[c]).norm().item()
                    if d < best_d:
                        best_d = d
                        best_c = c
                if best_c == c_true:
                    correct_per_class[c_true] += 1

    return {c: correct_per_class[c] / max(total_per_class[c], 1) for c in range(10)}
