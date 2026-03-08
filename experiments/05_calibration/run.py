# -*- coding: utf-8 -*-
"""Эксперимент: калибровка уверенности (ECE). Валидация, метрики, Platt scaling."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torchvision import transforms

from nwf import VAEEncoder, NWFStorage, get_mnist, encode_batch
from nwf.confidence import (
    min_mahalanobis,
    potential_at_query,
    trace_sigma,
    agreement_ratio,
    confidence_1_over_1_plus_d,
)
from nwf.continual import MLPClassifier, train_task

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data"


def get_metric_value(metric_name: str, storage, model, z, sigma_sq, y_true, device):
    """Возвращает значение метрики уверенности (чем выше - тем увереннее)."""
    if metric_name == "min_mahalanobis":
        d = min_mahalanobis(storage, z, sigma_sq, metric="symmetric")
        return float(confidence_1_over_1_plus_d(d).item())
    if metric_name == "potential":
        phi = potential_at_query(storage, z, sigma_sq, metric="symmetric", k=50)
        phi_max = 50.0
        return float((phi / phi_max).clamp(0, 1).item())
    if metric_name == "trace_sigma":
        t = trace_sigma(sigma_sq)
        t_max = 100.0
        return float(1.0 - (t / t_max).clamp(0, 1).item())
    if metric_name == "agreement_ratio":
        return float(agreement_ratio(storage, z, sigma_sq, int(y_true), k=10, metric="symmetric"))
    raise ValueError(metric_name)


def collect_metrics(storage, model, loader, device, metric_names, max_samples=None):
    """Собирает (conf, correct) для каждой метрики."""
    result = {m: {"conf": [], "correct": []} for m in metric_names}
    n = 0
    with torch.no_grad():
        for x, y in loader:
            if max_samples and n >= max_samples:
                break
            x = x.view(-1, 784).to(device)
            mu, sigma_sq = encode_batch(model, x)
            for i in range(mu.size(0)):
                if max_samples and n >= max_samples:
                    break
                idx, dists = storage.search(mu[i], sigma_sq[i], k=10, metric="symmetric")
                labels = storage.get_labels(idx)
                pred = labels.mode().values.item()
                correct = 1 if pred == y[i].item() else 0
                for m in metric_names:
                    conf = get_metric_value(m, storage, model, mu[i], sigma_sq[i], y[i].item(), device)
                    result[m]["conf"].append(conf)
                    result[m]["correct"].append(correct)
                n += 1
    for m in metric_names:
        result[m]["conf"] = np.array(result[m]["conf"])
        result[m]["correct"] = np.array(result[m]["correct"])
    return result


def ece_from_bins(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.clip(np.digitize(conf, bin_edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    n = len(conf)
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            acc = correct[mask].mean()
            avg_conf = conf[mask].mean()
            ece += mask.sum() * np.abs(acc - avg_conf)
    return float(ece / n)


def platt_scale(conf: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """p = 1 / (1 + exp(alpha * conf + beta)). Подбираем alpha, beta для калибровки."""
    z = alpha * conf + beta
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def fit_platt(conf: np.ndarray, correct: np.ndarray):
    """Подгонка Platt scaling: минимизируем NLL."""
    def nll(params):
        a, b = params[0], params[1]
        p = platt_scale(conf, a, b)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -np.sum(correct * np.log(p) + (1 - correct) * np.log(1 - p))

    res = minimize(nll, [1.0, 0.0], method="L-BFGS-B", options={"maxiter": 1000})
    return float(res.x[0]), float(res.x[1])


def reliability_diagram(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10):
    """Возвращает (bin_acc, bin_conf, bin_counts) для построения графика."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.clip(np.digitize(conf, bin_edges[1:-1]), 0, n_bins - 1)
    bin_acc = np.zeros(n_bins)
    bin_conf = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_acc[i] = correct[mask].mean()
            bin_conf[i] = conf[mask].mean()
            bin_counts[i] = mask.sum()
    return bin_acc, bin_conf, bin_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/vae_mnist.pt")
    parser.add_argument("--max_train", type=int, default=10000)
    parser.add_argument("--max_test", type=int, default=2000)
    parser.add_argument("--n_index", type=int, default=8000, help="Примеров в индексе (остальные - validation)")
    parser.add_argument("--n_bins", type=int, default=10)
    parser.add_argument("--out", type=str, default="results")
    parser.add_argument("--calibrate", action="store_true", help="Применить Platt scaling")
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.max_train < args.n_index:
        args.n_index = int(0.8 * args.max_train)
        log.info("Adjusted n_index to %d (quick mode)", args.n_index)

    ckpt = torch.load(args.checkpoint, map_location=device)
    vae = VAEEncoder(input_dim=784, hidden_dims=(512, 256), latent_dim=ckpt["latent_dim"])
    vae.load_state_dict(ckpt["model"])
    vae.to(device)
    vae.eval()

    train_loader, _, test_loader = get_mnist(batch_size=128, download=True)
    storage = NWFStorage(device=device)
    n_indexed = 0
    with torch.no_grad():
        for x, y in train_loader:
            if n_indexed >= args.n_index:
                break
            x = x.view(-1, 784).to(device)
            mu, sigma_sq = encode_batch(vae, x)
            storage.add_batch(mu, sigma_sq, y)
            n_indexed += x.size(0)

    log.info("Indexed %d samples", len(storage))

    metric_names = ["min_mahalanobis", "potential", "trace_sigma", "agreement_ratio"]
    n_val = max(0, min(2000, args.max_train - args.n_index))
    if n_val == 0:
        n_val = min(500, len(train_ds) - args.n_index)

    root = DATA_ROOT / "mnist"
    tf = transforms.ToTensor()
    train_ds = MNIST(root=str(root), train=True, download=True, transform=tf)
    val_indices = np.arange(args.n_index, min(args.n_index + n_val, len(train_ds)))
    if len(val_indices) == 0:
        val_indices = np.arange(min(500, len(train_ds)))
    val_loader = DataLoader(Subset(train_ds, val_indices.astype(int)), batch_size=128, shuffle=False)
    log.info("Validation samples: %d", len(val_indices))

    results_metrics = {}
    for m in metric_names:
        data = collect_metrics(storage, vae, val_loader, device, [m], max_samples=n_val)
        ece = ece_from_bins(data[m]["conf"], data[m]["correct"], args.n_bins)
        results_metrics[m] = {"ece": ece, "n_val": len(data[m]["conf"])}
        log.info("  %s: ECE = %.4f", m, ece)

    best_metric = min(results_metrics, key=lambda k: results_metrics[k]["ece"])
    log.info("Best metric: %s", best_metric)

    conf_val = collect_metrics(storage, vae, val_loader, device, [best_metric], max_samples=n_val)[best_metric]
    alpha, beta = fit_platt(conf_val["conf"], conf_val["correct"])
    log.info("Platt: alpha=%.4f, beta=%.4f", alpha, beta)

    test_data = collect_metrics(storage, vae, test_loader, device, [best_metric], max_samples=args.max_test)[best_metric]
    ece_before = ece_from_bins(test_data["conf"], test_data["correct"], args.n_bins)
    conf_cal = platt_scale(test_data["conf"], alpha, beta)
    ece_after = ece_from_bins(conf_cal, test_data["correct"], args.n_bins)
    log.info("Test: ECE before=%.4f, after=%.4f", ece_before, ece_after)

    results = {
        "metrics_ece": results_metrics,
        "best_metric": best_metric,
        "platt_alpha": alpha,
        "platt_beta": beta,
        "ece_test_before": ece_before,
        "ece_test_after": ece_after,
        "_meta": {"n_index": len(storage), "n_val": n_val, "n_bins": args.n_bins},
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "05_calibration_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results_metrics, f, indent=2)
    with open(out_dir / "calibration_params.json", "w", encoding="utf-8") as f:
        json.dump({"alpha": alpha, "beta": beta, "best_metric": best_metric}, f, indent=2)
    with open(out_dir / "05_calibration.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        acc_b, conf_b, _ = reliability_diagram(test_data["conf"], test_data["correct"], args.n_bins)
        acc_a, conf_a, _ = reliability_diagram(conf_cal, test_data["correct"], args.n_bins)
        axes[0].bar(conf_b, acc_b, width=0.08, alpha=0.7, label="acc")
        axes[0].plot([0, 1], [0, 1], "k--")
        axes[0].set_xlabel("Confidence")
        axes[0].set_ylabel("Accuracy")
        axes[0].set_title(f"Before (ECE={ece_before:.3f})")
        axes[0].set_xlim(0, 1)
        axes[1].bar(conf_a, acc_a, width=0.08, alpha=0.7)
        axes[1].plot([0, 1], [0, 1], "k--")
        axes[1].set_xlabel("Confidence")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title(f"After Platt (ECE={ece_after:.3f})")
        axes[1].set_xlim(0, 1)
        plt.tight_layout()
        plt.savefig(out_dir / "05_calibration_final.png", dpi=120)
        plt.close()
        log.info("Saved %s", out_dir / "05_calibration_final.png")
    except Exception as e:
        log.warning("Plot failed: %s", e)

    log.info("Saved %s", out_dir / "05_calibration.json")


if __name__ == "__main__":
    main()
