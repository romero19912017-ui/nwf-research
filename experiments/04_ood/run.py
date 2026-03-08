# -*- coding: utf-8 -*-
"""Эксперимент: OOD-детекция. MNIST (in) vs Fashion-MNIST (OOD)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from nwf import VAEEncoder, NWFStorage, encode_batch
from nwf.baselines import L2Index
from nwf.confidence import potential_at_query

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data"


def get_mnist_loaders(batch_size: int, max_samples: int = 2000):
    root = DATA_ROOT / "mnist"
    tf = transforms.ToTensor()
    ds = datasets.MNIST(root=str(root), train=False, download=True, transform=tf)
    idx = np.random.RandomState(42).choice(len(ds), min(max_samples, len(ds)), replace=False)
    return DataLoader(torch.utils.data.Subset(ds, idx), batch_size=batch_size, shuffle=False, num_workers=0)


def get_fashion_loaders(batch_size: int, max_samples: int = 2000):
    root = DATA_ROOT / "fashion"
    root.mkdir(parents=True, exist_ok=True)
    tf = transforms.ToTensor()
    ds = datasets.FashionMNIST(root=str(root), train=False, download=True, transform=tf)
    idx = np.random.RandomState(42).choice(len(ds), min(max_samples, len(ds)), replace=False)
    return DataLoader(torch.utils.data.Subset(ds, idx), batch_size=batch_size, shuffle=False, num_workers=0)


def compute_scores_nwf_dist(storage, model, loader, device):
    """Min Mahalanobis dist (higher = OOD)."""
    scores = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.view(-1, 784).to(device)
            mu, sigma_sq = encode_batch(model, x)
            for i in range(mu.size(0)):
                idx, dists = storage.search(mu[i], sigma_sq[i], k=1)
                d = dists.flatten()[0].item() if dists.numel() > 1 else dists[0].item()
                scores.append(d)
    return np.array(scores)


def compute_scores_nwf_potential(storage, model, loader, device):
    """Potential Phi(z). Lower = OOD, so score = -Phi for ROC (higher = OOD)."""
    scores = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.view(-1, 784).to(device)
            mu, sigma_sq = encode_batch(model, x)
            for i in range(mu.size(0)):
                phi = potential_at_query(storage, mu[i], sigma_sq[i], k=50)
                scores.append(-float(phi))
    return np.array(scores)


def compute_scores_l2(index, model, loader, device):
    """Min L2 dist to nearest neighbor."""
    scores = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.view(-1, 784).to(device)
            mu, _ = encode_batch(model, x)
            for i in range(mu.size(0)):
                idx, dists = index.search(mu[i].cpu().numpy(), k=1)
                scores.append(dists[0])
    return np.array(scores)


def auc_roc(in_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """OOD has higher distance -> positive class. AUC for OOD detection."""
    in_labels = np.zeros(len(in_scores))
    ood_labels = np.ones(len(ood_scores))
    labels = np.concatenate([in_labels, ood_labels])
    scores = np.concatenate([in_scores, ood_scores])
    order = np.argsort(scores)
    labels_sorted = labels[order]
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    tpr = np.cumsum(labels_sorted) / max(n_pos, 1)
    fpr = np.cumsum(1 - labels_sorted) / max(n_neg, 1)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(labels, scores)
    return float(auc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/vae_mnist.pt")
    parser.add_argument("--max_train", type=int, default=10000)
    parser.add_argument("--max_test", type=int, default=2000)
    parser.add_argument("--out", type=str, default="results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = VAEEncoder(input_dim=784, hidden_dims=(512, 256), latent_dim=ckpt["latent_dim"])
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    from nwf import get_mnist
    train_loader, _, _ = get_mnist(batch_size=128, download=True)
    in_loader = get_mnist_loaders(128, args.max_test)
    ood_loader = get_fashion_loaders(128, args.max_test)

    storage = NWFStorage(device=device)
    l2_index = L2Index()
    n = 0
    with torch.no_grad():
        for x, y in train_loader:
            if n >= args.max_train:
                break
            x = x.view(-1, 784).to(device)
            mu, sigma_sq = encode_batch(model, x)
            storage.add_batch(mu, sigma_sq, y)
            l2_index.add_batch(mu, y)
            n += x.size(0)

    in_nwf_dist = compute_scores_nwf_dist(storage, model, in_loader, device)
    ood_nwf_dist = compute_scores_nwf_dist(storage, model, ood_loader, device)
    auc_nwf_dist = auc_roc(in_nwf_dist, ood_nwf_dist)

    in_nwf_phi = compute_scores_nwf_potential(storage, model, in_loader, device)
    ood_nwf_phi = compute_scores_nwf_potential(storage, model, ood_loader, device)
    auc_nwf_phi = auc_roc(in_nwf_phi, ood_nwf_phi)

    in_l2 = compute_scores_l2(l2_index, model, in_loader, device)
    ood_l2 = compute_scores_l2(l2_index, model, ood_loader, device)
    auc_l2 = auc_roc(in_l2, ood_l2)

    results = {
        "NWF_Mahalanobis": {"auc_ood": auc_nwf_dist},
        "NWF_Potential": {"auc_ood": auc_nwf_phi},
        "FAISS_L2": {"auc_ood": auc_l2},
        "_meta": {"n_in": len(in_nwf_dist), "n_ood": len(ood_nwf_dist)},
    }

    print(f"NWF (Mahalanobis) OOD AUC = {auc_nwf_dist:.4f}")
    print(f"NWF (Potential) OOD AUC = {auc_nwf_phi:.4f}")
    print(f"FAISS L2 OOD AUC = {auc_l2:.4f}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "04_ood.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    try:
        from sklearn.metrics import roc_curve
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 6))
        labels = np.concatenate([np.zeros(len(in_nwf_dist)), np.ones(len(ood_nwf_dist))])
        auc_map = {"NWF_Mahalanobis": auc_nwf_dist, "NWF_Potential": auc_nwf_phi, "FAISS_L2": auc_l2}
        for name, in_s, ood_s, key in [
            ("NWF Mahalanobis", in_nwf_dist, ood_nwf_dist, "NWF_Mahalanobis"),
            ("NWF Potential", in_nwf_phi, ood_nwf_phi, "NWF_Potential"),
            ("FAISS L2", in_l2, ood_l2, "FAISS_L2"),
        ]:
            scores = np.concatenate([in_s, ood_s])
            fpr, tpr, _ = roc_curve(labels, scores)
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc_map[key]:.3f})")
        ax.plot([0, 1], [0, 1], "k--")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title("OOD detection ROC")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "04_ood_roc.png", dpi=120)
        plt.close()
        print(f"ROC plot saved to {out_dir / '04_ood_roc.png'}")
    except Exception as e:
        print(f"ROC plot skipped: {e}")
    print(f"Saved {out_dir / '04_ood.json'}")


if __name__ == "__main__":
    main()
