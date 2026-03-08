# -*- coding: utf-8 -*-
"""Эксперимент H2: устойчивость к шуму. NWF vs FAISS vs HDC."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from nwf import VAEEncoder, NWFStorage, get_mnist, encode_batch
from nwf.baselines import L2Index, HDCIndex
from nwf.kalman import KalmanEncoder

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


def eval_nwf(storage, model, loader, device, k, max_test, sigma, metric, kenc=None, seed=42):
    torch.manual_seed(seed)
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            if total >= max_test:
                break
            x = x.view(-1, 784).to(device)
            if sigma > 0:
                x = x + sigma * torch.randn_like(x)
            if kenc is not None:
                mu, sigma_sq = kenc.encode_static_batch(x, n_iter=3)
            else:
                mu, sigma_sq = encode_batch(model, x)
            for i in range(mu.size(0)):
                if metric == "euclidean":
                    idx, _ = storage.search_euclidean(mu[i], k=k)
                else:
                    idx, _ = storage.search(mu[i], sigma_sq[i], k=k, metric=metric)
                labels = storage.get_labels(idx)
                pred = labels.mode().values.item()
                if pred == y[i].item():
                    correct += 1
                total += 1
    return correct / total if total > 0 else 0.0


def eval_l2(index, model, loader, device, k, max_test, sigma, seed=42):
    torch.manual_seed(seed)
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            if total >= max_test:
                break
            x = x.view(-1, 784).to(device)
            if sigma > 0:
                x = x + sigma * torch.randn_like(x)
            mu, _ = encode_batch(model, x)
            for i in range(mu.size(0)):
                q = mu[i].cpu().numpy()
                idx, _ = index.search(q, k=k)
                labels = index.get_labels(idx)
                pred = int(np.bincount(labels).argmax())
                if pred == y[i].item():
                    correct += 1
                total += 1
    return correct / total if total > 0 else 0.0


def eval_hdc(index, loader, k, max_test, sigma, seed=42):
    rng = np.random.default_rng(seed)
    correct, total = 0, 0
    for x, y in loader:
        if total >= max_test:
            break
        x = x.view(-1, 784).numpy()
        if sigma > 0:
            x = x + sigma * rng.standard_normal(x.shape).astype(np.float32)
        x = np.clip(x, 0, 1)
        codes = index.encoder.encode(x)
        for i in range(codes.shape[0]):
            idx, _ = index.search(codes[i], k=k)
            labels = index.get_labels(idx)
            pred = int(np.bincount(labels).argmax())
            if pred == y[i].item():
                correct += 1
            total += 1
    return correct / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/vae_mnist.pt")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--noise_levels", type=str, default="0,0.1,0.2,0.3,0.5,0.7,1.0")
    parser.add_argument("--max_train", type=int, default=10000)
    parser.add_argument("--max_test", type=int, default=2000)
    parser.add_argument("--out", type=str, default="results")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--use_kalman", action="store_true", help="Kalman-кодирование для NWF запросов")
    parser.add_argument("--metric", type=str, default="symmetric", choices=["mahalanobis", "euclidean", "symmetric"],
                        help="Метрика NWF (для имени файла; все варианты запускаются)")
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = VAEEncoder(input_dim=784, hidden_dims=(512, 256), latent_dim=ckpt["latent_dim"])
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    kenc = KalmanEncoder(model, r_noise=0.1, device=device) if args.use_kalman else None
    log.info("metric=%s, use_kalman=%s", args.metric, args.use_kalman)

    _, _, test_loader = get_mnist(batch_size=128, download=True)
    train_loader, _, _ = get_mnist(batch_size=128, download=True)

    storage = NWFStorage(device=device)
    l2_index = L2Index()
    hdc_index = HDCIndex(dim=2000, seed=42)
    n_train = 0

    with torch.no_grad():
        for x, y in train_loader:
            if n_train >= args.max_train:
                break
            x = x.view(-1, 784).to(device)
            mu, sigma_sq = encode_batch(model, x)
            storage.add_batch(mu, sigma_sq, y)
            l2_index.add_batch(mu, y)
            hdc_index.add_batch(x, y)
            n_train += x.size(0)

    noise_levels = [float(s) for s in args.noise_levels.split(",")]
    results = {"methods": {}, "noise_levels": noise_levels}

    methods = [
        ("NWF_Mahalanobis", lambda s: eval_nwf(storage, model, test_loader, device, args.k, args.max_test, s, "mahalanobis", kenc)),
        ("NWF_Symmetric", lambda s: eval_nwf(storage, model, test_loader, device, args.k, args.max_test, s, "symmetric", kenc)),
        ("NWF_Euclidean", lambda s: eval_nwf(storage, model, test_loader, device, args.k, args.max_test, s, "euclidean", kenc)),
        ("FAISS_L2", lambda s: eval_l2(l2_index, model, test_loader, device, args.k, args.max_test, s)),
        ("HDC", lambda s: eval_hdc(hdc_index, test_loader, args.k, args.max_test, s)),
    ]

    for name, fn in methods:
        results["methods"][name] = {}
        for sigma in noise_levels:
            acc = fn(sigma)
            results["methods"][name][f"sigma_{sigma}"] = acc
            print(f"{name} sigma={sigma}  acc={acc:.4f}")
        print()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"02_noise_{args.metric}" + ("_kalman" if args.use_kalman else "") + ".json"
    with open(out_dir / out_name, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 6))
            for name, data in results["methods"].items():
                accs = [data[f"sigma_{s}"] for s in noise_levels]
                ax.plot(noise_levels, accs, "o-", label=name, linewidth=2, markersize=6)
            ax.set_xlabel("Noise level (sigma)")
            ax.set_ylabel("Accuracy")
            ax.set_title("Noise robustness: NWF vs Baselines")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_dir / "02_noise.png", dpi=120)
            plt.savefig(out_dir / "02_noise_comparison.png", dpi=120)
            plt.close()
            print(f"Plot saved to {out_dir / '02_noise_comparison.png'}")
        except Exception as e:
            print(f"Plot skipped: {e}")

    print(f"Saved {out_dir / out_name}")


if __name__ == "__main__":
    main()
