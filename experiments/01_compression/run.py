# -*- coding: utf-8 -*-
"""Эксперимент H1: семантическое сжатие. NWF vs FAISS vs HDC."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

from nwf import VAEEncoder, NWFStorage, get_mnist, encode_batch
from nwf.baselines import L2Index, HDCIndex
from nwf.kalman import KalmanEncoder

# Размеры в байтах (float32=4, MNIST raw=784)
BYTES_ORIGINAL = 784
BYTES_FAISS = 64 * 4  # z only
BYTES_NWF = 128 * 4   # z + sigma_sq
BYTES_HDC = 2000 // 8  # 2000 bits (dim=2000)


def eval_nwf(storage, model, loader, device, k, max_test, metric="mahalanobis", kenc=None):
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            if total >= max_test:
                break
            x = x.view(-1, 784).to(device)
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


def eval_l2(index: L2Index, model, loader, device, k, max_test):
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            if total >= max_test:
                break
            x = x.view(-1, 784).to(device)
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


def eval_pq(pq_index, model, loader, device, k, max_test):
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            if total >= max_test:
                break
            x = x.view(-1, 784).to(device)
            mu, sigma_sq = encode_batch(model, x)
            for i in range(mu.size(0)):
                idx, _ = pq_index.search(
                    mu[i].cpu().numpy().astype(np.float32),
                    sigma_sq[i].cpu().numpy().astype(np.float32),
                    k=k,
                )
                labels = pq_index.get_labels(idx)
                pred = int(np.bincount(labels.astype(int)).argmax())
                if pred == y[i].item():
                    correct += 1
                total += 1
    return correct / total if total > 0 else 0.0


def eval_hdc(index: HDCIndex, loader, device, k, max_test):
    correct, total = 0, 0
    for x, y in loader:
        if total >= max_test:
            break
        x = x.view(-1, 784)
        codes = index.encoder.encode_torch(x)
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
    parser.add_argument("--max_train", type=int, default=10000)
    parser.add_argument("--max_test", type=int, default=2000)
    parser.add_argument("--out", type=str, default="results")
    parser.add_argument("--plot", action="store_true", help="Save comparison plot")
    parser.add_argument("--use_kalman", action="store_true", help="Use Kalman encoding for NWF")
    parser.add_argument("--kalman_iters", type=int, default=3)
    parser.add_argument("--metric", type=str, default="mahalanobis",
                        choices=["mahalanobis", "symmetric", "euclidean"])
    parser.add_argument("--skip_nwf", action="store_true", help="Skip NWF, run only FAISS/HDC")
    parser.add_argument("--skip_faiss", action="store_true", help="Skip FAISS evaluation")
    parser.add_argument("--use_pq", action="store_true", help="Evaluate NWF with PQ compression")
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("metric=%s, use_kalman=%s", args.metric, args.use_kalman)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = VAEEncoder(input_dim=784, hidden_dims=(512, 256), latent_dim=ckpt["latent_dim"])
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    kenc = KalmanEncoder(model, r_noise=0.1, device=device) if args.use_kalman else None
    train_loader, _, test_loader = get_mnist(batch_size=128, download=True)

    # Build indices
    n_train = 0
    storage = NWFStorage(device=device)
    l2_index = L2Index()
    hdc_index = HDCIndex(dim=2000, seed=42)

    with torch.no_grad():
        for x, y in train_loader:
            if n_train >= args.max_train:
                break
            x = x.view(-1, 784).to(device)
            if kenc is not None:
                mu, sigma_sq = kenc.encode_static_batch(x, n_iter=args.kalman_iters)
            else:
                mu, sigma_sq = encode_batch(model, x)
            storage.add_batch(mu, sigma_sq, y)
            l2_index.add_batch(mu, y)
            hdc_index.add_batch(x, y)
            n_train += x.size(0)

    n_indexed = len(storage)
    log.info("Indexed %d samples", n_indexed)

    results = {}
    if not args.skip_nwf:
        precision_nwf = eval_nwf(storage, model, test_loader, device, args.k, args.max_test,
                                 metric=args.metric, kenc=kenc)
        key = f"NWF_{args.metric.capitalize()}"
        results[key] = {"precision_at_k": precision_nwf, "bytes_per_obj": BYTES_NWF}
        log.info("%s Precision@%d = %.4f", key, args.k, precision_nwf)

        precision_nwf_euc = eval_nwf(storage, model, test_loader, device, args.k, args.max_test,
                                     metric="euclidean", kenc=kenc)
        results["NWF_Euclidean"] = {"precision_at_k": precision_nwf_euc, "bytes_per_obj": BYTES_NWF}
        log.info("NWF_Euclidean Precision@%d = %.4f", args.k, precision_nwf_euc)
    else:
        precision_nwf = precision_nwf_euc = 0.0

    if not args.skip_faiss:
        precision_faiss = eval_l2(l2_index, model, test_loader, device, args.k, args.max_test)
        results["FAISS_L2"] = {"precision_at_k": precision_faiss, "bytes_per_obj": BYTES_FAISS}
        log.info("FAISS_L2 Precision@%d = %.4f", args.k, precision_faiss)

    precision_hdc = eval_hdc(hdc_index, test_loader, device, args.k, args.max_test)
    results["HDC"] = {"precision_at_k": precision_hdc, "bytes_per_obj": BYTES_HDC}
    log.info("HDC Precision@%d = %.4f", args.k, precision_hdc)

    if args.use_pq and not args.skip_nwf:
        try:
            from nwf.pq_index import PQChargeIndex, pack_charge
            Z = np.stack([storage._z[i].cpu().numpy() for i in range(len(storage))], axis=0)
            S = np.stack([storage._sigma_sq[i].cpu().numpy() for i in range(len(storage))], axis=0)
            L = [storage._labels[i] for i in range(len(storage))]
            d = Z.shape[1]
            pq_index = PQChargeIndex(d=d, m=16, nbits=8)
            pq_index.fit_and_add(Z, S, L)
            precision_pq = eval_pq(pq_index, model, test_loader, device, args.k, args.max_test)
            bytes_pq = pq_index.bytes_per_vector
            results["NWF_PQ"] = {"precision_at_k": precision_pq, "bytes_per_obj": bytes_pq}
            log.info("NWF_PQ Precision@%d = %.4f, bytes=%d", args.k, precision_pq, bytes_pq)
        except ImportError as e:
            log.warning("PQ unavailable: %s", e)

    # Compression ratio vs original
    for k, v in results.items():
        v["compression_ratio"] = BYTES_ORIGINAL / v["bytes_per_obj"]

    results["_meta"] = {
        "n_indexed": n_indexed, "n_test": min(args.max_test, 10000), "k": args.k,
        "use_kalman": args.use_kalman, "metric": args.metric,
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = "01_compression_symmetric.json" if args.metric == "symmetric" else "01_compression.json"
    with open(out_dir / out_name, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Summary table
    names = [k for k in results if not k.startswith("_")]
    log.info("--- Summary ---")
    for name in names:
        r = results.get(name, {})
        p = r.get("precision_at_k", 0)
        b = r.get("bytes_per_obj", 0)
        c = BYTES_ORIGINAL / b if b else 0
        r["compression_ratio"] = c
        log.info("%s: Precision=%.4f, bytes=%s, ratio=%.1fx", name, p, b, c)

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            precs = [results[k].get("precision_at_k", 0) for k in names]
            colors = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c"][:len(names)]
            bars = ax.bar(names, precs, color=colors)
            ax.set_ylabel("Precision@10")
            ax.set_title("Compression: NWF vs Baselines")
            for b, p in zip(bars, precs):
                ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01, f"{p:.3f}", ha="center", fontsize=10)
            plt.tight_layout()
            plt.savefig(out_dir / "01_compression.png", dpi=120)
            plt.close()
            print(f"Plot saved to {out_dir / '01_compression.png'}")
        except Exception as e:
            print(f"Plot skipped: {e}")

    log.info("Saved %s", out_dir / out_name)


if __name__ == "__main__":
    main()
