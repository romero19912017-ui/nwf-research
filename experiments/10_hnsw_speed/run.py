# -*- coding: utf-8 -*-
"""Сравнение скорости поиска: HNSW (white-transform) vs полный перебор NWF."""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from nwf import VAEEncoder, NWFStorage, get_mnist, encode_batch

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/vae_mnist.pt")
    parser.add_argument("--max_train", type=int, default=10000)
    parser.add_argument("--max_test", type=int, default=500)
    parser.add_argument("--out", type=str, default="results")
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    d = ckpt["latent_dim"]
    model = VAEEncoder(input_dim=784, hidden_dims=(512, 256), latent_dim=d)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    train_loader, _, test_loader = get_mnist(batch_size=128, download=True)
    storage = NWFStorage(device=device)
    has_hnsw = False
    try:
        from nwf.hnsw_index import HNSWMahalanobisIndex
        hnsw = HNSWMahalanobisIndex(d=d, n_links=32)
        has_hnsw = True
    except ImportError as e:
        log.warning("HNSW unavailable: %s", e)
        hnsw = None

    n = 0
    with torch.no_grad():
        for x, y in train_loader:
            if n >= args.max_train:
                break
            x = x.view(-1, 784).to(device)
            mu, sigma_sq = encode_batch(model, x)
            storage.add_batch(mu, sigma_sq, y)
            if hnsw is not None:
                hnsw.add_batch(mu, sigma_sq, y)
            n += x.size(0)

    if hnsw is not None:
        hnsw.build_index()
    log.info("Index size: %d", len(storage))

    correct_brute, correct_hnsw = 0, 0
    total = 0
    t_brute, t_hnsw = 0.0, 0.0
    k = 10

    with torch.no_grad():
        for x, y in test_loader:
            if total >= args.max_test:
                break
            x = x.view(-1, 784).to(device)
            mu, sigma_sq = encode_batch(model, x)
            for i in range(mu.size(0)):
                qz = mu[i]
                qs = sigma_sq[i]
                qz_np = qz.cpu().numpy().astype(np.float32)
                qs_np = qs.cpu().numpy().astype(np.float32)
                target = y[i].item()

                t0 = time.perf_counter()
                idx_b, dist_b = storage.search(qz, qs, k=k, metric="symmetric")
                t_brute += time.perf_counter() - t0
                labs = storage.get_labels(idx_b)
                pred_b = int(labs.mode().values.item())
                if pred_b == target:
                    correct_brute += 1

                if has_hnsw:
                    t0 = time.perf_counter()
                    idx_h, _ = hnsw.search(qz_np, qs_np, k=k)
                    t_hnsw += time.perf_counter() - t0
                    labs_h = hnsw.get_labels(idx_h)
                    pred_h = int(np.bincount(labs_h.astype(int)).argmax())
                    if pred_h == target:
                        correct_hnsw += 1
                total += 1

    prec_brute = correct_brute / total if total > 0 else 0
    prec_hnsw = correct_hnsw / total if has_hnsw and total > 0 else 0
    lat_brute_ms = (t_brute / total) * 1000 if total > 0 else 0
    lat_hnsw_ms = (t_hnsw / total) * 1000 if has_hnsw and total > 0 else 0
    speedup = t_brute / t_hnsw if has_hnsw and t_hnsw > 0 else 0

    results = {
        "brute_force": {"precision_at_10": prec_brute, "latency_ms_per_query": lat_brute_ms},
        "hnsw": {"precision_at_10": prec_hnsw, "latency_ms_per_query": lat_hnsw_ms, "speedup": speedup} if has_hnsw else None,
        "_meta": {"n_index": len(storage), "n_queries": total},
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "10_hnsw_speed.json", "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in results.items() if v is not None}, f, indent=2)

    log.info("Brute: Precision@10=%.4f, latency=%.3f ms/query", prec_brute, lat_brute_ms)
    if has_hnsw:
        log.info("HNSW:  Precision@10=%.4f, latency=%.3f ms/query, speedup=%.1fx", prec_hnsw, lat_hnsw_ms, speedup)
    log.info("Saved %s", out_dir / "10_hnsw_speed.json")


if __name__ == "__main__":
    main()
