# -*- coding: utf-8 -*-
"""Обучение VAE на MNIST. Сохранение чекпоинта."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.optim as optim

from nwf.data import get_mnist
from nwf.vae_encoder import VAEEncoder
from nwf.inference import vae_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--out", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, _ = get_mnist(batch_size=args.batch_size, download=True)
    model = VAEEncoder(input_dim=784, hidden_dims=(512, 256), latent_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for x, _ in train_loader:
            x = x.view(-1, 784).to(device)
            optimizer.zero_grad()
            recon, mu, log_var = model(x)
            loss = vae_loss(recon, x, mu, log_var)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.view(-1, 784).to(device)
                recon, mu, log_var = model(x)
                val_loss += vae_loss(recon, x, mu, log_var).item()

        n_train = len(train_loader.dataset)
        n_val = len(val_loader.dataset)
        print(f"Epoch {epoch+1}/{args.epochs}  train_loss={train_loss/n_train:.2f}  val_loss={val_loss/n_val:.2f}")

    ckpt = out_dir / "vae_mnist.pt"
    torch.save({"model": model.state_dict(), "latent_dim": args.latent_dim}, ckpt)
    print(f"Saved {ckpt}")


if __name__ == "__main__":
    main()
