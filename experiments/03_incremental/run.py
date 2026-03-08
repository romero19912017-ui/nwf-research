# -*- coding: utf-8 -*-
"""Эксперимент H3: инкрементальное обучение. NWF vs EWC vs Fine-tuning."""
from __future__ import annotations

import argparse
from typing import Optional
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

from nwf import VAEEncoder, NWFStorage, encode_batch
from nwf.continual import (
    MLPClassifier,
    get_split_mnist_loaders,
    train_task,
    evaluate,
    evaluate_per_class,
    run_icarl,
)

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "mnist"


def get_split_mnist(batch_size: int, task_id: int, max_per_task: Optional[int] = None):
    ranges = [(0, 3), (3, 6), (6, 10)]
    low, high = ranges[task_id]
    tf = transforms.ToTensor()
    ds = MNIST(root=str(DATA_ROOT), train=True, download=True, transform=tf)
    indices = [i for i, (_, y) in enumerate(ds) if low <= y < high]
    if max_per_task:
        indices = indices[:max_per_task]
    return DataLoader(
        torch.utils.data.Subset(ds, indices),
        batch_size=batch_size, shuffle=True, num_workers=0
    )


def get_full_test(batch_size: int):
    tf = transforms.ToTensor()
    ds = MNIST(root=str(DATA_ROOT), train=False, download=True, transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


def eval_nwf_per_class(storage, model, test_loader, device, k):
    """Точность по классам для NWF."""
    correct_per_class, total_per_class = {i: 0 for i in range(10)}, {i: 0 for i in range(10)}
    with torch.no_grad():
        for x, y in test_loader:
            x = x.view(-1, 784).to(device)
            mu, sigma_sq = encode_batch(model, x)
            for i in range(mu.size(0)):
                idx, _ = storage.search(mu[i], sigma_sq[i], k=k)
                labels = storage.get_labels(idx)
                pred = labels.mode().values.item()
                c = y[i].item()
                total_per_class[c] += 1
                if pred == c:
                    correct_per_class[c] += 1
    return {c: correct_per_class[c] / max(total_per_class[c], 1) for c in range(10)}


def run_nwf(args, device):
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = VAEEncoder(input_dim=784, hidden_dims=(512, 256), latent_dim=ckpt["latent_dim"])
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    storage = NWFStorage(device=device)
    test_loader = get_full_test(128)
    accs_after = []
    forgetting_matrix = []

    for task_id in range(3):
        loader = get_split_mnist(128, task_id, args.max_per_task)
        n_added = 0
        with torch.no_grad():
            for x, y in loader:
                if n_added >= args.max_per_task:
                    break
                x = x.view(-1, 784).to(device)
                mu, sigma_sq = encode_batch(model, x)
                storage.add_batch(mu, sigma_sq, y)
                n_added += x.size(0)

        per_class = eval_nwf_per_class(storage, model, test_loader, device, args.k)
        forgetting_matrix.append(per_class)
        acc = sum(per_class.values()) / 10
        accs_after.append(acc)
        print(f"NWF Task {task_id}: acc={acc:.4f}")

    return accs_after, forgetting_matrix


def run_ewc(args, device):
    loaders, test_ds = get_split_mnist_loaders(128, args.max_per_task)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)
    model = MLPClassifier(hidden=(256, 128), num_classes=10).to(device)
    fisher, old_params = None, None
    accs_after = []
    forgetting_matrix = []

    for task_id, loader in enumerate(loaders):
        fisher, old_params = train_task(
            model, loader, device, epochs=args.epochs, lr=args.lr,
            ewc_lambda=args.ewc_lambda, fisher=fisher, old_params=old_params,
        )
        per_class = evaluate_per_class(model, test_loader, device)
        forgetting_matrix.append(per_class)
        acc = evaluate(model, test_loader, device)
        accs_after.append(acc)
        print(f"EWC Task {task_id}: acc={acc:.4f}")

    return accs_after, forgetting_matrix


def run_finetuning(args, device):
    loaders, test_ds = get_split_mnist_loaders(128, args.max_per_task)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)
    model = MLPClassifier(hidden=(256, 128), num_classes=10).to(device)
    accs_after = []
    forgetting_matrix = []

    for task_id, loader in enumerate(loaders):
        train_task(model, loader, device, epochs=args.epochs, lr=args.lr, ewc_lambda=0)
        per_class = evaluate_per_class(model, test_loader, device)
        forgetting_matrix.append(per_class)
        acc = evaluate(model, test_loader, device)
        accs_after.append(acc)
        print(f"Fine-tuning Task {task_id}: acc={acc:.4f}")

    return accs_after, forgetting_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/vae_mnist.pt")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max_per_task", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=5, help="Epochs per task for MLP")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ewc_lambda", type=float, default=1000.0)
    parser.add_argument("--out", type=str, default="results")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    results = {}

    print("--- NWF ---")
    nwf_acc, nwf_fm = run_nwf(args, device)
    results["NWF"] = {"acc_after_task": nwf_acc, "forgetting_matrix": nwf_fm}

    print("\n--- EWC ---")
    ewc_acc, ewc_fm = run_ewc(args, device)
    results["EWC"] = {"acc_after_task": ewc_acc, "forgetting_matrix": ewc_fm}

    print("\n--- Fine-tuning ---")
    ft_acc, ft_fm = run_finetuning(args, device)
    results["Fine_tuning"] = {"acc_after_task": ft_acc, "forgetting_matrix": ft_fm}

    print("\n--- iCaRL ---")
    try:
        loaders, test_ds = get_split_mnist_loaders(128, args.max_per_task)
        icarl_acc, icarl_fm = run_icarl(
            loaders,
            test_ds,
            device,
            exemplars_per_class=min(20, args.max_per_task // 10 or 5),
            epochs=args.epochs,
            lr=args.lr,
        )
        results["iCaRL"] = {"acc_after_task": icarl_acc, "forgetting_matrix": icarl_fm}
    except Exception as e:
        print(f"iCaRL failed: {e}")
        results["iCaRL"] = {"acc_after_task": [0, 0, 0], "forgetting_matrix": []}

    results["_meta"] = {"max_per_task": args.max_per_task, "n_tasks": 3}

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "03_incremental.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n--- Summary ---")
    print(f"{'Method':<15} {'After T1':>10} {'After T2':>10} {'After T3':>10}")
    print("-" * 50)
    for name in ["NWF", "EWC", "Fine_tuning", "iCaRL"]:
        if name in results and "acc_after_task" in results[name]:
            accs = results[name]["acc_after_task"]
            print(f"{name:<15} {accs[0]:>10.4f} {accs[1]:>10.4f} {accs[2]:>10.4f}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            ax = axes[0]
            tasks = [1, 2, 3]
            for name in ["NWF", "EWC", "Fine_tuning", "iCaRL"]:
                if name not in results:
                    continue
                accs = results[name]["acc_after_task"]
                ax.plot(tasks, accs, "o-", label=name, linewidth=2, markersize=8)
            ax.set_xlabel("Tasks completed")
            ax.set_ylabel("Accuracy (all classes)")
            ax.set_title("Incremental learning: NWF vs EWC vs Fine-tuning")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax2 = axes[1]
            fm = np.array([[results["NWF"]["forgetting_matrix"][t].get(c, 0) for c in range(10)]
                          for t in range(3)])
            im = ax2.imshow(fm.T, aspect="auto", cmap="viridis", vmin=0, vmax=1)
            ax2.set_xlabel("Task")
            ax2.set_ylabel("Class")
            ax2.set_title("NWF forgetting matrix")
            plt.colorbar(im, ax=ax2)
            plt.tight_layout()
            plt.savefig(out_dir / "03_incremental.png", dpi=120)
            plt.close()
            print(f"Plot saved to {out_dir / '03_incremental.png'}")
        except Exception as e:
            print(f"Plot skipped: {e}")

    print(f"\nSaved {out_dir / '03_incremental.json'}")


if __name__ == "__main__":
    main()
