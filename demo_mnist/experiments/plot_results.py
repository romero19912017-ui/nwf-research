# -*- coding: utf-8 -*-
"""Построение графиков по результатам экспериментов.

Инкрементальность (сравнение NWF vs MLP), обучение с малым числом примеров,
устойчивость к шуму. Использует JSON и CSV из results/.
"""
import argparse
import csv
import json
import pathlib
import sys

path_root = pathlib.Path(__file__).resolve().parent.parent
if str(path_root) not in sys.path:
    sys.path.insert(0, str(path_root))

import matplotlib.pyplot as plt
import numpy as np


def plot_incremental(results_dir="results"):
    results_dir = pathlib.Path(results_dir)
    nwf_path = results_dir / "incremental_nwf.json"
    mlp_path = results_dir / "incremental_mlp.json"
    if not nwf_path.exists() or not mlp_path.exists():
        print(f"Missing {nwf_path} or {mlp_path}")
        return

    with open(nwf_path, encoding="utf-8") as f:
        nwf = json.load(f)
    with open(mlp_path, encoding="utf-8") as f:
        mlp = json.load(f)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(2)
    width = 0.35

    nwf_vals = [nwf["acc_old_only"], nwf["acc_old_after_add_new"]]
    mlp_vals = [mlp["acc_old_only"], mlp["acc_old_after_add_new"]]
    labels = ["Old classes only", "After adding new classes"]

    ax.bar(x - width/2, nwf_vals, width, label="NWF")
    ax.bar(x + width/2, mlp_vals, width, label="MLP")
    ax.set_ylabel("Accuracy on old classes")
    ax.set_title("Incremental learning: no catastrophic forgetting (NWF)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(results_dir / "incremental_comparison.png", dpi=150)
    plt.close()
    print(f"Saved {results_dir / 'incremental_comparison.png'}")


def plot_fewshot(csv_path="results/fewshot.csv", output_path="results/fewshot.png"):
    csv_path = pathlib.Path(csv_path)
    if not csv_path.exists():
        print(f"Missing {csv_path}")
        return

    data = {}
    with open(csv_path, encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            k = int(row["K"])
            m = row["method"]
            if k not in data:
                data[k] = {}
            data[k][m] = (float(row["mean"]), float(row["std"]))

    K_values = sorted(data.keys())
    methods = ["NWF", "1-NN", "LogReg", "MLP"]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(K_values))
    width = 0.2

    for i, m in enumerate(methods):
        means = [data[k][m][0] for k in K_values]
        stds = [data[k][m][1] for k in K_values]
        ax.bar(x + (i - 1.5) * width, means, width, yerr=stds, label=m, capsize=3)

    ax.set_ylabel("Accuracy")
    ax.set_xlabel("K (examples per class)")
    ax.set_title("Few-shot: accuracy vs K")
    ax.set_xticks(x)
    ax.set_xticklabels(K_values)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved {output_path}")


def plot_noise(csv_path="results/noise_robustness.csv", output_path="results/noise_robustness.png", title=None):
    csv_path = pathlib.Path(csv_path)
    if not csv_path.exists():
        print(f"Missing {csv_path}")
        return

    with open(csv_path, encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = list(r)
    if not rows:
        return

    # Support both "noise_std" and first column for x-axis
    first_row = rows[0]
    noise_key = "noise_std" if "noise_std" in first_row else next(iter(first_row.keys()))
    noise = [float(row[noise_key]) for row in rows]
    methods = ["NWF", "1-NN", "LogReg", "MLP"]
    col_map = {}
    for m in methods:
        col_map[m] = "noise" if (m == "NWF" and "NWF" not in first_row and "noise" in first_row) else m
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in methods:
        col = col_map.get(m, m)
        if col not in first_row:
            continue
        vals = [float(row[col]) for row in rows]
        ax.plot(noise, vals, "o-", label=m, linewidth=2, markersize=6)
    ax.set_xlabel("Noise std (added to test images)")
    ax.set_ylabel("Accuracy")
    ax.set_title(title or "Noise robustness")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved {output_path}")


def plot_incremental_aggressive(results_dir="results"):
    results_dir = pathlib.Path(results_dir)
    path = results_dir / "incremental_aggr.json"
    if not path.exists():
        print(f"Missing {path}")
        return
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    nwf = data["nwf"]
    mlp = data["mlp"]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(2)
    width = 0.35
    nwf_vals = [nwf["acc_old_only"], nwf["acc_old_after"]]
    mlp_vals = [mlp["acc_old_only"], mlp["acc_old_after"]]
    labels = ["Old only", "After adding new (MLP finetuned only on new)"]
    ax.bar(x - width/2, nwf_vals, width, label="NWF", color="C0")
    ax.bar(x + width/2, mlp_vals, width, label="MLP", color="C1")
    for i, (v1, v2) in enumerate(zip(nwf_vals, mlp_vals)):
        ax.text(i - width/2, max(v1, 0.05) + 0.02, f"{v1:.0%}", ha="center", fontsize=9)
        ax.text(i + width/2, max(v2, 0.05) + 0.02, f"{v2:.0%}", ha="center", fontsize=9)
    ax.set_ylabel("Accuracy on old classes")
    ax.set_title("Incremental learning: NWF 63.5% vs MLP 0.4% on old classes (160x gap)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(results_dir / "incremental_aggr.png", dpi=150)
    plt.close()
    print(f"Saved {results_dir / 'incremental_aggr.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--all", action="store_true", help="Plot all")
    parser.add_argument("--aggressive", action="store_true", help="Plot aggressive experiments only")
    parser.add_argument("--incremental", action="store_true")
    parser.add_argument("--fewshot", action="store_true")
    parser.add_argument("--noise", action="store_true")
    args = parser.parse_args()

    rd = pathlib.Path(args.results_dir)
    if args.aggressive:
        plot_incremental_aggressive(rd)
        plot_fewshot(rd / "fewshot_raw.csv", rd / "fewshot_raw.png")
        plot_noise(rd / "noise_aggr.csv", rd / "noise_aggr.png", title="Noise (raw): NWF 54.8% vs MLP 20.2% at std=1.0")
        return
    if args.all or args.incremental:
        plot_incremental(rd)
        plot_incremental_aggressive(rd)
    if args.all or args.fewshot:
        plot_fewshot(rd / "fewshot.csv", rd / "fewshot.png")
        plot_fewshot(rd / "fewshot_raw.csv", rd / "fewshot_raw.png")
    if args.all or args.noise:
        plot_noise(rd / "noise_robustness.csv", rd / "noise_robustness.png")
        plot_noise(rd / "noise_aggr.csv", rd / "noise_aggr.png", title="Noise (raw): NWF 54.8% vs MLP 20.2% at std=1.0")


if __name__ == "__main__":
    main()
