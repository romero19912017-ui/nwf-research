# -*- coding: utf-8 -*-
"""Интеграционная проверка: запуск экспериментов в тестовом режиме."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SEED = 42
QUICK = {"max_train": 500, "max_test": 200, "n_samples": 50}


def run(cmd: list[str], cwd: Path) -> tuple[bool, str]:
    env = {"PYTHONPATH": str(cwd), "CUDA_VISIBLE_DEVICES": ""}
    try:
        r = subprocess.run(
            [sys.executable] + cmd,
            cwd=cwd,
            env={**__import__("os").environ, **env},
            capture_output=True,
            text=True,
            timeout=300,
        )
        return r.returncode == 0, r.stdout + r.stderr
    except Exception as e:
        return False, str(e)


def main():
    root = Path(__file__).resolve().parent
    results_dir = root / "results"
    results_dir.mkdir(exist_ok=True)

    checks = []

    # 0. Unit tests
    ok, out = run(["-m", "pytest", "tests/", "-v", "--tb=line", "-q"], root)
    checks.append(("pytest", ok, out[:500] if out else ""))

    # 1. Experiment 01
    ok, out = run([
        "experiments/01_compression/run.py",
        "--max_train", "500", "--max_test", "200",
        "--metric", "symmetric",
    ], root)
    prec = None
    if (results_dir / "01_compression_symmetric.json").exists():
        with open(results_dir / "01_compression_symmetric.json") as f:
            d = json.load(f)
        prec = d.get("NWF_Symmetric", {}).get("precision_at_k")
    checks.append(("01_compression", ok and (prec is None or 0.5 < prec < 1.0), str(prec)))

    # 2. Experiment 02
    ok, out = run([
        "experiments/02_noise/run.py",
        "--max_train", "500", "--max_test", "200",
        "--noise_levels", "0,0.2,0.5",
    ], root)
    checks.append(("02_noise", ok, ""))

    # 3. Experiment 03
    ok, out = run([
        "experiments/03_incremental/run.py",
        "--max_per_task", "300", "--epochs", "2",
    ], root)
    acc = None
    if (results_dir / "03_incremental.json").exists():
        with open(results_dir / "03_incremental.json") as f:
            d = json.load(f)
        acc = d.get("NWF", {}).get("acc_after_task", [0])[-1]
    checks.append(("03_incremental", ok and (acc is None or 0.2 < acc < 1.0), str(acc)))

    # 4. Experiment 04
    ok, out = run([
        "experiments/04_ood/run.py",
        "--max_train", "500", "--max_test", "200",
    ], root)
    auc = None
    if (results_dir / "04_ood.json").exists():
        with open(results_dir / "04_ood.json") as f:
            d = json.load(f)
        auc = d.get("NWF_Mahalanobis", {}).get("auc_ood")
    checks.append(("04_ood", ok and (auc is None or auc > 0.5), str(auc)[:6] if auc else ""))

    # 5. Experiment 05 (calibration)
    ok, out = run([
        "experiments/05_calibration/run.py",
        "--max_train", "500", "--max_test", "200",
    ], root)
    ece = None
    if (results_dir / "05_calibration.json").exists():
        with open(results_dir / "05_calibration.json") as f:
            d = json.load(f)
        ece = d.get("ece_test_after", d.get("NWF", {}).get("ece"))
    checks.append(("05_calibration", ok and (ece is None or ece < 0.5), f"ECE={ece:.3f}" if ece is not None else ""))

    # 6. Experiment 07
    ok, out = run(["experiments/07_interpolation/run.py", "--n_pairs", "2", "--save_images"], root)
    checks.append(("07_interpolation", ok, ""))

    # 7. Experiment 08
    ok, out = run([
        "experiments/08_landscape/run.py",
        "--max_samples", "200", "--method", "pca",
    ], root)
    checks.append(("08_landscape", ok, ""))

    # 8. Experiment 09
    ok, out = run([
        "experiments/09_encoding_speed/run.py",
        "--n_samples", "100",
    ], root)
    checks.append(("09_encoding_speed", ok, ""))

    # 9. Experiment 10 (HNSW) - optional if faiss available
    ok, out = run([
        "experiments/10_hnsw_speed/run.py",
        "--max_train", "500", "--max_test", "100",
    ], root)
    checks.append(("10_hnsw_speed", ok, ""))

    # Summary
    print("\n=== Integration check results ===")
    all_ok = True
    for name, passed, extra in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status} {extra}")
        if not passed:
            all_ok = False
    print(f"\nOverall: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
