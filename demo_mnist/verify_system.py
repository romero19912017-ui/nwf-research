# -*- coding: utf-8 -*-
"""System verification for NWF experiment (PyTorch, Windows GPU)."""
import sys

print("=" * 50)
print("SYSTEM VERIFICATION FOR NWF EXPERIMENT (PyTorch)")
print("=" * 50)

# --- 1. PyTorch & CUDA ---
print("\n[1] PyTorch & CUDA:")
try:
    import torch

    print(f"   PyTorch: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print("   [OK] PyTorch GPU check PASSED")
    else:
        print("   [OK] PyTorch CPU mode")
except Exception as e:
    print(f"   [FAIL] PyTorch: {e}")
    sys.exit(1)

# --- 2. torchvision ---
print("\n[2] torchvision:")
try:
    import torchvision

    print(f"   Version: {torchvision.__version__}")
    print("   [OK] PASSED")
except Exception as e:
    print(f"   [FAIL] {e}")
    sys.exit(1)

# --- 3. MNIST load ---
print("\n[3] MNIST dataset:")
try:
    from torchvision import datasets, transforms

    t = transforms.ToTensor()
    ds = datasets.MNIST(root="./data", train=True, download=True, transform=t)
    print(f"   Train samples: {len(ds)}")
    print("   [OK] PASSED")
except Exception as e:
    print(f"   [FAIL] {e}")
    sys.exit(1)

# --- 4. sklearn ---
print("\n[4] scikit-learn:")
try:
    import sklearn

    print(f"   Version: {sklearn.__version__}")
    print("   [OK] PASSED")
except Exception as e:
    print(f"   [FAIL] {e}")
    sys.exit(1)

print("\n" + "=" * 50)
print("SYSTEM IS READY FOR NWF EXPERIMENT")
print("=" * 50)
