#!/usr/bin/env python
"""Check that the installed PyTorch build supports the current GPU.

Run after creating the conda environment:
    python scripts/check_gpu_compat.py
"""
import sys

try:
    import torch
except ImportError:
    print("PyTorch not installed.")
    sys.exit(1)

if not torch.cuda.is_available():
    print("CUDA not available — running CPU-only.")
    sys.exit(0)

gpu_name = torch.cuda.get_device_name()
cc_major, cc_minor = torch.cuda.get_device_capability()
print(f"GPU: {gpu_name} (sm_{cc_major}{cc_minor})")
print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")

# Smoke-test a conv2d on GPU
try:
    x = torch.randn(1, 3, 8, 8, device="cuda")
    torch.nn.Conv2d(3, 8, 3, padding=1).cuda()(x)
except RuntimeError as e:
    if "engine" in str(e).lower() or "no kernel image" in str(e).lower():
        print(
            f"\n ERROR: PyTorch {torch.__version__} does not include kernels "
            f"for your GPU (sm_{cc_major}{cc_minor}).\n"
        )
        if cc_major < 7:
            print(
                "Your GPU (Pascal / sm_6x) needs PyTorch <= 2.5.\n"
                "Fix:\n"
                "  pip install torch==2.5.1 torchvision==0.20.1 "
                "--index-url https://download.pytorch.org/whl/cu121\n"
            )
        else:
            print(f"Unexpected: sm_{cc_major}{cc_minor} should be supported. "
                  f"Full error:\n{e}")
        sys.exit(1)
    raise

print("✓ GPU compute is working.")
