"""Benchmark script.

Usage:
    python benchmarks/bench_<name>.py
    # or from a notebook:
    # %run benchmarks/bench_<name>.py

Output: Markdown text + Graph PNG (benchmarks/results/ text text)
"""
import sys
import os
from pathlib import Path

# Add the repository root to the path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Font setup (when available)
try:
    for p in ['/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf',
              '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']:
        try: fm.fontManager.addfont(p)
        except: pass
    plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'Nanum Gothic', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass

from llm_math.bench import time_fn, format_results_table, get_device

RESULTS_DIR = Path(__file__).resolve().parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

# === Bench: SFT vs RLHF Training ===

def main():
    print("=== SFT vs RLHF Training ===")
    print("Run this benchmark from the corresponding chapter notebook.")
    print("Run it directly in the notebook for more detailed results and visualizations.")
    print()
    print("Recommended steps:")
    print("  1. Open the corresponding chapter notebook in Colab")
    print("  2. Switch the runtime to GPU")
    print("  3. Run the benchmark cell in the notebook")
    print()
    # Simple demo
    print("text: Simple matrix multiplication timing")
    n = 1024
    A = torch.randn(n, n); B = torch.randn(n, n)
    res = time_fn(lambda A, B: A @ B, A, B, device='cpu', warmup=2, repeat=3)
    print(f"  n={n} Matrix Multiplication: {res['mean_ms']:.3f} ms")


if __name__ == '__main__':
    main()
