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


# === Bench 1: Matrix Multiplication text CPU vs GPU ===
def main():
    print("=== Bench 1: Matrix Multiplication text CPU vs GPU ===\n")
    sizes = [64, 256, 512, 1024, 2048]
    print(f"{'n':>6} | {'NumPy CPU (ms)':>16} | {'Torch CPU (ms)':>16} | {'Torch GPU (ms)':>16}")
    print("-" * 70)

    results = []
    for n in sizes:
        A_np = np.random.randn(n, n).astype(np.float32)
        B_np = np.random.randn(n, n).astype(np.float32)
        t_np = time_fn(lambda A, B: A @ B, A_np, B_np, device='cpu', warmup=2, repeat=3)['mean_ms']
        A_t = torch.from_numpy(A_np); B_t = torch.from_numpy(B_np)
        t_tc = time_fn(lambda A, B: A @ B, A_t, B_t, device='cpu', warmup=2, repeat=3)['mean_ms']
        t_tg = None
        if torch.cuda.is_available():
            A_g = A_t.cuda(); B_g = B_t.cuda()
            t_tg = time_fn(lambda A, B: A @ B, A_g, B_g, device='cuda', warmup=3, repeat=5)['mean_ms']
            print(f"{n:>6} | {t_np:>16.3f} | {t_tc:>16.3f} | {t_tg:>16.3f}")
        else:
            print(f"{n:>6} | {t_np:>16.3f} | {t_tc:>16.3f} | {'N/A':>16}")
        results.append({'n': n, 'numpy_cpu': t_np, 'torch_cpu': t_tc, 'torch_gpu': t_tg})

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot([r['n'] for r in results], [r['numpy_cpu'] for r in results], 'o-', label='NumPy CPU')
    ax.plot([r['n'] for r in results], [r['torch_cpu'] for r in results], 's-', label='PyTorch CPU')
    if results[0]['torch_gpu'] is not None:
        ax.plot([r['n'] for r in results], [r['torch_gpu'] for r in results], '^-', label='PyTorch GPU', linewidth=2)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Matrix text n'); ax.set_ylabel('Time (ms)')
    ax.set_title('Matrix Multiplication Benchmark: $O(n^3)$'); ax.legend(); ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'bench_matmul.png', dpi=100, bbox_inches='tight')
    plt.show()
    print(f"\nGraph text: {RESULTS_DIR / 'bench_matmul.png'}")


if __name__ == '__main__':
    main()
