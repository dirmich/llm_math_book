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

# === Bench: Attention Sequence Lengthtext ===

import torch.nn.functional as F

def main():
    print("=== Attention Benchmark ===")
    sizes = [128, 256, 512, 1024, 2048]
    print(f"{'n':>6} | {'CPU (ms)':>12} | {'GPU (ms)':>12}")
    print("-" * 35)
    results = []
    for n in sizes:
        d = 64
        Q = torch.randn(n, d); K = torch.randn(n, d); V = torch.randn(n, d)
        def attn(Q, K, V):
            return F.softmax(Q @ K.t() / (d ** 0.5), dim=-1) @ V
        t_cpu = time_fn(attn, Q, K, V, device='cpu', warmup=2, repeat=3)['mean_ms']
        t_gpu = None
        if torch.cuda.is_available():
            Qg, Kg, Vg = Q.cuda(), K.cuda(), V.cuda()
            t_gpu = time_fn(attn, Qg, Kg, Vg, device='cuda', warmup=3, repeat=5)['mean_ms']
            print(f"{n:>6} | {t_cpu:>12.3f} | {t_gpu:>12.3f}")
        else:
            print(f"{n:>6} | {t_cpu:>12.3f} | {'N/A':>12}")
        results.append({'n': n, 'cpu': t_cpu, 'gpu': t_gpu})

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot([r['n'] for r in results], [r['cpu'] for r in results], 'o-', label='CPU')
    if results[0]['gpu']:
        ax.plot([r['n'] for r in results], [r['gpu'] for r in results], '^-', label='GPU', linewidth=2)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Sequence Length n'); ax.set_ylabel('Time (ms)')
    ax.set_title('Attention $O(n^2 d)$ Benchmark'); ax.legend(); ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'bench_attention.png', dpi=100, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
