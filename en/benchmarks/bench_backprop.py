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

import torch.nn as nn

# === Bench 2: Backpropagation Time (By Model Size) ===
def make_mlp_step(in_d, hid, out_d, n_layers, device='cpu'):
    dims = [in_d] + [hid] * (n_layers - 1) + [out_d]
    layers = [nn.Linear(dims[i], dims[i+1]) for i in range(len(dims) - 1)]
    model = nn.Sequential(*layers).to(device)
    x = torch.randn(32, in_d, device=device)
    y = torch.randn(32, out_d, device=device)
    loss_fn = nn.MSELoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    def step():
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    return step

def main():
    print("=== Bench 2: Backpropagation Time ===\n")
    configs = [
        ('small',  100, 64, 10, 3),
        ('medium', 100, 256, 10, 5),
        ('large',  100, 512, 10, 10),
        ('xlarge', 100, 1024, 10, 20),
    ]
    print(f"{'Config':>8} | {'Params':>10} | {'CPU (ms)':>12} | {'GPU (ms)':>12}")
    print("-" * 50)
    results = []
    for name, id_, hid, od, nl in configs:
        step_cpu = make_mlp_step(id_, hid, od, nl, 'cpu')
        n_params = sum(p.numel() for p in step_cpu.__closure__[0].cell_contents.parameters()) if False else hid * hid * nl
        res_cpu = time_fn(step_cpu, device='cpu', warmup=2, repeat=5)
        if torch.cuda.is_available():
            step_gpu = make_mlp_step(id_, hid, od, nl, 'cuda')
            res_gpu = time_fn(step_gpu, device='cuda', warmup=3, repeat=5)
            print(f"{name:>8} | {n_params:>10,} | {res_cpu['mean_ms']:>12.3f} | {res_gpu['mean_ms']:>12.3f}")
            results.append({'name': name, 'cpu': res_cpu['mean_ms'], 'gpu': res_gpu['mean_ms']})
        else:
            print(f"{name:>8} | {n_params:>10,} | {res_cpu['mean_ms']:>12.3f} | {'N/A':>12}")
            results.append({'name': name, 'cpu': res_cpu['mean_ms'], 'gpu': None})

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(results))
    cpu_vals = [r['cpu'] for r in results]
    ax.bar([i - 0.2 for i in x], cpu_vals, 0.4, label='CPU')
    if results[0]['gpu'] is not None:
        gpu_vals = [r['gpu'] for r in results]
        ax.bar([i + 0.2 for i in x], gpu_vals, 0.4, label='GPU')
    ax.set_xticks(list(x)); ax.set_xticklabels([r['name'] for r in results])
    ax.set_ylabel('Time (ms)'); ax.set_title('Backpropagation Time Comparison')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'bench_backprop.png', dpi=100, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
