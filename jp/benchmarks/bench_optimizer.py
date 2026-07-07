"""ベンチマークスクリプト.

使い方:
    python benchmarks/bench_<name>.py
    # またはノートブックから:
    # %run benchmarks/bench_<name>.py

出力: Markdown テキスト + グラフ PNG (benchmarks/results/ テキスト テキスト)
"""
import sys
import os
from pathlib import Path

# リポジトリルートを path に追加
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# フォント設定 (可能な場合)
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

# === Bench 3: テキスト テキスト テキスト 時間 ===
def main():
    print("=== Bench 3: テキスト 比較 ===\n")
    from llm_math.data import load_mnist_small
    X_tr, y_tr, _, _ = load_mnist_small(n_train=2000, n_test=500)
    X_t = torch.tensor(X_tr, dtype=torch.float32)
    y_t = torch.tensor(y_tr, dtype=torch.long)

    def make_model():
        return nn.Sequential(nn.Linear(784, 128), nn.ReLU(),
                             nn.Linear(128, 64), nn.ReLU(),
                             nn.Linear(64, 10))

    opts = {
        'SGD': lambda p: torch.optim.SGD(p, lr=0.1),
        'Momentum': lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9),
        'Adam': lambda p: torch.optim.Adam(p, lr=0.001),
        'AdamW': lambda p: torch.optim.AdamW(p, lr=0.001, weight_decay=0.01),
    }
    loss_fn = nn.CrossEntropyLoss()
    results = {}
    for name, opt_fn in opts.items():
        torch.manual_seed(0)
        model = make_model()
        opt = opt_fn(model.parameters())
        losses = []
        for epoch in range(3):
            idx = torch.randperm(len(X_t))
            for i in range(0, len(X_t), 64):
                bi = idx[i:i+64]
                opt.zero_grad()
                loss = loss_fn(model(X_t[bi]), y_t[bi])
                loss.backward()
                opt.step()
                losses.append(loss.item())
        results[name] = losses
        print(f"{name}: テキスト loss = {losses[-1]:.4f}")

    # 可視化
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, losses in results.items():
        w = 50
        sm = np.convolve(losses, np.ones(w)/w, mode='valid')
        ax.plot(sm, label=name, linewidth=2)
    ax.set_xlabel('Step (テキスト)'); ax.set_ylabel('Loss')
    ax.set_title('テキスト テキスト 比較')
    ax.set_yscale('log'); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'bench_optimizer.png', dpi=100, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
