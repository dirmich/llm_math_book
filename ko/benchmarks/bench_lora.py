"""벤치마크 스크립트.

사용법:
    python benchmarks/bench_<name>.py
    # 또는 노트북에서:
    # %run benchmarks/bench_<name>.py

출력: Markdown 표 + 그래프 PNG (benchmarks/results/ 에 저장)
"""
import sys
import os
from pathlib import Path

# 레포 루트를 path에 추가
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정 (가능한 경우)
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

# === Bench: Full FT vs LoRA ===

import torch.nn as nn
import torch.nn.functional as F

def main():
    print("=== Full FT vs LoRA 벤치마크 ===")
    d = 256

    # Full
    model_full = nn.Sequential(nn.Linear(d, d*4), nn.ReLU(), nn.Linear(d*4, d), nn.ReLU(), nn.Linear(d, d))
    opt_full = torch.optim.AdamW(model_full.parameters(), lr=1e-3)

    # LoRA (간소화)
    class LoRAModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(d, d*4); self.fc2 = nn.Linear(d*4, d); self.fc3 = nn.Linear(d, d)
            for p in self.parameters(): p.requires_grad = False
            self.A1 = nn.Parameter(torch.randn(4, d) * 0.01); self.B1 = nn.Parameter(torch.zeros(d*4, 4))
            self.A2 = nn.Parameter(torch.randn(4, d*4) * 0.01); self.B2 = nn.Parameter(torch.zeros(d, 4))
            self.A3 = nn.Parameter(torch.randn(4, d) * 0.01); self.B3 = nn.Parameter(torch.zeros(d, 4))
        def forward(self, x):
            h1 = F.relu(self.fc1(x) + (x @ self.A1.t()) @ self.B1.t() * 2)
            h2 = F.relu(self.fc2(h1) + (h1 @ self.A2.t()) @ self.B2.t() * 2)
            return self.fc3(h2) + (h2 @ self.A3.t()) @ self.B3.t() * 2

    model_lora = LoRAModel()
    opt_lora = torch.optim.AdamW([p for p in model_lora.parameters() if p.requires_grad], lr=1e-3)

    x = torch.randn(8, d); y = torch.randn(8, d)
    loss_fn = nn.MSELoss()

    def step(model, opt):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()

    n_full = sum(p.numel() for p in model_full.parameters() if p.requires_grad)
    n_lora = sum(p.numel() for p in model_lora.parameters() if p.requires_grad)
    print(f"Full FT: {n_full:,} params")
    print(f"LoRA: {n_lora:,} params ({n_lora/n_full*100:.2f}%)")

    t_full = time_fn(step, model_full, opt_full, device='cpu', warmup=2, repeat=5)['mean_ms']
    t_lora = time_fn(step, model_lora, opt_lora, device='cpu', warmup=2, repeat=5)['mean_ms']
    print(f"\nFull FT: {t_full:.3f} ms/step")
    print(f"LoRA: {t_lora:.3f} ms/step")
    print(f"속도 향상: {t_full/t_lora:.2f}x")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(['Full FT', 'LoRA'], [t_full, t_lora], color=['blue', 'red'], alpha=0.7)
    ax.set_ylabel('시간 (ms/step)'); ax.set_title('Full FT vs LoRA 속도')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'bench_lora.png', dpi=100, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
