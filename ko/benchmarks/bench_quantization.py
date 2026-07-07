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

# === Bench: 정밀도별 추론 ===

import torch.nn as nn

def main():
    print("=== 양자화 벤치마크 ===")
    d = 1024
    model = nn.Linear(d, d)
    x = torch.randn(8, d)

    def infer_fp32():
        with torch.no_grad(): return model(x)

    # FP32
    t_fp32 = time_fn(infer_fp32, device='cpu', warmup=3, repeat=10)['mean_ms']
    print(f"FP32: {t_fp32:.3f} ms")

    # 양자화 (간소화)
    import copy
    q_model = copy.deepcopy(model)
    w = q_model.weight.data
    scale = w.abs().max() / 127.0
    q_model.weight.data = torch.round(w / scale).clamp(-128, 127).float() * scale
    def infer_int8():
        with torch.no_grad(): return q_model(x)
    t_int8 = time_fn(infer_int8, device='cpu', warmup=3, repeat=10)['mean_ms']
    print(f"INT8 (시뮬): {t_int8:.3f} ms")

    # 시각화
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(['FP32', 'INT8 (sim)'], [t_fp32, t_int8], color=['blue', 'red'], alpha=0.7)
    ax.set_ylabel('시간 (ms)'); ax.set_title('정밀도별 추론 속도')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'bench_quantization.png', dpi=100, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
