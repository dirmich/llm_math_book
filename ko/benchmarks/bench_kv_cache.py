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

# === Bench: KV Cache on/off ===

def main():
    print("=== KV Cache 벤치마크 ===")
    n_decode = 30
    results = []
    print(f"{'n_ctx':>8} | {'No Cache (ms)':>15} | {'Cache (ms)':>12} | {'Speedup':>10}")
    print("-" * 55)
    for n_ctx in [128, 512, 1024, 2048]:
        d = 64
        # No cache
        Q = torch.randn(1, n_ctx, d); K = torch.randn(1, n_ctx, d); V = torch.randn(1, n_ctx, d)
        def no_cache():
            for _ in range(n_decode):
                Qn = torch.randn(1, 1, d)
                Qc = torch.cat([Q, Qn], dim=1)
                Kc = torch.cat([K, torch.randn(1, 1, d)], dim=1)
                Vc = torch.cat([V, torch.randn(1, 1, d)], dim=1)
                _ = F.softmax(Qc @ Kc.transpose(-1, -2) / (d ** 0.5), dim=-1) @ Vc
        # With cache
        def with_cache():
            kc = torch.randn(1, n_ctx, d); vc = torch.randn(1, n_ctx, d)
            for _ in range(n_decode):
                qn = torch.randn(1, 1, d)
                kc = torch.cat([kc, torch.randn(1, 1, d)], dim=1)
                vc = torch.cat([vc, torch.randn(1, 1, d)], dim=1)
                _ = F.softmax(qn @ kc.transpose(-1, -2) / (d ** 0.5), dim=-1) @ vc

        t_no = time_fn(no_cache, device='cpu', warmup=1, repeat=2)['mean_ms']
        t_yes = time_fn(with_cache, device='cpu', warmup=1, repeat=2)['mean_ms']
        print(f"{n_ctx:>8} | {t_no:>15.3f} | {t_yes:>12.3f} | {t_no/t_yes:>9.2f}x")
        results.append({'n_ctx': n_ctx, 'no': t_no, 'yes': t_yes})

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot([r['n_ctx'] for r in results], [r['no'] for r in results], 'o-', label='No Cache', linewidth=2)
    ax.plot([r['n_ctx'] for r in results], [r['yes'] for r in results], 's-', label='With Cache', linewidth=2)
    ax.set_xlabel('컨텍스트 길이'); ax.set_ylabel('시간 (ms)')
    ax.set_title('KV Cache 효과'); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'bench_kv_cache.png', dpi=100, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
