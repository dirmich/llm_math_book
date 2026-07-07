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

# === Bench: QLoRA 推論 (CPU 4bit vs GPU FP16) ===

def main():
    print("=== QLoRA 推論 (CPU 4bit vs GPU FP16) ===")
    print("このベンチマークは対応する章のノートブックで実行することを推奨します。")
    print("ノートブックで直接実行すると詳細な結果と可視化を確認できます。")
    print()
    print("推奨手順:")
    print("  1. Colab で対応する章のノートブックを開く")
    print("  2. ランタイムを GPU に切り替える")
    print("  3. ノートブックのベンチマークセルを実行")
    print()
    # 簡単なデモ
    print("テキスト: 簡単な行列積の時間測定")
    n = 1024
    A = torch.randn(n, n); B = torch.randn(n, n)
    res = time_fn(lambda A, B: A @ B, A, B, device='cpu', warmup=2, repeat=3)
    print(f"  n={n} 行列積: {res['mean_ms']:.3f} ms")


if __name__ == '__main__':
    main()
