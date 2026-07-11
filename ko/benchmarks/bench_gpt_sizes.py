"""GPT 모델 크기별 correctness + forward benchmark."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from llm_math.experiments.benchmarks import run_named_benchmark

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def main() -> None:
    print("=== GPT 모델 크기별 correctness + forward benchmark ===")
    run_named_benchmark("gpt_sizes", RESULTS_DIR, lang="ko")


if __name__ == "__main__":
    main()
