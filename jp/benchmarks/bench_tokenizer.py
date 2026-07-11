"""トークナイザー correctness + throughput."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from llm_math.experiments.benchmarks import run_named_benchmark

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def main() -> None:
    print("=== トークナイザー correctness + throughput ===")
    run_named_benchmark("tokenizer", RESULTS_DIR, lang="jp")


if __name__ == "__main__":
    main()
