"""Run the shared, correctness-checked benchmark implementation."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from llm_math.experiments.benchmarks import run_named_benchmark


def main() -> None:
    language = Path(__file__).resolve().parents[1].name
    name = Path(__file__).stem.removeprefix("bench_")
    run_named_benchmark(name, Path(__file__).parent / "results", language)


if __name__ == "__main__":
    main()
