# LLM Math Book Exercises

Public hands-on exercises and experiment code.

Language folders: [English](en/), [Korean](ko/), [Japanese](jp/)

## Contents

- `ko/notebooks/`: Korean runnable notebooks
- `ko/solutions/`: Korean solution notebooks for Ch01-Ch05 only
- `ko/benchmarks/`: Korean CPU/GPU benchmark scripts
- `en/`: English materials, added as they become available
- `jp/`: Japanese materials, added as they become available
- `src/llm_math/`: shared utilities
- `data/`: small practice datasets

## Current Published Scope

This public repository currently publishes notebooks for Ch00-Ch32 and solution
notebooks only for Ch01-Ch05 in each language. The unfinished Ch06-Ch32 solution
placeholders were removed from `ko/solutions/`, `en/solutions/`, and
`jp/solutions/` until complete, chapter-matched solutions are available.

## Quick Start

```bash
pip install -e ".[notebook]"
jupyter lab
```

For development checks:

```bash
pip install -e ".[dev,notebook]"
python -m pytest
ruff check src tests scripts ko/benchmarks en/benchmarks jp/benchmarks
pyright
python scripts/check_notebook_syntax.py
python scripts/check_release_placeholders.py
python scripts/run_notebooks.py --lang ko,en,jp --kind notebooks,solutions
MPLBACKEND=Agg python scripts/run_benchmarks.py --device cpu --assert-correctness
```

On a CUDA host, validate the synchronized GPU kernel against the deterministic CPU reference:

```bash
python scripts/run_benchmarks.py --device cuda --record-environment --assert-correctness
```

For Colab:

```bash
bash colab_setup.sh
```

## License

Code is distributed under the MIT License.
