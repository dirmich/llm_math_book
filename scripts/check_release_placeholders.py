#!/usr/bin/env python3
"""Fail on unreleased placeholder notebooks and known translation artifacts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PLACEHOLDER_RE = re.compile(r"준비 중|해설이 준비 중|in progress|Solutions are in progress|準備中")
ARTIFACT_RE = re.compile(
    r"Vectortext|Matrixtext|VectorText|MatrixText|degreestext|text text|"
    r"Matrix (?:Multiplication )?text|Graph text|LoRA \(text\)|テキスト"
)


def cell_text(source: str | list[str]) -> str:
    return "".join(source) if isinstance(source, list) else source


def iter_notebooks(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.ipynb") if ".ipynb_checkpoints" not in p.parts)


def check_solution_range(root: Path) -> list[str]:
    errors: list[str] = []
    for lang in ("ko", "en", "jp"):
        for chapter in range(6, 33):
            path = root / lang / "solutions" / f"ch{chapter:02d}_solutions.ipynb"
            if path.exists():
                errors.append(f"{path}: Ch06-32 solution placeholder range must not be published")
    return errors


def check_notebook_text(path: Path) -> list[str]:
    errors: list[str] = []
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook.get("cells", [])):
        text = cell_text(cell.get("source", ""))
        for pattern_name, pattern in (
            ("placeholder", PLACEHOLDER_RE),
            ("translation artifact", ARTIFACT_RE),
        ):
            match = pattern.search(text)
            if match:
                start = max(0, match.start() - 30)
                end = match.end() + 30
                snippet = text[start:end].replace("\n", "\\n")
                errors.append(f"{path}: cell {index}: {pattern_name}: {snippet}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.root.resolve()

    errors = check_solution_range(root)
    for notebook in iter_notebooks(root):
        errors.extend(check_notebook_text(notebook))
    for language in ("en", "jp"):
        for path in sorted((root / language / "benchmarks").glob("*.py")):
            match = ARTIFACT_RE.search(path.read_text(encoding="utf-8"))
            if match:
                errors.append(f"{path}: translation artifact: {match.group(0)}")

    if errors:
        print("\n".join(errors))
        return 1
    print("OK: no release-blocking placeholders found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
