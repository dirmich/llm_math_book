#!/usr/bin/env python3
"""Validate notebook JSON and compile Python code cells."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


def iter_notebooks(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.ipynb") if ".ipynb_checkpoints" not in p.parts)


def source_text(source: str | list[str]) -> str:
    return "".join(source) if isinstance(source, list) else source


def check_notebook(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"{path}: invalid notebook JSON: {exc}"]

    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = source_text(cell.get("source", ""))
        try:
            ast.parse(source, filename=f"{path}:cell-{index}")
        except SyntaxError as exc:
            errors.append(f"{path}: cell {index}: {exc.msg} at line {exc.lineno}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", type=Path, default=[Path(".")])
    args = parser.parse_args()

    notebooks: list[Path] = []
    for path in args.paths:
        notebooks.extend([path] if path.suffix == ".ipynb" else iter_notebooks(path))

    unique_notebooks = sorted(set(notebooks))
    errors: list[str] = []
    for notebook in unique_notebooks:
        errors.extend(check_notebook(notebook))

    if errors:
        print("\n".join(errors))
        return 1
    print(f"OK: checked {len(unique_notebooks)} notebooks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
