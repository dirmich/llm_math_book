#!/usr/bin/env python3
"""Validate the complete multilingual public notebook release."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

PLACEHOLDER_RE = re.compile(r"준비 중|해설이 준비 중|in progress|Solutions are in progress|準備中")
ARTIFACT_RE = re.compile(
    r"Vectortext|Matrixtext|VectorText|MatrixText|degreestext|text text|"
    r"Matrix (?:Multiplication )?text|Graph text|LoRA \(text\)"
)
LANGUAGES = ("ko", "en", "jp")
EXPECTED_SOLUTIONS = {f"ch{chapter:02d}_solutions.ipynb" for chapter in range(1, 33)}
NOTEBOOK_RE = re.compile(r"^(ch\d{2})_.+\.ipynb$")
PROBLEM_RE = {
    "ko": re.compile(r"^## 문제 ([1-5])\s*$", re.MULTILINE),
    "en": re.compile(r"^## Problem ([1-5])\s*$", re.MULTILINE),
    "jp": re.compile(r"^## 問題 ([1-5])\s*$", re.MULTILINE),
}


def cell_text(source: str | list[str]) -> str:
    return "".join(source) if isinstance(source, list) else source


def iter_notebooks(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.ipynb") if ".ipynb_checkpoints" not in p.parts)


def check_inventory(root: Path) -> list[str]:
    errors: list[str] = []
    if len(iter_notebooks(root)) != 195:
        errors.append(f"{root}: expected exactly 195 published notebooks")
    notebook_names: dict[str, set[str]] = {}
    for lang in LANGUAGES:
        notebook_dir = root / lang / "notebooks"
        notebook_names[lang] = {path.name for path in notebook_dir.glob("*.ipynb")}
        chapters = [match.group(1) for name in notebook_names[lang] if (match := NOTEBOOK_RE.match(name))]
        expected_chapters = {f"ch{chapter:02d}" for chapter in range(33)}
        if len(notebook_names[lang]) != 33 or set(chapters) != expected_chapters:
            errors.append(f"{notebook_dir}: expected exactly one notebook for each Ch00-Ch32")

        solution_dir = root / lang / "solutions"
        actual_solutions = {path.name for path in solution_dir.glob("*.ipynb")}
        missing = sorted(EXPECTED_SOLUTIONS - actual_solutions)
        unexpected = sorted(actual_solutions - EXPECTED_SOLUTIONS)
        if missing:
            errors.append(f"{solution_dir}: missing notebooks: {', '.join(missing)}")
        if unexpected:
            errors.append(f"{solution_dir}: unexpected notebooks: {', '.join(unexpected)}")
    if len({frozenset(names) for names in notebook_names.values()}) != 1:
        errors.append("exercise notebook filenames differ across ko/en/jp")
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


def check_notebook_structure(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"{path}: invalid notebook JSON: {exc}"]

    cell_ids: list[str] = []
    for index, cell in enumerate(notebook.get("cells", [])):
        cell_id = cell.get("id")
        if not isinstance(cell_id, str) or not cell_id:
            errors.append(f"{path}: cell {index}: missing cell id")
        else:
            cell_ids.append(cell_id)
        if cell.get("cell_type") == "code":
            source = cell_text(cell.get("source", ""))
            try:
                ast.parse(source, filename=f"{path}:cell-{index}")
            except SyntaxError as exc:
                errors.append(f"{path}: cell {index}: invalid Python: {exc.msg}")
            if cell.get("outputs"):
                errors.append(f"{path}: cell {index}: outputs must be cleared")
            if cell.get("execution_count") is not None:
                errors.append(f"{path}: cell {index}: execution_count must be null")
    if len(cell_ids) != len(set(cell_ids)):
        errors.append(f"{path}: duplicate cell ids")
    return errors


def check_new_solutions(root: Path) -> list[str]:
    errors: list[str] = []
    for chapter in range(6, 33):
        code_cells: dict[str, list[str]] = {}
        for lang in LANGUAGES:
            path = root / lang / "solutions" / f"ch{chapter:02d}_solutions.ipynb"
            try:
                notebook = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            markdown = "\n".join(
                cell_text(cell.get("source", ""))
                for cell in notebook.get("cells", [])
                if cell.get("cell_type") == "markdown"
            )
            problems = PROBLEM_RE[lang].findall(markdown)
            if problems != ["1", "2", "3", "4", "5"]:
                errors.append(f"{path}: expected exactly five numbered problems, found {problems}")
            code_cells[lang] = [
                cell_text(cell.get("source", ""))
                for cell in notebook.get("cells", [])
                if cell.get("cell_type") == "code"
            ]
        if len(code_cells) == len(LANGUAGES) and len({tuple(v) for v in code_cells.values()}) != 1:
            errors.append(f"ch{chapter:02d}: code cells differ across ko/en/jp solutions")
    return errors


def check_release(root: Path) -> list[str]:
    errors = check_inventory(root)
    for notebook in iter_notebooks(root):
        try:
            errors.extend(check_notebook_text(notebook))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{notebook}: invalid notebook JSON: {exc}")
    for lang in LANGUAGES:
        for chapter in range(6, 33):
            path = root / lang / "solutions" / f"ch{chapter:02d}_solutions.ipynb"
            errors.extend(check_notebook_structure(path))
    errors.extend(check_new_solutions(root))
    for language in ("en", "jp"):
        for path in sorted((root / language / "benchmarks").glob("*.py")):
            match = ARTIFACT_RE.search(path.read_text(encoding="utf-8"))
            if match:
                errors.append(f"{path}: translation artifact: {match.group(0)}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.root.resolve()

    errors = check_release(root)

    if errors:
        print("\n".join(errors))
        return 1
    print("OK: validated 195 notebooks (96 solutions) across ko/en/jp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
