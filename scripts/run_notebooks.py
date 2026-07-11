#!/usr/bin/env python3
"""Execute published notebooks without modifying source files."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def selected_notebooks(root: Path, languages: list[str], kinds: list[str]) -> list[Path]:
    paths: list[Path] = []
    for language in languages:
        for kind in kinds:
            paths.extend(sorted((root / language / kind).glob("*.ipynb")))
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="ko", help="Comma-separated: ko,en,jp")
    parser.add_argument("--kind", default="notebooks,solutions")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--match", help="Only run paths containing this substring")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    paths = selected_notebooks(root, args.lang.split(","), args.kind.split(","))
    if args.match:
        paths = [path for path in paths if args.match in str(path.relative_to(root))]
    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="llm-math-notebooks-") as output_dir:
        for path in paths:
            try:
                notebook = nbformat.read(path, as_version=4)
                execution_dir = Path(output_dir) / path.parent.parent.name / path.parent.name
                execution_dir.mkdir(parents=True, exist_ok=True)
                (execution_dir.parent / "figures").mkdir(exist_ok=True)
                client = NotebookClient(
                    notebook,
                    timeout=args.timeout,
                    kernel_name="python3",
                    resources={"metadata": {"path": str(execution_dir)}},
                )
                client.execute()
                nbformat.write(notebook, Path(output_dir) / f"{path.parent.parent.name}-{path.name}")
                print(f"PASS {path.relative_to(root)}")
            except Exception as exc:  # report all notebook failures in one run
                failures.append(f"FAIL {path.relative_to(root)}: {type(exc).__name__}: {exc}")
                print(failures[-1])
    if failures:
        print(f"FAILED: {len(failures)}/{len(paths)} notebooks")
        return 1
    print(f"OK: executed {len(paths)} notebooks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
