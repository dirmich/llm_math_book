from pathlib import Path


def test_ch06_to_ch32_solution_placeholders_are_not_published() -> None:
    root = Path(__file__).resolve().parents[2]

    published = [
        root / lang / "solutions" / f"ch{chapter:02d}_solutions.ipynb"
        for lang in ("ko", "en", "jp")
        for chapter in range(6, 33)
    ]

    assert not [path for path in published if path.exists()]
