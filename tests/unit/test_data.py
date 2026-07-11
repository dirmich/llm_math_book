from pathlib import Path

import pytest

from llm_math.data import load_mini_korean, make_tiny_corpus


def test_load_mini_korean_uses_real_hangul_corpus() -> None:
    text = load_mini_korean()

    assert "한국어" in text
    assert any("가" <= char <= "힣" for char in text)
    assert "Natural language processing" not in text


def test_load_mini_korean_can_seed_custom_data_dir(tmp_path: Path) -> None:
    text = load_mini_korean(str(tmp_path))

    assert (tmp_path / "mini_korean.txt").exists()
    assert text == (tmp_path / "mini_korean.txt").read_text(encoding="utf-8")


def test_make_tiny_corpus_routes_korean_to_korean_loader() -> None:
    text = make_tiny_corpus(language="ko", max_chars=40)

    assert len(text) == 40
    assert any("가" <= char <= "힣" for char in text)


def test_make_tiny_corpus_rejects_unknown_language() -> None:
    with pytest.raises(ValueError, match="Unsupported language"):
        make_tiny_corpus(language="de")
