import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "check_release_placeholders", ROOT / "scripts" / "check_release_placeholders.py"
)
assert SPEC and SPEC.loader
RELEASE_CHECKS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RELEASE_CHECKS)


def test_published_notebook_inventory_is_complete() -> None:
    assert RELEASE_CHECKS.check_inventory(ROOT) == []


def test_published_notebooks_are_release_ready() -> None:
    assert RELEASE_CHECKS.check_release(ROOT) == []
