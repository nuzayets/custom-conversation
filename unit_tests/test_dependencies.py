"""Tests for deployment dependency alignment."""

from importlib.metadata import version
import json
from pathlib import Path


def test_verified_runtime_versions_are_installed():
    """Test the suite runs against deployment versions."""
    assert version("langfuse") == "4.0.0"
    assert version("litellm") == "1.81.4"


def test_dependency_declarations_match():
    """Test manifest and Pipfile keep verified versions aligned."""
    root = Path(__file__).parents[1]
    manifest = json.loads(
        (root / "custom_components/custom_conversation/manifest.json").read_text()
    )
    pipfile = (root / "Pipfile").read_text()
    lock = json.loads((root / "Pipfile.lock").read_text())

    assert "langfuse==4.0.0" in manifest["requirements"]
    assert "litellm==1.81.4" in manifest["requirements"]
    assert 'langfuse = "==4.0.0"' in pipfile
    assert 'litellm = "==1.81.4"' in pipfile
    assert lock["default"]["langfuse"]["version"] == "==4.0.0"
    assert lock["default"]["litellm"]["version"] == "==1.81.4"
