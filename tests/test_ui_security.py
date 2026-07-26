"""Static security regression checks for the dependency-free operator UI."""

from pathlib import Path


def test_metadata_is_not_inserted_as_html():
    app_js = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "hcmaic"
        / "ui"
        / "static"
        / "app.js"
    ).read_text(encoding="utf-8")
    assert 'meta.insertAdjacentHTML("beforeend"' not in app_js
