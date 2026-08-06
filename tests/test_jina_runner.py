import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.skillpixel_kis_jina_benchmark import jina_provider_kwargs


def test_jina_runner_never_allows_implicit_fallback() -> None:
    assert jina_provider_kwargs("jinaai/jina-clip-v2", allow_model_download=False) == {
        "jina_model": "jinaai/jina-clip-v2",
        "local_files_only": True,
        "allow_fallback": False,
    }
