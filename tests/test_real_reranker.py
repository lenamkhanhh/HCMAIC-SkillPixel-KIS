import pytest

from hcmaic.retrieval.rerank import CrossEncoderReranker, RealRerankerUnavailable


def test_cross_encoder_reranker_requires_explicit_weights_or_download_flag() -> None:
    with pytest.raises(RealRerankerUnavailable, match="not cached"):
        CrossEncoderReranker(local_files_only=True, allow_model_download=False)
