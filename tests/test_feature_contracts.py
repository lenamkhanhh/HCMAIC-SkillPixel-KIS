from __future__ import annotations

import pytest

from hcmaic.features.base import FeatureContext
from hcmaic.features.mock import (
    MockASRProvider,
    MockCaptionProvider,
    MockOCRProvider,
)


@pytest.mark.parametrize(
    "provider",
    [MockOCRProvider(), MockASRProvider(), MockCaptionProvider()],
)
def test_mock_modalities_emit_deterministic_feature_records(provider):
    context = FeatureContext(
        video_id="V1",
        entity_id="V1:001",
        start_ms=1000,
        end_ms=3000,
        text_hint="red bus near station",
    )
    first = provider.extract(context)
    second = provider.extract(context)
    assert first == second
    assert first and first[0].modality == provider.modality
    assert first[0].content_hash
    assert first[0].start_ms == 1000


def test_feature_context_rejects_invalid_time_range():
    with pytest.raises(ValueError, match="end_ms"):
        FeatureContext("V1", "V1:001", 3000, 1000)
