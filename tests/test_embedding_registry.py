from __future__ import annotations

import pytest

from hcmaic.embedding.registry import (
    get_provider_descriptor,
    list_provider_descriptors,
    provider_doctor,
)


def test_registry_keeps_control_and_optional_provider_descriptors():
    names = [item.name for item in list_provider_descriptors()]
    assert names == ["mock", "clip", "siglip2", "jina-clip-v2"]
    assert get_provider_descriptor("siglip2").lazy is True


def test_provider_doctor_is_non_downloading_and_actionable():
    report = provider_doctor("siglip2")
    assert report["provider"] == "siglip2"
    assert report["evidence_level"] in {"INTERFACE_ONLY", "BLOCKED"}
    assert report["model_revision"]
    assert report["install"]
    assert "download" not in report["action"].lower()


def test_unknown_provider_is_explicit():
    with pytest.raises(ValueError, match="Unknown embedding provider"):
        get_provider_descriptor("does-not-exist")
