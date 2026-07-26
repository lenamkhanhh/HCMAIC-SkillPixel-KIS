"""Evaluator tests: metric math on controlled results + fixture run."""

import json
from pathlib import Path

import pytest

from hcmaic.contracts.models import SearchResult
from hcmaic.evaluation.evaluator import (
    EvalQuery,
    Qrel,
    _percentile,
    evaluate,
    format_summary,
    load_qrels,
    load_queries,
    write_reports,
)


class FakeProvider:
    name = "mock"


class FakeService:
    """Duck-typed service returning scripted rankings per query text."""

    text_provider = FakeProvider()
    index_version = "fake-v1"

    def __init__(self, rankings: dict[str, list[str]]) -> None:
        self._rankings = rankings

    def search(self, request):
        frame_ids = self._rankings.get(request.text, [])
        return [
            SearchResult(
                rank=i + 1,
                final_score=1.0 - i * 0.01,
                signal_scores={"visual": 1.0 - i * 0.01},
                video_id=fid.split(":")[0],
                frame_id=fid,
                frame_idx=0,
                timestamp_ms=0,
                image_url=f"/frames/{fid}/image",
                index_version=self.index_version,
            )
            for i, fid in enumerate(frame_ids[: request.top_k])
        ]


def test_metric_math_controlled():
    service = FakeService(
        {
            "hit-at-1": ["v:001", "v:002"],
            "hit-at-3": ["v:009", "v:008", "v:001"],
            "miss": ["v:009", "v:008"],
        }
    )
    queries = [
        EvalQuery("q1", "hit-at-1"),
        EvalQuery("q2", "hit-at-3"),
        EvalQuery("q3", "miss"),
    ]
    qrels = {
        "q1": Qrel("q1", {"v:001"}),
        "q2": Qrel("q2", {"v:001"}),
        "q3": Qrel("q3", {"v:001"}),
    }
    report, per_query = evaluate(service, queries, qrels, top_k=10)
    assert report["n_scored"] == 3
    assert report["recall_at"]["1"] == pytest.approx(1 / 3)
    assert report["recall_at"]["5"] == pytest.approx(2 / 3)
    assert report["recall_at"]["10"] == pytest.approx(2 / 3)
    # MRR = (1 + 1/3 + 0) / 3
    assert report["mrr"] == pytest.approx((1 + 1 / 3) / 3)
    ranks = {r["query_id"]: r.get("first_relevant_rank") for r in per_query}
    assert ranks == {"q1": 1, "q2": 3, "q3": None}


def test_missing_qrels_and_empty_results_counted():
    service = FakeService({"has-results": ["v:001"], "empty": []})
    queries = [EvalQuery("q1", "has-results"), EvalQuery("q2", "empty")]
    qrels = {"q2": Qrel("q2", {"v:001"})}
    report, per_query = evaluate(service, queries, qrels)
    assert report["n_missing_qrels"] == 1
    assert report["n_empty_results"] == 1
    assert report["n_scored"] == 1
    assert per_query[0]["error"] == "no qrels for this query"


def test_percentile():
    values = [1.0, 2.0, 3.0, 4.0]
    assert _percentile(values, 0.5) == pytest.approx(2.5)
    assert _percentile(values, 0.95) == pytest.approx(3.85)
    assert _percentile([], 0.5) == 0.0
    assert _percentile([7.0], 0.95) == 7.0


def test_mode_label_and_disclaimer():
    report, _ = evaluate(FakeService({}), [], {})
    assert report["mode"] == "deterministic-mock"
    assert "plumbing only" in report["disclaimer"]
    assert "plumbing only" in format_summary(report)


def test_load_queries_and_qrels(sample_root: Path):
    queries = load_queries(sample_root / "queries.jsonl")
    qrels = load_qrels(sample_root / "qrels.jsonl")
    assert len(queries) == 6
    assert queries[0].query_id == "q1"
    assert qrels["q6"].relevant_frame_ids == {"L01_V001:003", "L01_V005:001"}


def test_load_queries_rejects_bad_line(tmp_path: Path):
    bad = tmp_path / "queries.jsonl"
    bad.write_text('{"text": "no id"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="query_id"):
        load_queries(bad)


def test_fixture_evaluation_end_to_end(service, sample_root: Path, tmp_path: Path):
    queries = load_queries(sample_root / "queries.jsonl")
    qrels = load_qrels(sample_root / "qrels.jsonl")
    report, per_query = evaluate(service, queries, qrels, top_k=10)
    # The committed fixture is constructed so the mock provider solves it;
    # this asserts plumbing, not competition quality.
    assert report["recall_at"]["1"] == 1.0
    assert report["mrr"] == 1.0
    assert report["latency_ms"]["p95"] >= report["latency_ms"]["p50"] >= 0

    report_path, per_query_path = write_reports(report, per_query, tmp_path / "eval")
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["mode"] == "deterministic-mock"
    lines = per_query_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 6
