"""The HTTP surface, end to end against a real database.

These tests commit (the API uses its own pooled connections), so the module cleans up
after itself to keep the rest of the suite's absolute assertions valid.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import psycopg
import pytest
from fastapi.testclient import TestClient
from google.protobuf import json_format

from tokenomics.api.deps import get_settings
from tokenomics.ingest.otlp import decode_request
from tokenomics.storage import database

pytestmark = pytest.mark.integration

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture
def client(pg_url: str, _schema: None, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setitem(os.environ, "TOKENOMICS_DATABASE_URL", pg_url)
    get_settings.cache_clear()
    database.close_pool()

    from tokenomics.api.app import create_app

    with TestClient(create_app()) as test_client:
        yield test_client

    with psycopg.connect(pg_url) as conn, conn.cursor() as cur:
        cur.execute("TRUNCATE usage_event, usage_rollup_hourly, budget, anomaly CASCADE")
    database.close_pool()
    get_settings.cache_clear()


def _target_moment() -> datetime:
    """Where the fixture's newest span should land, fixed once for the whole run.

    It has to be *one* value: ``ts`` is part of the ``(ts, trace_id, span_id)``
    idempotency key, so an exporter retry that re-based against a fresher clock would
    write a second row and read as double-billing.
    """
    now = datetime.now(UTC)
    # An hour ago -- but never before the start of the month, or a run in the first
    # hour of the 1st would push the spans into the previous month and put a monthly
    # budget back at zero, which is the bug this whole helper exists to kill.
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return max(now - timedelta(hours=1), month_start + timedelta(minutes=1))


TARGET = _target_moment()


def rebase(body: bytes, content_type: str) -> bytes:
    """Move the recorded spans into the current query window.

    The fixture is genuine SDK output, so its timestamps are whenever it was last
    generated -- and every assertion in this module reads through a *time window*: the
    30-day default on the spend endpoints, and the current calendar month on a monthly
    budget. A fixture a few weeks old therefore starts returning nothing, and these
    tests fail on the calendar rather than on the code. (That is exactly what happened:
    the budget test began failing the day the month rolled over.)

    Regenerating the fixture only resets the fuse. Re-basing here defuses it for good:
    the bytes stay a real recording and keep their relative offsets, and only the clock
    moves. The unit tests in ``tests/unit/test_ingest.py`` deliberately still parse the
    untouched file, so nothing is lost on the "we parse what the SDK really emits" side.

    Every call in a run shifts by the same amount (see :data:`TARGET`), so re-posting
    the fixture stays byte-identical -- which is what an exporter retry is.
    """
    request = decode_request(body, content_type)
    spans = [
        span
        for resource in request.resource_spans
        for scope in resource.scope_spans
        for span in scope.spans
    ]
    assert spans, "fixture contains no spans"

    shift = int(TARGET.timestamp() * 1_000_000_000) - max(span.end_time_unix_nano for span in spans)
    for span in spans:
        span.start_time_unix_nano += shift
        span.end_time_unix_nano += shift

    if content_type == "application/json":
        return json_format.MessageToJson(request).encode()
    return request.SerializeToString()


def post_fixture(client: TestClient) -> dict:
    body = rebase((FIXTURES / "otlp_traces.pb").read_bytes(), "application/x-protobuf")
    response = client.post(
        "/v1/traces", content=body, headers={"Content-Type": "application/x-protobuf"}
    )
    assert response.status_code == 200
    return response


def test_health_reports_the_pricing_snapshot(client: TestClient, frozen_snapshot_sha) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["pricing_snapshot"] == frozen_snapshot_sha


def test_otlp_protobuf_round_trip(client: TestClient) -> None:
    response = post_fixture(client)
    assert response.headers["content-type"].startswith("application/x-protobuf")

    rows = client.get("/api/spend/breakdown?group_by=provider").json()
    providers = {row["dimensions"]["provider"] for row in rows}
    assert providers == {"openai", "anthropic"}
    assert all(Decimal(row["cost_usd"]) > 0 for row in rows)


def test_otlp_json_is_accepted_too(client: TestClient) -> None:
    body = rebase((FIXTURES / "otlp_traces.json").read_bytes(), "application/json")
    response = client.post("/v1/traces", content=body, headers={"Content-Type": "application/json"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert client.get("/api/spend/breakdown").json()[0]["requests"] == 2


def test_replaying_a_batch_does_not_double_bill(client: TestClient) -> None:
    post_fixture(client)
    first = client.get("/api/spend/breakdown").json()[0]
    post_fixture(client)  # exporter retry
    second = client.get("/api/spend/breakdown").json()[0]

    assert first == second


def test_a_malformed_body_is_rejected_without_retry_advice(client: TestClient) -> None:
    response = client.post(
        "/v1/traces", content=b"not-protobuf", headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 400


def test_an_oversized_body_is_refused(client: TestClient, monkeypatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "max_body_bytes", 10)
    response = client.post(
        "/v1/traces", content=b"x" * 100, headers={"Content-Type": "application/x-protobuf"}
    )
    assert response.status_code == 413


def test_ingestion_updates_prometheus_counters(client: TestClient) -> None:
    post_fixture(client)
    body = client.get("/metrics").text

    assert 'tokenomics_events_ingested_total{project="checkout",provider="openai"}' in body
    assert "tokenomics_cost_usd_total" in body
    # The disjoint partition the cost function bills is what the metrics report.
    assert 'tokenomics_tokens_total{component="cache_read"' in body


def test_series_and_unit_economics(client: TestClient) -> None:
    post_fixture(client)

    series = client.get("/api/spend/series?granularity=day&group_by=project").json()
    assert series and "bucket" in series[0]

    economics = client.get("/api/spend/unit-economics?group_by=project").json()
    assert Decimal(economics[0]["cost_per_request"]) > 0


def test_an_unknown_group_by_is_a_client_error(client: TestClient) -> None:
    response = client.get("/api/spend/breakdown?group_by=project;DROP TABLE usage_event")
    assert response.status_code == 422
    assert client.get("/health").status_code == 200


def test_budget_lifecycle_over_http(client: TestClient) -> None:
    post_fixture(client)
    spend = Decimal(client.get("/api/spend/breakdown").json()[0]["cost_usd"])

    created = client.post(
        "/api/budgets",
        json={
            "name": "checkout",
            "amount_usd": str(spend),
            "scope": {"project": "checkout"},
            "thresholds": ["0.5", "1.0"],
            "webhook_secret": "s3cret",
        },
    )
    assert created.status_code == 201
    budget = created.json()
    assert budget["has_webhook_secret"] is True
    assert "webhook_secret" not in budget  # secrets never travel back out

    status = client.get(f"/api/budgets/{budget['id']}/status").json()
    assert Decimal(status["spend_usd"]) == spend
    assert status["utilization"] == pytest.approx(1.0)
    assert status["thresholds_crossed"] == ["0.5", "1.0"]

    first = client.post("/api/budgets/evaluate").json()
    second = client.post("/api/budgets/evaluate").json()
    assert len(first) == 2
    assert second == []  # fire-once holds across HTTP calls too

    assert client.delete(f"/api/budgets/{budget['id']}").status_code == 204
    assert client.get(f"/api/budgets/{budget['id']}").status_code == 404


def test_a_budget_with_a_bad_scope_is_refused(client: TestClient) -> None:
    response = client.post(
        "/api/budgets",
        json={"name": "typo", "amount_usd": "100", "scope": {"porject": "checkout"}},
    )
    assert response.status_code == 422


def test_chargeback_json_and_csv(client: TestClient) -> None:
    post_fixture(client)

    report = client.get("/api/reports/chargeback?group_by=project").json()
    assert report["kind"] == "chargeback"
    assert report["lines"][0]["dimensions"]["project"] == "checkout"

    csv_response = client.get("/api/reports/chargeback?group_by=project&format=csv")
    assert csv_response.headers["content-type"].startswith("text/csv")
    assert csv_response.text.splitlines()[0].startswith("project,")


def test_whatif_over_http(client: TestClient) -> None:
    post_fixture(client)

    response = client.post("/api/whatif", json={"targets": ["gpt-4o-mini", "claude-sonnet-4-5"]})
    results = response.json()

    assert {r["target_model"] for r in results} == {"gpt-4o-mini", "claude-sonnet-4-5"}
    assert results[0]["quality"]["status"] == "not-measured"
    assert Decimal(results[0]["projected_usd"]) > 0


def test_pricing_endpoints_never_leave_the_box(client: TestClient) -> None:
    # /snapshot and /models read the vendored snapshot; only /refresh would go outbound.
    assert client.get("/api/pricing/snapshot").json()["model_count"] > 3000
    assert "gpt-4o" in client.get("/api/pricing/models?q=gpt-4o").json()

    rates = client.get("/api/pricing/models/claude-sonnet-4-5").json()
    assert rates["prices_cache"] is True
    assert rates["context_tiers"] == [200000]

    assert client.get("/api/pricing/models/acme%2Fnope").status_code == 404


def test_the_api_key_guard_when_configured(pg_url: str, monkeypatch) -> None:
    monkeypatch.setitem(os.environ, "TOKENOMICS_DATABASE_URL", pg_url)
    monkeypatch.setitem(os.environ, "TOKENOMICS_API_KEY", "letmein")
    get_settings.cache_clear()
    database.close_pool()

    from tokenomics.api.app import create_app

    with TestClient(create_app()) as guarded:
        assert guarded.get("/health").status_code == 200  # liveness stays open
        assert guarded.get("/api/spend/breakdown").status_code == 401
        assert (
            guarded.get(
                "/api/spend/breakdown", headers={"Authorization": "Bearer letmein"}
            ).status_code
            == 200
        )
        assert (
            guarded.get(
                "/api/spend/breakdown", headers={"Authorization": "Bearer wrong"}
            ).status_code
            == 401
        )

    database.close_pool()
    get_settings.cache_clear()


def test_spend_rows_carry_tokens_alongside_the_money(client: TestClient) -> None:
    post_fixture(client)
    row = client.get("/api/spend/breakdown").json()[0]

    assert row["reasoning_tokens"] >= 0
    assert row["billable_input_tokens"] <= row["input_tokens"]
    components = row["cost_components"]
    assert sum(Decimal(value) for value in components.values()) == Decimal(row["cost_usd"])


def test_token_economics_splits_volume_from_spend(client: TestClient) -> None:
    post_fixture(client)
    rows = client.get("/api/spend/tokens?group_by=project").json()
    assert rows

    row = rows[0]
    assert sum(row["token_components"].values()) == row["total_tokens"]
    assert row["total_tokens"] == row["input_tokens"] + row["output_tokens"]
    shares = [share for share in row["token_shares"].values() if share is not None]
    assert sum(shares) == pytest.approx(1.0)
    assert Decimal(row["usd_per_1m_tokens"]) > 0


def test_token_summary_is_one_ungrouped_row_not_a_fold(client: TestClient) -> None:
    post_fixture(client)
    summary = client.get("/api/spend/tokens/summary").json()
    grouped = client.get("/api/spend/tokens?group_by=project").json()

    assert summary["dimensions"] == {}
    assert summary["total_tokens"] == sum(row["total_tokens"] for row in grouped)
    # A COUNT(DISTINCT) that was summed per slice would come out higher than the truth.
    assert summary["subjects"] <= sum(row["subjects"] for row in grouped)


def test_token_summary_survives_an_empty_window(client: TestClient) -> None:
    """The dashboard calls this before any data exists; it must not divide by zero."""
    summary = client.get("/api/spend/tokens/summary?since=2020-01-01T00:00:00Z").json()

    assert summary["total_tokens"] == 0
    assert summary["usd_per_1m_tokens"] is None
    assert summary["cache_hit_rate"] is None
    assert Decimal(summary["net_cache_benefit_usd"]) == 0


def test_token_economics_rejects_an_unknown_dimension(client: TestClient) -> None:
    response = client.get("/api/spend/tokens?group_by=api_key")
    assert response.status_code == 422


def test_per_component_cost_reaches_prometheus(client: TestClient) -> None:
    post_fixture(client)
    body = client.get("/metrics").text

    # Volume and money over the same partition, so one can be divided by the other.
    assert 'tokenomics_cost_usd_by_component_total{component="output"' in body


def test_chargeback_lines_carry_the_component_split(client: TestClient) -> None:
    post_fixture(client)
    report = client.get("/api/reports/chargeback?group_by=project").json()
    line = report["lines"][0]

    assert line["total_tokens"] == line["input_tokens"] + line["output_tokens"]
    assert Decimal(line["usd_per_1m_tokens"]) > 0
    components = ("input_usd", "output_usd", "cache_read_usd", "cache_write_usd", "reasoning_usd")
    assert sum(Decimal(line[key]) for key in components) == Decimal(line["direct_usd"])


def test_chargeback_csv_keeps_its_original_column_order(client: TestClient) -> None:
    """Finance imports this by position; new columns append, they never insert."""
    post_fixture(client)
    body = client.get("/api/reports/chargeback?group_by=project&format=csv").text
    header = body.splitlines()[0].split(",")

    assert header[:8] == [
        "project",
        "requests",
        "subjects",
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "direct_usd",
    ]
    assert "reasoning_tokens" in header
    assert "usd_per_1m_tokens" in header


def test_the_fixture_lands_inside_every_window_these_tests_query(client: TestClient) -> None:
    """The regression guard for `rebase`.

    Both windows below are moving targets, and a fixture that drifts out of either one
    fails tests elsewhere in this module with an error that points at the wrong thing:
    the budget test simply reports zero spend and looks like a budgeting bug.
    """
    post_fixture(client)
    now = datetime.now(UTC)
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    # The 30-day default window used by every spend endpoint.
    assert client.get("/api/spend/breakdown").json()[0]["requests"] == 2
    # The current calendar month, which is what a monthly budget measures.
    # Passed as a param, not interpolated: the `+` in an ISO offset decodes as a
    # space in a raw query string and the endpoint rejects it as unparseable.
    month = client.get("/api/spend/breakdown", params={"since": month_start.isoformat()}).json()
    assert month[0]["requests"] == 2
    assert now > TARGET  # spans are in the past, never the future
