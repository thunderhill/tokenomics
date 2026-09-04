"""The ``tokenomics`` command line.

Everything the API does, available without one -- useful for cron jobs, air-gapped
boxes and the demo. The only subcommand that touches the network is ``pricing refresh``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Annotated

import typer

from tokenomics.finops import anomaly as anomaly_service
from tokenomics.finops import budgets as budget_service
from tokenomics.finops import forecast as forecast_service
from tokenomics.finops import reports as report_service
from tokenomics.finops import tokens as token_service
from tokenomics.finops import whatif as whatif_service
from tokenomics.finops.reports import Allocation
from tokenomics.pricing.engine import default_engine
from tokenomics.pricing.pricebook import LITELLM_URL, PriceBook
from tokenomics.storage import database, queries, repository
from tokenomics.storage.queries import Filters

app = typer.Typer(
    name="tokenomics",
    help="FinOps for multi-LLM applications.",
    no_args_is_help=True,
    add_completion=False,
)
pricing_app = typer.Typer(help="Pricing snapshots.", no_args_is_help=True)
db_app = typer.Typer(help="Database maintenance.", no_args_is_help=True)
app.add_typer(pricing_app, name="pricing")
app.add_typer(db_app, name="db")

Days = Annotated[int, typer.Option(help="Look back this many days.")]


def _window(days: int) -> Filters:
    now = datetime.now(UTC)
    return Filters(since=now - timedelta(days=days), until=now)


def _echo_json(payload: object) -> None:
    typer.echo(json.dumps(payload, indent=2, default=str))


@app.command()
def serve(
    host: str = "0.0.0.0",  # containers must bind all interfaces
    port: int = 8000,
    reload: bool = False,
) -> None:
    """Run the API server."""
    import uvicorn

    uvicorn.run("tokenomics.api.app:app", host=host, port=port, reload=reload)


@db_app.command("migrate")
def db_migrate() -> None:
    """Apply migrations and create the current and next month's partitions."""
    with database.connection() as conn:
        applied = database.migrate(conn)
        now = datetime.now(UTC)
        partitions = database.ensure_partitions(conn, [now, now + timedelta(days=32)])
    typer.echo(f"applied {len(applied)} migration(s); partitions: {sorted(partitions)}")


@db_app.command("rollup")
def db_rollup(days: Days = 2) -> None:
    """Recompute hourly rollups for a recent window."""
    now = datetime.now(UTC)
    with database.connection() as conn:
        rows = repository.refresh_rollups(conn, now - timedelta(days=days), now)
    typer.echo(f"refreshed {rows} rollup row(s)")


@pricing_app.command("show")
def pricing_show() -> None:
    """Describe the active (vendored) pricing snapshot."""
    book = default_engine().book
    _echo_json(
        {
            "snapshot_id": book.snapshot_id,
            "source": book.source,
            "fetched_at": book.fetched_at,
            "models": len(book),
        }
    )


@pricing_app.command("refresh")
def pricing_refresh(
    url: str = LITELLM_URL,
    save: Annotated[bool, typer.Option(help="Also register the snapshot in the database.")] = True,
) -> None:
    """Fetch the current price list. The only outbound call in the whole system."""
    book = PriceBook.fetch(url)
    if save:
        with database.connection() as conn:
            repository.record_snapshot(conn, book)
    typer.echo(f"fetched {len(book)} models; snapshot {book.snapshot_id}")


@app.command()
def spend(
    days: Days = 30,
    group_by: Annotated[str, typer.Option(help="Comma-separated dimensions.")] = "project",
) -> None:
    """Spend broken down by attribution."""
    dimensions = [d.strip() for d in group_by.split(",") if d.strip()]
    with database.connection() as conn:
        rows = queries.spend_breakdown(conn, filters=_window(days), group_by=dimensions)
    _echo_json(rows)


@app.command()
def tokens(
    days: Days = 30,
    group_by: Annotated[str, typer.Option(help="Comma-separated dimensions.")] = "project",
    totals: Annotated[bool, typer.Option(help="Collapse to one window-wide row.")] = False,
) -> None:
    """Token usage beside the cost of that usage, per billable component."""
    dimensions = [d.strip() for d in group_by.split(",") if d.strip()]
    with database.connection() as conn:
        rows = queries.token_economics(conn, filters=_window(days), group_by=dimensions)
    if totals:
        _echo_json(token_service.summarize(rows))
        return
    _echo_json([token_service.derive(row) for row in rows])


@app.command()
def forecast(days: Days = 60) -> None:
    """Project month-end spend from the trailing daily series."""
    with database.connection() as conn:
        series = queries.daily_spend(conn, filters=_window(days))
    result = forecast_service.forecast_month_end(list(series))
    _echo_json(
        {
            "method": result.method,
            "month_to_date_usd": result.month_to_date_usd,
            "projected_month_end_usd": result.projected_month_end_usd,
            "range": [result.lower_usd, result.upper_usd],
            "daily_run_rate_usd": result.daily_run_rate_usd,
            "days_remaining": result.days_remaining,
        }
    )


@app.command()
def anomalies(
    days: Days = 14,
    threshold: float = anomaly_service.DEFAULT_THRESHOLD,
    persist: bool = False,
) -> None:
    """Detect spend anomalies and attribute a probable cause."""
    with database.connection() as conn:
        found = anomaly_service.scan(conn, filters=_window(days), threshold=threshold)
        if persist and found:
            anomaly_service.persist(conn, found)
    _echo_json(
        [
            {
                "bucket": item.bucket,
                "observed_usd": item.observed_usd,
                "baseline_usd": item.baseline_usd,
                "score": item.score,
                "multiple": round(item.multiple, 2),
                "probable_cause": [
                    {"dimension": c.dimension, "value": c.value, "share": round(c.share, 3)}
                    for c in item.probable_cause
                ],
            }
            for item in found
        ]
    )


@app.command()
def budgets(evaluate: bool = False) -> None:
    """List budgets, or evaluate them and fire any newly crossed thresholds."""
    with database.connection() as conn:
        if evaluate:
            fired = budget_service.run_budget_cycle(conn)
            _echo_json(
                [
                    {
                        "budget": alert.budget.name,
                        "threshold": alert.threshold,
                        "spend_usd": alert.spend_usd,
                        "amount_usd": alert.amount_usd,
                    }
                    for alert in fired
                ]
            )
            return
        statuses = [
            budget_service.evaluate(conn, budget)
            for budget in budget_service.list_budgets(conn, enabled_only=False)
        ]
    _echo_json(
        [
            {
                "name": s.budget.name,
                "period": str(s.budget.period),
                "spend_usd": s.spend_usd,
                "amount_usd": s.amount_usd,
                "utilization": round(float(s.utilization), 4),
                "unpriced_events": s.unpriced_events,
            }
            for s in statuses
        ]
    )


@app.command()
def report(
    days: Days = 30,
    group_by: str = "project",
    allocation: Allocation = Allocation.SHOW,
    csv: Annotated[bool, typer.Option(help="Emit CSV instead of JSON.")] = False,
) -> None:
    """Chargeback / showback report."""
    dimensions = tuple(d.strip() for d in group_by.split(",") if d.strip())
    with database.connection() as conn:
        result = report_service.chargeback(
            conn, filters=_window(days), group_by=dimensions, allocation=allocation
        )
    typer.echo(report_service.to_csv(result) if csv else json.dumps(result.to_dict(), indent=2))


@app.command()
def whatif(
    targets: Annotated[list[str], typer.Argument(help="Model keys to reprice against.")],
    days: Days = 30,
) -> None:
    """Reprice recent traffic against other models."""
    engine = default_engine()
    with database.connection() as conn:
        samples, scale = whatif_service.load_samples(conn, filters=_window(days))
    results = whatif_service.compare(engine, samples, targets=targets, scale_factor=scale)
    _echo_json(
        [
            {
                "target": r.target_model,
                "resolved": r.resolved_model_key,
                "baseline_usd": r.baseline_usd.quantize(Decimal("0.0001")),
                "projected_usd": r.projected_usd.quantize(Decimal("0.0001")),
                "delta_pct": None if r.delta_pct is None else round(r.delta_pct, 2),
                "refolded_cache_tokens": r.refolded_cache_tokens,
                "warnings": list(r.warnings),
                "quality": r.quality.status,
            }
            for r in results
        ]
    )


if __name__ == "__main__":
    app()
