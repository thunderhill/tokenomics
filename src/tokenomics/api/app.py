"""The Tokenomics API.

One FastAPI app serves three audiences: OTel exporters (``POST /v1/traces``), the
dashboard and any other client (``/api/*``, fully described by OpenAPI), and Prometheus
(``/metrics``). Nothing here calls out to a third party -- the only outbound request the
service can make is an explicit pricing refresh.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta

from fastapi import Depends, FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from tokenomics.api.deps import get_settings, require_api_key
from tokenomics.api.routers import budgets, finops, pricing, reports, spend, traces
from tokenomics.finops.budgets import InvalidScopeError
from tokenomics.pricing.engine import default_engine
from tokenomics.storage import database, repository
from tokenomics.storage.queries import InvalidDimensionError
from tokenomics.telemetry import metrics

DESCRIPTION = """
FinOps for multi-LLM applications: unified spend, attribution, budgets, forecasting,
anomaly detection and what-if simulation, built on the OpenTelemetry GenAI traces you
already emit.

Two properties the API guarantees:

* **Costs partition, they do not sum.** Cached and reasoning tokens are already inside
  the input and output totals, so they are billed once, at the right rate.
* **Unpriced is never zero.** An event we cannot price is stored with a NULL cost and
  counted separately in every response, so a resolution gap can never masquerade as a
  spend decrease.
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    engine = default_engine()
    metrics.pricebook_models.labels(snapshot=engine.snapshot_id).set(len(engine.book))

    if settings.auto_migrate:
        with database.connection() as conn:
            database.migrate(conn)
            now = datetime.now(UTC)
            # This month and next: ingestion never fails for want of a partition on the
            # 1st, and backfills create their own as needed.
            database.ensure_partitions(conn, [now, now + timedelta(days=32)])
            repository.record_snapshot(conn, engine.book)
    try:
        yield
    finally:
        database.close_pool()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Tokenomics",
        version="0.1.0",
        summary="FinOps observability and cost governance for multi-LLM applications.",
        description=DESCRIPTION,
        license_info={"name": "MIT", "identifier": "MIT"},
        lifespan=lifespan,
    )

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_methods=["*"],
            allow_headers=["*"],
        )

    guard = [Depends(require_api_key)]
    app.include_router(traces.router, dependencies=guard)
    for router in (spend.router, budgets.router, finops.router, reports.router, pricing.router):
        app.include_router(router, dependencies=guard)

    @app.exception_handler(InvalidDimensionError)
    @app.exception_handler(InvalidScopeError)
    async def _bad_dimension(request: Request, exc: Exception) -> JSONResponse:
        # These are the whitelist guards on group-by dimensions and budget scopes: a
        # miss is a client mistake, not a server fault.
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.get("/health", tags=["ops"], summary="Liveness and snapshot identity")
    def health() -> dict[str, object]:
        engine = default_engine()
        return {
            "status": "ok",
            "pricing_snapshot": engine.snapshot_id,
            "models_priced": len(engine.book),
        }

    @app.get(
        "/metrics",
        tags=["ops"],
        summary="Prometheus exposition",
        response_class=Response,
        include_in_schema=False,
    )
    def prometheus() -> Response:
        body, content_type = metrics.render()
        return Response(body, media_type=content_type)

    return app


app = create_app()
