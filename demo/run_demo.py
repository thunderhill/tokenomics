#!/usr/bin/env python3
"""End-to-end Tokenomics demo.

Simulates 45 days of traffic for three projects on four models, ships it to a running
Tokenomics API as **real OTLP/protobuf**, then prints the FinOps output: a chargeback
report, the token/cost split with what the prompt cache is worth, a month-end forecast,
the injected anomaly with its probable cause, budget burn-down, and a what-if
migration.

    docker compose up -d
    uv run python demo/run_demo.py

Everything runs offline against the vendored price list.
"""

from __future__ import annotations

import argparse
import sys
import time
from decimal import Decimal
from typing import Any

import httpx
from demo.simulate import batches, generate, to_otlp

DEFAULT_API = "http://localhost:8000"


def wait_for_api(client: httpx.Client, *, attempts: int = 30) -> dict[str, Any]:
    for attempt in range(attempts):
        try:
            response = client.get("/health", timeout=5.0)
            if response.status_code == 200:
                return response.json()
        except httpx.HTTPError:
            pass
        if attempt == 0:
            print("waiting for the API ...", flush=True)
        time.sleep(1.0)
    sys.exit(f"API at {client.base_url} never became healthy")


def ingest(client: httpx.Client, days: int, batch_size: int) -> int:
    calls = generate(days=days)
    print(f"simulated {len(calls):,} calls over {days} days", flush=True)

    started = time.perf_counter()
    for offset, batch in batches(calls, batch_size):
        response = client.post(
            "/v1/traces",
            content=to_otlp(batch, offset=offset),
            headers={"Content-Type": "application/x-protobuf"},
            timeout=120.0,
        )
        response.raise_for_status()
        done = offset + len(batch)
        print(f"\r  ingested {done:,}/{len(calls):,}", end="", flush=True)
    elapsed = time.perf_counter() - started
    print(f"\r  ingested {len(calls):,} spans in {elapsed:.1f}s{' ' * 20}")
    return len(calls)


def usd(value: Any, places: str = "0.01") -> str:
    if value is None:
        return "-"
    amount = Decimal(str(value)).quantize(Decimal(places))
    # The sign belongs outside the symbol: "-$5.78", not "$-5.78". Net cache benefit
    # can legitimately be negative, so this is not a hypothetical.
    return f"-${abs(amount):,}" if amount < 0 else f"${amount:,}"


def table(rows: list[list[str]], headers: list[str]) -> str:
    widths = [max(len(str(row[i])) for row in [headers, *rows]) for i in range(len(headers))]
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True))
    rule = "  ".join("-" * w for w in widths)
    body = [
        "  ".join(str(cell).ljust(w) for cell, w in zip(row, widths, strict=True)) for row in rows
    ]
    return "\n".join([line, rule, *body])


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def show_chargeback(client: httpx.Client) -> None:
    section("CHARGEBACK -- who owes what (45 days)")
    report = client.get(
        "/api/reports/chargeback",
        params={"group_by": "project", "since": _since(client), "allocation": "show"},
    ).json()

    rows = [
        [
            line["dimensions"]["project"],
            f"{line['requests']:,}",
            usd(line["total_usd"]),
            f"{line['share'] * 100:5.1f}%",
            usd(line["cost_per_request"], "0.000001"),
            usd(line["cost_per_subject"], "0.01"),
            f"{(line['cache_hit_rate'] or 0) * 100:4.0f}%",
            f"{line['unpriced_events']:,}" if line["unpriced_events"] else "-",
        ]
        for line in report["lines"]
    ]
    headers = [
        "project",
        "requests",
        "spend",
        "share",
        "$/request",
        "$/customer",
        "cache",
        "unpriced",
    ]
    print(table(rows, headers))
    print(f"\ntotal {usd(report['total_usd'])} | unpriced events: {report['unpriced_events']:,}")
    if report["unpriced_events"]:
        print(
            "  A line with spend but a non-zero unpriced count is understated: those\n"
            "  events are stored with cost NULL, never 0, so the gap is stated rather\n"
            "  than absorbed into the total."
        )

    section("TOP FEATURES BY SPEND")
    features = client.get(
        "/api/spend/breakdown",
        params={"group_by": "project,feature", "since": _since(client), "limit": 8},
    ).json()
    print(
        table(
            [
                [
                    f"{row['dimensions']['project']} / {row['dimensions']['feature']}",
                    f"{row['requests']:,}",
                    usd(row["cost_usd"]),
                ]
                for row in features
            ],
            ["project / feature", "requests", "spend"],
        )
    )


def show_token_economics(client: httpx.Client) -> None:
    section("TOKEN ECONOMICS -- volume is not the bill")
    summary = client.get("/api/spend/tokens/summary", params={"since": _since(client)}).json()

    components = summary["token_components"]
    costs = summary["cost_components"]
    rows = []
    for name in ("cache_read", "cache_write", "input", "output", "reasoning"):
        token_share = summary["token_shares"][name]
        cost_share = summary["cost_shares"][name]
        rate = summary["usd_per_1m_by_component"][name]
        rows.append(
            [
                name,
                f"{components[name]:,}",
                "-" if token_share is None else f"{token_share * 100:5.1f}%",
                usd(costs[f"{name}_usd"], "0.0001"),
                "-" if cost_share is None else f"{cost_share * 100:5.1f}%",
                "-" if rate is None else usd(rate, "0.01") + "/M",
            ]
        )
    print(table(rows, ["component", "tokens", "of volume", "spend", "of spend", "rate"]))
    print(
        f"\n  {int(summary['total_tokens']):,} tokens for {usd(summary['cost_usd'])} "
        f"-- a blended {usd(summary['usd_per_1m_tokens'], '0.01')} per million."
    )
    print(
        "  The two share columns are the point: the cheapest bucket is usually most of\n"
        "  the volume and almost none of the money."
    )

    # The reasoning row reads as a contradiction against the per-project table below
    # unless this is spelled out: only ~58 models price reasoning apart, and for
    # everyone else the tokens are already inside the output charge.
    bundled = int(summary["reasoning_tokens"]) - int(components["reasoning"])
    if bundled > 0:
        print(
            f"\n  {bundled:,} reasoning tokens are reported but have no reasoning row above:\n"
            "  these models quote no separate reasoning rate, so the tokens are already\n"
            "  inside the output charge. Billing them again would double-count them."
        )

    section("CACHE -- what prompt caching is actually worth")
    print(
        f"  saved by cache reads   {usd(summary['cache_savings_usd'], '0.0001')}\n"
        f"  premium paid to write  {usd(summary['cache_write_premium_usd'], '0.0001')}\n"
        f"  net benefit            {usd(summary['net_cache_benefit_usd'], '0.0001')}"
    )
    if summary["cache_hit_rate"] is not None:
        print(f"  cache hit rate         {summary['cache_hit_rate'] * 100:.1f}% of input tokens")
    coverage = summary["cache_priced_coverage"]
    if coverage is not None and coverage < 1:
        print(
            f"  NOTE: only {coverage * 100:.1f}% of cached tokens carried a usable rate, "
            "so the benefit above is a floor."
        )
    print(
        "\n  Priced from the rates each event was billed at, not from today's price\n"
        "  list -- a saving computed against a rate that has since changed is fiction."
    )

    losers = [
        row
        for row in client.get(
            "/api/spend/tokens", params={"since": _since(client), "group_by": "project"}
        ).json()
        if Decimal(row["net_cache_benefit_usd"]) < 0
    ]
    for row in losers:
        print(
            f"\n  !! {row['dimensions']['project']} is a net LOSS of "
            f"{usd(row['net_cache_benefit_usd'], '0.0001')}: it writes the cache more\n"
            f"     often than it reads it, so it pays the ~1.25x write premium without\n"
            "     collecting the ~0.1x read. On savings alone it would look like a win."
        )

    section("BY PROJECT -- reasoning and cache profiles differ")
    per_project = client.get(
        "/api/spend/tokens", params={"since": _since(client), "group_by": "project"}
    ).json()
    print(
        table(
            [
                [
                    row["dimensions"]["project"],
                    f"{row['total_tokens']:,}",
                    usd(row["cost_usd"]),
                    "-"
                    if row["usd_per_1m_tokens"] is None
                    else usd(row["usd_per_1m_tokens"], "0.01") + "/M",
                    "-"
                    if row["cache_hit_rate"] is None
                    else f"{row['cache_hit_rate'] * 100:4.0f}%",
                    "-"
                    if row["reasoning_share"] is None
                    else f"{row['reasoning_share'] * 100:4.0f}%",
                    usd(row["net_cache_benefit_usd"], "0.0001"),
                ]
                for row in per_project
            ],
            ["project", "tokens", "spend", "blended", "cached", "reasoning", "cache net"],
        )
    )


def show_forecast(client: httpx.Client) -> None:
    section("FORECAST -- month-end projection")
    forecast = client.get("/api/forecast", params={"since": _since(client)}).json()
    print(
        f"  month to date     {usd(forecast['month_to_date_usd'])}\n"
        f"  daily run rate    {usd(forecast['daily_run_rate_usd'])}\n"
        f"  projected close   {usd(forecast['projected_month_end_usd'])}  "
        f"(range {usd(forecast['lower_usd'])} - {usd(forecast['upper_usd'])})\n"
        f"  days remaining    {forecast['days_remaining']}\n"
        f"  method            {forecast['method']}"
    )
    factors = forecast["day_of_week_factors"]
    if factors:
        names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        print(
            "  weekday shape     "
            + "  ".join(f"{n} {f:.2f}" for n, f in zip(names, factors, strict=True))
        )


def show_anomalies(client: httpx.Client) -> None:
    section("ANOMALIES -- unusual for this hour, not merely large")
    found = client.get(
        "/api/anomalies", params={"since": _since(client, days=14), "persist": True}
    ).json()
    if not found:
        print("  nothing flagged")
        return
    for item in found:
        causes = ", ".join(
            f"{c['dimension']}={c['value']} ({c['share'] * 100:.0f}%)"
            for c in item["probable_cause"]
        )
        print(
            f"  {item['bucket']}  observed {usd(item['observed_usd'])} vs baseline "
            f"{usd(item['baseline_usd'])}  ({item['multiple']:.1f}x, z={item['score']:.1f})\n"
            f"      probable cause: {causes or 'unattributed'}"
        )


def show_budgets(client: httpx.Client) -> None:
    section("BUDGETS -- burn-down and fire-once alerts")
    existing = {b["name"]: b for b in client.get("/api/budgets").json()}
    wanted = [
        {"name": "checkout monthly", "amount_usd": "400", "scope": {"project": "checkout"}},
        {"name": "support monthly", "amount_usd": "900", "scope": {"project": "support"}},
        {"name": "cx team", "amount_usd": "1200", "scope": {"tags": {"team": "cx"}}},
    ]
    for payload in wanted:
        if payload["name"] not in existing:
            client.post("/api/budgets", json={**payload, "thresholds": ["0.5", "0.8", "1.0"]})

    rows = []
    for budget in client.get("/api/budgets").json():
        status = client.get(f"/api/budgets/{budget['id']}/status").json()
        bar_width = 24
        filled = min(int(status["utilization"] * bar_width), bar_width)
        bar = "#" * filled + "." * (bar_width - filled)
        rows.append(
            [
                budget["name"],
                usd(status["spend_usd"]),
                usd(status["amount_usd"]),
                f"[{bar}] {status['utilization'] * 100:5.1f}%",
                ", ".join(str(t) for t in status["thresholds_crossed"]) or "-",
            ]
        )
    print(table(rows, ["budget", "spend", "limit", "burn-down (this month)", "crossed"]))

    fired = client.post("/api/budgets/evaluate").json()
    again = client.post("/api/budgets/evaluate").json()
    print(f"\n  thresholds fired: {len(fired)}; re-running the cycle fired {len(again)} more")


def show_whatif(client: httpx.Client) -> None:
    section("WHAT-IF -- reprice the last 30 days on other models")
    results = client.post(
        "/api/whatif",
        params={"since": _since(client, days=30), "project": "support"},
        json={"targets": ["claude-haiku-4-5", "gpt-4o-mini", "gpt-4o", "gemini-2.5-flash"]},
        timeout=120.0,
    ).json()

    rows = [
        [
            r["target_model"],
            usd(r["baseline_usd"]),
            usd(r["projected_usd"]),
            "-" if r["delta_pct"] is None else f"{r['delta_pct']:+.1f}%",
            f"{r['refolded_cache_tokens']:,}" if r["refolded_cache_tokens"] else "-",
            ", ".join(r["warnings"]) or "-",
        ]
        for r in results
    ]
    print(table(rows, ["target", "today", "projected", "delta", "cache refolded", "notes"]))
    print(
        "\n  'cache refolded' counts tokens that are cache reads today but would be billed\n"
        "  at the full input rate on the target model. Quality is never inferred: every\n"
        "  simulation is labelled '" + results[0]["quality"]["status"] + "'."
    )


def show_unpriced(client: httpx.Client) -> None:
    unpriced = client.get("/api/spend/unpriced", params={"since": _since(client)}).json()
    if unpriced:
        section("UNPRICED -- stored with NULL cost, never zero")
        print(
            table(
                [[u["model"], u["provider"] or "-", f"{u['events']:,}"] for u in unpriced],
                ["model", "provider", "events"],
            )
        )


def _since(client: httpx.Client, days: int = 46) -> str:
    from datetime import UTC, datetime, timedelta

    return (datetime.now(UTC) - timedelta(days=days)).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default=DEFAULT_API)
    parser.add_argument("--days", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--skip-ingest", action="store_true", help="Report on existing data.")
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    with httpx.Client(base_url=args.api, headers=headers, timeout=30.0) as client:
        health = wait_for_api(client)
        print(
            f"Tokenomics at {args.api} | pricing snapshot {health['pricing_snapshot'][:12]} "
            f"({health['models_priced']:,} models priced)"
        )

        if not args.skip_ingest:
            ingest(client, args.days, args.batch_size)

        show_chargeback(client)
        show_token_economics(client)
        show_forecast(client)
        show_anomalies(client)
        show_budgets(client)
        show_whatif(client)
        show_unpriced(client)
        print()


if __name__ == "__main__":
    main()
