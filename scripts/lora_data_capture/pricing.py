"""Cost estimation for Grok API usage.

Prices are USD per million tokens from https://docs.x.ai/docs/models
(checked 2026-07-10). Update the table when xAI changes pricing, or override
a run with --price-input / --price-output.

Estimates bill all prompt tokens at the full input rate (no cached-input
discount), so they are a slight upper bound when xAI applies prompt caching.
"""

from __future__ import annotations

# model id -> (input $/Mtok, output $/Mtok)
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "grok-4.3": (1.25, 2.50),
    "grok-4.5": (2.00, 6.00),
    "grok-4.20-0309-reasoning": (1.25, 2.50),
    "grok-4.20-0309-non-reasoning": (1.25, 2.50),
    "grok-4.20-multi-agent-0309": (1.25, 2.50),
}


def resolve_prices(
    model: str,
    price_input: float | None = None,
    price_output: float | None = None,
) -> tuple[float, float] | None:
    """(input $/Mtok, output $/Mtok) for a model, or None when unknown.

    CLI overrides take precedence per side; a model absent from the table is
    priceable only when BOTH overrides are supplied.
    """
    table = PRICES_PER_MTOK.get(model)
    if table is None:
        if price_input is not None and price_output is not None:
            return price_input, price_output
        return None
    return (
        table[0] if price_input is None else price_input,
        table[1] if price_output is None else price_output,
    )


def estimate_cost_usd(
    prices: tuple[float, float] | None,
    input_tokens: int,
    output_tokens: int,
) -> float | None:
    if prices is None:
        return None
    return input_tokens * prices[0] / 1e6 + output_tokens * prices[1] / 1e6
