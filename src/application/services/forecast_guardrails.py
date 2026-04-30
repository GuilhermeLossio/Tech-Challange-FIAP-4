from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ForecastGuardrailResult:
    raw_model_close: float
    constrained_close: float
    raw_return: float
    constrained_return: float
    return_cap: float
    applied: bool
    method: str


def apply_standard_forecast_guardrail(
    *,
    raw_model_close: float,
    current_close: float,
    recent_window: np.ndarray,
    volatility_multiplier: float = 3.0,
    min_return_cap: float = 0.03,
    max_return_cap: float = 0.12,
    recent_periods: int = 20,
) -> ForecastGuardrailResult:
    if abs(current_close) <= 1e-8:
        return ForecastGuardrailResult(
            raw_model_close=float(raw_model_close),
            constrained_close=float(raw_model_close),
            raw_return=0.0,
            constrained_return=0.0,
            return_cap=0.0,
            applied=False,
            method="disabled_zero_reference_close",
        )

    recent_returns = _compute_daily_returns(recent_window)
    if len(recent_returns) > 0:
        effective_returns = recent_returns[-min(recent_periods, len(recent_returns)) :]
        realized_move = float(np.mean(np.abs(effective_returns)))
    else:
        realized_move = 0.01

    return_cap = float(
        min(
            max(realized_move * volatility_multiplier, min_return_cap),
            max_return_cap,
        )
    )
    raw_return = float(raw_model_close / current_close - 1.0)
    constrained_return = float(np.clip(raw_return, -return_cap, return_cap))
    constrained_close = float(max(current_close * (1.0 + constrained_return), 0.0))

    return ForecastGuardrailResult(
        raw_model_close=float(raw_model_close),
        constrained_close=constrained_close,
        raw_return=raw_return,
        constrained_return=constrained_return,
        return_cap=return_cap,
        applied=abs(constrained_return - raw_return) > 1e-9,
        method="recent_realized_volatility_return_cap",
    )


def clamp_interval_to_guardrail_band(
    *,
    predicted_close: float,
    interval_width: float,
    reference_close: float,
    return_cap: float,
) -> tuple[float, float]:
    if abs(reference_close) <= 1e-8 or return_cap <= 0.0:
        lower_bound = max(predicted_close - interval_width, 0.0)
        upper_bound = predicted_close + interval_width
        return float(lower_bound), float(upper_bound)

    band_lower = max(reference_close * (1.0 - return_cap), 0.0)
    band_upper = reference_close * (1.0 + return_cap)
    lower_bound = max(predicted_close - interval_width, band_lower)
    upper_bound = min(predicted_close + interval_width, band_upper)
    if upper_bound < lower_bound:
        upper_bound = lower_bound
    return float(lower_bound), float(upper_bound)


def _compute_daily_returns(values: np.ndarray) -> np.ndarray:
    if len(values) <= 1:
        return np.empty(0, dtype=np.float64)
    previous = values[:-1]
    current = values[1:]
    returns = np.zeros(len(previous), dtype=np.float64)
    non_zero_mask = np.abs(previous) > 1e-8
    returns[non_zero_mask] = current[non_zero_mask] / previous[non_zero_mask] - 1.0
    return returns
