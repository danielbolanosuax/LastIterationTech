"""
Black-Scholes engine and options analyzer integrated with the trading pipeline.
"""

from __future__ import annotations

from math import exp, log, sqrt
from typing import Dict, Literal, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

OptionType = Literal["call", "put"]


class BlackScholesEngine:
    """European option pricing and Greeks using Black-Scholes."""

    MIN_T = 1e-6
    MIN_SIGMA = 1e-6

    @classmethod
    def _normalize_inputs(cls, s: float, k: float, t: float, sigma: float):
        s = max(float(s), 1e-8)
        k = max(float(k), 1e-8)
        t = max(float(t), cls.MIN_T)
        sigma = max(float(sigma), cls.MIN_SIGMA)
        return s, k, t, sigma

    @classmethod
    def d1(cls, s: float, k: float, t: float, r: float, sigma: float) -> float:
        s, k, t, sigma = cls._normalize_inputs(s, k, t, sigma)
        return (log(s / k) + (r + 0.5 * sigma**2) * t) / (sigma * sqrt(t))

    @classmethod
    def d2(cls, s: float, k: float, t: float, r: float, sigma: float) -> float:
        s, k, t, sigma = cls._normalize_inputs(s, k, t, sigma)
        return cls.d1(s, k, t, r, sigma) - sigma * sqrt(t)

    @classmethod
    def call_price(cls, s: float, k: float, t: float, r: float, sigma: float) -> float:
        d1 = cls.d1(s, k, t, r, sigma)
        d2 = cls.d2(s, k, t, r, sigma)
        return float(s * norm.cdf(d1) - k * exp(-r * t) * norm.cdf(d2))

    @classmethod
    def put_price(cls, s: float, k: float, t: float, r: float, sigma: float) -> float:
        d1 = cls.d1(s, k, t, r, sigma)
        d2 = cls.d2(s, k, t, r, sigma)
        return float(k * exp(-r * t) * norm.cdf(-d2) - s * norm.cdf(-d1))

    @classmethod
    def price(cls, option_type: OptionType, s: float, k: float, t: float, r: float, sigma: float) -> float:
        if option_type == "call":
            return cls.call_price(s, k, t, r, sigma)
        return cls.put_price(s, k, t, r, sigma)

    @classmethod
    def calculate_greeks(
        cls, option_type: OptionType, s: float, k: float, t: float, r: float, sigma: float
    ) -> Dict[str, float]:
        s, k, t, sigma = cls._normalize_inputs(s, k, t, sigma)
        d1 = cls.d1(s, k, t, r, sigma)
        d2 = cls.d2(s, k, t, r, sigma)
        nd1 = norm.pdf(d1)
        sqrt_t = sqrt(t)

        if option_type == "call":
            delta = norm.cdf(d1)
            theta = (-s * nd1 * sigma / (2 * sqrt_t) - r * k * exp(-r * t) * norm.cdf(d2)) / 365.0
            rho = k * t * exp(-r * t) * norm.cdf(d2) / 100.0
        else:
            delta = norm.cdf(d1) - 1.0
            theta = (-s * nd1 * sigma / (2 * sqrt_t) + r * k * exp(-r * t) * norm.cdf(-d2)) / 365.0
            rho = -k * t * exp(-r * t) * norm.cdf(-d2) / 100.0

        gamma = nd1 / (s * sigma * sqrt_t)
        vega = (s * nd1 * sqrt_t) / 100.0

        return {
            "delta": float(delta),
            "gamma": float(gamma),
            "vega": float(vega),
            "theta": float(theta),
            "rho": float(rho),
        }

    @classmethod
    def implied_volatility(
        cls,
        market_price: float,
        option_type: OptionType,
        s: float,
        k: float,
        t: float,
        r: float,
        tol: float = 1e-5,
        max_iter: int = 120,
    ) -> float:
        """Robust implied vol via bisection."""
        market_price = max(float(market_price), 0.0)
        if market_price <= 0.0:
            return 0.001

        low = 1e-4
        high = 5.0

        for _ in range(max_iter):
            mid = (low + high) / 2.0
            model_price = cls.price(option_type, s, k, t, r, mid)
            error = model_price - market_price

            if abs(error) <= tol:
                return float(mid)

            if error > 0:
                high = mid
            else:
                low = mid

        return float((low + high) / 2.0)

    @classmethod
    def get_recommendation(cls, analysis: Dict[str, float]) -> str:
        mispricing = float(analysis.get("price_mispricing_pct", 0.0))
        iv = float(analysis.get("implied_volatility", 0.0))
        moneyness = float(analysis.get("moneyness", 1.0))

        if abs(mispricing) >= 8.0:
            return "UNDERPRICED" if mispricing < 0 else "OVERPRICED"
        if iv >= 0.55:
            return "HIGH_IV"
        if iv <= 0.15:
            return "LOW_IV"
        if moneyness >= 1.08:
            return "DEEP_ITM"
        return "FAIR_VALUE"

    @classmethod
    def option_analysis(
        cls,
        option_type: OptionType,
        s: float,
        k: float,
        t: float,
        r: float,
        market_price: float,
        model_sigma: Optional[float] = None,
    ) -> Dict[str, float]:
        base_sigma = max(float(model_sigma), 0.05) if model_sigma is not None else 0.20
        theoretical = cls.price(option_type, s, k, t, r, base_sigma)
        iv = cls.implied_volatility(market_price, option_type, s, k, t, r)
        greeks = cls.calculate_greeks(option_type, s, k, t, r, iv)
        moneyness = s / max(k, 1e-8)
        mispricing = 0.0 if theoretical <= 1e-8 else ((market_price - theoretical) / theoretical) * 100.0

        analysis = {
            "option_type": option_type.upper(),
            "underlying_price": float(s),
            "strike": float(k),
            "time_to_expiry_years": float(t),
            "time_to_expiry_days": float(t * 365.0),
            "risk_free_rate": float(r),
            "market_price": float(market_price),
            "theoretical_price": float(theoretical),
            "implied_volatility": float(iv),
            "moneyness": float(moneyness),
            "price_mispricing_pct": float(mispricing),
            "delta": greeks["delta"],
            "gamma": greeks["gamma"],
            "vega": greeks["vega"],
            "theta": greeks["theta"],
            "rho": greeks["rho"],
        }
        analysis["recommendation"] = cls.get_recommendation(analysis)
        return analysis


class OptionsAnalyzer:
    """Generates and summarizes a synthetic option chain around spot."""

    def __init__(self, risk_free_rate: float = 0.045):
        self.risk_free_rate = float(risk_free_rate)
        self.engine = BlackScholesEngine()

    def _market_price_with_skew(
        self, theoretical_price: float, strike: float, spot: float, option_type: OptionType
    ) -> float:
        moneyness_gap = (strike / max(spot, 1e-8)) - 1.0
        skew = 0.12 * moneyness_gap
        type_bias = 0.007 if option_type == "call" else -0.007
        spread = 0.01 + abs(moneyness_gap) * 0.02
        adjusted = theoretical_price * (1.0 + skew + type_bias) + spread
        return float(max(adjusted, 0.01))

    def analyze_symbol_options(
        self,
        symbol: str,
        spot_price: float,
        annualized_volatility: float,
        days_to_expiry: int = 30,
        strikes_range: float = 0.20,
        strike_step_pct: float = 0.05,
    ) -> pd.DataFrame:
        spot_price = float(max(spot_price, 0.01))
        days_to_expiry = int(max(days_to_expiry, 1))
        annualized_volatility = float(np.clip(annualized_volatility, 0.05, 2.0))
        t = days_to_expiry / 365.0

        low = spot_price * (1.0 - strikes_range)
        high = spot_price * (1.0 + strikes_range)
        strike_step = max(spot_price * strike_step_pct, 0.5)
        strikes = np.arange(low, high + strike_step, strike_step)

        rows = []
        for strike in strikes:
            k = float(max(strike, 0.01))
            for option_type in ("call", "put"):
                theoretical = self.engine.price(option_type, spot_price, k, t, self.risk_free_rate, annualized_volatility)
                market_price = self._market_price_with_skew(theoretical, k, spot_price, option_type)
                row = self.engine.option_analysis(
                    option_type=option_type,
                    s=spot_price,
                    k=k,
                    t=t,
                    r=self.risk_free_rate,
                    market_price=market_price,
                    model_sigma=annualized_volatility,
                )
                row["symbol"] = symbol.upper()
                rows.append(row)

        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return df.sort_values(["option_type", "strike"]).reset_index(drop=True)

    @staticmethod
    def _direction_from_row(row: pd.Series) -> str:
        rec = str(row.get("recommendation", "FAIR_VALUE"))
        option_type = str(row.get("option_type", "CALL")).upper()

        if rec == "UNDERPRICED":
            return "BUY_BIAS" if option_type == "CALL" else "SELL_BIAS"
        if rec == "OVERPRICED":
            return "SELL_BIAS" if option_type == "CALL" else "BUY_BIAS"
        return "NEUTRAL"

    def summarize_option_chain(self, option_chain: pd.DataFrame) -> Dict[str, object]:
        if option_chain is None or option_chain.empty:
            return {
                "available": False,
                "directional_bias": "NEUTRAL",
                "recommendation": "NO_DATA",
                "signal_confidence": 0.5,
                "avg_implied_volatility": 0.0,
            }

        idx = option_chain["price_mispricing_pct"].abs().idxmax()
        best = option_chain.loc[idx]
        bias = self._direction_from_row(best)
        mispricing_abs = float(abs(best["price_mispricing_pct"]))
        confidence = float(min(0.95, 0.5 + mispricing_abs / 40.0))

        top = (
            option_chain.reindex(option_chain["price_mispricing_pct"].abs().sort_values(ascending=False).index)
            .head(3)
            .copy()
        )
        top_rows = top[
            [
                "option_type",
                "strike",
                "market_price",
                "theoretical_price",
                "implied_volatility",
                "price_mispricing_pct",
                "recommendation",
            ]
        ].to_dict(orient="records")

        return {
            "available": True,
            "directional_bias": bias,
            "recommendation": str(best["recommendation"]),
            "signal_confidence": confidence,
            "avg_implied_volatility": float(option_chain["implied_volatility"].mean()),
            "best_option": {
                "option_type": str(best["option_type"]),
                "strike": float(best["strike"]),
                "market_price": float(best["market_price"]),
                "theoretical_price": float(best["theoretical_price"]),
                "price_mispricing_pct": float(best["price_mispricing_pct"]),
                "recommendation": str(best["recommendation"]),
            },
            "top_opportunities": top_rows,
        }
