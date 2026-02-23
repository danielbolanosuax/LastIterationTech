"""
Motor avanzado de indicadores tecnicos (estilo TradingView) sin dependencias externas.

Librerias permitidas:
- numpy
- pandas
- scipy.stats
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import warnings


# Alias solicitado: Dict con todas las senales y valores.
SignalBundle = Dict[str, Any]


class AdvancedIndicatorEngine:
    """
    Engine que calcula 6 indicadores avanzados y produce una senal compuesta.

    Fuentes matematicas:
    - RSI (Wilder RMA), Stochastic y MACD: equivalentes a formulas de TradingView.
    - Bulls vs Bears (normalizado): inspirado en BvB v6.
    - Koncorde (Blai5): adaptacion por lotes de NVI/PVI + osciladores.
    - Liquidity Pools: adaptacion por ventanas del enfoque de zonas de LuxAlgo.
    """

    def __init__(self):
        # Buffers acumulativos para escenarios incremental/streaming.
        self.nvi_buffer: float = 0.0
        self.pvi_buffer: float = 0.0
        self.rma_gain_buffer: Optional[float] = None
        self.rma_loss_buffer: Optional[float] = None
        self._warned_sequential = False

    # ------------------------------------------------------------------
    # API principal
    # ------------------------------------------------------------------
    def calculate_all(
        self,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        open_: np.ndarray,
        volume: np.ndarray,
    ) -> Optional[SignalBundle]:
        """
        Metodo principal.

        Requisitos:
        - Minimo 100 barras (si no, retorna None).
        - Limpieza robusta de NaN con forward fill + backfill.
        - Volumen 0 -> 1 para evitar divisiones por cero.
        """
        close, high, low, open_, volume = self._sanitize_ohlcv(
            close=close, high=high, low=low, open_=open_, volume=volume
        )

        n = len(close)
        if n < 100:
            return None

        rsi_data = self.calculate_rsi(close)
        stoch_data = self.calculate_stochastic(close=close, high=high, low=low)
        macd_data = self.calculate_macd(close=close)
        bvb_data = self.calculate_bulls_vs_bears(close=close, high=high, low=low)
        koncorde_data = self.calculate_koncorde(
            close=close, high=high, low=low, open_=open_, volume=volume
        )
        liquidity_data = self.calculate_liquidity_pools(
            close=close, high=high, low=low, open_=open_
        )

        raw_scores = {
            "rsi": self._score_rsi(rsi_data),
            "stochastic": self._score_stochastic(stoch_data),
            "macd": self._score_macd(macd_data),
            "bvb": self._score_bvb(bvb_data),
            "koncorde": self._score_koncorde(koncorde_data),
            "liquidity": self._score_liquidity(liquidity_data),
        }

        weights = {
            "rsi": 0.15,
            "stochastic": 0.10,
            "macd": 0.20,
            "bvb": 0.15,
            "koncorde": 0.25,
            "liquidity": 0.15,
        }

        weighted_scores = {k: float(raw_scores[k] * weights[k]) for k in raw_scores}
        composite_score = float(np.clip(sum(weighted_scores.values()), -1.0, 1.0))
        final_signal = self._final_signal_from_score(composite_score)
        confidence = float(self._compute_confidence(np.array(list(raw_scores.values()), dtype=float)))

        signal_breakdown = {
            name: {
                "raw_score": float(raw_scores[name]),
                "weight": float(weights[name]),
                "weighted_contribution": float(weighted_scores[name]),
            }
            for name in raw_scores
        }

        # Salida completa con estructura util y compatibilidad de campos clave.
        signal_bundle: SignalBundle = {
            "rsi": float(rsi_data["rsi"]),
            "macd": float(macd_data["macd"]),
            "stoch_k": float(stoch_data["stoch_k"]),
            "stoch_d": float(stoch_data["stoch_d"]),
            "bvb_total": float(bvb_data["bvb_total"]),
            "k_verde": float(koncorde_data["k_verde"]),
            "k_marron": float(koncorde_data["k_marron"]),
            "k_azul": float(koncorde_data["k_azul"]),
            "lp_signal": str(liquidity_data["lp_signal"]),
            "rsi_data": rsi_data,
            "stochastic_data": stoch_data,
            "macd_data": macd_data,
            "bvb_data": bvb_data,
            "koncorde_data": koncorde_data,
            "liquidity_data": liquidity_data,
            "composite_score": composite_score,
            "final_signal": final_signal,
            "confidence": confidence,
            "signal_breakdown": signal_breakdown,
        }
        return signal_bundle

    # ------------------------------------------------------------------
    # Indicador 1: RSI completo con divergencias
    # ------------------------------------------------------------------
    def calculate_rsi(
        self,
        close: np.ndarray,
        length: int = 14,
        lookback_left: int = 5,
        lookback_right: int = 5,
        range_lower: int = 5,
        range_upper: int = 60,
    ) -> Dict[str, Any]:
        """
        RSI con RMA de Wilder + divergencias regulares.

        Formula:
        - avg_gain = RMA(gain, length)
        - avg_loss = RMA(loss, length)
        - RSI = 100 - 100 / (1 + avg_gain/avg_loss)
        """
        close = self._sanitize_array(close)
        change = np.diff(close, prepend=close[0])
        gain = np.maximum(change, 0.0)
        loss = np.maximum(-change, 0.0)

        avg_gain = self._rma(gain, length)
        avg_loss = self._rma(loss, length)

        rsi_series = np.full(len(close), np.nan, dtype=float)
        valid = ~np.isnan(avg_gain) & ~np.isnan(avg_loss)
        with np.errstate(divide="ignore", invalid="ignore"):
            rs = np.divide(avg_gain, avg_loss, where=valid)
            rsi_calc = 100.0 - (100.0 / (1.0 + rs))
            rsi_series[valid] = rsi_calc[valid]

        # Reglas limite de TradingView para perdidas/ganancias cero.
        zero_loss = valid & (avg_loss == 0.0)
        zero_gain = valid & (avg_gain == 0.0)
        both_zero = zero_loss & zero_gain
        rsi_series[zero_loss] = 100.0
        rsi_series[zero_gain] = 0.0
        rsi_series[both_zero] = 50.0

        current_rsi = self._last_valid(rsi_series, default=50.0)

        if current_rsi > 70.0:
            rsi_signal = "OVERBOUGHT"
        elif current_rsi < 30.0:
            rsi_signal = "OVERSOLD"
        elif current_rsi > 50.0:
            rsi_signal = "BULLISH"
        elif current_rsi < 50.0:
            rsi_signal = "BEARISH"
        else:
            rsi_signal = "NEUTRAL"

        price_pivot_lows = self._pivot_flags(close, lookback_left, lookback_right, mode="low")
        price_pivot_highs = self._pivot_flags(close, lookback_left, lookback_right, mode="high")
        rsi_pivot_lows = self._pivot_flags(rsi_series, lookback_left, lookback_right, mode="low")
        rsi_pivot_highs = self._pivot_flags(rsi_series, lookback_left, lookback_right, mode="high")

        bullish_idx = self._find_latest_divergence(
            price_series=close,
            osc_series=rsi_series,
            price_pivots=price_pivot_lows,
            osc_pivots=rsi_pivot_lows,
            range_lower=range_lower,
            range_upper=range_upper,
            divergence_type="bullish",
        )
        bearish_idx = self._find_latest_divergence(
            price_series=close,
            osc_series=rsi_series,
            price_pivots=price_pivot_highs,
            osc_pivots=rsi_pivot_highs,
            range_lower=range_lower,
            range_upper=range_upper,
            divergence_type="bearish",
        )

        if bullish_idx is not None and (bearish_idx is None or bullish_idx >= bearish_idx):
            rsi_divergence = "BULLISH_DIV"
        elif bearish_idx is not None:
            rsi_divergence = "BEARISH_DIV"
        else:
            rsi_divergence = "NONE"

        rsi_strength = float(np.clip(abs(current_rsi - 50.0) / 50.0, 0.0, 1.0))

        # Guardamos buffers finales de RMA por compatibilidad incremental.
        self.rma_gain_buffer = self._last_valid(avg_gain, default=None)
        self.rma_loss_buffer = self._last_valid(avg_loss, default=None)

        return {
            "rsi": float(current_rsi),
            "rsi_signal": rsi_signal,
            "rsi_divergence": rsi_divergence,
            "rsi_series": rsi_series,
            "rsi_strength": rsi_strength,
        }

    # ------------------------------------------------------------------
    # Indicador 2: Stochastic Oscillator
    # ------------------------------------------------------------------
    def calculate_stochastic(
        self,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        period_k: int = 14,
        smooth_k: int = 1,
        period_d: int = 3,
    ) -> Dict[str, Any]:
        """
        Stochastic %K / %D (equivalente a ta.stoch + suavizado SMA en TradingView).
        """
        close = self._sanitize_array(close)
        high = self._sanitize_array(high)
        low = self._sanitize_array(low)

        lowest_low = self._rolling_min(low, period_k)
        highest_high = self._rolling_max(high, period_k)
        denom = highest_high - lowest_low

        stoch_raw = np.full(len(close), np.nan, dtype=float)
        valid = ~np.isnan(denom) & (denom != 0.0)
        stoch_raw[valid] = 100.0 * (close[valid] - lowest_low[valid]) / denom[valid]
        stoch_raw[~valid & ~np.isnan(denom)] = 50.0

        k = self._sma(stoch_raw, smooth_k)
        d = self._sma(k, period_d)

        current_k = self._last_valid(k, default=50.0)
        current_d = self._last_valid(d, default=50.0)
        stoch_momentum = float(current_k - current_d)

        if current_k > 80.0:
            stoch_signal = "OVERBOUGHT"
        elif current_k < 20.0:
            stoch_signal = "OVERSOLD"
        else:
            stoch_signal = "NEUTRAL"

        if len(k) >= 2 and len(d) >= 2 and self._is_finite_pair(k[-2], d[-2]) and self._is_finite_pair(k[-1], d[-1]):
            if k[-1] > d[-1] and k[-2] <= d[-2]:
                stoch_cross = "BULLISH_CROSS"
            elif k[-1] < d[-1] and k[-2] >= d[-2]:
                stoch_cross = "BEARISH_CROSS"
            else:
                stoch_cross = "NONE"
        else:
            stoch_cross = "NONE"

        return {
            "stoch_k": float(current_k),
            "stoch_d": float(current_d),
            "stoch_signal": stoch_signal,
            "stoch_cross": stoch_cross,
            "stoch_momentum": stoch_momentum,
        }

    # ------------------------------------------------------------------
    # Indicador 3: MACD completo
    # ------------------------------------------------------------------
    def calculate_macd(
        self,
        close: np.ndarray,
        fast: int = 12,
        slow: int = 26,
        signal_len: int = 9,
    ) -> Dict[str, Any]:
        """
        MACD clasico de TradingView:
        - macd = EMA(fast) - EMA(slow)
        - signal = EMA(macd, signal_len)
        - histogram = macd - signal
        """
        close = self._sanitize_array(close)
        ema_fast = self._ema_tv(close, fast)
        ema_slow = self._ema_tv(close, slow)
        macd_line = ema_fast - ema_slow
        signal_line = self._ema_tv(macd_line, signal_len)
        histogram = macd_line - signal_line

        current_macd = self._last_valid(macd_line, default=0.0)
        current_signal = self._last_valid(signal_line, default=0.0)
        current_hist = self._last_valid(histogram, default=0.0)

        if len(macd_line) >= 2 and len(signal_line) >= 2 and self._is_finite_pair(macd_line[-2], signal_line[-2]) and self._is_finite_pair(macd_line[-1], signal_line[-1]):
            if macd_line[-1] > signal_line[-1] and macd_line[-2] <= signal_line[-2]:
                macd_cross = "BULLISH_CROSS"
            elif macd_line[-1] < signal_line[-1] and macd_line[-2] >= signal_line[-2]:
                macd_cross = "BEARISH_CROSS"
            else:
                macd_cross = "NONE"
        else:
            macd_cross = "NONE"

        hist_delta = 0.0
        if len(histogram) >= 2 and np.isfinite(histogram[-1]) and np.isfinite(histogram[-2]):
            hist_delta = float(histogram[-1] - histogram[-2])

        if current_hist >= 0.0 and hist_delta > 0.0:
            macd_momentum = "ACCELERATING_UP"
        elif current_hist >= 0.0 and hist_delta <= 0.0:
            macd_momentum = "DECELERATING_UP"
        elif current_hist < 0.0 and hist_delta < 0.0:
            macd_momentum = "ACCELERATING_DOWN"
        else:
            macd_momentum = "DECELERATING_DOWN"

        if len(histogram) >= 2 and np.isfinite(histogram[-1]) and np.isfinite(histogram[-2]):
            if histogram[-2] <= 0.0 and histogram[-1] > 0.0:
                macd_zero_cross = "CROSS_UP"
            elif histogram[-2] >= 0.0 and histogram[-1] < 0.0:
                macd_zero_cross = "CROSS_DOWN"
            else:
                macd_zero_cross = "NONE"
        else:
            macd_zero_cross = "NONE"

        return {
            "macd": float(current_macd),
            "macd_signal": float(current_signal),
            "macd_histogram": float(current_hist),
            "macd_cross": macd_cross,
            "macd_momentum": macd_momentum,
            "macd_zero_cross": macd_zero_cross,
        }

    # ------------------------------------------------------------------
    # Indicador 4: Bulls vs Bears (BvB)
    # ------------------------------------------------------------------
    def calculate_bulls_vs_bears(
        self,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        length: int = 14,
        ma_type: str = "EMA",
        norm_type: str = "Normalized",
        bars_back: int = 120,
        tline: float = 80.0,
    ) -> Dict[str, Any]:
        """
        Bulls vs Bears normalizado:
        bulls_raw = high - MA(close)
        bears_raw = MA(close) - low
        total = norm_bulls - norm_bears
        """
        close = self._sanitize_array(close)
        high = self._sanitize_array(high)
        low = self._sanitize_array(low)

        ma = self._ma(close, length=length, ma_type=ma_type)
        bulls_raw = high - ma
        bears_raw = ma - low

        if norm_type.lower() == "normalized":
            min_bulls = self._rolling_min(bulls_raw, bars_back)
            max_bulls = self._rolling_max(bulls_raw, bars_back)
            min_bears = self._rolling_min(bears_raw, bars_back)
            max_bears = self._rolling_max(bears_raw, bars_back)

            norm_bulls = self._normalize_centered_100(bulls_raw, min_bulls, max_bulls)
            norm_bears = self._normalize_centered_100(bears_raw, min_bears, max_bears)
            total = norm_bulls - norm_bears
        else:
            total = bulls_raw - bears_raw

        current_total = self._last_valid(total, default=0.0)
        current_bulls_raw = self._last_valid(bulls_raw, default=0.0)
        current_bears_raw = self._last_valid(bears_raw, default=0.0)

        if current_total > tline:
            bvb_signal = "EXTREME_BULL"
        elif current_total < -tline:
            bvb_signal = "EXTREME_BEAR"
        elif current_total > 0:
            bvb_signal = "BULL"
        elif current_total < 0:
            bvb_signal = "BEAR"
        else:
            bvb_signal = "NEUTRAL"

        if len(total) >= 2 and np.isfinite(total[-1]) and np.isfinite(total[-2]):
            if total[-2] <= 0.0 and total[-1] > 0.0:
                bvb_zero_cross = "CROSS_UP"
            elif total[-2] >= 0.0 and total[-1] < 0.0:
                bvb_zero_cross = "CROSS_DOWN"
            else:
                bvb_zero_cross = "NONE"
        else:
            bvb_zero_cross = "NONE"

        bvb_strength = float(np.clip(abs(current_total) / 100.0, 0.0, 1.0))

        return {
            "bvb_total": float(current_total),
            "bvb_bulls_raw": float(current_bulls_raw),
            "bvb_bears_raw": float(current_bears_raw),
            "bvb_signal": bvb_signal,
            "bvb_zero_cross": bvb_zero_cross,
            "bvb_strength": bvb_strength,
        }

    # ------------------------------------------------------------------
    # Indicador 5: Koncorde (Blai5 adaptado)
    # ------------------------------------------------------------------
    def calculate_koncorde(
        self,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        open_: np.ndarray,
        volume: np.ndarray,
        length_ema: int = 255,
        m: int = 15,
    ) -> Dict[str, Any]:
        """
        Koncorde adaptado:
        - Azul: NVI normalizado
        - Marron: combinacion de RSI/MFI/BollOsc/Stoch
        - Verde: Marron + oscilador PVI
        """
        _ = length_ema  # parametro expuesto para compatibilidad con el indicador original
        close = self._sanitize_array(close)
        high = self._sanitize_array(high)
        low = self._sanitize_array(low)
        open_ = self._sanitize_array(open_)
        volume = self._sanitize_array(volume)
        volume = np.where(volume <= 0.0, 1.0, volume)

        n = len(close)
        if not self._warned_sequential:
            warnings.warn(
                "NVI/PVI se calcula con bucle secuencial Python (sin numba).",
                RuntimeWarning,
            )
            self._warned_sequential = True

        # ---------------------------
        # Componente AZUL (NVI)
        # ---------------------------
        nvi = np.zeros(n, dtype=float)
        for i in range(1, n):
            prev_close = close[i - 1] if close[i - 1] != 0.0 else 1e-12
            ret = (close[i] - close[i - 1]) / prev_close
            if volume[i] < volume[i - 1]:
                nvi[i] = nvi[i - 1] + ret * volume[i]
            else:
                nvi[i] = nvi[i - 1]

        nvim = self._ema_tv(nvi, m)
        nvimax = self._rolling_max(nvim, 90)
        nvimin = self._rolling_min(nvim, 90)
        azul = self._safe_ratio_times_100(nvi - nvim, nvimax - nvimin)

        # ---------------------------
        # Componente MARRON
        # ---------------------------
        tprice = (open_ + high + low + close) / 4.0
        src = (high + low + close) / 3.0  # hlc3

        src_delta = np.diff(src, prepend=src[0])
        pos_flow = np.where(src_delta > 0.0, volume * src, 0.0)
        neg_flow = np.where(src_delta < 0.0, volume * src, 0.0)
        upper = self._rolling_sum(pos_flow, 14)
        lower = self._rolling_sum(neg_flow, 14)
        xmf = self._rsi_from_up_down(upper, lower)

        basis = self._sma(tprice, 25)
        dev = 2.0 * self._rolling_std(tprice, 25)
        upper_bb = basis + dev
        lower_bb = basis - dev
        ob1 = (upper_bb + lower_bb) / 2.0
        ob2 = upper_bb - lower_bb
        boll_osc = np.zeros(n, dtype=float)
        valid_ob = np.isfinite(ob1) & np.isfinite(ob2) & (ob2 != 0.0)
        boll_osc[valid_ob] = ((tprice[valid_ob] - ob1[valid_ob]) / ob2[valid_ob]) * 100.0

        xrsi = self._rsi_wilder_series(tprice, length=14)

        ll = self._rolling_min(low, 21)
        hh = self._rolling_max(high, 21)
        stoc_raw = np.full(n, np.nan, dtype=float)
        den_stoc = hh - ll
        valid_stoc = np.isfinite(den_stoc) & (den_stoc != 0.0)
        stoc_raw[valid_stoc] = 100.0 * (tprice[valid_stoc] - ll[valid_stoc]) / den_stoc[valid_stoc]
        stoc_raw[~valid_stoc & np.isfinite(den_stoc)] = 50.0
        stoc = self._sma(stoc_raw, 3)

        marron = (xrsi + xmf + boll_osc + (stoc / 3.0)) / 2.0
        media = self._ema_tv(marron, m)

        # ---------------------------
        # Componente VERDE (PVI)
        # ---------------------------
        pvi = np.zeros(n, dtype=float)
        for i in range(1, n):
            prev_close = close[i - 1] if close[i - 1] != 0.0 else 1e-12
            ret = (close[i] - close[i - 1]) / prev_close
            if volume[i] > volume[i - 1]:
                pvi[i] = pvi[i - 1] + ret * volume[i]
            else:
                pvi[i] = pvi[i - 1]

        pvim = self._ema_tv(pvi, m)
        pvimax = self._rolling_max(pvim, 90)
        pvimin = self._rolling_min(pvim, 90)
        oscp = self._safe_ratio_times_100(pvi - pvim, pvimax - pvimin)

        verde = marron + oscp

        k_verde = self._last_valid(verde, default=0.0)
        k_marron = self._last_valid(marron, default=0.0)
        k_azul = self._last_valid(azul, default=0.0)
        k_media = self._last_valid(media, default=0.0)

        if k_verde > k_marron and k_azul > 0:
            k_signal = "STRONG_BUY"
        elif k_verde > k_marron and k_azul < 0:
            k_signal = "WEAK_BUY"
        elif k_verde < k_marron and k_azul < 0:
            k_signal = "STRONG_SELL"
        elif k_verde < k_marron and k_azul > 0:
            k_signal = "WEAK_SELL"
        else:
            k_signal = "NEUTRAL"

        if k_azul > 0 and k_verde > k_media:
            k_smart_money = "ACCUMULATING"
        elif k_azul < 0 and k_verde < k_media:
            k_smart_money = "DISTRIBUTING"
        else:
            k_smart_money = "NEUTRAL"

        k_verde_vs_media = "ABOVE" if k_verde >= k_media else "BELOW"

        self.nvi_buffer = float(nvi[-1])
        self.pvi_buffer = float(pvi[-1])

        return {
            "k_verde": float(k_verde),
            "k_marron": float(k_marron),
            "k_azul": float(k_azul),
            "k_media": float(k_media),
            "k_signal": k_signal,
            "k_smart_money": k_smart_money,
            "k_verde_vs_media": k_verde_vs_media,
        }

    # ------------------------------------------------------------------
    # Indicador 6: Liquidity Pools (adaptado)
    # ------------------------------------------------------------------
    def calculate_liquidity_pools(
        self,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        open_: np.ndarray,
        zone_contact_min: int = 2,
        bars_between_contacts: int = 5,
        confirmation_bars: int = 10,
        proximity_threshold: float = 0.005,
    ) -> Dict[str, Any]:
        """
        Detecta zonas de soporte/resistencia por multiples contactos con mecha.
        """
        close = self._sanitize_array(close)
        high = self._sanitize_array(high)
        low = self._sanitize_array(low)
        _ = self._sanitize_array(open_)  # reservado para futuras reglas de cuerpo de vela

        n = len(close)
        window = max(confirmation_bars * 4, 30)

        bear_zones: List[Tuple[float, float]] = []  # (bot, top)
        bull_zones: List[Tuple[float, float]] = []  # (bot, top)

        for end in range(window - 1, n):
            start = end - window + 1
            w_high = high[start : end + 1]
            w_low = low[start : end + 1]
            w_close = close[start : end + 1]

            # Zona de resistencia (bear pool)
            hst_high = float(np.max(w_high))
            res_touches = np.where((w_high >= hst_high * 0.998) & (w_close < hst_high))[0]
            if self._count_spaced_contacts(res_touches, bars_between_contacts) >= zone_contact_min:
                zone = (hst_high * 0.998, hst_high)
                self._append_zone_unique(bear_zones, zone)

            # Zona de soporte (bull pool)
            lst_low = float(np.min(w_low))
            sup_touches = np.where((w_low <= lst_low * 1.002) & (w_close > lst_low))[0]
            if self._count_spaced_contacts(sup_touches, bars_between_contacts) >= zone_contact_min:
                zone = (lst_low, lst_low * 1.002)
                self._append_zone_unique(bull_zones, zone)

        current = float(close[-1])
        prev = float(close[-2]) if len(close) > 1 else current

        resistance_levels = sorted({float(z[1]) for z in bear_zones})
        support_levels = sorted({float(z[0]) for z in bull_zones})

        near_resistance = any(abs(current - lvl) / current < proximity_threshold for lvl in resistance_levels)
        near_support = any(abs(current - lvl) / current < proximity_threshold for lvl in support_levels)
        inside_zone = any(bot <= current <= top for (bot, top) in (bear_zones + bull_zones))

        lp_breakout = "NONE"
        if any(prev <= top and current > top for (_, top) in bear_zones):
            lp_breakout = "BULLISH_BREAKOUT"
        elif any(prev >= bot and current < bot for (bot, _) in bull_zones):
            lp_breakout = "BEARISH_BREAKDOWN"

        if lp_breakout != "NONE":
            lp_signal = "BREAKOUT"
        elif near_resistance:
            lp_signal = "NEAR_RESISTANCE"
        elif near_support:
            lp_signal = "NEAR_SUPPORT"
        else:
            lp_signal = "NEUTRAL"

        all_levels = resistance_levels + support_levels
        if all_levels and current != 0.0:
            lp_distance_to_nearest = float(min(abs(current - lvl) / current for lvl in all_levels))
        else:
            lp_distance_to_nearest = 1.0

        return {
            "lp_near_resistance": bool(near_resistance),
            "lp_near_support": bool(near_support),
            "lp_inside_zone": bool(inside_zone),
            "lp_breakout": lp_breakout,
            "lp_resistance_levels": [float(x) for x in resistance_levels],
            "lp_support_levels": [float(x) for x in support_levels],
            "lp_signal": lp_signal,
            "lp_distance_to_nearest": float(lp_distance_to_nearest),
        }

    # ------------------------------------------------------------------
    # Scoring de cada indicador
    # ------------------------------------------------------------------
    def _score_rsi(self, rsi_data: Dict[str, Any]) -> float:
        """
        Reglas base:
        - Sobrevendido = +0.6, Sobrecomprado = -0.6
        - Divergencia alcista = +0.8, bajista = -0.8
        """
        score = 0.0
        signal = rsi_data.get("rsi_signal", "NEUTRAL")
        div = rsi_data.get("rsi_divergence", "NONE")
        strength = float(rsi_data.get("rsi_strength", 0.0))

        if signal == "OVERSOLD":
            score += 0.6
        elif signal == "OVERBOUGHT":
            score -= 0.6
        elif signal == "BULLISH":
            score += 0.25
        elif signal == "BEARISH":
            score -= 0.25

        if div == "BULLISH_DIV":
            score += 0.8
        elif div == "BEARISH_DIV":
            score -= 0.8

        score += np.sign(score if score != 0.0 else (rsi_data.get("rsi", 50.0) - 50.0)) * 0.1 * strength
        return float(np.clip(score, -1.0, 1.0))

    def _score_stochastic(self, stoch_data: Dict[str, Any]) -> float:
        """
        Reglas base:
        - Sobrevendido + cruce alcista = +0.7
        - Sobrecomprado + cruce bajista = -0.7
        """
        signal = stoch_data.get("stoch_signal", "NEUTRAL")
        cross = stoch_data.get("stoch_cross", "NONE")
        momentum = float(stoch_data.get("stoch_momentum", 0.0))

        if signal == "OVERSOLD" and cross == "BULLISH_CROSS":
            score = 0.7
        elif signal == "OVERBOUGHT" and cross == "BEARISH_CROSS":
            score = -0.7
        else:
            score = 0.0
            if signal == "OVERSOLD":
                score += 0.3
            elif signal == "OVERBOUGHT":
                score -= 0.3

            if cross == "BULLISH_CROSS":
                score += 0.4
            elif cross == "BEARISH_CROSS":
                score -= 0.4

            score += float(np.tanh(momentum / 20.0) * 0.2)

        return float(np.clip(score, -1.0, 1.0))

    def _score_macd(self, macd_data: Dict[str, Any]) -> float:
        """
        Reglas base:
        - Cruce alcista + histograma positivo creciente = +0.8
        - Cruce bajista + histograma negativo decreciente = -0.8
        """
        cross = macd_data.get("macd_cross", "NONE")
        momentum = macd_data.get("macd_momentum", "DECELERATING_UP")
        zero_cross = macd_data.get("macd_zero_cross", "NONE")
        hist = float(macd_data.get("macd_histogram", 0.0))

        if cross == "BULLISH_CROSS" and momentum == "ACCELERATING_UP":
            score = 0.8
        elif cross == "BEARISH_CROSS" and momentum == "ACCELERATING_DOWN":
            score = -0.8
        else:
            score = 0.0
            if cross == "BULLISH_CROSS":
                score += 0.5
            elif cross == "BEARISH_CROSS":
                score -= 0.5

            momentum_map = {
                "ACCELERATING_UP": 0.35,
                "DECELERATING_UP": 0.15,
                "DECELERATING_DOWN": -0.15,
                "ACCELERATING_DOWN": -0.35,
            }
            score += momentum_map.get(momentum, 0.0)

            if zero_cross == "CROSS_UP":
                score += 0.2
            elif zero_cross == "CROSS_DOWN":
                score -= 0.2

            score += float(np.tanh(hist * 20.0) * 0.1)

        return float(np.clip(score, -1.0, 1.0))

    def _score_bvb(self, bvb_data: Dict[str, Any]) -> float:
        """
        Reglas base:
        - EXTREME_BULL = -0.5 (enfoque contrarian)
        - EXTREME_BEAR = +0.5 (contrarian)
        - Zero cross up/down = +/-0.6
        """
        signal = bvb_data.get("bvb_signal", "NEUTRAL")
        zero_cross = bvb_data.get("bvb_zero_cross", "NONE")
        total = float(bvb_data.get("bvb_total", 0.0))

        score = 0.0
        if signal == "EXTREME_BULL":
            score -= 0.5
        elif signal == "EXTREME_BEAR":
            score += 0.5
        elif signal == "BULL":
            score += 0.25
        elif signal == "BEAR":
            score -= 0.25

        if zero_cross == "CROSS_UP":
            score += 0.6
        elif zero_cross == "CROSS_DOWN":
            score -= 0.6

        score += float(np.tanh(total / 60.0) * 0.1)
        return float(np.clip(score, -1.0, 1.0))

    def _score_koncorde(self, k_data: Dict[str, Any]) -> float:
        """
        Reglas base:
        - STRONG_BUY/STRONG_SELL = +/-1.0
        - WEAK signals = +/-0.4
        """
        signal = k_data.get("k_signal", "NEUTRAL")
        smart_money = k_data.get("k_smart_money", "NEUTRAL")
        verde_vs_media = k_data.get("k_verde_vs_media", "BELOW")

        signal_map = {
            "STRONG_BUY": 1.0,
            "WEAK_BUY": 0.4,
            "NEUTRAL": 0.0,
            "WEAK_SELL": -0.4,
            "STRONG_SELL": -1.0,
        }
        score = float(signal_map.get(signal, 0.0))

        if smart_money == "ACCUMULATING":
            score += 0.2
        elif smart_money == "DISTRIBUTING":
            score -= 0.2

        if verde_vs_media == "ABOVE":
            score += 0.1
        else:
            score -= 0.1

        return float(np.clip(score, -1.0, 1.0))

    def _score_liquidity(self, lp_data: Dict[str, Any]) -> float:
        """
        Reglas base:
        - Near support = +0.5
        - Near resistance = -0.5
        - Breakout = +/-0.9
        """
        near_support = bool(lp_data.get("lp_near_support", False))
        near_resistance = bool(lp_data.get("lp_near_resistance", False))
        inside_zone = bool(lp_data.get("lp_inside_zone", False))
        breakout = lp_data.get("lp_breakout", "NONE")

        score = 0.0
        if near_support:
            score += 0.5
        if near_resistance:
            score -= 0.5

        if breakout == "BULLISH_BREAKOUT":
            score = max(score, 0.9)
        elif breakout == "BEARISH_BREAKDOWN":
            score = min(score, -0.9)

        if inside_zone:
            score *= 0.8  # reducimos por ruido en zona de alta friccion

        return float(np.clip(score, -1.0, 1.0))

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------
    def _sanitize_ohlcv(
        self,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        open_: np.ndarray,
        volume: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        close = self._sanitize_array(close)
        high = self._sanitize_array(high)
        low = self._sanitize_array(low)
        open_ = self._sanitize_array(open_)
        volume = self._sanitize_array(volume)

        min_len = min(len(close), len(high), len(low), len(open_), len(volume))
        close = close[-min_len:]
        high = high[-min_len:]
        low = low[-min_len:]
        open_ = open_[-min_len:]
        volume = volume[-min_len:]

        volume = np.where(volume <= 0.0, 1.0, volume)
        return close, high, low, open_, volume

    @staticmethod
    def _sanitize_array(arr: np.ndarray) -> np.ndarray:
        values = np.asarray(arr, dtype=float).reshape(-1)
        s = pd.Series(values).replace([np.inf, -np.inf], np.nan).ffill().bfill()
        return s.to_numpy(dtype=float)

    @staticmethod
    def _sma(values: np.ndarray, length: int) -> np.ndarray:
        if length <= 1:
            return np.asarray(values, dtype=float)
        return (
            pd.Series(values, dtype=float)
            .rolling(window=length, min_periods=length)
            .mean()
            .to_numpy(dtype=float)
        )

    @staticmethod
    def _rolling_sum(values: np.ndarray, length: int) -> np.ndarray:
        return (
            pd.Series(values, dtype=float)
            .rolling(window=length, min_periods=length)
            .sum()
            .to_numpy(dtype=float)
        )

    @staticmethod
    def _rolling_min(values: np.ndarray, length: int) -> np.ndarray:
        return (
            pd.Series(values, dtype=float)
            .rolling(window=length, min_periods=length)
            .min()
            .to_numpy(dtype=float)
        )

    @staticmethod
    def _rolling_max(values: np.ndarray, length: int) -> np.ndarray:
        return (
            pd.Series(values, dtype=float)
            .rolling(window=length, min_periods=length)
            .max()
            .to_numpy(dtype=float)
        )

    @staticmethod
    def _rolling_std(values: np.ndarray, length: int) -> np.ndarray:
        return (
            pd.Series(values, dtype=float)
            .rolling(window=length, min_periods=length)
            .std(ddof=0)
            .to_numpy(dtype=float)
        )

    @staticmethod
    def _ema_tv(values: np.ndarray, length: int) -> np.ndarray:
        """
        EMA estilo TradingView:
        - seed: SMA de los primeros `length` valores validos.
        """
        arr = np.asarray(values, dtype=float)
        n = len(arr)
        out = np.full(n, np.nan, dtype=float)
        if length <= 0:
            return out

        valid_idx = np.where(np.isfinite(arr))[0]
        if len(valid_idx) == 0:
            return out

        start = int(valid_idx[0])
        if n - start < length:
            return out

        seed_slice = arr[start : start + length]
        if not np.all(np.isfinite(seed_slice)):
            # Si hay huecos inesperados, rellenamos localmente.
            seed_slice = pd.Series(seed_slice).ffill().bfill().to_numpy(dtype=float)

        out[start + length - 1] = float(np.mean(seed_slice))
        alpha = 2.0 / (length + 1.0)
        for i in range(start + length, n):
            x = arr[i]
            if not np.isfinite(x):
                x = out[i - 1]
            out[i] = x * alpha + out[i - 1] * (1.0 - alpha)
        return out

    @staticmethod
    def _rma(values: np.ndarray, length: int) -> np.ndarray:
        """
        Wilder's RMA (ta.rma de TradingView):
        rma[i] = (rma[i-1]*(length-1) + value[i]) / length
        """
        arr = np.asarray(values, dtype=float)
        n = len(arr)
        out = np.full(n, np.nan, dtype=float)
        if length <= 0 or n < length:
            return out

        seed = arr[:length]
        if not np.all(np.isfinite(seed)):
            seed = pd.Series(seed).ffill().bfill().to_numpy(dtype=float)

        out[length - 1] = float(np.mean(seed))
        for i in range(length, n):
            x = arr[i]
            if not np.isfinite(x):
                x = out[i - 1]
            out[i] = (out[i - 1] * (length - 1) + x) / length
        return out

    def _rsi_wilder_series(self, values: np.ndarray, length: int = 14) -> np.ndarray:
        values = self._sanitize_array(values)
        delta = np.diff(values, prepend=values[0])
        gain = np.maximum(delta, 0.0)
        loss = np.maximum(-delta, 0.0)
        avg_gain = self._rma(gain, length)
        avg_loss = self._rma(loss, length)

        rsi = np.full(len(values), np.nan, dtype=float)
        valid = ~np.isnan(avg_gain) & ~np.isnan(avg_loss)
        with np.errstate(divide="ignore", invalid="ignore"):
            rs = np.divide(avg_gain, avg_loss, where=valid)
            rsi_val = 100.0 - (100.0 / (1.0 + rs))
            rsi[valid] = rsi_val[valid]

        zero_loss = valid & (avg_loss == 0.0)
        zero_gain = valid & (avg_gain == 0.0)
        both_zero = zero_loss & zero_gain
        rsi[zero_loss] = 100.0
        rsi[zero_gain] = 0.0
        rsi[both_zero] = 50.0
        return rsi

    @staticmethod
    def _rsi_from_up_down(up: np.ndarray, down: np.ndarray) -> np.ndarray:
        """
        RSI generalizado a dos series positivas (usado por MFI-like):
        rs = up / down
        rsi = 100 - 100/(1+rs)
        """
        up = np.asarray(up, dtype=float)
        down = np.asarray(down, dtype=float)
        out = np.full(len(up), np.nan, dtype=float)

        valid = np.isfinite(up) & np.isfinite(down)
        with np.errstate(divide="ignore", invalid="ignore"):
            rs = np.divide(up, down, where=valid)
            out[valid] = 100.0 - (100.0 / (1.0 + rs[valid]))

        both_zero = valid & (up == 0.0) & (down == 0.0)
        down_zero = valid & (down == 0.0) & (up > 0.0)
        up_zero = valid & (up == 0.0) & (down > 0.0)
        out[both_zero] = 50.0
        out[down_zero] = 100.0
        out[up_zero] = 0.0
        return out

    @staticmethod
    def _normalize_centered_100(values: np.ndarray, roll_min: np.ndarray, roll_max: np.ndarray) -> np.ndarray:
        den = roll_max - roll_min
        out = np.zeros(len(values), dtype=float)
        valid = np.isfinite(values) & np.isfinite(roll_min) & np.isfinite(roll_max) & (den != 0.0)
        out[valid] = ((values[valid] - roll_min[valid]) / den[valid] - 0.5) * 100.0
        return out

    @staticmethod
    def _safe_ratio_times_100(num: np.ndarray, den: np.ndarray) -> np.ndarray:
        out = np.zeros(len(num), dtype=float)
        valid = np.isfinite(num) & np.isfinite(den) & (den != 0.0)
        out[valid] = (num[valid] * 100.0) / den[valid]
        return out

    def _ma(self, values: np.ndarray, length: int, ma_type: str) -> np.ndarray:
        t = ma_type.upper()
        if t == "SMA":
            return self._sma(values, length)
        if t == "WMA":
            weights = np.arange(1, length + 1, dtype=float)
            w_sum = float(np.sum(weights))
            return (
                pd.Series(values, dtype=float)
                .rolling(window=length, min_periods=length)
                .apply(lambda x: float(np.dot(x, weights) / w_sum), raw=True)
                .to_numpy(dtype=float)
            )
        # Default: EMA
        return self._ema_tv(values, length)

    @staticmethod
    def _pivot_flags(series: np.ndarray, left: int, right: int, mode: str = "low") -> np.ndarray:
        arr = np.asarray(series, dtype=float)
        n = len(arr)
        flags = np.zeros(n, dtype=bool)
        if n < left + right + 1:
            return flags

        for i in range(left, n - right):
            win = arr[i - left : i + right + 1]
            center = arr[i]
            if not np.all(np.isfinite(win)) or not np.isfinite(center):
                continue
            if mode == "low":
                if center == np.min(win):
                    flags[i] = True
            else:
                if center == np.max(win):
                    flags[i] = True
        return flags

    @staticmethod
    def _find_latest_divergence(
        price_series: np.ndarray,
        osc_series: np.ndarray,
        price_pivots: np.ndarray,
        osc_pivots: np.ndarray,
        range_lower: int,
        range_upper: int,
        divergence_type: str,
    ) -> Optional[int]:
        common_idx = np.where(price_pivots & osc_pivots)[0]
        if len(common_idx) < 2:
            return None

        # Buscamos desde el final para detectar la divergencia mas reciente.
        for j in range(len(common_idx) - 1, 0, -1):
            i2 = int(common_idx[j])
            for k in range(j - 1, -1, -1):
                i1 = int(common_idx[k])
                dist = i2 - i1
                if dist < range_lower:
                    continue
                if dist > range_upper:
                    break

                p1, p2 = price_series[i1], price_series[i2]
                o1, o2 = osc_series[i1], osc_series[i2]
                if not (np.isfinite(p1) and np.isfinite(p2) and np.isfinite(o1) and np.isfinite(o2)):
                    continue

                if divergence_type == "bullish":
                    # Precio LL y oscilador HL
                    if p2 < p1 and o2 > o1:
                        return i2
                else:
                    # Precio HH y oscilador LH
                    if p2 > p1 and o2 < o1:
                        return i2
        return None

    @staticmethod
    def _count_spaced_contacts(indices: np.ndarray, min_distance: int) -> int:
        if len(indices) == 0:
            return 0
        count = 1
        last = int(indices[0])
        for idx in indices[1:]:
            idx = int(idx)
            if idx - last >= min_distance:
                count += 1
                last = idx
        return count

    @staticmethod
    def _append_zone_unique(zones: List[Tuple[float, float]], zone: Tuple[float, float], tol: float = 0.002) -> None:
        z0, z1 = zone
        for a, b in zones:
            rel0 = abs(z0 - a) / max(abs(a), 1e-12)
            rel1 = abs(z1 - b) / max(abs(b), 1e-12)
            if rel0 <= tol and rel1 <= tol:
                return
        zones.append((float(z0), float(z1)))

    @staticmethod
    def _last_valid(arr: np.ndarray, default: Any = 0.0) -> Any:
        values = np.asarray(arr, dtype=float)
        valid = values[np.isfinite(values)]
        if len(valid) == 0:
            return default
        return float(valid[-1])

    @staticmethod
    def _is_finite_pair(a: float, b: float) -> bool:
        return bool(np.isfinite(a) and np.isfinite(b))

    @staticmethod
    def _final_signal_from_score(score: float) -> str:
        if score > 0.6:
            return "STRONG_BUY"
        if score > 0.2:
            return "BUY"
        if score < -0.6:
            return "STRONG_SELL"
        if score < -0.2:
            return "SELL"
        return "NEUTRAL"

    @staticmethod
    def _compute_confidence(scores: np.ndarray) -> float:
        """
        Confianza por alineacion:
        - baja entropia de direcciones -> mayor acuerdo
        - mayor magnitud media -> mayor conviccion
        """
        s = np.asarray(scores, dtype=float)
        if len(s) == 0:
            return 0.0

        pos = float(np.mean(s > 0.05))
        neg = float(np.mean(s < -0.05))
        neu = float(1.0 - pos - neg)

        probs = np.array([pos, neg, neu], dtype=float) + 1e-12
        ent = float(stats.entropy(probs, base=2))
        ent_norm = ent / np.log2(3.0)  # 0..1
        agreement = 1.0 - np.clip(ent_norm, 0.0, 1.0)

        strength = float(np.clip(np.mean(np.abs(s)), 0.0, 1.0))
        activity = float(np.clip(pos + neg, 0.0, 1.0))

        confidence = (0.5 * agreement + 0.5 * strength) * (0.5 + 0.5 * activity)
        return float(np.clip(confidence, 0.0, 1.0))


# ----------------------------------------------------------------------
# Demo completa solicitada
# ----------------------------------------------------------------------
def _generate_synthetic_ohlcv(n_bars: int = 500, seed: int = 42) -> Dict[str, np.ndarray]:
    """
    Genera OHLCV sintetico realista con GBM (Geometric Brownian Motion).
    """
    rng = np.random.default_rng(seed)

    # GBM para close
    s0 = 150.0
    mu = 0.12
    sigma = 0.28
    dt = 1.0 / 252.0

    shocks = rng.normal(loc=0.0, scale=1.0, size=n_bars)
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shocks
    close = s0 * np.exp(np.cumsum(log_returns))

    # Open cerca del close previo
    open_ = np.empty(n_bars, dtype=float)
    open_[0] = close[0] * (1.0 + rng.normal(0, 0.002))
    open_[1:] = close[:-1] * (1.0 + rng.normal(0, 0.002, size=n_bars - 1))

    # Rango intrabar realista
    intrabar_amp = np.abs(rng.normal(loc=0.004, scale=0.002, size=n_bars))
    high = np.maximum(open_, close) * (1.0 + intrabar_amp)
    low = np.minimum(open_, close) * (1.0 - intrabar_amp)

    # Volumen lognormal con relacion a magnitud del retorno
    ret_abs = np.abs(np.diff(close, prepend=close[0]) / np.where(close == 0.0, 1.0, close))
    base_volume = rng.lognormal(mean=13.0, sigma=0.35, size=n_bars)
    volume = base_volume * (1.0 + 8.0 * ret_abs)
    volume = np.maximum(volume, 1.0)

    return {
        "open": open_.astype(float),
        "high": high.astype(float),
        "low": low.astype(float),
        "close": close.astype(float),
        "volume": volume.astype(float),
    }


def _print_demo_output(bundle: SignalBundle) -> None:
    breakdown = bundle["signal_breakdown"]
    rsi = bundle["rsi_data"]
    st = bundle["stochastic_data"]
    mc = bundle["macd_data"]
    bvb = bundle["bvb_data"]
    kon = bundle["koncorde_data"]
    lp = bundle["liquidity_data"]

    print("\n" + "=" * 52)
    print(" ADVANCED INDICATOR ENGINE - SIGNAL BUNDLE")
    print("=" * 52)

    print("\n  RSI:")
    print(f"    Valor:        {rsi['rsi']:.2f}")
    print(f"    Senal:        {rsi['rsi_signal']}")
    print(f"    Divergencia:  {rsi['rsi_divergence']}")
    print(f"    Score:        {breakdown['rsi']['raw_score']:+.2f}")

    print("\n  Stochastic:")
    print(f"    %K: {st['stoch_k']:.2f}   %D: {st['stoch_d']:.2f}")
    print(f"    Senal:        {st['stoch_signal']}")
    print(f"    Cruce:        {st['stoch_cross']}")
    print(f"    Score:        {breakdown['stochastic']['raw_score']:+.2f}")

    print("\n  MACD:")
    print(f"    MACD:         {mc['macd']:.4f}")
    print(f"    Signal:       {mc['macd_signal']:.4f}")
    print(f"    Histograma:   {mc['macd_histogram']:+.4f} ({mc['macd_momentum']})")
    print(f"    Cruce:        {mc['macd_cross']}")
    print(f"    Score:        {breakdown['macd']['raw_score']:+.2f}")

    print("\n  Bulls vs Bears:")
    print(f"    Total:        {bvb['bvb_total']:+.2f}")
    print(f"    Senal:        {bvb['bvb_signal']}")
    print(f"    Zero Cross:   {bvb['bvb_zero_cross']}")
    print(f"    Score:        {breakdown['bvb']['raw_score']:+.2f}")

    print("\n  Koncorde:")
    print(f"    Verde:        {kon['k_verde']:+.2f}")
    print(f"    Marron:       {kon['k_marron']:+.2f}")
    print(f"    Azul:         {kon['k_azul']:+.2f}")
    print(f"    Media:        {kon['k_media']:+.2f}")
    print(f"    Senal:        {kon['k_signal']}")
    print(f"    Score:        {breakdown['koncorde']['raw_score']:+.2f}")

    print("\n  Liquidity Pools:")
    print(f"    Resistencias: {[round(x, 3) for x in lp['lp_resistance_levels'][-5:]]}")
    print(f"    Soportes:     {[round(x, 3) for x in lp['lp_support_levels'][-5:]]}")
    print(f"    Cerca de:     {lp['lp_signal']}")
    print(f"    Score:        {breakdown['liquidity']['raw_score']:+.2f}")

    alignment = int(round(bundle["confidence"] * 100))
    print("\n" + "-" * 52)
    print(f"  COMPOSITE SCORE:    {bundle['composite_score']:+.2f}")
    print(f"  SENAL FINAL:        {bundle['final_signal']}")
    print(f"  CONFIANZA:          {alignment}%")
    print("=" * 52)


def demo() -> None:
    data = _generate_synthetic_ohlcv(n_bars=500, seed=42)
    engine = AdvancedIndicatorEngine()
    bundle = engine.calculate_all(
        close=data["close"],
        high=data["high"],
        low=data["low"],
        open_=data["open"],
        volume=data["volume"],
    )

    if bundle is None:
        print("No hay suficientes barras (minimo 100).")
        return

    _print_demo_output(bundle)


if __name__ == "__main__":
    demo()

