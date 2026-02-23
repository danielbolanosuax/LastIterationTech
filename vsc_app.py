#!/usr/bin/env python3
"""
God Mode Trading - Professional Monitoring App
Signals + model audit + robustness tests in one command center.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from overfitting_audit import (
    deflated_sharpe_ratio,
    detect_overfitting,
    overfitting_alerts_from_metrics,
    probability_of_backtest_overfitting,
    romano_wolf_correction,
    white_reality_check,
)
from watchlists import TOP_50_SYMBOLS


BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

ACTION_COLOR = {"BUY": "#0f766e", "SELL": "#b91c1c", "HOLD": "#b45309", "NO_DATA": "#6b7280"}
ACTION_MAP = {"BUY": 1.0, "SELL": -1.0, "HOLD": 0.0, "NO_DATA": 0.0}
MODULE_COLS = ["temporal", "vision", "tabular", "nlp", "graph", "advanced_model", "options_model", "overall_model"]
STATUS_COLS = [
    "status_temporal",
    "status_vision",
    "status_tabular",
    "status_nlp",
    "status_graph",
    "status_sac",
    "status_timegan",
    "status_options",
    "status_advanced_indicators",
]


def set_theme() -> None:
    st.set_page_config(
        page_title="God Mode Professional Monitor",
        page_icon="GM",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
        :root {
            --card: rgba(255,255,255,0.9);
            --line: #b6cec8;
            --ink: #14313a;
            --muted: #4d6a73;
        }
        .stApp {
            background:
                radial-gradient(1300px 360px at 0% -12%, rgba(15,118,110,0.24), transparent 60%),
                radial-gradient(1200px 340px at 100% -10%, rgba(180,83,9,0.22), transparent 60%),
                linear-gradient(135deg, #fbf7ef, #dbe8e4);
        }
        .hero {
            background: var(--card);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            box-shadow: 0 14px 30px rgba(20, 49, 58, 0.10);
        }
        .hero h1 {
            margin: 0;
            color: var(--ink);
            font-family: "Space Grotesk", sans-serif;
            font-size: 1.55rem;
        }
        .hero p {
            margin: .3rem 0 0 0;
            color: var(--muted);
            font-family: "IBM Plex Mono", monospace;
            font-size: .9rem;
        }
        .kpi-wrap {
            background: var(--card);
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: .65rem .8rem;
        }
        .kpi-title {
            color: var(--muted);
            font-family: "IBM Plex Mono", monospace;
            font-size: .75rem;
            text-transform: uppercase;
        }
        .kpi-value {
            color: var(--ink);
            font-family: "Space Grotesk", sans-serif;
            font-size: 1.25rem;
            font-weight: 700;
            margin-top: .15rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def kpi(title: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="kpi-wrap">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def load_reports(limit: int = 400) -> List[Dict]:
    files = sorted(LOGS_DIR.glob("pipeline_report_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    reports: List[Dict] = []
    for file_path in files:
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
            payload["_file"] = str(file_path)
            reports.append(payload)
        except Exception:
            continue
    return reports


def reports_to_df(reports: List[Dict]) -> pd.DataFrame:
    rows = []
    for report in reports:
        run_ts = pd.to_datetime(report.get("timestamp"), format="%Y%m%d_%H%M%S", errors="coerce")
        run_id = str(report.get("timestamp", "N/A"))
        num_symbols = int(report.get("num_symbols", 0))

        for signal in report.get("signals", []) or []:
            model = signal.get("model_confidence", {}) or {}
            options = signal.get("options_analysis", {}) or {}
            advanced = signal.get("advanced", {}) or {}
            m_status = signal.get("model_status", {}) or {}
            sig_ts = pd.to_datetime(signal.get("timestamp"), errors="coerce")
            if pd.isna(sig_ts):
                sig_ts = run_ts

            rows.append(
                {
                    "run_id": run_id,
                    "run_timestamp": run_ts,
                    "run_num_symbols": num_symbols,
                    "timestamp": sig_ts,
                    "symbol": str(signal.get("symbol", "")).upper(),
                    "action": str(signal.get("action", "HOLD")).upper(),
                    "price": float(signal.get("price", 0.0)),
                    "confidence": float(signal.get("confidence", 0.0)),
                    "position_size": float(signal.get("position_size", 0.0)),
                    "reasoning": str(signal.get("reasoning", "")),
                    "sentiment": float(signal.get("sentiment", 0.0)),
                    "risk_passed": bool(signal.get("risk_check", {}).get("passed", False)),
                    "risk_message": str(signal.get("risk_check", {}).get("message", "")),
                    "temporal": float(model.get("temporal", np.nan)),
                    "vision": float(model.get("vision", np.nan)),
                    "tabular": float(model.get("tabular", np.nan)),
                    "nlp": float(model.get("nlp", np.nan)),
                    "graph": float(model.get("graph", np.nan)),
                    "advanced_model": float(model.get("advanced", np.nan)),
                    "options_model": float(model.get("options", np.nan)),
                    "overall_model": float(model.get("overall", np.nan)),
                    "advanced_score": float(advanced.get("composite_score", np.nan)),
                    "advanced_signal": str(advanced.get("final_signal", "NEUTRAL")).upper(),
                    "advanced_conf": float(advanced.get("confidence", np.nan)),
                    "adv_rsi": float(advanced.get("rsi", np.nan)),
                    "adv_rsi_signal": str(advanced.get("rsi_signal", "NEUTRAL")),
                    "adv_rsi_divergence": str(advanced.get("rsi_divergence", "NONE")),
                    "adv_macd": float(advanced.get("macd", np.nan)),
                    "adv_macd_hist": float(advanced.get("macd_histogram", np.nan)),
                    "adv_macd_cross": str(advanced.get("macd_cross", "NONE")),
                    "adv_bvb_total": float(advanced.get("bvb_total", np.nan)),
                    "adv_bvb_signal": str(advanced.get("bvb_signal", "NEUTRAL")),
                    "adv_k_signal": str(advanced.get("k_signal", "NEUTRAL")),
                    "adv_k_smart_money": str(advanced.get("k_smart_money", "NEUTRAL")),
                    "adv_lp_signal": str(advanced.get("lp_signal", "NEUTRAL")),
                    "adv_lp_breakout": str(advanced.get("lp_breakout", "NONE")),
                    "adv_lp_near_support": bool(advanced.get("lp_near_support", False)),
                    "adv_lp_near_resistance": bool(advanced.get("lp_near_resistance", False)),
                    "options_bias": str(options.get("directional_bias", "N/A")),
                    "options_rec": str(options.get("recommendation", "N/A")),
                    "options_iv": float(options.get("avg_implied_volatility", 0.0)),
                    "status_temporal": str(m_status.get("temporal", "UNKNOWN")),
                    "status_vision": str(m_status.get("vision", "UNKNOWN")),
                    "status_tabular": str(m_status.get("tabular", "UNKNOWN")),
                    "status_nlp": str(m_status.get("nlp", "UNKNOWN")),
                    "status_graph": str(m_status.get("graph", "UNKNOWN")),
                    "status_sac": str(m_status.get("sac", "UNKNOWN")),
                    "status_timegan": str(m_status.get("timegan", "UNKNOWN")),
                    "status_options": str(m_status.get("options", "UNKNOWN")),
                    "status_advanced_indicators": str(m_status.get("advanced_indicators", "UNKNOWN")),
                    "source_file": report.get("_file", ""),
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["run_timestamp", "symbol"], ascending=[False, True])
    return df


def load_signal_tape(limit: int = 3000) -> pd.DataFrame:
    file_path = LOGS_DIR / "signals.jsonl"
    if not file_path.exists():
        return pd.DataFrame()

    lines: deque = deque(maxlen=limit)
    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)

    rows = []
    for line in lines:
        try:
            d = json.loads(line)
            rows.append(
                {
                    "timestamp": pd.to_datetime(d.get("timestamp"), errors="coerce"),
                    "symbol": str(d.get("symbol", "")).upper(),
                    "action": str(d.get("action", "HOLD")).upper(),
                    "price": float(d.get("price", 0.0)),
                    "confidence": float(d.get("confidence", 0.0)),
                    "position_size": float(d.get("position_size", 0.0)),
                }
            )
        except Exception:
            continue

    tape = pd.DataFrame(rows)
    if not tape.empty:
        tape = tape.sort_values("timestamp", ascending=False)
    return tape


def load_backtests(limit: int = 200) -> pd.DataFrame:
    files = sorted(LOGS_DIR.glob("backtest_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    rows = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            data["file"] = f.name
            rows.append(data)
        except Exception:
            continue
    return pd.DataFrame(rows)


def report_label(report: Dict) -> str:
    ts = str(report.get("timestamp", "N/A"))
    n = int(report.get("num_symbols", 0))
    return f"{ts} | symbols={n}"


def select_report(reports: List[Dict], expected_n: int) -> Optional[Dict]:
    if not reports:
        return None
    sorted_reports = sorted(
        reports,
        key=lambda r: (int(r.get("num_symbols", 0)), str(r.get("timestamp", ""))),
        reverse=True,
    )
    # Primero intentar uno que cubra el universo esperado.
    for r in sorted_reports:
        if int(r.get("num_symbols", 0)) >= expected_n:
            return r
    return sorted_reports[0]


def build_universe_signals(selected_df: pd.DataFrame, tape_df: pd.DataFrame, universe: List[str]) -> pd.DataFrame:
    if selected_df.empty and tape_df.empty:
        return pd.DataFrame()

    latest_report_by_symbol: Dict[str, Dict] = {}
    if not selected_df.empty:
        for _, row in selected_df.sort_values("timestamp", ascending=False).iterrows():
            sym = row["symbol"]
            if sym not in latest_report_by_symbol:
                latest_report_by_symbol[sym] = row.to_dict()

    latest_tape_by_symbol: Dict[str, Dict] = {}
    if not tape_df.empty:
        for _, row in tape_df.sort_values("timestamp", ascending=False).iterrows():
            sym = row["symbol"]
            if sym not in latest_tape_by_symbol:
                latest_tape_by_symbol[sym] = row.to_dict()

    rows = []
    now = pd.Timestamp.utcnow().tz_localize(None)
    for sym in universe:
        if sym in latest_report_by_symbol:
            item = latest_report_by_symbol[sym]
            source = "report"
        elif sym in latest_tape_by_symbol:
            item = latest_tape_by_symbol[sym]
            source = "tape"
        else:
            rows.append(
                {
                    "symbol": sym,
                    "action": "NO_DATA",
                    "price": np.nan,
                    "confidence": np.nan,
                    "advanced_signal": "NO_DATA",
                    "advanced_score": np.nan,
                    "advanced_conf": np.nan,
                    "position_size": np.nan,
                    "risk_passed": False,
                    "options_bias": "N/A",
                    "options_rec": "N/A",
                    "timestamp": pd.NaT,
                    "source": "missing",
                    "staleness_min": np.nan,
                }
            )
            continue

        ts = pd.to_datetime(item.get("timestamp"), errors="coerce")
        stale = np.nan
        if pd.notna(ts):
            stale = float((now - ts).total_seconds() / 60.0)

        rows.append(
            {
                "symbol": sym,
                "action": str(item.get("action", "NO_DATA")).upper(),
                "price": float(item.get("price", np.nan)),
                "confidence": float(item.get("confidence", np.nan)),
                "advanced_signal": str(item.get("advanced_signal", "NEUTRAL")).upper() if source == "report" else "N/A",
                "advanced_score": float(item.get("advanced_score", np.nan)) if source == "report" else np.nan,
                "advanced_conf": float(item.get("advanced_conf", np.nan)) if source == "report" else np.nan,
                "position_size": float(item.get("position_size", np.nan)),
                "risk_passed": bool(item.get("risk_passed", False)) if source == "report" else np.nan,
                "options_bias": str(item.get("options_bias", "N/A")) if source == "report" else "N/A",
                "options_rec": str(item.get("options_rec", "N/A")) if source == "report" else "N/A",
                "timestamp": ts,
                "source": source,
                "staleness_min": stale,
            }
        )

    df = pd.DataFrame(rows).sort_values("symbol")
    return df


def compute_operational_returns(all_df: pd.DataFrame) -> pd.DataFrame:
    if all_df.empty:
        return pd.DataFrame()

    rows = []
    for sym, g in all_df.sort_values("timestamp").groupby("symbol"):
        g = g.copy().sort_values("timestamp")
        g["next_price"] = g["price"].shift(-1)
        g["raw_return"] = (g["next_price"] / g["price"]) - 1.0
        g["direction"] = g["action"].map(ACTION_MAP).fillna(0.0)
        g["strategy_return"] = g["direction"] * g["raw_return"]
        g["hit"] = (g["strategy_return"] > 0).astype(float)
        valid = g.dropna(subset=["raw_return"])
        for _, row in valid.iterrows():
            rows.append(
                {
                    "timestamp": row["timestamp"],
                    "symbol": sym,
                    "action": row["action"],
                    "raw_return": float(row["raw_return"]),
                    "strategy_return": float(row["strategy_return"]),
                    "hit": float(row["hit"]) if row["direction"] != 0 else np.nan,
                }
            )
    ret_df = pd.DataFrame(rows)
    if not ret_df.empty:
        ret_df = ret_df.sort_values("timestamp")
    return ret_df


def sharpe(returns: pd.Series) -> float:
    r = returns.dropna().values
    if len(r) < 3:
        return np.nan
    return float(np.mean(r) / (np.std(r) + 1e-12) * np.sqrt(252))


def max_drawdown(returns: pd.Series) -> float:
    r = returns.dropna().values
    if len(r) < 2:
        return np.nan
    equity = np.cumprod(1.0 + r)
    peaks = np.maximum.accumulate(equity)
    dd = (equity - peaks) / (peaks + 1e-12)
    return float(abs(np.min(dd)))


def split_operational_segments(ret_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    if ret_df.empty:
        return {"train": pd.DataFrame(), "val": pd.DataFrame(), "test": pd.DataFrame(), "oos": pd.DataFrame()}
    n = len(ret_df)
    i1 = int(n * 0.5)
    i2 = int(n * 0.7)
    i3 = int(n * 0.85)
    return {
        "train": ret_df.iloc[:i1],
        "val": ret_df.iloc[i1:i2],
        "test": ret_df.iloc[i2:i3],
        "oos": ret_df.iloc[i3:],
    }


def segment_metrics(seg_df: pd.DataFrame) -> Dict[str, float]:
    if seg_df.empty:
        return {"accuracy": np.nan, "sharpe_ratio": np.nan, "max_drawdown": np.nan}
    no_hold = seg_df.dropna(subset=["hit"])
    acc = float(no_hold["hit"].mean()) if not no_hold.empty else np.nan
    return {
        "accuracy": acc,
        "sharpe_ratio": sharpe(seg_df["strategy_return"]),
        "max_drawdown": max_drawdown(seg_df["strategy_return"]),
    }


def build_diagnostic_matrix(ret_df: pd.DataFrame, backtest_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    seg = split_operational_segments(ret_df)
    m_train = segment_metrics(seg["train"])
    m_val = segment_metrics(seg["val"])
    m_test = segment_metrics(seg["test"])
    m_oos = segment_metrics(seg["oos"])

    # Si existe backtest, usar su sharpe/drawdown como referencia train.
    if not backtest_df.empty:
        m_train["sharpe_ratio"] = float(backtest_df.iloc[0].get("sharpe_ratio", m_train["sharpe_ratio"]))
        m_train["max_drawdown"] = float(backtest_df.iloc[0].get("max_drawdown_pct", m_train["max_drawdown"]))

    overfit_map = detect_overfitting(m_train, m_val, threshold=0.2)
    rows = []
    for metric in ["accuracy", "sharpe_ratio", "max_drawdown"]:
        tr = m_train.get(metric, np.nan)
        va = m_val.get(metric, np.nan)
        te = m_test.get(metric, np.nan)
        oo = m_oos.get(metric, np.nan)

        status = "🟢 OK"
        if metric in overfit_map and overfit_map[metric]["overfit"]:
            status = "🔴 OVERFIT"

        if metric == "sharpe_ratio" and pd.notna(tr) and pd.notna(oo):
            if tr > 0 and oo < tr * 0.4:
                status = "🔴 CRITICAL"
        if metric == "max_drawdown" and pd.notna(tr) and pd.notna(oo):
            if tr > 0 and oo > tr * 1.5:
                status = "🔴 CRITICAL"

        rows.append(
            {
                "metric": metric,
                "train": tr,
                "val": va,
                "test": te,
                "oos": oo,
                "status": status,
            }
        )

    alerts = overfitting_alerts_from_metrics(m_train, m_oos)
    return pd.DataFrame(rows), alerts


def compute_statistical_tests(ret_df: pd.DataFrame) -> Dict[str, object]:
    if ret_df.empty:
        return {}

    pivot = ret_df.pivot_table(index="timestamp", columns="symbol", values="strategy_return", aggfunc="mean").fillna(0.0)
    out: Dict[str, object] = {}

    if pivot.shape[0] >= 40 and pivot.shape[1] >= 2:
        wrc = white_reality_check(pivot.values, n_bootstrap=800, block_size=10)
        out["wrc"] = wrc

        avg_sr = sharpe(pivot.mean(axis=1))
        trials = [sharpe(pivot[c]) for c in pivot.columns]
        trials = [x for x in trials if pd.notna(x)]
        if len(trials) >= 3:
            out["dsr"] = deflated_sharpe_ratio(avg_sr, trials, n_obs=len(pivot))

        if pivot.shape[0] >= 80 and pivot.shape[1] >= 4:
            try:
                out["pbo"] = probability_of_backtest_overfitting(pivot, n_splits=8)
            except Exception:
                pass

        # Romano-Wolf sobre medias de retorno por símbolo
        means = pivot.mean(axis=0).values
        stds = pivot.std(axis=0).values + 1e-12
        t_obs = means / (stds / np.sqrt(len(pivot)))

        rng = np.random.default_rng(42)
        boot = []
        for _ in range(500):
            idx = rng.integers(0, len(pivot), size=len(pivot))
            sample = pivot.values[idx]
            m = sample.mean(axis=0)
            s = sample.std(axis=0) + 1e-12
            boot.append(m / (s / np.sqrt(len(sample))))
        boot_stats = np.vstack(boot)
        out["romano_wolf"] = romano_wolf_correction(t_obs, boot_stats)

    return out


def show_header(selected_report: Optional[Dict], represented: int, missing: int) -> None:
    ts = str(selected_report.get("timestamp", "N/A")) if selected_report else "N/A"
    n = int(selected_report.get("num_symbols", 0)) if selected_report else 0
    st.markdown(
        f"""
        <div class="hero">
            <h1>God Mode - Professional Trading Intelligence</h1>
            <p>Run selected: {ts} | Signals in report: {n} | Universe represented: {represented} | Missing: {missing}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_signal_tab(universe_df: pd.DataFrame, all_df: pd.DataFrame) -> None:
    st.subheader("Signal Command Center")
    if universe_df.empty:
        st.warning("No signals available.")
        return

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("BUY", int((universe_df["action"] == "BUY").sum()))
    c2.metric("SELL", int((universe_df["action"] == "SELL").sum()))
    c3.metric("HOLD", int((universe_df["action"] == "HOLD").sum()))
    c4.metric("No Data", int((universe_df["action"] == "NO_DATA").sum()))
    c5.metric("Avg Conf", f"{universe_df['confidence'].dropna().mean() * 100:.1f}%")

    col_a, col_b, col_c = st.columns([1.6, 1, 1])
    with col_a:
        action_filter = st.multiselect(
            "Action filter",
            ["BUY", "SELL", "HOLD", "NO_DATA"],
            default=["BUY", "SELL", "HOLD", "NO_DATA"],
        )
    with col_b:
        source_filter = st.multiselect("Source", ["report", "tape", "missing"], default=["report", "tape", "missing"])
    with col_c:
        symbol_search = st.text_input("Search symbol", value="")

    view = universe_df[universe_df["action"].isin(action_filter) & universe_df["source"].isin(source_filter)].copy()
    if symbol_search:
        view = view[view["symbol"].str.contains(symbol_search.upper(), na=False)]

    fmt = view.copy()
    fmt["price"] = fmt["price"].map(lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A")
    fmt["confidence"] = fmt["confidence"].map(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "N/A")
    fmt["advanced_score"] = fmt["advanced_score"].map(lambda x: f"{x:+.2f}" if pd.notna(x) else "N/A")
    fmt["advanced_conf"] = fmt["advanced_conf"].map(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "N/A")
    fmt["timestamp"] = pd.to_datetime(fmt["timestamp"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    fmt["staleness_min"] = fmt["staleness_min"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
    st.dataframe(
        fmt[
            [
                "symbol",
                "action",
                "price",
                "confidence",
                "advanced_signal",
                "advanced_score",
                "advanced_conf",
                "position_size",
                "risk_passed",
                "options_bias",
                "options_rec",
                "source",
                "staleness_min",
                "timestamp",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    left, right = st.columns((1, 1))
    with left:
        pie_src = universe_df["action"].value_counts().rename_axis("action").reset_index(name="count")
        fig = px.pie(pie_src, names="action", values="count", hole=0.55, color="action", color_discrete_map=ACTION_COLOR)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        if not all_df.empty:
            trend = all_df.dropna(subset=["timestamp"]).sort_values("timestamp").tail(400)
            fig = px.line(trend, x="timestamp", y="confidence", color="symbol")
            fig.update_yaxes(range=[0, 1], tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)


def show_model_audit_tab(universe_df: pd.DataFrame, selected_df: pd.DataFrame, all_df: pd.DataFrame, backtest_df: pd.DataFrame) -> None:
    st.subheader("Model Evaluation & Overfitting Audit")

    st.markdown("#### Model Connectivity Health")
    if selected_df.empty:
        st.info("Sin run seleccionado para inspeccionar conectividad.")
    else:
        latest_row = selected_df.iloc[0]
        connectivity = pd.DataFrame(
            [
                {"module": "temporal", "status": latest_row.get("status_temporal", "UNKNOWN")},
                {"module": "vision", "status": latest_row.get("status_vision", "UNKNOWN")},
                {"module": "tabular", "status": latest_row.get("status_tabular", "UNKNOWN")},
                {"module": "nlp", "status": latest_row.get("status_nlp", "UNKNOWN")},
                {"module": "graph", "status": latest_row.get("status_graph", "UNKNOWN")},
                {"module": "sac", "status": latest_row.get("status_sac", "UNKNOWN")},
                {"module": "timegan", "status": latest_row.get("status_timegan", "UNKNOWN")},
                {"module": "options", "status": latest_row.get("status_options", "UNKNOWN")},
                {"module": "advanced_indicators", "status": latest_row.get("status_advanced_indicators", "UNKNOWN")},
            ]
        )
        st.dataframe(connectivity, use_container_width=True, hide_index=True)
        if (connectivity["status"] != "CONNECTED").any():
            st.error("Hay módulos no conectados. Revisa estados != CONNECTED.")
        else:
            st.success("Todos los módulos están CONNECTED para la toma de decisiones.")

    # Checklist de calidad (estado actual del sistema)
    checklist = pd.DataFrame(
        [
            ["Walk-forward validation", "❌ No implementado en pipeline principal actual"],
            ["Purged k-fold + embargo", "❌ Disponible en herramienta de auditoría, no integrado al entrenamiento"],
            ["Train/Val learning curves persistidas", "⚠️ No hay historia de entrenamiento real en logs"],
            ["Regularización / Early stopping", "⚠️ No visible en módulos actuales (sistema en modo mock)"],
            ["Leakage guards formales", "⚠️ Parcial (sin framework completo de leakage tests)"],
            ["Stress / robustness tests", "✅ Disponibles en panel estadístico de esta app"],
        ],
        columns=["Control", "Estado"],
    )
    st.dataframe(checklist, use_container_width=True, hide_index=True)

    # Matriz de diagnostico
    op_ret = compute_operational_returns(all_df if not all_df.empty else selected_df)
    diag_df, alerts = build_diagnostic_matrix(op_ret, backtest_df)

    st.markdown("#### Diagnostic Matrix (Train / Val / Test / OOS)")
    if diag_df.empty:
        st.info("No hay suficientes datos para construir la matriz diagnóstica.")
    else:
        disp = diag_df.copy()
        for c in ["train", "val", "test", "oos"]:
            disp[c] = disp[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
        st.dataframe(disp, use_container_width=True, hide_index=True)

    if alerts:
        for a in alerts:
            st.error(a)
    else:
        st.success("Sin alertas críticas de overfitting con los datos actuales.")

    # Módulos por ticker
    st.markdown("#### Module Confidence by Ticker")
    if selected_df.empty:
        st.info("No hay datos de módulos en el run seleccionado.")
    else:
        module_df = selected_df[["symbol", *MODULE_COLS]].set_index("symbol")
        module_df = module_df.reindex(TOP_50_SYMBOLS)
        heat = module_df.copy().fillna(0.0)
        fig = px.imshow(
            heat.values,
            x=heat.columns,
            y=heat.index,
            color_continuous_scale="RdYlGn",
            aspect="auto",
            zmin=0,
            zmax=1,
        )
        fig.update_layout(height=700)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Advanced Indicators Snapshot")
        adv_cols = [
            "symbol",
            "advanced_signal",
            "advanced_score",
            "advanced_conf",
            "adv_rsi",
            "adv_rsi_signal",
            "adv_rsi_divergence",
            "adv_macd",
            "adv_macd_hist",
            "adv_macd_cross",
            "adv_bvb_total",
            "adv_bvb_signal",
            "adv_k_signal",
            "adv_k_smart_money",
            "adv_lp_signal",
            "adv_lp_breakout",
            "adv_lp_near_support",
            "adv_lp_near_resistance",
        ]
        adv_view = selected_df[adv_cols].copy().set_index("symbol")
        adv_view = adv_view.reindex(TOP_50_SYMBOLS)
        st.dataframe(adv_view, use_container_width=True)

    # Drift de módulos
    st.markdown("#### Module Drift Over Time")
    if all_df.empty:
        st.info("Sin histórico suficiente.")
    else:
        m = all_df.dropna(subset=["timestamp"] + MODULE_COLS, how="all").copy()
        if m.empty:
            st.info("Sin datos de módulos históricos.")
        else:
            long = m[["timestamp", *MODULE_COLS]].melt(
                id_vars="timestamp", value_vars=MODULE_COLS, var_name="module", value_name="confidence"
            ).dropna()
            fig = px.line(long, x="timestamp", y="confidence", color="module")
            fig.update_yaxes(range=[0, 1], tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

    # Learning curves opcionales via upload
    st.markdown("#### Learning Curves (Upload opcional)")
    st.caption("Sube CSV con columnas: epoch, train_loss, val_loss")
    history_file = st.file_uploader("Training history CSV", type=["csv"], key="history_uploader")
    if history_file is not None:
        try:
            h = pd.read_csv(history_file)
            if {"epoch", "train_loss", "val_loss"}.issubset(h.columns):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=h["epoch"], y=h["train_loss"], mode="lines+markers", name="train_loss"))
                fig.add_trace(go.Scatter(x=h["epoch"], y=h["val_loss"], mode="lines+markers", name="val_loss"))
                st.plotly_chart(fig, use_container_width=True)
                if h["val_loss"].min() > h["train_loss"].iloc[-1]:
                    st.warning("Gap train/val elevado: potencial overfitting.")
            else:
                st.error("CSV inválido. Requiere columnas: epoch, train_loss, val_loss.")
        except Exception as e:
            st.error(f"No se pudo leer el CSV: {str(e)}")


def show_stat_tab(all_df: pd.DataFrame) -> None:
    st.subheader("Robustness & Statistical Tests")
    op_ret = compute_operational_returns(all_df)
    if op_ret.empty:
        st.info("Sin datos para tests estadísticos.")
        return

    tests = compute_statistical_tests(op_ret)
    c1, c2, c3 = st.columns(3)

    wrc = tests.get("wrc")
    if wrc:
        c1.metric("White Reality Check p-value", f"{wrc['p_value']:.4f}")
    else:
        c1.metric("White Reality Check", "N/A")

    dsr = tests.get("dsr")
    if dsr:
        c2.metric("Deflated Sharpe Prob", f"{dsr['deflated_sharpe_ratio_prob']:.4f}")
    else:
        c2.metric("Deflated Sharpe", "N/A")

    pbo = tests.get("pbo")
    if pbo:
        c3.metric("Probability of Backtest Overfitting", f"{pbo['pbo']:.4f}")
    else:
        c3.metric("PBO", "N/A")

    if wrc and wrc["p_value"] > 0.1:
        st.warning("WRC no rechaza nulo con fuerza: edge estadístico débil.")
    if dsr and dsr["deflated_sharpe_ratio_prob"] < 0.5:
        st.warning("DSR bajo: Sharpe observado puede ser producto de sobreajuste.")
    if pbo and pbo["pbo"] > 0.5:
        st.error("PBO alto (>0.5): alto riesgo de overfitting en selección de estrategia.")

    rw = tests.get("romano_wolf")
    if isinstance(rw, pd.DataFrame) and not rw.empty:
        st.markdown("#### Romano-Wolf Correction")
        st.dataframe(rw.head(30), use_container_width=True, hide_index=True)

    # Curva de equity operacional proxy
    st.markdown("#### Operational Equity Proxy")
    daily = op_ret.groupby(op_ret["timestamp"].dt.floor("D"))["strategy_return"].mean().reset_index()
    daily["equity"] = (1.0 + daily["strategy_return"].fillna(0.0)).cumprod()
    fig = px.line(daily, x="timestamp", y="equity")
    st.plotly_chart(fig, use_container_width=True)


def show_raw_tab(reports: List[Dict], selected_report: Optional[Dict], selected_df: pd.DataFrame, tape_df: pd.DataFrame, backtest_df: pd.DataFrame) -> None:
    st.subheader("Raw Data & Traceability")
    st.caption(f"Reports loaded: {len(reports)}")
    if selected_report:
        st.json(
            {
                "timestamp": selected_report.get("timestamp"),
                "num_symbols": selected_report.get("num_symbols"),
                "file": selected_report.get("_file"),
            }
        )
    st.markdown("#### Selected report signals")
    st.dataframe(selected_df.head(200), use_container_width=True, hide_index=True)
    st.markdown("#### Signal tape")
    st.dataframe(tape_df.head(200), use_container_width=True, hide_index=True)
    st.markdown("#### Backtest summaries")
    st.dataframe(backtest_df.head(80), use_container_width=True, hide_index=True)


def main() -> None:
    set_theme()

    reports = load_reports(limit=500)
    all_df = reports_to_df(reports)
    tape_df = load_signal_tape(limit=4000)
    backtest_df = load_backtests(limit=200)

    with st.sidebar:
        st.title("Control")
        expected_universe = st.number_input("Expected tickers", min_value=1, max_value=500, value=len(TOP_50_SYMBOLS), step=1)
        auto_refresh = st.checkbox("Auto refresh (5s)", value=True)
        if st.button("Refresh now", use_container_width=True):
            st.rerun()

        selected_report = select_report(reports, expected_universe)
        labels = [report_label(r) for r in reports] if reports else []
        if reports:
            default_label = report_label(selected_report) if selected_report else labels[0]
            try:
                default_idx = labels.index(default_label)
            except ValueError:
                default_idx = 0
            mode = st.radio("Run selection", ["Most complete", "Latest", "Manual"], index=0)
            if mode == "Latest":
                selected_report = reports[0]
            elif mode == "Manual":
                choice = st.selectbox("Choose run", labels, index=default_idx)
                selected_report = reports[labels.index(choice)]
        else:
            selected_report = None

    selected_df = reports_to_df([selected_report]) if selected_report else pd.DataFrame()
    universe = TOP_50_SYMBOLS[: int(expected_universe)] if expected_universe <= len(TOP_50_SYMBOLS) else TOP_50_SYMBOLS
    universe_df = build_universe_signals(selected_df, tape_df, universe)
    represented = int((universe_df["action"] != "NO_DATA").sum()) if not universe_df.empty else 0
    missing = int((universe_df["action"] == "NO_DATA").sum()) if not universe_df.empty else len(universe)

    show_header(selected_report, represented, missing)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi("Universe Coverage", f"{represented}/{len(universe)}")
    with k2:
        kpi("Selected Run Signals", f"{len(selected_df)}")
    with k3:
        avg_conf = selected_df["confidence"].mean() if not selected_df.empty else np.nan
        kpi("Avg Confidence", f"{avg_conf*100:.1f}%" if pd.notna(avg_conf) else "N/A")
    with k4:
        risk_pass_rate = selected_df["risk_passed"].mean() if "risk_passed" in selected_df.columns and not selected_df.empty else np.nan
        kpi("Risk Pass Rate", f"{risk_pass_rate*100:.1f}%" if pd.notna(risk_pass_rate) else "N/A")

    if missing > 0:
        st.warning(
            f"Faltan {missing} tickers del universo en el run seleccionado. "
            "Se rellenan con últimas señales del tape cuando existen."
        )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Signals", "Model Audit", "Statistical Tests", "Raw Data"]
    )
    with tab1:
        show_signal_tab(universe_df, all_df)
    with tab2:
        show_model_audit_tab(universe_df, selected_df, all_df, backtest_df)
    with tab3:
        show_stat_tab(all_df)
    with tab4:
        show_raw_tab(reports, selected_report, selected_df, tape_df, backtest_df)

    if auto_refresh:
        time.sleep(5)
        st.rerun()


if __name__ == "__main__":
    main()
