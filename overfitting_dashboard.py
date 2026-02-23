#!/usr/bin/env python3
"""
Dashboard de monitoreo de sobreajuste / degradacion.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st

from overfitting_audit import overfitting_alerts_from_metrics
from watchlists import TOP_50_SYMBOLS


BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = BASE_DIR / "logs"


def load_pipeline_reports(limit: int = 200) -> List[Dict]:
    files = sorted(LOGS_DIR.glob("pipeline_report_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    out = []
    for f in files:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            d["_file"] = str(f)
            out.append(d)
        except Exception:
            continue
    return out


def reports_to_df(reports: List[Dict]) -> pd.DataFrame:
    rows = []
    for r in reports:
        ts = pd.to_datetime(r.get("timestamp"), format="%Y%m%d_%H%M%S", errors="coerce")
        for s in r.get("signals", []) or []:
            rows.append(
                {
                    "timestamp": pd.to_datetime(s.get("timestamp"), errors="coerce")
                    if s.get("timestamp")
                    else ts,
                    "symbol": str(s.get("symbol", "")).upper(),
                    "action": str(s.get("action", "HOLD")).upper(),
                    "confidence": float(s.get("confidence", 0.0)),
                    "risk_passed": bool(s.get("risk_check", {}).get("passed", False)),
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("timestamp", ascending=False)
    return df


def load_backtests(limit: int = 100) -> pd.DataFrame:
    files = sorted(LOGS_DIR.glob("backtest_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    rows = []
    for f in files:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            d["file"] = f.name
            rows.append(d)
        except Exception:
            continue
    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(page_title="Overfitting Monitor", page_icon="OF", layout="wide")
    st.title("Overfitting Monitor")
    st.caption("Monitoreo continuo de degradacion train/test/OOS")

    with st.sidebar:
        auto = st.checkbox("Auto refresh (10s)", value=True)
        if st.button("Refresh"):
            st.rerun()

    reports = load_pipeline_reports(260)
    df = reports_to_df(reports)
    bt = load_backtests(120)

    if df.empty:
        st.warning("No hay pipeline_report en logs/")
        return

    latest_ts = df["timestamp"].max()
    latest_df = df[df["timestamp"] == latest_ts]
    represented = sorted(set(latest_df["symbol"]))
    missing = [s for s in TOP_50_SYMBOLS if s not in represented]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Signals latest run", len(latest_df))
    c2.metric("Tickers represented", len(represented))
    c3.metric("Missing from Top50", len(missing))
    c4.metric("Risk fails", int((latest_df["risk_passed"] == False).sum()))  # noqa: E712

    if missing:
        st.error("Missing tickers in latest run: " + ", ".join(missing[:20]) + (" ..." if len(missing) > 20 else ""))

    st.subheader("Confidence drift")
    roll = df.dropna(subset=["timestamp"]).sort_values("timestamp").tail(1200).copy()
    g = roll.groupby("timestamp", as_index=False)["confidence"].mean()
    fig = px.line(g, x="timestamp", y="confidence")
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    if len(g) > 20:
        recent = g["confidence"].tail(10).mean()
        past = g["confidence"].head(max(1, len(g) - 10)).tail(30).mean()
        if past > 0:
            drop = (past - recent) / past
            if drop > 0.2:
                st.error(f"ALERTA: confidence media cayó {drop*100:.1f}% vs ventana previa.")

    st.subheader("Action concentration")
    mix = latest_df["action"].value_counts(normalize=True)
    if not mix.empty and mix.max() > 0.8:
        st.warning(f"ALERTA: concentracion de accion alta ({mix.idxmax()} {mix.max()*100:.1f}%).")
    st.dataframe((mix * 100).rename("pct").reset_index(), use_container_width=True, hide_index=True)

    st.subheader("Backtest health")
    if bt.empty:
        st.info("No hay archivos backtest_*.json en logs/.")
    else:
        st.dataframe(bt.head(30), use_container_width=True, hide_index=True)
        train_metrics = {
            "sharpe_ratio": float(bt["sharpe_ratio"].head(1).iloc[0]) if "sharpe_ratio" in bt.columns else 0.0,
            "max_drawdown": float(bt["max_drawdown_pct"].head(1).iloc[0]) if "max_drawdown_pct" in bt.columns else 0.0,
        }
        # Proxy OOS: pipeline confidence/risk (no hay PnL OOS real persistido aún)
        test_metrics = {
            "accuracy": float(latest_df["confidence"].mean()),
            "sharpe_ratio": float(latest_df["confidence"].mean()) * 2.0,
            "max_drawdown": float((latest_df["risk_passed"] == False).mean()) * 100.0,  # noqa: E712
        }
        alerts = overfitting_alerts_from_metrics(train_metrics, test_metrics)
        for a in alerts:
            st.warning(a)

    if auto:
        time.sleep(10)
        st.rerun()


if __name__ == "__main__":
    main()

