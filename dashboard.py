import json
import subprocess
import sys
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from watchlists import TOP_50_SYMBOLS_CSV

st.set_page_config(page_title="God Mode Command Deck", page_icon="GM", layout="wide")

try:
    from god_mode_complete import GodModeComplete
except ImportError:
    st.error("No se pudo importar god_mode_complete.py")
    st.stop()


BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

ACTION_EMOJI = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}
ACTION_COLOR = {"BUY": "#0f766e", "SELL": "#b91c1c", "HOLD": "#b45309"}
MODEL_COLS = ["temporal", "vision", "tabular", "nlp", "graph", "options", "base_model", "overall"]


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
        .stApp {
            background:
                radial-gradient(1200px 400px at -5% -10%, rgba(15,118,110,0.24), transparent 60%),
                radial-gradient(900px 300px at 110% -15%, rgba(180,83,9,0.25), transparent 60%),
                linear-gradient(135deg, #f6f1e8, #dbe7e4);
        }
        .hero, .glass {
            background: rgba(255,255,255,0.88);
            border: 1px solid #adc7c0;
            border-radius: 14px;
            box-shadow: 0 12px 26px rgba(18,32,39,0.10);
            padding: .85rem .95rem;
        }
        .hero h1 { margin: 0; font-size: 1.5rem; font-family: "Space Grotesk", sans-serif; }
        .hero p { margin: .25rem 0 0 0; color: #3f5a63; }
        .mono { font-family: "IBM Plex Mono", monospace; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state() -> None:
    defaults = {
        "pipeline_process": None,
        "pipeline_log_handle": None,
        "pipeline_log_path": None,
        "pipeline_command": None,
        "pipeline_started_at": None,
        "pipeline_exit_code": None,
        "analysis_result": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_resource
def init_system() -> GodModeComplete:
    return GodModeComplete()


def parse_symbols(raw: str) -> List[str]:
    return [s.strip().upper() for s in raw.replace(";", ",").split(",") if s.strip()]


def sync_process_state() -> None:
    proc = st.session_state.get("pipeline_process")
    if proc is None:
        return
    if proc.poll() is not None:
        st.session_state["pipeline_exit_code"] = proc.returncode
        handle = st.session_state.get("pipeline_log_handle")
        if handle and not handle.closed:
            handle.close()
        st.session_state["pipeline_process"] = None
        st.session_state["pipeline_log_handle"] = None


def process_running() -> bool:
    proc = st.session_state.get("pipeline_process")
    return proc is not None and proc.poll() is None


def start_pipeline(symbols: List[str], execute: bool, loop: bool, interval: int) -> str:
    if process_running():
        return "Pipeline ya esta corriendo."
    if not symbols:
        return "Debes indicar al menos 1 simbolo."

    cmd = [sys.executable, "-u", "main.py", "--symbols", *symbols]
    if execute:
        cmd.append("--execute")
    if loop:
        cmd.extend(["--loop", "--interval", str(max(1, int(interval)))])

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"dashboard_pipeline_{stamp}.log"
    log_handle = open(log_path, "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(cmd, cwd=str(BASE_DIR), stdout=log_handle, stderr=subprocess.STDOUT)

    st.session_state["pipeline_process"] = proc
    st.session_state["pipeline_log_handle"] = log_handle
    st.session_state["pipeline_log_path"] = str(log_path)
    st.session_state["pipeline_command"] = " ".join(cmd)
    st.session_state["pipeline_started_at"] = datetime.now().isoformat(timespec="seconds")
    st.session_state["pipeline_exit_code"] = None
    return "Pipeline iniciado."


def stop_pipeline() -> str:
    proc = st.session_state.get("pipeline_process")
    if proc is None:
        return "No hay pipeline en ejecucion."
    try:
        proc.terminate()
        proc.wait(timeout=8)
    except Exception:
        proc.kill()
    handle = st.session_state.get("pipeline_log_handle")
    if handle and not handle.closed:
        handle.close()
    st.session_state["pipeline_exit_code"] = proc.returncode
    st.session_state["pipeline_process"] = None
    st.session_state["pipeline_log_handle"] = None
    return "Pipeline detenido."


def read_log_tail(path: Optional[str], max_lines: int = 140) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.exists():
        return ""
    lines: deque = deque(maxlen=max_lines)
    with p.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            lines.append(line.rstrip("\n"))
    return "\n".join(lines)


def load_reports(limit: int = 200) -> List[Dict]:
    files = sorted(LOGS_DIR.glob("pipeline_report_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)[:limit]
    out = []
    for file_path in files:
        try:
            with file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            data["_file"] = str(file_path)
            out.append(data)
        except Exception:
            pass
    return out


def reports_to_df(reports: List[Dict]) -> pd.DataFrame:
    rows = []
    for report in reports:
        run_ts = pd.to_datetime(report.get("timestamp"), format="%Y%m%d_%H%M%S", errors="coerce")
        for signal in report.get("signals", []) or []:
            model = signal.get("model_confidence", {}) or {}
            sig_ts = pd.to_datetime(signal.get("timestamp"), errors="coerce")
            if pd.isna(sig_ts):
                sig_ts = run_ts
            rows.append(
                {
                    "timestamp": sig_ts,
                    "symbol": str(signal.get("symbol", "")).upper(),
                    "action": str(signal.get("action", "HOLD")).upper(),
                    "price": float(signal.get("price", 0.0)),
                    "confidence": float(signal.get("confidence", 0.0)),
                    "position_size": float(signal.get("position_size", 0.0)),
                    "risk_passed": bool(signal.get("risk_check", {}).get("passed", False)),
                    "options_bias": str(signal.get("options_analysis", {}).get("directional_bias", "N/A")),
                    "temporal": float(model.get("temporal", float("nan"))),
                    "vision": float(model.get("vision", float("nan"))),
                    "tabular": float(model.get("tabular", float("nan"))),
                    "nlp": float(model.get("nlp", float("nan"))),
                    "graph": float(model.get("graph", float("nan"))),
                    "options": float(model.get("options", float("nan"))),
                    "base_model": float(model.get("base_model", float("nan"))),
                    "overall": float(model.get("overall", float("nan"))),
                    "source": "pipeline_report",
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp", ascending=False)
    return df


def load_signal_tape(limit: int = 300) -> pd.DataFrame:
    path = LOGS_DIR / "signals.jsonl"
    if not path.exists():
        return pd.DataFrame()
    lines: deque = deque(maxlen=limit)
    with path.open("r", encoding="utf-8", errors="ignore") as f:
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
                    "source": "signals.jsonl",
                }
            )
        except Exception:
            pass
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("timestamp", ascending=False)
    return df


def pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def action_label(action: str) -> str:
    return f"{ACTION_EMOJI.get(action, '⚪')} {action}"


def radar(conf_map: Dict[str, float]) -> go.Figure:
    labels = list(conf_map.keys())
    values = [float(conf_map[k]) * 100 for k in labels]
    labels += [labels[0]]
    values += [values[0]]
    fig = go.Figure(go.Scatterpolar(r=values, theta=labels, fill="toself", fillcolor="rgba(15,118,110,0.2)", line=dict(color="#0f766e")))
    fig.update_layout(margin=dict(l=8, r=8, t=25, b=8), showlegend=False, polar=dict(radialaxis=dict(range=[0, 100], visible=True)))
    return fig


def gauge(confidence: float, action: str) -> go.Figure:
    fig = go.Figure(go.Indicator(mode="gauge+number", value=max(0.0, min(confidence * 100, 100.0)), number={"suffix": "%"}, title={"text": f"SAC: {action}"}))
    fig.update_traces(gauge={"axis": {"range": [0, 100]}, "bar": {"color": ACTION_COLOR.get(action, "#3f5a63")}})
    fig.update_layout(margin=dict(l=8, r=8, t=40, b=8))
    return fig


def main() -> None:
    inject_css()
    init_state()
    sync_process_state()
    god_mode = init_system()

    reports = load_reports()
    report_df = reports_to_df(reports)
    tape_df = load_signal_tape()

    with st.sidebar:
        st.title("Command Deck")
        page = st.radio("Vista", ["Dashboard", "Pipeline", "Modelos", "Senales", "Analisis"], index=0)
        auto_refresh = st.checkbox("Auto refresh (5s)", value=False)
        st.caption(f"Reportes: {len(reports)}")
        st.caption(f"Tape: {len(tape_df)}")

    status = "RUNNING" if process_running() else "IDLE"
    st.markdown(
        f"""
        <div class="hero">
            <h1>God Mode Trading - Performance Deck</h1>
            <p>Control de pipeline, rendimiento de modelos y resultados BUY/SELL/HOLD en una sola interfaz.</p>
            <p class="mono">Pipeline: {status}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if page == "Dashboard":
        portfolio = god_mode.paper_trading.get_portfolio_summary()
        latest = reports[0]["signals"] if reports else []
        latest_counter = Counter([str(s.get("action", "HOLD")).upper() for s in latest])
        avg_conf = report_df["confidence"].head(80).mean() if not report_df.empty else 0.0

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Portfolio", f"${portfolio['total_value']:,.2f}")
        k2.metric("Cash", f"${portfolio['cash']:,.2f}")
        k3.metric("Avg Confidence", pct(avg_conf))
        k4.metric("Last Run Mix", f"B {latest_counter.get('BUY', 0)} / S {latest_counter.get('SELL', 0)} / H {latest_counter.get('HOLD', 0)}")

        c1, c2 = st.columns((1.2, 1))
        with c1:
            if report_df.empty:
                st.info("No hay reportes aun.")
            else:
                fig = px.line(report_df.head(150), x="timestamp", y="confidence", color="symbol", markers=True)
                fig.update_yaxes(range=[0, 1], tickformat=".0%")
                st.plotly_chart(fig, use_container_width=True)
        with c2:
            if report_df.empty:
                st.info("No hay acciones para distribucion.")
            else:
                pie = report_df.head(160)["action"].value_counts().rename_axis("action").reset_index(name="count")
                fig = px.pie(pie, names="action", values="count", hole=0.55, color="action", color_discrete_map=ACTION_COLOR)
                st.plotly_chart(fig, use_container_width=True)

    if page == "Pipeline":
        i1, i2, i3 = st.columns(3)
        with i1:
            symbols_raw = st.text_input("Symbols", TOP_50_SYMBOLS_CSV)
        with i2:
            execute = st.checkbox("Execute paper trades", value=False)
        with i3:
            loop = st.checkbox("Loop mode", value=False)
        interval = st.number_input("Interval (seconds)", min_value=30, value=3600, step=30, disabled=not loop)

        b1, b2, b3 = st.columns(3)
        if b1.button("Run Pipeline", use_container_width=True):
            st.success(start_pipeline(parse_symbols(symbols_raw), execute, loop, int(interval)))
        if b2.button("Stop Pipeline", use_container_width=True):
            st.warning(stop_pipeline())
        if b3.button("Refresh", use_container_width=True):
            st.rerun()

        st.markdown("#### Estado")
        if process_running():
            st.success("Pipeline activo.")
        elif st.session_state.get("pipeline_exit_code") is not None:
            st.info(f"Pipeline finalizado con exit code {st.session_state['pipeline_exit_code']}.")
        else:
            st.info("Pipeline inactivo.")

        if st.session_state.get("pipeline_command"):
            st.code(st.session_state["pipeline_command"], language="bash")
        if st.session_state.get("pipeline_log_path"):
            st.caption(f"Log: {st.session_state['pipeline_log_path']}")
            st.code(read_log_tail(st.session_state["pipeline_log_path"]), language="bash")

        st.markdown("#### Ultimo resultado BUY / SELL / HOLD")
        if reports and reports[0].get("signals"):
            latest_df = pd.DataFrame(reports[0]["signals"])
            latest_df["action"] = latest_df["action"].str.upper().apply(action_label)
            latest_df["price"] = latest_df["price"].map(lambda x: f"${x:,.2f}")
            latest_df["confidence"] = (latest_df["confidence"] * 100).map(lambda x: f"{x:.1f}%")
            st.dataframe(latest_df[["symbol", "action", "price", "confidence", "position_size"]], use_container_width=True, hide_index=True)
        else:
            st.info("Sin reporte reciente.")

    if page == "Modelos":
        mod_df = report_df.dropna(subset=MODEL_COLS, how="all") if not report_df.empty else pd.DataFrame()
        if mod_df.empty:
            st.warning("No hay metricas por modulo. Ejecuta pipeline actualizado para llenar este panel.")
        else:
            avg = mod_df[MODEL_COLS].mean(numeric_only=True).dropna()
            cols = st.columns(min(4, len(avg)) if len(avg) else 1)
            for idx, (name, val) in enumerate(avg.items()):
                cols[idx % len(cols)].metric(name.upper(), pct(float(val)))

            long = mod_df[["timestamp", *MODEL_COLS]].melt(id_vars=["timestamp"], var_name="module", value_name="confidence").dropna()
            fig = px.line(long, x="timestamp", y="confidence", color="module")
            fig.update_yaxes(range=[0, 1], tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

            latest = mod_df.iloc[0]
            st.plotly_chart(
                radar(
                    {
                        "Temporal": latest["temporal"],
                        "Vision": latest["vision"],
                        "Tabular": latest["tabular"],
                        "NLP": latest["nlp"],
                        "Graph": latest["graph"],
                    }
                ),
                use_container_width=True,
            )

    if page == "Senales":
        frames = []
        if not report_df.empty:
            frames.append(report_df[["timestamp", "symbol", "action", "price", "confidence", "position_size", "source"]])
        if not tape_df.empty:
            frames.append(tape_df[["timestamp", "symbol", "action", "price", "confidence", "position_size", "source"]])
        if not frames:
            st.info("No hay senales disponibles.")
        else:
            df = pd.concat(frames, ignore_index=True).sort_values("timestamp", ascending=False)
            actions = st.multiselect("Filtrar accion", ["BUY", "SELL", "HOLD"], default=["BUY", "SELL", "HOLD"])
            syms = sorted(df["symbol"].dropna().unique().tolist())
            symbols_filter = st.multiselect("Filtrar simbolo", syms, default=syms[:10])
            filt = df[df["action"].isin(actions) & df["symbol"].isin(symbols_filter if symbols_filter else syms)].copy()

            c1, c2, c3 = st.columns(3)
            counts = filt["action"].value_counts()
            c1.metric("BUY", int(counts.get("BUY", 0)))
            c2.metric("SELL", int(counts.get("SELL", 0)))
            c3.metric("HOLD", int(counts.get("HOLD", 0)))

            fig = px.histogram(filt, x="confidence", color="action", nbins=20, barmode="overlay", color_discrete_map=ACTION_COLOR)
            fig.update_xaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

            filt["action"] = filt["action"].map(action_label)
            filt["price"] = filt["price"].map(lambda x: f"${x:,.2f}")
            filt["confidence"] = (filt["confidence"] * 100).map(lambda x: f"{x:.1f}%")
            filt["timestamp"] = pd.to_datetime(filt["timestamp"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
            st.dataframe(filt.head(300), use_container_width=True, hide_index=True)

    if page == "Analisis":
        i1, i2, i3 = st.columns(3)
        with i1:
            symbol = st.text_input("Symbol", value="AAPL").upper()
        with i2:
            period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y"], index=1)
        with i3:
            do_trade = st.checkbox("Execute trade", value=False)

        if st.button("Analizar simbolo", use_container_width=True):
            with st.spinner(f"Analizando {symbol}..."):
                try:
                    st.session_state["analysis_result"] = god_mode.analyze_symbol(symbol, period=period, execute_trade=do_trade)
                except Exception as e:
                    st.error(str(e))

        result = st.session_state.get("analysis_result")
        if result:
            signal = result["signal"]
            st.metric("Decision", action_label(signal.action))
            c1, c2, c3 = st.columns(3)
            c1.metric("Confidence", pct(signal.confidence))
            c2.metric("Price", f"${result['current_price']:,.2f}")
            c3.metric("Change", f"{result['price_change']:+.2f}%")
            st.plotly_chart(gauge(signal.confidence, signal.action), use_container_width=True)
            conf = result.get("confidence_breakdown", {})
            st.plotly_chart(
                radar(
                    {
                        "Temporal": conf.get("temporal", 0.0),
                        "Vision": conf.get("vision", 0.0),
                        "Tabular": conf.get("tabular", 0.0),
                        "NLP": conf.get("nlp", 0.0),
                        "Graph": conf.get("graph", 0.0),
                    }
                ),
                use_container_width=True,
            )
        else:
            st.info("Ejecuta un analisis para ver BUY/SELL/HOLD y performance por modelo.")

    if auto_refresh:
        time.sleep(5)
        st.rerun()


if __name__ == "__main__":
    main()
