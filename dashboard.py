
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time

# Importar sistema
try:
    from god_mode_complete import GodModeComplete, Config
    st.set_page_config(page_title="God Mode Trading", page_icon="🚀", layout="wide")
except ImportError:
    st.error("❌ No se pudo importar god_mode_complete.py")
    st.stop()

# ============================================================================
# INICIALIZACIÓN
# ============================================================================

@st.cache_resource
def init_system():
    """Inicializar sistema (cached)"""
    return GodModeComplete()

god_mode = init_system()

# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.title("🚀 God Mode Trading")
st.sidebar.markdown("---")

page = st.sidebar.selectbox(
    "Navegación",
    ["📊 Dashboard", "🔍 Análisis", "💼 Portfolio", "📈 Backtest", "⚙️ Configuración"]
)

st.sidebar.markdown("---")
st.sidebar.info(f"""
**Capital**: ${god_mode.risk_manager.capital:,.0f}  
**Posiciones**: {len(god_mode.paper_trading.portfolio['positions'])}  
**Trades**: {len(god_mode.paper_trading.portfolio['trades_history'])}
""")

# ============================================================================
# PÁGINA: DASHBOARD
# ============================================================================

if page == "📊 Dashboard":
    st.title("📊 Trading Dashboard")

    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)

    portfolio_value = god_mode.paper_trading.get_portfolio_value()
    cash = god_mode.paper_trading.portfolio['cash']
    positions = len(god_mode.paper_trading.portfolio['positions'])
    trades = len(god_mode.paper_trading.portfolio['trades_history'])

    col1.metric("Portfolio Value", f"${portfolio_value:,.0f}", 
                f"{((portfolio_value/Config.DEFAULT_CAPITAL - 1)*100):+.1f}%")
    col2.metric("Cash", f"${cash:,.0f}")
    col3.metric("Positions", positions)
    col4.metric("Total Trades", trades)

    st.markdown("---")

    # Watch list
    st.subheader("📋 Watchlist")

    watchlist_symbols = st.multiselect(
        "Seleccionar símbolos",
        ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'META', 'AMZN', 'NFLX'],
        default=['AAPL', 'MSFT', 'TSLA']
    )

    if st.button("🔄 Actualizar Datos"):
        with st.spinner("Analizando símbolos..."):
            results = []
            progress_bar = st.progress(0)

            for i, symbol in enumerate(watchlist_symbols):
                try:
                    result = god_mode.analyze_symbol(symbol, period="1mo")
                    results.append({
                        'Symbol': symbol,
                        'Price': f"${result['current_price']:.2f}",
                        'Change': f"{result['price_change']:+.2f}%",
                        'Signal': result['signal'].action,
                        'Confidence': f"{result['confidence_breakdown']['overall']:.1%}",
                        'Sentiment': f"{result['sentiment']:.2f}"
                    })
                except Exception as e:
                    st.error(f"Error en {symbol}: {str(e)[:100]}")

                progress_bar.progress((i + 1) / len(watchlist_symbols))
                time.sleep(0.5)

            if results:
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)

    # Gráfico de equity curve
    st.markdown("---")
    st.subheader("📈 Equity Curve")

    if god_mode.paper_trading.portfolio['trades_history']:
        trades_df = pd.DataFrame(god_mode.paper_trading.portfolio['trades_history'])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=trades_df.index,
            y=[Config.DEFAULT_CAPITAL] * len(trades_df),
            mode='lines',
            name='Capital Inicial',
            line=dict(dash='dash')
        ))

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No hay trades para mostrar")

# ============================================================================
# PÁGINA: ANÁLISIS
# ============================================================================

elif page == "🔍 Análisis":
    st.title("🔍 Análisis de Símbolo")

    col1, col2 = st.columns([2, 1])

    with col1:
        symbol = st.text_input("Símbolo", value="AAPL").upper()

    with col2:
        period = st.selectbox("Período", ["1mo", "3mo", "6mo", "1y"], index=1)

    execute_trade = st.checkbox("Ejecutar trade automático", value=False)

    if st.button("🚀 Analizar"):
        with st.spinner(f"Analizando {symbol}..."):
            try:
                result = god_mode.analyze_symbol(symbol, period=period, execute_trade=execute_trade)

                # Información básica
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Precio", f"${result['current_price']:.2f}")
                col2.metric("Cambio", f"{result['price_change']:+.2f}%")
                col3.metric("P/E", f"{result['quote'].get('pe_ratio', 0):.2f}")
                col4.metric("Volume", f"{result['quote'].get('volume', 0):,}")

                st.markdown("---")

                # Señal de trading
                st.subheader("🎯 Señal de Trading")

                signal = result['signal']
                action_color = {'BUY': 'green', 'SELL': 'red', 'HOLD': 'orange'}[signal.action]

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown(f"### {signal.action}")
                    st.markdown(f"**Confianza**: {signal.confidence:.1%}")

                with col2:
                    st.markdown(f"**Position Size**: {signal.position_size:.4f}")
                    st.markdown(f"**Valor**: ${signal.position_size * signal.price:,.2f}")

                with col3:
                    st.markdown(f"**Stop Loss**: {signal.stop_loss:.1%}")
                    st.markdown(f"**Take Profit**: {signal.take_profit:.1%}")

                # Risk check
                risk_check = result['risk_check']
                if risk_check['passed']:
                    st.success(f"✅ Risk Check: {risk_check['message']}")
                else:
                    st.error(f"❌ Risk Check: {risk_check['message']}")

                st.markdown("---")

                # Confianza por módulo
                st.subheader("📊 Confianza por Módulo")

                confidence = result['confidence_breakdown']
                conf_df = pd.DataFrame({
                    'Módulo': ['Temporal', 'Visión', 'Tabular', 'NLP', 'Grafos'],
                    'Confianza': [
                        confidence['temporal'],
                        confidence['vision'],
                        confidence['tabular'],
                        confidence['nlp'],
                        confidence['graph']
                    ]
                })

                fig = px.bar(conf_df, x='Módulo', y='Confianza', 
                            title=f"Confianza General: {confidence['overall']:.1%}")
                st.plotly_chart(fig, use_container_width=True)

                # Trade execution result
                if 'trade_result' in result:
                    st.markdown("---")
                    st.subheader("💼 Resultado del Trade")

                    trade_result = result['trade_result']
                    if trade_result['success']:
                        st.success("✅ Trade ejecutado exitosamente")
                        st.json(trade_result)
                    else:
                        st.error(f"❌ Error: {trade_result.get('error', 'Unknown')}")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# ============================================================================
# PÁGINA: PORTFOLIO
# ============================================================================

elif page == "💼 Portfolio":
    st.title("💼 Portfolio Management")

    portfolio = god_mode.paper_trading.get_portfolio_summary()

    # Métricas
    col1, col2, col3 = st.columns(3)
    col1.metric("Cash", f"${portfolio['cash']:,.2f}")
    col2.metric("Portfolio Value", f"${portfolio['total_value']:,.2f}")
    col3.metric("P&L", f"${portfolio['total_value'] - Config.DEFAULT_CAPITAL:+,.2f}")

    st.markdown("---")

    # Posiciones abiertas
    st.subheader("📍 Posiciones Abiertas")

    if portfolio['positions_detail']:
        positions_data = []
        for symbol, pos in portfolio['positions_detail'].items():
            positions_data.append({
                'Symbol': symbol,
                'Quantity': f"{pos['quantity']:.4f}",
                'Avg Price': f"${pos['avg_price']:.2f}",
                'Current Value': f"${pos['quantity'] * pos['avg_price']:,.2f}",
                'Stop Loss': f"{pos['stop_loss']:.1%}",
                'Take Profit': f"{pos['take_profit']:.1%}"
            })

        st.dataframe(pd.DataFrame(positions_data), use_container_width=True)
    else:
        st.info("No hay posiciones abiertas")

    st.markdown("---")

    # Historial de trades
    st.subheader("📜 Historial de Trades")

    if portfolio['num_trades'] > 0:
        trades_df = pd.DataFrame(god_mode.paper_trading.portfolio['trades_history'])
        st.dataframe(trades_df, use_container_width=True)

        # Botón de descarga
        csv = trades_df.to_csv(index=False)
        st.download_button(
            "⬇️ Descargar Historial",
            csv,
            "trades_history.csv",
            "text/csv"
        )
    else:
        st.info("No hay trades ejecutados")

# ============================================================================
# PÁGINA: BACKTEST
# ============================================================================

elif page == "📈 Backtest":
    st.title("📈 Backtesting")

    col1, col2, col3 = st.columns(3)

    with col1:
        symbol = st.text_input("Símbolo", value="AAPL").upper()

    with col2:
        start_date = st.date_input("Fecha inicio", datetime.now() - timedelta(days=365))

    with col3:
        end_date = st.date_input("Fecha fin", datetime.now())

    if st.button("🚀 Ejecutar Backtest"):
        with st.spinner("Ejecutando backtest..."):
            try:
                results = god_mode.backtest(
                    symbol,
                    start_date=str(start_date),
                    end_date=str(end_date)
                )

                # Métricas
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Retorno", f"{results['total_return_pct']:+.2f}%")
                col2.metric("Sharpe", f"{results['sharpe_ratio']:.2f}")
                col3.metric("Max DD", f"{results['max_drawdown_pct']:.2f}%")
                col4.metric("Win Rate", f"{results['win_rate']:.1f}%")

                st.markdown("---")

                # Equity curve
                st.subheader("Equity Curve")

                equity_df = pd.DataFrame({
                    'Day': range(len(results['equity_curve'])),
                    'Value': results['equity_curve']
                })

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=equity_df['Day'],
                    y=equity_df['Value'],
                    mode='lines',
                    name='Portfolio Value',
                    fill='tozeroy'
                ))
                fig.add_hline(y=Config.DEFAULT_CAPITAL, line_dash="dash", 
                             annotation_text="Capital Inicial")

                st.plotly_chart(fig, use_container_width=True)

                # Trades
                st.subheader("Trades Ejecutados")
                trades_df = pd.DataFrame(results['trades'])
                st.dataframe(trades_df, use_container_width=True)

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# ============================================================================
# PÁGINA: CONFIGURACIÓN
# ============================================================================

elif page == "⚙️ Configuración":
    st.title("⚙️ Configuración")

    st.subheader("Trading Parameters")

    col1, col2 = st.columns(2)

    with col1:
        capital = st.number_input("Capital Inicial", value=Config.DEFAULT_CAPITAL, step=10000)
        max_position = st.slider("Max Position Size", 0.0, 1.0, Config.MAX_POSITION_SIZE)

    with col2:
        risk_per_trade = st.slider("Risk per Trade", 0.0, 0.10, Config.RISK_PER_TRADE)
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.75)

    if st.button("💾 Guardar Configuración"):
        Config.DEFAULT_CAPITAL = capital
        Config.MAX_POSITION_SIZE = max_position
        Config.RISK_PER_TRADE = risk_per_trade

        god_mode.risk_manager.capital = capital

        st.success("✅ Configuración guardada")

    st.markdown("---")

    st.subheader("API Keys")

    av_key = st.text_input("Alpha Vantage API Key", value=Config.ALPHA_VANTAGE_KEY, type="password")
    alpaca_key = st.text_input("Alpaca API Key", value=Config.ALPACA_API_KEY, type="password")
    alpaca_secret = st.text_input("Alpaca Secret Key", value=Config.ALPACA_SECRET_KEY, type="password")

    if st.button("🔑 Actualizar Keys"):
        Config.ALPHA_VANTAGE_KEY = av_key
        Config.ALPACA_API_KEY = alpaca_key
        Config.ALPACA_SECRET_KEY = alpaca_secret
        st.success("✅ Keys actualizadas")

# ============================================================================
# FOOTER
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 🚀 God Mode v3.0
**Powered by:**
- ARIMA, GARCH, TFT, N-BEATS
- CNN, ResNet, YOLO
- LightGBM, CatBoost
- FinBERT, GPT-4, NER
- GCN, GAT
- SAC (Reinforcement Learning)
- TimeGAN

""")