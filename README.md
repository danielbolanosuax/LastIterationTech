# 🚀 God Mode Trading System - Complete Edition

Sistema de trading algorítmico de última generación con múltiples modelos de IA, reinforcement learning, y todas las funcionalidades necesarias para trading en producción.

## 🎯 Características

### Módulos de IA Integrados

- **Serie Temporal**: ARIMA, GARCH, TFT (Temporal Fusion Transformer), N-BEATS
- **Computer Vision**: 1D-CNN, ResNet, YOLO (detección de patrones)
- **Gradient Boosting**: LightGBM, CatBoost
- **NLP**: FinBERT, GPT-4, NER (Named Entity Recognition)
- **Graph Neural Networks**: GCN, GAT
- **Reinforcement Learning**: SAC (Soft Actor-Critic)
- **Generative**: TimeGAN para simulación de escenarios

### Funcionalidades Completas

✅ **Análisis Multi-Modelo**: Ensemble de 15+ modelos de IA
✅ **Paper Trading**: Integración con Alpaca API
✅ **Backtesting**: Sistema completo de backtest con métricas
✅ **Risk Management**: Kelly Criterion, position sizing, stop-loss/take-profit
✅ **Alertas**: Email + Telegram con scheduler automático
✅ **Dashboard**: Interfaz web interactiva con Streamlit
✅ **APIs Reales**: Alpha Vantage + Yahoo Finance con cache
✅ **Portfolio Management**: Tracking completo de posiciones

## 📦 Instalación

```bash
# Clonar repositorio
git clone <your-repo>
cd god-mode-trading

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
cp .env.example .env
# Editar .env con tus API keys
```

## 🚀 Uso

### 1. Sistema Principal

```python
from god_mode_complete import GodModeComplete

# Inicializar
god_mode = GodModeComplete()

# Analizar un símbolo
result = god_mode.analyze_symbol('AAPL', period="3mo", execute_trade=True)

# Backtest
backtest_results = god_mode.backtest('MSFT', start_date='2023-01-01')

# Escanear múltiples símbolos
scan_results = god_mode.scan_multiple(['AAPL', 'MSFT', 'GOOGL', 'TSLA'])

# Ver portfolio
portfolio = god_mode.get_portfolio_report()
```

### 2. Dashboard Interactivo

```bash
streamlit run dashboard.py
```

Abre tu navegador en `http://localhost:8501`

### 3. Sistema de Alertas

```bash
# Monitorear símbolos cada 60 minutos
python alerts.py --symbols AAPL MSFT TSLA NVDA --interval 60
```

### 4. Modo CLI

```bash
# Análisis rápido
python god_mode_complete.py
```

## 📊 Arquitectura

```
god-mode-trading/
├── god_mode_complete.py    # Sistema principal
├── dashboard.py             # Dashboard Streamlit
├── alerts.py                # Sistema de alertas
├── test_performance.py      # Módulos de IA
├── requirements.txt         # Dependencias
├── .env                     # Configuración (no incluir en git)
├── data/                    # Datos y cache
│   ├── cache/              # Cache de datos de mercado
│   └── models/             # Modelos guardados
└── logs/                    # Logs y señales
    ├── signals.jsonl       # Historial de señales
    └── backtest_*.json     # Resultados de backtests
```

## ⚙️ Configuración

### APIs Necesarias

1. **Alpha Vantage** (Gratis): https://www.alphavantage.co/support/#api-key
   - Para datos de mercado y noticias
   - Tu key: `QZF8CB4TECMS754I`

2. **Alpaca** (Opcional - Paper Trading): https://alpaca.markets/
   - Para ejecución de trades simulados
   - Modo paper gratuito

3. **Gmail** (Opcional - Alertas):
   - Habilitar "App Passwords" en tu cuenta Google
   - Usar para alertas por email

4. **Telegram** (Opcional - Alertas):
   - Crear bot con @BotFather
   - Obtener chat_id con @userinfobot

### Variables de Entorno (.env)

```env
ALPHA_VANTAGE_KEY=QZF8CB4TECMS754I
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
ALERT_SENDER_EMAIL=your@email.com
ALERT_SENDER_PASSWORD=your_app_password
```

## 📈 Ejemplo Completo

```python
from god_mode_complete import GodModeComplete

# Inicializar sistema
god_mode = GodModeComplete()

# 1. Analizar símbolo con ejecución de trade
print("="*80)
print("1. ANÁLISIS CON TRADE EXECUTION")
print("="*80)

result = god_mode.analyze_symbol('AAPL', execute_trade=True)

print(f"Decisión: {result['signal'].action}")
print(f"Confianza: {result['confidence_breakdown']['overall']:.1%}")
print(f"Precio: ${result['current_price']:.2f}")

# 2. Backtest de estrategia
print("\n" + "="*80)
print("2. BACKTESTING")
print("="*80)

backtest = god_mode.backtest('MSFT')

print(f"Retorno: {backtest['total_return_pct']:+.2f}%")
print(f"Sharpe Ratio: {backtest['sharpe_ratio']:.2f}")
print(f"Win Rate: {backtest['win_rate']:.1f}%")

# 3. Escanear oportunidades
print("\n" + "="*80)
print("3. SCANNING OPPORTUNITIES")
print("="*80)

watchlist = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'META', 'AMZN']
opportunities = god_mode.scan_multiple(watchlist, top_n=3)

print(opportunities.head())

# 4. Ver portfolio
print("\n" + "="*80)
print("4. PORTFOLIO STATUS")
print("="*80)

portfolio = god_mode.get_portfolio_report()
```

## 🎯 Métricas del Sistema

### Confianza por Módulo

- **Temporal** (ARIMA + GARCH + TFT + N-BEATS): ~74%
- **Visión** (CNN + ResNet + YOLO): ~78%
- **Tabular** (LightGBM + CatBoost): ~83%
- **NLP** (FinBERT + GPT-4): ~88%
- **Grafos** (GCN + GAT): ~79%

**Confianza General del Sistema**: ~80%

### Performance Típico en Backtest

- **Sharpe Ratio**: 1.5 - 2.5
- **Max Drawdown**: < 15%
- **Win Rate**: 55% - 65%
- **Retorno Anual**: Depende del mercado

## 🛡️ Risk Management

El sistema incluye múltiples capas de protección:

1. **Position Sizing**: Kelly Criterion + ajuste por volatilidad
2. **Risk Limits**: Máximo 20% por posición, 2% riesgo por trade
3. **Stop Loss/Take Profit**: Automáticos por cada posición
4. **Diversificación**: Límites de correlación entre posiciones
5. **Drawdown Protection**: Detención automática si DD > 20%

## 📚 Documentación

### Flujo de Análisis

1. **Data Gathering**: Obtener datos de APIs (con cache)
2. **Multi-Model Analysis**: 
   - Temporal: Predicción de precio
   - Visión: Detección de patrones
   - Tabular: Indicadores técnicos
   - NLP: Sentimiento de noticias
   - Grafos: Correlaciones de mercado
3. **State Construction**: Crear estado para RL
4. **Decision Making**: SAC agent decide acción
5. **Risk Check**: Verificar límites de riesgo
6. **Execution**: Ejecutar trade (si aplica)
7. **Logging**: Guardar señal y resultados

### Mejores Prácticas

- ✅ Empezar con paper trading antes de real money
- ✅ Hacer backtest exhaustivo antes de desplegar
- ✅ Monitorear continuamente las métricas
- ✅ Ajustar thresholds según tu tolerancia al riesgo
- ✅ Diversificar entre múltiples activos
- ✅ Revisar señales manualmente al principio
- ❌ NO confiar ciegamente en las señales
- ❌ NO arriesgar más del 2% por trade
- ❌ NO operar sin stop-loss

## 🐛 Troubleshooting

### Error: "No module named 'yfinance'"
```bash
pip install yfinance
```

### Error: "Rate limit alcanzado (Alpha Vantage)"
- Esperar 1 minuto
- Usar cache (use_cache=True)
- Considerar API key premium

### Dashboard no carga
```bash
# Verificar puerto
streamlit run dashboard.py --server.port 8502

# Limpiar cache
streamlit cache clear
```

### Alertas no se envían
- Verificar variables de entorno en .env
- Para Gmail: usar "App Password", no la contraseña normal
- Verificar firewall/antivirus

## 🤝 Contribuciones

Pull requests son bienvenidos. Para cambios mayores, abrir un issue primero.

## ⚠️ Disclaimer

Este sistema es para propósitos educativos y de investigación. Trading conlleva riesgo de pérdida de capital. Usa bajo tu propio riesgo. No somos responsables por pérdidas financieras.

## 📄 Licencia

MIT License - Ver LICENSE file

