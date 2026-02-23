
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')
import requests
import time
import json
import os
import sys
import re
from datetime import datetime, timedelta
from pathlib import Path

try:
    from black_scholes_engine import OptionsAnalyzer
except Exception:
    OptionsAnalyzer = None


def configure_console_output():
    """Evitar errores de encoding al imprimir unicode en Windows."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding='utf-8', errors='replace')
            except Exception:
                pass


configure_console_output()

# ============================================================================
# CONFIGURACIÓN GLOBAL
# ============================================================================

class Config:
    """Configuración centralizada del sistema"""

    # APIs
    ALPHA_VANTAGE_KEY = "QZF8CB4TECMS754I"
    ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
    ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")

    # Trading
    DEFAULT_CAPITAL = 100000
    MAX_POSITION_SIZE = 0.20  # 20% del capital por posición
    RISK_PER_TRADE = 0.02  # 2% de riesgo por trade

    # Backtesting
    BACKTEST_PERIOD_DAYS = 252  # 1 año
    COMMISSION_RATE = 0.001  # 0.1% comisión

    # Alertas
    ALERT_CONFIDENCE_THRESHOLD = 0.85
    ALERT_EMAIL = os.getenv("ALERT_EMAIL", "")

    # Data storage
    DATA_DIR = Path("data")
    LOGS_DIR = Path("logs")
    MODELS_DIR = Path("models")

    @classmethod
    def setup_directories(cls):
        """Crear directorios necesarios"""
        for dir_path in [cls.DATA_DIR, cls.LOGS_DIR, cls.MODELS_DIR]:
            dir_path.mkdir(exist_ok=True)

Config.setup_directories()

# ============================================================================
# API DE MERCADO MEJORADA
# ============================================================================

class MarketDataAPI:
    """Gestor avanzado de APIs con caching"""

    def __init__(self, alpha_vantage_key: str = None):
        self.av_key = alpha_vantage_key or Config.ALPHA_VANTAGE_KEY
        self.av_base_url = "https://www.alphavantage.co/query"
        self.cache_dir = Config.DATA_DIR / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.stock_cache_ttl_seconds = 3600
        self.news_cache_ttl_seconds = 1800
        self.yahoo_enabled = True

    def _read_cached_stock_data(self, cache_file: Path) -> Optional[pd.DataFrame]:
        """Leer cache de mercado desde CSV (sin dependencias extra)."""
        try:
            df = pd.read_csv(cache_file, parse_dates=['datetime'])
            if df.empty or 'close' not in df.columns:
                return None
            return df
        except Exception:
            return None

    def _write_cached_stock_data(self, cache_file: Path, data: pd.DataFrame):
        """Persistir cache de mercado de forma segura."""
        try:
            data.to_csv(cache_file, index=False)
        except Exception as e:
            print(f"  ⚠ No se pudo guardar cache: {str(e)[:50]}")

    def get_stock_data(self, symbol: str, period: str = "3mo", use_cache: bool = True) -> pd.DataFrame:
        """Obtener datos con sistema de cache"""
        cache_file = self.cache_dir / f"{symbol}_{period}.csv"

        # Verificar cache
        if use_cache and cache_file.exists():
            cache_age = time.time() - cache_file.stat().st_mtime
            if cache_age < self.stock_cache_ttl_seconds:
                cached_df = self._read_cached_stock_data(cache_file)
                if cached_df is not None:
                    print(f"  ✓ Usando datos en cache ({cache_age/60:.0f}m antiguos)")
                    return cached_df

        print(f"\n📡 Descargando datos de {symbol}...")

        # Intentar Yahoo Finance
        if self.yahoo_enabled:
            try:
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period)

                if not df.empty:
                    df.columns = [col.lower() for col in df.columns]
                    df = df.reset_index()
                    df.rename(columns={'date': 'datetime'}, inplace=True)

                    # Guardar en cache
                    self._write_cached_stock_data(cache_file, df)
                    print(f"  ✓ Yahoo Finance: {len(df)} registros")
                    return df
            except Exception as e:
                error_text = str(e)
                print(f"  ⚠ Yahoo Finance: {error_text[:50]}")
                if "curl: (77)" in error_text or "certificate" in error_text.lower():
                    self.yahoo_enabled = False
                    print("  ⚠ Yahoo Finance deshabilitado por error de certificados")

        # Fallback a Alpha Vantage
        try:
            df = self._get_alpha_vantage(symbol)
            if df is not None and len(df) > 0:
                self._write_cached_stock_data(cache_file, df)
                print(f"  ✓ Alpha Vantage: {len(df)} registros")
                return df
        except Exception as e:
            print(f"  ⚠ Alpha Vantage: {str(e)[:50]}")

        # Datos sintéticos como último recurso
        print(f"  ⚠ Usando datos sintéticos")
        df = self._generate_synthetic_data(symbol)
        self._write_cached_stock_data(cache_file, df)
        return df

    def _get_alpha_vantage(self, symbol: str) -> pd.DataFrame:
        """Alpha Vantage con manejo de rate limits"""
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'apikey': self.av_key,
            'outputsize': 'full'
        }

        response = requests.get(self.av_base_url, params=params, timeout=10)
        data = response.json()

        if 'Note' in data:
            raise Exception("Rate limit alcanzado")

        if 'Time Series (Daily)' not in data:
            raise Exception("Sin datos")

        time_series = data['Time Series (Daily)']
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        df.index = pd.to_datetime(df.index)
        df = df.astype(float)
        df = df.sort_index()
        df = df.reset_index()
        df.rename(columns={'index': 'datetime'}, inplace=True)

        return df

    def _generate_synthetic_data(self, symbol: str, days: int = 252) -> pd.DataFrame:
        """GBM mejorado para datos sintéticos"""
        np.random.seed(hash(symbol) % (2**32))

        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

        # Parámetros realistas según el símbolo
        S0 = 100
        mu = 0.0005
        sigma = 0.02

        if symbol in ['TSLA', 'NVDA']:
            sigma = 0.04  # Más volatilidad para tech

        returns = np.random.normal(mu, sigma, days)
        price = S0 * np.exp(np.cumsum(returns))

        df = pd.DataFrame({
            'datetime': dates,
            'open': price * np.random.uniform(0.995, 1.005, days),
            'high': price * np.random.uniform(1.005, 1.03, days),
            'low': price * np.random.uniform(0.97, 0.995, days),
            'close': price,
            'volume': np.random.randint(1000000, 20000000, days)
        })

        return df

    def get_news_sentiment(self, symbol: str, use_cache: bool = True) -> List[str]:
        """Noticias con cache"""
        cache_file = self.cache_dir / f"news_{symbol}.json"

        if use_cache and cache_file.exists():
            cache_age = time.time() - cache_file.stat().st_mtime
            if cache_age < self.news_cache_ttl_seconds:
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cached_news = json.load(f)
                    if isinstance(cached_news, list):
                        return cached_news
                except Exception:
                    pass

        try:
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': symbol,
                'apikey': self.av_key,
                'limit': 10
            }

            response = requests.get(self.av_base_url, params=params, timeout=10)
            data = response.json()

            if 'feed' in data:
                news = []
                for item in data['feed'][:5]:
                    title = item.get('title', '')
                    summary = item.get('summary', '')
                    news.append(f"{title}. {summary}"[:200])

                if news:
                    with open(cache_file, 'w', encoding='utf-8') as f:
                        json.dump(news, f)
                    return news
        except:
            pass

        return self._get_default_news()

    def _get_default_news(self) -> List[str]:
        return [
            "Markets show positive momentum with tech sector leading gains",
            "Federal Reserve signals potential rate adjustments ahead",
            "Strong earnings reports boost investor confidence",
            "Global markets react to economic data releases",
            "Trading volumes remain elevated amid market volatility"
        ]

    def get_realtime_quote(self, symbol: str) -> Dict:
        """Quote en tiempo real"""
        if not self.yahoo_enabled:
            return {'symbol': symbol, 'price': 0, 'volume': 0}

        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = ticker.info

            return {
                'symbol': symbol,
                'price': info.get('currentPrice', info.get('regularMarketPrice', 0)),
                'volume': info.get('volume', 0),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'beta': info.get('beta', 1.0)
            }
        except Exception as e:
            error_text = str(e)
            if "curl: (77)" in error_text or "certificate" in error_text.lower():
                self.yahoo_enabled = False
            return {'symbol': symbol, 'price': 0, 'volume': 0}

# ============================================================================
# UTILIDADES
# ============================================================================

@dataclass
class ModelOutput:
    prediction: np.ndarray
    confidence: float
    metadata: Dict[str, Any]
    model_name: str

@dataclass
class MarketState:
    price_features: np.ndarray
    technical_indicators: np.ndarray
    sentiment_score: float
    volatility: float
    graph_embeddings: np.ndarray
    time_features: np.ndarray

@dataclass
class TradeSignal:
    """Señal de trading completa"""
    symbol: str
    action: str  # BUY, SELL, HOLD
    price: float
    position_size: float
    confidence: float
    stop_loss: float
    take_profit: float
    timestamp: datetime
    reasoning: str

    def to_dict(self):
        return asdict(self)

def extract_numeric_data(df: pd.DataFrame) -> np.ndarray:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    return df[numeric_cols].values

def get_price_column(df: pd.DataFrame) -> np.ndarray:
    for col in ['close', 'Close', 'price', 'Price']:
        if col in df.columns:
            return df[col].values

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        return df[numeric_cols[0]].values

    raise ValueError("No se encontraron columnas de precio")

# ============================================================================
# MÃ“DULOS AI BASE (CONECTADOS)
# ============================================================================

class TemporalModule:
    """SeÃ±al temporal ligera basada en drift/momentum de retornos."""

    def __init__(self, horizon: int = 10):
        self.horizon = horizon
        self.is_mock = False

    def predict(self, price_data: pd.DataFrame) -> ModelOutput:
        price = pd.Series(get_price_column(price_data)).astype(float).dropna()
        if len(price) < 5:
            pred = np.zeros(self.horizon, dtype=float)
            return ModelOutput(pred, 0.50, {'reason': 'insufficient_data'}, 'temporal')

        returns = price.pct_change().dropna()
        recent_ret = returns.tail(min(30, len(returns)))
        ret_mu = float(recent_ret.mean()) if len(recent_ret) > 0 else 0.0
        ret_sigma = float(recent_ret.std()) if len(recent_ret) > 0 else 0.01

        k = min(40, len(price))
        x = np.arange(k, dtype=float)
        slope = float(np.polyfit(x, price.iloc[-k:].values, 1)[0] / (price.iloc[-1] + 1e-12))

        lookback = min(15, len(price) - 1)
        momentum = float((price.iloc[-1] / price.iloc[-1 - lookback]) - 1.0) if lookback > 0 else ret_mu
        momentum_rate = momentum / max(lookback, 1)

        drift = float(np.clip(0.45 * ret_mu + 0.35 * slope + 0.20 * momentum_rate, -0.03, 0.03))
        forecast_returns = drift + np.linspace(0.0, drift * 0.5, self.horizon)
        prediction = forecast_returns.astype(float)

        conf = 0.55 + min(0.35, abs(drift) * 8.0 + max(0.0, 0.10 - ret_sigma) * 0.8)
        confidence = float(np.clip(conf, 0.50, 0.95))

        metadata = {
            'drift': drift,
            'ret_mu': ret_mu,
            'ret_sigma': ret_sigma,
            'momentum': momentum,
        }
        return ModelOutput(prediction, confidence, metadata, 'temporal')


class VisionModule:
    """DetecciÃ³n de patrones simple sobre la serie de precios."""

    def __init__(self):
        self.is_mock = False

    def analyze(self, price_data: pd.DataFrame) -> ModelOutput:
        price = pd.Series(get_price_column(price_data)).astype(float).dropna()
        if len(price) < 25:
            return ModelOutput(np.array([0.0]), 0.50, {'detected_patterns': ['insufficient_data']}, 'vision')

        r = price.pct_change().fillna(0.0)
        sma_fast = float(price.rolling(8).mean().iloc[-1])
        sma_slow = float(price.rolling(21).mean().iloc[-1])
        breakout_ref = float(price.tail(20).iloc[:-1].max()) if len(price) >= 21 else float(price.max())
        breakout = float((price.iloc[-1] / (breakout_ref + 1e-12)) - 1.0)

        up_streak = int((r.tail(5) > 0).sum())
        down_streak = int((r.tail(5) < 0).sum())

        score = 0.0
        patterns = []

        if sma_fast > sma_slow:
            score += 1.0
            patterns.append('sma_bull_cross')
        else:
            score -= 1.0
            patterns.append('sma_bear_cross')

        if breakout > 0.01:
            score += 1.0
            patterns.append('breakout_up')
        elif breakout < -0.01:
            score -= 1.0
            patterns.append('breakout_down')

        if up_streak >= 4:
            score += 0.5
            patterns.append('momentum_up')
        if down_streak >= 4:
            score -= 0.5
            patterns.append('momentum_down')

        norm_score = float(np.clip(score / 3.0, -1.0, 1.0))
        conf = 0.55 + 0.35 * min(1.0, abs(norm_score))
        confidence = float(np.clip(conf, 0.50, 0.95))

        metadata = {
            'detected_patterns': patterns if patterns else ['none'],
            'pattern_score': norm_score,
            'sma_fast': sma_fast,
            'sma_slow': sma_slow,
        }
        return ModelOutput(np.array([norm_score]), confidence, metadata, 'vision')


class TabularModule:
    """Scoring tabular ligero sobre indicadores tÃ©cnicos."""

    def __init__(self):
        self.is_mock = False

    def predict(self, features_df: pd.DataFrame) -> ModelOutput:
        row = features_df.iloc[0] if not features_df.empty else pd.Series(dtype=float)
        rsi = float(row.get('rsi', 50.0))
        macd = float(row.get('macd', 0.0))
        returns = float(row.get('returns', 0.0))
        vol = float(row.get('volatility', 0.01))
        bb_pos = float(row.get('bb_position', 0.5))

        rsi_score = float(np.clip((50.0 - rsi) / 25.0, -1.0, 1.0))
        macd_score = float(np.tanh(macd * 5.0))
        ret_score = float(np.tanh(returns * 20.0))
        bb_score = float(np.clip((0.5 - bb_pos) * 2.0, -1.0, 1.0))

        raw = 0.30 * rsi_score + 0.30 * macd_score + 0.20 * ret_score + 0.20 * bb_score
        raw = float(np.clip(raw, -1.0, 1.0))

        conf = 0.55 + 0.30 * abs(raw) + 0.20 * max(0.0, 0.08 - vol)
        confidence = float(np.clip(conf, 0.50, 0.95))

        metadata = {
            'feature_scores': {
                'rsi': rsi_score,
                'macd': macd_score,
                'returns': ret_score,
                'bb_position': bb_score,
            },
            'volatility': vol,
            'raw_score': raw,
        }
        return ModelOutput(np.array([raw]), confidence, metadata, 'tabular')


class NLPModule:
    """Sentimiento basado en lÃ©xico financiero ligero."""

    POSITIVE_WORDS = {
        'beat', 'growth', 'strong', 'up', 'surge', 'bullish', 'optimistic',
        'record', 'profit', 'expansion', 'upgrade', 'outperform', 'momentum'
    }
    NEGATIVE_WORDS = {
        'miss', 'weak', 'down', 'drop', 'bearish', 'risk', 'uncertain',
        'loss', 'downgrade', 'recession', 'volatility', 'lawsuit', 'decline'
    }

    def __init__(self):
        self.is_mock = False

    def analyze(self, news_items: List[str]) -> ModelOutput:
        if not news_items:
            return ModelOutput(np.array([0.0]), 0.50, {'coverage': 0.0}, 'nlp')

        scores = []
        covered = 0
        for item in news_items:
            tokens = re.findall(r"[a-zA-Z]+", str(item).lower())
            pos = sum(1 for t in tokens if t in self.POSITIVE_WORDS)
            neg = sum(1 for t in tokens if t in self.NEGATIVE_WORDS)
            tot = pos + neg
            if tot > 0:
                covered += 1
                scores.append((pos - neg) / tot)
            else:
                scores.append(0.0)

        sentiment = float(np.mean(scores)) if scores else 0.0
        coverage = covered / max(1, len(news_items))
        conf = 0.50 + 0.25 * coverage + 0.20 * min(1.0, abs(sentiment))
        confidence = float(np.clip(conf, 0.50, 0.95))

        metadata = {
            'coverage': coverage,
            'num_news': len(news_items),
        }
        return ModelOutput(np.array([sentiment]), confidence, metadata, 'nlp')


class GraphModule:
    """SeÃ±al relacional ligera basada en estructura de red simulada."""

    def __init__(self):
        self.is_mock = False

    def analyze(self, adjacency: np.ndarray, features: np.ndarray) -> ModelOutput:
        adj = np.asarray(adjacency, dtype=float)
        feat = np.asarray(features, dtype=float)
        if adj.ndim != 2 or feat.ndim != 2:
            return ModelOutput(np.zeros(10), 0.50, {'reason': 'invalid_input'}, 'graph')

        degree = adj.mean(axis=1)
        centrality = degree / (degree.sum() + 1e-12)
        dispersion = float(np.std(centrality))
        avg_degree = float(np.mean(degree))

        emb = feat.mean(axis=0).astype(float)
        if len(emb) < 10:
            emb = np.pad(emb, (0, 10 - len(emb)))
        else:
            emb = emb[:10]

        conf = 0.55 + 0.25 * max(0.0, 1.0 - min(1.0, dispersion * 8.0))
        confidence = float(np.clip(conf, 0.50, 0.95))
        metadata = {'avg_degree': avg_degree, 'dispersion': dispersion}
        return ModelOutput(emb, confidence, metadata, 'graph')


class SACAgent:
    """Policy ligera para combinar seÃ±ales multimodales."""

    def __init__(self, state_dim: int = 128, action_dim: int = 3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.is_mock = False

    def _construct_state_vector(self, state: MarketState) -> np.ndarray:
        vec = np.concatenate([
            np.asarray(state.price_features).flatten(),
            np.asarray(state.technical_indicators).flatten(),
            np.asarray([state.sentiment_score, state.volatility]),
            np.asarray(state.graph_embeddings).flatten(),
            np.asarray(state.time_features).flatten(),
        ])
        if len(vec) < self.state_dim:
            vec = np.pad(vec, (0, self.state_dim - len(vec)))
        elif len(vec) > self.state_dim:
            vec = vec[:self.state_dim]
        return vec.astype(float)

    def get_trading_decision(self, state: MarketState) -> Dict[str, float]:
        price_signal = float(np.tanh(np.mean(state.price_features) * 20.0))
        tech_signal = float(np.tanh(np.mean(state.technical_indicators)))
        sentiment_signal = float(np.clip(state.sentiment_score, -1.0, 1.0))
        graph_signal = float(np.tanh(np.mean(state.graph_embeddings)))
        vol = float(np.clip(state.volatility, 0.001, 1.0))

        combined = (
            0.38 * price_signal
            + 0.27 * tech_signal
            + 0.20 * sentiment_signal
            + 0.15 * graph_signal
        )

        if combined > 0.15:
            action = 'BUY'
        elif combined < -0.15:
            action = 'SELL'
        else:
            action = 'HOLD'

        confidence = float(np.clip(0.58 + 0.35 * abs(combined) - 0.10 * vol, 0.50, 0.95))
        position_size = float(np.clip(0.03 + 0.17 * abs(combined), 0.01, Config.MAX_POSITION_SIZE))
        stop_loss = -float(np.clip(0.01 + vol * 0.04, 0.01, 0.08))
        take_profit = float(np.clip(abs(stop_loss) * 2.2, 0.02, 0.18))

        return {
            'action': action,
            'position_size': position_size,
            'confidence': confidence,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'policy_score': combined,
        }


class TimeGANSimulator:
    """Simulador de escenarios (proxy liviano de TimeGAN)."""

    def __init__(self):
        self.is_mock = False

    def generate_scenarios(self, n_scenarios: int = 10, horizon: int = 10) -> np.ndarray:
        drift = 0.0002
        vol = 0.01
        return np.random.normal(drift, vol, size=(n_scenarios, horizon))

# ============================================================================
# POSITION SIZER - Kelly Criterion
# ============================================================================

class PositionSizer:
    """Position sizing con Kelly Criterion"""

    @staticmethod
    def kelly_criterion(win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Kelly Criterion clásico"""
        if avg_loss == 0:
            return 0

        win_loss_ratio = avg_win / abs(avg_loss)
        kelly = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio

        # Half Kelly para ser conservador
        return max(0, min(kelly * 0.5, Config.MAX_POSITION_SIZE))

    @staticmethod
    def volatility_adjusted_size(confidence: float, volatility: float, capital: float) -> float:
        """Tamaño ajustado por volatilidad"""
        base_size = confidence * Config.MAX_POSITION_SIZE

        # Ajustar por volatilidad
        vol_adjustment = 1 / (1 + volatility * 10)

        adjusted_size = base_size * vol_adjustment

        return min(adjusted_size, Config.MAX_POSITION_SIZE)

# ============================================================================
# RISK MANAGER
# ============================================================================

class RiskManager:
    """Gestión de riesgo avanzada"""

    def __init__(self, capital: float = Config.DEFAULT_CAPITAL):
        self.capital = capital
        self.max_drawdown = 0.20  # 20%
        self.max_correlation = 0.70  # Máx correlación entre posiciones
        self.positions = {}

    def check_risk_limits(self, signal: TradeSignal) -> Tuple[bool, str]:
        """Verificar límites de riesgo"""

        # 1. Verificar capital disponible
        if signal.position_size * signal.price > self.capital * Config.MAX_POSITION_SIZE:
            return False, "Excede límite de capital por posición"

        # 2. Verificar drawdown
        current_dd = self.calculate_drawdown()
        if current_dd > self.max_drawdown:
            return False, f"Drawdown {current_dd:.1%} excede límite {self.max_drawdown:.1%}"

        # 3. Verificar confianza mínima
        if signal.confidence < 0.70:
            return False, f"Confianza {signal.confidence:.1%} por debajo del mínimo"

        # 4. Stop loss razonable
        if signal.action == 'BUY' and signal.stop_loss < -0.10:
            return False, "Stop loss demasiado amplio"

        return True, "OK"

    def calculate_drawdown(self) -> float:
        """Calcular drawdown actual"""
        if not self.positions:
            return 0.0

        total_value = sum(pos['value'] for pos in self.positions.values())
        peak_value = max(self.capital, total_value)

        return (peak_value - total_value) / peak_value

    def update_position(self, symbol: str, quantity: float, price: float):
        """Actualizar posición"""
        self.positions[symbol] = {
            'quantity': quantity,
            'entry_price': price,
            'value': quantity * price
        }

# ============================================================================
# PAPER TRADING ENGINE
# ============================================================================

class PaperTradingEngine:
    """Motor de paper trading con Alpaca"""

    def __init__(self, api_key: str = None, secret_key: str = None):
        self.api_key = api_key or Config.ALPACA_API_KEY
        self.secret_key = secret_key or Config.ALPACA_SECRET_KEY
        self.use_alpaca = bool(self.api_key and self.secret_key)

        self.portfolio = {
            'cash': Config.DEFAULT_CAPITAL,
            'positions': {},
            'trades_history': []
        }

        if self.use_alpaca:
            try:
                from alpaca.trading.client import TradingClient
                from alpaca.trading.requests import MarketOrderRequest
                from alpaca.trading.enums import OrderSide, TimeInForce

                self.client = TradingClient(self.api_key, self.secret_key, paper=True)
                self.MarketOrderRequest = MarketOrderRequest
                self.OrderSide = OrderSide
                self.TimeInForce = TimeInForce

                print("  ✓ Alpaca Paper Trading conectado")
            except Exception as e:
                print(f"  ⚠ Alpaca no disponible: {str(e)[:50]}")
                self.use_alpaca = False

    def execute_signal(self, signal: TradeSignal) -> Dict:
        """Ejecutar señal de trading"""

        if self.use_alpaca:
            return self._execute_alpaca(signal)
        else:
            return self._execute_simulated(signal)

    def _execute_alpaca(self, signal: TradeSignal) -> Dict:
        """Ejecutar en Alpaca paper trading"""
        try:
            side = self.OrderSide.BUY if signal.action == 'BUY' else self.OrderSide.SELL

            order_data = self.MarketOrderRequest(
                symbol=signal.symbol,
                qty=signal.position_size,
                side=side,
                time_in_force=self.TimeInForce.DAY
            )

            order = self.client.submit_order(order_data)

            return {
                'success': True,
                'order_id': order.id,
                'symbol': signal.symbol,
                'side': signal.action,
                'quantity': signal.position_size,
                'status': 'submitted'
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def _execute_simulated(self, signal: TradeSignal) -> Dict:
        """Ejecutar en simulación local"""

        cost = signal.position_size * signal.price

        if signal.action == 'BUY':
            if self.portfolio['cash'] < cost:
                return {'success': False, 'error': 'Fondos insuficientes'}

            self.portfolio['cash'] -= cost

            if signal.symbol in self.portfolio['positions']:
                pos = self.portfolio['positions'][signal.symbol]
                old_quantity = pos['quantity']
                new_quantity = old_quantity + signal.position_size
                pos['avg_price'] = (
                    (pos['avg_price'] * old_quantity) + (signal.price * signal.position_size)
                ) / new_quantity
                pos['quantity'] = new_quantity
            else:
                self.portfolio['positions'][signal.symbol] = {
                    'quantity': signal.position_size,
                    'avg_price': signal.price,
                    'stop_loss': signal.stop_loss,
                    'take_profit': signal.take_profit
                }

        elif signal.action == 'SELL':
            if signal.symbol not in self.portfolio['positions']:
                return {'success': False, 'error': 'No hay posición para vender'}

            pos = self.portfolio['positions'][signal.symbol]

            if pos['quantity'] < signal.position_size:
                return {'success': False, 'error': 'Cantidad insuficiente'}

            self.portfolio['cash'] += signal.position_size * signal.price
            pos['quantity'] -= signal.position_size

            if pos['quantity'] <= 1e-10:
                del self.portfolio['positions'][signal.symbol]

        # Registrar trade
        trade = {
            'timestamp': signal.timestamp,
            'symbol': signal.symbol,
            'action': signal.action,
            'quantity': signal.position_size,
            'price': signal.price,
            'confidence': signal.confidence
        }

        self.portfolio['trades_history'].append(trade)

        return {
            'success': True,
            'trade': trade,
            'portfolio_value': self.get_portfolio_value()
        }

    def get_portfolio_value(self, current_prices: Dict = None) -> float:
        """Calcular valor del portfolio"""
        total = self.portfolio['cash']

        for symbol, pos in self.portfolio['positions'].items():
            price = current_prices.get(symbol, pos['avg_price']) if current_prices else pos['avg_price']
            total += pos['quantity'] * price

        return total

    def get_portfolio_summary(self) -> Dict:
        """Resumen del portfolio"""
        return {
            'cash': self.portfolio['cash'],
            'positions': len(self.portfolio['positions']),
            'total_value': self.get_portfolio_value(),
            'num_trades': len(self.portfolio['trades_history']),
            'positions_detail': self.portfolio['positions']
        }

# ============================================================================
# SISTEMA PRINCIPAL CON TODAS LAS FUNCIONALIDADES
# ============================================================================

class GodModeComplete:
    """Sistema God Mode COMPLETO"""

    def __init__(self, alpha_vantage_key: str = None):
        print("\n🚀 Inicializando GOD MODE COMPLETE...")

        self.market_api = MarketDataAPI(alpha_vantage_key)
        self.position_sizer = PositionSizer()
        self.risk_manager = RiskManager()
        self.paper_trading = PaperTradingEngine()
        self.options_analyzer = OptionsAnalyzer() if OptionsAnalyzer else None

        # Importar módulos
        try:
            self.temporal_module = TemporalModule()
            self.vision_module = VisionModule()
            self.tabular_module = TabularModule()
            self.nlp_module = NLPModule()
            self.graph_module = GraphModule()
            self.sac_agent = SACAgent(state_dim=128, action_dim=3)
            self.timegan = TimeGANSimulator()

            print("  ✓ Módulos AI conectados")
        except Exception as e:
            print(f"  ⚠ Fallback a módulos simulados: {str(e)[:80]}")
            self._init_mock_modules()

        self.model_status = self._collect_model_status()
        status_line = ", ".join(f"{k}:{v}" for k, v in self.model_status.items())
        print(f"  ✓ Estado de conectividad -> {status_line}")

        # Historial de señales
        self.signals_history = []

        if self.options_analyzer is not None:
            print("  ✓ Black-Scholes engine cargado")
        else:
            print("  ⚠ Black-Scholes engine no disponible")

        print("  ✓ Sistema listo\n")

    def _init_mock_modules(self):
        """Módulos mock para testing"""
        class MockModule:
            is_mock = True
            def predict(self, data):
                return ModelOutput(np.random.randn(10), 0.75, {}, 'mock')
            def analyze(self, *args, **kwargs):
                return ModelOutput(np.random.randn(10), 0.75, {'detected_patterns': ['mock']}, 'mock')
            def forward(self, *args):
                return np.random.randn(10, 20)

        class MockSAC:
            is_mock = True
            def _construct_state_vector(self, state):
                return np.random.randn(128)
            def get_trading_decision(self, state):
                return {
                    'action': np.random.choice(['BUY', 'SELL', 'HOLD']),
                    'position_size': np.random.rand(),
                    'confidence': np.random.rand(),
                    'stop_loss': -0.02,
                    'take_profit': 0.05
                }

        class MockTimeGAN:
            is_mock = True
            def generate_scenarios(self, n_scenarios=10, horizon=10):
                return np.random.randn(n_scenarios, horizon)

        self.temporal_module = MockModule()
        self.vision_module = MockModule()
        self.tabular_module = MockModule()
        self.nlp_module = MockModule()
        self.graph_module = MockModule()
        self.sac_agent = MockSAC()
        self.timegan = MockTimeGAN()

    def _collect_model_status(self) -> Dict[str, str]:
        """Estado de conectividad de todos los módulos para decisiones."""
        modules = {
            'temporal': self.temporal_module,
            'vision': self.vision_module,
            'tabular': self.tabular_module,
            'nlp': self.nlp_module,
            'graph': self.graph_module,
            'sac': self.sac_agent,
            'timegan': self.timegan,
            'options': self.options_analyzer,
        }
        out = {}
        for name, obj in modules.items():
            if obj is None:
                out[name] = 'DISCONNECTED'
            elif getattr(obj, 'is_mock', False):
                out[name] = 'MOCK'
            else:
                out[name] = 'CONNECTED'
        return out

    def _estimate_annualized_volatility(self, price_data: pd.DataFrame) -> float:
        """Estimacion robusta de volatilidad anualizada."""
        try:
            close_prices = pd.Series(get_price_column(price_data)).astype(float)
            returns = close_prices.pct_change().dropna()
            if returns.empty:
                return 0.20
            annual_vol = float(returns.std() * np.sqrt(252))
            return float(np.clip(annual_vol, 0.05, 2.0))
        except Exception:
            return 0.20

    @staticmethod
    def _blend_confidence(base_confidence: float, options_confidence: float) -> float:
        blended = np.mean([base_confidence, options_confidence])
        return float(np.clip(blended, 0.0, 0.99))

    def _analyze_options_overlay(self, symbol: str, spot_price: float, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Ejecutar analisis de opciones Black-Scholes para enriquecer la señal."""
        if self.options_analyzer is None:
            return {
                'available': False,
                'directional_bias': 'NEUTRAL',
                'recommendation': 'NO_DATA',
                'signal_confidence': 0.5,
                'avg_implied_volatility': 0.0
            }

        try:
            annual_volatility = self._estimate_annualized_volatility(price_data)
            option_chain = self.options_analyzer.analyze_symbol_options(
                symbol=symbol,
                spot_price=float(spot_price),
                annualized_volatility=annual_volatility,
                days_to_expiry=30
            )
            summary = self.options_analyzer.summarize_option_chain(option_chain)
            summary['annualized_volatility_input'] = annual_volatility
            return summary
        except Exception as e:
            return {
                'available': False,
                'directional_bias': 'NEUTRAL',
                'recommendation': 'ERROR',
                'signal_confidence': 0.5,
                'avg_implied_volatility': 0.0,
                'error': str(e)[:120]
            }

    def analyze_symbol(self, symbol: str, period: str = "3mo", execute_trade: bool = False) -> Dict:
        """Análisis completo con opción de ejecutar trade"""

        print("\n" + "="*80)
        print(f"GOD MODE ANALYSIS - {symbol.upper()}")
        print("="*80)

        # Obtener datos
        price_data = self.market_api.get_stock_data(symbol, period)
        news_data = self.market_api.get_news_sentiment(symbol)
        quote = self.market_api.get_realtime_quote(symbol)

        quote_price = quote.get('price')
        latest_price = quote_price if quote_price and quote_price > 0 else price_data['close'].iloc[-1]
        price_change = ((latest_price / price_data['close'].iloc[0]) - 1) * 100

        print(f"\n📊 Información:")
        print(f"   Precio: ${latest_price:.2f}")
        print(f"   Cambio: {price_change:+.2f}%")
        print(f"   Volume: {quote.get('volume', 0):,}")
        print(f"   P/E: {quote.get('pe_ratio', 0):.2f}")

        # Análisis multi-modelo
        market_data = {'price_data': price_data, 'news': news_data}
        result = self._analyze_and_decide(market_data)

        # Overlay de opciones con Black-Scholes
        options_analysis = self._analyze_options_overlay(symbol.upper(), latest_price, price_data)
        result['options_analysis'] = options_analysis

        if options_analysis.get('available'):
            base_confidence = float(result['confidence_breakdown']['overall'])
            options_confidence = float(options_analysis.get('signal_confidence', 0.5))
            result['confidence_breakdown']['base_model'] = base_confidence
            result['confidence_breakdown']['options'] = options_confidence
            result['confidence_breakdown']['overall'] = self._blend_confidence(base_confidence, options_confidence)

            print("\n[OPTIONS] BLACK-SCHOLES:")
            print(f"   Bias: {options_analysis.get('directional_bias', 'NEUTRAL')}")
            print(f"   Recommendation: {options_analysis.get('recommendation', 'N/A')}")
            print(f"   Avg IV: {options_analysis.get('avg_implied_volatility', 0.0):.1%}")
            print(f"   Options confidence: {options_confidence:.1%}")

        self.model_status = self._collect_model_status()

        # Crear señal de trading
        decision = result['decision']
        confidence = result['confidence_breakdown']['overall']
        volatility = result['market_state'].volatility

        # Position sizing
        position_size = self.position_sizer.volatility_adjusted_size(
            confidence, volatility, self.risk_manager.capital
        )

        signal = TradeSignal(
            symbol=symbol.upper(),
            action=decision['action'],
            price=latest_price,
            position_size=position_size,
            confidence=confidence,
            stop_loss=decision['stop_loss'],
            take_profit=decision['take_profit'],
            timestamp=datetime.now(),
            reasoning=(
                f"Confianza: {confidence:.1%}, Vol: {volatility:.2%}, "
                f"Sentiment: {result['sentiment']:.2f}, "
                f"OptionsBias: {options_analysis.get('directional_bias', 'NEUTRAL')}"
            )
        )

        # Verificar riesgo
        risk_ok, risk_msg = self.risk_manager.check_risk_limits(signal)

        print(f"\n🎯 SEÑAL GENERADA:")
        print(f"   {self._get_action_emoji(signal.action)} Acción: {signal.action}")
        print(f"   Tamaño: {signal.position_size:.4f} unidades (${signal.position_size * signal.price:,.2f})")
        print(f"   Confianza: {signal.confidence:.1%}")
        print(f"   Stop Loss: {signal.stop_loss:.1%}")
        print(f"   Take Profit: {signal.take_profit:.1%}")
        print(f"   Risk Check: {'✅ PASS' if risk_ok else '❌ FAIL'} - {risk_msg}")

        # Ejecutar trade si está habilitado y pasa risk check
        if execute_trade and risk_ok and signal.action != 'HOLD':
            print(f"\n💼 Ejecutando trade...")
            trade_result = self.paper_trading.execute_signal(signal)

            if trade_result['success']:
                print(f"   ✅ Trade ejecutado exitosamente")
                self.risk_manager.update_position(symbol, signal.position_size, latest_price)
            else:
                print(f"   ❌ Error: {trade_result.get('error', 'Unknown')}")

            result['trade_result'] = trade_result

        # Guardar señal
        self.signals_history.append(signal)
        self._save_signal(signal)

        # Añadir info adicional
        result.update({
            'symbol': symbol.upper(),
            'current_price': float(latest_price),
            'price_change': float(price_change),
            'signal': signal,
            'risk_check': {'passed': risk_ok, 'message': risk_msg},
            'quote': quote,
            'model_status': self.model_status.copy(),
        })

        return result

    def _analyze_and_decide(self, market_data: Dict) -> Dict:
        """Pipeline de análisis multi-modelo"""

        print("\n" + "="*80)
        print("ANÁLISIS MULTI-MODELO")
        print("="*80)

        # 1. Temporal
        print("\n[1/7] Módulo Temporal...")
        temporal_output = self.temporal_module.predict(market_data['price_data'])
        print(f"  ✓ Predicción: ${temporal_output.prediction[0]:.2f} → ${temporal_output.prediction[-1]:.2f}")
        print(f"  ✓ Confianza: {temporal_output.confidence:.2%}")

        # 2. Visión
        print("\n[2/7] Módulo Visión...")
        vision_output = self.vision_module.analyze(market_data['price_data'])
        patterns = vision_output.metadata.get('detected_patterns', ['none'])
        print(f"  ✓ Patrones: {patterns}")
        print(f"  ✓ Confianza: {vision_output.confidence:.2%}")

        # 3. Tabular
        print("\n[3/7] Módulo Tabular...")
        tabular_features = self._prepare_tabular_features(market_data)
        tabular_output = self.tabular_module.predict(tabular_features)
        print(f"  ✓ Features: RSI={tabular_features['rsi'].iloc[0]:.1f}, Vol={tabular_features['volatility'].iloc[0]:.2%}")
        print(f"  ✓ Confianza: {tabular_output.confidence:.2%}")

        # 4. NLP
        print("\n[4/7] Módulo NLP...")
        nlp_output = self.nlp_module.analyze(market_data['news'])
        sentiment = nlp_output.prediction[0]
        print(f"  ✓ Sentimiento: {sentiment:.2f} ({'POSITIVO ✅' if sentiment > 0.5 else 'NEGATIVO ❌'})")
        print(f"  ✓ Confianza: {nlp_output.confidence:.2%}")

        # 5. Grafos
        print("\n[5/7] Módulo Grafos...")
        graph_data = self._prepare_graph_data(market_data)
        graph_output = self.graph_module.analyze(graph_data['adjacency'], graph_data['features'])
        print(f"  ✓ Red: {graph_data['adjacency'].shape[0]} nodos")
        print(f"  ✓ Confianza: {graph_output.confidence:.2%}")

        # 6. Market State
        print("\n[6/7] Market State...")
        market_state = self._construct_market_state(
            temporal_output, vision_output, tabular_output, nlp_output, graph_output
        )
        print(f"  ✓ State vector construido")

        # 7. Decisión SAC
        print("\n[7/7] Decisión SAC...")
        decision = self.sac_agent.get_trading_decision(market_state)
        print(f"  {self._get_action_emoji(decision['action'])} {decision['action']}")

        # 8. Scenarios TimeGAN
        scenarios = self.timegan.generate_scenarios(n_scenarios=10, horizon=10)

        confidence_overall = np.mean([
            temporal_output.confidence, vision_output.confidence,
            tabular_output.confidence, nlp_output.confidence, graph_output.confidence
        ])

        print(f"\n{'='*80}")
        print(f"CONFIANZA GENERAL: {confidence_overall:.1%}")
        print(f"{'='*80}")

        return {
            'decision': decision,
            'market_state': market_state,
            'module_outputs': {
                'temporal': temporal_output,
                'vision': vision_output,
                'tabular': tabular_output,
                'nlp': nlp_output,
                'graph': graph_output
            },
            'scenarios': scenarios,
            'confidence_breakdown': {
                'temporal': temporal_output.confidence,
                'vision': vision_output.confidence,
                'tabular': tabular_output.confidence,
                'nlp': nlp_output.confidence,
                'graph': graph_output.confidence,
                'overall': confidence_overall
            },
            'sentiment': float(sentiment)
        }

    def _prepare_tabular_features(self, market_data: Dict) -> pd.DataFrame:
        """Features técnicos avanzados"""
        df = market_data['price_data'].copy()
        price_col = get_price_column(df)

        features = {
            'returns': 0.0, 'volatility': 0.01,
            'sma_20': price_col[-1], 'sma_50': price_col[-1],
            'rsi': 50.0, 'macd': 0.0, 'bb_position': 0.5
        }

        if len(price_col) > 1:
            returns = pd.Series(price_col).pct_change().dropna()
            features['returns'] = returns.iloc[-1] if len(returns) > 0 else 0.0
            features['volatility'] = returns.std() if len(returns) > 0 else 0.01

        if len(price_col) >= 20:
            sma20 = pd.Series(price_col).rolling(20).mean()
            features['sma_20'] = sma20.iloc[-1]

            # Bollinger Bands
            bb_std = pd.Series(price_col).rolling(20).std()
            bb_upper = sma20 + 2 * bb_std
            bb_lower = sma20 - 2 * bb_std
            bb_position = (price_col[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
            features['bb_position'] = bb_position

        if len(price_col) >= 50:
            features['sma_50'] = pd.Series(price_col).rolling(50).mean().iloc[-1]

        # RSI
        if len(price_col) >= 14:
            delta = pd.Series(price_col).diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / (loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            features['rsi'] = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0

        # MACD
        if len(price_col) >= 26:
            ema12 = pd.Series(price_col).ewm(span=12).mean()
            ema26 = pd.Series(price_col).ewm(span=26).mean()
            macd = ema12 - ema26
            features['macd'] = macd.iloc[-1]

        return pd.DataFrame([features])

    def _prepare_graph_data(self, market_data: Dict) -> Dict:
        n_assets = 10
        price_df = market_data.get('price_data', pd.DataFrame())
        try:
            close = pd.Series(get_price_column(price_df)).astype(float).dropna()
            ret = close.pct_change().fillna(0.0).tail(240).values
        except Exception:
            ret = np.random.normal(0.0, 0.01, size=240)

        if len(ret) < 40:
            ret = np.pad(ret, (40 - len(ret), 0))

        feature_rows = []
        for i in range(n_assets):
            shifted = np.roll(ret, i + 1)
            w_short = shifted[-20:]
            w_mid = shifted[-60:]
            w_long = shifted[-120:]
            feat = np.array([
                np.mean(w_short), np.std(w_short),
                np.mean(w_mid), np.std(w_mid),
                np.mean(w_long), np.std(w_long),
                np.percentile(w_short, 25), np.percentile(w_short, 75),
                np.percentile(w_mid, 25), np.percentile(w_mid, 75),
                np.min(w_short), np.max(w_short),
                np.min(w_mid), np.max(w_mid),
                np.mean(np.abs(w_short)), np.mean(np.abs(w_mid)),
                np.mean(np.sign(w_short)), np.mean(np.sign(w_mid)),
                np.mean(shifted[-5:]), np.mean(shifted[-10:]),
            ], dtype=float)
            feature_rows.append(feat)

        features = np.vstack(feature_rows)

        # Adyacencia basada en similitud coseno entre nodos.
        norm = np.linalg.norm(features, axis=1, keepdims=True) + 1e-12
        f_norm = features / norm
        adjacency = np.clip(f_norm @ f_norm.T, -1.0, 1.0)
        adjacency = (adjacency + 1.0) / 2.0
        np.fill_diagonal(adjacency, 1.0)
        return {'adjacency': adjacency, 'features': features}

    def _construct_market_state(self, temporal, vision, tabular, nlp, graph) -> MarketState:
        price_features = temporal.prediction[:10]
        if len(price_features) < 10:
            price_features = np.pad(price_features, (0, 10 - len(price_features)))

        tab_pred = float(tabular.prediction[0]) if len(tabular.prediction) > 0 else 0.0
        temp_pred = float(np.mean(temporal.prediction[-3:])) if len(temporal.prediction) > 0 else 0.0
        vis_pred = float(np.mean(vision.prediction)) if len(vision.prediction) > 0 else 0.0
        technical = np.array([tab_pred, temp_pred, vis_pred], dtype=float)
        sentiment = float(nlp.prediction[0])
        base_vol = float(tabular.metadata.get('volatility', np.std(temporal.prediction)))
        volatility = float(np.clip(abs(base_vol), 0.005, 0.20))

        graph_emb = graph.prediction.flatten()[:10]
        if len(graph_emb) < 10:
            graph_emb = np.pad(graph_emb, (0, 10 - len(graph_emb)))

        now = datetime.now()
        time_features = np.array([now.hour / 24.0, now.weekday() / 7.0, now.day / 31.0])

        return MarketState(price_features, technical, sentiment, volatility, graph_emb, time_features)

    def _get_action_emoji(self, action: str) -> str:
        return {'BUY': '🟢', 'SELL': '🔴', 'HOLD': '🟡'}.get(action, '⚪')

    def _save_signal(self, signal: TradeSignal):
        """Guardar señal en archivo"""
        signals_file = Config.LOGS_DIR / "signals.jsonl"
        with open(signals_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(signal.to_dict(), default=str) + '\n')

    def backtest(self, symbol: str, start_date: str = None, end_date: str = None) -> Dict:
        """Backtest completo con métricas"""

        print(f"\n🔄 BACKTESTING {symbol}")
        print("="*80)

        # Obtener datos históricos
        price_data = self.market_api.get_stock_data(symbol, period="1y")

        if start_date:
            price_data = price_data[price_data['datetime'] >= start_date]
        if end_date:
            price_data = price_data[price_data['datetime'] <= end_date]

        if price_data.empty:
            raise ValueError("No hay datos disponibles para el rango seleccionado")

        if len(price_data) < 60:
            raise ValueError("Datos insuficientes para backtest (mínimo recomendado: 60 filas)")

        print(f"  Período: {price_data['datetime'].iloc[0]} a {price_data['datetime'].iloc[-1]}")
        print(f"  Datos: {len(price_data)} días\n")

        # Inicializar
        capital = Config.DEFAULT_CAPITAL
        position = 0
        trades = []
        equity_curve = [capital]

        # Simular día a día
        for i in range(50, len(price_data) - 1):
            window = price_data.iloc[:i+1]

            market_data = {
                'price_data': window,
                'news': self.market_api._get_default_news()
            }

            try:
                result = self._analyze_and_decide(market_data)
                decision = result['decision']['action']
                confidence = result['confidence_breakdown']['overall']
                price = window['close'].iloc[-1]

                # Trading logic
                if decision == 'BUY' and position == 0 and confidence > 0.75:
                    position = (capital * 0.95) / price  # 95% del capital
                    trades.append({
                        'date': window['datetime'].iloc[-1],
                        'type': 'BUY',
                        'price': price,
                        'quantity': position,
                        'confidence': confidence
                    })
                    print(f"  📈 {window['datetime'].iloc[-1].strftime('%Y-%m-%d')} BUY @ ${price:.2f}")

                elif decision == 'SELL' and position > 0:
                    capital = position * price * (1 - Config.COMMISSION_RATE)
                    trades.append({
                        'date': window['datetime'].iloc[-1],
                        'type': 'SELL',
                        'price': price,
                        'quantity': position,
                        'pnl': capital - Config.DEFAULT_CAPITAL
                    })
                    print(f"  📉 {window['datetime'].iloc[-1].strftime('%Y-%m-%d')} SELL @ ${price:.2f} | PnL: ${capital - Config.DEFAULT_CAPITAL:+,.2f}")
                    position = 0

                # Equity curve
                current_value = capital if position == 0 else position * price
                equity_curve.append(current_value)

            except Exception as e:
                print(f"  ⚠ Error día {i}: {str(e)[:50]}")
                equity_curve.append(equity_curve[-1])
                continue

        # Cerrar posición final si existe
        if position > 0:
            final_price = price_data['close'].iloc[-1]
            capital = position * final_price
            trades.append({
                'date': price_data['datetime'].iloc[-1],
                'type': 'SELL',
                'price': final_price,
                'quantity': position,
                'pnl': capital - Config.DEFAULT_CAPITAL
            })

        # Calcular métricas
        final_value = equity_curve[-1]
        total_return = (final_value / Config.DEFAULT_CAPITAL - 1) * 100

        equity_array = np.array(equity_curve)
        returns = np.diff(equity_array) / equity_array[:-1]
        sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)

        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / (running_max + 1e-10)
        max_dd = abs(np.min(drawdown)) * 100

        win_trades = [t for t in trades if t.get('pnl', 0) > 0]
        win_rate = len(win_trades) / len([t for t in trades if t['type'] == 'SELL']) if trades else 0

        # Resultados
        results = {
            'initial_capital': Config.DEFAULT_CAPITAL,
            'final_capital': final_value,
            'total_return_pct': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown_pct': max_dd,
            'num_trades': len(trades),
            'win_rate': win_rate * 100,
            'equity_curve': equity_curve,
            'trades': trades
        }

        print(f"\n{'='*80}")
        print("RESULTADOS DEL BACKTEST")
        print(f"{'='*80}")
        print(f"  Capital inicial:  ${results['initial_capital']:,.2f}")
        print(f"  Capital final:    ${results['final_capital']:,.2f}")
        print(f"  Retorno total:    {results['total_return_pct']:+.2f}%")
        print(f"  Sharpe Ratio:     {results['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown:     {results['max_drawdown_pct']:.2f}%")
        print(f"  Número de trades: {results['num_trades']}")
        print(f"  Win Rate:         {results['win_rate']:.1f}%")
        print(f"{'='*80}\n")

        # Guardar resultados
        results_file = Config.LOGS_DIR / f"backtest_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump({k: v for k, v in results.items() if k not in ['equity_curve', 'trades']}, f, indent=2, default=str)

        return results

    def scan_multiple(self, symbols: List[str], top_n: int = 5) -> pd.DataFrame:
        """Escanear múltiples símbolos y rankear"""

        print(f"\n🔍 ESCANEANDO {len(symbols)} SÍMBOLOS")
        print("="*80)

        results = []

        for symbol in symbols:
            try:
                result = self.analyze_symbol(symbol, period="1mo")

                results.append({
                    'Symbol': symbol,
                    'Price': result['current_price'],
                    'Change%': result['price_change'],
                    'Action': result['signal'].action,
                    'Confidence': result['confidence_breakdown']['overall'],
                    'Sentiment': result['sentiment'],
                    'Patterns': ', '.join(result['module_outputs']['vision'].metadata.get('detected_patterns', [])),
                    'Score': result['confidence_breakdown']['overall'] * (1 + abs(result['price_change'])/100)
                })

                time.sleep(1)

            except Exception as e:
                print(f"  ❌ Error en {symbol}: {str(e)[:50]}")
                continue

        df = pd.DataFrame(results)
        if df.empty:
            print("\n⚠ No se generaron resultados válidos")
            return df
        df = df.sort_values('Score', ascending=False)

        print(f"\n{'='*80}")
        print(f"TOP {top_n} OPORTUNIDADES")
        print(f"{'='*80}")
        print(df.head(top_n).to_string(index=False))
        print(f"{'='*80}\n")

        return df

    def get_portfolio_report(self) -> Dict:
        """Reporte del portfolio"""
        portfolio = self.paper_trading.get_portfolio_summary()

        print(f"\n📊 REPORTE DE PORTFOLIO")
        print("="*80)
        print(f"  Cash disponible:  ${portfolio['cash']:,.2f}")
        print(f"  Posiciones:       {portfolio['positions']}")
        print(f"  Valor total:      ${portfolio['total_value']:,.2f}")
        print(f"  Trades ejecutados: {portfolio['num_trades']}")

        if portfolio['positions_detail']:
            print(f"\n  Posiciones abiertas:")
            for symbol, pos in portfolio['positions_detail'].items():
                print(f"    {symbol}: {pos['quantity']:.4f} @ ${pos['avg_price']:.2f}")

        print("="*80)

        return portfolio

# ============================================================================
# MAIN DEMO
# ============================================================================

def main():
    """Demo completo del sistema"""

    print("\n" + "="*80)
    print(" "*20 + "GOD MODE TRADING SYSTEM COMPLETE")
    print(" "*15 + "Production Ready - All Features Included")
    print("="*80)

    # Inicializar
    god_mode = GodModeComplete()

    # 1. Análisis individual con ejecución de trade
    print("\n\n[DEMO 1] ANÁLISIS CON TRADE EXECUTION")
    result = god_mode.analyze_symbol('AAPL', period="3mo", execute_trade=True)

    # 2. Backtest
    print("\n\n[DEMO 2] BACKTESTING")
    backtest_results = god_mode.backtest('MSFT')

    # 3. Escaneo múltiple
    print("\n\n[DEMO 3] MULTI-SYMBOL SCAN")
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'META']
    scan_results = god_mode.scan_multiple(symbols, top_n=3)

    # 4. Reporte de portfolio
    print("\n\n[DEMO 4] PORTFOLIO REPORT")
    portfolio = god_mode.get_portfolio_report()

    print("\n\n✅ DEMO COMPLETADO")
    print("="*80)

    return god_mode

if __name__ == "__main__":
    system = main()
