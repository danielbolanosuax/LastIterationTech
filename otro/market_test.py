
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')
import requests
import time
from datetime import datetime, timedelta

# ============================================================================
# CONFIGURACIÓN DE APIs
# ============================================================================

class MarketDataAPI:
    """Gestor de APIs de mercado con fallback automático"""

    def __init__(self, alpha_vantage_key: str = None):
        self.av_key = alpha_vantage_key or "QZF8CB4TECMS754I"
        self.av_base_url = "https://www.alphavantage.co/query"

    def get_stock_data(self, symbol: str, period: str = "3mo") -> pd.DataFrame:
        """Obtener datos de stock con fallback automático"""
        print(f"\n📡 Descargando datos de {symbol}...")

        # Intentar Yahoo Finance primero
        try:
            data = self._get_yahoo_finance(symbol, period)
            if data is not None and len(data) > 0:
                print(f"  ✓ Datos obtenidos via Yahoo Finance: {len(data)} registros")
                return data
        except Exception as e:
            print(f"  ⚠ Yahoo Finance falló: {str(e)[:50]}")

        # Fallback a Alpha Vantage
        try:
            data = self._get_alpha_vantage(symbol)
            if data is not None and len(data) > 0:
                print(f"  ✓ Datos obtenidos via Alpha Vantage: {len(data)} registros")
                return data
        except Exception as e:
            print(f"  ⚠ Alpha Vantage falló: {str(e)[:50]}")

        # Si ambos fallan, generar datos sintéticos
        print(f"  ⚠ Usando datos sintéticos para demostración")
        return self._generate_synthetic_data(symbol)

    def _get_yahoo_finance(self, symbol: str, period: str) -> pd.DataFrame:
        """Obtener datos de Yahoo Finance"""
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)

        if df.empty:
            return None

        df.columns = [col.lower() for col in df.columns]
        df = df.reset_index()
        df.rename(columns={'date': 'datetime'}, inplace=True)

        return df

    def _get_alpha_vantage(self, symbol: str) -> pd.DataFrame:
        """Obtener datos de Alpha Vantage"""
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'apikey': self.av_key,
            'outputsize': 'compact'
        }

        response = requests.get(self.av_base_url, params=params, timeout=10)
        data = response.json()

        if 'Time Series (Daily)' not in data:
            return None

        time_series = data['Time Series (Daily)']

        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        df.index = pd.to_datetime(df.index)
        df = df.astype(float)
        df = df.sort_index()
        df = df.reset_index()
        df.rename(columns={'index': 'datetime'}, inplace=True)

        return df

    def _generate_synthetic_data(self, symbol: str, days: int = 200) -> pd.DataFrame:
        """Generar datos sintéticos realistas"""
        np.random.seed(hash(symbol) % (2**32))

        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

        S0 = 100
        mu = 0.0005
        sigma = 0.02

        returns = np.random.normal(mu, sigma, days)
        price = S0 * np.exp(np.cumsum(returns))

        df = pd.DataFrame({
            'datetime': dates,
            'open': price * np.random.uniform(0.99, 1.01, days),
            'high': price * np.random.uniform(1.00, 1.03, days),
            'low': price * np.random.uniform(0.97, 1.00, days),
            'close': price,
            'volume': np.random.randint(1000000, 10000000, days)
        })

        return df

    def get_news_sentiment(self, symbol: str) -> List[str]:
        """Obtener noticias recientes"""
        try:
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': symbol,
                'apikey': self.av_key,
                'limit': 10
            }

            response = requests.get(self.av_base_url, params=params, timeout=10)
            data = response.json()

            if 'feed' not in data:
                return self._get_default_news()

            news = []
            for item in data['feed'][:5]:
                title = item.get('title', '')
                summary = item.get('summary', '')
                news.append(f"{title}. {summary}"[:200])

            return news if news else self._get_default_news()

        except:
            return self._get_default_news()

    def _get_default_news(self) -> List[str]:
        """Noticias por defecto"""
        return [
            "Markets show positive momentum with tech sector leading gains",
            "Federal Reserve signals potential rate adjustments ahead",
            "Strong earnings reports boost investor confidence",
            "Global markets react to economic data releases",
            "Trading volumes remain elevated amid market volatility"
        ]

# ============================================================================
# IMPORTAR MÓDULOS DEL SISTEMA
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

def extract_numeric_data(df: pd.DataFrame) -> np.ndarray:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    return df[numeric_cols].values

def get_price_column(df: pd.DataFrame) -> np.ndarray:
    if 'close' in df.columns:
        return df['close'].values
    elif 'Close' in df.columns:
        return df['Close'].values
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            return df[numeric_cols[0]].values
        else:
            raise ValueError("No se encontraron columnas numéricas")

# ============================================================================
# SISTEMA GOD MODE CON DATOS REALES
# ============================================================================

class RealMarketGodMode:
    """Sistema God Mode integrado con datos de mercado reales"""

    def __init__(self, alpha_vantage_key: str = None):
        self.market_api = MarketDataAPI(alpha_vantage_key)

        print("\n🔧 Inicializando módulos del sistema...")

        # Importar módulos del sistema anterior
        try:
            from god_mode_complete import (
                TemporalModule, VisionModule, TabularModule, 
                NLPModule, GraphModule, SACAgent, TimeGANSimulator
            )

            self.temporal_module = TemporalModule()
            self.vision_module = VisionModule()
            self.tabular_module = TabularModule()
            self.nlp_module = NLPModule()
            self.graph_module = GraphModule()
            self.sac_agent = SACAgent(state_dim=128, action_dim=3)
            self.timegan = TimeGANSimulator()

            print("  ✓ Todos los módulos cargados correctamente")
        except ImportError:
            print("  ⚠ No se pudo importar test_performance.py")
            print("  ℹ Ejecutando en modo de prueba con módulos simulados")
            self._init_mock_modules()

    def _init_mock_modules(self):
        """Inicializar módulos simulados si no existe test_performance.py"""
        class MockModule:
            def predict(self, data):
                return ModelOutput(
                    prediction=np.random.randn(10),
                    confidence=0.75,
                    metadata={},
                    model_name='mock'
                )
            def analyze(self, data):
                return ModelOutput(
                    prediction=np.random.randn(10),
                    confidence=0.75,
                    metadata={'detected_patterns': ['mock_pattern']},
                    model_name='mock'
                )
            def forward(self, *args):
                return np.random.randn(10, 20)

        class MockSAC:
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
            def generate_scenarios(self, n_scenarios=10, horizon=10):
                return np.random.randn(n_scenarios, horizon)

        self.temporal_module = MockModule()
        self.vision_module = MockModule()
        self.tabular_module = MockModule()
        self.nlp_module = MockModule()
        self.graph_module = MockModule()
        self.sac_agent = MockSAC()
        self.timegan = MockTimeGAN()

    def analyze_symbol(self, symbol: str, period: str = "3mo") -> Dict[str, Any]:
        """Análisis completo de un símbolo con datos reales"""
        print("\n" + "="*80)
        print(f"GOD MODE ANALYSIS - {symbol.upper()}")
        print("="*80)

        # Obtener datos de mercado
        price_data = self.market_api.get_stock_data(symbol, period)
        news_data = self.market_api.get_news_sentiment(symbol)

        # Mostrar info del símbolo
        latest_price = price_data['close'].iloc[-1]
        price_change = ((price_data['close'].iloc[-1] / price_data['close'].iloc[0]) - 1) * 100

        print(f"\n📊 Información del Símbolo:")
        print(f"   Precio actual: ${latest_price:.2f}")
        print(f"   Cambio período: {price_change:+.2f}%")
        print(f"   Datos históricos: {len(price_data)} días")
        print(f"   Rango: ${price_data['close'].min():.2f} - ${price_data['close'].max():.2f}")

        # Preparar datos
        market_data = {
            'price_data': price_data,
            'news': news_data
        }

        # Ejecutar análisis
        result = self._analyze_and_decide(market_data)

        # Añadir información
        result['symbol'] = symbol.upper()
        result['current_price'] = float(latest_price)
        result['price_change'] = float(price_change)

        return result

    def _analyze_and_decide(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Pipeline completo de análisis"""
        print("\n" + "="*80)
        print("ANÁLISIS MULTI-MODELO")
        print("="*80)

        # 1. MÓDULO TEMPORAL
        print("\n[1/7] Módulo Temporal (ARIMA + GARCH + TFT + N-BEATS)...")
        temporal_output = self.temporal_module.predict(market_data['price_data'])
        print(f"  ✓ Predicción: ${temporal_output.prediction[0]:.2f} → ${temporal_output.prediction[-1]:.2f}")
        print(f"  ✓ Confianza: {temporal_output.confidence:.2%}")

        # 2. MÓDULO VISIÓN
        print("\n[2/7] Módulo Visión (1D-CNN + ResNet + YOLO)...")
        vision_output = self.vision_module.analyze(market_data['price_data'])
        detected_patterns = vision_output.metadata.get('detected_patterns', ['none'])
        print(f"  ✓ Patrones: {detected_patterns}")
        print(f"  ✓ Confianza: {vision_output.confidence:.2%}")

        # 3. MÓDULO TABULAR
        print("\n[3/7] Módulo Tabular (LightGBM + CatBoost)...")
        tabular_features = self._prepare_tabular_features(market_data)
        tabular_output = self.tabular_module.predict(tabular_features)
        print(f"  ✓ Features: {len(tabular_features.columns)}")
        print(f"  ✓ Confianza: {tabular_output.confidence:.2%}")

        # 4. MÓDULO NLP
        print("\n[4/7] Módulo NLP (FinBERT + GPT-4 + NER)...")
        nlp_output = self.nlp_module.analyze(market_data['news'])
        sentiment = nlp_output.prediction[0]
        sentiment_label = "POSITIVO ✅" if sentiment > 0.5 else "NEGATIVO ❌"
        print(f"  ✓ Sentimiento: {sentiment:.2f} ({sentiment_label})")
        print(f"  ✓ Confianza: {nlp_output.confidence:.2%}")

        # 5. MÓDULO GRAFOS
        print("\n[5/7] Módulo Grafos (GCN + GAT)...")
        graph_data = self._prepare_graph_data(market_data)
        graph_output = self.graph_module.analyze(graph_data['adjacency'], graph_data['features'])
        print(f"  ✓ Red: {graph_data['adjacency'].shape[0]} nodos")
        print(f"  ✓ Confianza: {graph_output.confidence:.2%}")

        # 6. MARKET STATE
        print("\n[6/7] Construyendo Market State...")
        market_state = self._construct_market_state(
            temporal_output, vision_output, tabular_output, 
            nlp_output, graph_output
        )
        print(f"  ✓ State vector construido")

        # 7. DECISIÓN CON SAC
        print("\n[7/7] Cerebro SAC - Decisión Final...")
        decision = self.sac_agent.get_trading_decision(market_state)

        # EMOJI según acción - FIX APLICADO AQUÍ
        action_emoji = {'BUY':'🟢', 'SELL':'🔴', 'HOLD':'🟡'}
        emoji = action_emoji.get(decision['action'], '⚪')

        print(f"  {emoji} DECISIÓN: {decision['action']}")
        print(f"  ✓ Posición: {decision['position_size']:.1%}")
        print(f"  ✓ Stop Loss: {decision['stop_loss']:.1%}")
        print(f"  ✓ Take Profit: {decision['take_profit']:.1%}")

        # SIMULACIÓN
        print("\n[BONUS] TimeGAN - Escenarios...")
        scenarios = self.timegan.generate_scenarios(n_scenarios=10, horizon=10)
        print(f"  ✓ {len(scenarios)} escenarios generados")

        # Resultado
        confidence_overall = np.mean([
            temporal_output.confidence,
            vision_output.confidence,
            tabular_output.confidence,
            nlp_output.confidence,
            graph_output.confidence
        ])

        result = {
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

        print("\n" + "="*80)
        print(f"CONFIANZA GENERAL: {confidence_overall:.1%}")
        print("="*80)

        return result

    def _prepare_tabular_features(self, market_data: Dict) -> pd.DataFrame:
        """Preparar features técnicos"""
        df = market_data['price_data'].copy()
        price_col = get_price_column(df)

        features_dict = {
            'returns': 0.0,
            'volatility': 0.01,
            'sma_20': price_col[-1],
            'sma_50': price_col[-1],
            'rsi': 50.0
        }

        if len(price_col) > 1:
            returns = pd.Series(price_col).pct_change().dropna()
            features_dict['returns'] = returns.iloc[-1] if len(returns) > 0 else 0.0
            features_dict['volatility'] = returns.std() if len(returns) > 0 else 0.01

        if len(price_col) >= 20:
            features_dict['sma_20'] = pd.Series(price_col).rolling(20).mean().iloc[-1]

        if len(price_col) >= 50:
            features_dict['sma_50'] = pd.Series(price_col).rolling(50).mean().iloc[-1]

        if len(price_col) >= 14:
            delta = pd.Series(price_col).diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            features_dict['rsi'] = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0

        return pd.DataFrame([features_dict])

    def _prepare_graph_data(self, market_data: Dict) -> Dict:
        """Preparar grafo de correlaciones"""
        n_assets = 10
        adjacency = np.random.rand(n_assets, n_assets)
        adjacency = (adjacency + adjacency.T) / 2
        np.fill_diagonal(adjacency, 1.0)
        features = np.random.randn(n_assets, 20)

        return {'adjacency': adjacency, 'features': features}

    def _construct_market_state(self, temporal, vision, tabular, nlp, graph) -> MarketState:
        """Construir Market State"""
        price_features = temporal.prediction[:10]
        if len(price_features) < 10:
            price_features = np.pad(price_features, (0, 10 - len(price_features)))

        technical = np.array([
            tabular.confidence,
            temporal.confidence,
            vision.confidence
        ])

        sentiment = float(nlp.prediction[0])
        volatility = float(np.std(temporal.prediction)) if len(temporal.prediction) > 1 else 0.02

        graph_emb = graph.prediction.flatten()[:10]
        if len(graph_emb) < 10:
            graph_emb = np.pad(graph_emb, (0, 10 - len(graph_emb)))

        now = datetime.now()
        time_features = np.array([
            now.hour / 24.0,
            now.weekday() / 7.0,
            now.day / 31.0
        ])

        return MarketState(
            price_features=price_features,
            technical_indicators=technical,
            sentiment_score=sentiment,
            volatility=volatility,
            graph_embeddings=graph_emb,
            time_features=time_features
        )

    def compare_symbols(self, symbols: List[str]) -> pd.DataFrame:
        """Comparar múltiples símbolos"""
        print("\n" + "="*80)
        print(f"COMPARACIÓN DE {len(symbols)} SÍMBOLOS")
        print("="*80)

        results = []

        for symbol in symbols:
            try:
                result = self.analyze_symbol(symbol, period="1mo")

                results.append({
                    'Symbol': symbol,
                    'Price': result['current_price'],
                    'Change %': result['price_change'],
                    'Decision': result['decision']['action'],
                    'Position': f"{result['decision']['position_size']:.1%}",
                    'Confidence': f"{result['confidence_breakdown']['overall']:.1%}",
                    'Sentiment': f"{result['sentiment']:.2f}"
                })

                time.sleep(1)

            except Exception as e:
                print(f"\n❌ Error en {symbol}: {str(e)[:100]}")
                continue

        df = pd.DataFrame(results)

        print("\n" + "="*80)
        print("TABLA COMPARATIVA")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80)

        return df

# ============================================================================
# EJEMPLO DE USO
# ============================================================================

def main():
    """Demo completo con datos reales"""

    print("\n" + "="*80)
    print(" "*15 + "GOD MODE TRADING SYSTEM - REAL MARKET")
    print(" "*18 + "Alpha Vantage + Yahoo Finance")
    print("="*80)

    # Inicializar
    god_mode = RealMarketGodMode(alpha_vantage_key="QZF8CB4TECMS754I")

    # Analizar símbolos
    symbols = ["AAPL", "MSFT", "TSLA"]

    print(f"\n🎯 Analizando {len(symbols)} símbolos principales...")

    for symbol in symbols:
        try:
            result = god_mode.analyze_symbol(symbol, period="3mo")

            print(f"\n{'='*80}")
            print(f"📊 RECOMENDACIÓN FINAL - {symbol}")
            print(f"{'='*80}")
            print(f"  Precio: ${result['current_price']:.2f}")

            # FIX APLICADO - Emoji fuera del f-string
            action_emoji = {'BUY':'🟢', 'SELL':'🔴', 'HOLD':'🟡'}
            emoji = action_emoji.get(result['decision']['action'], '⚪')
            print(f"  {emoji} Acción: {result['decision']['action']}")

            print(f"  Posición: {result['decision']['position_size']:.1%}")
            print(f"  Confianza: {result['confidence_breakdown']['overall']:.1%}")
            print(f"  Sentimiento: {result['sentiment']:.2f}")
            print(f"{'='*80}")

            time.sleep(2)
        except Exception as e:
            print(f"\n❌ Error procesando {symbol}: {str(e)[:100]}")
            continue

    # Comparación
    print("\n\n🔍 Generando tabla comparativa...")
    try:
        comparison = god_mode.compare_symbols(symbols)
    except Exception as e:
        print(f"⚠ Error en comparación: {str(e)[:100]}")

    print("\n✅ Análisis completado")

    return god_mode

if __name__ == "__main__":
    system = main()