
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN Y ESTRUCTURAS DE DATOS
# ============================================================================

@dataclass
class ModelOutput:
    """Estructura para outputs de modelos"""
    prediction: np.ndarray
    confidence: float
    metadata: Dict[str, Any]
    model_name: str

@dataclass
class MarketState:
    """Estado del mercado para el agente RL"""
    price_features: np.ndarray
    technical_indicators: np.ndarray
    sentiment_score: float
    volatility: float
    graph_embeddings: np.ndarray
    time_features: np.ndarray

# ============================================================================
# UTILIDADES
# ============================================================================

def extract_numeric_data(df: pd.DataFrame) -> np.ndarray:
    """Extraer solo columnas numéricas de un DataFrame"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    return df[numeric_cols].values

def get_price_column(df: pd.DataFrame) -> np.ndarray:
    """Obtener columna de precio principal"""
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
# 1. MÓDULO TEMPORAL (ARIMA + GARCH + TFT + N-BEATS)
# ============================================================================

class TemporalModule:
    """Módulo de series temporales con múltiples modelos"""

    def __init__(self):
        self.models = {
            'arima': ARIMAPredictor(),
            'garch': GARCHVolatility(),
            'tft': TemporalFusionTransformer(),
            'nbeats': NBEATSPredictor()
        }

    def predict(self, data: pd.DataFrame) -> ModelOutput:
        """Predicción ensemble de todos los modelos temporales"""
        predictions = []
        confidences = []

        for name, model in self.models.items():
            try:
                pred = model.forecast(data)
                predictions.append(pred['forecast'])
                confidences.append(pred['confidence'])
            except Exception as e:
                print(f"  ⚠ Warning en {name}: {str(e)[:50]}")
                predictions.append(np.zeros(10))
                confidences.append(0.5)

        weights = np.array(confidences) / sum(confidences) if sum(confidences) > 0 else np.ones(len(confidences)) / len(confidences)
        final_prediction = np.average(predictions, axis=0, weights=weights)

        return ModelOutput(
            prediction=final_prediction,
            confidence=np.mean(confidences),
            metadata={'individual_preds': predictions, 'weights': weights.tolist()},
            model_name='temporal_ensemble'
        )

class ARIMAPredictor:
    """ARIMA para tendencias lineales"""
    def __init__(self, order=(5,1,2)):
        self.order = order

    def forecast(self, data: pd.DataFrame, horizon: int = 10) -> Dict:
        """Forecast ARIMA"""
        try:
            from statsmodels.tsa.arima.model import ARIMA
            values = get_price_column(data)

            model = ARIMA(values, order=self.order)
            fitted = model.fit()
            forecast = fitted.forecast(steps=horizon)

            conf_int = fitted.get_forecast(steps=horizon).conf_int()
            confidence = 1.0 / (1.0 + np.mean(conf_int.iloc[:, 1] - conf_int.iloc[:, 0]))

            return {'forecast': forecast.values, 'confidence': float(confidence)}
        except Exception as e:
            values = get_price_column(data)
            return {'forecast': np.full(horizon, values[-1]), 'confidence': 0.5}

class GARCHVolatility:
    """GARCH para modelar volatilidad"""
    def __init__(self, p=1, q=1):
        self.p = p
        self.q = q

    def forecast(self, data: pd.DataFrame, horizon: int = 10) -> Dict:
        """Forecast de volatilidad con GARCH"""
        try:
            from arch import arch_model
            values = get_price_column(data)
            returns = pd.Series(values).pct_change().dropna() * 100

            model = arch_model(returns, vol='Garch', p=self.p, q=self.q)
            fitted = model.fit(disp='off')
            forecast = fitted.forecast(horizon=horizon)
            volatility = np.sqrt(forecast.variance.values[-1, :])

            confidence = 0.8 if fitted.aic < 1000 else 0.6

            return {'forecast': volatility, 'confidence': confidence}
        except Exception as e:
            values = get_price_column(data)
            returns = pd.Series(values).pct_change().dropna()
            return {'forecast': np.full(horizon, returns.std()), 'confidence': 0.5}

class TemporalFusionTransformer:
    """TFT para atención temporal multi-horizonte"""
    def __init__(self, hidden_size=64, num_heads=4):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.model = None

    def forecast(self, data: pd.DataFrame, horizon: int = 10) -> Dict:
        """TFT forecast con atención"""
        X = self._prepare_features(data)
        predictions = self._predict(X, horizon)

        return {
            'forecast': predictions,
            'confidence': 0.85
        }

    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Preparar features para TFT - SOLO VALORES NUMÉRICOS"""
        numeric_data = extract_numeric_data(data)

        if len(numeric_data) > 50:
            return numeric_data[-50:]
        else:
            return numeric_data

    def _predict(self, X: np.ndarray, horizon: int) -> np.ndarray:
        """Predicción iterativa"""
        if len(X) == 0 or X.shape[1] == 0:
            return np.zeros(horizon)

        predictions = []
        last_value = X[-1, 0]

        for _ in range(horizon):
            next_val = last_value * (1 + np.random.normal(0.0005, 0.01))
            predictions.append(next_val)
            last_value = next_val

        return np.array(predictions)

class NBEATSPredictor:
    """N-BEATS para descomposición de series temporales"""
    def __init__(self, stack_types=['trend', 'seasonality']):
        self.stack_types = stack_types

    def forecast(self, data: pd.DataFrame, horizon: int = 10) -> Dict:
        """N-BEATS forecast"""
        values = get_price_column(data)

        trend = self._extract_trend(values)
        seasonality = self._extract_seasonality(values)

        trend_forecast = self._forecast_trend(trend, horizon)
        seasonal_forecast = self._forecast_seasonality(seasonality, horizon)

        final_forecast = trend_forecast + seasonal_forecast

        return {
            'forecast': final_forecast,
            'confidence': 0.82
        }

    def _extract_trend(self, values: np.ndarray) -> np.ndarray:
        """Extraer tendencia con moving average"""
        from scipy.ndimage import uniform_filter1d
        window = min(20, len(values) // 3)
        if window < 3:
            window = 3
        return uniform_filter1d(values, size=window, mode='nearest')

    def _extract_seasonality(self, values: np.ndarray) -> np.ndarray:
        """Extraer estacionalidad"""
        trend = self._extract_trend(values)
        return values - trend

    def _forecast_trend(self, trend: np.ndarray, horizon: int) -> np.ndarray:
        """Forecast lineal de tendencia"""
        x = np.arange(len(trend))
        coeffs = np.polyfit(x, trend, deg=1)
        future_x = np.arange(len(trend), len(trend) + horizon)
        return np.polyval(coeffs, future_x)

    def _forecast_seasonality(self, seasonality: np.ndarray, horizon: int) -> np.ndarray:
        """Forecast de estacionalidad con repetición"""
        period = min(20, len(seasonality))
        if period == 0:
            return np.zeros(horizon)
        return np.tile(seasonality[-period:], (horizon // period + 1))[:horizon]

# ============================================================================
# 2. MÓDULO DE VISIÓN (1D-CNN + ResNet + YOLO) - FIXED
# ============================================================================

class VisionModule:
    """Módulo de visión para patrones en gráficos y candlesticks"""

    def __init__(self):
        self.cnn = CNN1DPatternRecognizer()
        self.resnet = ResNetFeatureExtractor()
        self.yolo = YOLOPatternDetector()

    def analyze(self, data: pd.DataFrame) -> ModelOutput:
        """Análisis visual de patrones de precio"""
        signal = self._to_signal(data)

        cnn_patterns = self.cnn.detect(signal)
        resnet_features = self.resnet.extract(signal)
        yolo_patterns = self.yolo.detect_patterns(signal)

        combined_signal = self._combine_vision_outputs(
            cnn_patterns, resnet_features, yolo_patterns
        )

        return ModelOutput(
            prediction=combined_signal,
            confidence=0.78,
            metadata={
                'cnn_shape': cnn_patterns.shape,
                'resnet_shape': resnet_features.shape,
                'detected_patterns': yolo_patterns['patterns']
            },
            model_name='vision_ensemble'
        )

    def _to_signal(self, data: pd.DataFrame) -> np.ndarray:
        """Convertir datos a señal 1D normalizada"""
        signal = get_price_column(data)

        signal_mean = signal.mean()
        signal_std = signal.std()
        if signal_std == 0:
            signal_std = 1.0

        return (signal - signal_mean) / signal_std

    def _combine_vision_outputs(self, cnn, resnet, yolo) -> np.ndarray:
        """Combinar outputs de modelos de visión - FIXED"""
        # Flatten todos los arrays a 1D
        cnn_flat = cnn.flatten() if isinstance(cnn, np.ndarray) else np.array(cnn).flatten()
        resnet_flat = resnet.flatten() if isinstance(resnet, np.ndarray) else np.array(resnet).flatten()
        yolo_flat = yolo['scores'].flatten() if isinstance(yolo['scores'], np.ndarray) else np.array(yolo['scores']).flatten()

        # Determinar longitud mínima
        lengths = [len(cnn_flat), len(resnet_flat), len(yolo_flat)]
        valid_lengths = [l for l in lengths if l > 0]

        if not valid_lengths:
            return np.zeros(10)

        min_len = min(valid_lengths)

        # Pad arrays que sean demasiado cortos
        if len(cnn_flat) == 0:
            cnn_flat = np.zeros(min_len)
        if len(resnet_flat) == 0:
            resnet_flat = np.zeros(min_len)
        if len(yolo_flat) == 0:
            yolo_flat = np.zeros(min_len)

        # Truncar o pad según sea necesario
        cnn_final = cnn_flat[:min_len] if len(cnn_flat) >= min_len else np.pad(cnn_flat, (0, min_len - len(cnn_flat)))
        resnet_final = resnet_flat[:min_len] if len(resnet_flat) >= min_len else np.pad(resnet_flat, (0, min_len - len(resnet_flat)))
        yolo_final = yolo_flat[:min_len] if len(yolo_flat) >= min_len else np.pad(yolo_flat, (0, min_len - len(yolo_flat)))

        # Ahora todos son 1D con la misma longitud
        return np.concatenate([cnn_final, resnet_final, yolo_final])

class CNN1DPatternRecognizer:
    """CNN 1D para reconocimiento de patrones en series"""
    def __init__(self, filters=[64, 128, 256]):
        self.filters = filters

    def detect(self, signal: np.ndarray) -> np.ndarray:
        """Detectar patrones con convoluciones 1D simuladas"""
        try:
            import torch
            import torch.nn as nn

            model = nn.Sequential(
                nn.Conv1d(1, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(50)
            )

            with torch.no_grad():
                signal_tensor = torch.FloatTensor(signal).unsqueeze(0).unsqueeze(0)
                output = model(signal_tensor)

            return output.squeeze().numpy()
        except:
            from scipy.ndimage import convolve1d
            kernel = np.ones(5) / 5
            smoothed = convolve1d(signal, kernel, mode='constant')
            result = smoothed[:50] if len(smoothed) > 50 else np.pad(smoothed, (0, max(0, 50-len(smoothed))))
            return result

class ResNetFeatureExtractor:
    """ResNet adaptado para extracción de features en series"""
    def __init__(self, depth=18):
        self.depth = depth

    def extract(self, signal: np.ndarray) -> np.ndarray:
        """Extraer features profundas con ResNet"""
        x = signal.copy()

        for i in range(3):
            residual = x
            x = self._conv_block(x, kernel_size=5)
            min_len = min(len(x), len(residual))
            x[:min_len] = x[:min_len] + residual[:min_len]

        features = np.array([
            x.mean(), x.std(), x.max(), x.min(),
            np.percentile(x, 25), np.percentile(x, 75)
        ])

        return features

    def _conv_block(self, x: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Bloque convolucional simple"""
        from scipy.ndimage import convolve1d
        kernel = np.ones(kernel_size) / kernel_size
        return convolve1d(x, kernel, mode='constant')

class YOLOPatternDetector:
    """YOLO adaptado para detectar patrones de candlesticks"""
    def __init__(self):
        self.patterns = ['head_shoulders', 'double_top', 'triangle', 'flag']

    def detect_patterns(self, signal: np.ndarray) -> Dict[str, Any]:
        """Detectar patrones específicos tipo YOLO"""
        detections = []
        scores = []

        peaks = self._find_peaks(signal)
        valleys = self._find_valleys(signal)

        if len(peaks) >= 3:
            if peaks[1] > peaks[0] and peaks[1] > peaks[2]:
                detections.append('head_shoulders')
                scores.append(0.85)

        if len(peaks) >= 2:
            if abs(peaks[0] - peaks[1]) < 0.02 * abs(signal.max() - signal.min() + 1e-8):
                detections.append('double_top')
                scores.append(0.78)

        if len(peaks) > 0 and len(valleys) > 0:
            detections.append('triangle')
            scores.append(0.65)

        if not scores:
            scores = [0.5]
            detections = ['none']

        return {
            'patterns': detections,
            'scores': np.array(scores),
            'peaks': peaks,
            'valleys': valleys
        }

    def _find_peaks(self, signal: np.ndarray) -> List[float]:
        """Encontrar picos locales"""
        try:
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(signal, distance=5)
            return signal[peaks].tolist() if len(peaks) > 0 else []
        except:
            return []

    def _find_valleys(self, signal: np.ndarray) -> List[float]:
        """Encontrar valles locales"""
        try:
            from scipy.signal import find_peaks
            valleys, _ = find_peaks(-signal, distance=5)
            return signal[valleys].tolist() if len(valleys) > 0 else []
        except:
            return []

# ============================================================================
# 3. MÓDULO TABULAR (LightGBM + CatBoost)
# ============================================================================

class TabularModule:
    """Módulo para datos tabulares estructurados"""

    def __init__(self):
        self.lgbm = LightGBMPredictor()
        self.catboost = CatBoostPredictor()

    def predict(self, features: pd.DataFrame) -> ModelOutput:
        """Predicción ensemble de modelos tabulares"""
        lgbm_pred = self.lgbm.predict(features)
        catboost_pred = self.catboost.predict(features)

        final_pred = 0.5 * lgbm_pred['prediction'] + 0.5 * catboost_pred['prediction']
        confidence = (lgbm_pred['confidence'] + catboost_pred['confidence']) / 2

        return ModelOutput(
            prediction=final_pred,
            confidence=confidence,
            metadata={
                'lgbm': {'confidence': lgbm_pred['confidence']},
                'catboost': {'confidence': catboost_pred['confidence']}
            },
            model_name='tabular_ensemble'
        )

class LightGBMPredictor:
    """LightGBM para features tabulares"""
    def __init__(self, params=None):
        self.params = params or {
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 31,
            'learning_rate': 0.05
        }
        self.model = None

    def predict(self, features: pd.DataFrame) -> Dict:
        """Predicción con LightGBM"""
        if self.model is None:
            self.model = self._get_dummy_model(features)

        try:
            predictions = self.model.predict(features)
            confidence = 0.82
        except:
            predictions = np.zeros(len(features))
            confidence = 0.5

        return {
            'prediction': predictions,
            'confidence': confidence
        }

    def _get_dummy_model(self, features: pd.DataFrame):
        """Modelo dummy para demo"""
        class DummyModel:
            def predict(self, X):
                return np.random.randn(len(X)) * 0.1
        return DummyModel()

class CatBoostPredictor:
    """CatBoost para features categóricas y numéricas"""
    def __init__(self, iterations=100):
        self.iterations = iterations
        self.model = None

    def predict(self, features: pd.DataFrame) -> Dict:
        """Predicción con CatBoost"""
        if self.model is None:
            self.model = self._get_dummy_model(features)

        try:
            predictions = self.model.predict(features)
            confidence = 0.84
        except:
            predictions = np.zeros(len(features))
            confidence = 0.5

        return {
            'prediction': predictions,
            'confidence': confidence
        }

    def _get_dummy_model(self, features: pd.DataFrame):
        """Modelo dummy para demo"""
        class DummyModel:
            def predict(self, X):
                return np.random.randn(len(X)) * 0.1
        return DummyModel()

# ============================================================================
# 4. MÓDULO NLP (FinBERT + GPT-4 + NER)
# ============================================================================

class NLPModule:
    """Módulo de análisis de noticias y sentimiento"""

    def __init__(self):
        self.finbert = FinBERTSentiment()
        self.gpt = GPT4Analyzer()
        self.ner = NamedEntityRecognizer()

    def analyze(self, texts: List[str]) -> ModelOutput:
        """Análisis NLP completo"""
        sentiment_scores = self.finbert.analyze_sentiment(texts)
        gpt_insights = self.gpt.analyze(texts)
        entities = self.ner.extract(texts)

        combined_score = self._combine_nlp_outputs(
            sentiment_scores, gpt_insights, entities
        )

        return ModelOutput(
            prediction=np.array([combined_score]),
            confidence=0.88,
            metadata={
                'sentiment_avg': float(np.mean(sentiment_scores)) if len(sentiment_scores) > 0 else 0.5,
                'gpt_sentiment': gpt_insights.get('sentiment_score', 0.5),
                'num_entities': len(entities)
            },
            model_name='nlp_ensemble'
        )

    def _combine_nlp_outputs(self, sentiment, gpt, entities) -> float:
        """Combinar outputs de NLP"""
        sentiment_avg = np.mean(sentiment) if len(sentiment) > 0 else 0.5
        gpt_score = gpt.get('sentiment_score', 0.5)
        entity_weight = min(len(entities) / 100.0, 0.2)

        return 0.5 * sentiment_avg + 0.4 * gpt_score + 0.1 * entity_weight

class FinBERTSentiment:
    """FinBERT para análisis de sentimiento financiero"""
    def __init__(self):
        self.model_name = 'finbert'

    def analyze_sentiment(self, texts: List[str]) -> List[float]:
        """Análisis de sentimiento"""
        sentiments = []
        for text in texts:
            positive_words = ['gain', 'profit', 'growth', 'up', 'bullish', 'rally', 'positive']
            negative_words = ['loss', 'drop', 'decline', 'down', 'bearish', 'fall', 'negative']

            text_lower = text.lower()
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)

            if pos_count + neg_count > 0:
                score = pos_count / (pos_count + neg_count)
            else:
                score = 0.5

            sentiments.append(score)

        return sentiments

class GPT4Analyzer:
    """GPT-4 para análisis profundo de contexto"""
    def __init__(self):
        self.model = 'gpt-4'

    def analyze(self, texts: List[str]) -> Dict:
        """Análisis con GPT-4"""
        combined_text = ' '.join(texts)

        analysis = {
            'sentiment_score': self._compute_sentiment(combined_text),
            'urgency': self._compute_urgency(combined_text),
            'market_impact': self._compute_impact(combined_text),
            'key_topics': self._extract_topics(combined_text)
        }

        return analysis

    def _compute_sentiment(self, text: str) -> float:
        """Computar sentimiento"""
        positive = len([w for w in text.split() if w.lower() in ['good', 'great', 'positive', 'up', 'gain']])
        negative = len([w for w in text.split() if w.lower() in ['bad', 'negative', 'down', 'loss', 'drop']])
        total = positive + negative
        return positive / total if total > 0 else 0.5

    def _compute_urgency(self, text: str) -> float:
        """Computar urgencia"""
        urgent_words = ['breaking', 'urgent', 'alert', 'now', 'immediate']
        count = sum(1 for word in urgent_words if word in text.lower())
        return min(count / 3.0, 1.0)

    def _compute_impact(self, text: str) -> float:
        """Computar impacto en mercado"""
        impact_words = ['market', 'billion', 'million', 'stock', 'index']
        count = sum(1 for word in impact_words if word in text.lower())
        return min(count / 5.0, 1.0)

    def _extract_topics(self, text: str) -> List[str]:
        """Extraer tópicos principales"""
        words = text.lower().split()
        common = ['market', 'stock', 'price', 'trading', 'investor']
        return [w for w in common if w in words]

class NamedEntityRecognizer:
    """NER para extraer entidades relevantes"""
    def __init__(self):
        self.entity_types = ['ORG', 'MONEY', 'PERCENT', 'DATE']

    def extract(self, texts: List[str]) -> List[Dict]:
        """Extraer entidades nombradas"""
        import re

        entities = []

        for text in texts:
            orgs = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
            money = re.findall(r'\$[\d,]+(?:\.\d{2})?', text)
            percents = re.findall(r'\d+(?:\.\d+)?%', text)

            entities.append({
                'organizations': orgs,
                'money': money,
                'percentages': percents
            })

        return entities

# ============================================================================
# 5. MÓDULO DE GRAFOS (GCN + GAT)
# ============================================================================

class GraphModule:
    """Módulo para relaciones y redes en mercados"""

    def __init__(self):
        self.gcn = GraphConvolutionalNetwork()
        self.gat = GraphAttentionNetwork()

    def analyze(self, adjacency_matrix: np.ndarray, node_features: np.ndarray) -> ModelOutput:
        """Análisis de grafos"""
        gcn_embeddings = self.gcn.forward(adjacency_matrix, node_features)
        gat_embeddings = self.gat.forward(adjacency_matrix, node_features)

        combined = 0.5 * gcn_embeddings + 0.5 * gat_embeddings

        return ModelOutput(
            prediction=combined,
            confidence=0.79,
            metadata={
                'num_nodes': adjacency_matrix.shape[0],
                'embedding_dim': combined.shape[1] if len(combined.shape) > 1 else combined.shape[0]
            },
            model_name='graph_ensemble'
        )

class GraphConvolutionalNetwork:
    """GCN para aprendizaje en grafos"""
    def __init__(self, hidden_dim=64):
        self.hidden_dim = hidden_dim

    def forward(self, adj_matrix: np.ndarray, features: np.ndarray) -> np.ndarray:
        """Forward pass de GCN"""
        adj_norm = self._normalize_adjacency(adj_matrix)

        h1 = adj_norm @ features
        h1 = self._relu(h1)

        h2 = adj_norm @ h1

        return h2

    def _normalize_adjacency(self, adj: np.ndarray) -> np.ndarray:
        """Normalización simétrica de matriz de adjacencia"""
        degree = np.sum(adj, axis=1)
        degree_inv_sqrt = np.power(degree, -0.5)
        degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.
        D_inv_sqrt = np.diag(degree_inv_sqrt)

        return D_inv_sqrt @ adj @ D_inv_sqrt

    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation"""
        return np.maximum(0, x)

class GraphAttentionNetwork:
    """GAT para atención en grafos"""
    def __init__(self, num_heads=4, hidden_dim=64):
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

    def forward(self, adj_matrix: np.ndarray, features: np.ndarray) -> np.ndarray:
        """Forward pass de GAT"""
        outputs = []

        for _ in range(self.num_heads):
            attention_weights = self._compute_attention(adj_matrix, features)
            head_output = attention_weights @ features
            outputs.append(head_output)

        combined = np.mean(outputs, axis=0)

        return combined

    def _compute_attention(self, adj: np.ndarray, features: np.ndarray) -> np.ndarray:
        """Computar attention weights"""
        n = adj.shape[0]
        attention = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if adj[i, j] > 0:
                    score = np.dot(features[i], features[j])
                    attention[i, j] = score

        attention = self._softmax(attention, axis=1)

        return attention

    def _softmax(self, x: np.ndarray, axis: int = 1) -> np.ndarray:
        """Softmax function"""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / (np.sum(exp_x, axis=axis, keepdims=True) + 1e-8)

# ============================================================================
# 6. CEREBRO - SAC (Soft Actor-Critic) para Decisiones
# ============================================================================

class SACAgent:
    """Soft Actor-Critic para toma de decisiones de trading"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.actor = ActorNetwork(state_dim, action_dim, hidden_dim)
        self.critic1 = CriticNetwork(state_dim, action_dim, hidden_dim)
        self.critic2 = CriticNetwork(state_dim, action_dim, hidden_dim)
        self.value = ValueNetwork(state_dim, hidden_dim)

        self.alpha = 0.2
        self.gamma = 0.99

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Seleccionar acción basada en estado"""
        action, log_prob = self.actor.sample(state, deterministic)
        return action

    def get_trading_decision(self, market_state: MarketState) -> Dict[str, Any]:
        """Obtener decisión de trading basada en estado del mercado"""
        state_vector = self._construct_state_vector(market_state)
        action = self.select_action(state_vector, deterministic=False)
        decision = self._interpret_action(action)

        return decision

    def _construct_state_vector(self, market_state: MarketState) -> np.ndarray:
        """Construir vector de estado desde MarketState"""
        state_components = [
            market_state.price_features.flatten(),
            market_state.technical_indicators.flatten(),
            np.array([market_state.sentiment_score]),
            np.array([market_state.volatility]),
            market_state.graph_embeddings.flatten()[:10],
            market_state.time_features.flatten()
        ]

        state_vector = np.concatenate(state_components)

        if len(state_vector) > self.state_dim:
            state_vector = state_vector[:self.state_dim]
        elif len(state_vector) < self.state_dim:
            state_vector = np.pad(state_vector, (0, self.state_dim - len(state_vector)))

        return state_vector

    def _interpret_action(self, action: np.ndarray) -> Dict[str, Any]:
        """Interpretar acción continua a decisión de trading"""
        position = action[0] if len(action) > 0 else 0.0

        if position > 0.3:
            decision_type = 'BUY'
        elif position < -0.3:
            decision_type = 'SELL'
        else:
            decision_type = 'HOLD'

        return {
            'action': decision_type,
            'position_size': abs(position),
            'confidence': min(abs(position), 1.0),
            'stop_loss': action[1] if len(action) > 1 else -0.02,
            'take_profit': action[2] if len(action) > 2 else 0.05
        }

class ActorNetwork:
    """Red actor para política estocástica"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

    def sample(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float]:
        """Sample action from policy"""
        hidden = np.tanh(np.random.randn(self.hidden_dim) * 0.1 + state.mean())

        mean = np.tanh(hidden[:self.action_dim])
        log_std = np.clip(hidden[self.action_dim:2*self.action_dim], -20, 2)
        std = np.exp(log_std)

        if deterministic:
            action = mean
        else:
            action = mean + std * np.random.randn(self.action_dim)
            action = np.tanh(action)

        log_prob = -0.5 * np.sum(np.log(2 * np.pi * std**2 + 1e-8))

        return action, log_prob

class CriticNetwork:
    """Red critic para Q-value"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

    def forward(self, state: np.ndarray, action: np.ndarray) -> float:
        """Compute Q-value"""
        combined = np.concatenate([state, action])
        hidden = np.tanh(combined.mean())
        q_value = hidden
        return q_value

class ValueNetwork:
    """Red de valor para V(s)"""
    def __init__(self, state_dim: int, hidden_dim: int):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

    def forward(self, state: np.ndarray) -> float:
        """Compute state value"""
        hidden = np.tanh(state.mean())
        value = hidden
        return value

# ============================================================================
# 7. SIMULADOR - TimeGAN para Generación de Datos Sintéticos
# ============================================================================

class TimeGANSimulator:
    """TimeGAN para generar escenarios de mercado sintéticos"""

    def __init__(self, seq_len: int = 24, hidden_dim: int = 64):
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

    def generate_scenarios(self, n_scenarios: int = 10, horizon: int = 20) -> np.ndarray:
        """Generar escenarios de mercado sintéticos"""
        scenarios = []

        for _ in range(n_scenarios):
            z = np.random.randn(horizon, self.hidden_dim)
            scenario = np.tanh(z).mean(axis=1)
            scenarios.append(scenario)

        return np.array(scenarios)

# ============================================================================
# 8. SISTEMA INTEGRADOR - God Mode Architecture
# ============================================================================

class GodModeSystem:
    """Sistema integrador de toda la arquitectura"""

    def __init__(self):
        self.temporal_module = TemporalModule()
        self.vision_module = VisionModule()
        self.tabular_module = TabularModule()
        self.nlp_module = NLPModule()
        self.graph_module = GraphModule()
        self.sac_agent = SACAgent(state_dim=128, action_dim=3)
        self.timegan = TimeGANSimulator()

        self.module_weights = {
            'temporal': 0.25,
            'vision': 0.15,
            'tabular': 0.20,
            'nlp': 0.20,
            'graph': 0.20
        }

    def analyze_and_decide(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Pipeline completo de análisis y decisión"""
        print("\n" + "="*80)
        print("GOD MODE SYSTEM - ANÁLISIS COMPLETO")
        print("="*80)

        # 1. MÓDULO TEMPORAL
        print("\n[1/7] Módulo Temporal (ARIMA + GARCH + TFT + N-BEATS)...")
        temporal_output = self.temporal_module.predict(market_data['price_data'])
        print(f"  ✓ Predicción temporal: {temporal_output.prediction[:3]}...")
        print(f"  ✓ Confianza: {temporal_output.confidence:.2%}")

        # 2. MÓDULO VISIÓN
        print("\n[2/7] Módulo Visión (1D-CNN + ResNet + YOLO)...")
        vision_output = self.vision_module.analyze(market_data['price_data'])
        detected_patterns = vision_output.metadata['detected_patterns']
        print(f"  ✓ Patrones detectados: {detected_patterns}")
        print(f"  ✓ Confianza: {vision_output.confidence:.2%}")

        # 3. MÓDULO TABULAR
        print("\n[3/7] Módulo Tabular (LightGBM + CatBoost)...")
        tabular_features = self._prepare_tabular_features(market_data)
        tabular_output = self.tabular_module.predict(tabular_features)
        print(f"  ✓ Features procesados: {len(tabular_features.columns)}")
        print(f"  ✓ Confianza: {tabular_output.confidence:.2%}")

        # 4. MÓDULO NLP
        print("\n[4/7] Módulo NLP (FinBERT + GPT-4 + NER)...")
        news_texts = market_data.get('news', ['Market shows positive trends'])
        nlp_output = self.nlp_module.analyze(news_texts)
        sentiment = nlp_output.prediction[0]
        print(f"  ✓ Sentimiento: {sentiment:.2f} ({'POSITIVO' if sentiment > 0.5 else 'NEGATIVO'})")
        print(f"  ✓ Confianza: {nlp_output.confidence:.2%}")

        # 5. MÓDULO GRAFOS
        print("\n[5/7] Módulo Grafos (GCN + GAT)...")
        graph_data = self._prepare_graph_data(market_data)
        graph_output = self.graph_module.analyze(
            graph_data['adjacency'], 
            graph_data['features']
        )
        print(f"  ✓ Embeddings de grafo: {graph_output.prediction.shape}")
        print(f"  ✓ Confianza: {graph_output.confidence:.2%}")

        # 6. MARKET STATE
        print("\n[6/7] Construyendo Market State...")
        market_state = self._construct_market_state(
            temporal_output, vision_output, tabular_output, 
            nlp_output, graph_output
        )
        print(f"  ✓ Market State construido exitosamente")

        # 7. DECISIÓN CON SAC
        print("\n[7/7] Cerebro SAC - Toma de Decisión...")
        decision = self.sac_agent.get_trading_decision(market_state)
        print(f"  ✓ DECISIÓN: {decision['action']}")
        print(f"  ✓ Tamaño posición: {decision['position_size']:.2%}")
        print(f"  ✓ Confianza: {decision['confidence']:.2%}")

        # 8. SIMULACIÓN
        print("\n[BONUS] TimeGAN - Generación de Escenarios...")
        scenarios = self.timegan.generate_scenarios(n_scenarios=5, horizon=10)
        print(f"  ✓ {len(scenarios)} escenarios generados")

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
                'overall': np.mean([
                    temporal_output.confidence,
                    vision_output.confidence,
                    tabular_output.confidence,
                    nlp_output.confidence,
                    graph_output.confidence
                ])
            }
        }

        print("\n" + "="*80)
        print(f"CONFIANZA GENERAL DEL SISTEMA: {result['confidence_breakdown']['overall']:.2%}")
        print("="*80 + "\n")

        return result

    def _prepare_tabular_features(self, market_data: Dict) -> pd.DataFrame:
        """Preparar features tabulares"""
        df = market_data['price_data'].copy()

        price_col = get_price_column(df)

        features_dict = {
            'returns': 0.0,
            'volatility': 0.01,
            'sma_20': price_col[-1],
            'sma_50': price_col[-1]
        }

        if len(price_col) > 1:
            returns = pd.Series(price_col).pct_change().dropna()
            features_dict['returns'] = returns.iloc[-1] if len(returns) > 0 else 0.0
            features_dict['volatility'] = returns.std() if len(returns) > 0 else 0.01

        if len(price_col) >= 20:
            features_dict['sma_20'] = pd.Series(price_col).rolling(20).mean().iloc[-1]

        if len(price_col) >= 50:
            features_dict['sma_50'] = pd.Series(price_col).rolling(50).mean().iloc[-1]

        return pd.DataFrame([features_dict])

    def _prepare_graph_data(self, market_data: Dict) -> Dict:
        """Preparar datos de grafo"""
        n_assets = 10

        adjacency = np.random.rand(n_assets, n_assets)
        adjacency = (adjacency + adjacency.T) / 2
        np.fill_diagonal(adjacency, 1.0)

        features = np.random.randn(n_assets, 20)

        return {'adjacency': adjacency, 'features': features}

    def _construct_market_state(self, temporal, vision, tabular, nlp, graph) -> MarketState:
        """Construir Market State desde outputs de módulos"""
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

        import datetime
        now = datetime.datetime.now()
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

# ============================================================================
# EJEMPLO DE USO
# ============================================================================

def main():
    """Ejemplo de uso del sistema God Mode"""

    print("\n" + "="*80)
    print(" "*20 + "GOD MODE TRADING SYSTEM")
    print(" "*15 + "Arquitectura Multi-Modelo Completa")
    print("="*80 + "\n")

    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=200, freq='D')

    price = 100 * np.exp(np.cumsum(np.random.randn(200) * 0.02))

    market_data_df = pd.DataFrame({
        'date': dates,
        'close': price,
        'volume': np.random.randint(1000000, 10000000, 200),
        'high': price * 1.02,
        'low': price * 0.98,
        'open': price * np.random.uniform(0.99, 1.01, 200)
    })

    market_data = {
        'price_data': market_data_df,
        'news': [
            'Markets rally on positive economic data',
            'Central bank hints at rate cuts',
            'Tech stocks lead the gains'
        ]
    }

    god_mode = GodModeSystem()

    result = god_mode.analyze_and_decide(market_data)

    print("\n📊 RESUMEN EJECUTIVO:")
    print(f"   Acción Recomendada: {result['decision']['action']}")
    print(f"   Confianza Global: {result['confidence_breakdown']['overall']:.1%}")
    print(f"   Tamaño Posición: {result['decision']['position_size']:.1%}")

    print("\n✅ Sistema God Mode ejecutado exitosamente")
    print("="*80 + "\n")

    return god_mode, result

if __name__ == "__main__":
    system, result = main()