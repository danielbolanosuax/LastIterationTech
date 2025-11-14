# src/features/technical.py
from typing import Protocol
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator
from ta.volatility import AverageTrueRange

class FeatureTransformer(Protocol):
    """Protocolo para transformadores de features."""
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        ...

class TechnicalFeatures:
    """Generador de indicadores técnicos con parámetros configurables."""
    
    def __init__(self, config: dict):
        self.config = config
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Momentum
        df['rsi_14'] = RSIIndicator(df['Close'], window=14).rsi()
        df['rsi_21'] = RSIIndicator(df['Close'], window=21).rsi()
        
        # Trend
        macd = MACD(df['Close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Volatility
        df['atr_14'] = AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
        df['volatility_20'] = df['Close'].pct_change().rolling(20).std()
        
        # Volume
        df['volume_ma_20'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_ma_20']
        
        # Price patterns
        df['returns_1d'] = df['Close'].pct_change()
        df['returns_5d'] = df['Close'].pct_change(5)
        df['returns_20d'] = df['Close'].pct_change(20)
        
        return df

# src/features/pipeline.py
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA

class FeaturePipeline:
    """Pipeline completo de feature engineering."""
    
    def __init__(self, transformers: list[FeatureTransformer]):
        self.transformers = transformers
        self.scaler = RobustScaler()  # Robusto a outliers
        self.pca = None
    
    def fit_transform(self, df: pd.DataFrame, use_pca: bool = False, n_components: int = 0.95) -> pd.DataFrame:
        # Aplicar transformadores
        for transformer in self.transformers:
            df = transformer.transform(df)
        
        # Limpiar
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Escalar
        feature_cols = [c for c in df.columns if c not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        df[feature_cols] = self.scaler.fit_transform(df[feature_cols])
        
        # PCA opcional
        if use_pca:
            self.pca = PCA(n_components=n_components)
            pca_features = self.pca.fit_transform(df[feature_cols])
            pca_df = pd.DataFrame(
                pca_features,
                columns=[f'PC{i+1}' for i in range(pca_features.shape[1])],
                index=df.index
            )
            df = pd.concat([df.drop(columns=feature_cols), pca_df], axis=1)
        
        return df
