# src/data/providers/base.py
from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd
from datetime import datetime, timedelta
import hashlib
import pickle
from pathlib import Path

class DataProvider(ABC):
    """Interfaz base para proveedores de datos."""
    
    def __init__(self, cache_dir: Path, cache_ttl_hours: int = 24):
        self.cache_dir = cache_dir
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
    
    @abstractmethod
    def fetch(self, ticker: str, start: str, end: Optional[str]) -> pd.DataFrame:
        """Obtiene datos OHLCV."""
        pass
    
    def get_with_cache(self, ticker: str, start: str, end: Optional[str]) -> pd.DataFrame:
        """Wrapper con cache inteligente."""
        cache_key = self._generate_cache_key(ticker, start, end)
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        
        # Verificar cache válido
        if cache_path.exists():
            cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
            if cache_age < self.cache_ttl:
                return pd.read_pickle(cache_path)
        
        # Fetch y guardar
        df = self.fetch(ticker, start, end)
        df.to_pickle(cache_path)
        return df
    
    def _generate_cache_key(self, ticker: str, start: str, end: Optional[str]) -> str:
        key_str = f"{self.__class__.__name__}_{ticker}_{start}_{end}"
        return hashlib.md5(key_str.encode()).hexdigest()

# src/data/providers/yahoo.py
import yfinance as yf
from .base import DataProvider

class YahooProvider(DataProvider):
    """Proveedor Yahoo Finance con retry y rate limiting."""
    
    def fetch(self, ticker: str, start: str, end: Optional[str]) -> pd.DataFrame:
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(start=start, end=end, auto_adjust=True)
        
        if df.empty:
            raise ValueError(f"No data returned for {ticker}")
        
        return df[['Open', 'High', 'Low', 'Close', 'Volume']].rename(
            columns=str.title
        )

# src/data/cache.py
from functools import wraps
import redis
import json
from typing import Callable

class RedisCache:
    """Cache distribuido con Redis para producción."""
    
    def __init__(self, host: str = 'localhost', port: int = 6379):
        self.client = redis.Redis(host=host, port=port, decode_responses=True)
    
    def cache_dataframe(self, ttl: int = 3600):
        """Decorator para cachear DataFrames."""
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                cache_key = f"{func.__name__}:{args}:{kwargs}"
                
                # Intentar leer del cache
                cached = self.client.get(cache_key)
                if cached:
                    return pd.read_json(cached)
                
                # Ejecutar función y cachear
                result = func(*args, **kwargs)
                self.client.setex(cache_key, ttl, result.to_json())
                return result
            return wrapper
        return decorator
