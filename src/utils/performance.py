# src/utils/performance.py
from numba import jit, prange
import numpy as np

@jit(nopython=True)
def fast_rolling_sharpe(returns: np.ndarray, window: int = 252) -> np.ndarray:
    """
    Cálculo ultra-rápido de Sharpe ratio.
    nopython=True significa que no usa el intérprete de Python (más rápido).
    """
    n = len(returns)
    sharpe = np.zeros(n)
    
    for i in range(window, n):
        window_returns = returns[i-window:i]
        mean_return = np.mean(window_returns)
        std_return = np.std(window_returns)
        
        if std_return > 0:
            sharpe[i] = np.sqrt(252) * mean_return / std_return
        else:
            sharpe[i] = 0.0
    
    return sharpe

@jit(nopython=True, parallel=True)
def fast_rolling_metrics(prices: np.ndarray, window: int = 20) -> tuple:
    """
    Calcula múltiples métricas rolling en paralelo.
    parallel=True usa múltiples cores de CPU.
    """
    n = len(prices)
    rolling_mean = np.zeros(n)
    rolling_std = np.zeros(n)
    rolling_max = np.zeros(n)
    
    for i in prange(window, n):  # prange = parallel range
        window_data = prices[i-window:i]
        rolling_mean[i] = np.mean(window_data)
        rolling_std[i] = np.std(window_data)
        rolling_max[i] = np.max(window_data)
    
    return rolling_mean, rolling_std, rolling_max

@jit(nopython=True)
def fast_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI optimizado con Numba."""
    n = len(prices)
    rsi = np.zeros(n)
    
    # Calcular cambios de precio
    deltas = np.diff(prices)
    
    for i in range(period, n):
        window = deltas[i-period:i]
        gains = window[window > 0]
        losses = -window[window < 0]
        
        avg_gain = np.mean(gains) if len(gains) > 0 else 0.0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0.0
        
        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi
