# test_performance.py
import numpy as np
import time
from utils.performance import fast_rolling_sharpe

# Generar datos de prueba
returns = np.random.randn(10000) * 0.01

# Versión normal (Python puro)
def slow_rolling_sharpe(returns, window=252):
    n = len(returns)
    sharpe = np.zeros(n)
    for i in range(window, n):
        window_returns = returns[i-window:i]
        sharpe[i] = np.sqrt(252) * np.mean(window_returns) / np.std(window_returns)
    return sharpe

# Benchmark versión lenta
start = time.time()
result_slow = slow_rolling_sharpe(returns)
time_slow = time.time() - start

# Benchmark versión rápida (primera llamada compila)
start = time.time()
result_fast = fast_rolling_sharpe(returns)
time_fast_first = time.time() - start

# Segunda llamada (ya compilado)
start = time.time()
result_fast = fast_rolling_sharpe(returns)
time_fast = time.time() - start

print(f"Versión lenta: {time_slow:.4f}s")
print(f"Versión rápida (primera llamada): {time_fast_first:.4f}s")
print(f"Versión rápida (compilada): {time_fast:.4f}s")
print(f"Speedup: {time_slow/time_fast:.1f}x más rápido")
