# src/backtesting/engine.py
import pandas as pd
import numpy as np
from typing import Dict
from dataclasses import dataclass

@dataclass
class BacktestResults:
    """Resultados estructurados de backtesting."""
    equity_curve: pd.Series
    trades: pd.DataFrame
    metrics: Dict[str, float]
    
    def plot(self):
        """Genera visualizaciones."""
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Equity curve
        axes[0].plot(self.equity_curve)
        axes[0].set_title('Equity Curve')
        
        # Drawdown
        rolling_max = self.equity_curve.cummax()
        drawdown = (self.equity_curve - rolling_max) / rolling_max
        axes[1].fill_between(drawdown.index, 0, drawdown, alpha=0.3)
        axes[1].set_title('Drawdown')
        
        # Monthly returns
        monthly_returns = self.equity_curve.resample('M').last().pct_change()
        axes[2].bar(monthly_returns.index, monthly_returns)
        axes[2].set_title('Monthly Returns')
        
        plt.tight_layout()
        return fig

class VectorizedBacktest:
    """Motor de backtesting completamente vectorizado."""
    
    def __init__(self, data: pd.DataFrame, config: RiskConfig):
        self.data = data
        self.config = config
    
    def run(self, signals: pd.Series, initial_capital: float = 100000) -> BacktestResults:
        """Ejecuta backtest vectorizado con gestión de riesgo."""
        df = self.data.copy()
        df['signal'] = signals
        df['returns'] = df['Close'].pct_change()
        
        # Position sizing con volatility targeting
        df['volatility'] = df['returns'].rolling(20).std()
        df['position_size'] = (self.config.volatility_target / df['volatility']).clip(0, self.config.max_position_size)
        
        # Aplicar señales
        df['position'] = df['signal'].shift(1) * df['position_size']
        
        # Calcular returns con costos
        df['strategy_returns'] = df['position'] * df['returns']
        df['costs'] = df['position'].diff().abs() * 0.001  # 10 bps
        df['net_returns'] = df['strategy_returns'] - df['costs']
        
        # Equity curve
        equity = (1 + df['net_returns']).cumprod() * initial_capital
        
        # Extraer trades
        trades = self._extract_trades(df)
        
        # Calcular métricas
        metrics = self._calculate_metrics(df['net_returns'], equity)
        
        return BacktestResults(equity_curve=equity, trades=trades, metrics=metrics)
    
    def _calculate_metrics(self, returns: pd.Series, equity: pd.Series) -> Dict[str, float]:
        """Métricas completas de performance."""
        total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
        n_years = len(returns) / 252
        cagr = (1 + total_return) ** (1/n_years) - 1
        
        sharpe = np.sqrt(252) * returns.mean() / returns.std()
        sortino = np.sqrt(252) * returns.mean() / returns[returns < 0].std()
        
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        win_rate = (returns > 0).mean()
        
        return {
            'total_return': total_return,
            'cagr': cagr,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'calmar_ratio': cagr / abs(max_drawdown) if max_drawdown != 0 else 0
        }
