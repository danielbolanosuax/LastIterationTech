# src/core/config.py
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal
from pathlib import Path
import os

class DataConfig(BaseModel):
    """Configuración de fuentes de datos con validación."""
    ticker: str = Field(..., pattern=r'^[A-Z]{1,5}$')
    start_date: str = Field(..., pattern=r'^\d{4}-\d{2}-\d{2}$')
    end_date: Optional[str] = None
    source: Literal['alpha_vantage', 'yahoo', 'polygon'] = 'yahoo'
    cache_dir: Path = Path('data/cache')
    
    @field_validator('cache_dir')
    def create_cache_dir(cls, v):
        v.mkdir(parents=True, exist_ok=True)
        return v

class ModelConfig(BaseModel):
    """Configuración de modelos ML."""
    model_type: Literal['xgboost', 'lstm', 'ensemble'] = 'xgboost'
    window_size: int = Field(60, ge=20, le=252)
    horizon: int = Field(20, ge=1, le=60)
    n_splits: int = Field(5, ge=3, le=10)
    
    # XGBoost específico
    xgb_n_estimators: int = 500
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.03
    xgb_early_stopping_rounds: int = 50
    
    # LSTM específico
    lstm_units: list[int] = [128, 64, 32]
    lstm_dropout: float = 0.2

class RiskConfig(BaseModel):
    """Configuración de gestión de riesgo."""
    max_position_size: float = Field(0.1, ge=0.01, le=0.5)
    stop_loss_atr_mult: float = 2.0
    take_profit_atr_mult: float = 3.0
    max_drawdown_threshold: float = 0.15
    volatility_target: float = 0.15

class TradingSystemConfig(BaseModel):
    """Configuración completa del sistema."""
    data: DataConfig
    model: ModelConfig
    risk: RiskConfig
    
    @classmethod
    def from_yaml(cls, path: Path) -> 'TradingSystemConfig':
        import yaml
        with open(path) as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
