# src/models/base_model.py
from abc import ABC, abstractmethod
from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import joblib

class BaseModel(ABC):
    """Clase base para todos los modelos."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.feature_importance_ = None
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaseModel':
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        pass
    
    def walk_forward_validation(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Walk-forward validation con TimeSeriesSplit."""
        tscv = TimeSeriesSplit(n_splits=self.config.n_splits)
        predictions = np.zeros(len(y))
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            self.fit(X_train, y_train)
            predictions[val_idx] = self.predict_proba(X_val)[:, 1]
        
        return predictions, y
    
    def save(self, path: Path):
        joblib.dump(self, path)
    
    @classmethod
    def load(cls, path: Path) -> 'BaseModel':
        return joblib.load(path)

# src/models/xgboost_model.py
import xgboost as xgb
from .base_model import BaseModel

class XGBoostModel(BaseModel):
    """XGBoost con early stopping y GPU support."""
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'XGBoostModel':
        self.model = xgb.XGBClassifier(
            n_estimators=self.config.xgb_n_estimators,
            max_depth=self.config.xgb_max_depth,
            learning_rate=self.config.xgb_learning_rate,
            tree_method='hist',  # GPU: 'gpu_hist'
            early_stopping_rounds=self.config.xgb_early_stopping_rounds,
            eval_metric='auc',
            random_state=42
        )
        
        # Split para early stopping
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        self.feature_importance_ = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)

# src/models/ensemble.py
from typing import List
import numpy as np
from .base_model import BaseModel

class EnsembleModel:
    """Ensemble de modelos con voting estratégico."""
    
    def __init__(self, models: List[BaseModel], weights: Optional[List[float]] = None):
        self.models = models
        self.weights = weights or [1/len(models)] * len(models)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        predictions = np.array([
            model.predict_proba(X)[:, 1] * weight
            for model, weight in zip(self.models, self.weights)
        ])
        return predictions.sum(axis=0) / sum(self.weights)
    
    def calibrate_weights(self, X_val: pd.DataFrame, y_val: pd.Series):
        """Optimiza pesos basándose en performance de validación."""
        from scipy.optimize import minimize
        
        def objective(weights):
            pred = np.average(
                [m.predict_proba(X_val)[:, 1] for m in self.models],
                weights=weights,
                axis=0
            )
            return -roc_auc_score(y_val, pred)
        
        result = minimize(
            objective,
            x0=self.weights,
            bounds=[(0, 1)] * len(self.models),
            constraints={'type': 'eq', 'fun': lambda w: sum(w) - 1}
        )
        self.weights = result.x
