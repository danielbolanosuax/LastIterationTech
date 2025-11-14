# src/deployment/monitor.py
from prometheus_client import Counter, Histogram, Gauge
import logging
from typing import Dict
import smtplib
from email.mime.text import MIMEText

class PerformanceMonitor:
    """Monitoreo de performance en producción."""
    
    def __init__(self):
        # Métricas Prometheus
        self.trades_counter = Counter('trades_total', 'Total number of trades')
        self.pnl_gauge = Gauge('current_pnl', 'Current P&L')
        self.latency_histogram = Histogram('prediction_latency_seconds', 'Model prediction latency')
        
        self.logger = logging.getLogger(__name__)
    
    def log_trade(self, trade_data: Dict):
        self.trades_counter.inc()
        self.logger.info(f"Trade executed: {trade_data}")
    
    def check_drift(self, current_features: pd.DataFrame, reference_features: pd.DataFrame):
        """Detecta drift en distribución de features."""
        from scipy.stats import ks_2samp
        
        drift_detected = {}
        for col in current_features.columns:
            stat, p_value = ks_2samp(current_features[col], reference_features[col])
            if p_value < 0.05:
                drift_detected[col] = p_value
        
        if drift_detected:
            self.send_alert(f"Feature drift detected: {drift_detected}")
    
    def send_alert(self, message: str):
        """Envía alertas por email/Slack."""
        # Implementación de alertas
        self.logger.warning(f"ALERT: {message}")
