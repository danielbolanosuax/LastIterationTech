
"""
God Mode Alert System
Sistema de alertas automáticas con schedule
"""

import schedule
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import os
from typing import List

from god_mode_complete import GodModeComplete, Config, TradeSignal

# ============================================================================
# CONFIGURACIÓN DE ALERTAS
# ============================================================================

class AlertConfig:
    """Configuración de alertas"""

    # Email
    SMTP_SERVER = "smtp.gmail.com"
    SMTP_PORT = 587
    SENDER_EMAIL = os.getenv("ALERT_SENDER_EMAIL", "")
    SENDER_PASSWORD = os.getenv("ALERT_SENDER_PASSWORD", "")
    RECEIVER_EMAIL = os.getenv("ALERT_RECEIVER_EMAIL", "")

    # Telegram (opcional)
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

    # Thresholds
    CONFIDENCE_THRESHOLD = 0.85
    PRICE_CHANGE_THRESHOLD = 5.0  # %

# ============================================================================
# ALERT MANAGER
# ============================================================================

class AlertManager:
    """Gestor de alertas"""

    def __init__(self, god_mode: GodModeComplete):
        self.god_mode = god_mode
        self.sent_alerts = set()  # Evitar alertas duplicadas

    def send_email(self, subject: str, body: str):
        """Enviar alerta por email"""

        if not AlertConfig.SENDER_EMAIL or not AlertConfig.RECEIVER_EMAIL:
            print("  ⚠ Email no configurado")
            return False

        try:
            msg = MIMEMultipart()
            msg['From'] = AlertConfig.SENDER_EMAIL
            msg['To'] = AlertConfig.RECEIVER_EMAIL
            msg['Subject'] = subject

            msg.attach(MIMEText(body, 'html'))

            server = smtplib.SMTP(AlertConfig.SMTP_SERVER, AlertConfig.SMTP_PORT)
            server.starttls()
            server.login(AlertConfig.SENDER_EMAIL, AlertConfig.SENDER_PASSWORD)

            server.send_message(msg)
            server.quit()

            print(f"  ✅ Email enviado: {subject}")
            return True

        except Exception as e:
            print(f"  ❌ Error enviando email: {str(e)[:100]}")
            return False

    def send_telegram(self, message: str):
        """Enviar alerta por Telegram"""

        if not AlertConfig.TELEGRAM_BOT_TOKEN or not AlertConfig.TELEGRAM_CHAT_ID:
            print("  ⚠ Telegram no configurado")
            return False

        try:
            import requests

            url = f"https://api.telegram.org/bot{AlertConfig.TELEGRAM_BOT_TOKEN}/sendMessage"
            data = {
                'chat_id': AlertConfig.TELEGRAM_CHAT_ID,
                'text': message,
                'parse_mode': 'HTML'
            }

            response = requests.post(url, data=data)

            if response.status_code == 200:
                print(f"  ✅ Telegram enviado")
                return True
            else:
                print(f"  ❌ Telegram error: {response.status_code}")
                return False

        except Exception as e:
            print(f"  ❌ Error Telegram: {str(e)[:100]}")
            return False

    def format_signal_email(self, signal: TradeSignal, analysis: dict) -> str:
        """Formatear señal para email HTML"""

        action_emoji = {'BUY': '🟢', 'SELL': '🔴', 'HOLD': '🟡'}[signal.action]
        action_color = {'BUY': '#00FF00', 'SELL': '#FF0000', 'HOLD': '#FFA500'}[signal.action]

        html = f"""
        <html>
        <body style="font-family: Arial, sans-serif;">
            <h2 style="color: {action_color};">
                {action_emoji} SEÑAL DE TRADING: {signal.action}
            </h2>

            <h3>Símbolo: {signal.symbol}</h3>

            <table style="border-collapse: collapse; width: 100%;">
                <tr style="background-color: #f2f2f2;">
                    <td style="padding: 10px; border: 1px solid #ddd;"><b>Precio</b></td>
                    <td style="padding: 10px; border: 1px solid #ddd;">${signal.price:.2f}</td>
                </tr>
                <tr>
                    <td style="padding: 10px; border: 1px solid #ddd;"><b>Confianza</b></td>
                    <td style="padding: 10px; border: 1px solid #ddd;">{signal.confidence:.1%}</td>
                </tr>
                <tr style="background-color: #f2f2f2;">
                    <td style="padding: 10px; border: 1px solid #ddd;"><b>Position Size</b></td>
                    <td style="padding: 10px; border: 1px solid #ddd;">{signal.position_size:.4f} (${signal.position_size * signal.price:,.2f})</td>
                </tr>
                <tr>
                    <td style="padding: 10px; border: 1px solid #ddd;"><b>Stop Loss</b></td>
                    <td style="padding: 10px; border: 1px solid #ddd;">{signal.stop_loss:.1%}</td>
                </tr>
                <tr style="background-color: #f2f2f2;">
                    <td style="padding: 10px; border: 1px solid #ddd;"><b>Take Profit</b></td>
                    <td style="padding: 10px; border: 1px solid #ddd;">{signal.take_profit:.1%}</td>
                </tr>
            </table>

            <h3>Análisis</h3>
            <p>{signal.reasoning}</p>

            <h4>Confianza por Módulo:</h4>
            <ul>
                <li>Temporal: {analysis['confidence_breakdown']['temporal']:.1%}</li>
                <li>Visión: {analysis['confidence_breakdown']['vision']:.1%}</li>
                <li>Tabular: {analysis['confidence_breakdown']['tabular']:.1%}</li>
                <li>NLP: {analysis['confidence_breakdown']['nlp']:.1%}</li>
                <li>Grafos: {analysis['confidence_breakdown']['graph']:.1%}</li>
            </ul>

            <p><small>Generado por God Mode Trading System - {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</small></p>
        </body>
        </html>
        """

        return html

    def format_signal_telegram(self, signal: TradeSignal) -> str:
        """Formatear señal para Telegram"""

        action_emoji = {'BUY': '🟢', 'SELL': '🔴', 'HOLD': '🟡'}[signal.action]

        message = f"""
{action_emoji} <b>SEÑAL {signal.action}</b>

<b>{signal.symbol}</b>
💰 Precio: ${signal.price:.2f}
📊 Confianza: {signal.confidence:.1%}
📈 Position: {signal.position_size:.4f}
🛑 Stop Loss: {signal.stop_loss:.1%}
🎯 Take Profit: {signal.take_profit:.1%}

⏰ {signal.timestamp.strftime('%H:%M:%S')}
"""

        return message

    def check_and_alert(self, symbols: List[str]):
        """Verificar símbolos y enviar alertas si aplica"""

        print(f"\n🔔 Verificando alertas para {len(symbols)} símbolos...")
        print(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        for symbol in symbols:
            try:
                # Evitar alertas duplicadas recientes
                alert_key = f"{symbol}_{datetime.now().strftime('%Y%m%d%H')}"
                if alert_key in self.sent_alerts:
                    continue

                # Analizar símbolo
                result = self.god_mode.analyze_symbol(symbol, period="1mo")
                signal = result['signal']

                # Verificar si cumple criterios para alerta
                should_alert = (
                    signal.confidence >= AlertConfig.CONFIDENCE_THRESHOLD and
                    signal.action in ['BUY', 'SELL'] and
                    abs(result['price_change']) >= AlertConfig.PRICE_CHANGE_THRESHOLD
                )

                if should_alert:
                    print(f"\n  🚨 ALERTA para {symbol}:")
                    print(f"     Acción: {signal.action}")
                    print(f"     Confianza: {signal.confidence:.1%}")
                    print(f"     Cambio: {result['price_change']:+.2f}%")

                    # Enviar por email
                    subject = f"🚨 {signal.action} Signal: {symbol} (Confidence {signal.confidence:.0%})"
                    body = self.format_signal_email(signal, result)
                    self.send_email(subject, body)

                    # Enviar por Telegram
                    telegram_msg = self.format_signal_telegram(signal)
                    self.send_telegram(telegram_msg)

                    # Marcar como enviado
                    self.sent_alerts.add(alert_key)

            except Exception as e:
                print(f"  ❌ Error en {symbol}: {str(e)[:100]}")
                continue

        print(f"  ✅ Verificación completada\n")

# ============================================================================
# SCHEDULER
# ============================================================================

def start_alert_scheduler(symbols: List[str], interval_minutes: int = 60):
    """Iniciar scheduler de alertas"""

    print("\n" + "="*80)
    print("🔔 SISTEMA DE ALERTAS INICIADO")
    print("="*80)
    print(f"  Símbolos: {', '.join(symbols)}")
    print(f"  Intervalo: {interval_minutes} minutos")
    print(f"  Threshold confianza: {AlertConfig.CONFIDENCE_THRESHOLD:.0%}")
    print(f"  Threshold cambio precio: {AlertConfig.PRICE_CHANGE_THRESHOLD}%")
    print("="*80)

    # Inicializar sistema
    god_mode = GodModeComplete()
    alert_manager = AlertManager(god_mode)

    # Programar checks
    def job():
        alert_manager.check_and_alert(symbols)

    # Primera ejecución inmediata
    job()

    # Programar ejecuciones periódicas
    schedule.every(interval_minutes).minutes.do(job)

    print(f"\n⏰ Scheduler activo. Próximo check en {interval_minutes} minutos...")
    print("   Presiona Ctrl+C para detener\n")

    # Loop infinito
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check cada minuto
    except KeyboardInterrupt:
        print("\n\n🛑 Scheduler detenido")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='God Mode Alert System')
    parser.add_argument('--symbols', nargs='+', default=['AAPL', 'MSFT', 'TSLA', 'NVDA'],
                       help='Símbolos a monitorear')
    parser.add_argument('--interval', type=int, default=60,
                       help='Intervalo en minutos entre checks')

    args = parser.parse_args()

    start_alert_scheduler(args.symbols, args.interval)
