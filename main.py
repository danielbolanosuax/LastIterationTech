#!/usr/bin/env python3
"""
God Mode Trading - MAIN PIPELINE
Ejecuta el sistema completo de forma automatizada
"""

import os
import sys
import time
import argparse
from datetime import datetime, timedelta
from typing import List

# Añadir directorio actual al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def configure_console_output():
    """Evitar caídas por encoding en consolas no UTF-8."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding='utf-8', errors='replace')
            except Exception:
                pass


configure_console_output()


def get_action_emoji(action: str) -> str:
    """Obtener emoji de accion sin lanzar errores por acciones desconocidas."""
    return {'BUY': '🟢', 'SELL': '🔴', 'HOLD': '🟡'}.get(action, '⚪')

def print_banner():
    """Banner del sistema"""
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║        🚀 GOD MODE TRADING SYSTEM - AUTO PIPELINE 🚀         ║
    ║                                                               ║
    ║              Production-Ready Trading Bot v3.0                ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    print(f"    🕐 Iniciado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("    " + "="*60)

def check_dependencies():
    """Verificar dependencias instaladas"""
    print("\n[STEP 1] Verificando dependencias...\n")

    required = ['numpy', 'pandas', 'requests', 'torch', 'streamlit']
    missing = []

    for module in required:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except ImportError:
            print(f"  ❌ {module}")
            missing.append(module)

    if missing:
        print(f"\n⚠️  Módulos faltantes: {', '.join(missing)}")
        print("\n💡 Ejecuta primero:")
        print("   bash install.sh")
        print("   source venv/bin/activate")
        return False

    print("\n✅ Todas las dependencias OK")
    return True

def load_system():
    """Cargar el sistema God Mode"""
    print("\n[STEP 2] Cargando sistema God Mode...")

    try:
        from god_mode_complete import GodModeComplete

        print("  ✅ Módulos importados")
        print("  🔧 Inicializando sistema...")

        god_mode = GodModeComplete()

        print("  ✅ Sistema inicializado correctamente\n")
        return god_mode

    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        return None

def analyze_watchlist(god_mode, symbols: List[str], execute_trades: bool = False):
    """Analizar lista de símbolos"""
    print(f"\n[STEP 3] Analizando watchlist ({len(symbols)} símbolos)...")
    print(f"  Ejecución de trades: {'✅ HABILITADO' if execute_trades else '❌ DESHABILITADO'}")
    print("  " + "-"*60)

    results = []

    for i, symbol in enumerate(symbols, 1):
        print(f"\n  [{i}/{len(symbols)}] Procesando {symbol}...")

        try:
            result = god_mode.analyze_symbol(
                symbol, 
                period="3mo", 
                execute_trade=execute_trades
            )

            signal = result['signal']
            confidence = result['confidence_breakdown']['overall']

            # Emoji según acción
            action_emoji = get_action_emoji(signal.action)

            print(f"       {action_emoji} {signal.action} @ ${result['current_price']:.2f}")
            print(f"       Confianza: {confidence:.1%}")
            print(f"       Position: {signal.position_size:.4f} (${signal.position_size * signal.price:,.2f})")

            results.append({
                'symbol': symbol,
                'action': signal.action,
                'price': result['current_price'],
                'confidence': confidence,
                'signal': signal,
                'result': result
            })

            # Rate limiting
            if i < len(symbols):
                time.sleep(2)

        except Exception as e:
            print(f"       ❌ Error: {str(e)[:100]}")
            continue

    return results

def generate_report(results: List[dict], god_mode):
    """Generar reporte final"""
    print("\n" + "="*80)
    print("[STEP 4] REPORTE FINAL")
    print("="*80)

    if not results:
        print("  ⚠️ No hay resultados para mostrar")
        return

    # Tabla de señales
    print("\n📊 SEÑALES GENERADAS:")
    print("-"*80)
    print(f"{'Symbol':<10} {'Action':<8} {'Price':<12} {'Confidence':<12} {'Position':<15}")
    print("-"*80)

    for r in results:
        signal = r['signal']
        action_emoji = get_action_emoji(r['action'])

        print(f"{r['symbol']:<10} {action_emoji} {r['action']:<6} "
              f"${r['price']:<10.2f} {r['confidence']:<11.1%} "
              f"{signal.position_size:.4f}")

    print("-"*80)

    # Resumen por acción
    buy_count = sum(1 for r in results if r['action'] == 'BUY')
    sell_count = sum(1 for r in results if r['action'] == 'SELL')
    hold_count = sum(1 for r in results if r['action'] == 'HOLD')

    print(f"\n📈 RESUMEN:")
    print(f"  🟢 BUY:  {buy_count}")
    print(f"  🔴 SELL: {sell_count}")
    print(f"  🟡 HOLD: {hold_count}")

    # Top oportunidades
    sorted_results = sorted(results, key=lambda x: x['confidence'], reverse=True)

    print(f"\n⭐ TOP 3 OPORTUNIDADES:")
    for i, r in enumerate(sorted_results[:3], 1):
        print(f"  {i}. {r['symbol']}: {r['action']} (Confianza: {r['confidence']:.1%})")

    # Portfolio status
    print("\n💼 ESTADO DEL PORTFOLIO:")
    portfolio = god_mode.get_portfolio_report()

    # Guardar reporte en archivo
    save_report_to_file(results, portfolio)

def save_report_to_file(results: List[dict], portfolio: dict):
    """Guardar reporte en archivo"""
    import json
    from pathlib import Path

    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = logs_dir / f"pipeline_report_{timestamp}.json"

    report_data = {
        'timestamp': timestamp,
        'num_symbols': len(results),
        'signals': [
            {
                'symbol': r['symbol'],
                'action': r['action'],
                'price': float(r['price']),
                'confidence': float(r['confidence']),
                'position_size': float(r['signal'].position_size),
                'stop_loss': float(r['signal'].stop_loss),
                'take_profit': float(r['signal'].take_profit)
            }
            for r in results
        ],
        'portfolio': {
            'cash': float(portfolio['cash']),
            'total_value': float(portfolio['total_value']),
            'num_positions': portfolio['positions'],
            'num_trades': portfolio['num_trades']
        }
    }

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2)

    print(f"\n📝 Reporte guardado: {filename}")

def run_backtest(god_mode, symbol: str):
    """Ejecutar backtest"""
    print(f"\n[OPTIONAL] Ejecutando backtest para {symbol}...")

    try:
        results = god_mode.backtest(symbol)

        print(f"\n📊 RESULTADOS BACKTEST {symbol}:")
        print(f"  Retorno:      {results['total_return_pct']:+.2f}%")
        print(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {results['max_drawdown_pct']:.2f}%")
        print(f"  Win Rate:     {results['win_rate']:.1f}%")
        print(f"  Num Trades:   {results['num_trades']}")

        return results

    except Exception as e:
        print(f"  ❌ Error: {str(e)[:100]}")
        return None

def main():
    """Pipeline principal"""

    # Parsear argumentos
    parser = argparse.ArgumentParser(description='God Mode Trading - Auto Pipeline')

    parser.add_argument(
        '--symbols',
        nargs='+',
        default=['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'],
        help='Símbolos a analizar'
    )

    parser.add_argument(
        '--execute',
        action='store_true',
        help='Ejecutar trades reales (paper trading)'
    )

    parser.add_argument(
        '--backtest',
        type=str,
        default=None,
        help='Ejecutar backtest para un símbolo'
    )

    parser.add_argument(
        '--loop',
        action='store_true',
        help='Ejecutar en loop continuo'
    )

    parser.add_argument(
        '--interval',
        type=int,
        default=3600,
        help='Intervalo en segundos para loop (default: 3600 = 1 hora)'
    )

    parser.add_argument(
        '--retry-delay',
        type=int,
        default=60,
        help='Segundos de espera antes de reintentar tras un error en modo loop'
    )

    args = parser.parse_args()

    if args.interval <= 0:
        print("❌ --interval debe ser mayor a 0")
        sys.exit(1)

    if args.retry_delay <= 0:
        print("❌ --retry-delay debe ser mayor a 0")
        sys.exit(1)

    # Banner
    print_banner()

    # Verificar dependencias
    if not check_dependencies():
        sys.exit(1)

    # Cargar sistema
    god_mode = load_system()

    if god_mode is None:
        print("\n❌ Error cargando el sistema. Abortando...")
        sys.exit(1)

    # Función para ejecutar un ciclo completo
    def run_cycle():
        # Analizar watchlist
        results = analyze_watchlist(god_mode, args.symbols, args.execute)

        # Generar reporte
        generate_report(results, god_mode)

        # Backtest opcional
        if args.backtest:
            run_backtest(god_mode, args.backtest)

    # Ejecutar
    if args.loop:
        print(f"\n🔄 MODO LOOP ACTIVADO")
        print(f"   Intervalo: {args.interval} segundos ({args.interval/60:.0f} minutos)")
        print(f"   Reintento ante error: {args.retry_delay} segundos")
        print(f"   Presiona Ctrl+C para detener\n")

        cycle_count = 0
        consecutive_failures = 0

        try:
            while True:
                cycle_count += 1

                print(f"\n{'='*80}")
                print(f"CICLO #{cycle_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'='*80}")

                try:
                    run_cycle()
                    consecutive_failures = 0
                except Exception as e:
                    consecutive_failures += 1
                    print(f"\n❌ Error en ciclo #{cycle_count}: {str(e)[:200]}")
                    print(f"   Fallos consecutivos: {consecutive_failures}")
                    retry_at = datetime.now() + timedelta(seconds=args.retry_delay)
                    print(f"   Reintentando a las {retry_at.strftime('%H:%M:%S')}\n")
                    time.sleep(args.retry_delay)
                    continue

                print(f"\n⏰ Próximo ciclo en {args.interval/60:.0f} minutos...")
                print(f"   Esperando hasta {(datetime.now() + timedelta(seconds=args.interval)).strftime('%H:%M:%S')}\n")

                time.sleep(args.interval)

        except KeyboardInterrupt:
            print("\n\n🛑 Loop detenido por usuario")
            print(f"   Ciclos completados: {cycle_count}")

    else:
        # Ejecución única
        try:
            run_cycle()
        except Exception as e:
            print(f"\n❌ Error en ejecución única: {str(e)[:200]}")
            sys.exit(1)

    # Final
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETADO")
    print("="*80)
    print(f"\n🕐 Finalizado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

if __name__ == "__main__":
    main()
