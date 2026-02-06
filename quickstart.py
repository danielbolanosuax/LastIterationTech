#!/usr/bin/env python3
"""
God Mode Trading - Quick Start Script
Ejecuta una demo completa del sistema
"""

from god_mode_complete import GodModeComplete

def main():
    print("""
    ╔═══════════════════════════════════════════════════════╗
    ║     🚀 GOD MODE TRADING SYSTEM - QUICK START 🚀      ║
    ╚═══════════════════════════════════════════════════════╝
    """)

    print("Inicializando sistema...\n")
    god_mode = GodModeComplete()

    # Demo 1: Análisis Simple
    print("\n" + "="*80)
    print("DEMO 1: ANÁLISIS DE SÍMBOLO")
    print("="*80)

    result = god_mode.analyze_symbol('AAPL', period="1mo")

    print(f"\n📊 Resultado:")
    print(f"   Símbolo: {result['symbol']}")
    print(f"   Precio: ${result['current_price']:.2f}")
    print(f"   Señal: {result['signal'].action}")
    print(f"   Confianza: {result['confidence_breakdown']['overall']:.1%}")

    # Demo 2: Comparación
    print("\n" + "="*80)
    print("DEMO 2: COMPARACIÓN DE SÍMBOLOS")
    print("="*80)

    symbols = ['AAPL', 'MSFT', 'GOOGL']
    comparison = god_mode.scan_multiple(symbols, top_n=3)

    # Demo 3: Portfolio
    print("\n" + "="*80)
    print("DEMO 3: ESTADO DEL PORTFOLIO")
    print("="*80)

    portfolio = god_mode.get_portfolio_report()

    print("\n\n✅ Quick Start Completado!")
    print("\nPróximos pasos:")
    print("  1. Ejecutar dashboard: streamlit run dashboard.py")
    print("  2. Configurar alertas: python alerts.py --symbols AAPL MSFT")
    print("  3. Leer documentación: README.md")

    return god_mode

if __name__ == "__main__":
    system = main()
