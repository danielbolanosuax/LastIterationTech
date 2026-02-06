#!/bin/bash
# God Mode Trading - Script de ejecución rápida

# Verificar si venv existe
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment no encontrado"
    echo "Ejecuta primero: bash install.sh"
    exit 1
fi

# Activar venv
source venv/bin/activate

# Verificar argumentos
if [ $# -eq 0 ]; then
    echo "================================================================================"
    echo "       🚀 GOD MODE TRADING - MENÚ DE EJECUCIÓN 🚀"
    echo "================================================================================"
    echo ""
    echo "Uso: bash run.sh [opción]"
    echo ""
    echo "Opciones:"
    echo "  1, pipeline    - Ejecutar pipeline principal (una vez)"
    echo "  2, loop        - Ejecutar pipeline en loop continuo"
    echo "  3, quick       - Ejecutar quickstart demo"
    echo "  4, dashboard   - Lanzar dashboard web"
    echo "  5, alerts      - Activar sistema de alertas"
    echo "  6, backtest    - Ejecutar backtest"
    echo ""
    echo "Ejemplos:"
    echo "  bash run.sh pipeline"
    echo "  bash run.sh loop"
    echo "  bash run.sh dashboard"
    echo ""
    exit 0
fi

case $1 in
    1|pipeline)
        echo "🚀 Ejecutando pipeline principal..."
        python main.py --symbols AAPL MSFT GOOGL TSLA NVDA
        ;;

    2|loop)
        echo "🔄 Ejecutando pipeline en loop continuo (1 hora)..."
        python main.py --symbols AAPL MSFT GOOGL TSLA NVDA --loop --interval 3600
        ;;

    3|quick|quickstart)
        echo "⚡ Ejecutando quick start..."
        python quickstart.py
        ;;

    4|dashboard|dash)
        echo "📊 Lanzando dashboard..."
        streamlit run dashboard.py
        ;;

    5|alerts)
        echo "🔔 Activando sistema de alertas..."
        python alerts.py --symbols AAPL MSFT TSLA NVDA --interval 60
        ;;

    6|backtest)
        echo "📈 Ejecutando backtest..."
        python main.py --symbols AAPL --backtest AAPL
        ;;

    *)
        echo "❌ Opción no válida: $1"
        echo "Ejecuta 'bash run.sh' para ver opciones disponibles"
        exit 1
        ;;
esac
