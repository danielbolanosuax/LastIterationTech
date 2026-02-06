#!/bin/bash
# God Mode Trading System - Instalación Automática con Virtual Environment

set -e  # Exit on error

echo "================================================================================"
echo "       🚀 GOD MODE TRADING - INSTALACIÓN AUTOMÁTICA 🚀"
echo "================================================================================"
echo ""

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Verificar Python
echo "[1/6] Verificando Python..."
if ! command -v python3 &> /dev/null; then
    echo "${RED}❌ Python3 no encontrado${NC}"
    echo "Instala Python3: sudo apt install python3 python3-venv python3-pip"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo "${GREEN}✅ $PYTHON_VERSION${NC}"

# Crear virtual environment
echo ""
echo "[2/6] Creando virtual environment..."
if [ -d "venv" ]; then
    echo "${YELLOW}⚠️  venv ya existe, eliminando...${NC}"
    rm -rf venv
fi

python3 -m venv venv
echo "${GREEN}✅ Virtual environment creado${NC}"

# Activar venv
echo ""
echo "[3/6] Activando virtual environment..."
source venv/bin/activate
echo "${GREEN}✅ Activado: $(which python)${NC}"

# Actualizar pip
echo ""
echo "[4/6] Actualizando pip..."
pip install --upgrade pip > /dev/null 2>&1
echo "${GREEN}✅ pip actualizado: $(pip --version)${NC}"

# Instalar dependencias
echo ""
echo "[5/6] Instalando dependencias..."
echo "     (Esto puede tomar varios minutos...)"

# Instalar en orden para evitar conflictos
echo "     📦 Instalando paquetes básicos..."
pip install numpy pandas scipy requests python-dateutil pytz > /dev/null 2>&1

echo "     📦 Instalando ML frameworks..."
pip install scikit-learn lightgbm catboost > /dev/null 2>&1

echo "     📦 Instalando statsmodels y arch..."
pip install statsmodels arch > /dev/null 2>&1

echo "     📦 Instalando PyTorch..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu > /dev/null 2>&1

echo "     📦 Instalando transformers y NLP..."
pip install transformers nltk spacy > /dev/null 2>&1

echo "     📦 Instalando visualization..."
pip install plotly matplotlib seaborn > /dev/null 2>&1

echo "     📦 Instalando APIs..."
pip install yfinance schedule python-dotenv > /dev/null 2>&1

echo "     📦 Instalando Streamlit..."
pip install streamlit > /dev/null 2>&1

echo "     📦 Instalando Alpaca..."
pip install alpaca-py > /dev/null 2>&1

echo "     📦 Instalando utils finales..."
pip install tqdm networkx > /dev/null 2>&1

echo "${GREEN}✅ Todas las dependencias instaladas${NC}"

# Crear estructura de directorios
echo ""
echo "[6/6] Creando estructura de directorios..."
mkdir -p data/cache
mkdir -p logs
mkdir -p models
echo "${GREEN}✅ Directorios creados${NC}"

# Configurar .env
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "${GREEN}✅ .env creado desde .env.example${NC}"
    fi
fi

# Resumen
echo ""
echo "================================================================================"
echo "                   ✅ INSTALACIÓN COMPLETADA"
echo "================================================================================"
echo ""
echo "📝 Próximos pasos:"
echo ""
echo "   1. Activar el entorno virtual:"
echo "      ${GREEN}source venv/bin/activate${NC}"
echo ""
echo "   2. Ejecutar el pipeline principal:"
echo "      ${GREEN}python main.py --symbols AAPL MSFT GOOGL${NC}"
echo ""
echo "   3. Ejecutar quick start:"
echo "      ${GREEN}python quickstart.py${NC}"
echo ""
echo "   4. Lanzar dashboard:"
echo "      ${GREEN}streamlit run dashboard.py${NC}"
echo ""
echo "   5. Activar alertas:"
echo "      ${GREEN}python alerts.py --symbols AAPL MSFT --interval 60${NC}"
echo ""
echo "   6. Pipeline en loop continuo:"
echo "      ${GREEN}python main.py --loop --interval 3600${NC}"
echo ""
echo "================================================================================"
echo "⚠️  IMPORTANTE: Siempre activar venv antes de ejecutar:"
echo "   ${YELLOW}source venv/bin/activate${NC}"
echo "================================================================================"
echo ""
