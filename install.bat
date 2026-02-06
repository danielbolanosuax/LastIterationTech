@echo off
REM God Mode Trading System - Instalación para Windows

echo ================================================================================
echo        GOD MODE TRADING - INSTALACION AUTOMATICA (Windows)
echo ================================================================================
echo.

REM Verificar Python
echo [1/6] Verificando Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python no encontrado
    echo Descarga Python desde: https://www.python.org/downloads/
    pause
    exit /b 1
)
python --version
echo OK: Python encontrado
echo.

REM Crear virtual environment
echo [2/6] Creando virtual environment...
if exist venv (
    echo Eliminando venv anterior...
    rmdir /s /q venv
)
python -m venv venv
echo OK: Virtual environment creado
echo.

REM Activar venv
echo [3/6] Activando virtual environment...
call venv\Scripts\activate.bat
echo OK: Activado
echo.

REM Actualizar pip
echo [4/6] Actualizando pip...
python -m pip install --upgrade pip --quiet
echo OK: pip actualizado
echo.

REM Instalar dependencias
echo [5/6] Instalando dependencias (esto tomara varios minutos)...
echo      Instalando paquetes basicos...
pip install numpy pandas scipy requests python-dateutil pytz --quiet

echo      Instalando ML frameworks...
pip install scikit-learn lightgbm catboost --quiet

echo      Instalando statsmodels...
pip install statsmodels arch --quiet

echo      Instalando PyTorch (CPU)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet

echo      Instalando NLP...
pip install transformers nltk spacy --quiet

echo      Instalando visualization...
pip install plotly matplotlib seaborn --quiet

echo      Instalando APIs...
pip install yfinance schedule python-dotenv --quiet

echo      Instalando Streamlit...
pip install streamlit --quiet

echo      Instalando Alpaca...
pip install alpaca-py --quiet

echo      Instalando utils...
pip install tqdm networkx --quiet

echo OK: Todas las dependencias instaladas
echo.

REM Crear directorios
echo [6/6] Creando directorios...
if not exist data mkdir data
if not exist data\cache mkdir data\cache
if not exist logs mkdir logs
if not exist models mkdir models
echo OK: Directorios creados
echo.

REM Configurar .env
if not exist .env (
    if exist .env.example (
        copy .env.example .env >nul
        echo OK: .env creado
    )
)

echo ================================================================================
echo                    INSTALACION COMPLETADA
echo ================================================================================
echo.
echo Proximos pasos:
echo.
echo    1. Activar el entorno virtual:
echo       venv\Scripts\activate
echo.
echo    2. Ejecutar el pipeline:
echo       python main.py --symbols AAPL MSFT
echo.
echo    3. Quick start:
echo       python quickstart.py
echo.
echo    4. Dashboard:
echo       streamlit run dashboard.py
echo.
echo ================================================================================
echo IMPORTANTE: Siempre activar venv antes de ejecutar
echo             venv\Scripts\activate
echo ================================================================================
echo.
pause
