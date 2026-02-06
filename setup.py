#!/usr/bin/env python3
"""
God Mode Trading - Setup Automático
Configuración inicial del sistema
"""

import os
import sys
import subprocess
from pathlib import Path

def print_header(text):
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80)

def run_command(cmd, check=True):
    """Ejecutar comando del sistema"""
    try:
        result = subprocess.run(cmd, shell=True, check=check, 
                              capture_output=True, text=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        return False

def check_python_version():
    """Verificar versión de Python"""
    print_header("Verificando Python")

    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Se requiere Python 3.8 o superior")
        return False

    print("✅ Versión de Python OK")
    return True

def create_directories():
    """Crear estructura de directorios"""
    print_header("Creando Directorios")

    dirs = ['data', 'data/cache', 'logs', 'models']

    for dir_path in dirs:
        Path(dir_path).mkdir(exist_ok=True, parents=True)
        print(f"  ✅ {dir_path}/")

    return True

def install_dependencies():
    """Instalar dependencias"""
    print_header("Instalando Dependencias")

    print("Esto puede tomar varios minutos...\n")

    # Actualizar pip
    print("📦 Actualizando pip...")
    run_command(f"{sys.executable} -m pip install --upgrade pip", check=False)

    # Instalar requirements
    print("📦 Instalando paquetes...")

    if Path("requirements.txt").exists():
        success = run_command(f"{sys.executable} -m pip install -r requirements.txt")

        if success:
            print("\n✅ Todas las dependencias instaladas")
            return True
        else:
            print("\n⚠ Hubo algunos errores, pero continuando...")
            return True
    else:
        print("❌ No se encontró requirements.txt")
        return False

def setup_env_file():
    """Configurar archivo .env"""
    print_header("Configurando Variables de Entorno")

    if Path(".env").exists():
        print("⚠ .env ya existe")
        overwrite = input("¿Sobrescribir? (y/n): ").lower()
        if overwrite != 'y':
            return True

    # Copiar ejemplo
    if Path(".env.example").exists():
        with open(".env.example", 'r') as f:
            content = f.read()

        with open(".env", 'w') as f:
            f.write(content)

        print("✅ .env creado desde .env.example")
        print("\nℹ️  Edita .env para agregar tus API keys:")
        print("   - ALPHA_VANTAGE_KEY (ya incluida)")
        print("   - ALPACA_API_KEY (opcional)")
        print("   - ALERT_SENDER_EMAIL (opcional)")

        return True
    else:
        print("❌ No se encontró .env.example")
        return False

def test_imports():
    """Probar imports principales"""
    print_header("Probando Imports")

    required_modules = [
        'numpy',
        'pandas',
        'requests',
        'statsmodels',
        'sklearn',
        'torch',
        'streamlit'
    ]

    failed = []

    for module in required_modules:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except ImportError:
            print(f"  ❌ {module}")
            failed.append(module)

    if failed:
        print(f"\n⚠ Módulos faltantes: {', '.join(failed)}")
        print("   Ejecuta: pip install " + " ".join(failed))
        return False

    print("\n✅ Todos los módulos disponibles")
    return True

def run_quick_test():
    """Ejecutar test rápido del sistema"""
    print_header("Test del Sistema")

    try:
        print("Importando sistema...")
        from god_mode_complete import GodModeComplete, Config

        print("✅ Import exitoso")

        print("\nInicializando sistema...")
        god_mode = GodModeComplete()

        print("✅ Inicialización exitosa")

        print("\n✅ Sistema funcionando correctamente!")
        return True

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nVerifica que test_performance.py esté en el mismo directorio")
        return False

def main():
    """Setup principal"""

    print("""
    ╔═══════════════════════════════════════════════════════╗
    ║       🚀 GOD MODE TRADING - SETUP WIZARD 🚀          ║
    ╚═══════════════════════════════════════════════════════╝
    """)

    steps = [
        ("Verificar Python", check_python_version),
        ("Crear Directorios", create_directories),
        ("Instalar Dependencias", install_dependencies),
        ("Configurar .env", setup_env_file),
        ("Probar Imports", test_imports),
        ("Test del Sistema", run_quick_test)
    ]

    results = []

    for step_name, step_func in steps:
        try:
            success = step_func()
            results.append((step_name, success))

            if not success:
                print(f"\n⚠ {step_name} falló pero continuando...")
        except Exception as e:
            print(f"\n❌ Error en {step_name}: {str(e)}")
            results.append((step_name, False))

    # Resumen
    print("\n" + "="*80)
    print("RESUMEN DE INSTALACIÓN")
    print("="*80)

    for step_name, success in results:
        status = "✅" if success else "❌"
        print(f"  {status} {step_name}")

    all_success = all(success for _, success in results)

    if all_success:
        print("\n" + "="*80)
        print("🎉 ¡INSTALACIÓN COMPLETADA EXITOSAMENTE!")
        print("="*80)
        print("\n🚀 Próximos pasos:\n")
        print("  1. Ejecutar demo:")
        print("     python quickstart.py\n")
        print("  2. Lanzar dashboard:")
        print("     streamlit run dashboard.py\n")
        print("  3. Activar alertas:")
        print("     python alerts.py --symbols AAPL MSFT\n")
        print("  4. Leer documentación:")
        print("     cat README.md\n")
    else:
        print("\n" + "="*80)
        print("⚠️  INSTALACIÓN COMPLETADA CON WARNINGS")
        print("="*80)
        print("\nRevisar los errores arriba y solucionarlos manualmente.")
        print("\nPara ayuda, consulta README.md o corre:")
        print("  python setup.py")

if __name__ == "__main__":
    main()
