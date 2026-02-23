#!/usr/bin/env python3
"""
Supervisor 24/7 para relanzar main.py automáticamente.
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import List
from watchlists import TOP_50_SYMBOLS


def configure_console_output():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def build_main_command(args) -> List[str]:
    cmd = [
        sys.executable,
        "main.py",
        "--loop",
        "--interval",
        str(args.interval),
        "--retry-delay",
        str(args.retry_delay),
        "--symbols",
        *args.symbols,
    ]

    if args.execute:
        cmd.append("--execute")

    if args.backtest:
        cmd.extend(["--backtest", args.backtest])

    return cmd


def main():
    configure_console_output()

    parser = argparse.ArgumentParser(description="Supervisor 24/7 para main.py")
    parser.add_argument("--symbols", nargs="+", default=TOP_50_SYMBOLS)
    parser.add_argument("--interval", type=int, default=3600)
    parser.add_argument("--retry-delay", type=int, default=60)
    parser.add_argument("--restart-delay", type=int, default=15)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--backtest", type=str, default=None)
    args = parser.parse_args()

    if args.interval <= 0 or args.retry_delay <= 0 or args.restart_delay <= 0:
        print("❌ interval, retry-delay y restart-delay deben ser mayores a 0")
        sys.exit(1)

    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    restart_count = 0
    print("🛡️ Supervisor 24/7 iniciado")
    print(f"🕐 Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Presiona Ctrl+C para detener supervisor y bot.\n")

    try:
        while True:
            cmd = build_main_command(args)
            print(f"🚀 Lanzando: {' '.join(cmd)}")

            result = subprocess.run(cmd, check=False)
            exit_code = result.returncode

            # Códigos típicos de interrupción manual
            if exit_code in (0, 130, -2):
                print(f"\n🛑 Bot detenido (exit code {exit_code}). Supervisor finalizado.")
                break

            restart_count += 1
            next_retry = datetime.now().timestamp() + args.restart_delay
            retry_time = datetime.fromtimestamp(next_retry).strftime("%H:%M:%S")
            print(f"\n❌ Bot terminó con exit code {exit_code}")
            print(f"🔁 Reinicio #{restart_count} en {args.restart_delay}s (a las {retry_time})\n")
            time.sleep(args.restart_delay)

    except KeyboardInterrupt:
        print("\n🛑 Supervisor detenido por usuario")


if __name__ == "__main__":
    main()
