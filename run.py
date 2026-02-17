#!/usr/bin/env python3
"""
OmniTranscribe 启动脚本
从项目根目录运行此脚本来启动 OmniTranscribe

Usage:
    python run.py              # 启动命令行界面
    python run.py --gui        # 启动图形界面 (GUI)
    python run.py --gui --share  # 启动GUI并创建公共链接
"""

import sys
import os
from pathlib import Path

# 添加 src 目录到 Python 路径
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Load environment variables from .env file (for run.py itself)
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

# 导入并运行 main
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OmniTranscribe - 多语言音频转录与翻译工具")
    parser.add_argument("--gui", action="store_true", help="启动图形界面 (GUI)")
    parser.add_argument("--share", action="store_true", help="创建公共链接 (仅用于GUI模式)")

    # Parse known args to allow other args to pass through to main
    args, remaining = parser.parse_known_args()

    if args.gui:
        # Launch GUI
        from src import gui
        sys.argv = ["gui.py"] + (["--share"] if args.share else [])
        gui.main()
    else:
        # Launch CLI
        from src import main
        sys.argv = ["main.py"] + remaining
        main()
