#!/usr/bin/env python3
"""OmniTranscribe launcher.

Usage:
    python run.py                 # CLI
    python run.py --gui           # product workspace in browser
    python run.py --desktop       # product workspace in native desktop window
    python run.py --legacy-gui    # previous one-shot Gradio GUI
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

try:
    from dotenv import load_dotenv
    env_path = ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OmniTranscribe - 多语言音频转录与翻译工具")
    parser.add_argument("--gui", action="store_true", help="启动新的项目式 GUI")
    parser.add_argument("--desktop", action="store_true", help="在原生桌面窗口中启动产品界面")
    parser.add_argument("--legacy-gui", action="store_true", help="启动旧版一次性处理 GUI")
    parser.add_argument("--share", action="store_true", help="创建 Gradio 公共链接（仅浏览器 GUI）")
    args, remaining = parser.parse_known_args()

    if args.desktop:
        from product_gui import ProductGUI
        ProductGUI().launch_desktop()
    elif args.gui:
        from product_gui import ProductGUI
        ProductGUI().launch(share=args.share)
    elif args.legacy_gui:
        from src import gui
        sys.argv = ["gui.py"] + (["--share"] if args.share else [])
        gui.main()
    else:
        from src import main
        sys.argv = ["main.py"] + remaining
        main()
