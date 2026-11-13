#!/usr/bin/env python3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omnitranscribe.product_gui import ProductGUI


if __name__ == "__main__":
    ProductGUI().launch_desktop()
