"""Pytest bootstrap: ensure the ``src`` layout is importable.

Allows ``import meridian`` to resolve when the package has not been installed
in editable mode (for example, a fresh checkout running ``pytest`` directly).
"""

import os
import sys

_SRC_DIR = os.path.join(os.path.dirname(__file__), "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
