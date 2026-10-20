"""pytest configuration: project root first on sys.path (so the local ``datasets``
package beats the HuggingFace one) and synthetic PathMNIST-shaped data by default,
so the whole suite runs offline.  Set PRAA_SYNTHETIC_DATA=0 to use real PathMNIST."""

import os
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent)
if ROOT in sys.path:
    sys.path.remove(ROOT)
sys.path.insert(0, ROOT)
os.environ.setdefault("PRAA_SYNTHETIC_DATA", "1")
