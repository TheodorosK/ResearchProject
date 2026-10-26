import os
import sys
import tempfile
from pathlib import Path

# Must be set before lumen is imported: config reads them at import time.
_home = tempfile.mkdtemp(prefix="lumen-test-")
os.environ["LUMEN_HOME"] = _home
os.environ["LUMEN_EMBEDDER"] = "fake"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
