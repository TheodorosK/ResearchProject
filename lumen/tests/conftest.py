import os
import shutil
import sys
import tempfile
from pathlib import Path

# Must be set before lumen is imported: config reads them at import time.
_home = tempfile.mkdtemp(prefix="lumen-test-")
os.environ["LUMEN_HOME"] = _home
os.environ["LUMEN_EMBEDDER"] = "fake"
os.environ["XDG_DATA_HOME"] = os.path.join(_home, "xdg")  # where send2trash puts the Trash on Linux

# Face tests need the two OpenCV models. Point LUMEN_TEST_FACE_MODELS at a folder holding them
# to run those tests offline; without it, face recognition is switched off and they are skipped.
_models = os.environ.get("LUMEN_TEST_FACE_MODELS")
if _models and os.path.isdir(_models):
    shutil.copytree(_models, os.path.join(_home, "models", "faces"))
else:
    os.environ["LUMEN_FACES"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
