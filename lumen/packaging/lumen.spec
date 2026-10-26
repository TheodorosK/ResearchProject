# PyInstaller spec for the Lumen desktop app.  Build from the `lumen/` folder:
#     pyinstaller packaging/lumen.spec
# Models are not bundled: they are downloaded on first run into ~/.lumen/models.
import sys

from PyInstaller.utils.hooks import collect_all, collect_submodules

datas = [("../lumen/static", "lumen/static")]
binaries = []
hiddenimports = collect_submodules("uvicorn") + collect_submodules("lumen") + ["pystray"]
for pkg in ("fastembed", "rapidocr_onnxruntime", "reverse_geocoder", "pillow_heif", "av", "onnxruntime", "segno"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

a = Analysis(["launch.py"], pathex=[".."], datas=datas, binaries=binaries, hiddenimports=hiddenimports,
             excludes=["tkinter", "matplotlib", "torch", "pytest", "skimage"])
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="Lumen", console=False, icon=None)
coll = COLLECT(exe, a.binaries, a.datas, name="Lumen")
if sys.platform == "darwin":
    app = BUNDLE(coll, name="Lumen.app", bundle_identifier="com.lumen.photos",
                 info_plist={"LSUIElement": True, "CFBundleShortVersionString": "0.1.0"})
