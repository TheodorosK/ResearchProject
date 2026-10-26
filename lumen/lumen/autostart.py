"""'Start Lumen with my computer': a login item for macOS, Windows and Linux desktops."""
import os
import platform
import sys
from pathlib import Path

LABEL = "com.lumen.photos"
SOURCE_DIR = Path(__file__).resolve().parents[1]   # where `python -m lumen` works from


def _command():
    if getattr(sys, "frozen", False):  # the packaged desktop app
        return [sys.executable, "--no-browser"]
    exe = Path(sys.executable)
    if platform.system() == "Windows" and (exe.parent / "pythonw.exe").exists():
        exe = exe.parent / "pythonw.exe"  # no console window
    return [str(exe), "-m", "lumen", "--no-browser"]


def _path():
    system = platform.system()
    if system == "Darwin":
        return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"
    if system == "Windows":
        return Path(os.environ.get("APPDATA", Path.home())) / "Microsoft" / "Windows" / "Start Menu" / \
            "Programs" / "Startup" / "Lumen.vbs"
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "autostart" / "lumen.desktop"


def is_enabled() -> bool:
    return _path().exists()


def set_enabled(on: bool):
    path = _path()
    if not on:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = _command()
    system = platform.system()
    if system == "Darwin":
        args = "".join(f"<string>{a}</string>" for a in cmd)
        path.write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>{LABEL}</string>
  <key>ProgramArguments</key><array>{args}</array>
  <key>WorkingDirectory</key><string>{SOURCE_DIR}</string>
  <key>RunAtLoad</key><true/>
</dict></plist>
""")
    elif system == "Windows":
        line = " ".join(f'""{a}""' for a in cmd)
        path.write_text(f'Set sh = CreateObject("WScript.Shell")\nsh.CurrentDirectory = "{SOURCE_DIR}"\n'
                        f'sh.Run "{line}", 0, False\n')
    else:
        exec_line = " ".join(f'"{a}"' for a in cmd)
        path.write_text(f"[Desktop Entry]\nType=Application\nName=Lumen\nComment=Search your photos by describing them\n"
                        f"Exec={exec_line}\nPath={SOURCE_DIR}\nX-GNOME-Autostart-enabled=true\n")
