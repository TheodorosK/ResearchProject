import argparse
import os
import sys
import threading
import webbrowser


def main():
    frozen = getattr(sys, "frozen", False)
    parser = argparse.ArgumentParser(prog="lumen", description="Search your photos by describing them.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    parser.add_argument("--lan", action="store_true",
                        help="also open Lumen to phones on your Wi-Fi (asks for a PIN; port + 1)")
    parser.add_argument("--tray", action=argparse.BooleanOptionalAction, default=frozen,
                        help="show a menu-bar/tray icon (default in the desktop app)")
    args = parser.parse_args()

    os.environ["LUMEN_PORT"] = str(args.port)
    import uvicorn

    if args.lan:
        from . import db, lan

        db.init()
        db.set_setting("lan", "1")
        if not lan.pin_is_set():
            print(f"\n  📱 Phone access PIN: {lan.set_pin()}  (change it in Sources → Phone access)")

    url = f"http://127.0.0.1:{args.port}"
    if not args.no_browser:
        threading.Timer(1.5, webbrowser.open, (url,)).start()
    print(f"\n  ✨ Lumen is running at {url}  (Ctrl+C to stop)\n")
    if args.lan:
        from . import lan

        print(f"  📱 On your phone (same Wi-Fi): http://{lan.lan_ip()}:{args.port + 1}\n")

    from .server import app

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=args.port, log_level="warning"))
    if args.tray:
        try:
            from . import tray

            threading.Thread(target=server.run, daemon=True).start()
            tray.run(url, on_quit=lambda: setattr(server, "should_exit", True))
            return
        except Exception as e:  # no tray available (e.g. a headless Linux box): carry on without it
            print(f"  (no tray icon: {e})")
    server.run()
