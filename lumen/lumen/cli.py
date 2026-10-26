import argparse
import threading
import webbrowser


def main():
    parser = argparse.ArgumentParser(prog="lumen", description="Search your photos by describing them.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    args = parser.parse_args()

    import uvicorn

    url = f"http://127.0.0.1:{args.port}"
    if not args.no_browser:
        threading.Timer(1.5, webbrowser.open, (url,)).start()
    print(f"\n  ✨ Lumen is running at {url}  (Ctrl+C to stop)\n")
    uvicorn.run("lumen.server:app", host="127.0.0.1", port=args.port, log_level="warning")
