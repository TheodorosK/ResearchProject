"""Live folder watching: new, changed or deleted photos are picked up within seconds."""
import os
import threading

from . import config, db

DEBOUNCE_SECONDS = 3.0


class FolderWatcher:
    def __init__(self, on_change):
        self.on_change = on_change          # called with a set of folder ids
        self._observer = None
        self._watches = {}                  # folder id -> watch handle
        self._pending = set()
        self._timer = None
        self._lock = threading.Lock()
        self.state = "off"

    def start(self):
        try:
            from watchdog.observers import Observer
        except ImportError:
            self.state = "unavailable (pip install watchdog)"
            return
        self._observer = Observer()
        self._observer.daemon = True
        self._observer.start()
        self.state = "on"
        self.sync()

    def sync(self):
        """Watch exactly the folders in the library."""
        if self._observer is None:
            return
        from watchdog.events import FileSystemEventHandler

        watcher = self

        class Handler(FileSystemEventHandler):
            def __init__(self, folder_id):
                self.folder_id = folder_id

            def on_any_event(self, event):
                paths = [getattr(event, "src_path", ""), getattr(event, "dest_path", "") or ""]
                if any(os.path.splitext(str(p))[1].lower() in config.MEDIA_EXTENSIONS for p in paths) \
                        or event.is_directory:
                    watcher._changed(self.folder_id)

        folders = {r["id"]: r["path"] for r in db.q("SELECT id, path FROM folders")}
        with self._lock:
            for fid in list(self._watches):
                if fid not in folders:
                    try:
                        self._observer.unschedule(self._watches.pop(fid))
                    except Exception:
                        pass
            for fid, path in folders.items():
                if fid not in self._watches and os.path.isdir(path):
                    try:
                        self._watches[fid] = self._observer.schedule(Handler(fid), path, recursive=True)
                    except Exception:  # e.g. too many inotify watches, or a network share
                        pass

    def _changed(self, folder_id):
        with self._lock:
            self._pending.add(folder_id)
            if self._timer:
                self._timer.cancel()
            # Wait for a quiet moment: copying 500 photos shouldn't trigger 500 scans.
            self._timer = threading.Timer(DEBOUNCE_SECONDS, self._fire)
            self._timer.daemon = True
            self._timer.start()

    def _fire(self):
        with self._lock:
            pending, self._pending = self._pending, set()
        if pending:
            self.on_change(pending)

    def watching(self):
        return len(self._watches)
