"""Small background jobs the UI can follow (e.g. copying a collection to a folder)."""
import itertools
import os
import shutil
import threading

_jobs = {}
_ids = itertools.count(1)
_lock = threading.Lock()


def get(job_id: int):
    with _lock:
        j = _jobs.get(job_id)
        return dict(j) if j else None


def _update(job_id, **kw):
    with _lock:
        _jobs[job_id].update(kw)


def start(kind: str, total: int, fn) -> int:
    """Run fn(progress) in a thread, where progress(done) reports how far it got."""
    job_id = next(_ids)
    with _lock:
        _jobs[job_id] = {"id": job_id, "kind": kind, "state": "running", "done": 0, "total": total,
                         "error": None, "result": None}

    def run():
        try:
            result = fn(lambda done: _update(job_id, done=done))
            _update(job_id, state="done", done=total, result=result)
        except Exception as e:
            _update(job_id, state="failed", error=f"{type(e).__name__}: {e}")

    threading.Thread(target=run, name=f"lumen-job-{job_id}", daemon=True).start()
    return job_id


def unique_name(dest_dir: str, name: str, taken: set) -> str:
    """A filename in dest_dir that collides neither with existing files nor with ones we just wrote."""
    stem, ext = os.path.splitext(name)
    candidate, n = name, 1
    while candidate.lower() in taken or os.path.exists(os.path.join(dest_dir, candidate)):
        candidate = f"{stem} ({n}){ext}"
        n += 1
    taken.add(candidate.lower())
    return candidate


def copy_files(paths, dest_dir: str):
    """Start copying originals (with their dates) into dest_dir. Never overwrites anything."""
    os.makedirs(dest_dir, exist_ok=True)

    def work(progress):
        taken, copied, missing = set(), 0, 0
        for i, p in enumerate(paths, 1):
            if os.path.exists(p):
                shutil.copy2(p, os.path.join(dest_dir, unique_name(dest_dir, os.path.basename(p), taken)))
                copied += 1
            else:
                missing += 1
            progress(i)
        return {"copied": copied, "missing": missing, "dest": dest_dir}

    return start("copy", len(paths), work)
