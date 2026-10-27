"""In-memory matrix of every ready photo's embedding, for fast cosine search."""
import threading

import numpy as np

from . import db


class VectorIndex:
    def __init__(self):
        self._lock = threading.RLock()
        self.ids = np.zeros(0, dtype=np.int64)
        self.mat = np.zeros((0, 0), dtype=np.float32)
        self._pending_ids = []
        self._pending_vecs = []
        self.version = 0

    def load(self, model_key: str):
        rows = db.q(
            "SELECT id, embedding FROM photos WHERE status='ready' AND model=? AND embedding IS NOT NULL",
            (model_key,),
        )
        with self._lock:
            self._pending_ids, self._pending_vecs = [], []
            if rows:
                self.ids = np.array([r["id"] for r in rows], dtype=np.int64)
                self.mat = np.stack(
                    [np.frombuffer(r["embedding"], dtype=np.float16) for r in rows]
                ).astype(np.float32)
            else:
                self.ids = np.zeros(0, dtype=np.int64)
                self.mat = np.zeros((0, 0), dtype=np.float32)
            self.version += 1

    def add(self, ids, vecs):
        with self._lock:
            self._pending_ids.append(np.asarray(ids, dtype=np.int64))
            self._pending_vecs.append(np.asarray(vecs, dtype=np.float16).astype(np.float32))
            self.version += 1

    def remove(self, ids):
        ids = np.asarray(list(ids), dtype=np.int64)
        if not len(ids):
            return
        with self._lock:
            self._consolidate()
            keep = ~np.isin(self.ids, ids)
            self.ids, self.mat = self.ids[keep], self.mat[keep]
            self.version += 1

    def _consolidate(self):
        if not self._pending_ids:
            return
        new_ids = np.concatenate(self._pending_ids)
        new_vecs = np.concatenate(self._pending_vecs)
        self._pending_ids, self._pending_vecs = [], []
        # Re-embedded photos replace their old vector.
        keep = ~np.isin(self.ids, new_ids)
        if self.mat.size:
            self.ids = np.concatenate([self.ids[keep], new_ids])
            self.mat = np.concatenate([self.mat[keep], new_vecs])
        else:
            self.ids, self.mat = new_ids, new_vecs

    def snapshot(self):
        """(ids, matrix) consistent with each other; callers must not mutate them."""
        with self._lock:
            self._consolidate()
            return self.ids, self.mat

    def vector_of(self, photo_id: int):
        ids, mat = self.snapshot()
        hit = np.nonzero(ids == photo_id)[0]
        return mat[hit[0]] if len(hit) else None

    def __len__(self):
        ids, _ = self.snapshot()
        return len(ids)


index = VectorIndex()
