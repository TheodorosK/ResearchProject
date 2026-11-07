"""CPU multiprocessing with the ``spawn`` start method (never fork an initialised TensorFlow runtime).

The process that OWNS the pool never imports TensorFlow; every worker imports it fresh, hides the GPU, and is limited to
one intra-op and one inter-op thread so W workers use W cores (no oversubscription).  The parent polls the summed worker
RSS against an explicit cap and terminates the whole pool on any limit violation.  GPU multiprocessing is intentionally
NOT supported: several processes would compete for the same Metal device and unified memory.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time
from typing import Any, Callable, Dict, List, Optional

GB = 1024 ** 3
_STATE: Dict[str, Any] = {}


# ------------------------------------------------------------ functions executed INSIDE worker processes
def init_worker(grappler: str, threads: int, settings_dict: Dict[str, Any]) -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    from .. import tfenv
    from .workload import Settings, dataset_for
    tf = tfenv.configure("cpu", "strict", grappler)          # hides any GPU before TensorFlow initialises
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    st = Settings.from_dict(settings_dict)
    _STATE.update(tf=tf, st=st, data=dataset_for(st))


def _evaluate(payload: Dict[str, Any]) -> Dict[str, Any]:
    import psutil
    from .execute import eval_individual
    from .workload import genome_from_dict
    g = genome_from_dict(payload["genome"])
    r = eval_individual(_STATE["tf"], "cpu", g, _STATE["data"], _STATE["st"], payload["steps"])
    r["pid"] = os.getpid()
    r["rss_gb"] = psutil.Process().memory_info().rss / GB
    return r


def warm_task(i: int) -> int:
    from . import workload as W
    st = _STATE["st"]
    g = W.initial_population(W.Point("small", 2, 2, "hetero", 1), st)[0]
    _evaluate({"genome": W.genome_to_dict(g), "steps": 2})
    return os.getpid()


def eval_task(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _evaluate(payload)


# ---------------------------------------------------------------------------------------- parent side
class MPEvaluator:
    def __init__(self, workers: int, settings, rss_cap_total_gb: float, threads_per_worker: int = 1):
        import psutil
        self._psutil = psutil
        self.workers, self.rss_cap_gb = workers, rss_cap_total_gb
        self.threads = threads_per_worker
        ctx = mp.get_context("spawn")
        t = time.perf_counter()
        self.pool = ctx.Pool(workers, initializer=init_worker, initargs=(settings.grappler, threads_per_worker, settings.as_dict()))
        self.pool.map(warm_task, range(workers * 3), chunksize=1)
        self.startup_s = time.perf_counter() - t
        self._last_rss_check = 0.0
        self.peak_children_rss_gb = 0.0

    def children_rss_gb(self) -> float:
        try:
            kids = self._psutil.Process().children(recursive=True)
            return sum(k.memory_info().rss for k in kids) / GB
        except Exception:
            return 0.0

    def evaluate(self, genomes, steps: int, check: Callable[[], None]) -> List[Dict[str, Any]]:
        from .workload import genome_to_dict
        from ..limits import LimitExceeded
        asyncs = [self.pool.apply_async(eval_task, ({"genome": genome_to_dict(g), "steps": steps},)) for g in genomes]
        results: List[Optional[Dict[str, Any]]] = [None] * len(asyncs)
        try:
            while True:
                pending = False
                for i, a in enumerate(asyncs):
                    if results[i] is None:
                        if a.ready():
                            results[i] = a.get()
                        else:
                            pending = True
                if not pending:
                    return results                                    # type: ignore[return-value]
                check()
                now = time.monotonic()
                if now - self._last_rss_check > 0.5:
                    self._last_rss_check = now
                    rss = self.children_rss_gb()
                    self.peak_children_rss_gb = max(self.peak_children_rss_gb, rss)
                    if rss > self.rss_cap_gb:
                        raise LimitExceeded(f"summed worker RSS {rss:.2f} GB > cap {self.rss_cap_gb:.2f} GB")
                time.sleep(0.005)
        except BaseException:
            self.close()                                              # never leave workers running after an abort
            raise

    def close(self) -> None:
        try:
            self.pool.terminate()
            self.pool.join()
        except Exception:
            pass


PER_WORKER_GB = 0.75      # budget per worker (TF runtime + model + data); measured children were ~0.4 GB each on small models


def max_workers_by_memory(max_extra_gb: float, per_worker_gb: float = PER_WORKER_GB, requested: int = 4) -> int:
    """Conservative worker cap: each worker is budgeted ``per_worker_gb`` and workers may use at most half the allowance.
    (The summed worker RSS is additionally checked at run time against 80 % of the allowance.)"""
    return max(1, min(requested, int((max_extra_gb * 0.5) // per_worker_gb)))
