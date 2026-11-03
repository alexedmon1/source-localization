"""Process pools that do not oversubscribe the machine.

A worker process inherits a numerical-library thread pool (OpenBLAS, MKL, OpenMP) sized to
every core, so ``n`` workers on ``n`` cores run about ``n * n`` threads that compete for the
same cores. In November 2026 an 8-worker run on 10 cores held 88 runnable threads, each
worker's own thread got about 13% of a core, and from outside the slowdown could not be told
apart from a hang.

Every process pool in this package is opened through :func:`process_pool` or :func:`mp_pool`,
which cap each worker at ``cores // workers`` library threads (at least one) before it does
any work. Scripts that open their own pools should do the same.

The cap can change the last bits of a floating-point result, because a BLAS call summing over
fewer threads adds in a different order. It changes nothing larger.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

__all__ = ["THREAD_ENV_VARS", "available_cores", "threads_per_worker", "limit_threads", "process_pool", "mp_pool"]

THREAD_ENV_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
                   "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def available_cores() -> int:
    """Cores this process may run on (its CPU affinity where the platform reports it)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def threads_per_worker(n_workers: int, n_cores: int | None = None) -> int:
    """Library threads each of ``n_workers`` processes may use so the total stays within the cores."""
    n_cores = available_cores() if n_cores is None else max(1, int(n_cores))
    return max(1, n_cores // max(1, int(n_workers)))


def limit_threads(n_threads: int) -> None:
    """Cap this process's numerical-library threads at ``n_threads``, for the rest of its life.

    Sets the environment variables as well, so any process this one starts inherits the cap,
    and applies it to libraries already loaded through ``threadpoolctl``.
    """
    n_threads = max(1, int(n_threads))
    for var in THREAD_ENV_VARS:
        os.environ[var] = str(n_threads)
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=n_threads)


def _init_worker(n_threads, initializer, initargs):
    limit_threads(n_threads)
    if initializer is not None:
        initializer(*initargs)


def process_pool(n_workers: int, *, threads: int | None = None, initializer=None, initargs=(),
                 mp_context=None) -> ProcessPoolExecutor:
    """A :class:`~concurrent.futures.ProcessPoolExecutor` whose workers are thread-capped.

    ``threads`` defaults to :func:`threads_per_worker`; ``initializer`` runs after the cap.
    """
    n_threads = threads_per_worker(n_workers) if threads is None else threads
    return ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_context, initializer=_init_worker,
                               initargs=(n_threads, initializer, tuple(initargs)))


def mp_pool(n_workers: int, *, threads: int | None = None, initializer=None, initargs=(), context: str = "fork"):
    """A :class:`multiprocessing.pool.Pool` (``fork`` by default) whose workers are thread-capped."""
    from multiprocessing import get_context
    n_threads = threads_per_worker(n_workers) if threads is None else threads
    return get_context(context).Pool(n_workers, initializer=_init_worker, initargs=(n_threads, initializer, tuple(initargs)))
