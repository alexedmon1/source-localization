"""Process pools cap each worker's numerical-library threads (utils/parallel.py)."""

import os

import numpy as np
import pytest

from source_localization.utils import parallel as P


def test_threads_per_worker_divides_the_cores():
    assert P.threads_per_worker(8, n_cores=10) == 1
    assert P.threads_per_worker(2, n_cores=10) == 5
    assert P.threads_per_worker(3, n_cores=10) == 3
    assert P.threads_per_worker(1, n_cores=10) == 10


def test_threads_per_worker_never_below_one():
    assert P.threads_per_worker(16, n_cores=10) == 1
    assert P.threads_per_worker(0, n_cores=10) == 10
    assert P.threads_per_worker(4, n_cores=0) == 1


def _report(_=None):
    from threadpoolctl import threadpool_info
    np.dot(np.ones((4, 4)), np.ones((4, 4)))           # make sure BLAS is loaded
    return {v: os.environ.get(v) for v in P.THREAD_ENV_VARS}, [i["num_threads"] for i in threadpool_info()]


def _blas_threads_in_parent():
    from threadpoolctl import threadpool_info
    np.dot(np.ones((4, 4)), np.ones((4, 4)))
    return [i["num_threads"] for i in threadpool_info()]


def _mark(value):
    os.environ["SL_TEST_INIT"] = value


def _read_mark(_=None):
    return os.environ.get("SL_TEST_INIT")


@pytest.mark.parametrize("threads", [1, 2])
def test_process_pool_caps_workers(threads):
    with P.process_pool(2, threads=threads) as ex:
        env, pools = ex.submit(_report).result()
    assert set(env.values()) == {str(threads)}
    assert pools and all(n <= threads for n in pools)


@pytest.mark.parametrize("threads", [1, 2])
def test_mp_pool_caps_workers(threads):
    with P.mp_pool(2, threads=threads) as pool:
        out = pool.map(_report, range(2))
    for env, pools in out:
        assert set(env.values()) == {str(threads)}
        assert pools and all(n <= threads for n in pools)


def test_default_cap_is_cores_over_workers():
    with P.process_pool(2) as ex:
        env, _ = ex.submit(_report).result()
    assert env["OPENBLAS_NUM_THREADS"] == str(P.threads_per_worker(2))


def test_initializer_runs_after_the_cap():
    with P.process_pool(1, threads=1, initializer=_mark, initargs=("ran",)) as ex:
        assert ex.submit(_read_mark).result() == "ran"
    with P.mp_pool(1, threads=1, initializer=_mark, initargs=("ran",)) as pool:
        assert pool.map(_read_mark, [0]) == ["ran"]


def test_parent_is_left_alone():
    before = _blas_threads_in_parent()
    with P.process_pool(2, threads=1) as ex:
        ex.submit(_report).result()
    assert _blas_threads_in_parent() == before
