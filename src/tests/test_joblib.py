# ruff: noqa E402
import os
import sys

os.environ["RAY_DEDUP_LOGS"] = "0"
this_dir = os.path.dirname(__file__)
sys.path.append(this_dir)

import joblib
import pytest
from conftest import return_big_object  # type: ignore


@pytest.mark.parametrize(
    "backend",
    [
        "serial",
        "loky",
        "multiprocessing",
        "ray",
        # "dask",
    ],
)
def test_joblib(backend: str, set_ray_env: None) -> None:
    n_jobs, return_as, n = -1, "generator", 50
    print(backend)
    if backend == "serial":
        backend, n_jobs, n = "loky", 1, 10
    elif backend in ("multiprocessing", "ray"):
        return_as = "list"

    if backend == "ray":
        os.environ["RAY_DEDUP_LOGS"] = "0"
        print("BBBB")
        import ray
        from ray.util.joblib import register_ray

        ray.init(
            runtime_env={
                "env_vars": {
                    "PYTHONPATH": this_dir,
                }
            }
        )
        register_ray()
        print("Initialize & Register Ray.")

    print("AAA")
    with joblib.parallel_config(backend=backend, n_jobs=n_jobs):
        for res in joblib.Parallel(return_as=return_as)(
            joblib.delayed(return_big_object)(i)  #
            for i in range(n)
        ):
            print(res)
