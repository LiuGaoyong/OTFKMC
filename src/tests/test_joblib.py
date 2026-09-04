import os
import sys
from typing import Literal

os.environ["RAY_DEDUP_LOGS"] = "0"
this_dir = os.path.dirname(__file__)
sys.path.append(this_dir)

import joblib  # noqa: E402
import pytest  # noqa: E402
from conftest import return_big_object  # type: ignore # noqa: E402


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
def test_joblib(backend: str) -> None:
    n_jobs, return_as, n = -1, "generator", 150
    if backend == "serial":
        backend, n_jobs, n = "loky", 1, 10
    elif backend in ("multiprocessing", "ray"):
        return_as = "list"

    if backend == "ray":
        import ray

        ray.init(
            runtime_env={
                "env_vars": {
                    "PYTHONPATH": this_dir,
                }
            }
        )

    joblib_register(backend)

    with joblib.parallel_config(backend=backend, n_jobs=n_jobs):
        for res in joblib.Parallel(return_as=return_as)(
            joblib.delayed(return_big_object)(i)  #
            for i in range(n)
        ):
            print(res)


def joblib_register(
    backend: str | Literal["ray", "dask", "daskmpi"],
) -> None:
    if backend.lower() == "ray":
        from ray.util.joblib import register_ray

        register_ray(), joblib  # type: ignore
    elif backend.lower() == "daskmpi":
        raise NotImplementedError
    else:
        return
