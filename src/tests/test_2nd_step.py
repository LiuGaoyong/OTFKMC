from pathlib import Path

import numpy as np
import pytest
from ase.build import fcc111
from graphatoms.system import Cluster, Gas, System
from hydra import compose, initialize

from otfkmc.abc import FirstStep
from otfkmc.config import CONFIG_DIR, Config
from otfkmc.exploration import Exploration as SecondStep
from otfkmc.exploration._0base import CheckMinimaFailed, OptimizationFailed

this_dir = Path(__file__).parent


class Mock(SecondStep, FirstStep):
    pass

def _with_energetics(
    cluster: Cluster,
    *,
    fmax: float = 0.01,
    frequencies: np.ndarray,
) -> Cluster:
    return cluster.model_copy(
        update={
            "energy": -1.0,
            "fmax": fmax,
            "frequencies": frequencies,
        }
    )


@pytest.fixture
def explorer(tmp_path: Path) -> Mock:
    with initialize(
        config_path=CONFIG_DIR.relative_to(this_dir, walk_up=True).as_posix(),
        job_name="config",
    ):
        cfg: Config = compose(  # type: ignore
            config_name="run",
            overrides=["calculator=emt", "atoms=octahedron"],
        )
    cfg.outputs = str(tmp_path)
    return Mock(config=cfg)


@pytest.fixture
def cluster() -> Cluster:
    slab = fcc111("Pd", size=(3, 3, 3), a=3.89, vacuum=10.0)
    system = System.from_ase(
        slab,
        parse_bonds={"method": "raw"},
        attach_is_adsorbate=True,
        parse_bonds_outer=True,
    )
    return Cluster.from_select(
        system,
        core=np.array([0]),
        method="distance",
        max_moved_threshold=8.0,
        env_threshold=15.0,
    )


@pytest.fixture
def gas() -> Gas:
    return Gas.from_molecule("CO")


def test_cluster_path_minima(explorer: Mock, cluster: Cluster) -> None:
    p = explorer.cluster_path(cluster, type="minima")
    fml = cluster.symbols.get_chemical_formula("metal")
    assert p == explorer.path / "minima" / fml / f"{cluster.hash}.npz"
    assert p.parent.is_dir()


def test_cluster_path_ts(explorer: Mock, cluster: Cluster) -> None:
    p = explorer.cluster_path(cluster, type="ts")
    fml = cluster.symbols.get_chemical_formula("metal")
    assert p == explorer.path / "ts" / fml / f"{cluster.hash}.npz"


def test_cluster_path_gas_ignores_type(explorer: Mock, gas: Gas) -> None:
    p = explorer.cluster_path(gas, type="minima")
    fml = gas.symbols.get_chemical_formula("metal")
    assert p == explorer.path / "gas" / fml / f"{gas.hash}.npz"


def test_cluster_save(explorer: Mock, cluster: Cluster, gas: Gas) -> None:
    p = explorer.cluster_path(cluster, type="minima")
    explorer.cluster_save(cluster, type="minima")
    assert p.is_file()

    pgas = explorer.cluster_path(gas)
    explorer.cluster_save(gas)
    assert pgas.is_file()


def test_cluster_check_minima(explorer: Mock, cluster: Cluster) -> None:
    good = _with_energetics(
        cluster, frequencies=np.array([100.0, 200.0, 300.0])
    )
    assert explorer.cluster_check(good, type="minima")

    bad = _with_energetics(
        cluster, fmax=0.1, frequencies=np.array([100.0, 200.0, 300.0])
    )
    assert not explorer.cluster_check(bad, type="minima")

    imaginary = _with_energetics(
        cluster, frequencies=np.array([-50.0, 200.0, 300.0])
    )
    assert not explorer.cluster_check(imaginary, type="minima")


def test_cluster_check_ts(explorer: Mock, cluster: Cluster) -> None:
    ts = _with_energetics(cluster, frequencies=np.array([-50.0, 200.0, 300.0]))
    assert explorer.cluster_check(ts, type="ts")

    not_ts = _with_energetics(
        cluster, frequencies=np.array([100.0, 200.0, 300.0])
    )
    assert not explorer.cluster_check(not_ts, type="ts")


def test_cluster_check_gas(explorer: Mock, gas: Gas) -> None:
    good = gas.model_copy(
        update={
            "energy": -1.0,
            "fmax": 0.01,
            "frequencies": np.array([100.0, 200.0]),
        }
    )
    assert explorer.cluster_check(good)


def test_cluster_check_unknown_type(explorer: Mock, cluster: Cluster) -> None:
    with pytest.raises(ValueError):
        explorer.cluster_check(cluster, type="gas")


def test_cluster_optimization_read_existing(
    explorer: Mock,
    cluster: Cluster,
) -> None:
    saved = _with_energetics(
        cluster, frequencies=np.array([100.0, 200.0, 300.0])
    )
    explorer.cluster_save(saved, type="minima")

    result = explorer.cluster_optimization(cluster, type="minima")

    assert isinstance(result, Cluster)
    assert result.hash == cluster.hash
    assert result.energy == -1.0
    assert explorer.network.vcount() == 1


def test_cluster_optimization_check_failed(
    explorer: Mock,
    cluster: Cluster,
) -> None:
    saved = _with_energetics(
        cluster, fmax=0.1, frequencies=np.array([100.0, 200.0, 300.0])
    )
    explorer.cluster_save(saved, type="minima")

    with pytest.raises(CheckMinimaFailed):
        explorer.cluster_optimization(cluster, type="minima")


def test_cluster_optimization_optimize_failed(
    explorer: Mock,
    cluster: Cluster,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "otfkmc.exploration._0base.call_optimize",
        lambda *args, **kwargs: ([], False),
    )

    with pytest.raises(OptimizationFailed):
        explorer.cluster_optimization(cluster, type="minima")



