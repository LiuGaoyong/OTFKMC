import os

os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"

import numpy as np
import ray
from ase import Atoms
from graphatoms.system import Cluster, Gas, System
from omegaconf import DictConfig

from ._base import Base
from ._expl import hydra_parse


class RunnerBase(Base):
    """The abstract base class for the runner."""

    def __init__(self, *, config: DictConfig) -> None:
        super().__init__(config=config)

        # parse the gas list
        self.gas_lst: list[Gas] = []
        for gas in config.gas:
            if gas is None:
                continue
            gas = hydra_parse(gas, Gas)
            assert isinstance(gas, Gas)
            assert gas.sticking is not None
            assert gas.pressure is not None
            self.logger.info("Read the gas:", gas)
            self.gas_lst.append(gas)

        # initialize the ray cluster
        if str(self.config.parallel) == "ray":
            ray.init(ignore_reinit_error=True)
            ncpus = int(ray.cluster_resources()["CPU"])
            assert ncpus != 0, "Number of CPUs must be greater than 0"
        self.ncpu_for_runner = int(config.ray.ncpu_for_runner)

    def atoms2system(self, inp: Atoms | None) -> None:
        if inp is None:
            # parse system for first step
            try:
                self.catalyst: System = hydra_parse(self.config.system, System)
            except Exception:
                self.catalyst: System = System.from_ase(
                    atoms=hydra_parse(self.config.system, Atoms),
                    parse_bonds=self.config.bonds,
                    attach_is_adsorbate=True,
                    parse_bonds_outer=True,
                )
        elif isinstance(inp, Atoms):
            self.catalyst: System = System.from_ase(
                atoms=inp,
                parse_bonds=self.config.bonds,
                attach_is_adsorbate=True,
                parse_bonds_outer=True,
            )
        else:
            raise ValueError(f"Unknown type of input: {type(inp)}")

        assert isinstance(self.catalyst, System)
        assert self.catalyst.pair is not None
        assert self.catalyst.is_outer is not None
        assert self.catalyst.is_adsorbate is not None
        self.logger.info(f"Read the system: {self.catalyst}")
        return self.catalyst

    def system2cluster(self, system: System) -> tuple[Cluster, ...]:
        lst: list[Cluster] = [
            Cluster.from_select(
                system,
                np.unique(np.where(core)).astype(int),
                env_threshold=self.config.site.env_threshold,
                max_moved_threshold=self.config.site.max_moved_threshold,
                method=self.config.site.method,
            )
            for core in system.get_site_core(max_ncore=3)
        ]
        _, idxs = np.unique([i.hash for i in lst], return_index=True)
        self.logger.info(f"Find {len(lst)} cluster for sys={system.hash}.")
        self.logger.info(f"Find {len(idxs)} unique cluster.")
        return tuple(lst[i] for i in idxs)

    # def __handle_cluster_by_core(
    #     self,
    #     system: System,
    #     core: np.ndarray,
    # ) -> Cluster:
    #     """Handle the cluster by core."""
    #     core = np.asarray(core)
    #     if core.dtype in (bool, np.bool_):
    #         core = np.unique(np.where(core))
    #     return Cluster.from_select(
    #         system,
    #         core.astype(int),
    #         method=self.config.site.method,
    #         env_threshold=self.config.site.env_threshold,
    #         max_moved_threshold=self.config.site.max_moved_threshold,
    #     )


FirstStep = RunnerBase
