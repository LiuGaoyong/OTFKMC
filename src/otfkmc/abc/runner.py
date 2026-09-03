import numpy as np
from ase import Atoms
from graphatoms.system import Cluster, System
from omegaconf import DictConfig

from .expl import ExplABC, hydra_parse


class RunnerABC(ExplABC):
    """The abstract base class for the runner."""

    def __init__(self, *, config: DictConfig) -> None:
        super().__init__(config=config)

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
        print("Read the system:", self.catalyst)

    def handle_cluster_by_core(
        self,
        system: System,
        core: np.ndarray,
    ) -> Cluster:
        """Handle the cluster by core."""
        core = np.asarray(core)
        if core.dtype in (bool, np.bool_):
            core = np.unique(np.where(core))

        return Cluster.from_select(
            system,
            core.astype(int),
            method=self.config.site.method,
            env_threshold=self.config.site.env_threshold,
            max_moved_threshold=self.config.site.max_moved_threshold,
        )
