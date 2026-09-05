"""The basic class combining the runner and the explorer for the
On-The-Fly Kinetic Monte Carlo Simulation.

The concrete operations are implemented as functions:
    - geometry optimization: `_funcs.optimize`
    - single end transition state search: `_funcs.call_dimer`
    - double end transition state search: `_funcs.call_neb`
    - vibration analysis: `_funcs.call_vib`
"""

from abc import abstractmethod
from typing import override

from ase.calculators.calculator import Calculator
from graphatoms.system import Cluster, Gas  # type: ignore

from otfkmc.abc import Base, hydra_parse
from otfkmc.config import Config


class ExplBaseABC(Base):
    """The abstract base class for the explorer."""

    @override
    def __init__(self, *, config: Config) -> None:
        super().__init__(config=config)
        self.calculator: Calculator = hydra_parse(
            self.config.calculator,  # type: ignore
            Calculator,
        )

    @property
    def gas_lst(self) -> list[Gas]:
        # The `_gas_lst` was not implemented in this class.
        #   Please use `class XXX(THIS_CALSS, FirstStep)` make it available.
        return self._gas_lst  # type: ignore

    def explore(self, cluster: Cluster, gas: Gas | None = None) -> None:
        """Exploration of the catalyst elemental reaction."""
        if self.pmode == "serial":
            self._explore_serial(cluster=cluster, gas=gas)
        elif self.pmode == "joblib":
            self._explore_joblib(cluster=cluster, gas=gas)
        elif self.pmode == "ray":
            self._explore_ray(cluster=cluster, gas=gas)
        else:
            raise ValueError(f"Parallel mode '{self.pmode}' is not supported.")
        self.network.write(self.network_path)

    @abstractmethod
    def _explore_serial(self, cluster: Cluster, gas: Gas | None = None) -> None:
        """Explore the catalyst elemental reaction in serial mode."""
        pass

    @abstractmethod
    def _explore_joblib(self, cluster: Cluster, gas: Gas | None = None) -> None:
        """Explore the catalyst elemental reaction in joblib mode."""
        pass

    @abstractmethod
    def _explore_ray(self, cluster: Cluster, gas: Gas | None = None) -> None:
        """Explore the catalyst elemental reaction in ray mode."""
        pass
