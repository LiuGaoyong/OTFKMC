from abc import abstractmethod
from typing import TYPE_CHECKING, Any, override

from ase.calculators.calculator import Calculator
from graphatoms.system import Cluster, Gas  # type: ignore
from graphatoms.utils.parser import hydra_parse

from ._base import Base, Config, DictConfig

if TYPE_CHECKING:

    def hydra_parse(
        cfg: DictConfig,
        cls: type,
        debug: bool = False,
        **kw,
    ) -> Any:
        """Parse Hydra DictConfig to instantiate an object.

        Args:
            cfg: Hydra DictConfig object.
            cls: Target class type.
            debug: Whether to print config for debugging.
            **kw: Additional arguments for instantiate.

        Returns:
            Instance of cls.
        """


class ExplABC(Base):
    """The abstract base class for the explorer."""

    @override
    def __init__(self, *, config: Config) -> None:
        super().__init__(config=config)
        self.calculator: Calculator = hydra_parse(
            self.config.calculator,  # type: ignore
            Calculator,
        )

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
        """Explore the catalyst elemental reaction in serial mode."""
        pass

    @abstractmethod
    def _explore_ray(self, cluster: Cluster, gas: Gas | None = None) -> None:
        """Explore the catalyst elemental reaction in ray mode."""
        pass
