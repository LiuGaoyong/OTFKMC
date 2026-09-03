from abc import abstractmethod

from graphatoms.system import System

from otfkmc.runner._1parser import FirstStep


class SecondStepABC(FirstStep):
    def explore(self, system: System | None = None) -> None:
        """Exploration of the catalyst elemental reaction."""
        if system is None:
            system = self.catalyst

        parallel = str(self.config.parallel).lower()
        if parallel == "serial":
            self._explore_serial(system)
        elif parallel == "ray":
            self._explore_ray(system)
        else:
            raise ValueError(f"Parallel mode '{parallel}' is not supported.")

        self.network.write(self.network_path)

    @abstractmethod
    def _explore_serial(self, system: System) -> None:
        """Explore the catalyst elemental reaction in serial mode."""
        pass

    @abstractmethod
    def _explore_ray(self, system: System) -> None:
        """Explore the catalyst elemental reaction in ray mode."""
        pass
