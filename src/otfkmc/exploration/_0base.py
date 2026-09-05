"""The basic class combining the runner and the explorer for the
On-The-Fly Kinetic Monte Carlo Simulation.

The cluster-related logic is implemented as methods of this class:
    - `cluster_path`: resolve the path of a saved cluster
    - `cluster_save`: save the cluster to the disk
    - `cluster_check`: check whether the cluster is a valid minima or ts
    - `cluster_optimization`: optimize the cluster, analyze its vibrations
      and save it
The concrete operations are implemented as functions:
    - geometry optimization: `_funcs.call_optimize`
    - single end transition state search: `_funcs.call_dimer`
    - double end transition state search: `_funcs.call_neb`
    - vibration analysis: `_funcs.call_vib`
"""

from abc import abstractmethod
from pathlib import Path
from typing import Literal, override

import numpy as np
from ase.calculators.calculator import Calculator
from graphatoms.system import Cluster, Gas  # type: ignore

from otfkmc.abc import Base, hydra_parse
from otfkmc.config import Config, EventConfig

from ._funcs import call_optimize, call_vib


class OptimizationFailed(RuntimeError):
    """Optimization failed."""


class CheckMinimaFailed(RuntimeError):
    """Check minima failed."""


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

    def cluster_path(
        self,
        cluster: Cluster | Gas,
        *,
        type: str | Literal["minima", "gas", "ts"] = "minima",
    ) -> Path:
        """Get the path to the cluster."""
        if isinstance(cluster, Gas):
            type = "gas"
        else:
            assert isinstance(cluster, Cluster)
            assert type in ("minima", "ts")
        symbols = cluster.symbols
        fml: str = symbols.get_chemical_formula("metal")
        p = self.path / type / fml
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)
        return p / f"{cluster.hash}.npz"

    def cluster_save(
        self,
        cluster: Cluster | Gas,
        *,
        type: str | Literal["minima", "gas", "ts"] = "minima",
    ) -> None:
        """Save the cluster."""
        cluster.write_npz(self.cluster_path(cluster, type=type))

    def cluster_check(
        self,
        cluster: Cluster | Gas,
        *,
        type: str | Literal["minima", "gas", "ts"] = "minima",
    ) -> bool:
        """Check whether the cluster is a valid minima or ts."""
        event: EventConfig = self.config.event
        fmax = float(event.max_force)
        if type == "ts":
            assert isinstance(cluster, Cluster)
            mfreq_ts = float(event.min_frequency_for_ts)
            return cluster.check_ts(fmax, mfreq_ts)
        elif isinstance(cluster, Gas) or type == "minima":
            mfreq_minima = float(event.min_frequency)
            return cluster.check_minima(fmax, mfreq_minima)
        else:
            raise ValueError(
                f"Unknown type={type}, or type(cluster)="  #
                f"{cluster.__class__.__name__}"
            )

    def cluster_optimization(
        self,
        cluster: Cluster | Gas,
        *,
        type: str | Literal["minima", "gas"] = "minima",
    ) -> Cluster | Gas:
        """Optimize the cluster, analyze its vibrations and save it.

        If the optimized cluster exists on the disk, read it directly.
        """
        if isinstance(cluster, Gas):
            CLS, type = Gas, "gas"
        else:
            CLS, type = Cluster, "minima"

        p = self.cluster_path(cluster, type=type)
        if p.exists():
            result: Cluster | Gas = CLS.read_npz(p)
            print(f"Read {result.__class__.__name__.lower()}:", p)
        else:
            print("Optimization (start):", cluster)
            lst, cvrg = call_optimize(
                atoms=cluster.to_ase().copy(),
                calc=self.calculator,
                method=str(self.config.optimizer.method).upper(),
                max_steps=int(self.config.optimizer.steps),
                fmax=float(self.config.optimizer.fmax),
            )
            if cvrg:
                new_atoms = lst[-1]
                print("Optimization (end):", cluster)
                freq, _ = call_vib(atoms=new_atoms, calc=self.calculator)
                f = new_atoms.get_forces()
                if type == "minima":
                    result = Cluster.from_ase(
                        atoms=new_atoms,
                        parse_bonds=self.config.bonds,
                        parse_bonds_distance=False,
                        parse_bonds_order=False,
                        energy=new_atoms.get_potential_energy(),
                        fmax=np.linalg.norm(f, axis=1).max(),
                        frequencies=freq,
                        nadsorbate=0,
                    )
                else:
                    result = Gas.from_ase(
                        atoms=new_atoms,
                        sticking=cluster.sticking,  # type: ignore
                        pressure=cluster.pressure,  # type: ignore
                        parse_bonds=self.config.bonds,
                        energy=new_atoms.get_potential_energy(),
                        fmax=np.linalg.norm(f, axis=1).max(),
                        parse_bonds_distance=False,
                        parse_bonds_order=False,
                        frequencies=freq,
                    )
                assert isinstance(result, (Cluster, Gas))
                print("Vibration frequencies:", freq[:2], "for", result)
            else:
                raise OptimizationFailed(f"Optimization (failed): {cluster}.")
            self.cluster_save(result, type=type)
        if not self.cluster_check(result, type=type):
            raise CheckMinimaFailed(
                result.energy, result.fmax, result.frequencies
            )
        if type == "minima":
            p = self.cluster_path(result, type=type)
            s = p.relative_to(self.path).as_posix()
            self.network.add_vertex(name=s)
        return result

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
