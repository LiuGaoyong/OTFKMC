"""The basic class for On-The-Fly Kinetic Monte Carlo Simulation.

This class is for the first step of the simulation.
    - handle the input parameters
    - calculate the bond list
    - calculate each atom is outer or inner
"""

import os
from typing import Literal

os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"

import numpy as np
import ray
from ase import Atoms
from graphatoms.system import Cluster, Gas, System
from graphatoms.utils.parser import hydra_parse
from omegaconf import DictConfig

from otfkmc.runner._0base import BaseOTFKMC


class FirstStep(BaseOTFKMC):
    def __init__(self, *, config: DictConfig) -> None:
        super().__init__(config=config)

        # 1. parse the system
        try:
            self.catalyst: System = hydra_parse(config.system, System)
        except Exception:
            self.catalyst: System = System.from_ase(
                atoms=hydra_parse(config.system, Atoms),
                attach_is_adsorbate=True,
                parse_bonds=config.bonds,
                parse_bonds_outer=True,
            )

        assert isinstance(self.catalyst, System)
        assert self.catalyst.pair is not None
        assert self.catalyst.is_outer is not None
        assert self.catalyst.is_adsorbate is not None
        print("Read the system:", self.catalyst)

        # 2. parse the gas list
        self.gas_lst: list[Gas] = []
        for gas in config.gas:
            if gas is None:
                continue
            gas = hydra_parse(gas, Gas)
            assert isinstance(gas, Gas)
            assert gas.sticking is not None
            assert gas.pressure is not None
            print("Read the gas:", gas)
            self.gas_lst.append(gas)

        # 3. initialize the ray cluster
        if str(self.config.parallel) == "ray":
            ray.init(ignore_reinit_error=True)
            ncpus = int(ray.cluster_resources()["CPU"])
            assert ncpus != 0, "Number of CPUs must be greater than 0"
        self.ncpu_for_runner = int(config.ray.ncpu_for_runner)

    def handle_cluster_by_core(
        self,
        core: np.ndarray,
        *,
        system: System | None = None,
    ) -> Cluster:
        """Handle the cluster by core."""
        if system is None:
            system = self.catalyst

        core = np.asarray(core)
        if core.dtype in (bool, np.bool_):
            core = np.unique(np.where(core))

        return Cluster.from_select(
            self.catalyst,
            core.astype(int),
            method=self.config.site.method,
            env_threshold=self.config.site.env_threshold,
            max_moved_threshold=self.config.site.max_moved_threshold,
        )

    def cluster_optimization(
        self,
        cluster: Cluster | Gas,
        *,
        type: str | Literal["minima", "gas"] = "minima",
    ) -> Cluster | Gas:
        """Optimize the clusters."""
        if isinstance(cluster, Gas):
            CLS, type = Gas, "gas"
        else:
            CLS, type = Cluster, "minima"

        p = self.cluster_path(cluster=cluster, type=type)
        if p.exists():
            result: Cluster | Gas = CLS.read_npz(p)
            print(f"Read {result.__class__.__name__.lower()}:", p)
        else:
            print("Optimization (start):", cluster)
            lst, cvrg = BaseOTFKMC.optimize(
                self,
                atoms=cluster.to_ase().copy(),
                method=str(self.config.optimizer.method).upper(),
                max_steps=int(self.config.optimizer.steps),
                fmax=float(self.config.optimizer.fmax),
            )

            if cvrg:
                new_atoms = lst[-1]
                print("Optimization (end):", cluster)
                freq, _ = self.vib(atoms=new_atoms)
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
                raise self.OptimizationFailed(
                    f"Optimization (failed): {cluster}."
                )
            self.cluster_save(cluster=result, type=type)
        if not self.cluster_check(result, type=type):
            raise self.CheckMinimaFailed(
                result.energy,
                result.fmax,
                result.frequencies,
            )
        if type == "minima":
            p = self.cluster_path(cluster=result, type=type)
            s = p.relative_to(self.path).as_posix()
            self.network.add_vertex(name=s)
        return result

    class OptimizationFailed(RuntimeError):
        """Optimization failed."""

    class CheckMinimaFailed(RuntimeError):
        """Check minima failed."""


if __name__ == "__main__":
    from pathlib import Path

    from omegaconf import OmegaConf

    fname = Path(__file__).parent / "config.yaml"
    config = OmegaConf.load(fname)
    assert isinstance(config, DictConfig)
    obj = FirstStep(config=config)

    print(ray.util.inspect_serializability(obj))
    print(ray.util.inspect_serializability(obj.catalyst))
    print(ray.util.inspect_serializability(obj.gas_lst))
    ray.shutdown()
