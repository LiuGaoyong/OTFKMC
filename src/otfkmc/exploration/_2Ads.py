"""The class for exploring the adsorption process.

1. Got the site core for the adsorption process.
2. Try adsorption
"""

import itertools
from typing import override

import numpy as np
from ase import Atoms
from graphatoms.system import Cluster, Gas, System
from omegaconf import DictConfig

from otfkmc.exploration._2expl_0_Base import SecondStepABC
from otfkmc.runner._1parser import FirstStep


class SecondStepAds(SecondStepABC):
    """The class for exploring the adsorption process."""

    @override
    def _explore_serial(self, system: System) -> None:
        """Explore the adsorption process."""
        result = itertools.starmap(
            helper_adsorption,
            itertools.product(
                [self],
                [
                    self.handle_cluster_by_core(core, system=system)
                    for core in system.get_site_core(max_ncore=3)
                ],
                self.gas_lst,
            ),
        )
        list(result)

    @override
    def _explore_ray(self, system: System) -> None:
        """Explore the adsorption process in ray mode."""
        raise NotImplementedError("Ray mode is not supported.")


try:
    from adsorption.interfaces import DirectAdsorption

    def helper_adsorption(
        self: FirstStep,
        cluster: Cluster,
        adsorbate: Gas,
    ) -> None:
        """Helper function for adsorption process."""
        expl: DictConfig = self.config.exploration
        assert not self.catalyst.check_induced_graph(), (
            "The system is not a valid graph."
        )
        assert cluster.check_induced_graph(), (
            "The cluster is not a valid graph."
        )

        # 1. save & optimize the substrate
        cluster_optimized = self.cluster_optimization(cluster, type="minima")
        assert cluster_optimized.hash == cluster.hash, (
            "Cluster'hash changed after optimization."
        )
        core_atoms: Atoms = cluster_optimized.to_ase(
            exclude_bond_attibutes=True,
            exclude_energetics=True,
        )
        p = self.cluster_path(cluster=cluster_optimized, type="minima")
        name_r = p.relative_to(self.path).as_posix()

        # 2. save & optimize the gas molecule
        gas_optimized = self.cluster_optimization(adsorbate, type="gas")
        assert gas_optimized.hash == adsorbate.hash, (
            "Gas'hash changed after optimization."
        )
        p = self.cluster_path(cluster=gas_optimized, type="gas")
        name_g = p.relative_to(self.path).as_posix()

        # 3. try adsorption process & optimize the result
        ads = DirectAdsorption(nfibonacci=expl.get("nfibonacci", 100))
        try_result_atoms, _ = ads.__call__(
            atoms=Atoms(
                numbers=core_atoms.numbers,
                positions=core_atoms.positions,
                cell=core_atoms.cell,
                pbc=core_atoms.pbc,
            ),
            adsorbate=adsorbate.to_ase(),
            core=np.unique(np.where(cluster_optimized.iscore)),
        )
        try_result_atoms.info = core_atoms.info
        try:
            new_cluster_optimized = self.cluster_optimization(
                Cluster.from_ase(
                    try_result_atoms,
                    parse_bonds=self.config.bonds,
                    parse_bonds_distance=False,
                    parse_bonds_order=False,
                    nadsorbate=len(adsorbate),
                ),
                type="minima",
            )
            p = self.cluster_path(cluster=new_cluster_optimized, type="minima")
            name_p = p.relative_to(self.path).as_posix()
            self.network.add_edge(name_r, name_p, name=name_g)
        except self.OptimizationFailed as e:
            print(e)

except ImportError:

    def helper_adsorption(
        self: FirstStep,
        cluster: Cluster,
        adsorbate: Gas,
    ) -> None:
        """Helper function for adsorption process."""
        pass
