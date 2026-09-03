import itertools
from typing import override

import numpy as np
from ase import Atoms
from graphatoms.system import Cluster, System
from omegaconf import DictConfig

from otfkmc.runner._1parser import FirstStep
from otfkmc.runner._2expl_0_Base import SecondStepABC


class SecondStepSurface(SecondStepABC):
    """The class for exploring the surface process."""

    @override
    def _explore_serial(self, system: System) -> None:
        """Explore the surface process."""
        result = itertools.starmap(
            helper_dimer,
            itertools.product(
                [self],
                [
                    self.handle_cluster_by_core(core, system=system)
                    for core in system.get_site_core(max_ncore=1)
                ],
            ),
        )
        list(result)

    @override
    def _explore_ray(self, system: System) -> None:
        """Explore the surface process in ray mode."""
        raise NotImplementedError("Ray mode is not supported.")


def helper_dimer(self: FirstStep, cluster: Cluster) -> None:
    """Helper function for dimer process."""
    expl: DictConfig = self.config.exploration
    assert not self.catalyst.check_induced_graph(), (
        "The system is not a valid graph."
    )
    assert cluster.check_induced_graph(), "The cluster is not a valid graph."

    # 1. save & optimize the reactant
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

    # 2. run single end transition state search
    print("Dimer (start):", cluster)
    lst, cvrg = self.dimer(
        atoms=core_atoms.copy(),
        max_steps=expl.get("max_steps", 1000),
        fmax=expl.get("fmax", 0.05),
    )
    if not cvrg:
        print(f"Dimer failed for {cluster}.")
        return

    # 3. convert the transition state to cluster object
    new_atoms = lst[-1]
    print("Dimer (end):", cluster)
    freq, modes = self.vib(atoms=new_atoms)
    f = new_atoms.get_forces()
    ts = cluster.from_ase(
        new_atoms,
        parse_bonds=self.config.bonds,
        parse_bonds_distance=False,
        parse_bonds_order=False,
        energy=new_atoms.get_potential_energy(),
        fmax=np.linalg.norm(f, axis=1).max(),
        frequencies=freq,
        nadsorbate=0,
    )
    self.cluster_save(cluster=ts, type="ts")
    if not self.cluster_check(ts, type="ts"):
        print(f"TS check failed for {ts}.")
        return
    p = self.cluster_path(cluster=ts, type="ts")
    name_ts = p.relative_to(self.path).as_posix()

    # 4. optimize the transition state in the two direction of the mode
    vdiff = new_atoms.positions - core_atoms.positions
    ldiff = np.linalg.norm(vdiff, axis=1)
    ldiff[ldiff < 1e-5] = np.inf
    imin_ldiff = np.argmin(ldiff)
    lmin_mode = np.linalg.norm(modes[0][imin_ldiff])
    mode = modes[0] * vdiff[imin_ldiff] / lmin_mode
    name_p = ""
    for sign in (1, -1):
        atoms = ts.to_ase(
            exclude_bond_attibutes=True,
            exclude_energetics=True,
        ).copy()
        atoms.info.pop("hashes", None)
        print(atoms.info.keys())
        atoms.positions += sign * mode
        try:
            cluster_optimized_0 = self.cluster_optimization(
                cluster.from_ase(
                    atoms,
                    parse_bonds=self.config.bonds,
                    parse_bonds_distance=False,
                    parse_bonds_order=False,
                    energy=None,
                    fmax=None,
                    frequencies=None,
                    nadsorbate=0,
                ),
                type="minima",
            )
        except self.CheckMinimaFailed:
            return
        p = self.cluster_path(cluster=cluster_optimized_0, type="minima")
        name_p = p.relative_to(self.path).as_posix()
        if name_p != name_r:
            break
    if name_p == "":
        print(
            f"TS {name_ts} is not optimized in the two direction of the mode."
        )
        return

    self.network.add_edge(name_r, name_p, name=name_ts)
