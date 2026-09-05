"""The helper functions for the exploration.

The helpers orchestrate the exploration flow on top of the `ExplBaseABC`
methods, which encapsulate the cluster-related logic. The concrete
operations are delegated to `_funcs`.
"""

import numpy as np
from ase import Atoms
from graphatoms.system import Cluster, Gas, System

from otfkmc.config import ExplConfig

from ._0base import CheckMinimaFailed, ExplBaseABC, OptimizationFailed
from ._funcs import call_dimer, call_vib


def _optimize_cluster(
    explorer: ExplBaseABC,
    cluster: Cluster,
) -> tuple[Cluster, Atoms, str]:
    """Stage 1: optimize the input cluster, analyze its vibrations and save it.

    Returns the optimized cluster, its core atoms and the relative path name
    of the saved minima.
    """
    cluster_optimized = explorer.cluster_optimization(cluster, type="minima")
    assert cluster_optimized.hash == cluster.hash, (
        "Cluster'hash changed after optimization."
    )
    core_atoms: Atoms = cluster_optimized.to_ase(
        exclude_bond_attibutes=True,
        exclude_energetics=True,
    )
    p = explorer.cluster_path(cluster_optimized, type="minima")
    name_r = p.relative_to(explorer.path).as_posix()
    return cluster_optimized, core_atoms, name_r


def _search_ts(
    explorer: ExplBaseABC,
    cluster: Cluster,
    core_atoms: Atoms,
) -> tuple[Cluster, Atoms, np.ndarray] | None:
    """Stage 2 & 3: run the dimer method and check whether the result is a TS.

    Returns the TS cluster together with its atoms and vibrational modes if
    the dimer converged and the result passed the TS check, otherwise `None`.
    """
    expl: ExplConfig = explorer.config.exploration
    print("Dimer (start):", cluster)
    lst, cvrg = call_dimer(
        atoms=core_atoms.copy(),
        calc=explorer.calculator,
        max_steps=expl.maxtry,
        fmax=explorer.config.optimizer.fmax,
    )
    if not cvrg:
        print(f"Dimer failed for {cluster}.")
        return None

    # convert the dimer result to a TS cluster & check it
    new_atoms = lst[-1]
    print("Dimer (end):", cluster)
    freq, modes = call_vib(atoms=new_atoms, calc=explorer.calculator)
    f = new_atoms.get_forces()
    ts = cluster.from_ase(
        new_atoms,
        parse_bonds=explorer.config.bonds,
        parse_bonds_distance=False,
        parse_bonds_order=False,
        energy=new_atoms.get_potential_energy(),
        fmax=np.linalg.norm(f, axis=1).max(),
        frequencies=freq,
        nadsorbate=0,
    )
    explorer.cluster_save(ts, type="ts")
    if not explorer.cluster_check(ts, type="ts"):
        print(f"TS check failed for {ts}.")
        return None
    return ts, new_atoms, modes


def _descend_mode(
    explorer: ExplBaseABC,
    cluster: Cluster,
    ts: Cluster,
    new_atoms: Atoms,
    core_atoms: Atoms,
    modes: np.ndarray,
    name_r: str,
) -> str:
    """Stage 4: optimize the TS along its first vibrational mode.

    Displaces the TS by plus/minus the first mode and optimizes both
    directions. Returns the relative path name of the product minima, or an
    empty string if no distinct minima is found.
    """
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
            cluster_optimized_0 = explorer.cluster_optimization(
                cluster.from_ase(
                    atoms,
                    parse_bonds=explorer.config.bonds,
                    parse_bonds_distance=False,
                    parse_bonds_order=False,
                    energy=None,
                    fmax=None,
                    frequencies=None,
                    nadsorbate=0,
                ),
                type="minima",
            )
        except CheckMinimaFailed:
            return ""
        p = explorer.cluster_path(cluster_optimized_0, type="minima")
        name_p = p.relative_to(explorer.path).as_posix()
        if name_p != name_r:
            break
    return name_p


def helper_dimer(
    explorer: ExplBaseABC,
    cluster: Cluster,
) -> None:
    """Helper function for dimer process.

    The flow of dimer:
        1. optimize the input cluster, analyze its vibrations and save it
        2. call the dimer method on the optimized result
        3. check whether the dimer result is a transition state
        4. if it is a TS, optimize along its first vibrational mode
    """
    assert cluster.check_induced_graph(), "The cluster is not a valid graph."

    # 1. optimize the input cluster & save it as the reactant
    _, core_atoms, name_r = _optimize_cluster(explorer, cluster)

    # 2 & 3. run dimer on the optimized result & check whether it is a TS
    result = _search_ts(explorer, cluster, core_atoms)
    if result is None:
        return
    ts, new_atoms, modes = result
    p = explorer.cluster_path(ts, type="ts")
    name_ts = p.relative_to(explorer.path).as_posix()

    # 4. optimize the TS along its first vibrational mode
    name_p = _descend_mode(
        explorer, cluster, ts, new_atoms, core_atoms, modes, name_r
    )
    if name_p == "":
        print(
            f"TS {name_ts} is not optimized in the two direction of the mode."
        )
        return

    explorer.network.add_edge(name_r, name_p, name=name_ts)


try:
    from adsorption.interfaces import DirectAdsorption

    def helper_adsorption(
        explorer: ExplBaseABC,
        catalyst: System,
        cluster: Cluster,
        adsorbate: Gas,
    ) -> None:
        """Helper function for adsorption process."""
        expl: ExplConfig = explorer.config.exploration
        assert not catalyst.check_induced_graph(), (
            "The system is not a valid graph."
        )
        assert cluster.check_induced_graph(), (
            "The cluster is not a valid graph."
        )

        # 1. save & optimize the substrate
        cluster_optimized = explorer.cluster_optimization(
            cluster, type="minima"
        )
        assert cluster_optimized.hash == cluster.hash, (
            "Cluster'hash changed after optimization."
        )
        core_atoms: Atoms = cluster_optimized.to_ase(
            exclude_bond_attibutes=True,
            exclude_energetics=True,
        )
        p = explorer.cluster_path(cluster_optimized, type="minima")
        name_r = p.relative_to(explorer.path).as_posix()

        # 2. save & optimize the gas molecule
        gas_optimized = explorer.cluster_optimization(adsorbate, type="gas")
        assert gas_optimized.hash == adsorbate.hash, (
            "Gas'hash changed after optimization."
        )
        p = explorer.cluster_path(gas_optimized, type="gas")
        name_g = p.relative_to(explorer.path).as_posix()

        # 3. try adsorption process & optimize the result
        ads = DirectAdsorption(nfibonacci=expl.nfibonacci)
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
            new_cluster_optimized = explorer.cluster_optimization(
                Cluster.from_ase(
                    try_result_atoms,
                    parse_bonds=explorer.config.bonds,
                    parse_bonds_distance=False,
                    parse_bonds_order=False,
                    nadsorbate=len(adsorbate),
                ),
                type="minima",
            )
            p = explorer.cluster_path(new_cluster_optimized, type="minima")
            name_p = p.relative_to(explorer.path).as_posix()
            explorer.network.add_edge(name_r, name_p, name=name_g)
        except OptimizationFailed as e:
            print(e)

except ImportError:

    def helper_adsorption(
        explorer: ExplBaseABC,
        catalyst: System,
        cluster: Cluster,
        adsorbate: Gas,
    ) -> None:
        """Helper function for adsorption process."""
        pass


def helper_neb(
    explorer: ExplBaseABC,
    cluster: Cluster,
) -> None:
    """Helper function for neb process.

    The logic of neb:
        1. generate a far-end trial structure based on the input cluster,
           e.g. move an atom of the cluster to an adjacent site.
        2. call the neb method (`_funcs.call_neb`) to search the
           transition state between the input structure and the trial
           structure.
    """
    raise NotImplementedError("The neb logic is not implemented.")
