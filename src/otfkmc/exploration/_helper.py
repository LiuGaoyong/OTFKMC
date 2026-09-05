"""The helper functions for the exploration.

The helpers are decoupled from the explorer class: they only depend on
the explicit arguments (calculator, config, path, network) and call the
concrete operations from `_funcs`.
"""

from pathlib import Path
from typing import Literal

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from graphatoms.system import Cluster, Gas, System
from igraph import Graph

from otfkmc.config import Config, EventConfig, ExplConfig

from ._funcs import call_dimer, call_vib, optimize


class OptimizationFailed(RuntimeError):
    """Optimization failed."""


class CheckMinimaFailed(RuntimeError):
    """Check minima failed."""


def _cluster_path(
    path: Path,
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
    p = path / type / fml
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)
    return p / f"{cluster.hash}.npz"


def _cluster_save(
    path: Path,
    cluster: Cluster | Gas,
    *,
    type: str | Literal["minima", "gas", "ts"] = "minima",
) -> None:
    """Save the cluster."""
    cluster.write_npz(_cluster_path(path, cluster, type=type))


def _cluster_check(
    config: Config,
    cluster: Cluster | Gas,
    *,
    type: str | Literal["minima", "gas", "ts"] = "minima",
) -> bool:
    """Check whether the cluster is a valid minima or ts."""
    event: EventConfig = config.event
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
    calculator: Calculator,
    config: Config,
    path: Path,
    network: Graph,
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

    p = _cluster_path(path, cluster, type=type)
    if p.exists():
        result: Cluster | Gas = CLS.read_npz(p)
        print(f"Read {result.__class__.__name__.lower()}:", p)
    else:
        print("Optimization (start):", cluster)
        lst, cvrg = optimize(
            atoms=cluster.to_ase().copy(),
            calc=calculator,
            method=str(config.optimizer.method).upper(),
            max_steps=int(config.optimizer.steps),
            fmax=float(config.optimizer.fmax),
        )
        if cvrg:
            new_atoms = lst[-1]
            print("Optimization (end):", cluster)
            freq, _ = call_vib(atoms=new_atoms, calc=calculator)
            f = new_atoms.get_forces()
            if type == "minima":
                result = Cluster.from_ase(
                    atoms=new_atoms,
                    parse_bonds=config.bonds,
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
                    parse_bonds=config.bonds,
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
        _cluster_save(path, result, type=type)
    if not _cluster_check(config, result, type=type):
        raise CheckMinimaFailed(result.energy, result.fmax, result.frequencies)
    if type == "minima":
        p = _cluster_path(path, result, type=type)
        s = p.relative_to(path).as_posix()
        network.add_vertex(name=s)
    return result


def helper_dimer(
    calculator: Calculator,
    config: Config,
    path: Path,
    network: Graph,
    cluster: Cluster,
) -> None:
    """Helper function for dimer process.

    The flow of dimer:
        1. optimize the input cluster, analyze its vibrations and save it
        2. call the dimer method on the optimized result
        3. check whether the dimer result is a transition state
        4. if it is a TS, optimize along its first vibrational mode
    """
    expl: ExplConfig = config.exploration
    assert cluster.check_induced_graph(), "The cluster is not a valid graph."

    # 1. optimize the input cluster & save it as the reactant
    cluster_optimized = cluster_optimization(
        calculator, config, path, network, cluster, type="minima"
    )
    assert cluster_optimized.hash == cluster.hash, (
        "Cluster'hash changed after optimization."
    )
    core_atoms: Atoms = cluster_optimized.to_ase(
        exclude_bond_attibutes=True,
        exclude_energetics=True,
    )
    p = _cluster_path(path, cluster_optimized, type="minima")
    name_r = p.relative_to(path).as_posix()

    # 2. run single end transition state search on the optimized result
    print("Dimer (start):", cluster)
    lst, cvrg = call_dimer(
        atoms=core_atoms.copy(),
        calc=calculator,
        max_steps=expl.maxtry,
        fmax=config.optimizer.fmax,
    )
    if not cvrg:
        print(f"Dimer failed for {cluster}.")
        return

    # 3. convert the dimer result to a TS cluster & check it
    new_atoms = lst[-1]
    print("Dimer (end):", cluster)
    freq, modes = call_vib(atoms=new_atoms, calc=calculator)
    f = new_atoms.get_forces()
    ts = cluster.from_ase(
        new_atoms,
        parse_bonds=config.bonds,
        parse_bonds_distance=False,
        parse_bonds_order=False,
        energy=new_atoms.get_potential_energy(),
        fmax=np.linalg.norm(f, axis=1).max(),
        frequencies=freq,
        nadsorbate=0,
    )
    _cluster_save(path, ts, type="ts")
    if not _cluster_check(config, ts, type="ts"):
        print(f"TS check failed for {ts}.")
        return
    p = _cluster_path(path, ts, type="ts")
    name_ts = p.relative_to(path).as_posix()

    # 4. optimize the TS along its first vibrational mode
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
            cluster_optimized_0 = cluster_optimization(
                calculator,
                config,
                path,
                network,
                cluster.from_ase(
                    atoms,
                    parse_bonds=config.bonds,
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
            return
        p = _cluster_path(path, cluster_optimized_0, type="minima")
        name_p = p.relative_to(path).as_posix()
        if name_p != name_r:
            break
    if name_p == "":
        print(
            f"TS {name_ts} is not optimized in the two direction of the mode."
        )
        return

    network.add_edge(name_r, name_p, name=name_ts)


try:
    from adsorption.interfaces import DirectAdsorption

    def helper_adsorption(
        calculator: Calculator,
        config: Config,
        path: Path,
        network: Graph,
        catalyst: System,
        cluster: Cluster,
        adsorbate: Gas,
    ) -> None:
        """Helper function for adsorption process."""
        expl: ExplConfig = config.exploration
        assert not catalyst.check_induced_graph(), (
            "The system is not a valid graph."
        )
        assert cluster.check_induced_graph(), (
            "The cluster is not a valid graph."
        )

        # 1. save & optimize the substrate
        cluster_optimized = cluster_optimization(
            calculator, config, path, network, cluster, type="minima"
        )
        assert cluster_optimized.hash == cluster.hash, (
            "Cluster'hash changed after optimization."
        )
        core_atoms: Atoms = cluster_optimized.to_ase(
            exclude_bond_attibutes=True,
            exclude_energetics=True,
        )
        p = _cluster_path(path, cluster_optimized, type="minima")
        name_r = p.relative_to(path).as_posix()

        # 2. save & optimize the gas molecule
        gas_optimized = cluster_optimization(
            calculator, config, path, network, adsorbate, type="gas"
        )
        assert gas_optimized.hash == adsorbate.hash, (
            "Gas'hash changed after optimization."
        )
        p = _cluster_path(path, gas_optimized, type="gas")
        name_g = p.relative_to(path).as_posix()

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
            new_cluster_optimized = cluster_optimization(
                calculator,
                config,
                path,
                network,
                Cluster.from_ase(
                    try_result_atoms,
                    parse_bonds=config.bonds,
                    parse_bonds_distance=False,
                    parse_bonds_order=False,
                    nadsorbate=len(adsorbate),
                ),
                type="minima",
            )
            p = _cluster_path(path, new_cluster_optimized, type="minima")
            name_p = p.relative_to(path).as_posix()
            network.add_edge(name_r, name_p, name=name_g)
        except OptimizationFailed as e:
            print(e)

except ImportError:

    def helper_adsorption(
        calculator: Calculator,
        config: Config,
        path: Path,
        network: Graph,
        catalyst: System,
        cluster: Cluster,
        adsorbate: Gas,
    ) -> None:
        """Helper function for adsorption process."""
        pass


def helper_neb(
    calculator: Calculator,
    config: Config,
    path: Path,
    network: Graph,
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
