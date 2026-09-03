"""The basic class for On-The-Fly Kinetic Monte Carlo Simulation.

The three methods are:
    - geometry optimization
    - single end transition state search: dimer
    - double end transition state search: neb
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import IO, Literal

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.symbols import Symbols
from ase.units import invcm
from ase.vibrations import Vibrations, VibrationsData
from graphatoms.system import Cluster, Gas
from graphatoms.utils.parser import hydra_parse
from igraph import Graph
from omegaconf import DictConfig

# from adsorption.common.optimize import call_dimer as _dimer
# from adsorption.common.optimize import call_neb as _neb
# from adsorption.common.optimize import optimize as _optimize
from otfkmc.exploration import call_dimer as _dimer
from otfkmc.exploration import call_neb as _neb
from otfkmc.exploration import optimize as _optimize


class ExplorationBase:
    def __init__(self, *, calculator: Calculator, **kwargs) -> None:
        assert isinstance(calculator, Calculator)
        self.calculator = calculator

    def optimize(
        self,
        atoms: Atoms,
        *,
        method: str = "LBFGS",
        logfile: IO | Path | str | None = None,
        trajectory: str | Path | None = None,
        append_trajectory: bool = False,
        max_steps: int = 100,
        fmax: float = 0.05,
    ) -> tuple[list[Atoms], bool]:
        """Optimize the `ase.Atoms`.

        Parameters
        ----------
        atoms : Atoms
            The atoms to optimize.
        calc : Calculator
            The calculator to use.
        method : str, optional
            The method to use. Defaults to "LBFGS".
        max_steps : int, optional
            The maximum number of steps. Defaults to 100.
        fmax : float, optional
            The maximum force. Defaults to 0.05.

        Returns:
        -------
        tuple[list[Atoms], bool]
            The optimized trajectory and whether the optimization converged.
        """
        return _optimize(
            atoms=atoms,
            calc=self.calculator,
            method=method,
            logfile=logfile,
            trajectory=trajectory,
            append_trajectory=append_trajectory,
            max_steps=max_steps,
            fmax=fmax,
        )

    def dimer(
        self,
        atoms: Atoms,
        *,
        displacement: np.ndarray | None = None,
        logfile: IO | Path | str | None = None,
        trajectory: str | Path | None = None,
        append_trajectory: bool = False,
        parse_mask_from_atoms: bool = True,
        mask: list[bool] | np.ndarray | None = None,
        max_steps: int = 1000,
        fmax: float = 0.02,
        **kwargs,
    ) -> tuple[list[Atoms], bool]:
        """Call dimer method to search transition state.

        Parameters
        ----------
        atoms : Atoms
            The atoms to optimize.
        calc : Calculator
            The calculator to use.
        displacement : np.ndarray | None, optional
            The displacement vector. Defaults to None.
        mask : list[bool] | np.ndarray | None, optional
            The mask to use. Defaults to None.
        parse_mask_from_atoms : bool, optional
            Whether to parse mask from atoms. Defaults to True.
        max_steps : int, optional
            The maximum number of steps. Defaults to 1000.
        fmax : float, optional
            The maximum force. Defaults to 0.01.
        debug : bool, optional
            Whether to print debug information. Defaults to False.
            If True, the optimization will print log information to stdout.

        Returns:
        -------
        tuple[list[Atoms], bool]
            The optimized trajectory and whether the optimization converged.

        """
        return _dimer(
            atoms=atoms,
            calc=self.calculator,
            displacement=displacement,
            logfile=logfile,
            trajectory=trajectory,
            append_trajectory=append_trajectory,
            parse_mask_from_atoms=parse_mask_from_atoms,
            mask=mask,
            max_steps=max_steps,
            fmax=fmax,
        )

    def neb(
        self,
        atoms: Atoms,
        calc: Calculator,
        final_atoms: Atoms,
        *,
        nimages: int = 5,
        climb: bool = True,
        k4spring: float = 0.1,
        dynamic_relaxation: bool = True,
        method4spring: str = "improvedtangent",
        method4interpolate: str = "linear",
        logfile: IO | Path | str | None = None,
        trajectory: str | Path | None = None,
        append_trajectory: bool = False,
        method4opt: str = "FIRE",
        max_steps: int = 100,
        fmax: float = 0.05,
    ) -> tuple[list[Atoms], bool]:
        """Call NEB method to search transition state.

        Parameters
        ----------
        atoms : Atoms
            The atoms to optimize.
        calc : Calculator
            The calculator to use.
        final_atoms : Atoms
            The final atoms to optimize.
        nimages : int, optional
            The number of images. Defaults to 5.
        climb : bool, optional
            Whether to climb. Defaults to True.
        k4spring : float, optional
            The spring constant. Defaults to 0.1.
        dynamic_relaxation : bool, optional
            Whether to use dynamic relaxation. Defaults to True.
            True skips images with forces below the convergence criterion.
            This is updated after each force call; if a previously converged
            image goes out of tolerance (due to spring adjustments between
            the image and its neighbors), it will be optimized again.
            False reverts to the default NEB implementation.
        method4spring : str | None, optional
            The method to use for spring optimization. Defaults to None.
            Choice betweeen five methods:

                * aseneb: legacy ase NEB implementation
                * improvedtangent: Paper I NEB implementation (default)
                * eb: Paper III full spring force implementation
                * spline: Paper IV spline interpolation (supports precon)
                * string: Paper IV string method (supports precon)

        method4interpolate : str, optional
            The method to use for interpolation. Defaults to "FIRE".
            Method by which to interpolate: 'linear' or 'idpp'.
                linear provides a standard straight-line interpolation, while
                idpp uses an image-dependent pair potential.


        Paper I:
            G. Henkelman and H. Jonsson, Chem. Phys, 113, 9978 (2000).
            :doi:`10.1063/1.1323224`

        Paper II:
            G. Henkelman, B. P. Uberuaga, and H. Jonsson, Chem. Phys,
            113, 9901 (2000).
            :doi:`10.1063/1.1329672`

        Paper III:
            E. L. Kolsbjerg, M. N. Groves, and B. Hammer, J. Chem. Phys,
            145, 094107 (2016)
            :doi:`10.1063/1.4961868`

        Paper IV:
            S. Makri, C. Ortner and J. R. Kermode, J. Chem. Phys.
            150, 094109 (2019)
            https://dx.doi.org/10.1063/1.5064465


        Returns:
        -------
        tuple[list[Atoms], bool]
            The atoms trajectory and whether the optimization converged.
        """
        return _neb(
            atoms=atoms,
            calc=self.calculator,
            final_atoms=final_atoms,
            nimages=nimages,
            climb=climb,
            k4spring=k4spring,
            dynamic_relaxation=dynamic_relaxation,
            method4spring=method4spring,
            method4interpolate=method4interpolate,
            logfile=logfile,
            trajectory=trajectory,
            append_trajectory=append_trajectory,
            method4opt=method4opt,
            max_steps=max_steps,
            fmax=fmax,
        )

    def vib(self, atoms: Atoms) -> tuple[np.ndarray, np.ndarray]:
        """Vibration atoms (frequencies in cm^-1 & harmonic modes)."""
        self.calculator.reset()
        atoms.calc = self.calculator
        with TemporaryDirectory() as tmpdir:
            vib = Vibrations(atoms, name=tmpdir)
            vib.run()
            vibdata: VibrationsData = vib.get_vibrations()
        eng, modes = vibdata.get_energies_and_modes(all_atoms=True)
        freq = np.asarray(eng / invcm, dtype=complex)
        freq = np.real(freq) - np.imag(freq)
        freq[np.abs(freq) < 1e-5] = 1e-5
        return freq, modes


class BaseOTFKMC(ExplorationBase):
    def __init__(self, *, config: DictConfig) -> None:
        calc = hydra_parse(config.calculator, Calculator)
        super().__init__(calculator=calc)
        self.config: DictConfig = config
        self.path = Path(config.output)
        self.path.mkdir(parents=True, exist_ok=True)
        for k in ["minima", "gas", "ts"]:
            (self.path / k).mkdir(parents=True, exist_ok=True)
        self.network_path = self.path / "network.lgl"
        self.network = Graph(directed=False)

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
        symbols: Symbols = cluster.symbols
        fml: str = symbols.get_chemical_formula("metal")
        p = self.path / type / fml
        # if type != "gas":
        #     p = p.parent / f"{cluster.ncore}_{p.name}"
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
        cluster.write_npz(self.cluster_path(cluster=cluster, type=type))

    def cluster_exists(
        self,
        cluster: Cluster | Gas,
        *,
        type: str | Literal["minima", "gas", "ts"] = "minima",
    ) -> bool:
        """Check if the cluster exists."""
        return self.cluster_path(cluster=cluster, type=type).exists()

    def cluster_check(
        self,
        cluster: Cluster | Gas,
        *,
        type: str | Literal["minima", "gas", "ts"] = "minima",
    ) -> bool:
        event: DictConfig = self.config.event
        fmax = float(event.get("max_force", 0.05))
        if type == "ts":
            assert isinstance(cluster, Cluster)
            mfreq_ts = float(event.get("min_frequency_for_ts", 50.0))
            return cluster.check_ts(fmax, mfreq_ts)
        elif isinstance(cluster, Gas) or type == "minima":
            mfreq_minima = float(event.get("min_frequency", 30.0))
            return cluster.check_minima(fmax, mfreq_minima)
        else:
            raise ValueError(
                f"Unknown type={type}, or type(cluster)="  #
                f"{cluster.__class__.__name__}"
            )
