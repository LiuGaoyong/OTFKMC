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
from ase.units import invcm
from ase.vibrations import Vibrations, VibrationsData
from graphatoms.system import Cluster, Gas

from otfkmc.abc.expl import ExplABC

from ._funcs import call_dimer as _dimer
from ._funcs import call_neb as _neb
from ._funcs import optimize as _optimize


class ExplorationBase(ExplABC):
    class OptimizationFailed(RuntimeError):
        """Optimization failed."""

    class CheckMinimaFailed(RuntimeError):
        """Check minima failed."""

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
            lst, cvrg = self.optimize(
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
