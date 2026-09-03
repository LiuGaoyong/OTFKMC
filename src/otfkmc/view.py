import sys
from pathlib import Path

from ase.visualize import view as _ase_view
from graphatoms.system import SysGraph


def view() -> None:
    """View the system."""
    p = Path(sys.argv[1])  # type: ignore
    system = SysGraph.read_npz(p)
    _ase_view(system.to_ase())
