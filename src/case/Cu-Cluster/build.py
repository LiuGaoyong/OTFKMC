"""Build Octahedron for Cu."""

import sys

from ase.cluster import Octahedron

lc = 3.61
atoms = Octahedron("Cu", int(sys.argv[1]), latticeconstant=lc)
print(f"Natoms={len(atoms)}")
atoms.write("structure.xyz")
