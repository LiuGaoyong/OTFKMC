"""Build wulff construction for Pt."""

from ase.cluster import wulff_construction

surface_energies_Pt = {  # Unit: eV*A^-2
    # Reference:
    #   https://www.nature.com/articles/sdata201680
    #   http://crystalium.materialsvirtuallab.org/
    (1, 1, 1): 0.093,
    (3, 3, 2): 0.097,
    (3, 2, 2): 0.099,
    (2, 2, 1): 0.100,
    (1, 1, 0): 0.105,
    (3, 3, 1): 0.106,
    # (1, 1, 0): 0.117,
    # (1, 0, 0): 0.116,
}
lattice_constant_Pt = 3.92

atoms = wulff_construction(
    symbol="Pt",
    surfaces=list(surface_energies_Pt.keys()),
    energies=list(surface_energies_Pt.values()),
    size=6000,
    structure="fcc",
    latticeconstant=lattice_constant_Pt,
)
print(f"Natoms={len(atoms)}")
atoms.write("structure.xyz")
