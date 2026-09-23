"""Public coordinate fixtures and independent geometry checks for context tests."""

import gzip
from math import acos, asin, degrees

import gemmi
import numpy as np
from rdkit import Chem

from prolif.datafiles import datapath
from prolif.io import MoleculeStandardizer


def named(mol, name):
    return next(
        a for a in mol.GetAtoms() if a.GetPDBResidueInfo().GetName().strip() == name
    )


def load_pair(source_format="pdb"):
    data = datapath / "implicitHbond"
    text = gzip.decompress((data / f"1HSG.{source_format}.gz").read_bytes()).decode()
    if source_format == "cif":
        structure = gemmi.make_structure_from_block(
            gemmi.cif.read_string(text).sole_block()
        )
        assert len(structure) == 1
        text = structure.make_pdb_string()
    protein = Chem.MolFromPDBBlock(
        "\n".join(line for line in text.splitlines() if line.startswith("ATOM  "))
        + "\nEND\n",
        removeHs=False,
    )
    ligand = Chem.MolFromPDBBlock(
        "\n".join(
            line
            for line in text.splitlines()
            if line.startswith("HETATM")
            and line[17:20] == "MK1"
            and line[21] == "B"
            and line[22:26].strip() == "902"
        )
        + "\nEND\n",
        removeHs=False,
    )
    assert protein is not None and ligand is not None
    template = gemmi.cif.read_string(
        gzip.decompress((data / "MK1.cif.gz").read_bytes()).decode()
    )
    ligand = next(iter(MoleculeStandardizer([template])(ligand)))
    return protein, ligand


def donor_geometry(mol, index, remote):
    xyz = mol.GetConformer().GetPositions()
    center = xyz[index]
    vectors = [
        xyz[n.GetIdx()] - center
        for n in mol.GetAtomWithIdx(index).GetNeighbors()
        if n.GetAtomicNum() != 1
    ]
    target = remote - center
    angles = sorted(
        degrees(
            acos(
                np.clip(
                    np.dot(v, target) / (np.linalg.norm(v) * np.linalg.norm(target)),
                    -1,
                    1,
                )
            )
        )
        for v in vectors
    )
    normal = np.cross(vectors[0], vectors[1])
    plane = degrees(
        asin(
            np.clip(
                abs(np.dot(normal, target))
                / (np.linalg.norm(normal) * np.linalg.norm(target)),
                0,
                1,
            )
        )
    )
    return angles, plane
