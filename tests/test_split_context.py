"""Splits own their mappings and never silently certify a covalently cut graph."""

import pickle

import dill
import numpy as np
import pytest
from rdkit import Chem

from prolif import Fingerprint, Molecule
from prolif.interactions import Hydrophobic, ImplicitHBAcceptor, ImplicitHBDonor
from prolif.io import MoleculeStandardizer
from prolif.molecule import split_molecule
from prolif.pickling import PICKLE_HANDLER
from prolif.residue import Residue
from tests.context_helpers import load_pair, named


def mapping(mol: Molecule) -> list[list[int]]:
    return [[a.GetUnsignedProp("mapindex") for a in r.GetAtoms()] for r in mol]


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("use_segid", [False, True])
def test_noncovalent_split_preserves_source_and_child_metadata(
    reverse: bool, use_segid: bool
) -> None:
    source, ligand = load_pair()
    protein = MoleculeStandardizer()(source)
    raw = Chem.CombineMols(protein, ligand)
    if use_segid:
        for atom in raw.GetAtoms():
            info = atom.GetPDBResidueInfo()
            info.SetSegmentNumber(1 if info.GetChainId() == "A" else 2)
    if reverse:
        raw = Chem.RenumberAtoms(raw, list(reversed(range(raw.GetNumAtoms()))))
    combined = Molecule.from_rdkit(raw, use_segid=use_segid)
    # Exercise local ordering independently of parent ordering.
    residues = [
        Residue(
            Chem.RenumberAtoms(r, list(reversed(range(r.GetNumAtoms())))),
            use_segid=use_segid,
        )
        for r in combined
    ]
    combined = Molecule(raw, residues=residues, use_segid=use_segid)
    before = mapping(combined)
    coordinates = combined.xyz.copy()
    lig, prot = split_molecule(combined, lambda r: r.name == "MK1")
    assert mapping(combined) == before
    np.testing.assert_array_equal(combined.xyz, coordinates)
    for child in (lig, prot):
        for residue in child:
            assert residue is not combined[residue.resid]
            for atom in residue.GetAtoms():
                parent = child.GetAtomWithIdx(atom.GetUnsignedProp("mapindex"))
                assert (
                    parent.GetPDBResidueInfo().GetName()
                    == atom.GetPDBResidueInfo().GetName()
                )
                np.testing.assert_array_equal(
                    child.xyz[parent.GetIdx()], residue.xyz[atom.GetIdx()]
                )
    fp = Fingerprint(
        ["ImplicitHBAcceptor", "ImplicitHBDonor"], count=True, use_segid=use_segid
    )
    for left, right, kind, role in [
        (lig, prot, "ImplicitHBAcceptor", "protein"),
        (prot, lig, "ImplicitHBDonor", "ligand"),
    ]:
        result = fp.generate(left, right, metadata=True)
        assert result
        nitrogen = named(prot["ASP29.2" if use_segid else "ASP29.B"], "N")
        expected = nitrogen.GetUnsignedProp("mapindex")
        assert any(
            row["parent_indices"][role] == (expected,)
            for kinds in result.values()
            for row in kinds.get(kind, ())
        )


def peptide_and_water() -> Molecule:
    peptide = Chem.MolFromSequence("AGK")
    peptide.AddConformer(Chem.Conformer(peptide.GetNumAtoms()))
    water = Chem.MolFromSmiles("O")
    water.GetAtomWithIdx(0).SetMonomerInfo(
        Chem.AtomPDBResidueInfo(" O  ", residueName="HOH", residueNumber=99)
    )
    water.AddConformer(Chem.Conformer(1))
    return Molecule.from_rdkit(Chem.CombineMols(peptide, water))


def assert_implicit_rejected(residue: Residue) -> None:
    for cls in (ImplicitHBAcceptor, ImplicitHBDonor):
        with pytest.raises(ValueError, match=r"covalent.*split"):
            list(cls(ignore_geometry_checks=True).detect(residue, residue))


def test_covalent_cut_does_not_modify_source_or_disable_other_detectors() -> None:
    original = peptide_and_water()
    before = mapping(original)
    lhs, rhs = split_molecule(original, lambda r: r.number == 2)
    assert mapping(original) == before
    for child in (lhs, rhs):
        assert_implicit_rejected(child[0])
        list(Hydrophobic().detect(child[0], child[0]))
    residue = original["GLY2.A"]
    assert residue._context.matches(
        Chem.MolFromSmarts("[$(N-C=O)]"), residue._context_owner
    )


@pytest.mark.parametrize(
    "operation", ["rewrap", "repeat", "detached", "standardize", "pickle", "dill"]
)
def test_known_cut_stays_rejected_through_supported_lifecycles(operation: str) -> None:
    lhs, _ = split_molecule(peptide_and_water(), lambda r: r.number == 1)
    if operation == "rewrap":
        result = Molecule.from_rdkit(Chem.Mol(lhs))[0]
    elif operation == "repeat":
        result = split_molecule(lhs, lambda _r: True)[0][0]
    elif operation == "detached":
        result = Residue(Chem.Mol(lhs[0]))
    elif operation == "standardize":
        result = MoleculeStandardizer()(lhs)[0]
    else:
        previous = PICKLE_HANDLER.get()
        PICKLE_HANDLER.set()
        try:
            transport = pickle if operation == "pickle" else dill
            result = transport.loads(transport.dumps(lhs))[0]
        finally:
            PICKLE_HANDLER.set(previous)
    assert_implicit_rejected(result)


def test_later_water_descendant_remains_conservatively_rejected() -> None:
    _, cut_remainder = split_molecule(peptide_and_water(), lambda r: r.number == 2)
    water, _ = split_molecule(cut_remainder, lambda r: r.name == "HOH")
    assert_implicit_rejected(water[0])
