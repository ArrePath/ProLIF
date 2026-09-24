"""Preparation must update authoritative chemistry without changing source identity."""

import numpy as np
import pytest
from rdkit import Chem

from prolif import Molecule
from prolif.io import MoleculeStandardizer
from prolif.io.template_engine import RDKitMolTemplateEngine
from prolif.residue import Residue
from tests.context_helpers import load_pair, named


def signature(mol: Chem.Mol) -> tuple:
    return (
        Chem.MolToSmiles(mol),
        [
            (
                a.GetAtomicNum(),
                a.GetIsotope(),
                a.GetChiralTag(),
                a.GetPDBResidueInfo().GetName(),
                a.GetPDBResidueInfo().GetResidueName(),
                a.GetPDBResidueInfo().GetResidueNumber(),
                a.GetPDBResidueInfo().GetChainId(),
            )
            for a in mol.GetAtoms()
        ],
        mol.GetConformer().GetPositions().tolist() if mol.GetNumConformers() else [],
    )


def test_charges_reach_complete_parent() -> None:
    source, _ = load_pair()
    prepared = MoleculeStandardizer()(source)
    for resid, name, charge in [
        ("ASP29.B", "OD2", -1),
        ("LYS55.A", "NZ", 1),
        ("ARG8.B", "NH2", 1),
    ]:
        a = named(prepared[resid], name)
        assert a.GetFormalCharge() == charge
        assert (
            prepared.GetAtomWithIdx(a.GetUnsignedProp("mapindex")).GetFormalCharge()
            == charge
        )


@pytest.mark.parametrize("mixed_h", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_source_atoms_coordinates_and_hydrogens_survive(
    mixed_h: bool, reverse: bool
) -> None:
    source, _ = load_pair()
    if mixed_h:
        source = Chem.AddHs(
            source, onlyOnAtoms=[972], addCoords=True, addResidueInfo=True
        )
    if reverse:
        source = Chem.RenumberAtoms(source, list(reversed(range(source.GetNumAtoms()))))
    before = signature(source)
    result = MoleculeStandardizer()(source)
    assert signature(source) == before
    assert signature(result)[1:] == before[1:]
    for residue in result:
        for atom in residue.GetAtoms():
            parent = result.GetAtomWithIdx(atom.GetUnsignedProp("mapindex"))
            assert parent.GetAtomicNum() == atom.GetAtomicNum()
            np.testing.assert_array_equal(
                residue.xyz[atom.GetIdx()], result.xyz[parent.GetIdx()]
            )


@pytest.mark.parametrize(
    "bad_result",
    ["raise", "duplicate", "missing", "identity", "coordinate", "inventory"],
)
def test_failed_later_engine_leaves_prolif_input_unchanged(bad_result: str) -> None:
    raw = Chem.MolFromSequence("AH")
    for atom in raw.GetAtoms():
        if atom.GetPDBResidueInfo().GetResidueNumber() == 2:
            atom.GetPDBResidueInfo().SetResidueName("HSD")
    source = Molecule.from_rdkit(raw)
    before = signature(source), [signature(r) for r in source]
    standardizer = MoleculeStandardizer()
    engine = standardizer.engines["HID"]

    class BadEngine:
        name = "HID"

        def n_heavy_atoms(self) -> int:
            return engine.n_heavy_atoms()

        def apply(self, residue: Residue) -> Residue:
            if bad_result == "raise":
                raise ValueError("deliberate later engine failure")
            result = engine.apply(residue)
            if bad_result == "duplicate":
                result.GetAtomWithIdx(1).SetUnsignedProp(
                    "mapindex", result.GetAtomWithIdx(0).GetUnsignedProp("mapindex")
                )
            elif bad_result == "missing":
                result.GetAtomWithIdx(0).ClearProp("mapindex")
            elif bad_result == "coordinate":
                result.AddConformer(Chem.Conformer(result.GetNumAtoms()))
            elif bad_result == "inventory":
                changed = Chem.RWMol(result)
                changed.RemoveAtom(changed.GetNumAtoms() - 1)
                result = Residue(changed.GetMol())
            else:
                result.GetAtomWithIdx(0).SetIsotope(15)
            return result

    standardizer.engines["HID"] = BadEngine()
    with pytest.raises(
        ValueError, match=r"deliberate|provenance|identity|coordinate|inventory"
    ):
        standardizer(source)
    assert (signature(source), [signature(r) for r in source]) == before


def test_inplace_commit_retains_old_residue_snapshot() -> None:
    source, _ = load_pair()
    mol = Molecule.from_rdkit(source)
    old = mol["ASP29.B"]
    before = signature(old)
    assert MoleculeStandardizer()(mol) is mol
    assert signature(old) == before
    assert named(mol["ASP29.B"], "OD2").GetFormalCharge() == -1
    idx = named(mol["ASP29.B"], "OD2").GetUnsignedProp("mapindex")
    assert mol.GetAtomWithIdx(idx).GetFormalCharge() == -1


def test_engine_reordering_is_mapped_not_zipped() -> None:
    source = Chem.MolFromSequence("ADK")
    normal = MoleculeStandardizer()(source)
    standardizer = MoleculeStandardizer()
    engine = standardizer.engines["ASP"]

    class ReorderEngine:
        name = "ASP"

        def n_heavy_atoms(self) -> int:
            return engine.n_heavy_atoms()

        def apply(self, residue: Residue) -> Residue:
            prepared = engine.apply(residue)
            return Residue(
                Chem.RenumberAtoms(
                    prepared, list(reversed(range(prepared.GetNumAtoms())))
                )
            )

    standardizer.engines["ASP"] = ReorderEngine()
    reordered = standardizer(source)
    assert signature(reordered) == signature(normal)
    idx = named(reordered[1], "OD2").GetUnsignedProp("mapindex")
    assert reordered.GetAtomWithIdx(idx).GetFormalCharge() == -1


def test_rdkit_template_preserves_reordered_existing_h() -> None:
    source, _ = load_pair()
    source = Chem.AddHs(source, onlyOnAtoms=[972], addCoords=True, addResidueInfo=True)
    source = Chem.RenumberAtoms(source, list(reversed(range(source.GetNumAtoms()))))
    before = signature(source)
    standardizer = MoleculeStandardizer()
    standardizer.engines["ASP"] = RDKitMolTemplateEngine(
        "ASP", Chem.MolFromSmiles("N[C@@H](CC(=O)[O-])C=O")
    )
    prepared = standardizer(source)
    assert signature(source) == before
    assert signature(prepared)[1:] == before[1:]
    residue = prepared["ASP29.B"]
    atom = named(residue, "OD2")
    assert (
        prepared.GetAtomWithIdx(atom.GetUnsignedProp("mapindex")).GetFormalCharge()
        == -1
    )


@pytest.mark.parametrize("change", ["hydrogen", "stereo"])
def test_template_cannot_rewire_source_hydrogen_or_assigned_stereo(change: str) -> None:
    raw = Chem.MolFromSequence("K")
    nz = named(raw, "NZ").GetIdx()
    raw = Chem.AddHs(raw, onlyOnAtoms=[nz], addResidueInfo=True)
    raw.AddConformer(Chem.Conformer(raw.GetNumAtoms()))
    standardizer = MoleculeStandardizer()
    engine = standardizer.engines["LYS"]

    class RewireEngine:
        name = "LYS"

        def n_heavy_atoms(self) -> int:
            return engine.n_heavy_atoms()

        def apply(self, residue: Residue) -> Residue:
            prepared = engine.apply(residue)
            changed = Chem.RWMol(prepared)
            if change == "hydrogen":
                h = next(a for a in prepared.GetAtoms() if a.GetAtomicNum() == 1)
                changed.RemoveBond(h.GetIdx(), named(prepared, "NZ").GetIdx())
                changed.AddBond(
                    h.GetIdx(), named(prepared, "N").GetIdx(), Chem.BondType.SINGLE
                )
            else:
                ca = named(prepared, "CA").GetIdx()
                changed.RemoveBond(ca, named(prepared, "CB").GetIdx())
                changed.AddBond(
                    ca, named(prepared, "CG").GetIdx(), Chem.BondType.SINGLE
                )
            return Residue(changed.GetMol())

    standardizer.engines["LYS"] = RewireEngine()
    before = signature(raw)
    with pytest.raises(ValueError, match=r"hydrogen|stereochemistry"):
        standardizer(raw)
    assert signature(raw) == before


def test_template_can_restore_missing_intra_residue_bond() -> None:
    source = Chem.MolFromSequence("ADK")
    expected = MoleculeStandardizer()(source)
    r = Molecule.from_rdkit(source)[1]
    i = named(r, "CG").GetUnsignedProp("mapindex")
    j = named(r, "OD2").GetUnsignedProp("mapindex")
    damaged = Chem.RWMol(source)
    damaged.RemoveBond(i, j)
    repaired = MoleculeStandardizer()(damaged.GetMol())
    assert repaired.GetBondBetweenAtoms(i, j) is not None
    assert Chem.MolToSmiles(repaired) == Chem.MolToSmiles(expected)
