"""Snapshot chemistry, ownership, completeness and lifetime."""

import copy
import pickle
from collections.abc import Iterator
from types import ModuleType

import dill
import numpy as np
import pytest
from rdkit import Chem

from prolif import Molecule
from prolif.interactions import Hydrophobic
from prolif.io import MoleculeStandardizer
from prolif.residue import Residue
from tests.context_helpers import load_pair, named


def matches(residue: Residue, query: str) -> tuple[tuple[int, ...], ...]:
    return residue._context.matches(Chem.MolFromSmarts(query), residue._context_owner)


def test_local_permutation_preserves_atom_ownership() -> None:
    raw = Molecule.from_rdkit(Chem.MolFromSequence("AGK"))
    residues = [
        Residue(Chem.RenumberAtoms(r, list(reversed(range(r.GetNumAtoms())))))
        for r in raw
    ]
    mol = Molecule(raw, residues=residues)
    for residue in mol:
        actual = {
            residue.GetAtomWithIdx(row[0]).GetUnsignedProp("mapindex")
            for row in matches(residue, "[#7]")
        }
        expected = {
            a.GetIdx()
            for a in mol.GetAtoms()
            if a.GetAtomicNum() == 7
            and a.GetPDBResidueInfo().GetResidueNumber() == residue.resid.number
        }
        assert actual == expected


@pytest.mark.parametrize(
    "bad", ["charge", "bond", "duplicate", "missing", "coordinate", "namespace"]
)
def test_invalid_custom_cache_is_rejected_at_implicit_use_only(bad: str) -> None:
    source, _ = load_pair()
    parent = Molecule.from_rdkit(source)
    residues = [Residue(Chem.Mol(r)) for r in parent]
    bad_res = next(r for r in residues if str(r.resid) == "ASP29.B")
    atom = named(bad_res, "OD2")
    if bad == "charge":
        atom.SetFormalCharge(-1)
    elif bad == "bond":
        bad_res.GetBondBetweenAtoms(
            named(bad_res, "CG").GetIdx(), atom.GetIdx()
        ).SetBondType(Chem.BondType.DOUBLE)
    elif bad == "duplicate":
        atom.SetUnsignedProp("mapindex", 0)
    elif bad == "missing":
        atom.ClearProp("mapindex")
    elif bad == "coordinate":
        bad_res.GetConformer().SetAtomPosition(atom.GetIdx(), (0, 0, 0))
    else:
        bad_res.resid.chain = "OTHER"
    mol = Molecule(source, residues=residues)
    # A divergent residue elsewhere must not be silently used as recursive context.
    probe = mol[0]
    list(Hydrophobic().detect(probe, mol[1]))
    with pytest.raises(ValueError, match="context"):
        matches(probe, "[#7]")


def test_duplicate_custom_owners_are_rejected_at_implicit_use() -> None:
    parent = Molecule.from_rdkit(Chem.MolFromSequence("AA"))
    residues = [Residue(Chem.Mol(r)) for r in parent]
    # Both fragments genuinely advertise the same namespace, rather than merely
    # having a cache ID that disagrees with the authoritative parent atoms.
    for mol in [parent, *residues]:
        for atom in mol.GetAtoms():
            atom.GetPDBResidueInfo().SetResidueNumber(1)
    residues[1].resid.number = 1
    mol = Molecule(parent, residues=residues)
    list(Hydrophobic().detect(mol[0], mol[0]))
    with pytest.raises(ValueError, match="duplicate residue owners"):
        matches(mol[0], "[#7]")


def test_rebinding_an_attached_residue_does_not_change_its_snapshot() -> None:
    raw = Chem.MolFromSequence("ADK")
    first = Molecule.from_rdkit(raw)
    residue = first[1]
    old_context = residue._context
    second = Molecule(raw, residues=list(first))
    assert residue._context is old_context
    assert first[1] is not second[1]
    assert matches(first[1], "[#8]") == matches(second[1], "[#8]")


def test_rebuild_gets_new_geometry_without_changing_old_snapshot() -> None:
    source, _ = load_pair()
    first = Molecule.from_rdkit(source)
    old = first["ASP29.B"]
    xyz = old._context.xyz.copy()
    source.GetConformer().SetAtomPosition(972, (1, 2, 3))
    second = Molecule.from_rdkit(source)
    np.testing.assert_array_equal(old._context.xyz, xyz)
    np.testing.assert_array_equal(second["ASP29.B"]._context.xyz[972], [1, 2, 3])
    assert not old._context.xyz.flags.writeable


def test_standardization_creates_new_coherent_snapshot() -> None:
    source, _ = load_pair()
    mol = Molecule.from_rdkit(source)
    old = mol["ASP29.B"]
    old_charge = old._context.mol.GetAtomWithIdx(
        named(old, "OD2").GetUnsignedProp("mapindex")
    ).GetFormalCharge()
    MoleculeStandardizer()(mol)
    fresh = mol["ASP29.B"]
    idx = named(fresh, "OD2").GetUnsignedProp("mapindex")
    assert old._context.mol.GetAtomWithIdx(idx).GetFormalCharge() == old_charge
    assert fresh._context.mol.GetAtomWithIdx(idx).GetFormalCharge() == -1
    assert matches(fresh, "[O-]")


@pytest.fixture
def property_pickling() -> Iterator[None]:
    from prolif.pickling import PICKLE_HANDLER

    previous = PICKLE_HANDLER.get()
    PICKLE_HANDLER.set()
    try:
        yield
    finally:
        PICKLE_HANDLER.set(previous)


@pytest.mark.parametrize("transport", [pickle, dill])
def test_shared_context_survives_supported_transport(
    transport: ModuleType, property_pickling: None
) -> None:
    mol = Molecule.from_rdkit(Chem.MolFromSequence("AGK"))
    expected = [matches(r, "[#7]") for r in mol]
    restored = transport.loads(transport.dumps(list(mol)))
    assert [matches(r, "[#7]") for r in restored] == expected
    assert all(r._context is restored[0]._context for r in restored)
    assert not restored[0]._context.mol.__dict__


def test_plain_rdkit_deepcopy_can_be_rebuilt_with_fresh_context() -> None:
    mol = Molecule.from_rdkit(Chem.MolFromSequence("AGK"))
    copied = copy.deepcopy(mol)
    rebuilt = Molecule.from_rdkit(copied)
    assert [matches(r, "[#7]") for r in rebuilt] == [matches(r, "[#7]") for r in mol]


def test_segment_namespace_controls_cut_bonds_and_metadata() -> None:
    raw = Chem.MolFromSequence("AA")
    for atom in raw.GetAtoms():
        info = atom.GetPDBResidueInfo()
        segment = info.GetResidueNumber()
        info.SetResidueNumber(1)
        info.SetSegmentNumber(segment)
    mol = Molecule.from_rdkit(raw, use_segid=True)
    for residue in mol:
        actual = {
            residue.GetAtomWithIdx(row[0]).GetUnsignedProp("mapindex")
            for row in matches(residue, "[#7]")
        }
        expected = {
            a.GetIdx()
            for a in mol.GetAtoms()
            if a.GetAtomicNum() == 7
            and str(a.GetPDBResidueInfo().GetSegmentNumber()) == residue.resid.chain
        }
        assert actual == expected


def test_recursive_matches_are_complete_and_cached_per_snapshot_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from prolif import _context

    seen = []
    original = _context._all_matches

    def observe(mol: Chem.Mol, query: Chem.Mol) -> tuple[tuple[int, ...], ...]:
        seen.append((id(mol), query.ToBinary()))
        return original(mol, query)

    monkeypatch.setattr(_context, "_all_matches", observe)
    source = Chem.MolFromSmiles(".".join(["O"] * 1105))
    for i, atom in enumerate(source.GetAtoms()):
        atom.SetMonomerInfo(
            Chem.AtomPDBResidueInfo(" O  ", residueName="HOH", residueNumber=i + 1)
        )
    molecules = [Molecule.from_rdkit(source), Molecule.from_rdkit(source)]
    for mol in molecules:
        for query in ["[$(O)]", "[$(O)]", "[#7]"]:
            actual = {
                r.GetAtomWithIdx(row[0]).GetUnsignedProp("mapindex")
                for r in mol
                for row in matches(r, query)
            }
            expected = {
                a.GetIdx()
                for a in source.GetAtoms()
                if a.GetAtomicNum() == (7 if query == "[#7]" else 8)
            }
            assert actual == expected
    assert len(seen) == len(set(seen)), "No repeated global match per snapshot/query"


def test_ambiguous_insertion_code_namespace_is_not_silently_merged() -> None:
    raw = Chem.MolFromSequence("AA")
    for atom in raw.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info.GetResidueNumber() == 2:
            info.SetResidueNumber(1)
            info.SetInsertionCode("A")
    mol = Molecule.from_rdkit(raw)
    with pytest.raises(ValueError, match="namespace"):
        matches(mol[0], "[#7]")


def test_multi_atom_matches_keep_all_local_indices_and_reject_cross_owner() -> None:
    mol = Molecule.from_rdkit(Chem.MolFromSequence("ADK"))
    for r in mol:
        for row in matches(r, "[#6]=[#8]"):
            bond = r.GetBondBetweenAtoms(*row)
            assert bond.GetBondType() == Chem.BondType.DOUBLE
    assert matches(mol[1], "[$(N-C=O)]"), "Recursive context may cross owners"
    with pytest.raises(ValueError, match=r"cross.*residue"):
        matches(mol[1], "N-C=O")
