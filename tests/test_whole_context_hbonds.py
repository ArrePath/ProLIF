"""Whole-peptide chemistry and geometry, without contact-count ground truth."""

import pytest
from rdkit import Chem

from prolif import Fingerprint, Molecule
from prolif.interactions import ImplicitHBAcceptor, ImplicitHBDonor
from prolif.io import MoleculeStandardizer
from prolif.residue import Residue
from prolif.typeshed import InteractionMetadata
from tests.context_helpers import donor_geometry, load_pair, named


@pytest.fixture(params=["pdb", "cif"])
def pair(request: pytest.FixtureRequest) -> tuple[Chem.Mol, Molecule, Residue]:
    source, ligand = load_pair(request.param)
    return source, MoleculeStandardizer()(source), ligand


def records(
    protein: Molecule, ligand: Residue, kind: str, protein_name: str, ligand_name: str
) -> list[InteractionMetadata]:
    residue = protein["ASP29.B"]
    ni = named(residue, protein_name).GetUnsignedProp("mapindex")
    oi = named(ligand, ligand_name).GetIdx()
    result = Fingerprint(count=True, implicit_hydrogens=True).generate(
        Molecule.from_rdkit(ligand),
        protein,
        metadata=True,
    )
    return [
        row
        for pair, kinds in result.items()
        if str(pair[1]) == "ASP29.B"
        for row in kinds.get(kind, ())
        if row["parent_indices"]["protein"] == (ni,)
        and row["parent_indices"]["ligand"] == (oi,)
    ]


def test_internal_peptide_n_is_not_an_acceptor(pair: tuple) -> None:
    _, protein, ligand = pair
    assert not records(protein, ligand, "ImplicitHBDonor", "N", "O4")


def test_backbone_donor_uses_complete_peptide_plane(pair: tuple) -> None:
    source, protein, ligand = pair
    nitrogen = named(protein["ASP29.B"], "N").GetUnsignedProp("mapindex")
    oxygen = named(ligand, "O4").GetIdx()
    angles, plane = donor_geometry(source, nitrogen, ligand.xyz[oxygen])
    rows = records(protein, ligand, "ImplicitHBAcceptor", "N", "O4")
    assert rows, "Complete-source geometry is inside the configured H-bond tolerances"
    assert rows[0]["ideal_donor_angle"] == 120.0
    assert sorted(rows[0]["donor_atom_angles"]) == pytest.approx(angles)
    assert rows[0]["donor_plane_angle"] == pytest.approx(plane)


@pytest.mark.parametrize("sequence", ["ADK", "APD", "AGGK"])
@pytest.mark.parametrize("terminal_charge", [0, 1])
def test_peptide_classification_uses_bonds_across_residue_boundaries(
    sequence: str, terminal_charge: int
) -> None:
    # Synthetic placement only enables distance testing; geometry is deliberately
    # disabled here. The deposited-coordinate tests above are the geometry oracle.
    raw = Chem.Mol(MoleculeStandardizer()(Chem.MolFromSequence(sequence)))
    nitrogen = next(
        a for a in raw.GetAtoms() if a.GetPDBResidueInfo().GetName().strip() == "N"
    )
    nitrogen.SetFormalCharge(terminal_charge)
    Chem.SanitizeMol(raw)
    raw.AddConformer(Chem.Conformer(raw.GetNumAtoms()))
    protein = Molecule.from_rdkit(raw)
    probe = Chem.MolFromSmiles("CO")
    conf = Chem.Conformer(probe.GetNumAtoms())
    for i in range(probe.GetNumAtoms()):
        conf.SetAtomPosition(i, (0, 0, 3))
    probe.AddConformer(conf)
    ligand = Molecule.from_rdkit(probe)[0]
    donor_n: set[int] = set()
    acceptor_n: set[int] = set()
    for residue in protein:
        for interaction, actual in [
            (ImplicitHBAcceptor(ignore_geometry_checks=True), donor_n),
            (ImplicitHBDonor(ignore_geometry_checks=True), acceptor_n),
        ]:
            for row in interaction.detect(ligand, residue):
                atom = residue.GetAtomWithIdx(row["indices"]["protein"][0])
                if atom.GetPDBResidueInfo().GetName().strip() == "N":
                    actual.add(residue.resid.number)
    assert acceptor_n == ({1} if terminal_charge == 0 else set())
    assert donor_n == {r.resid.number for r in protein if r.resid.name != "PRO"}


def test_both_roles_count_modes_and_custom_geometry_limits(pair: tuple) -> None:
    _, protein, ligand = pair
    ligand_mol = Molecule.from_rdkit(ligand)
    for count in (False, True):
        fp = Fingerprint(["ImplicitHBAcceptor", "ImplicitHBDonor"], count=count)
        forward = fp.generate(ligand_mol, protein, residues=["ASP29.B"], metadata=True)
        backward = fp.generate(protein, ligand_mol, metadata=True)
        assert forward and backward
        inverse_rows: list[InteractionMetadata] = []
        for kinds in backward.values():
            value = kinds.get("ImplicitHBDonor", ())
            inverse_rows.extend(value)
        target = named(protein["ASP29.B"], "N").GetUnsignedProp("mapindex")
        assert any(row["parent_indices"]["ligand"] == (target,) for row in inverse_rows)
    residue = protein["ASP29.B"]
    oxygen = named(ligand, "O4").GetIdx()
    nitrogen = named(residue, "N").GetIdx()

    def target_rows(interaction: ImplicitHBAcceptor) -> list[InteractionMetadata]:
        return [
            r
            for r in interaction.detect(ligand, residue)
            if r["indices"]["ligand"] == (oxygen,)
            and r["indices"]["protein"] == (nitrogen,)
        ]

    assert target_rows(ImplicitHBAcceptor())
    assert not target_rows(ImplicitHBAcceptor(tolerance_dev_dpa=1))
    assert not target_rows(ImplicitHBAcceptor(distance=2))
    assert target_rows(
        ImplicitHBAcceptor(tolerance_dev_dpa=1, ignore_geometry_checks=True)
    )
