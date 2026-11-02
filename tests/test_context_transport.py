"""Real iterable and trajectory workers retain chemistry and fresh coordinates."""

from typing import Literal

import MDAnalysis as mda
import numpy as np
import pytest
from rdkit import Chem

from prolif import Fingerprint, Molecule
from prolif.ifp import IFP
from prolif.io import MoleculeStandardizer
from prolif.typeshed import InteractionMetadata
from tests.context_helpers import donor_geometry, load_pair, named


def make_pair() -> tuple[Molecule, Molecule]:
    source, ligand = load_pair()
    return MoleculeStandardizer()(source), Molecule.from_rdkit(ligand)


def target_rows(ifp: IFP) -> list[InteractionMetadata]:
    return [
        row
        for pair, kinds in ifp.items()
        if str(pair[1]) == "ASP29.B"
        for row in kinds.get("ImplicitHBAcceptor", ())
        if row["parent_indices"]["protein"] == (972,)
    ]


def test_iterable_workers_preserve_metadata_and_pose_coordinates() -> None:
    protein, ligand = make_pair()
    distant = Chem.Mol(ligand)
    conf = distant.GetConformer()
    for i, position in enumerate(conf.GetPositions()):
        conf.SetAtomPosition(i, position + 100)
    ligands = [ligand, Molecule.from_rdkit(distant)]
    results = []
    for jobs in (1, 2):
        fp = Fingerprint(["ImplicitHBAcceptor", "ImplicitHBDonor"], count=True)
        fp.run_from_iterable(ligands, protein, n_jobs=jobs, progress=False)
        assert target_rows(fp.ifp[0])
        assert not fp.ifp[1]
        results.append(fp.ifp)
    assert results[0] == results[1]


@pytest.mark.parametrize("strategy", ["chunk", "queue"])
def test_real_trajectory_workers_refresh_context_with_reused_topology(
    strategy: Literal["chunk", "queue"],
) -> None:
    inferring = pytest.importorskip("MDAnalysis.converters.RDKitInferring")
    protein, ligand = make_pair()
    oxygen = named(ligand[0], "O4").GetIdx()
    _, expected_plane = donor_geometry(protein, 972, ligand.xyz[oxygen])
    combined = Chem.CombineMols(protein, ligand)
    u = mda.Universe(combined)
    frames = np.repeat(u.atoms.positions[None, :, :], 2, axis=0)
    ligand_ag = u.select_atoms("resname MK1")
    protein_ag = u.select_atoms("not resname MK1")
    frames[1, ligand_ag.indices] += [100, 100, 100]
    u.load_new(frames)
    # This native inferrer restores/sanitizes template chemistry. Its default
    # H adjustment only restores existing H atoms; none are present or added here.
    configs = [
        {
            "implicit_hydrogens": True,
            "force": True,
            "inferrer": inferring.TemplateInferrer(Chem.Mol(template)),
        }
        for template in (ligand, protein)
    ]
    for atoms, config in zip((ligand_ag, protein_ag), configs, strict=True):
        converted = atoms.convert_to.rdkit(**config)
        assert converted.GetNumAtoms() == len(atoms)
        assert all(a.GetAtomicNum() != 1 for a in converted.GetAtoms())
    results = []
    for jobs in (1, 2):
        fp = Fingerprint(
            ["ImplicitHBAcceptor", "ImplicitHBDonor"], count=True, use_segid=False
        )
        fp.run(
            u.trajectory,
            ligand_ag,
            protein_ag,
            n_jobs=jobs,
            progress=False,
            parallel_strategy=strategy,
            converter_kwargs=(configs[0], configs[1]),
        )
        rows = target_rows(fp.ifp[0])
        assert rows
        # MDA trajectory coordinates are float32; compare with original coordinates.
        assert rows[0]["donor_plane_angle"] == pytest.approx(expected_plane, abs=1e-4)
        assert not fp.ifp[1]
        results.append(fp.ifp)
    assert results[0] == results[1]
