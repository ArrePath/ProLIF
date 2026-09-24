"""Private molecular snapshots for implicit H-bond chemistry and geometry.

No parent/residue back-references: residues share one graph and query cache.
Rebuild a Molecule after deliberately editing its graph or coordinates.
"""

from typing import TYPE_CHECKING, cast

import numpy as np
from rdkit import Chem

from prolif.pickling import PROLIF_PICKLE_OPTIONS
from prolif.residue import ResidueId

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from prolif.residue import Residue


# Non-private/noncomputed so existing MolProps transport retains the provenance.
_INCOMPLETE_CONTEXT = "prolif.covalentSplit"


def _require_complete(mol: Chem.Mol) -> None:
    if mol.HasProp(_INCOMPLETE_CONTEXT):
        raise ValueError(
            "Implicit H-bond analysis cannot use a covalent split or its descendants; "
            "supply independently prepared complete chemistry."
        )


def _all_matches(mol: Chem.Mol, query: Chem.Mol) -> tuple[tuple[int, ...], ...]:
    params = Chem.SubstructMatchParameters()
    params.maxMatches = 0
    params.maxRecursiveMatches = 0
    return cast(tuple[tuple[int, ...], ...], mol.GetSubstructMatches(query, params))


def _identity(atom: Chem.Atom) -> tuple:
    info = atom.GetPDBResidueInfo()
    return (
        atom.GetAtomicNum(),
        atom.GetIsotope(),
        None
        if info is None
        else (
            info.GetName(),
            info.GetResidueName(),
            info.GetResidueNumber(),
            info.GetChainId(),
            info.GetSegmentNumber(),
            info.GetInsertionCode(),
            info.GetAltLoc(),
        ),
    )


def _chemistry(atom: Chem.Atom) -> tuple:
    # Implicit H, hybridization and ring membership are fragment-derived; they
    # cannot establish divergence. Matching reads those from the intact parent.
    return (
        atom.GetFormalCharge(),
        atom.GetNumExplicitHs(),
        atom.GetNoImplicit(),
        atom.GetIsAromatic(),
        atom.GetChiralTag(),
    )


class _MolecularContext:
    def __init__(self, mol: Chem.Mol, residues: list["Residue"], *, use_segid: bool):
        self.mol = Chem.Mol(mol)
        # Removing disconnected components invalidates RDKit's ring cache.
        # Recompute perception on the snapshot without altering supplied chemistry.
        Chem.GetSymmSSSR(self.mol)
        self._xyz: NDArray[np.float64] | None = (
            self.mol.GetConformer().GetPositions() if mol.GetNumConformers() else None
        )
        if self._xyz is not None:
            self._xyz.flags.writeable = False
        self._match_cache: dict[bytes, tuple[tuple[tuple[int, ...], ...], ...]] = {}
        self._error: str | None = None
        self.local_to_parent: list[tuple[int, ...]] = []
        self.parent_to_local: dict[int, tuple[int, int]] = {}
        try:
            self._validate(residues, use_segid=use_segid)
        except ValueError as error:
            # Custom cache validation must not disable unrelated interactions.
            self._error = str(error)

    def _atom_index(self, atom: Chem.Atom, resid: ResidueId, *, use_segid: bool) -> int:
        if not atom.HasProp("mapindex"):
            raise ValueError("missing atom mapindex")
        index = atom.GetUnsignedProp("mapindex")
        if index >= self.mol.GetNumAtoms() or index in self.parent_to_local:
            raise ValueError("atom mapping is not bijective")
        parent_atom = self.mol.GetAtomWithIdx(index)
        if _identity(atom) != _identity(parent_atom):
            raise ValueError("atom identity differs from parent")
        if ResidueId.from_atom(parent_atom, use_segid=use_segid) != resid:
            raise ValueError("residue identity/namespace differs from parent")
        if _chemistry(atom) != _chemistry(parent_atom):
            raise ValueError("residue chemistry differs from parent")
        return index

    def _validate(self, residues: list["Residue"], *, use_segid: bool) -> None:
        if len({residue.resid for residue in residues}) != len(residues):
            raise ValueError("duplicate residue owners in custom cache")
        if self._xyz is not None and not np.isfinite(self._xyz).all():
            raise ValueError("non-finite coordinates")
        cache_edges = {}
        for owner, residue in enumerate(residues):
            mapping = []
            for atom in residue.GetAtoms():
                index = self._atom_index(atom, residue.resid, use_segid=use_segid)
                self.parent_to_local[index] = owner, atom.GetIdx()
                mapping.append(index)
            insertion_codes = {
                info.GetInsertionCode().strip()
                for atom in residue.GetAtoms()
                if (info := atom.GetPDBResidueInfo()) is not None
            }
            if len(insertion_codes) > 1:
                raise ValueError("residue namespace cannot distinguish insertion codes")
            self.local_to_parent.append(tuple(mapping))
            if bool(residue.GetNumConformers()) != (self._xyz is not None):
                raise ValueError("missing coordinate correspondence")
            if self._xyz is not None and not np.array_equal(
                self._xyz[mapping], residue.xyz
            ):
                raise ValueError("coordinates differ from parent")
            for bond in residue.GetBonds():
                key = tuple(
                    sorted(
                        (mapping[bond.GetBeginAtomIdx()], mapping[bond.GetEndAtomIdx()])
                    )
                )
                cache_edges[key] = (
                    bond.GetBondType(),
                    bond.GetIsAromatic(),
                    bond.GetStereo(),
                )
        if len(self.parent_to_local) != self.mol.GetNumAtoms():
            raise ValueError("residues do not cover the complete parent")
        parent_edges = {}
        for bond in self.mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if self.parent_to_local[i][0] == self.parent_to_local[j][0]:
                parent_edges[tuple(sorted((i, j)))] = (
                    bond.GetBondType(),
                    bond.GetIsAromatic(),
                    bond.GetStereo(),
                )
        if parent_edges != cache_edges:
            raise ValueError("intra-residue bonds differ from parent")

    def require_valid(self) -> None:
        _require_complete(self.mol)
        self.require_coherent()

    def require_coherent(self) -> None:
        if self._error is not None:
            raise ValueError(
                f"Invalid implicit H-bond context: {self._error}. "
                "Rebuild coherent residues from the authoritative parent."
            )

    @property
    def xyz(self) -> "NDArray[np.float64]":
        self.require_valid()
        if self._xyz is None:
            raise ValueError("Implicit H-bond geometry requires source coordinates")
        return self._xyz

    def parent_index(self, owner: int, local_index: int) -> int:
        self.require_valid()
        return self.local_to_parent[owner][local_index]

    def matches(self, query: Chem.Mol, owner: int) -> tuple[tuple[int, ...], ...]:
        self.require_valid()
        key = query.ToBinary()
        if key not in self._match_cache:
            groups: list[list[tuple[int, ...]]] = [[] for _ in self.local_to_parent]
            for match in _all_matches(self.mol, query):
                mapped = [self.parent_to_local[i] for i in match]
                anchor_owner = mapped[0][0]
                if any(other_owner != anchor_owner for other_owner, _ in mapped):
                    raise ValueError(
                        "Implicit H-bond SMARTS returns atoms across residue owners; "
                        "use a recursive single-anchor query for "
                        "cross-residue constraints."
                    )
                groups[anchor_owner].append(tuple(local for _, local in mapped))
            self._match_cache[key] = tuple(tuple(sorted(group)) for group in groups)
        return self._match_cache[key][owner]

    def __getstate__(self) -> dict:
        return {
            **self.__dict__,
            "_match_cache": {},
            "mol": self.mol.ToBinary(
                int(PROLIF_PICKLE_OPTIONS)
                | int(Chem.PropertyPickleOptions.CoordsAsDouble)
            ),
        }

    def __setstate__(self, state: dict) -> None:
        state["mol"] = Chem.Mol(state["mol"])
        self.__dict__.update(state)
        if self._xyz is not None:
            self._xyz.flags.writeable = False


def _matches(residue: "Residue", query: Chem.Mol) -> tuple[tuple[int, ...], ...]:
    _require_complete(residue)
    context: _MolecularContext | None = getattr(residue, "_context", None)
    if context is None:
        return _all_matches(residue, query)
    return context.matches(query, residue._context_owner)


def _geometry(
    residue: "Residue", local_index: int
) -> tuple[Chem.Mol, "NDArray[np.float64]", int]:
    _require_complete(residue)
    context = getattr(residue, "_context", None)
    if context is None:
        return residue, residue.xyz, local_index
    index = context.parent_index(residue._context_owner, local_index)
    return context.mol, context.xyz, index
