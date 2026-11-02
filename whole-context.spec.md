# Whole-context implicit H-bonds

## intent

Preserve complete supplied chemistry, deposited geometry and source atom ownership
through preparation, residue views, noncovalent splitting and supported transport.

## invariants

- Standardization validates a complete candidate before mutating its destination and projects template chemistry into the intact parent without adding atoms or coordinates.
- Implicit H-bond classification and geometry use one coherent complete-source snapshot with unambiguous source ownership and complete recursive query matching.
- Splitting preserves source state and child atom correspondence; covalently cut children and their descendants reject implicit H-bond analysis.
- Supported transport preserves namespace, snapshot chemistry and source geometry while discarding query caches.

## works when

- `tests/test_parent_standardization.py` covers parent charges, existing H/stereo/coordinates, reordered templates, missing-bond restoration and failure atomicity.
- `tests/test_whole_context_hbonds.py` derives donor angles/plane from deposited vectors and exercises peptide/proline/glycine/terminal-charge behavior in both roles.
- `tests/test_context.py` covers source namespace collisions, custom-cache divergence, recursive queries beyond 1000 anchors and snapshot rebuilds.
- `tests/test_split_context.py` covers mapping, namespace, input preservation and persistent covalent-cut rejection.
- `tests/test_context_transport.py` crosses actual iterable and trajectory worker boundaries with different poses/frames.
- `tests/test_molecule.py` covers upstream fragmented-residue errors and multi-component residue handling.

## chokepoints

`MoleculeStandardizer.__call__` and `_candidate_from_rdkit`, `Molecule.__init__`,
`_MolecularContext.require_valid`, `split_molecule` and `TrajectoryPoolQueue.process`.

## seams

Template engines → intact-parent preparation; constructors/splitting → residue
snapshots; actual serialization/workers → implicit detectors. Ariadne separately
owns the released extraction/JSON/rendering consumer seam.

## verification boundary

These are representative regressions, not proof of arbitrary custom-template
chemistry or complete support for deposited structures. Explicit-H and unrelated
interaction algorithms remain upstream algorithms. Standalone residues have only
their supplied graph; deliberate edits require a rebuilt snapshot. Exports dropping
properties can lose covalent-cut provenance. Match-dense 100k-atom custom SMARTS
resource use is not qualified. Fork Actions remain disabled; qualification is local
and the consumer repository's CI supplies its own seam coverage.

## why

The [whole-context integration decision](DECISIONS.md) explains why fragment-local
hybridization/classification corrections alone do not satisfy this boundary.
