# ArrePath ProLIF decisions

## 2026-11-02 — Integrate 2.2.3 without replacing complete-context chemistry

Retain transactional whole-parent template projection and complete-source implicit
H-bond snapshots on upstream 2.2.3. Adopt upstream ring perception/fused-ring SMARTS,
explicit-H proline exclusion, repeated-residue fragmentation, segment propagation,
fragmented-owner errors and queue error forwarding. Keep one fork splitter because
its mapping validation and covalent-cut provenance extend upstream's atom-order fix.

Whole-parent sanitization owns hybridization and hydrogen perception. Do not layer
upstream's coordinate-based hybridization guessing or fragment-terminal radical/H
adjustments onto the intact parent: these address cut fragments, require coordinates
for chemical preparation and duplicate chemical perception. Remove their otherwise
unused runtime-context hook, geometry helpers/constants and helper-only tests;
independent deposited-coordinate and chemical-graph regressions remain the oracles.

Ordinary molecule construction now rejects duplicate fragmented owners by default.
Only the transactional standardizer groups disconnected pieces with their existing
identity while templates can restore missing intra-residue bonds; the completed
candidate must still have coherent, unique source ownership. Do not suppress the
upstream error globally. Implicit snapshots also reject chain/segment/insertion-code
owners collapsed by the selected namespace, even if optional upstream errors are
suppressed. This does not add insertion codes to ProLIF's public residue IDs.

Use a fresh `2.2.3+arrepath.1` integration version. Scientific qualification uses
Ariadne's exact dependency versions and released `cheml-core-prolif==0.1.2`.
Ariadne retains immutable archive consumption; any internal package publication is
an operator-owned action after artifact qualification, not public PyPI publication.
The contract and remaining boundaries are in [whole-context.spec.md](whole-context.spec.md).
