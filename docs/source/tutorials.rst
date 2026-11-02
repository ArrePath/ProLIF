Tutorials
=========

In this section you will learn how to generate an interaction fingerprint for different
scenarios, how to extract information from the fingerprint data, and how to visualize
it.

You can get a very brief overview of how ProLIF is implemented in the
:ref:`source/api:Summary` section of this documentation, with links to the different
modules for a more in-depth explanation.

All the examples here showcase a transmembrane protein (GPCR family) in complex with a
small molecule. For convenience, the different files used for this tutorial are included
with the ProLIF installation, and you can access these through either:

- ``prolif.datafiles.TOP`` and ``prolif.datafiles.TRAJ`` for the topology (PDB file) and
  trajectory (XTC file) used in the molecular dynamics tutorials.
- ``prolif.datafiles.datapath``, which points to the directory containing all the other
  tutorial files. This is a :class:`pathlib.Path` object which offers a convenient way
  to navigate filesystems as shown in their
  `basic use <https://docs.python.org/3/library/pathlib.html#basic-use>`__ section.

.. warning::
    Outside of the tutorials, remember to switch any reference to ``prolif.datafiles``
    with the actual paths to your inputs.

.. tip::
    At the top of each tutorial's page you can find links to either download the
    notebook or run it in Google Colab. You can install the dependencies for the
    tutorials with the command::
      
      pip install prolif[tutorials]


Molecular dynamics
------------------

There are two tutorial notebooks for MD simulations, depending on the type of components
that are being analyzed:

- :ref:`notebooks/md-ligand-protein:Ligand-protein MD`
- :ref:`notebooks/md-protein-protein:Protein-protein MD`

Docking
-------

Follow this tutorial for docking of a ligand with a protein:

:ref:`notebooks/docking:Docking`

PDB file
--------

This tutorial showcases how to use ProLIF from a PDB file:

:ref:`notebooks/pdb:PDB`

Advanced usage
--------------

For investigating water-mediated interactions, you can follow this tutorial:
:ref:`notebooks/water-bridge:Water-bridge Interactions`

For more advanced usecases such as modifying interaction parameters, defining your own
interactions, ignoring backbone interactions and such, you can have a look at this
tutorial:

:ref:`notebooks/advanced:Advanced usage`

**Implicit hydrogen methods**: If your topology does not contain hydrogen atoms, you can
still use the implicit hydrogen methods to generate interaction fingerprints. This method is 
useful for quickly comparing your experimental structure with computational results.

- :ref:`notebooks/implicit-hbond:Implicit hydrogen bond interaction`

Molecular context for implicit hydrogen bonds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implicit H-bond classification and geometry use the complete supplied
:class:`~prolif.molecule.Molecule`, not an isolated residue fragment. For example,
an internal peptide nitrogen must retain its bond to the preceding carbonyl when
its donor/acceptor character and donor plane are evaluated. Returned atom indices
still identify the requested residues and their owning molecules. This requires
RDKit 2024.3.1 or newer, including for complete recursive SMARTS matching.

Given a ligand and a complete protein in the same coordinate frame, use the
existing standardizer and fingerprint interfaces:

.. code-block:: python

    import prolif as plf
    from prolif.io import MoleculeStandardizer

    # ligand is a prepared Molecule; protein is a complete RDKit/ProLIF molecule.
    protein = MoleculeStandardizer()(protein)
    fp = plf.Fingerprint(count=True, implicit_hydrogens=True)
    interactions = fp.generate(ligand, protein, metadata=True)

Standardization corrects the intact parent and its residue chemistry together.
An ordinary RDKit input is not modified. A ProLIF input is updated as the same
object only after successful validation; a failure leaves it unchanged. Existing
atoms, hydrogens and coordinates are retained: this does not add missing H atoms,
choose an experimental protonation state, or certify source completeness.

Snapshots and deliberate edits
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Attached residues share a snapshot of their complete molecule. Retained residue
references continue to describe that earlier snapshot after standardization or a
rebuild; they are not live views. After intentionally changing coordinates or
chemistry, rebuild the molecule and obtain fresh residue references:

.. code-block:: python

    # edited_protein is the complete, deliberately edited RDKit molecule.
    protein = plf.Molecule.from_rdkit(edited_protein)
    current_residue = protein[0]

Do not continue using a residue saved before the rebuild. Arbitrary edits are not
tracked automatically. A standalone :class:`~prolif.residue.Residue` can use only
its own supplied graph; rebuilding a fragment cannot recover absent chemistry.
Custom residue caches must agree with their authoritative parent. Incoherent
caches raise an error when used for implicit analysis rather than falling back to
fragment chemistry.

Splitting complexes
^^^^^^^^^^^^^^^^^^^

:func:`~prolif.molecule.split_molecule` returns independently owned residue copies
with child-local parent indices. It does not change the input's atom mappings;
code must no longer rely on shared residue object identity between input and
children. For a known noncovalently bound ligand in a prepared complex:

.. code-block:: python

    ligand_id = plf.ResidueId("LIG", 1, "B")  # replace with the actual residue ID
    ligand, protein = plf.split_molecule(
        complex_mol, lambda resid: resid == ligand_id
    )
    interactions = fp.generate(ligand, protein, metadata=True)

.. warning::

    If any covalent bond crosses the partition, implicit H-bond analysis of
    **both children and all their nonempty descendants** raises an error.
    Ordinary splitting and unrelated interaction detectors remain available.
    Rewrapping, standardizing or repeatedly splitting a marked child does not
    repair missing chemistry; even water subsequently separated from it remains
    conservatively marked. Supply independently prepared complete input instead.
    The marker survives supported copying and serialization, but arbitrary exports
    that discard RDKit properties are outside this guarantee.

Custom SMARTS and scope
^^^^^^^^^^^^^^^^^^^^^^^

Recursive SMARTS may inspect atoms across residue boundaries. A returned
multi-atom match must, however, belong to one residue so its metadata remains
representable. Cross-owner returned tuples raise an explicit error; use a
recursive single-anchor query when neighboring residues are constraints rather
than reported interaction atoms.

Only implicit H-bond detectors use this complete-context matching and geometry.
Explicit-H and other interaction algorithms are unchanged. This is not a general
solution for every interaction affected by fragmentation, nor does it reconstruct
missing source bonds or omitted components.

