.. _cli-label:

The pyAFQ CLI
~~~~~~~~~~~~~

pyAFQ provides two command line interfaces that run the same pipeline
underneath:

* ``pyAFQ`` takes explicit paths to a single subject's DWI and T1w files
  (in the style of MRtrix commands).
* ``pyAFQ-bids`` takes a BIDS dataset and discovers the inputs with pyBIDS
  (in the style of BIDS Apps). See :ref:`cli-bids-label`.

``pyAFQ``: single subject with explicit paths
---------------------------------------------

The following usage is available:

.. code-block:: none

   pyAFQ [OPTIONS] dwi t1 o_folder
   pyAFQ download
   pyAFQ qsiprep

The first form runs the pyAFQ tractometry pipeline on a single subject.

Two subcommands are also available, and take no further arguments:

``download``
   Fetch every template, atlas, and model pyAFQ may need and cache them
   locally, then exit. Run this once if you intend to use pyAFQ on a
   machine without internet access, or to build a Docker image.

``qsiprep``
   Write default JSON configuration files into the current working
   directory and exit. These can be used to define a pyAFQ recon workflow
   in qsiprep; no pipeline is run.

Positional arguments
^^^^^^^^^^^^^^^^^^^^

``dwi``
   Path to DWI data file.

``t1``
   Path to T1-weighted image file. Must already be registered to the DWI
   data, though not resampled.

``o_folder``
   Path to output folder.

Options
^^^^^^^

-h, --help              Show this help message and exit.
--bval BVAL             Path to bval file. If none, the DWI data file path
                        will be used to find it.
--bvec BVEC             Path to bvec file. If none, the DWI data file path
                        will be used to find it.
-v, --verbose           Verbose logging.
-d, --dry-run           Perform a dry run — print the recognized arguments
                        without running pyAFQ.
-c TO_CALL, --call TO_CALL
                        AFQ.api attribute to get using the specified config
                        file. Defaults to ``all``, which performs the entire
                        tractometry pipeline.

Note that all other pyAFQ optional parameters can also be passed in. To see them,
use the ``--help`` option, i.e., ``pyAFQ --help``. Here is a full example call to
the pyAFQ CLI:

.. code-block:: none

    pyAFQ ~/AFQ_data/HBN/derivatives/qsiprep/sub-NDARAA948VFH/ses-HBNsiteRU/dwi/sub-NDARAA948VFH_ses-HBNsiteRU_acq-64dir_space-T1w_desc-preproc_dwi.nii.gz ~/AFQ_data/HBN/derivatives/qsiprep/sub-NDARAA948VFH/anat/sub-NDARAA948VFH_desc-preproc_T1w.nii.gz ~/AFQ_data/HBN/derivatives/afq/sub-NDARAA948VFH/ses-HBNsiteRU/dwi --rng_seed=2026 --return_idx=True --pve="multiaxial+brainchop+synthseg"


.. _cli-bids-label:

``pyAFQ-bids``: BIDS App style
------------------------------

.. code-block:: none

   pyAFQ-bids [OPTIONS] bids_dir {participant,group}

Inputs are found with pyBIDS from the preprocessed derivatives inside
``bids_dir`` (for example the output of qsiprep), so no individual file paths
are required. This uses :class:`AFQ.api.group.GroupAFQ` underneath.

Positional arguments
^^^^^^^^^^^^^^^^^^^^

``bids_dir``
   Root of the BIDS dataset. Must contain a ``dataset_description.json`` and
   a ``derivatives`` folder with preprocessed DWI data.

``analysis_level``
   ``participant`` runs the tractometry pipeline for each subject (and
   session) found. ``group`` combines tract profiles computed
   at the participant level into a single ``tract_profiles.csv``.

Options
^^^^^^^

--participant-label LABEL [LABEL ...]
                        Participants to process, with or without the
                        ``sub-`` prefix. Default: all participants.
--session-id ID [ID ...]
                        Sessions to process, with or without the ``ses-``
                        prefix. Default: all sessions.
--dwi-preproc-pipeline NAME
                        Derivatives pipeline containing the preprocessed
                        DWI data, e.g. ``qsiprep``. Default: ``all``.
--t1-preproc-pipeline NAME
                        Derivatives pipeline containing the preprocessed
                        T1w data. Default: same as ``--dwi-preproc-pipeline``.
--bids-filter-file FILE
                        JSON file of pyBIDS entity filters used to select
                        the DWI files, e.g.
                        ``{"acquisition": "64dir", "space": "T1w"}``.
                        A qsiprep style file with a top-level ``"dwi"`` key
                        is also accepted.
--nprocs N              Number of subject/sessions to process in parallel;
                        ``-1`` uses all CPUs. Default: 1.
--parallel-engine NAME  Parallelization engine (``serial``, ``joblib``,
                        ``ray``, ``dask``). When ``--nprocs`` is not 1 and
                        this is left as ``serial``, ``joblib`` is used.
--skip-bids-validation  Do not validate or index metadata with pyBIDS.
                        Speeds up indexing of large datasets.
-v, --verbose           Verbose logging.
-d, --dry-run           Print the recognized arguments without running pyAFQ.
-c TO_CALL, --call TO_CALL
                        AFQ.api attribute to export at the participant level.
                        Defaults to ``all``. Ignored at the group level.

All other pyAFQ optional parameters accepted by ``pyAFQ`` are also accepted
here (``pyAFQ-bids --help`` lists them). Example:

.. code-block:: none

    pyAFQ-bids ~/AFQ_data/HBN participant --dwi-preproc-pipeline qsiprep --participant-label NDARAA948VFH --nprocs 4 --rng_seed=2026
    pyAFQ-bids ~/AFQ_data/HBN group
