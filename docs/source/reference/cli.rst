.. _cli-label:

The pyAFQ CLI
~~~~~~~~~~~~~

pyAFQ can be called from the command line. The following usage is available:

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
--------------------

``dwi``
   Path to DWI data file.

``t1``
   Path to T1-weighted image file. Must already be registered to the DWI
   data, though not resampled.

``o_folder``
   Path to output folder.

Options
-------

-h, --help              Show this help message and exit.
--bval BVAL             Path to bval file. If none, the DWI data file path
                        will be used to find it.
--bvec BVEC             Path to bvec file. If none, the DWI data file path
                        will be used to find it.
-v, --verbose           Verbose when reading the TOML file.
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
    
    pyAFQ /home/john/AFQ_data/HBN/derivatives/qsiprep/sub-NDARAA948VFH/ses-HBNsiteRU/dwi/sub-NDARAA948VFH_ses-HBNsiteRU_acq-64dir_space-T1w_desc-preproc_dwi.nii.gz /home/john/AFQ_data/HBN/derivatives/qsiprep/sub-NDARAA948VFH/anat/sub-NDARAA948VFH_desc-preproc_T1w.nii.gz /home/john/AFQ_data/HBN/derivatives/afq/sub-NDARAA948VFH/ses-HBNsiteRU/dwi --rng_seed=2026 --return_idx=True --pve="multiaxial+brainchop+synthseg"
