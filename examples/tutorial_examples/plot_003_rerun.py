"""
================
Re-running pyAFQ
================
Sometimes you want to change arguments and re-run pyAFQ. This could be to
try different parameters, to run on changed data, or re-run after updating
parameters due to an error.

pyAFQ saves derivatives as it goes, so if you re-run pyAFQ after changing
parameters, it could use derivatives from previous runs with the
old parameters.

To solve this, use the myafq.clobber() or myafq.cmd_outputs() methods. They
are the same methods. They will delete previous derivatives so you can
re-run your pipeline.
"""
from AFQ.api.group import GroupAFQ
import AFQ.data.fetch as afd
import os.path as op
import os


##########################################################################
# We start with some example data. The data we will use here is
# generated from the
# `Stanford HARDI dataset <https://purl.stanford.edu/ng782rw8378>`_.
# We then setup our myafq object which we will use to demonstrate
# the clobber method.

afd.organize_stanford_data()

tracking_params = dict(n_seeds=100,
                       random_seeds=True,
                       rng_seed=2022,
                       trx=True,
                       num_chunks=True)

myafq = GroupAFQ(
    bids_path=op.join(afd.afq_home, 'stanford_hardi'),
    preproc_pipeline='vistasoft',
    tracking_params=tracking_params)

###################
# Delete Everything
# -----------------
# To delete all pyAFQ outputs in the output directory, simply call::
#
#     myafq.cmd_outputs()
#
# or::
#
#     myafq.clobber()
#
# This is equivalent to running ``rm -r`` on all pyAFQ outputs. After this,
# you can re-run your pipeline from scratch.
#
# Here, we will delete everything and re-run with a different b0 threshold.
# The b0_threshold determines which b-values are considered b0.
# The default is 50.

myafq.cmd_outputs()

myafq = GroupAFQ(
    bids_path=op.join(afd.afq_home, 'stanford_hardi'),
    preproc_pipeline='vistasoft',
    b0_threshold=100,
    tracking_params=tracking_params)

myafq.export("b0")

####################
# Delete Some Things
# ------------------
# To delete only specific types of derivatives while preserving others,
# use the ``dependent_on`` parameter::
#
#     # Delete only tractography-dependent files
#     myafq.cmd_outputs(dependent_on="track")
#
#     # Delete only bundle recognition-dependent files
#     myafq.cmd_outputs(dependent_on="recog")
#
#     # Delete only profiling-dependent files
#     myafq.cmd_outputs(dependent_on="prof")
#
# You can also specify exceptions - files to preserve::
#
#     # Delete all outputs except the tractography
#     myafq.cmd_outputs(exceptions=["streamlines"])
#
# Here, we will change the tractography parameters, but we want to keep all
# derivatives not dependent on tractography. Typically, this means keeping
# The mapping from MNI space and fitted models, but deleting recognized
# bundles and tract profiles.

myafq.clobber(dependent_on="track")

tracking_params = dict(n_seeds=100,
                       random_seeds=True,
                       max_angle=60,
                       rng_seed=12,
                       trx=True,
                       num_chunks=True)

myafq = GroupAFQ(
    bids_path=op.join(afd.afq_home, 'stanford_hardi'),
    preproc_pipeline='vistasoft',
    b0_threshold=100,
    tracking_params=tracking_params)

myafq.export("streamlines")

##################
# Move Some Things
# ----------------
# The ``cmd_outputs`` method is flexible and can perform other file operations
# besides deletion. For example, to copy files::

#     # Copy only files dependent on tractography
#     myafq.cmd_outputs(
#           cmd="cp",
#           dependent_on="track",
#           suffix="/path/to/backup/")

# Note: The method automatically adds ``-r``
#       for "cp" and "rm" operations.

# Create backup directory
backup_dir = op.join(afd.afq_home, "stanford_hardi_backup")
os.makedirs(backup_dir, exist_ok=True)

# Move the outupts of AFQ to this directory
myafq.cmd_outputs(cmd="mv", suffix=backup_dir)

##############
# How It Works
# ------------
# The method works by:
# 1. Identifying all pyAFQ outputs in the output directory
# 2. Filtering based on the ``dependent_on`` parameter (if provided)
# 3. Removing any files listed in ``exceptions``
# 4. Executing the specified command (``rm``, ``mv``, ``cp`` etc.) on the remaining files
# 5. Resetting the workflow to ensure subsequent runs regenerate affected derivatives
#
# We plan to automate this process in the future.
