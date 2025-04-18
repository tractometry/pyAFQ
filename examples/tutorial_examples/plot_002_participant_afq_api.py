"""
======================================
Getting started with pyAFQ - ParticipantAFQ
======================================
"""

import os
import os.path as op

import matplotlib.pyplot as plt
import nibabel as nib
import plotly
import pandas as pd

from AFQ.api.participant import ParticipantAFQ
import AFQ.data.fetch as afd
import AFQ.viz.altair as ava

##########################################################################
# Example data
# ------------
# The following call dowloads a dataset that contains a single subject"s
# high angular resolution diffusion imaging (HARDI) data, collected at the
# Stanford Vista Lab
#
# .. note::
#   See https://purl.stanford.edu/ng782rw8378 for details on dataset.
#
# The data are downloaded and organized locally into a BIDS compliant
# anatomical data folder (``anat``) and a diffusion-weighted imaging data
# (``dwi``) folder, which are both placed in the user's home directory under::
#
#   ``~/AFQ_data/stanford_hardi/``
#
# The data is also placed in a derivatives directory, signifying that it has
# already undergone the required preprocessing necessary for pyAFQ to run.
#
# The clear_previous_afq is used to remove any previous runs of the afq object
# stored in the ``~/AFQ_data/stanford_hardi/`` BIDS directory. Set it to None if
# you want to use the results of previous runs.

afd.organize_stanford_data(clear_previous_afq = "track")

##########################################################################
# Defining data files
# --------------------
# If your data is not in BIDS format, you can still use pyAFQ. If you have BIDS
# compliant dataset, you can use ``GroupAFQ`` instead (:doc:`plot_001_group_afq_api`). 
# Otherwise, You will need to define the data files that you want to use. In 
# this case, we will define the data files for the subject we downloaded above. 
# The data files are located in the ``~/AFQ_data/stanford_hardi/derivatives/vistasoft``
# directory, and are organized into a BIDS compliant directory structure. The 
# data files are located in the ``dwi`` directories. The data files
# are:  
#
# - ``sub-01_dwi.nii.gz`` (the diffusion-weighted imaging data)
# - ``sub-01_dwi.bval``   (the b-values)
# - ``sub-01_dwi.bvec``   (the b-vectors)

data_dir = op.join(afd.afq_home, "stanford_hardi", "derivatives", "vistasoft", 
                   "sub-01", "ses-01", "dwi")

dwi_data_file = op.join(data_dir, "sub-01_ses-01_dwi.nii.gz")
bval_file     = op.join(data_dir, "sub-01_ses-01_dwi.bval")
bvec_file     = op.join(data_dir, "sub-01_ses-01_dwi.bvec")

# You will also need to define the output directory where you want to store the
# results. The output directory needs to exist before exporting ParticipantAFQ
# results.

output_dir = op.join(afd.afq_home, "stanford_hardi", "derivatives", "afq", "sub-01")
os.makedirs(output_dir, exist_ok = True)

##########################################################################
# Set tractography parameters (optional)
# ---------------------------------------
# We make create a `tracking_params` variable, which we will pass to the
# ParticipantAFQ object which specifies that we want 25,000 seeds randomly
# distributed in the white matter. We only do this to make this example
# faster and consume less space. We also set ``num_chunks`` to `True`,
# which will use ray to parallelize the tracking across all cores.
# This can be removed to process in serial, or set to use a particular
# distribution of work by setting `n_chunks` to an integer number.

tracking_params = dict(n_seeds=25000,
                       random_seeds=True,
                       rng_seed=2022,
                       trx=True,
                       num_chunks=True)

##########################################################################
# Initialize a ParticipantAFQ object:
# -------------------------
#
# Creates a ParticipantAFQ object, that encapsulates tractometry. This object 
# can be used to manage the entire :doc:`/explanations/tractometry_pipeline`, including:
#
# - Tractography
# - Registration
# - Segmentation
# - Cleaning
# - Profiling
# - Visualization
#
# To initialize the object, we will pass in the diffusion data files and specify
# the output directory where we want to store the results. We will also
# pass in the tracking parameters we defined above. 

myafq = ParticipantAFQ(
    dwi_data_file = dwi_data_file, 
    bval_file = bval_file,
    bvec_file = bvec_file,
    output_dir = output_dir,
    tracking_params = tracking_params,
)

##########################################################################
# Calculating DTI FA (Diffusion Tensor Imaging Fractional Anisotropy)
# ------------------------------------------------------------------
# The ParticipantAFQ object has a method called ``export``, which allows the user
# to calculate various derived quantities from the data.
#
# For example, FA can be computed using the DTI model, by explicitly
# calling ``myafq.export("dti_fa")``. This triggers the computation of DTI
# parameters, and stores the results in the AFQ derivatives directory. 
# In addition, it calculates the FA from these parameters and stores it in a 
# different file in the same directory.
#
# .. note::
#
#    The AFQ API computes quantities lazily. This means that DTI parameters
#    are not computed until they are required. This means that the first
#    line below is the one that requires time.
#
# The result of the call to ``export`` is the filename of the corresponding FA 
# files.

FA_fname = myafq.export("dti_fa")

##########################################################################
# We will then use ``nibabel`` to load the deriviative file and retrieve the
# data array.

FA_img = nib.load(FA_fname)
FA = FA_img.get_fdata()

##########################################################################
# Visualize the result with Matplotlib
# -------------------------------------
# At this point ``FA`` is an array, and we can use standard Python tools to
# visualize it or perform additional computations with it.
#
# In this case we are going to take an axial slice halfway through the
# FA data array and plot using a sequential color map.
#
# .. note::
#
#    The data array is structured as a xyz coordinate system.

fig, ax = plt.subplots(1)
ax.matshow(FA[:, :, FA.shape[-1] // 2], cmap="viridis")
ax.axis("off")

##########################################################################
# Recognizing the bundles and calculating tract profiles:
# -----------------------------------------------------
# Typically, users of pyAFQ are interested in calculating not only an overall
# map of the FA, but also the major white matter pathways (or bundles) and
# tract profiles of tissue properties along their length. To trigger the
# pyAFQ pipeline that calculates the profiles, users can call the
# ``export("profiles")`` method:
#
# .. note::
#    Running the code below triggers the full pipeline of operations
#    leading to the computation of the tract profiles. Therefore, it
#    takes a little while to run (about 40 minutes, typically).

myafq.export("profiles")

##########################################################################
# Visualizing the bundles and calculating tract profiles:
# -----------------------------------------------------
# The pyAFQ API provides several ways to visualize bundles and profiles.
#
# First, we will run a function that exports an html file that contains
# an interactive visualization of the bundles that are segmented.
#
# .. note::
#    By default we resample a 100 points within a bundle, however to reduce
#    processing time we will only resample 50 points.
#
# Once it is done running, it should pop a browser window open and let you
# interact with the bundles.
#
# .. note::
#    You can hide or show a bundle by clicking the legend, or select a
#    single bundle by double clicking the legend. The interactive
#    visualization will also all you to pan, zoom, and rotate.

bundle_html = myafq.export("all_bundles_figure")
plotly.io.show(bundle_html[0])

##########################################################################
# We can also visualize the tract profiles in all of the bundles. These
# plots show both FA (left) and MD (right) layed out anatomically.
# To make this plot, it is required that you install with
# ``pip install pyAFQ[plot]`` so that you have the necessary dependencies.
#

fig_files = myafq.export("tract_profile_plots")

##########################################################################
# .. figure:: {{ fig_files[0] }}
#

