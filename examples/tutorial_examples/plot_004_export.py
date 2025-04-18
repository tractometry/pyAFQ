"""
======================================
Exporting pyAFQ Results
======================================

This example shows how to use the ``export`` methods to obtain results from
the ParticipantAFQ object. The ``export`` methods are used to calculate
derived quantities from the data, such as DTI parameters, tract profiles,
and bundle segmentations. 

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
# Preparing the ParticipantAFQ object
# -------------------------
# In this example, we will create a ParticipantAFQ object based on the 
# :doc:`plot_002_participant_afq_api` example. Please refer to that
# example for a detailed description of the parameters.

afd.organize_stanford_data(clear_previous_afq = "track")

data_dir = op.join(afd.afq_home, "stanford_hardi", "derivatives", "vistasoft", 
                   "sub-01", "ses-01", "dwi")

dwi_data_file = op.join(data_dir, "sub-01_ses-01_dwi.nii.gz")
bval_file     = op.join(data_dir, "sub-01_ses-01_dwi.bval")
bvec_file     = op.join(data_dir, "sub-01_ses-01_dwi.bvec")

output_dir = op.join(afd.afq_home, "stanford_hardi", "derivatives", "afq", "sub-01")
os.makedirs(output_dir, exist_ok = True)

# Initialize the ParticipantAFQ object
myafq = ParticipantAFQ(
    dwi_data_file = dwi_data_file, 
    bval_file = bval_file,
    bvec_file = bvec_file,
    output_dir = output_dir,
    tracking_params = {
        "n_seeds": 25000,
        "random_seeds": True,
        "rng_seed": 2022,
        "trx": True,
        "num_chunks": True
    },
)

##########################################################################
# The Export Method
# ------------------------------------------------------------------
# The ParticipantAFQ object has a method called ``export``, which allows the user
# to calculate various derived quantities from the data.
# 
# The ``export`` method can be called with a string argument that specifies the
# type of quantity to be calculated. For example, ``myafq.export("OPTIONS")``.
#
# To list the available options, you can call the ``export`` method with the 
# argument "help". This will return a list of all the available options for
# the ``export`` method. 

myafq.export("help")

##########################################################################
# .. note::
#
#    No all options are possible even if they are valid. This will depend on 
#    you dataset and pyAFQ API parameters. For example, you cannot calculate 
#    DKI model from single shell data. Please refer to 
#    :doc:`/howto/usage/tractography_params` for more documentation.


##########################################################################
# Calculating DTI FA (Diffusion Tensor Imaging Fractional Anisotropy)
# ------------------------------------------------------------------
# FA can be computed using the DTI model, by explicitly calling 
# ``myafq.export("dti_fa")``. This triggers the computation of DTI parameters, 
# and stores the results in the AFQ derivatives directory. In addition, it 
# calculates the FA from these parameters and stores it in a different file in
# the same directory.
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
# The Export All Method
# -----------------------------------------------------
# There is a ``export_all`` method that will export all the results from the
# AFQ pipeline. Undeerneath the hood, ``export_all`` calls a series of ``export`` 
# methods. This method was added for convenience, and is the recommended method 
# to use when you want to export everything the results from the AFQ pipeline.
# 
# This method will export the following results, if possible:
# - Transformation maps and files
# - Start and stop mask images, associated diffusion scalar files
# - Tractography, segmented tractography in to bundles
# - Tract profiles, streamline counts, median tract length
# - Visuzalizations of the bundles and tract profiles

myafq.export_all()

##########################################################################
# The Export Up To Method
# -----------------------------------------------------
# The ``export_up_to`` method allows you to export results up to a certain
# point (but not including) in the AFQ pipeline. 
#
# For example, if you want to export all the results up to the bundle 
# segmentation step, you can call the ``export_up_to`` method with the
# argument "bundles". This will export all the required derivatives 
# prior to the bundle segmentation step, where you can then take the 
# derivatives and debug your own custom segmentation pipeline. 

myafq.export_up_to("bundles")