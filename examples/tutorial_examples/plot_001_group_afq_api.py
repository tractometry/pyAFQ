"""
======================================
Getting started with pyAFQ - GroupAFQ
======================================

There are two ways to :doc:`use pyAFQ </tutorials/index>`: through the
command line interface, and by writing Python code. This tutorial will walk you
through the basics of the latter, using pyAFQ's Python Application Programming
Interface (API).

"""
import os.path as op

import matplotlib.pyplot as plt
import nibabel as nib
import plotly
import pandas as pd

from AFQ.api.group import GroupAFQ
import AFQ.data.fetch as afd
import AFQ.viz.altair as ava
import AFQ.definitions.image as afm

##########################################################################
# Example data
# ------------
# pyAFQ can be called using GroupAFQ to handle data
# organized in a BIDS compliant directory.
# If this is not the case, refer to the Participant AFQ example.
# To get users started with this tutorial, we will download some example
# data and organize it in a BIDS compliant way (for more details on how
# BIDS is used in pyAFQ, refer to :doc:`plot_006_bids_layout`).
#
# The following call dowloads a a single subject's data from the Healthy Brain
# Network Processed Open Diffusion Derivatives dataset (HBN-POD2) [1]_, [2]_
# and organizes it in BIDS in the user's home directory under::
#
#   ``~/AFQ_data/HBN/``
#
# The data is also placed in a derivatives directory, signifying that it has
# already undergone the required preprocessing necessary for pyAFQ to run.
#
# The clear_previous_afq is used to remove any previous runs of the afq object
# stored in the `~/AFQ_data/HBN/` BIDS directory. Set it to None if
# you want to use the results of previous runs.

bids_path = afd.fetch_hbn_preproc(
    ["NDARAA948VFH"],
    clear_previous_afq="all")[1]

##########################################################################
# Set tractography parameters (optional)
# ---------------------------------------
# We make create a `tracking_params` variable, which we will pass to the
# GroupAFQ object which specifies that we want 50,000 seeds randomly
# distributed in the white matter. We only do this to make this example faster
# and consume less space; normally, we use more seeds

tracking_params = dict(n_seeds=50000,
                       random_seeds=True,
                       rng_seed=2025,
                       odf_model="csd_aodf",
                       trx=False)

#####################################################################
# Define PVE images (optional)
# ----------------------------
# To improve segmentation and tractography results, we can provide
# partial volume estimate (PVE) images for the cerebrospinal fluid (CSF),
# gray matter (GM), and white matter (WM). Here, we define these images
# using the AFQ.definitions.image.PVEImages class, which takes as input
# three AFQ.definitions.image.ImageFile objects, one for each tissue type.
# One can also provide a single PVE image with all three tissue types
# using the AFQ.definitions.image.PVEImage class. Finally, by default,
# if no PVE images are provided, pyAFQ will use SynthSeg2 to compute
# these images.

pve = afm.PVEImages(
    afm.ImageFile(
        suffix="probseg", filters={"scope": "qsiprep", "label": "CSF"}),
    afm.ImageFile(
        suffix="probseg", filters={"scope": "qsiprep", "label": "GM"}),
    afm.ImageFile(
        suffix="probseg", filters={"scope": "qsiprep", "label": "WM"}))

##########################################################################
# Initialize a GroupAFQ object:
# -------------------------
#
# Creates a GroupAFQ object, that encapsulates tractometry. This object can be
# used to manage the entire :doc:`/explanations/index`, including:
#
# - Tractography
# - Registration
# - Segmentation
# - Cleaning
# - Profiling
# - Visualization
#
# This will also create an output folder for the corresponding AFQ derivatives
# in the AFQ data directory: ``AFQ_data/HBN/derivatives/afq/``
#
# To initialize this object we will pass in the path location to our BIDS
# compliant data, the name of the preprocessing pipeline we want to use, 
# the name of the t1 preprocessing pipeline we want to use (in this case,
# its the same, qsiprep [3]), the participant labels we want to process
# (in this case, just a single subject), the PVE images we defined above, and
# the tracking parameters we defined above. We set ray_n_cpus=1 and
# low_memory=True to avoid memory issues running this example on
# Github actions. In most other cases, these should not be necessary.

myafq = GroupAFQ(
    bids_path=op.join(afd.afq_home, 'HBN'),
    preproc_pipeline='qsiprep',
    t1_pipeline='qsiprep',
    participant_labels=['NDARAA948VFH'],
    pve=pve,
    tracking_params=tracking_params,
    ray_n_cpus=1,
    low_memory=True)

##########################################################################
# Calculating DKI FA (Diffusion Kurtosis Imaging Fractional Anisotropy)
# ------------------------------------------------------------------
# The GroupAFQ object has a method called `export`, which allows the user
# to calculate various derived quantities from the data.
#
# For example, FA can be computed using the DKI model, by explicitly
# calling `myafq.export("dki_fa")`. This triggers the computation of DKI
# parameters for all subjects in the dataset, and stores the results in
# the AFQ derivatives directory. In addition, it calculates the FA
# from these parameters and stores it in a different file in the same
# directory.
#
# .. note::
#
#    The AFQ API computes quantities lazily. This means that DKI parameters
#    are not computed until they are required. This means that the first
#    line below is the one that requires time.
#
# The result of the call to `export` is a dictionary, with the subject
# IDs as keys, and the filenames of the corresponding files as values.
# This means that to extract the filename corresponding to the FA of the first
# subject, we can do:

FA_fname = myafq.export("dki_fa", collapse=False)["NDARAA948VFH"]["HBNsiteRU"]

# We will then use `nibabel` to load the deriviative file and retrieve the
# data array.

FA_img = nib.load(FA_fname)
FA = FA_img.get_fdata()

##########################################################################
# Visualize the result with Matplotlib
# -------------------------------------
# At this point `FA` is an array, and we can use standard Python tools to
# visualize it or perform additional computations with it.
#
# In this case we are going to take an axial slice halfway through the
# FA data array and plot using a sequential color map.
#
# .. note::
#
#    The data array is structured as a xyz coordinate system.

fig, ax = plt.subplots(1)
ax.matshow(FA[:, :, FA.shape[-1] // 2], cmap='viridis')
ax.axis("off")

##########################################################################
# Recognizing the bundles and calculating tract profiles:
# -----------------------------------------------------
# Typically, users of pyAFQ are interested in calculating not only an overall
# map of the FA, but also the major white matter pathways (or bundles) and
# tract profiles of tissue properties along their length. To trigger the
# pyAFQ pipeline that calculates the profiles, users can call the
# `export('profiles')` method:
#
# .. note::
#    Running the code below triggers the full pipeline of operations
#    leading to the computation of the tract profiles. Therefore, it
#    takes a little while to run (about 40 minutes, typically).

myafq.export('profiles')

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

bundle_html = myafq.export("all_bundles_figure", collapse=False)
plotly.io.show(bundle_html["NDARAA948VFH"]["HBNsiteRU"][0])

##########################################################################
# We can also visualize the tract profiles in all of the bundles. These
# plots show both FA (left) and MD (right) layed out anatomically.
# To make this plot, it is required that you install with
# `pip install pyAFQ[plot]` so that you have the necessary dependencies.
#

fig_files = myafq.export("tract_profile_plots", collapse=False)[
    "NDARAA948VFH"]["HBNsiteRU"]

##########################################################################
# .. figure:: {{ fig_files[0] }}
#


##########################################################################
# We can even use altair to visualize the tract profiles in all
# of the bundles. We provide a more customizable interface for visualizing
# the tract profiles using altair.
# Again, to make this plot, it is required that you install with
# `pip install pyAFQ[plot]` so that you have the necessary dependencies.
#
profiles_df = myafq.combine_profiles()
altair_df = ava.combined_profiles_df_to_altair_df(
    profiles_df,
    tissue_properties=['dki_fa', 'dki_md'])
altair_chart = ava.altair_df_to_chart(altair_df)
altair_chart.display()


##########################################################################
# We can check the number of streamlines per bundle, to make sure
# every bundle is found with a reasonable amount of streamlines.

bundle_counts = pd.read_csv(
    myafq.export("sl_counts", collapse=False)[
        "NDARAA948VFH"]["HBNsiteRU"], index_col=[0])
for ind in bundle_counts.index:
    if ind == "Total Recognized":
        threshold = 1000
    elif "Fronto-occipital" in ind or "Orbital" in ind:
        threshold = 5
    else:
        threshold = 15
    if bundle_counts["n_streamlines"][ind] < threshold:
        raise ValueError((
            "Small number of streamlines found "
            f"for bundle(s):\n{bundle_counts}"))


#############################################################################
# References
# ----------
# .. [1] Alexander LM, Escalera J, Ai L, et al. An open resource for
#     transdiagnostic research in pediatric mental health and learning
#     disorders. Sci Data. 2017;4:170181.
#
# .. [2] Richie-Halford A, Cieslak M, Ai L, et al. An analysis-ready and quality
#     controlled resource for pediatric brain white-matter research. Scientific
#     Data. 2022;9(1):1-27.
#
# .. [3] Cieslak M, Cook PA, He X, et al. QSIPrep: an integrative platform for
#     preprocessing and reconstructing diffusion MRI data. Nat Methods.
#     2021;18(7):775-778.
