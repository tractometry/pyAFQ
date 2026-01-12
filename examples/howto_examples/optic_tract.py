"""
===============================================================
How to track the Optic Tract and Posterior Optic Nerve in pyAFQ
===============================================================

pyAFQ is designed to be customizable and extensible, even to
relatively small bundles. This example will be based on the work 
of Kruper et al. [1]_. Here, part of the trick is that most preprocessing
pipelines will cut off a portion or all of the optic nerve. So, we
only attempt to track the most posterior portion of the optic nerve,
the optic tract, and the optic chiasm. In addition, we will use our own
custom PVE maps from the T1-weighted image to help with segmentation.

"""

import plotly
import os.path as op

import AFQ.api.bundle_dict as abd
from AFQ.api.group import GroupAFQ

import AFQ.data.fetch as afd
from AFQ.definitions.image import RoiImage
import AFQ.definitions.image as afm

#############################################################################
# Get dMRI data
# ---------------
# We will analyze one subject from the Healthy Brain Network Processed Open
# Diffusion Derivatives dataset (HBN-POD2) [2]_, [3]_. We'll use a fetcher to
# get preprocessed dMRI data for one of the >2,000 subjects in that study. The
# data gets organized into a BIDS-compatible format in the `~/AFQ_data/HBN`
# folder:

study_dir = afd.fetch_hbn_preproc(["NDARAA948VFH"])[1]

#############################################################################
# Define Optic Tract and Optic Nerve `BundleDict` object
# --------------------------------
# The `BundleDict` object holds information about the ROIs used to define the
# optic tract and optic nerve bundles. In this case, there is also a curvature
# criterion applied to the optic tract bundles to help separate them from
# other nearby bundles. Additionally, qb_thresh is set to 6 to use a clustering
# approach to cleaning as opposed to our standard mahalanobis distance approach.

oton_rois = afd.read_oton_templates(as_img=False)

otoc_bd = abd.BundleDict({
        "Left Optic": {
            "include": [
                oton_rois["left_OT_1"],
                oton_rois["left_OT_0"],
                oton_rois["left_ON_0"],
            ],
            "curvature": {"path": oton_rois["left_OTOC_curve"], "thresh": 20, "cut": True},
            "qb_thresh": 6
        },
        "Right Optic": {
            "include": [
                oton_rois["right_OT_1"],
                oton_rois["right_OT_0"],
                oton_rois["right_ON_0"],
            ],
            "curvature": {"path": oton_rois["right_OTOC_curve"], "thresh": 20, "cut": True},
            "qb_thresh": 6
        },
        "Left Optic Cross": {
            "include": [
                oton_rois["left_OT_1"],
                oton_rois["left_OT_0"],
                oton_rois["right_ON_0"],
            ],
            "qb_thresh": 6
        },
        "Right Optic Cross": {
            "include": [
                oton_rois["right_OT_1"],
                oton_rois["right_OT_0"],
                oton_rois["left_ON_0"],
            ],
            "qb_thresh": 6
        },
})

#############################################################################
# Tractography and segmentation parameters
# ----------------------------------------
# Here, we define some custom parameters for tractography and segmentation.
# For tractography, we use a higher max_angle to account for the
# sharp turn the optic tract makes around the midbrain. Additionally, we seed
# densely around the ROIs.
# For segmentation, we use more lenient cleaning parameters to account for
# the small size of these bundles. 

tractography_params = {
    "seed_mask": RoiImage(),
    "n_seeds": 20,
    "random_seeds": False,
    "max_angle": 60,
    "trx": True,
}

segmentation_params = {
    "cleaning_params": {
        "distance_threshold": 2,
        "length_threshold": 3,
    }
}

#############################################################################
# Define PVE images for segmentation
# ----------------------------------
# Finally, we define the PVE images that will be used to guide tracking.
# For these bundles in particular, this is the trickiest part. Portions of
# the optic nerve often have low FA, fall outside of the brain mask, or are
# simply misclassified as gray matter or CSF. In this case, we threshold
# on the T1-weighted image using manually set thresholds. We only divide it into
# gray and white matter and accept all streamlines (we do not attempt to filter
# out streamlines terminating in the CSF; these are normally handled in the bundle
# recognition and cleaning steps). In your case, this can also be done manually,
# or done using DIPY's Markov Random Field (MRF;
# https://docs.dipy.org/stable/examples_built/segmentation/tissue_classification.html).
# Just remember to either use the unmasked T1 (as we do here) or ensure that
# the brain mask includes the optic nerve.
# 
# In the event that tractography fails,
# you can also check pyAFQ's outputs to see the PVE images and white matter gray
# matter interface (wmgmi) that pyAFQ used for tractography. You can then adjust
# pyAFQ parameters, delete these files, and re-run accordingly.

pve = afm.PVEImages(
    afm.ThresholdedScalarImage(
        "t1_file",
        upper_bound=0),
    afm.ThresholdedScalarImage(
        "t1_file",
        upper_bound=180),
    afm.ThresholdedScalarImage(
        "t1_file",
        lower_bound=180))

#############################################################################
# Define GroupAFQ object
# ----------------------
# Finally, we define the `GroupAFQ` object and export all the results.

my_afq = GroupAFQ(
    bids_path=study_dir,
    dwi_preproc_pipeline="qsiprep",
    participant_labels=["NDARAA948VFH"],
    output_dir=op.join(study_dir, "derivatives", "afq_otoc"),
    pve=pve,
    tracking_params=tractography_params,
    ray_n_cpus=4,
    segmentation_params=segmentation_params,
    bundle_info=otoc_bd)

my_afq.export_all()

bundle_html = my_afq.export("all_bundles_figure")
plotly.io.show(bundle_html["NDARAA948VFH"])

#############################################################################
# References
# ----------
# .. [1] Kruper, John, and Ariel Rokem. "Automatic fast and reliable
#     recognition of a small brain white matter bundle." International
#     Workshop on Computational Diffusion MRI. Cham: Springer Nature
#     Switzerland, 2023.
#
# .. [2] Alexander LM, Escalera J, Ai L, et al. An open resource for
#     transdiagnostic research in pediatric mental health and learning
#     disorders. Sci Data. 2017;4:170181.
#
# .. [3] Richie-Halford A, Cieslak M, Ai L, et al. An analysis-ready and quality
#     controlled resource for pediatric brain white-matter research. Scientific
#     Data. 2022;9(1):1-27.
