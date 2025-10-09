"""
===================
PyAFQ Endpoint Maps
===================
Here we extract endpoint maps for pyAFQ run under different configurations
for an HBN subject.
"""

####################################################
# Import libraries, load the defautl tract templates
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for headless plotting
import matplotlib.pyplot as plt

import nibabel as nib
import numpy as np

from AFQ.api.group import GroupAFQ
import AFQ.definitions.image as afm
from dipy.data import get_sphere

import os.path as op
import os
import AFQ.data.fetch as afd

from AFQ.viz.utils import COLOR_DICT
from dipy.align import resample

############################################################
# Use an example subject from the Healthy Brain Network (HBN).

subject_id = "NDARKP893TWU"  # Example subject ID
ses_id = "HBNsiteRU"  # Example session ID
_, study_dir = afd.fetch_hbn_preproc([subject_id])

endpoint_maps = {  # Endpoint maps by threshold in mm
    "2": {},
    "3": {},
    "4": {}
}

bundle_names = [  # pyAFQ defaults
    "Left Anterior Thalamic", "Right Anterior Thalamic",
    "Left Cingulum Cingulate", "Right Cingulum Cingulate",
    "Left Corticospinal", "Right Corticospinal",
    "Left Inferior Fronto-occipital", "Right Inferior Fronto-occipital",
    "Left Inferior Longitudinal", "Right Inferior Longitudinal",
    "Left Superior Longitudinal", "Right Superior Longitudinal",
    "Left Arcuate", "Right Arcuate",
    "Left Uncinate", "Right Uncinate",
    "Left Posterior Arcuate", "Right Posterior Arcuate",
    "Left Vertical Occipital", "Right Vertical Occipital",
    "Callosum Anterior Frontal", "Callosum Motor",
    "Callosum Occipital", "Callosum Orbital",
    "Callosum Posterior Parietal", "Callosum Superior Frontal",
    "Callosum Superior Parietal", "Callosum Temporal"
]

###########################################################################
# Compare endpoint maps for single-shell and multi-shell CSD
# For both local (probabilistic) and PFT (particle filtering) tractography.

for odf_model in ["csd", "msmtcsd"]:
    for tracker in ["local", "pft"]:
        output_dir = op.join(
            study_dir, "derivatives",
            f"afq_{odf_model}_{tracker}")

        myafq = GroupAFQ(
            op.join(afd.afq_home, "HBN"),
            participant_labels=[subject_id],
            preproc_pipeline="qsiprep",
            tracking_params={
                "tracker": tracker,
                "odf_model": odf_model,
                "sphere": get_sphere(name="repulsion724"),
                "seed_mask": afm.ScalarImage("wm_gm_interface"),
                "seed_threshold": 0.5,
                "stop_mask": afm.ThreeTissueImage(),
                "stop_threshold": "ACT",
                "n_seeds": 2000000,
                "random_seeds": True},
            output_dir=output_dir,
            endpoint_threshold=None)

        endpoints_maps = myafq.export("endpoint_maps")

        # Copy outputs of first runs for later use
        # Up to but not including tractography
        if odf_model == "csd" and tracker == "local":
            other_output_paths = [
                op.join(study_dir, (
                    "derivatives/afq_csd_pft/"
                    f"sub-{subject_id}/ses-{ses_id}/dwi")),
                op.join(study_dir, (
                    "derivatives/afq_msmtcsd_local/"
                    f"sub-{subject_id}/ses-{ses_id}/dwi")),
                op.join(study_dir, (
                    "derivatives/afq_msmtcsd_pft/"
                    f"sub-{subject_id}/ses-{ses_id}/dwi")),
            ]
            for other_output_path in other_output_paths:
                os.makedirs(other_output_path, exist_ok=True)
                myafq.cmd_outputs(
                    "cp", suffix=other_output_path, up_to="track")

        endpoint_data = nib.load(
            endpoints_maps[subject_id]).get_fdata()
        endpoint_maps["2"][f"{odf_model}_{tracker}"] = \
            np.logical_and(endpoint_data < 2.0, endpoint_data != 0.0)
        endpoint_maps["3"][f"{odf_model}_{tracker}"] = \
            np.logical_and(endpoint_data < 3.0, endpoint_data != 0.0)
        endpoint_maps["4"][f"{odf_model}_{tracker}"] = \
            np.logical_and(endpoint_data < 4.0, endpoint_data != 0.0)

t1 = nib.load(myafq.export("t1_file")[subject_id])
b0 = nib.load(myafq.export("b0")[subject_id])

t1_dwi_space = resample(
    t1.get_fdata(),
    b0.get_fdata(),
    moving_affine=t1.affine,
    static_affine=b0.affine).get_fdata()

# Find the best z-slice for visualization
sum_of_all_maps = np.zeros(endpoint_maps["2"]["csd_local"].shape[:3])
for threshold, maps in endpoint_maps.items():
    for map_name, _map in maps.items():
        sum_of_all_maps += np.sum(_map, axis=-1)
best_z = np.argmax(np.sum(sum_of_all_maps, axis=(0, 1)))

t1_slice = t1_dwi_space[..., best_z]
t1_slice = (t1_slice - t1_slice.min()) / (t1_slice.max() - t1_slice.min())

for threshold, maps in endpoint_maps.items():
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(f"Endpoint Maps for Threshold < {threshold} mm")
    for ii, (map_name, _map) in enumerate(maps.items()):
        image = np.zeros(_map.shape[:2] + (3,))
        image_counts = np.zeros(_map.shape[:2])
        for z in range(_map.shape[3]):
            # Assign each z-slice a unique color from our colormap
            rgb = COLOR_DICT[bundle_names[z]]
            image += _map[..., best_z, z][..., np.newaxis] * rgb
            image_counts += _map[..., best_z, z]

        # Interesting Metrics
        endpoint_voxel_count = np.sum(image_counts)
        median_vc = np.median(np.sum(_map, axis=(0, 1, 2)))

        # Normalize the image by the number of bundles
        image_counts[image_counts == 0] = 1
        image /= image_counts[..., np.newaxis]

        # Blend the T1 slice with the endpoint map
        image = (np.stack([t1_slice] * 3, axis=-1) + image) / 2.0
        axes[ii // 2, ii % 2].imshow(np.rot90(image), interpolation='none')
        axes[ii // 2, ii % 2].axis("off")
        axes[ii // 2, ii % 2].set_title(map_name.replace("_", " ").upper())

        axes[ii // 2, ii % 2].text(
            0.5, 0.1,
            f'Total Endpoint Voxel Count {endpoint_voxel_count}',
            color='white',
            ha='center', va='center',
            transform=axes[ii // 2, ii % 2].transAxes)

        axes[ii // 2, ii % 2].text(
            0.5, 0.05,
            f'Median across bundles {median_vc}',
            color='white',
            ha='center', va='center',
            transform=axes[ii // 2, ii % 2].transAxes)

    plt.tight_layout()
    plt.savefig(f"endpoint_maps_threshold_{threshold}.png")
    plt.close(fig)

###########################################################################
# This Example would take too long to run in the documentation.
# So, we provide the results as static images.
#  .. image:: ../../_static/endpoint_maps_threshold_2.png
#  .. image:: ../../_static/endpoint_maps_threshold_3.png
#  .. image:: ../../_static/endpoint_maps_threshold_4.png
