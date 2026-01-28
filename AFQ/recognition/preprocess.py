import logging
from time import time

import immlib
import nibabel as nib
import numpy as np

import AFQ.recognition.utils as abu

logger = logging.getLogger("AFQ")


@immlib.calc("tol", "dist_to_atlas", "vox_dim")
def tolerance_mm_to_vox(img, dist_to_waypoint, input_dist_to_atlas):
    return abu.tolerance_mm_to_vox(img, dist_to_waypoint, input_dist_to_atlas)


@immlib.calc("fgarray")
def fgarray(tg):
    """
    Streamlines resampled to 20 points.
    """
    logger.info("Resampling Streamlines...")
    start_time = time()
    fg_array = np.array(abu.resample_tg(tg, 20))
    logger.info((f"Streamlines Resampled (time: {time() - start_time}s)"))
    return fg_array


@immlib.calc("crosses")
def crosses(fgarray, img):
    """
    Classify the streamlines by whether they cross the midline.
    Creates a crosses attribute which is an array of booleans. Each boolean
    corresponds to a streamline, and is whether or not that streamline
    crosses the midline.
    """
    # What is the x,y,z coordinate of 0,0,0 in the template space?
    zero_coord = np.dot(np.linalg.inv(img.affine), np.array([0, 0, 0, 1]))

    orientation = nib.orientations.aff2axcodes(img.affine)
    lr_axis = 0
    for idx, axis_label in enumerate(orientation):
        if axis_label in ["L", "R"]:
            lr_axis = idx
            break

    return np.logical_and(
        np.any(fgarray[:, :, lr_axis] > zero_coord[lr_axis], axis=1),
        np.any(fgarray[:, :, lr_axis] < zero_coord[lr_axis], axis=1),
    )


# Things that can be calculated for multiple bundles at once
# (i.e., for a whole tractogram) go here
def get_preproc_plan(img, tg, dist_to_waypoint, dist_to_atlas):
    preproc_plan = immlib.plan(
        tolerance_mm_to_vox=tolerance_mm_to_vox, fgarray=fgarray, crosses=crosses
    )
    return preproc_plan(
        img=img,
        tg=tg,
        dist_to_waypoint=dist_to_waypoint,
        input_dist_to_atlas=dist_to_atlas,
    )
