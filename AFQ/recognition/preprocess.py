import logging
from time import time

import immlib
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
def crosses(fgarray):
    """
    Classify the streamlines by whether they cross the midline.
    Creates a crosses attribute which is an array of booleans. Each boolean
    corresponds to a streamline, and is whether or not that streamline
    crosses the midline.
    """
    return np.logical_and(
        np.any(fgarray[:, :, 0] > 0, axis=1),
        np.any(fgarray[:, :, 0] < 0, axis=1),
    )


@immlib.calc("lengths")
def lengths(fgarray):
    """
    Calculate the lengths of the streamlines.
    Using resampled fgarray biases lengths to be lower. However,
    this is not meant to be a precise selection requirement, and
    is more meant for efficiency.
    """
    segments = np.diff(fgarray, axis=1)
    segment_lengths = np.sqrt(np.sum(segments**2, axis=2))
    return np.sum(segment_lengths, axis=1)


@immlib.calc("endpoint_dists")
def endpoint_dists(fgarray):
    """
    Calculate the distances between the endpoints of the streamlines.
    """
    return np.linalg.norm(fgarray[:, 0, :] - fgarray[:, -1, :], axis=1)


# Things that can be calculated for multiple bundles at once
# (i.e., for a whole tractogram) go here
def get_preproc_plan(img, tg, dist_to_waypoint, dist_to_atlas):
    preproc_plan = immlib.plan(
        tolerance_mm_to_vox=tolerance_mm_to_vox,
        fgarray=fgarray,
        crosses=crosses,
        lengths=lengths,
        endpoint_dists=endpoint_dists,
    )
    return preproc_plan(
        img=img,
        tg=tg,
        dist_to_waypoint=dist_to_waypoint,
        input_dist_to_atlas=dist_to_atlas,
    )
