import numpy as np
import nibabel as nib
import logging

import dipy.tracking.utils as dtu
import dipy.tracking.streamline as dts

from scipy.spatial.distance import cdist
logger = logging.getLogger('AFQ')


def clean_by_other_density_map(this_bundle_sls, other_bundle_sls,
                               node_thresh, img):
    """
    Cleans a set of streamlines by removing those with significant overlap with 
    another set of streamlines.

    Parameters
    ----------
    this_bundle_sls : array-like
        A list or array of streamlines to be cleaned.
    other_bundle_sls : array-like
        A reference list or array of streamlines to determine overlapping regions.
    node_thresh : int
        The maximum number of nodes allowed to overlap between `this_bundle_sls`
        and `other_bundle_sls`. Streamlines with overlaps beyond this threshold 
        are removed.
    img : nibabel.Nifti1Image or ndarray
        A reference 3D image that defines the spatial dimensions for the density 
        map.

    Returns
    -------
    cleaned_idx : ndarray of bool
        An array of boolean values indicating which streamlines from 
        `this_bundle_sls` pass the overlap threshold (True for streamlines to 
        keep, False for streamlines to discard).

    Notes
    -----
    This function computes a density map from `other_bundle_sls` to represent 
    the spatial occupancy of the streamlines. It then calculates the probability
    of each streamline in `this_bundle_sls` overlapping with this map. 
    Streamlines that overlap in more than `node_thresh` nodes are flagged for 
    removal.

    Examples
    --------
    >>> clean_idx = clean_by_other_density_map(bundle1, bundle2, 5, img)
    >>> cleaned_bundle = [s for i, s in enumerate(bundle1) if clean_idx[i]]
    """
    other_bundle_density_map = dtu.density_map(
        other_bundle_sls, np.eye(4), img.shape[:3])
    fiber_probabilities = dts.values_from_volume(
        other_bundle_density_map, this_bundle_sls, np.eye(4))
    cleaned_idx = np.zeros(len(this_bundle_sls), dtype=np.bool_)
    for ii, fp in enumerate(fiber_probabilities):
        cleaned_idx[ii] = np.sum(np.asarray(fp) >= 1) <= node_thresh
    return cleaned_idx


def clean_relative_to_other_core(core, this_fgarray, other_fgarray, affine):
    """
    Removes streamlines from a set that lie on the opposite side of a specified 
    core axis compared to another set of streamlines.

    Parameters
    ----------
    core : {'anterior', 'posterior', 'superior', 'inferior', 'right', 'left'}
        The anatomical axis used to define the core direction. This determines 
        the side of the core from which streamlines in `this_fgarray` are 
        retained.
    this_fgarray : ndarray
        An array of streamlines to be cleaned.
    other_fgarray : ndarray
        An array of reference streamlines to define the core.
    affine : ndarray
        The affine transformation matrix.

    Returns
    -------
    cleaned_idx_core : ndarray of bool
        An array of boolean values indicating which streamlines in `this_fgarray`
        lie on the correct side of the core (True for streamlines to keep, False 
        for streamlines to discard).

    Notes
    -----
    This function first calculates the median streamline of `other_fgarray`, 
    which acts as the core line. It then determines whether each streamline in 
    `this_fgarray` is on the specified side of this core, based on the specified
    anatomical axis (`core`). Streamlines on the opposite side are flagged for 
    removal.

    Examples
    --------
    >>> cleaned_core_idx = clean_relative_to_other_core('anterior', 
    ...                                                 streamlines1, 
    ...                                                 streamlines2, 
    ...                                                 np.eye(4))
    >>> cleaned_streamlines = [s for i, s in enumerate(streamlines1) 
    ...                        if cleaned_core_idx[i]]
    """
    if len(other_fgarray) == 0:
        logger.warning("Cleaning relative to core skipped, no core found.")
        return np.ones(this_fgarray.shape[0], dtype=np.bool_)

    # find dimension of core axis
    orientation = nib.orientations.aff2axcodes(affine)
    core_axis = None
    core_upper = core[0].upper()
    axis_groups = {
        'L': ('L', 'R'),
        'R': ('L', 'R'),
        'P': ('P', 'A'),
        'A': ('P', 'A'),
        'I': ('I', 'S'),
        'S': ('I', 'S'),
    }

    direction_signs = {
        'L': 1,
        'R': -1,
        'P': 1,
        'A': -1,
        'I': 1,
        'S': -1,
    }

    core_axis = None
    for idx, axis_label in enumerate(orientation):
        if core_upper in axis_groups[axis_label]:
            core_axis = idx
            core_direc = direction_signs[core_upper]
            break

    if affine[core_axis, core_axis] < 0:
        core_direc = -core_direc

    if core_axis is None:
        raise ValueError(f"Invalid core axis: {core}")

    core_bundle = np.median(other_fgarray, axis=0)
    cleaned_idx_core = np.zeros(this_fgarray.shape[0], dtype=np.bool_)
    for ii, sl in enumerate(this_fgarray):
        dist_matrix = cdist(core_bundle, sl, 'sqeuclidean')
        min_dist_indices = np.unravel_index(np.argmin(dist_matrix),
                                            dist_matrix.shape)
        closest_core = core_bundle[min_dist_indices[0], core_axis]
        closest_sl = sl[min_dist_indices[1], core_axis]

        cleaned_idx_core[ii] = core_direc * (closest_sl - closest_core) > 0
    return cleaned_idx_core
