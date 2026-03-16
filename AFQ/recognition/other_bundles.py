import logging

import dipy.tracking.streamline as dts
import dipy.tracking.utils as dtu
import nibabel as nib
import numpy as np
from scipy.spatial.distance import cdist

import AFQ.recognition.utils as abu

logger = logging.getLogger("AFQ")


def clean_by_overlap(
    this_bundle_sls,
    other_bundle_sls,
    overlap,
    img,
    remove=False,
    project=None,
    other_bundle_min_density=0.05,
):
    """
    Cleans a set of streamlines by only keeping (or removing) those with
    significant overlap with another set of streamlines.

    Parameters
    ----------
    this_bundle_sls : array-like
        A list or array of streamlines to be cleaned.
        Assumed to be in RASMM space.
    other_bundle_sls : array-like
        A reference list or array of streamlines to determine overlapping regions.
    overlap : int
        The minimum number of nodes allowed to overlap between `this_bundle_sls`
        and `other_bundle_sls`. Streamlines with overlaps beyond this threshold
        are removed.
    img : nibabel.Nifti1Image or ndarray
        A reference 3D image that defines the spatial dimensions for the density
        map.
    remove : bool, optional
        If True, streamlines that overlap in less than `overlap` nodes are
        removed. If False, streamlines that overlap in more than `overlap` nodes
        are removed.
        Default: False.
    project : {'A/P', 'I/S', 'L/R', None}, optional
        If specified, the overlap calculation is projected along the given axis
        before cleaning. For example, 'A/P' projects the streamlines along the
        anterior-posterior axis.
        Default: None.
    other_bundle_min_density : float, optional
        A threshold to binarize the density map of `other_bundle_sls`. Voxels
        with density values above this threshold (as a fraction of the maximum
        density) are considered occupied.
        Default: 0.05.

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
    Streamlines that overlap in less than `overlap` nodes are flagged for
    removal (or more, if remove is True).

    Examples
    --------
    >>> clean_idx = clean_by_overlap(bundle1, bundle2, 5, img, True)
    >>> cleaned_bundle = [s for i, s in enumerate(bundle1) if clean_idx[i]]
    """
    other_bundle_density_map = dtu.density_map(
        other_bundle_sls, img.affine, img.shape[:3]
    )

    if remove:
        max_val = other_bundle_density_map.max()
        if max_val > 0:
            other_bundle_density_map = (
                other_bundle_density_map / max_val
            ) > other_bundle_min_density
        else:
            other_bundle_density_map = np.zeros_like(
                other_bundle_density_map, dtype=bool
            )

    if project is not None:
        orientation = nib.orientations.aff2axcodes(img.affine)
        core_axis = next(
            idx for idx, label in enumerate(orientation) if label in project.upper()
        )

        projection = np.sum(other_bundle_density_map, axis=core_axis)

        other_bundle_density_map = np.broadcast_to(
            np.expand_dims(projection, axis=core_axis), other_bundle_density_map.shape
        )

    fiber_probabilities = dts.values_from_volume(
        other_bundle_density_map, this_bundle_sls, img.affine
    )
    cleaned_idx = np.zeros(len(this_bundle_sls), dtype=np.bool_)
    for ii, fp in enumerate(fiber_probabilities):
        if remove:
            cleaned_idx[ii] = np.sum(np.asarray(fp) >= 1) <= overlap
        else:
            cleaned_idx[ii] = np.sum(np.asarray(fp) >= 1) > overlap
    return cleaned_idx


def clean_relative_to_other_core(
    core,
    this_fgarray,
    other_fgarray,
    consideration,
):
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
        Assumed to be in RASMM space.
    other_fgarray : ndarray
        An array of reference streamlines to define the core.
        Assumed to be in RASMM space.
    consideration : float or string, optional
        If float, the distance threshold (in voxels) for considering a
        streamline's position relative to the core. All points on
        the streamline within distance from the core are considered
        when determining if the streamline lies on the correct side.
        If string, must be one of 'entire' or 'closest'.
        If 'entire', the entire streamline must lie on the correct
        side of the core to be retained.
        If 'closest', only the closest point on the streamline to
        the core is considered.

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
    core_axis = abu.axes_dict[core[0].upper()]

    direction_signs = {
        "L": 1,
        "R": -1,
        "P": 1,
        "A": -1,
        "I": 1,
        "S": -1,
    }

    core_direc = direction_signs[core[0].upper()]

    core_bundle = np.median(other_fgarray, axis=0)
    cleaned_idx_core = np.zeros(this_fgarray.shape[0], dtype=np.bool_)
    for ii, sl in enumerate(this_fgarray):
        if isinstance(consideration, float):
            dist_matrix = cdist(core_bundle, sl, "sqeuclidean")
            closest_core_indices = np.argmin(dist_matrix, axis=0)

            min_dists_sq = np.min(dist_matrix, axis=0)
            within_threshold = min_dists_sq < consideration**2
            if np.any(within_threshold):
                relevant_sl_pts = sl[within_threshold, core_axis]
                relevant_core_pts = core_bundle[
                    closest_core_indices[within_threshold], core_axis
                ]

                cleaned_idx_core[ii] = np.all(
                    core_direc * (relevant_sl_pts - relevant_core_pts) > 0
                )
            else:
                cleaned_idx_core[ii] = True
        elif consideration == "entire":
            cleaned_idx_core[ii] = np.all(
                core_direc * (sl[:, core_axis] - core_bundle[:, core_axis]) > 0
            )
        elif consideration == "closest":
            dist_matrix = cdist(core_bundle, sl, "sqeuclidean")
            min_dist_indices = np.unravel_index(
                np.argmin(dist_matrix), dist_matrix.shape
            )
            closest_core = core_bundle[min_dist_indices[0], core_axis]
            closest_sl = sl[min_dist_indices[1], core_axis]

            cleaned_idx_core[ii] = core_direc * (closest_sl - closest_core) > 0
        else:
            raise ValueError(
                "Invalid value for consideration. Must be a "
                "float or one of 'entire' or 'closest'. You have provided: "
                f"{consideration}"
            )

    return cleaned_idx_core
