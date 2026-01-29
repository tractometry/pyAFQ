import numpy as np
from dipy.core.interpolation import interpolate_scalar_3d


def _interp3d(roi, sl):
    return interpolate_scalar_3d(roi.get_fdata(), np.asarray(sl))[0]


def check_sls_with_inclusion(sls, include_rois, include_roi_tols):
    inc_results = np.zeros(len(sls), dtype=tuple)
    include_rois = [roi_.get_fdata().copy() for roi_ in include_rois]
    for jj, sl in enumerate(sls):
        closest = np.zeros(len(include_rois), dtype=np.int32)
        sl = np.asarray(sl)
        valid = True
        for ii, roi in enumerate(include_rois):
            dist = interpolate_scalar_3d(roi, sl)[0]

            closest[ii] = np.argmin(dist)
            if dist[closest[ii]] > include_roi_tols[ii]:
                # Too far from one of them:
                inc_results[jj] = (False, [])
                valid = False
                break

        # Checked all the ROIs and it was close to all of them
        if valid:
            inc_results[jj] = (True, closest)
    return inc_results


def check_sl_with_exclusion(sl, exclude_rois, exclude_roi_tols):
    """Helper function to check that a streamline is not too close to a
    list of exclusion ROIs.
    """
    for ii, roi in enumerate(exclude_rois):
        # if any part of the streamline is near any exclusion ROI,
        # return False
        if np.any(_interp3d(roi, sl) <= exclude_roi_tols[ii]):
            return False
    # Either there are no exclusion ROIs, or you are not close to any:
    return True


def clean_by_endpoints(fgarray, target, target_idx, tol=0, flip_sls=None):
    """
    Clean a collection of streamlines based on an endpoint ROI.
    Filters down to only include items that have their start or end points
    close to the targets.
    Parameters
    ----------
    fgarray : ndarray of shape (N, M, 3)
        Where N is number of streamlines, M is number of nodes.
    target: Nifti1Image
        Nifti1Image containing a distance transform of the ROI.
    target_idx: int.
        Index within each streamline to check if within the target region.
        Typically 0 for startpoint ROIs or -1 for endpoint ROIs.
        If using flip_sls, this becomes (len(sl) - this_idx - 1) % len(sl)
    tol : int, optional
        A distance tolerance (in units that the coordinates
        of the streamlines are represented in). Default: 0, which means that
        the endpoint is exactly in the coordinate of the target ROI.
    flip_sls : 1d array, optional
        Length is len(streamlines), whether to flip the streamline.
    Yields
    -------
    boolean array of streamlines that survive cleaning.
    """
    n_sls, n_nodes, _ = fgarray.shape

    # handle target_idx negative values as wrapping around
    effective_idx = target_idx if target_idx >= 0 else (n_nodes + target_idx)
    indices = np.full(n_sls, effective_idx)

    if flip_sls is not None:
        flipped_indices = n_nodes - 1 - effective_idx
        indices = np.where(flip_sls.astype(bool), flipped_indices, indices)

    distances = interpolate_scalar_3d(
        target.get_fdata(), fgarray[np.arange(n_sls), indices]
    )[0]

    return distances <= tol
