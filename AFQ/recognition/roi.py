import numpy as np
from dipy.core.interpolation import interpolate_scalar_3d


def _interp3d(roi, sl):
    return interpolate_scalar_3d(roi.get_fdata(), np.asarray(sl))[0]


def check_sls_with_inclusion(
        sls, include_rois, include_roi_tols):
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


def check_sl_with_exclusion(sl, exclude_rois,
                            exclude_roi_tols):
    """ Helper function to check that a streamline is not too close to a
    list of exclusion ROIs.
    """
    for ii, roi in enumerate(exclude_rois):
        # if any part of the streamline is near any exclusion ROI,
        # return False
        if np.any(_interp3d(roi, sl) <= exclude_roi_tols[ii]):
            return False
    # Either there are no exclusion ROIs, or you are not close to any:
    return True


def clean_by_endpoints(streamlines, target, target_idx, tol=0,
                       flip_sls=None, accepted_idxs=None):
    """
    Clean a collection of streamlines based on an endpoint ROI.
    Filters down to only include items that have their start or end points
    close to the targets.
    Parameters
    ----------
    streamlines : sequence of N by 3 arrays
        Where N is number of nodes in the array, the collection of
        streamlines to filter down to.
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
    accepted_idxs : 1d array, optional
        Boolean array, where entries correspond to eachs streamline,
        and streamlines that pass cleaning will be set to 1.
    Yields
    -------
    boolean array of streamlines that survive cleaning.
    """
    if accepted_idxs is None:
        accepted_idxs = np.zeros(len(streamlines), dtype=np.bool_)

    if flip_sls is None:
        flip_sls = np.zeros(len(streamlines))
    flip_sls = flip_sls.astype(int)

    for ii, sl in enumerate(streamlines):
        this_idx = target_idx
        if flip_sls[ii]:
            this_idx = (len(sl) - this_idx - 1) % len(sl)
        accepted_idxs[ii] = _interp3d(target, [sl[this_idx]])[0] <= tol

    return accepted_idxs
