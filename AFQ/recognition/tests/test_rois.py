import numpy as np
import AFQ.recognition.roi as abr
import nibabel as nib
import numpy.testing as npt
import numpy as np
from scipy.ndimage import distance_transform_edt
from AFQ.recognition.roi import (
    check_sls_with_inclusion,
    check_sl_with_inclusion,
    check_sl_with_exclusion)

shape = (15, 15, 15)
affine = np.eye(4)

streamline1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
streamline2 = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
streamlines = [
    np.array([[1, 1, 1],
            [2, 1, 1],
            [3, 1, 1],
            [4, 1, 1]]),
    np.array([[1, 1, 2],
            [2, 1, 2],
            [3, 1, 2],
            [4, 1, 2]]),
    np.array([[1, 1, 1],
            [2, 1, 1],
            [3, 1, 1]]),
    np.array([[1, 1, 1],
            [2, 1, 1]])]

roi1 = np.ones(shape, dtype=np.float32)
roi1[1, 2, 3] = 0
roi1[4, 5, 6] = 0
roi1 = nib.Nifti1Image(distance_transform_edt(roi1), affine)
roi2 = np.ones(shape, dtype=np.float32)
roi2[7, 8, 9] = 0
roi2[10, 11, 12] = 0
roi2 = nib.Nifti1Image(distance_transform_edt(roi2), affine)

start_roi = np.ones(shape, dtype=np.float32)
start_roi[1, 1, 1] = 0
start_roi[1, 1, 2] = 0
start_roi = nib.Nifti1Image(distance_transform_edt(start_roi), affine)
end_roi = np.ones(shape, dtype=np.float32)
end_roi[4, 1, 1] = 0
end_roi[4, 1, 2] = 0
end_roi = nib.Nifti1Image(distance_transform_edt(end_roi), affine)

include_rois = [roi1, roi2]
exclude_rois = [roi1]
include_roi_tols = [8, 8]
exclude_roi_tols = [1]


def test_clean_by_endpoints():
    clean_idx_start = list(abr.clean_by_endpoints(
        streamlines, start_roi, 0))
    clean_idx_end = list(abr.clean_by_endpoints(
        streamlines, end_roi, -1))
    npt.assert_array_equal(np.logical_and(
        clean_idx_start, clean_idx_end), np.array([1, 1, 0, 0]))

    # If tol=1, the third streamline also gets included
    clean_idx_start = list(abr.clean_by_endpoints(
        streamlines, start_roi, 0, tol=1))
    clean_idx_end = list(abr.clean_by_endpoints(
        streamlines, end_roi, -1, tol=1))
    npt.assert_array_equal(np.logical_and(
        clean_idx_start, clean_idx_end), np.array([1, 1, 1, 0]))


def test_check_sls_with_inclusion():
    sls = [streamline1, streamline2]
    result = list(check_sls_with_inclusion(
        sls, include_rois, include_roi_tols))
    assert result[0][0] is True
    assert np.allclose(
        result[0][1][0], 0)
    assert np.allclose(
        result[0][1][1], 2)
    assert result[1][0] is False


def test_check_sl_with_inclusion_pass():
    result, dists = check_sl_with_inclusion(
        streamline1, include_rois, include_roi_tols)
    assert result is True
    assert len(dists) == 2


def test_check_sl_with_inclusion_fail():
    result, dists = check_sl_with_inclusion(
        streamline2, include_rois, include_roi_tols)
    assert result is False
    assert dists == []


def test_check_sl_with_exclusion_pass():
    result = check_sl_with_exclusion(
        streamline1, exclude_rois, exclude_roi_tols)
    assert result is False


def test_check_sl_with_exclusion_fail():
    result = check_sl_with_exclusion(
        streamline2, exclude_rois, exclude_roi_tols)
    assert result is True
