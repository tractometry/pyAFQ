import os.path as op

import dipy.data.fetcher as fetcher
import nibabel as nib
import numpy as np
import numpy.testing as npt
from dipy.io.stateful_tractogram import Space, StatefulTractogram

import AFQ.data.fetch as afd
import AFQ.recognition.cleaning as abc
import AFQ.recognition.curvature as abv
import AFQ.recognition.other_bundles as abo
import AFQ.recognition.utils as abu

hardi_dir = op.join(fetcher.dipy_home, "stanford_hardi")
hardi_fdata = op.join(hardi_dir, "HARDI150.nii.gz")
hardi_img = nib.load(hardi_fdata)
file_dict = afd.read_stanford_hardi_tractography()
streamlines = file_dict["tractography_subsampled.trk"]
tg = StatefulTractogram(streamlines, hardi_img, Space.RASMM)
tg.to_vox()
streamlines = tg.streamlines


def _make_straight_streamlines(n_sl=5, length=10, axis=0, offset=0):
    """Utility: make straight streamlines along one axis."""
    sls = []
    for _ in range(n_sl):
        sl = np.zeros((length, 3))
        sl[:, axis] = np.linspace(0, length - 1, length) + offset
        sls.append(sl)
    return sls


def test_segment_sl_curve():
    sl_disp_0 = abv.sl_curve(streamlines[4], 4)
    npt.assert_array_almost_equal(
        sl_disp_0,
        [
            [-0.236384, -0.763855, 0.60054],
            [0.232594, -0.867859, -0.439],
            [0.175343, 0.001082, -0.984507],
        ],
    )

    sl_disp_1 = abv.sl_curve(streamlines[2], 4)
    mean_angle_diff = abv.sl_curve_dist(sl_disp_0, sl_disp_1)
    npt.assert_almost_equal(mean_angle_diff, 1.701458, decimal=3)


def test_cleaning():
    out, returned_idx = abc.clean_bundle(
        tg,
        n_points=100,
        clean_rounds=5,
        distance_threshold=2,
        length_threshold=4,
        min_sl=20,
        return_idx=True,
    )
    for idx, sl in enumerate(out.streamlines):
        idx_sl = tg.streamlines[returned_idx][idx]
        for node_idx, node in enumerate(sl):
            npt.assert_equal(node, idx_sl[node_idx])


def test_segment_clip_edges():
    sls = tg.streamlines
    idx = np.arange(len(tg.streamlines))
    accepted_sls = sls[[4, 10, 11]]
    accepted_ix = idx[[4, 10, 11]]
    bundle_roi_closest = np.zeros((len(sls), 3), dtype=np.int32)
    bundle_roi_closest[4, :] = [5, 10, 15]
    bundle_roi_closest[10, :] = [3, 6, 9]
    bundle_roi_closest[11, :] = [10, 10, 10]
    cut_sls = abu.cut_sls_by_closest(
        accepted_sls, bundle_roi_closest[accepted_ix], [0, 2]
    )
    npt.assert_array_equal(cut_sls[0], accepted_sls[0][5:15])
    npt.assert_array_equal(cut_sls[1], accepted_sls[1][3:9])
    npt.assert_array_equal(cut_sls[2], accepted_sls[2][9:11])


def test_segment_orientation():
    cleaned_idx = abc.clean_by_orientation(
        streamlines, primary_axis="P/A", affine=np.eye(4)
    )
    npt.assert_equal(np.sum(cleaned_idx), 93)
    cleaned_idx_tol = abc.clean_by_orientation(
        streamlines, primary_axis="P/A", affine=np.eye(4), tol=50
    )
    npt.assert_(np.sum(cleaned_idx_tol) < np.sum(cleaned_idx))

    cleaned_idx = abc.clean_by_orientation(
        streamlines, primary_axis="I/S", affine=np.eye(4)
    )
    cleaned_idx_tol = abc.clean_by_orientation(
        streamlines, primary_axis="I/S", affine=np.eye(4), tol=33
    )
    npt.assert_array_equal(cleaned_idx_tol, cleaned_idx)


def test_clean_isolation_forest_basic():
    cleaned_idx = abc.clean_by_isolation_forest(streamlines, n_points=20, min_sl=10)
    # Should return either a boolean mask or integer indices
    npt.assert_(isinstance(cleaned_idx, (np.ndarray,)))
    npt.assert_(cleaned_idx.shape[0] <= len(streamlines))


def test_clean_isolation_forest_outlier_thresh():
    cleaned_loose = abc.clean_by_isolation_forest(
        streamlines, n_points=20, distance_threshold=2, min_sl=10, random_state=42
    )
    cleaned_strict = abc.clean_by_isolation_forest(
        streamlines, n_points=20, distance_threshold=3, min_sl=10, random_state=42
    )
    npt.assert_(np.sum(cleaned_loose) >= np.sum(cleaned_strict))


def test_clean_by_overlap_keep_remove():
    img = nib.Nifti1Image(np.zeros((20, 20, 20)), np.eye(4))

    this_bundle = _make_straight_streamlines(n_sl=3, length=10, axis=0)
    other_bundle = _make_straight_streamlines(n_sl=3, length=10, axis=0)

    cleaned_remove = abo.clean_by_overlap(
        this_bundle, other_bundle, overlap=5, img=img, remove=True
    )
    npt.assert_equal(cleaned_remove, np.zeros(3, dtype=bool))

    cleaned_keep = abo.clean_by_overlap(
        this_bundle, other_bundle, overlap=5, img=img, remove=False
    )
    npt.assert_equal(cleaned_keep, np.ones(3, dtype=bool))


def test_clean_by_overlap_partial_overlap():
    img = nib.Nifti1Image(np.zeros((20, 20, 20)), np.eye(4))

    this_bundle = _make_straight_streamlines(n_sl=2, length=10, axis=0)
    other_bundle = _make_straight_streamlines(n_sl=2, length=10, axis=1)

    # These bundles are orthogonal, so minimal overlap
    cleaned = abo.clean_by_overlap(
        this_bundle, other_bundle, overlap=2, img=img, remove=False
    )
    npt.assert_equal(cleaned, np.zeros(2, dtype=bool))


def test_clean_relative_to_other_core_entire_vs_closest():
    # Two bundles along x axis, separated along z
    this_bundle = np.array(_make_straight_streamlines(n_sl=2, length=5, axis=0))
    this_bundle[0, :, 2] += 5
    this_bundle[1, :, 2] -= 5
    other_bundle = np.array(_make_straight_streamlines(n_sl=2, length=5, axis=0))
    affine = np.eye(4)
    cleaned_entire = abo.clean_relative_to_other_core(
        "inferior", this_bundle, other_bundle, affine, entire=True
    )
    npt.assert_equal(cleaned_entire, [True, False])

    # With entire=False, same result in this synthetic case
    cleaned_closest = abo.clean_relative_to_other_core(
        "inferior", this_bundle, other_bundle, affine, entire=False
    )
    npt.assert_equal(cleaned_closest, [True, False])
