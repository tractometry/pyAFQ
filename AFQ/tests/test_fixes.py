import os.path as op

import dipy.core.gradients as dpg
import nibabel as nib
import nibabel.tmpdirs as nbtmp
import numpy as np
import numpy.testing as npt
from dipy.data import default_sphere
from dipy.reconst.gqi import GeneralizedQSamplingModel

import AFQ.data.fetch as afd
from AFQ._fixes import gaussian_weights, gwi_odf
from AFQ.utils.testing import make_dki_data


def test_GQI_fix():
    with nbtmp.InTemporaryDirectory() as tmpdir:
        fbval = op.join(tmpdir, "dki.bval")
        fbvec = op.join(tmpdir, "dki.bvec")
        fdata = op.join(tmpdir, "dki.nii.gz")
        make_dki_data(fbval, fbvec, fdata)
        gtab = dpg.gradient_table(bvals=fbval, bvecs=fbvec)
        data = nib.load(fdata).get_fdata()

        gqmodel = GeneralizedQSamplingModel(gtab, sampling_length=1.2)

        odf_ours = gwi_odf(gqmodel, data, default_sphere)

        odf_theirs = gqmodel.fit(data).odf(default_sphere)

        npt.assert_array_almost_equal(odf_ours, odf_theirs)


def test_gaussian_weights():
    file_dict = afd.read_stanford_hardi_tractography()
    streamlines = file_dict["tractography_subsampled"]

    weights = gaussian_weights(streamlines)
    assert not np.any(np.isnan(weights))

    # test consistency
    assignment_idxs = np.tile(np.arange(100), (len(streamlines), 1))
    assignment_method_weights = gaussian_weights(
        streamlines, assignment_idxs=assignment_idxs
    )

    assert np.allclose(
        weights, assignment_method_weights[: len(weights)], rtol=1e-6, atol=1e-6
    )

    assert np.allclose(np.sum(weights), 100)
    assert np.allclose(np.sum(assignment_method_weights), 100)


def test_mahal_fix():
    sls = [
        [[30, 41, 61], [28, 61, 38]],
        [[30, 41, 62], [20, 44, 34]],
        [[50, 67, 88], [10, 10, 20]],
        [[35, 43, 65], [25, 55, 35]],
        [[40, 50, 70], [15, 15, 25]],
        [[45, 54, 75], [12, 22, 32]],
        [[32, 48, 68], [28, 58, 40]],
        [[38, 52, 72], [18, 38, 28]],
        [[34, 44, 64], [21, 41, 31]],
        [[36, 46, 66], [23, 53, 33]],
        [[37, 47, 67], [24, 54, 34]],
        [[39, 49, 69], [19, 39, 29]],
        [[33, 53, 73], [22, 42, 32]],
        [[31, 51, 71], [26, 56, 36]],
        [[29, 59, 79], [27, 57, 37]],
        [[28, 58, 78], [17, 47, 27]],
        [[27, 57, 77], [16, 36, 26]],
        [[26, 56, 76], [14, 24, 34]],
        [[25, 55, 75], [13, 23, 33]],
        [[24, 54, 74], [11, 21, 31]],
    ]
    sls_array = np.asarray(sls).astype(float)
    results = np.asarray(
        [
            [1.718654, 1.550252],
            [2.202227, 0.7881],
            [3.415999, 2.689814],
        ]
    )
    npt.assert_array_almost_equal(
        gaussian_weights(
            sls_array, n_points=None, return_mahalanobis=True, stat=np.mean
        )[:3],
        results,
    )
