import os.path as op

import dipy.data as dpd
import nibabel as nib
import nibabel.tmpdirs as nbtmp
import numpy as np
import numpy.testing as npt
import pytest
from dipy.reconst.shm import calculate_max_order

from AFQ.models import csd
from AFQ.models.asym_filtering import unified_filtering as pyafq_unified_filtering

try:
    from scilpy.denoise.asym_filtering import (
        unified_filtering as scilpy_unified_filtering,
    )

    has_scilpy = True
except ImportError:
    has_scilpy = False


def test_fit_csd():
    fdata, fbval, fbvec = dpd.get_fnames("small_64D")
    with nbtmp.InTemporaryDirectory() as tmpdir:
        # Convert from npy to txt:
        bvals = np.loadtxt(fbval)
        bvecs = np.loadtxt(fbvec)
        np.savetxt(op.join(tmpdir, "bvals.txt"), bvals)
        np.savetxt(op.join(tmpdir, "bvecs.txt"), bvecs)
        for sh_order_max in [4, 6]:
            fname = csd.fit_csd(
                str(fdata),
                op.join(tmpdir, "bvals.txt"),
                op.join(tmpdir, "bvecs.txt"),
                out_dir=tmpdir,
                sh_order_max=sh_order_max,
            )

            npt.assert_(op.exists(fname))
            sh_coeffs_img = nib.load(fname)
            npt.assert_equal(sh_order_max, calculate_max_order(sh_coeffs_img.shape[-1]))


# Note we do not want to run this by default, Scilpy
# Has many specific requirements for dependency versions
# That we do not want to interfere with pyAFQ testing generally
@pytest.mark.skipif(not has_scilpy, reason="scilpy is not installed")
def test_afod():
    fdata, fbval, fbvec = dpd.get_fnames("small_64D")
    sphere = dpd.get_sphere("repulsion100")
    with nbtmp.InTemporaryDirectory() as tmpdir:
        # Convert from npy to txt:
        bvals = np.loadtxt(fbval)
        bvecs = np.loadtxt(fbvec)
        np.savetxt(op.join(tmpdir, "bvals.txt"), bvals)
        np.savetxt(op.join(tmpdir, "bvecs.txt"), bvecs)
        fname = csd.fit_csd(
            str(fdata),
            op.join(tmpdir, "bvals.txt"),
            op.join(tmpdir, "bvecs.txt"),
            out_dir=tmpdir,
            sh_order_max=6,
        )

        npt.assert_(op.exists(fname))
        sh_coeffs_img = nib.load(fname)

        aodf_pyafq = pyafq_unified_filtering(
            sh_coeffs_img.get_fdata(),
            sphere,
        )

        aodf_scilpy = scilpy_unified_filtering(
            sh_coeffs_img.get_fdata(),
            6,
            "descoteaux07",
            True,
            False,
            "repulsion100",
        )

        npt.assert_allclose(aodf_pyafq, aodf_scilpy, atol=1e-6)
