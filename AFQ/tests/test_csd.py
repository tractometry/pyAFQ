import os.path as op

import dipy.data as dpd
import nibabel as nib
import nibabel.tmpdirs as nbtmp
import numpy as np
import numpy.testing as npt
import pytest
from dipy.reconst import shm
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


def test_tournier07_to_descoteaux07():
    """
    The projection matches, on the sphere, cs_dmri's own export of an SS3T
    fit to DIPY's descoteaux07 basis
    """
    cs = pytest.importorskip("cs_dmri")
    from dipy.io.peaks import load_pam

    from AFQ.tasks.data import _tournier07_to_descoteaux07

    shape, lmax = (5, 5, 4), 8
    rng = np.random.default_rng(0)
    wm = rng.normal(size=shape + (45,)).astype(np.float32)
    responses = cs.ResponseSet(
        wm=cs.TissueResponse(
            np.array([[3.5, 0, 0, 0, 0], [1.2, -0.6, 0.2, -0.05, 0.01]]), lmax
        ),
        gm=cs.TissueResponse(np.array([[3.0], [1.0]]), 0),
        csf=cs.TissueResponse(np.array([[4.0], [0.3]]), 0),
    )
    # on a RAS+ grid cs_dmri's world frame is the image frame
    fit = cs.SS3TFit(
        wm=wm,
        gm=np.full(shape, 0.05, np.float32),
        csf=np.full(shape, 0.02, np.float32),
        lmax_wm=lmax,
        mask=np.ones(shape, bool),
        affine=np.diag([2.0, 2.0, 2.0, 1.0]),
        frame="world",
        responses=responses,
    )
    with nbtmp.InTemporaryDirectory() as tmpdir:
        fit.export(op.join(tmpdir, "fit.pam5"))
        ref = load_pam(op.join(tmpdir, "fit.pam5")).shm_coeff
    B, _ = shm.sh_to_sf_matrix(dpd.get_sphere(name="symmetric362"), sh_order_max=lmax)
    npt.assert_allclose(_tournier07_to_descoteaux07(wm) @ B, ref @ B, atol=1e-4)
