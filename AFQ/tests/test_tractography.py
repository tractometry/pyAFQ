import os.path as op
import random
import numpy as np
import numpy.testing as npt
import pytest

import nibabel as nib
import tempfile

from AFQ.models.csd import fit_csd
from AFQ.models.dti import fit_dti
from AFQ.tractography.tractography import track
from AFQ.utils.testing import make_tracking_data


seeds = np.array([[-80., -120., -60.],
                  [-81, -121, -61],
                  [-81, -120, -60]])


tmpdir = tempfile.TemporaryDirectory()
fbval = op.join(tmpdir.name, 'dti.bval')
fbvec = op.join(tmpdir.name, 'dti.bvec')
fdata = op.join(tmpdir.name, 'dti.nii.gz')
make_tracking_data(fbval, fbvec, fdata)

minlen = 20
step_size = 0.5


def test_csd_local_tracking():
    random.seed(1234)
    for sh_order_max in [4, 8, 10]:
        fname = fit_csd(fdata, fbval, fbvec,
                        response=((0.0015, 0.0003, 0.0003), 100),
                        sh_order_max=sh_order_max, lambda_=1,
                        tau=0.1, mask=None,
                        out_dir=tmpdir.name)
        for directions in ["det", "prob"]:
            sls = track(
                fname,
                directions,
                odf_model="CSD",
                max_angle=30.,
                sphere=None,
                seed_mask=None,
                n_seeds=seeds,
                stop_mask=None,
                step_size=step_size,
                minlen=minlen,
                tracker="local").streamlines

            for sl in sls:
                npt.assert_(len(sl) >= minlen / step_size)


def test_dti_local_tracking():
    fdict = fit_dti(fdata, fbval, fbvec)
    for directions in ["det", "prob"]:
        sls = track(
            fdict['params'],
            directions,
            max_angle=30.,
            sphere=None,
            seed_mask=None,
            n_seeds=1,
            step_size=step_size,
            minlen=minlen,
            odf_model="DTI",
            tracker="local").streamlines
        for sl in sls:
            npt.assert_(len(sl) >= minlen / step_size)


def test_pft_tracking():
    for fname, odf in zip(
            [
                fit_dti(fdata, fbval, fbvec)['params'],
                fit_csd(
                    fdata, fbval, fbvec,
                    response=((0.0015, 0.0003, 0.0003), 100),
                    sh_order_max=8, lambda_=1, tau=0.1, mask=None,
                    out_dir=tmpdir.name)],
            ["DTI", "CSD"]):
        img = nib.load(fdata)
        data_shape  = img.shape
        pve_wm_data = np.ones(data_shape[:3])
        pve_gm_data = np.ones(data_shape[:3])
        pve_csf_data = np.ones(data_shape[:3])
        stop_mask = (pve_wm_data, pve_gm_data, pve_csf_data)

        for directions in ["det", "prob"]:
            for stop_threshold in ["ACT", "CMC"]:
                sls = track(
                    fname,
                    directions,
                    max_angle=30.,
                    sphere=None,
                    seed_mask=None,
                    stop_mask=stop_mask,
                    stop_threshold=stop_threshold,
                    n_seeds=1,
                    step_size=step_size,
                    minlen=minlen,
                    odf_model=odf,
                    tracker="pft").streamlines

                for sl in sls:
                    npt.assert_(len(sl) >= minlen / step_size)

    # Test error handling:
    with pytest.raises(RuntimeError):
        track(
            fname,
            directions,
            max_angle=30.,
            sphere=None,
            seed_mask=None,
            stop_mask=0,  # Stop mask needs to be a tuple!
            stop_threshold=stop_threshold,
            n_seeds=1,
            step_size=step_size,
            minlen=minlen,
            tracker="pft")

    with pytest.raises(RuntimeError):
        track(
            fname,
            directions,
            max_angle=30.,
            sphere=None,
            seed_mask=None,
            stop_mask=stop_mask,
            stop_threshold=None,  # Stop threshold needs to be a string!
            n_seeds=1,
            step_size=step_size,
            minlen=minlen,
            tracker="pft")