import os
import os.path as op

import nibabel as nib
import numpy as np
from dipy.core.gradients import gradient_table, unique_bvals_magnitude
from dipy.reconst import csdeconv as csd
from dipy.reconst import shm

import AFQ.utils.models as ut

# Monkey patch fixed spherical harmonics for conda from
# DIPY dev:
from AFQ._fixes import spherical_harmonics

shm.spherical_harmonics = spherical_harmonics

__all__ = ["fit_csd"]


class CsdNanResponseError(Exception):
    pass


def _model(gtab, data, response=None, sh_order_max=None, csd_fa_thr=0.7):
    """
    Helper function that defines a CSD model.
    """
    if sh_order_max is None:
        ndata = np.sum(~gtab.b0s_mask)
        # See dipy.reconst.shm.calculate_max_order
        L1 = (-3 + np.sqrt(1 + 8 * ndata)) / 2.0
        sh_order_max = int(L1)
        if np.mod(sh_order_max, 2) != 0:
            sh_order_max = sh_order_max - 1
        if sh_order_max > 8:
            sh_order_max = 8

    my_model = csd.ConstrainedSphericalDeconvModel
    if response is None:
        unique_bvals = unique_bvals_magnitude(gtab.bvals)
        if len(unique_bvals[unique_bvals > 0]) > 1:
            low_shell_idx = gtab.bvals <= unique_bvals[unique_bvals > 0][0]
            response_gtab = gradient_table(
                bvals=gtab.bvals[low_shell_idx], bvecs=gtab.bvecs[low_shell_idx]
            )
            data = data[..., low_shell_idx]
        else:
            response_gtab = gtab
        response, _ = csd.auto_response_ssst(
            response_gtab, data, roi_radii=10, fa_thr=csd_fa_thr
        )
    # Catch conditions where an auto-response could not be calculated:
    if np.all(np.isnan(response[0])):
        raise CsdNanResponseError

    csdmodel = my_model(gtab, response, sh_order_max=sh_order_max)
    return csdmodel, sh_order_max


def _fit(
    gtab,
    data,
    mask,
    response=None,
    sh_order_max=None,
    lambda_=1,
    tau=0.1,
    csd_fa_thr=0.7,
):
    """
    Helper function that does the core of fitting a model to data.
    """
    model, sh_order_max = _model(gtab, data, response, sh_order_max, csd_fa_thr)
    return model, model.fit(data, mask=mask), sh_order_max


def fit_csd(
    data_files,
    bval_files,
    bvec_files,
    mask=None,
    response=None,
    b0_threshold=50,
    sh_order_max=None,
    lambda_=1,
    tau=0.1,
    out_dir=None,
):
    """
    Fit the CSD model and save file with SH coefficients.

    Parameters
    ----------
    data_files : str or list.
        Files containing DWI data. If this is a str, that's the full path to a
        single file. If it's a list, each entry is a full path.
    bval_files : str or list.
        Equivalent to `data_files`.
    bvec_files : str or list.
        Equivalent to `data_files`.
    mask : ndarray, optional.
        Binary mask, set to True or 1 in voxels to be processed.
        Default: Process all voxels.
    response: tuple, optional.
        The response function to be used by CSD, as a tuple with two elements.
        The first is the eigen-values as an (3,) ndarray and the second is
        the signal value for the response function without diffusion-weighting
        (i.e. S0). If not provided, auto_response will be used to calculate
        these values.
    b0_threshold : float,optional.
      The value of diffusion-weighting under which we consider it to be
      equivalent to 0. Default:50
    sh_order_max : int, optional.
        default: infer the number of parameters from the number of data
        volumes, but no larger than 8.
    lambda_ : float, optional.
        weight given to the constrained-positivity regularization part of
        the deconvolution equation. Default: 1
    tau : float, optional.
        threshold controlling the amplitude below which the corresponding
        fODF is assumed to be zero.  Ideally, tau should be set to
        zero. However, to improve the stability of the algorithm, tau is
        set to tau*100 % of the mean fODF amplitude (here, 10% by default)
        (see [1]_). Default: 0.1
    out_dir : str, optional
        A full path to a directory to store the maps that get computed.
        Default: file with coefficients gets stored in the same directory as
        the first DWI file in `data_files`.

    Returns
    -------
    fname : the full path to the file containing the SH coefficients.

    References
    ----------
    .. [1] Tournier, J.D., et al. NeuroImage 2007. Robust determination of
            the fibre orientation distribution in diffusion MRI:
            Non-negativity constrained super-resolved spherical
            deconvolution
    """
    img, data, gtab, mask = ut.prepare_data(
        data_files, bval_files, bvec_files, b0_threshold=b0_threshold, mask=mask
    )

    _, csdfit, _ = _fit(
        gtab,
        data,
        mask,
        response=response,
        sh_order_max=sh_order_max,
        lambda_=lambda_,
        tau=tau,
    )

    if out_dir is None:
        out_dir = op.join(op.split(data_files)[0], "dki")

    if not op.exists(out_dir):
        os.makedirs(out_dir)

    aff = img.affine
    fname = op.join(out_dir, "csd_sh_coeff.nii.gz")
    nib.save(nib.Nifti1Image(csdfit.shm_coeff, aff), fname)
    return fname
