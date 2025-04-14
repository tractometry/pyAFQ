import os
import os.path as op

import numpy as np
import nibabel as nib

from dipy.reconst import csdeconv as csd
from dipy.reconst import shm
from dipy.core.gradients import gradient_table, unique_bvals_magnitude
from dipy.data import get_sphere, small_sphere

import AFQ.utils.models as ut
from AFQ.models.asym_csd import AsymConstrainedSphericalDeconvModel


# Monkey patch fixed spherical harmonics for conda from
# DIPY dev:
from AFQ._fixes import spherical_harmonics
shm.spherical_harmonics = spherical_harmonics

__all__ = ["fit_csd"]


class CsdNanResponseError(Exception):
    pass


def _fit(gtab, data, mask, response=None, sh_order=None,
         asym=False, csd_fa_thr=0.7):
    """
    Helper function that does the core of fitting a model to data.
    """
    if sh_order is None:
        ndata = np.sum(~gtab.b0s_mask)
        # See dipy.reconst.shm.calculate_max_order
        L1 = (-3 + np.sqrt(1 + 8 * ndata)) / 2.0
        sh_order = int(L1)
        if np.mod(sh_order, 2) != 0:
            sh_order = sh_order - 1
        if sh_order > 8:
            sh_order = 8

    unique_bvals = unique_bvals_magnitude(gtab.bvals)
    if len(unique_bvals[unique_bvals > 0]) > 1:
        low_shell_idx = gtab.bvals <= unique_bvals[unique_bvals > 0][0]
        response_gtab = gradient_table(gtab.bvals[low_shell_idx],
                                       gtab.bvecs[low_shell_idx])
        data = data[..., low_shell_idx]
    else:
        response_gtab = gtab

    if response is None:
        response, _ = csd.auto_response_ssst(
            response_gtab,
            data,
            roi_radii=10,
            fa_thr=csd_fa_thr)
    # Catch conditions where an auto-response could not be calculated:
    if np.all(np.isnan(response[0])):
        raise CsdNanResponseError

    if asym:
        acsdmodel = AsymConstrainedSphericalDeconvModel(
            gtab, response,
            reg_sphere=small_sphere,
            sh_order=sh_order)
        return acsdmodel.fit(
            data, mask=mask)
    else:
        csdmodel = csd.ConstrainedSphericalDeconvModel(
            gtab, response, sh_order=sh_order)
        return csdmodel.fit(
            data, mask=mask)


def fit_csd(data_files, bval_files, bvec_files, mask=None, response=None,
            b0_threshold=50, sh_order=None, asym=False, out_dir=None):
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
    sh_order : int, optional.
        default: infer the number of parameters from the number of data
        volumes, but no larger than 8.
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
    img, data, gtab, mask = ut.prepare_data(data_files, bval_files, bvec_files,
                                            b0_threshold=b0_threshold,
                                            mask=mask)

    csdfit = _fit(gtab, data, mask, response=response, sh_order=sh_order,
                  asym=asym)

    if out_dir is None:
        out_dir = op.join(op.split(data_files)[0], 'dki')

    if not op.exists(out_dir):
        os.makedirs(out_dir)

    aff = img.affine
    fname = op.join(out_dir, 'csd_sh_coeff.nii.gz')
    nib.save(nib.Nifti1Image(csdfit.shm_coeff, aff), fname)
    return fname
