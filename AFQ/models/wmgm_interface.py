import numpy as np
import nibabel as nib

from dipy.align import resample

from scipy.ndimage import gaussian_filter
from skimage.segmentation import find_boundaries


def fit_wm_gm_interface(PVE_img, dwiref_img):
    """
    Compute the white matter/gray matter interface from a PVE image.

    Parameters
    ----------
    PVE_img : Nifti1Image
        PVE image containing CSF, GM, and WM segmentations from T1
    dwiref_img : Nifti1Image
        Reference image to find boundary in that space.
    """
    PVE = PVE_img.get_fdata()

    csf = PVE[..., 0]
    gm = PVE[..., 1]
    wm = PVE[..., 2]

    # Put in diffusion space
    wm = resample(
        wm,
        dwiref_img.get_fdata(),
        moving_affine=PVE_img.affine,
        static_affine=dwiref_img.affine).get_fdata()
    gm = resample(
        gm,
        dwiref_img.get_fdata(),
        moving_affine=PVE_img.affine,
        static_affine=dwiref_img.affine).get_fdata()
    csf = resample(
        csf,
        dwiref_img.get_fdata(),
        moving_affine=PVE_img.affine,
        static_affine=dwiref_img.affine).get_fdata()

    wm_boundary = find_boundaries(wm, mode='inner')
    gm_smoothed = gaussian_filter(gm, 1)
    csf_smoothed = gaussian_filter(csf, 1)

    wm_boundary[~gm_smoothed.astype(bool)] = 0
    wm_boundary[csf_smoothed > gm_smoothed] = 0

    return nib.Nifti1Image(
        wm_boundary.astype(np.float32), dwiref_img.affine)


def pve_from_subcortex(t1_subcortex_data):
    """
    Compute the PVE (Partial Volume Estimation) from the subcortex T1 image.

    Parameters
    ----------
    t1_subcortex_data : ndarray
        T1 subcortex data from brainchop

    Returns
    -------
    pve_img : ndarray
        PVE data with CSF, GM, and WM segmentations.
    """
    CSF_labels = [0, 3, 4, 11, 12]
    GM_labels = [2, 6, 7, 8, 9, 10, 14, 15, 16]
    WM_labels = [1, 5]
    mixed_labels = [13, 17]

    PVE = np.zeros(t1_subcortex_data.shape + (3,), dtype=np.float32)

    PVE[np.isin(t1_subcortex_data, CSF_labels), 0] = 1.0
    PVE[np.isin(t1_subcortex_data, GM_labels), 1] = 1.0
    PVE[np.isin(t1_subcortex_data, WM_labels), 2] = 1.0

    # For mixed labels, we assume they are WM interior, GM exterior
    # This is a simplification, basically so they do not cause problems
    # with ACT
    wm_fuzzed = gaussian_filter(PVE[..., 2], 1)
    nwm_fuzzed = gaussian_filter(PVE[..., 0] + PVE[..., 1], 1)
    bs_exterior = np.logical_and(
        find_boundaries(
            np.isin(t1_subcortex_data, mixed_labels),
            mode='inner'),
        nwm_fuzzed >= wm_fuzzed)
    bs_interior = np.logical_and(
        np.isin(t1_subcortex_data, mixed_labels),
        ~bs_exterior)
    PVE[bs_exterior, 1] = 1.0
    PVE[bs_interior, 2] = 1.0

    return PVE
