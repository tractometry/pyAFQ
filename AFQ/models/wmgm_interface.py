import nibabel as nib
import numpy as np
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

    wm_boundary = find_boundaries(wm > 0.5, mode='inner')
    gm_smoothed = gaussian_filter(gm, 1)
    csf_smoothed = gaussian_filter(csf, 1)

    wm_boundary[~gm_smoothed.astype(bool)] = 0
    wm_boundary[csf_smoothed > gm_smoothed] = 0
    wm_boundary[wm < 0.5] = 0

    return nib.Nifti1Image(
        wm_boundary.astype(np.float32), dwiref_img.affine)
