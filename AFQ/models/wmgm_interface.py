import numpy as np
import nibabel as nib

from dipy.align import resample

from skimage.morphology import remove_small_objects
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

    csf = PVE[..., 0].copy()
    gm = PVE[..., 1].copy()
    wm = PVE[..., 2].copy()

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

    # 13824 = 2^9*3^3
    # Needs to be enough to
    # remove small objects in the skull
    wm = remove_small_objects(wm > 0.5, min_size=13824)

    wm_boundary = find_boundaries(wm, mode='inner')
    gm_smoothed = gaussian_filter(gm, 1)
    csf_smoothed = gaussian_filter(csf, 1)

    wm_boundary[~gm_smoothed.astype(bool)] = 0
    wm_boundary[csf_smoothed > gm_smoothed] = 0

    return nib.Nifti1Image(
        wm_boundary.astype(np.float32), dwiref_img.affine)
