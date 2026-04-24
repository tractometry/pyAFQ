import logging

import dipy.tracking.utils as dtu
import nibabel as nib
import numpy as np
from dipy.io.utils import create_nifti_header, get_reference_info
from dipy.tracking.streamline import select_random_set_of_streamlines
from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import dice
from skimage.morphology import dilation

logger = logging.getLogger("AFQ")


def transform_roi(roi, mapping, is_boolean):
    """
    After being non-linearly transformed, ROIs tend to have holes in them.
    We perform a couple of computational geometry operations on the ROI to
    fix that up.

    Parameters
    ----------
    roi : Nifti1Image, str, ndarray
        The ROI to transform. Can be a path or image, which will be
        converted to an ndarray.

    mapping : DiffeomorphicMap object
        A mapping between DWI space and a template.

    is_boolean : bool
        Whether the ROI is boolean or not.

    Returns
    -------
    ROI after dilation and hole-filling
    """
    if isinstance(roi, str):
        roi = nib.load(roi)
    if isinstance(roi, nib.Nifti1Image):
        roi = roi.get_fdata()

    if is_boolean:
        # dilate binary images to avoid losing small ROIs
        roi = dilation(roi)
        roi = distance_transform_edt(roi.astype(bool))

    roi = mapping.transform((roi.astype(float)), interpolation="linear")

    if is_boolean:
        return (roi > 0.5).astype(np.uint8)
    else:
        return roi.astype(np.float32)


def density_map(tractogram, n_sls=None, normalize=False):
    """
    Create a streamline density map.
    based on:
    https://dipy.org/documentation/1.1.1./examples_built/streamline_formats/

    Parameters
    ----------
    tractogram : StatefulTractogram
        Stateful tractogram whose streamlines are used to make
        the density map.
    n_sls : int or None, optional
        n_sls to randomly select to make the density map.
        If None, all streamlines are used.
        Default: None
    normalize : bool, optional
        Whether to normalize maximum values to 1.
        Default: False

    Returns
    -------
    Nifti1Image containing the density map.
    """
    tractogram.to_vox()

    sls = tractogram.streamlines
    if n_sls is not None:
        sls = select_random_set_of_streamlines(sls, n_sls)

    affine, vol_dims, voxel_sizes, voxel_order = get_reference_info(tractogram)
    tractogram_density = dtu.density_map(sls, np.eye(4), vol_dims)
    if normalize:
        tractogram_density = tractogram_density / tractogram_density.max()

    nifti_header = create_nifti_header(affine, vol_dims, voxel_sizes)
    density_map_img = nib.Nifti1Image(tractogram_density, affine, nifti_header)

    return density_map_img


def dice_coeff(arr1, arr2, weighted=True):
    """
    Compute Dice's coefficient between two images.

    Parameters
    ----------
    arr1 : Nifti1Image, str, ndarray
        One ndarray to compare. Can be a path or image, which will be
        converted to an ndarray.
    arr2 : Nifti1Image, str, ndarray
        The other ndarray to compare. Can be a path or image, which will be
        converted to an ndarray.
    weighted : bool, optional
        Whether or not to weight the DICE coefficient as in [Cousineau2017]_.
        The weighted Dice coefficient is calculated by adding the sum of all
        values in arr1 where arr2 is nonzero to the sum of all values in arr2
        where arr1 is nonzero, then dividing that by the sum of all values in
        arr1 and arr2.
        Default: True

    Returns
    -------
    The dice similarity between the images.

    Notes
    -----
    .. [Cousineau2017] Cousineau M, Jodoin PM, Morency FC, et al.
           A test-retest study on
           Parkinson's PPMI dataset yields statistically significant white
           matter fascicles. Neuroimage Clin. 2017;16:222-233. Published 2017
           Jul 25. doi:10.1016/j.nicl.2017.07.020
    """
    if isinstance(arr1, str):
        arr1 = nib.load(arr1)
    if isinstance(arr2, str):
        arr2 = nib.load(arr2)

    if isinstance(arr1, nib.Nifti1Image):
        arr1 = arr1.get_fdata()
    if isinstance(arr2, nib.Nifti1Image):
        arr2 = arr2.get_fdata()

    arr1 = arr1.flatten()
    arr2 = arr2.flatten()

    if weighted:
        return (np.sum(arr1 * arr2.astype(bool)) + np.sum(arr2 * arr1.astype(bool))) / (
            np.sum(arr1) + np.sum(arr2)
        )
    else:
        # scipy's dice function returns the dice *dissimilarity*
        return 1 - dice(arr1.astype(bool), arr2.astype(bool))
