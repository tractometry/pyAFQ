import logging

import dipy.tracking.utils as dtu
import nibabel as nib
import numpy as np
import scipy.ndimage as ndim
from dipy.io.utils import create_nifti_header, get_reference_info
from dipy.tracking.streamline import select_random_set_of_streamlines
from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import dice
from skimage.measure import euler_number

logger = logging.getLogger("AFQ")


def _signed_distance(mask, max_tol):
    """
    Compute the signed distance transform of a binary mask, with a maximum
    tolerance for the distance values. This allows us to only have to
    calculated edt in a bounding box, which is faster.
    """
    margin = int(np.ceil(max_tol)) + 2

    coords = np.argwhere(mask)
    min_coords = np.maximum(0, coords.min(axis=0) - margin)
    max_coords = np.minimum(mask.shape, coords.max(axis=0) + margin + 1)
    box = tuple(slice(min_c, max_c) for min_c, max_c in zip(min_coords, max_coords))

    sub = mask[box]
    phi = np.full(mask.shape, -(max_tol + 2), dtype=np.float32)
    phi[box] = (distance_transform_edt(sub) - distance_transform_edt(~sub)).astype(
        np.float32
    )

    return phi


def _betti(mask):
    """Calculate Betti numbers"""
    _CROSS = ndim.generate_binary_structure(3, 1)
    _FULL = ndim.generate_binary_structure(3, 3)

    b0 = ndim.label(mask, structure=_FULL)[1]
    b2 = ndim.label(~mask, structure=_CROSS)[1] - 1
    return b0, b0 + b2 - euler_number(mask, connectivity=3), b2


def transform_roi(roi, mapping, is_boolean, max_tol=3.0, return_tol=False):
    """
    After being non-linearly transformed, ROIs can have holes in them,
    or even disappear entirely, depending on the mapping, domain, and codomain.
    This method gives transformed ROIs that preserve, as close as possible,
    the topology of the original ROI: its number of connected components,
    the tunnels through it, and the cavities inside it. Specifically,
    using Betti numbers. Holes are repaired by dilating the warped ROI,
    up to a maximum tolerance. When the topology can be preserved, it
    will be preserved with minimal dilation.

    Parameters
    ----------
    roi : Nifti1Image, str, ndarray
        The ROI to transform. Can be a path or image, which will be
        converted to an ndarray.

    mapping : DiffeomorphicMap object
        A mapping between DWI space and a template.

    is_boolean : bool
        Whether the ROI is boolean or not.

    max_tol : float, optional
        Maximum tolerance for the signed distance field used to warp
        boolean masks.
        Default: 3.0

    return_tol : bool, optional
        Whether to return the tolerance used to warp the boolean mask.
        If is_boolean is False, this parameter is ignored.
        Default: False

    Returns
    -------
    The transformed ROI, as a Nifti1Image.
    """
    if isinstance(roi, str):
        roi = nib.load(roi)
    if isinstance(roi, nib.Nifti1Image):
        roi = roi.get_fdata()

    if not is_boolean:
        return mapping.transform(roi.astype(np.float32), interpolation="linear").astype(
            np.float32
        )

    # remove dipy adding 0s at edges
    mask = roi.astype(bool)
    ones = np.ones(mask.shape, dtype=np.float32)
    valid = mapping.transform(ones, interpolation="linear") > 0.5

    # Preserve the number of connected components in the original mask
    phi = _signed_distance(mask, max_tol)
    phi = mapping.transform(phi, interpolation="linear")

    b0_in, b1_in, b2_in = _betti(mask)
    for t in np.arange(0, max_tol + 0.1, 0.1):
        out = (phi > -t) & valid
        b0_out, b1_out, b2_out = _betti(out)
        intact = (b0_out <= b0_in) and (b1_out <= b1_in) and (b2_out <= b2_in)
        if intact or t >= max_tol:
            break

    if return_tol:
        return out.astype(np.uint8), t
    else:
        return out.astype(np.uint8)


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
