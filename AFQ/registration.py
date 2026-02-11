"""
Registration tools
"""

import nibabel as nib
import numpy as np
from dipy.align.imaffine import AffineMap
from dipy.align.imwarp import DiffeomorphicMap

__all__ = [
    "read_affine_mapping",
    "read_syn_mapping",
    "read_old_mapping",
]


def reduce_shape(shape):
    """
    Reduce dimension in shape to 3 if possible
    """
    try:
        return shape[:3]
    except TypeError:
        return shape


def read_syn_mapping(disp, codisp):
    """
    Read a syn registration mapping from a nifti file

    Parameters
    ----------
    disp : str or Nifti1Image
        If string, file must of an image or ndarray.
        If image, contains the mapping displacement field in each voxel
        from subject to template

    codisp : str or Nifti1Image
        If string, file must of an image or ndarray.
        If image, contains the mapping displacement field in each voxel
        from template to subject

    Returns
    -------
    A :class:`DiffeomorphicMap` object
    """
    if isinstance(disp, str):
        disp = nib.load(disp)

    if isinstance(codisp, str):
        codisp = nib.load(codisp)

    mapping = DiffeomorphicMap(
        dim=3,
        disp_shape=codisp.get_fdata().shape[:3],
        disp_grid2world=None,
        domain_shape=disp.get_fdata().shape[:3],
        domain_grid2world=None,
        codomain_shape=codisp.get_fdata().shape[:3],
        codomain_grid2world=None,
    )
    mapping.forward = disp.get_fdata().astype(np.float32)
    mapping.backward = codisp.get_fdata().astype(np.float32)

    return mapping


def read_affine_mapping(affine, domain_img, codomain_img):
    """
    Read a syn registration mapping from a nifti file

    Parameters
    ----------
    affine : str or ndarray
        If string, file must of an ndarray.
        If ndarray, contains affine transformation used for mapping

    domain_img : str or Nifti1Image

    codomain_img : str or Nifti1Image

    Returns
    -------
    A :class:`AffineMap` object
    """
    if isinstance(affine, str):
        affine = np.load(affine)

    if isinstance(domain_img, str):
        domain_img = nib.load(domain_img)

    if isinstance(codomain_img, str):
        codomain_img = nib.load(codomain_img)

    mapping = AffineMap(
        affine,
        domain_grid_shape=reduce_shape(domain_img.shape),
        domain_grid2world=domain_img.affine,
        codomain_grid_shape=reduce_shape(codomain_img.shape),
        codomain_grid2world=codomain_img.affine,
    )

    return mapping


def read_old_mapping(disp, domain_img, codomain_img):
    """
    Warning: This is only used for pyAFQ tests and backwards compatibility.
    Read old-style registration mapping from a nifti file.

    Parameters
    ----------
    disp : str or Nifti1Image
        If string, file must of an image or ndarray.
        If image, contains the mapping displacement field in each voxel
        Shape (x, y, z, 3, 2)

    domain_img : str or Nifti1Image

    codomain_img : str or Nifti1Image

    Returns
    -------
    A :class:`DiffeomorphicMap` object
    """
    if isinstance(disp, str):
        disp = nib.load(disp)

    if isinstance(domain_img, str):
        domain_img = nib.load(domain_img)

    if isinstance(codomain_img, str):
        codomain_img = nib.load(codomain_img)

    mapping = DiffeomorphicMap(
        3,
        disp.shape[:3],
        disp_grid2world=np.linalg.inv(disp.affine),
        domain_shape=domain_img.shape[:3],
        domain_grid2world=domain_img.affine,
        codomain_shape=codomain_img.shape,
        codomain_grid2world=codomain_img.affine,
    )

    disp_data = disp.get_fdata().astype(np.float32)
    mapping.forward = disp_data[..., 0]
    mapping.backward = disp_data[..., 1]
    mapping.is_inverse = True

    return mapping
