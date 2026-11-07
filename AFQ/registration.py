"""
Registration tools
"""

import nibabel as nib
import numpy as np
from dipy.align.imaffine import AffineMap
from dipy.align.imwarp import DiffeomorphicMap
from dipy.align.vector_fields import invert_vector_field_fixed_point_3d

__all__ = [
    "read_affine_mapping",
    "read_syn_mapping",
    "read_fireants_mapping",
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


def read_fireants_mapping(coords, domain_img, codomain_img, prealign=None):
    """
    Read a syn registration mapping from FireANTs warped coordinates

    Parameters
    ----------
    coords : ndarray, shape (1, Z, Y, X, 3)
        Normalized sampling coordinates (x, y, z order, align_corners=True)
        on the codomain grid, from FireANTs get_warped_coordinates.

    domain_img : Nifti1Image
        Subject image.

    codomain_img : Nifti1Image
        Template image. FireANTs registered the prealigned subject
        on this grid.

    prealign : ndarray, shape (4, 4), optional
        Affine from template world to subject world
        applied before FireANTs.

    Returns
    -------
    A :class:`DiffeomorphicMap` object
    """
    shape = np.array(codomain_img.shape[:3])
    # normalized coordinates to voxel displacements, then to mm
    disp = (coords[0].transpose(2, 1, 0, 3).astype(np.float64) + 1) * (shape - 1) / 2
    disp -= np.moveaxis(np.indices(shape), 0, -1)
    disp = np.ascontiguousarray(disp @ codomain_img.affine[:3, :3].T, dtype=np.float32)

    # starting from -disp keeps translations exact
    codisp = invert_vector_field_fixed_point_3d(
        disp,
        np.linalg.inv(codomain_img.affine),
        nib.affines.voxel_sizes(codomain_img.affine),
        100,
        1e-3,
        start=-disp,
    )
    mapping = DiffeomorphicMap(
        dim=3,
        disp_shape=codomain_img.shape[:3],
        disp_grid2world=codomain_img.affine,
        domain_shape=domain_img.shape[:3],
        domain_grid2world=domain_img.affine,
        codomain_shape=codomain_img.shape[:3],
        codomain_grid2world=codomain_img.affine,
        prealign=None if prealign is None else np.linalg.inv(prealign),
    )
    mapping.forward = codisp
    mapping.backward = disp

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


def read_old_mapping(disp, domain_img, codomain_img, prealign=None):
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
        prealign=prealign,
    )

    disp_data = disp.get_fdata().astype(np.float32)
    mapping.forward = disp_data[..., 0]
    mapping.backward = disp_data[..., 1]
    mapping.is_inverse = False

    return mapping
