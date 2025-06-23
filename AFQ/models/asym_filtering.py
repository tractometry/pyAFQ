# -*- coding: utf-8 -*-
# Original source: github.com/scilus/scilpy
# Copyright (c) 2012-- Sherbrooke Connectivity Imaging Lab [SCIL], Université de Sherbrooke.
# Licensed under the MIT License (https://opensource.org/licenses/MIT).
# Modified by John Kruper for pyAFQ
# OpenCL and cosine filtering removed

import numpy as np
import logging
from dipy.reconst.shm import sh_to_sf_matrix
from dipy.data import get_sphere


__all__ = ["unified_filtering"]


def _get_sh_order_and_fullness(ncoeffs):
    """
    Get the order of the SH basis from the number of SH coefficients
    as well as a boolean indicating if the basis is full.
    """
    # the two curves (sym and full) intersect at ncoeffs = 1, in what
    # case both bases correspond to order 1.
    sym_order = (-3.0 + np.sqrt(1.0 + 8.0 * ncoeffs)) / 2.0
    if sym_order.is_integer():
        return sym_order, False
    full_order = np.sqrt(ncoeffs) - 1.0
    if full_order.is_integer():
        return full_order, True
    raise ValueError('Invalid number of coefficients for SH basis.')


def unified_filtering(sh_data, sphere,
                      sh_basis='descoteaux07', is_legacy=False,
                      sigma_spatial=1.0, sigma_align=0.8,
                      sigma_angle=None, rel_sigma_range=0.2,
                      win_hwidth=None, exclude_center=False):
    """
    Unified asymmetric filtering as described in [1].

    Parameters
    ----------
    sh_data: ndarray
        SH coefficients image.
    sphere: str or DIPY sphere
        Name of the DIPY sphere to use for SH to SF projection.
    sh_basis: str
        SH basis definition used for input and output SH image.
        One of 'descoteaux07' or 'tournier07'.
        Default: 'descoteaux07'.
    is_legacy: bool
        Whether the legacy SH basis definition should be used.
        Default: False.
    sigma_spatial: float or None
        Standard deviation of spatial filter. Can be None to replace
        by mean filter, in what case win_hwidth must be given.
    sigma_align: float or None
        Standard deviation of alignment filter. `None` disables
        alignment filtering.
    sigma_angle: float or None
        Standard deviation of the angle filter. `None` disables
        angle filtering.
    rel_sigma_range: float or None
        Standard deviation of the range filter, relative to the
        range of SF amplitudes. `None` disables range filtering.
    disable_spatial: bool, optional
        Replace gaussian filter by a mean filter for spatial filter.
        The value from `sigma_spatial` is still used for setting the
        size of the filtering window.
    win_hwidth: int, optional
        Half-width of the filtering window. When None, the
        filtering window half-width is given by (6*sigma_spatial + 1).
    exclude_center: bool, optional
        Assign a weight of 0 to the center voxel of the filter.

    References
    ----------
    [1] Poirier and Descoteaux, 2024, "A Unified Filtering Method for
        Estimating Asymmetric Orientation Distribution Functions",
        Neuroimage, https://doi.org/10.1016/j.neuroimage.2024.120516
    """
    if sigma_spatial is None and win_hwidth is None:
        raise ValueError('sigma_spatial and win_hwidth cannot both be None')

    if isinstance(sphere, str):
        sphere = get_sphere(name=sphere)

    if sigma_spatial is not None:
        if sigma_spatial <= 0.0:
            raise ValueError('sigma_spatial cannot be <= 0.')
        # calculate half-width from sigma_spatial
        half_width = int(round(3 * sigma_spatial))
    if sigma_align is not None:
        if sigma_align <= 0.0:
            raise ValueError('sigma_align cannot be <= 0.')
    if sigma_angle is not None:
        if sigma_angle <= 0.0:
            raise ValueError('sigma_align cannot be <= 0.')

    sh_order, full_basis = _get_sh_order_and_fullness(sh_data.shape[-1])

    # overwrite half-width if win_hwidth is supplied
    if win_hwidth is not None:
        half_width = win_hwidth

    # filter shape computed from half_width
    filter_shape = (half_width * 2 + 1, half_width * 2 + 1, half_width * 2 + 1)

    # build filters
    uv_filter = _unified_filter_build_uv(sigma_angle, sphere)
    nx_filter = _unified_filter_build_nx(filter_shape, sigma_spatial,
                                         sigma_align, sphere, exclude_center)

    B = sh_to_sf_matrix(sphere, sh_order, sh_basis, full_basis,
                        legacy=is_legacy, return_inv=False)
    _, B_inv = sh_to_sf_matrix(sphere, sh_order, sh_basis, True,
                               legacy=is_legacy, return_inv=True)

    # compute "real" sigma_range scaled by sf amplitudes
    # if rel_sigma_range is supplied
    sigma_range = None
    if rel_sigma_range is not None:
        if rel_sigma_range <= 0.0:
            raise ValueError('sigma_rangel cannot be <= 0.')
        sigma_range = rel_sigma_range * _get_sf_range(sh_data, B)

    return _unified_filter_call_python(sh_data, nx_filter, uv_filter,
                                       sigma_range, B, B_inv, sphere)


def _unified_filter_build_uv(sigma_angle, sphere):
    """
    Build the angle filter, weighted on angle between current direction u
    and neighbour direction v.

    Parameters
    ----------
    sigma_angle: float
        Standard deviation of filter. Values at distances greater than
        sigma_angle are clipped to 0 to reduce computation time.
    sphere: DIPY sphere
        Sphere used for sampling the SF.

    Returns
    -------
    weights: ndarray
        Angle filter of shape (N_dirs, N_dirs).
    """
    directions = sphere.vertices
    if sigma_angle is not None:
        dot = directions.dot(directions.T)
        x = np.arccos(np.clip(dot, -1.0, 1.0))
        weights = _evaluate_gaussian_distribution(x, sigma_angle)
        mask = x > (3.0 * sigma_angle)
        weights[mask] = 0.0
        weights /= np.sum(weights, axis=-1)
    else:
        weights = np.eye(len(directions))
    return weights


def _unified_filter_build_nx(filter_shape, sigma_spatial, sigma_align,
                             sphere, exclude_center):
    """
    Build the combined spatial and alignment filter.

    Parameters
    ----------
    filter_shape: tuple
        Dimensions of filtering window.
    sigma_spatial: float or None
        Standard deviation of spatial filter. None disables Gaussian
        weighting for spatial filtering.
    sigma_align: float or None
        Standard deviation of the alignment filter. None disables Gaussian
        weighting for alignment filtering.
    sphere: DIPY sphere
        Sphere for SH to SF projection.
    exclude_center: bool
        Whether the center voxel is included in the neighbourhood.

    Returns
    -------
    weights: ndarray
        Combined spatial + alignment filter of shape (W, H, D, N) where
        N is the number of sphere directions.
    """
    directions = sphere.vertices.astype(np.float32)

    grid_directions = _get_window_directions(filter_shape).astype(np.float32)
    distances = np.linalg.norm(grid_directions, axis=-1)
    grid_directions[distances > 0] = grid_directions[distances > 0] /\
        distances[distances > 0][..., None]

    if sigma_spatial is None:
        w_spatial = np.ones(filter_shape)
    else:
        w_spatial = _evaluate_gaussian_distribution(distances, sigma_spatial)

    if sigma_align is None:
        w_align = np.ones(np.append(filter_shape, (len(directions),)))
    else:
        cos_theta = np.clip(grid_directions.dot(directions.T), -1.0, 1.0)
        theta = np.arccos(cos_theta)
        theta[filter_shape[0] // 2,
              filter_shape[1] // 2,
              filter_shape[2] // 2] = 0.0
        w_align = _evaluate_gaussian_distribution(theta, sigma_align)

    # resulting filter
    w = w_spatial[..., None] * w_align

    if exclude_center:
        w[filter_shape[0] // 2,
          filter_shape[1] // 2,
          filter_shape[2] // 2] = 0.0

    # normalize and return
    w /= np.sum(w, axis=(0, 1, 2), keepdims=True)
    return w


def _get_sf_range(sh_data, B_mat):
    """
    Get the range of SF amplitudes for input `sh_data`.

    Parameters
    ----------
    sh_data: ndarray
        Spherical harmonics coefficients image.
    B_mat: ndarray
        SH to SF projection matrix.

    Returns
    -------
    sf_range: float
        Range of SF amplitudes.
    """
    sf = np.array([np.dot(i, B_mat) for i in sh_data],
                  dtype=sh_data.dtype)
    sf[sf < 0.0] = 0.0
    sf_max = np.max(sf)
    sf_min = np.min(sf)
    return sf_max - sf_min


def _unified_filter_call_python(sh_data, nx_filter, uv_filter, sigma_range,
                                B_mat, B_inv, sphere):
    """
    Run filtering using pure python implementation.

    Parameters
    ----------
    sh_data: ndarray
        Input SH data.
    nx_filter: ndarray
        Combined spatial and alignment filter.
    uv_filter: ndarray
        Angle filter.
    sigma_range: float or None
        Standard deviation of range filter. None disables range filtering.
    B_mat: ndarray
        SH to SF projection matrix.
    B_inv: ndarray
        SF to SH projection matrix.
    sphere: DIPY sphere
        Sphere for SH to SF projection.

    Returns
    -------
    out_sh: ndarray
        Filtered output as SH coefficients.
    """
    nb_sf = len(sphere.vertices)
    mean_sf = np.zeros(sh_data.shape[:-1] + (nb_sf,))

    # Apply filter to each sphere vertice
    for u_sph_id in range(nb_sf):
        if u_sph_id % 20 == 0:
            logging.info('Processing direction: {}/{}'
                         .format(u_sph_id, nb_sf))
        mean_sf[..., u_sph_id] = _correlate(sh_data, nx_filter, uv_filter,
                                            sigma_range, u_sph_id, B_mat)

    out_sh = np.array([np.dot(i, B_inv) for i in mean_sf],
                      dtype=sh_data.dtype)
    return out_sh


def _correlate(sh_data, nx_filter, uv_filter, sigma_range, u_index, B_mat):
    """
    Apply the filters to the SH image for the sphere direction
    described by `u_index`.

    Parameters
    ----------
    sh_data: ndarray
        Input SH coefficients.
    nx_filter: ndarray
        Combined spatial and alignment filter.
    uv_filter: ndarray
        Angle filter.
    sigma_range: float or None
        Standard deviation of range filter. None disables range filtering.
    u_index: int
        Index of the current sphere direction to process.
    B_mat: ndarray
        SH to SF projection matrix.

    Returns
    -------
    out_sf: ndarray
        Output SF amplitudes along the direction described by `u_index`.
    """
    v_indices = np.flatnonzero(uv_filter[u_index])
    nx_filter = nx_filter[..., u_index]
    h_w, h_h, h_d = nx_filter.shape[:3]
    half_w, half_h, half_d = h_w // 2, h_h // 2, h_d // 2
    out_sf = np.zeros(sh_data.shape[:3])
    sh_data = np.pad(sh_data, ((half_w, half_w),
                               (half_h, half_h),
                               (half_d, half_d),
                               (0, 0)))

    sf_u = np.dot(sh_data, B_mat[:, u_index])
    sf_v = np.dot(sh_data, B_mat[:, v_indices])
    uv_filter = uv_filter[u_index, v_indices]

    _get_range = _evaluate_gaussian_distribution\
        if sigma_range is not None else lambda x, _: np.ones_like(x)

    for ii in range(out_sf.shape[0]):
        for jj in range(out_sf.shape[1]):
            for kk in range(out_sf.shape[2]):
                a = sf_v[ii:ii + h_w, jj:jj + h_h, kk:kk + h_d]
                b = sf_u[ii + half_w, jj + half_h, kk + half_d]
                x_range = a - b
                range_filter = _get_range(x_range, sigma_range)

                # the resulting filter for the current voxel and v_index
                res_filter = range_filter * nx_filter[..., None]
                res_filter =\
                    res_filter * np.reshape(uv_filter,
                                            (1, 1, 1, len(uv_filter)))
                out_sf[ii, jj, kk] = np.sum(
                    sf_v[ii:ii + h_w, jj:jj + h_h, kk:kk + h_d] * res_filter)
                out_sf[ii, jj, kk] /= np.sum(res_filter)

    return out_sf


def _evaluate_gaussian_distribution(x, sigma):
    """
    1-dimensional 0-centered Gaussian distribution
    with standard deviation sigma.

    Parameters
    ----------
    x: ndarray or float
        Points where the distribution is evaluated.
    sigma: float
        Standard deviation.

    Returns
    -------
    out: ndarray or float
        Values at x.
    """
    assert sigma > 0.0, "Sigma must be greater than 0."
    cnorm = 1.0 / sigma / np.sqrt(2.0 * np.pi)
    return cnorm * np.exp(-x**2 / 2 / sigma**2)


def _get_window_directions(shape):
    """
    Get directions from center voxel to all neighbours
    for a window of given shape.

    Parameters
    ----------
    shape: tuple
        Dimensions of the window.

    Returns
    -------
    grid: ndarray
        Grid containing the direction from the center voxel to
        the current position for all positions inside the window.
    """
    grid = np.indices(shape)
    grid = np.moveaxis(grid, 0, -1)
    grid = grid - np.asarray(shape) // 2
    return grid
