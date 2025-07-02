# -*- coding: utf-8 -*-
# Original source: github.com/scilus/scilpy
# Copyright (c) 2012-- Sherbrooke Connectivity Imaging Lab [SCIL], Université de Sherbrooke.
# Licensed under the MIT License (https://opensource.org/licenses/MIT).
# Modified by John Kruper for pyAFQ
# OpenCL and cosine filtering removed
# Replaced with numba

import numpy as np
import multiprocessing
from tqdm import tqdm

from numba import njit, prange, set_num_threads
import ray

from dipy.reconst.shm import sh_to_sf_matrix
from dipy.data import get_sphere

from AFQ.utils.stats import chunk_indices


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
                      n_threads=None, n_cpus=None):
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
    n_threads: int or None
        Number of threads to use for numba. If None, uses
        the number of available threads.
        Default: None.
    n_cpus: int or None
        Number of CPUs to use for parallel processing with Ray.
        If None, uses the number of available CPUs minus one.
        Default: None.

    References
    ----------
    [1] Poirier and Descoteaux, 2024, "A Unified Filtering Method for
        Estimating Asymmetric Orientation Distribution Functions",
        Neuroimage, https://doi.org/10.1016/j.neuroimage.2024.120516
    """
    if isinstance(sphere, str):
        sphere = get_sphere(name=sphere)

    if sigma_spatial is not None:
        if sigma_spatial <= 0.0:
            raise ValueError('sigma_spatial cannot be <= 0.')
    if sigma_align is not None:
        if sigma_align <= 0.0:
            raise ValueError('sigma_align cannot be <= 0.')
    if sigma_angle is not None:
        if sigma_angle <= 0.0:
            raise ValueError('sigma_align cannot be <= 0.')

    if n_threads is not None:
        set_num_threads(n_threads)

    if n_cpus is None:
        n_cpus = multiprocessing.cpu_count() - 1

    sh_order, full_basis = _get_sh_order_and_fullness(sh_data.shape[-1])

    # build filters
    uv_filter = _unified_filter_build_uv(sigma_angle,
                                         sphere.vertices.astype(np.float64))
    nx_filter = _unified_filter_build_nx(sphere.vertices.astype(np.float64),
                                         sigma_spatial, sigma_align,
                                         False, False)
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

    return _unified_filter_call_python(
        sh_data, nx_filter, uv_filter,
        sigma_range, B, B_inv, sphere,
        n_cpus)


@njit(fastmath=True, cache=True)
def _unified_filter_build_uv(sigma_angle, directions):
    """
    Build the angle filter, weighted on angle between current direction u
    and neighbour direction v.

    Parameters
    ----------
    sigma_angle: float
        Standard deviation of filter. Values at distances greater than
        sigma_angle are clipped to 0 to reduce computation time.
    directions: DIPY sphere directions.
        Vertices from DIPY sphere for sampling the SF.

    Returns
    -------
    weights: ndarray
        Angle filter of shape (N_dirs, N_dirs).
    """
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


@njit(fastmath=True, cache=True)
def _unified_filter_build_nx(directions, sigma_spatial, sigma_align,
                             disable_spatial,
                             disable_align, j_invariance=False):
    """
    Original source: github.com/CHrlS98/aodf-toolkit
    Copyright (c) 2023 Charles Poirier
    Licensed under the MIT License (https://opensource.org/licenses/MIT).
    """
    directions = np.ascontiguousarray(directions.astype(np.float32))

    half_width = int(round(3 * sigma_spatial))
    nx_weights = np.zeros((2 * half_width + 1, 2 * half_width + 1,
                           2 * half_width + 1, len(directions)),
                          dtype=np.float32)

    for i in range(-half_width, half_width + 1):
        for j in range(-half_width, half_width + 1):
            for k in range(-half_width, half_width + 1):
                dxy = np.array([[i, j, k]], dtype=np.float32)
                len_xy = np.sqrt(dxy[0, 0]**2 + dxy[0, 1]**2 + dxy[0, 2]**2)

                if disable_spatial:
                    w_spatial = 1.0
                else:
                    # the length controls spatial weight
                    w_spatial = np.exp(-len_xy**2 / (2 * sigma_spatial**2))

                # the direction controls the align weight
                if i == j == k == 0 or disable_align:
                    # hack for main direction to have maximal weight
                    # w_align = np.ones((1, len(directions)), dtype=np.float32)
                    w_align = np.zeros((1, len(directions)), dtype=np.float32)
                else:
                    dxy /= len_xy
                    w_align = np.arccos(np.clip(np.dot(dxy, directions.T),
                                                -1.0, 1.0))  # 1, N
                w_align = np.exp(-w_align**2 / (2 * sigma_align**2))

                nx_weights[half_width + i, half_width + j, half_width + k] =\
                    w_align * w_spatial

    if j_invariance:
        # A filter is j-invariant if its prediction does not
        # depend on the content of the current voxel
        nx_weights[half_width, half_width, half_width, :] = 0.0

    for ui in range(len(directions)):
        w_sum = np.sum(nx_weights[..., ui])
        nx_weights /= w_sum

    return nx_weights


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
                                B_mat, B_inv, sphere, n_cpus):
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
    n_cpus: int
        Number of CPUs to use for parallel processing with Ray.

    Returns
    -------
    out_sh: ndarray
        Filtered output as SH coefficients.
    """
    nb_sf = len(sphere.vertices)
    mean_sf = np.zeros(sh_data.shape[:-1] + (nb_sf,))
    sh_data = np.ascontiguousarray(sh_data, dtype=np.float64)
    B_mat = np.ascontiguousarray(B_mat, dtype=np.float64)

    h_w, h_h, h_d = nx_filter.shape[:3]
    half_w, half_h, half_d = h_w // 2, h_h // 2, h_d // 2
    sh_data_padded = np.ascontiguousarray(np.pad(
        sh_data,
        ((half_w, half_w), (half_h, half_h), (half_d, half_d), (0, 0)),
        mode='constant'
    ), dtype=np.float64)

    # Apply filter to each sphere vertice
    if n_cpus > 1:
        ray.init(ignore_reinit_error=True)

        sh_data_id = ray.put(sh_data)
        sh_data_padded_id = ray.put(sh_data_padded)
        nx_filter_id = ray.put(nx_filter)
        uv_filter_id = ray.put(uv_filter)
        B_mat_id = ray.put(B_mat)

        @ray.remote(num_cpus=n_cpus)
        def _correlate_batch_remote(batch_u_ids, sh_data, sh_data_padded,
                                    nx_filter, uv_filter, sigma_range, B_mat):
            results = []
            for u_sph_id in batch_u_ids:
                corr = _correlate(
                    sh_data, sh_data_padded, nx_filter,
                    uv_filter, sigma_range, u_sph_id, B_mat
                )
                results.append((u_sph_id, corr))
            return results

        all_u_ids = list(range(nb_sf))
        futures = [
            _correlate_batch_remote.remote(
                batch, sh_data_id, sh_data_padded_id,
                nx_filter_id, uv_filter_id,
                sigma_range, B_mat_id
            )
            for batch in chunk_indices(all_u_ids, n_cpus * 2)
        ]

        # Gather and write results
        for future in tqdm(futures):
            batch_results = ray.get(future)
            for u_sph_id, correlation in batch_results:
                mean_sf[..., u_sph_id] = correlation
    else:
        for u_sph_id in tqdm(range(nb_sf)):
            mean_sf[..., u_sph_id] = _correlate(sh_data, sh_data_padded,
                                                nx_filter, uv_filter,
                                                sigma_range, u_sph_id, B_mat)

    out_sh = np.array([np.dot(i, B_inv) for i in mean_sf],
                      dtype=sh_data.dtype)
    return out_sh


@njit(fastmath=True, parallel=True)
def _correlate(sh_data, sh_data_padded, nx_filter, uv_filter,
               sigma_range, u_index, B_mat):
    """
    Apply the filters to the SH image for the sphere direction
    described by `u_index`.

    Parameters
    ----------
    sh_data: ndarray
        Input SH coefficients.
    sh_data: ndarray
        Input SH coefficients, pre-padded.
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

    # sf_u = np.dot(sh_data, B_mat[:, u_index])
    # sf_v = np.dot(sh_data, B_mat[:, v_indices])
    sf_u = np.zeros(sh_data_padded.shape[:3])
    sf_v = np.zeros(sh_data_padded.shape[:3] + (len(v_indices),))
    for i in prange(sh_data_padded.shape[0]):
        for j in range(sh_data_padded.shape[1]):
            for k in range(sh_data_padded.shape[2]):
                for c in range(sh_data_padded.shape[3]):
                    sf_u[i, j, k] += sh_data_padded[i,
                                                    j, k, c] * B_mat[c, u_index]
                    for vi in range(len(v_indices)):
                        sf_v[i, j, k, vi] += sh_data_padded[i,
                                                            j, k, c] * B_mat[c, v_indices[vi]]

    uv_filter = uv_filter[u_index, v_indices]

    for ii in prange(out_sf.shape[0]):
        for jj in range(out_sf.shape[1]):
            for kk in range(out_sf.shape[2]):
                a = sf_v[ii:ii + h_w, jj:jj + h_h, kk:kk + h_d]
                b = sf_u[ii + half_w, jj + half_h, kk + half_d]
                x_range = a - b

                if sigma_range is None:
                    range_filter = np.ones_like(x_range)
                else:
                    range_filter = _evaluate_gaussian_distribution(
                        x_range, sigma_range)

                # the resulting filter for the current voxel and v_index
                res_filter = range_filter * nx_filter[..., None]
                res_filter =\
                    res_filter * np.reshape(uv_filter,
                                            (1, 1, 1, len(uv_filter)))
                out_sf[ii, jj, kk] = np.sum(
                    sf_v[ii:ii + h_w, jj:jj + h_h, kk:kk + h_d] * res_filter)
                out_sf[ii, jj, kk] /= np.sum(res_filter)

    return out_sf


@njit(fastmath=True, cache=True)
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
    if sigma <= 0.0:
        raise ValueError("Sigma must be greater than 0.")
    cnorm = 1.0 / sigma / np.sqrt(2.0 * np.pi)
    return cnorm * np.exp(-x**2 / 2.0 / sigma**2)
