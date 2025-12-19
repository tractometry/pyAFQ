# -*- coding: utf-8 -*-
# Original source: github.com/scilus/scilpy
# Copyright (c) 2012--
# Sherbrooke Connectivity Imaging Lab [SCIL], Université de Sherbrooke.
# Licensed under the MIT License (https://opensource.org/licenses/MIT).
# Modified by John Kruper for pyAFQ
# OpenCL and cosine filtering removed
# Replaced with numba

import logging

import numpy as np
from dipy.data import get_sphere
from dipy.direction import peak_directions
from dipy.reconst.shm import sh_to_sf, sh_to_sf_matrix, sph_harm_ind_list
from numba import config, njit, prange, set_num_threads
from tqdm import tqdm

logger = logging.getLogger("AFQ")


__all__ = [
    "unified_filtering",
    "compute_asymmetry_index",
    "compute_odd_power_map",
    "compute_nufid_asym",
]


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
    raise ValueError("Invalid number of coefficients for SH basis.")


def unified_filtering(
    sh_data,
    sphere,
    sh_basis="descoteaux07",
    is_legacy=False,
    sigma_spatial=1.0,
    sigma_align=0.8,
    sigma_angle=None,
    rel_sigma_range=0.2,
    n_threads=None,
    low_mem=False,
):
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
    low_mem: bool
        Whether to use the low-memory version of the filtering.
        It will be between 50% and 100% slower.
        Default: False.

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
            raise ValueError("sigma_spatial cannot be <= 0.")
    if sigma_align is not None:
        if sigma_align <= 0.0:
            raise ValueError("sigma_align cannot be <= 0.")
    if sigma_angle is not None:
        if sigma_angle <= 0.0:
            raise ValueError("sigma_angle cannot be <= 0.")

    if n_threads is not None:
        set_num_threads(n_threads)

    if low_mem:
        sh_data = np.ascontiguousarray(sh_data, dtype=np.float32)
        sphere.vertices = sphere.vertices.astype(np.float32)
    else:
        sphere.vertices = sphere.vertices.astype(np.float64)

    sh_order, full_basis = _get_sh_order_and_fullness(sh_data.shape[-1])

    # build filters
    config.THREADING_LAYER = "workqueue"
    uv_filter = _unified_filter_build_uv(sigma_angle, sphere.vertices)
    nx_filter = _unified_filter_build_nx(
        sphere.vertices, sigma_spatial, sigma_align, False, False
    )

    B = sh_to_sf_matrix(
        sphere,
        sh_order_max=sh_order,
        basis_type=sh_basis,
        full_basis=full_basis,
        legacy=is_legacy,
        return_inv=False,
    )
    _, B_inv = sh_to_sf_matrix(
        sphere,
        sh_order_max=sh_order,
        basis_type=sh_basis,
        full_basis=True,
        legacy=is_legacy,
        return_inv=True,
    )

    # compute "real" sigma_range scaled by sf amplitudes
    # if rel_sigma_range is supplied
    sigma_range = None
    if rel_sigma_range is not None:
        if rel_sigma_range <= 0.0:
            raise ValueError("sigma_rangel cannot be <= 0.")
        sigma_range = rel_sigma_range * _get_sf_range(sh_data, B)

    if low_mem:
        return _unified_filter_call_lowmem(
            sh_data, nx_filter, uv_filter, sigma_range, B, B_inv, sphere
        )
    else:
        return _unified_filter_call_python(
            sh_data, nx_filter, uv_filter, sigma_range, B, B_inv, sphere
        )


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
        weights = np.eye(len(directions), dtype=np.float32)
    return weights


@njit(fastmath=True, cache=True)
def _unified_filter_build_nx(
    directions,
    sigma_spatial,
    sigma_align,
    disable_spatial,
    disable_align,
    j_invariance=False,
):
    """
    Original source: github.com/CHrlS98/aodf-toolkit
    Copyright (c) 2023 Charles Poirier
    Licensed under the MIT License (https://opensource.org/licenses/MIT).
    """
    directions = np.ascontiguousarray(directions.astype(np.float32))

    half_width = int(round(3 * sigma_spatial))
    nx_weights = np.zeros(
        (2 * half_width + 1, 2 * half_width + 1, 2 * half_width + 1, len(directions)),
        dtype=np.float32,
    )

    for i in range(-half_width, half_width + 1):
        for j in range(-half_width, half_width + 1):
            for k in range(-half_width, half_width + 1):
                dxy = np.array([[i, j, k]], dtype=np.float32)
                len_xy = np.sqrt(dxy[0, 0] ** 2 + dxy[0, 1] ** 2 + dxy[0, 2] ** 2)

                if disable_spatial:
                    w_spatial = 1.0
                else:
                    # the length controls spatial weight
                    w_spatial = np.exp(-(len_xy**2) / (2 * sigma_spatial**2))

                # the direction controls the align weight
                if i == j == k == 0 or disable_align:
                    # hack for main direction to have maximal weight
                    # w_align = np.ones((1, len(directions)), dtype=np.float32)
                    w_align = np.zeros((1, len(directions)), dtype=np.float32)
                else:
                    dxy /= len_xy
                    w_align = np.arccos(
                        np.clip(np.dot(dxy, directions.T), -1.0, 1.0)
                    )  # 1, N
                w_align = np.exp(-(w_align**2) / (2 * sigma_align**2))

                nx_weights[half_width + i, half_width + j, half_width + k] = (
                    w_align * w_spatial
                )

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
    sf = np.array([np.dot(i, B_mat) for i in sh_data], dtype=sh_data.dtype)
    sf[sf < 0.0] = 0.0
    sf_max = np.max(sf)
    sf_min = np.min(sf)
    return sf_max - sf_min


def _unified_filter_call_python(
    sh_data, nx_filter, uv_filter, sigma_range, B_mat, B_inv, sphere
):
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
    sh_data = np.ascontiguousarray(sh_data, dtype=np.float64)
    B_mat = np.ascontiguousarray(B_mat, dtype=np.float64)

    config.THREADING_LAYER = "workqueue"

    h_w, h_h, h_d = nx_filter.shape[:3]
    half_w, half_h, half_d = h_w // 2, h_h // 2, h_d // 2
    sh_data_padded = np.ascontiguousarray(
        np.pad(
            sh_data,
            ((half_w, half_w), (half_h, half_h), (half_d, half_d), (0, 0)),
            mode="constant",
        ),
        dtype=np.float64,
    )

    for u_sph_id in tqdm(range(nb_sf)):
        mean_sf[..., u_sph_id] = _correlate(
            sh_data, sh_data_padded, nx_filter, uv_filter, sigma_range, u_sph_id, B_mat
        )

    out_sh = np.array([np.dot(i, B_inv) for i in mean_sf], dtype=sh_data.dtype)
    return out_sh


@njit(fastmath=True, parallel=True)
def _correlate(
    sh_data, sh_data_padded, nx_filter, uv_filter, sigma_range, u_index, B_mat
):
    """
    Apply the filters to the SH image for the sphere direction
    described by `u_index`.

    Parameters
    ----------
    sh_data: ndarray
        Input SH coefficients.
    sh_data_padded: ndarray
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
                    sf_u[i, j, k] += sh_data_padded[i, j, k, c] * B_mat[c, u_index]
                    for vi in range(len(v_indices)):
                        sf_v[i, j, k, vi] += (
                            sh_data_padded[i, j, k, c] * B_mat[c, v_indices[vi]]
                        )

    uv_filter = uv_filter[u_index, v_indices]

    for ii in prange(out_sf.shape[0]):
        for jj in range(out_sf.shape[1]):
            for kk in range(out_sf.shape[2]):
                a = sf_v[ii : ii + h_w, jj : jj + h_h, kk : kk + h_d]
                b = sf_u[ii + half_w, jj + half_h, kk + half_d]
                x_range = a - b

                if sigma_range is None:
                    range_filter = np.ones_like(x_range)
                else:
                    range_filter = _evaluate_gaussian_distribution(x_range, sigma_range)

                # the resulting filter for the current voxel and v_index
                res_filter = range_filter * nx_filter[..., None]
                res_filter = res_filter * np.reshape(
                    uv_filter, (1, 1, 1, len(uv_filter))
                )
                out_sf[ii, jj, kk] = np.sum(
                    sf_v[ii : ii + h_w, jj : jj + h_h, kk : kk + h_d] * res_filter
                )
                out_sf[ii, jj, kk] /= np.sum(res_filter)

    return out_sf


def _unified_filter_call_lowmem(
    sh_data, nx_filter, uv_filter, sigma_range, B_mat, B_inv, sphere
):
    """
    Low-memory version of the filtering function.
    """
    nb_sf = len(sphere.vertices)
    mean_sf = np.zeros(sh_data.shape[:-1] + (nb_sf,), dtype=np.float32)
    sh_data = np.ascontiguousarray(sh_data, dtype=np.float32)
    B_mat = np.ascontiguousarray(B_mat, dtype=np.float32)

    config.THREADING_LAYER = "workqueue"

    for u_sph_id in tqdm(range(nb_sf)):
        mean_sf[..., u_sph_id] = _correlate_low_mem(
            sh_data, nx_filter, uv_filter, sigma_range, u_sph_id, B_mat
        )
    out_sh = np.array([np.dot(i, B_inv) for i in mean_sf], dtype=np.float32)
    return out_sh


@njit(fastmath=True, parallel=True)
def _correlate_low_mem(sh_data, nx_filter, uv_filter, sigma_range, u_index, B_mat):
    """
    Low-memory version of the correlate function.
    """
    v_indices = np.flatnonzero(uv_filter[u_index])
    n_v = v_indices.shape[0]

    h_w = nx_filter.shape[0]
    h_h = nx_filter.shape[1]
    h_d = nx_filter.shape[2]
    half_w = h_w // 2
    half_h = h_h // 2
    half_d = h_d // 2
    nx_filter_u = nx_filter[:, :, :, u_index]

    X = sh_data.shape[0]
    Y = sh_data.shape[1]
    Z = sh_data.shape[2]
    C = sh_data.shape[3]
    out_sf = np.zeros((X, Y, Z))

    uv_filter_u = np.empty(n_v)
    for vi in range(n_v):
        uv_filter_u[vi] = uv_filter[u_index, v_indices[vi]]

    B_u = np.empty(C)
    for c in range(C):
        B_u[c] = B_mat[c, u_index]

    B_v = np.empty((C, n_v))
    for vi in range(n_v):
        v_idx = v_indices[vi]
        for c in range(C):
            B_v[c, vi] = B_mat[c, v_idx]

    use_range = sigma_range is not None

    for ii in prange(X):
        for jj in range(Y):
            for kk in range(Z):
                sf_u_center = 0.0
                for c in range(C):
                    sf_u_center += sh_data[ii, jj, kk, c] * B_u[c]

                num = 0.0
                den = 0.0

                for wx in range(h_w):
                    i2 = ii + wx - half_w
                    for wy in range(h_h):
                        j2 = jj + wy - half_h
                        for wz in range(h_d):
                            k2 = kk + wz - half_d
                            if (
                                i2 < 0
                                or i2 >= X
                                or j2 < 0
                                or j2 >= Y
                                or k2 < 0
                                or k2 >= Z
                            ):
                                continue

                            base_nx = nx_filter_u[wx, wy, wz]
                            if base_nx == 0.0:
                                continue

                            for vi in range(n_v):
                                sf_v_val = 0.0
                                for c in range(C):
                                    sf_v_val += sh_data[i2, j2, k2, c] * B_v[c, vi]

                                if use_range:
                                    x = sf_v_val - sf_u_center
                                    x_norm = x / sigma_range
                                    range_w = np.exp(-0.5 * x_norm * x_norm)
                                else:
                                    range_w = 1.0

                                w = base_nx * uv_filter_u[vi] * range_w
                                num += sf_v_val * w
                                den += w

                if den > 0.0:
                    out_sf[ii, jj, kk] = num / den
                else:
                    out_sf[ii, jj, kk] = 0.0

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
    return cnorm * np.exp(-(x**2) / 2.0 / sigma**2)


def compute_asymmetry_index(sh_coeffs, mask):
    """
    Compute asymmetry index (ASI) [1] from
    asymmetric ODF volume expressed in full SH basis.

    Parameters
    ----------
    sh_coeffs: ndarray (x, y, z, ncoeffs)
         Input spherical harmonics coefficients.
    mask: ndarray (x, y, z), bool
         Mask inside which ASI should be computed.

    Returns
    -------
    asi_map: ndarray (x, y, z)
         Asymmetry index map.

    References
    ----------
    [1] S. Cetin Karayumak, E. Özarslan, and G. Unal,
        "Asymmetric Orientation Distribution Functions (AODFs)
        revealing intravoxel geometry in diffusion MRI"
        Magnetic Resonance Imaging, vol. 49, pp. 145-158, Jun. 2018,
        doi: https://doi.org/10.1016/j.mri.2018.03.006.
    """
    order, full_basis = _get_sh_order_and_fullness(sh_coeffs.shape[-1])

    _, l_list = sph_harm_ind_list(order, full_basis=full_basis)

    sign = np.power(-1.0, l_list)
    sign = np.reshape(sign, (1, 1, 1, len(l_list)))
    sh_squared = sh_coeffs**2
    mask = np.logical_and(sh_squared.sum(axis=-1) > 0.0, mask)

    asi_map = np.zeros(sh_coeffs.shape[:-1])
    asi_map[mask] = (
        np.sum(sh_squared * sign, axis=-1)[mask] / np.sum(sh_squared, axis=-1)[mask]
    )

    # Negatives should not happen (amplitudes always positive)
    asi_map = np.clip(asi_map, 0.0, 1.0)
    asi_map = np.sqrt(1 - asi_map**2) * mask

    return asi_map


def compute_odd_power_map(sh_coeffs, mask):
    """
    Compute odd-power map [1] from
    asymmetric ODF volume expressed in full SH basis.

    Parameters
    ----------
    sh_coeffs: ndarray (x, y, z, ncoeffs)
         Input spherical harmonics coefficients.
    mask: ndarray (x, y, z), bool
         Mask inside which odd-power map should be computed.

    Returns
    -------
    odd_power_map: ndarray (x, y, z)
         Odd-power map.

    References
    ----------
    [1] C. Poirier, E. St-Onge, and M. Descoteaux,
        "Investigating the Occurrence of Asymmetric Patterns in
        White Matter Fiber Orientation Distribution Functions"
        [Abstract], In: Proc. Intl. Soc. Mag. Reson. Med. 29 (2021),
        2021 May 15-20, Vancouver, BC, Abstract number 0865.
    """
    order, full_basis = _get_sh_order_and_fullness(sh_coeffs.shape[-1])
    _, l_list = sph_harm_ind_list(order, full_basis=full_basis)
    odd_l_list = (l_list % 2 == 1).reshape((1, 1, 1, -1))

    odd_order_norm = np.linalg.norm(sh_coeffs * odd_l_list, ord=2, axis=-1)

    full_order_norm = np.linalg.norm(sh_coeffs, ord=2, axis=-1)

    asym_map = np.zeros(sh_coeffs.shape[:-1])
    mask = np.logical_and(full_order_norm > 0, mask)
    asym_map[mask] = odd_order_norm[mask] / full_order_norm[mask]

    return asym_map


def compute_nufid_asym(sh_coeffs, sphere, csf, mask):
    """
    Number of fiber directions (nufid) map [1].

    Parameters
    ----------
    sh_coeffs: ndarray (x, y, z, ncoeffs)
        Input spherical harmonics coefficients.

    sphere: DIPY sphere
        Sphere for SH to SF projection.

    csf: ndarray (x, y, z)
        CSF probability map, used to guess the absolute threshold.

    mask: ndarray (x, y, z), bool
         Mask inside which ASI should be computed.

    References
    ----------
    [1] C. Poirier and M. Descoteaux,
        "Filtering Methods for Asymmetric ODFs:
        Where and How Asymmetry Occurs in the White Matter."
        bioRxiv. 2022 Jan 1; 2022.12.18.520881.
        doi: https://doi.org/10.1101/2022.12.18.520881
    """
    sh_order, full_basis = _get_sh_order_and_fullness(sh_coeffs.shape[-1])
    odf = sh_to_sf(
        sh_coeffs,
        sphere,
        sh_order_max=sh_order,
        basis_type="descoteaux07",
        full_basis=full_basis,
        legacy=False,
    )

    # Guess at threshold from 2.0 * mean of ODF maxes in CSF
    absolute_threshold = 2.0 * np.mean(np.max(odf[csf > 0.99], axis=-1))
    odf[odf < absolute_threshold] = 0.0

    nufid_data = np.zeros(sh_coeffs.shape[:-1], dtype=np.float32)
    for ii in tqdm(range(sh_coeffs.shape[0])):
        for jj in range(sh_coeffs.shape[1]):
            for kk in range(sh_coeffs.shape[2]):
                if mask[ii, jj, kk]:
                    _, peaks, _ = peak_directions(
                        odf[ii, jj, kk], sphere, is_symmetric=False
                    )

                    nufid_data[ii, jj, kk] = np.count_nonzero(peaks)

    return nufid_data
