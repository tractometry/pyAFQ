import logging
import math
import tempfile
from math import radians

import numpy as np
from dipy.align import vector_fields as vfu
from dipy.align.imwarp import DiffeomorphicMap, mult_aff
from dipy.data import default_sphere
from dipy.reconst.gqi import squared_radial_component
from dipy.tracking.streamline import set_number_of_points
from PIL import Image
from scipy.linalg import blas, pinvh
from scipy.special import gammaln, lpmv
from tqdm import tqdm

logger = logging.getLogger("AFQ")


def get_simplified_transform(self):
    """Constructs a simplified version of this Diffeomorhic Map

    The simplified version incorporates the pre-align transform, as well as
    the domain and codomain affine transforms into the displacement field.
    The resulting transformation may be regarded as operating on the
    image spaces given by the domain and codomain discretization. As a
    result, self.prealign, self.disp_grid2world, self.domain_grid2world and
    self.codomain affine will be None (denoting Identity) in the resulting
    diffeomorphic map.
    """
    if self.dim == 2:
        simplify_f = vfu.simplify_warp_function_2d
    else:
        simplify_f = vfu.simplify_warp_function_3d
    # Simplify the forward transform
    D = self.domain_grid2world
    P = self.prealign
    Rinv = self.disp_world2grid
    Cinv = self.codomain_world2grid

    # this is the matrix which we need to multiply the voxel coordinates
    # to interpolate on the forward displacement field ("in"side the
    # 'forward' brackets in the expression above)
    affine_idx_in = mult_aff(Rinv, mult_aff(P, D))

    # this is the matrix which we need to multiply the voxel coordinates
    # to add to the displacement ("out"side the 'forward' brackets in the
    # expression above)
    affine_idx_out = mult_aff(Cinv, mult_aff(P, D))

    # this is the matrix which we need to multiply the displacement vector
    # prior to adding to the transformed input point
    affine_disp = Cinv

    new_forward = simplify_f(
        self.forward, affine_idx_in, affine_idx_out, affine_disp, self.domain_shape
    )

    # Simplify the backward transform
    C = self.codomain_grid2world
    Pinv = self.prealign_inv
    Dinv = self.domain_world2grid

    affine_idx_in = mult_aff(Rinv, C)
    affine_idx_out = mult_aff(Dinv, mult_aff(Pinv, C))
    affine_disp = mult_aff(Dinv, Pinv)
    new_backward = simplify_f(
        self.backward,
        affine_idx_in,
        affine_idx_out,
        affine_disp,
        self.codomain_shape,
    )
    simplified = DiffeomorphicMap(
        dim=self.dim,
        disp_shape=self.disp_shape,
        disp_grid2world=None,
        domain_shape=self.domain_shape,
        domain_grid2world=None,
        codomain_shape=self.codomain_shape,
        codomain_grid2world=None,
        prealign=None,
    )
    simplified.forward = new_forward
    simplified.backward = new_backward
    return simplified


def gwi_odf(gqmodel, data):
    gqi_vector = np.real(
        squared_radial_component(
            np.dot(gqmodel.b_vector, default_sphere.vertices.T) * gqmodel.Lambda
        )
    )
    odf = blas.dgemm(
        alpha=1.0, a=data.reshape(-1, gqi_vector.shape[0]), b=gqi_vector
    ).reshape((*data.shape[:-1], gqi_vector.shape[1]))
    return odf


def spherical_harmonics(m, n, theta, phi):
    """
    An implementation of spherical harmonics that overcomes conda compilation
    issues. See: https://github.com/nipy/dipy/issues/852
    """
    x = np.cos(phi)
    val = lpmv(m, n, x).astype(complex)
    val *= np.sqrt((2 * n + 1) / 4.0 / np.pi)
    val *= np.exp(0.5 * (gammaln(n - m + 1) - gammaln(n + m + 1)))
    val = val * np.exp(1j * m * theta)
    return val


def in_place_norm(vec, axis=-1, keepdims=False, delvec=True):
    """Return Vectors with Euclidean (L2) norm

    See :term:`unit vector` and :term:`Euclidean norm`

    Parameters
    -------------
    vec : array_like
        Vectors to norm. Squared in the process of calculating the norm.
    axis : int, optional
        Axis over which to norm. By default norm over last axis. If `axis` is
        None, `vec` is flattened then normed. Default is -1.
    keepdims : bool, optional
        If True, the output will have the same number of dimensions as `vec`,
        with shape 1 on `axis`. Default is False.
    delvec : bool, optional
        If True, vec is deleted as soon as possible.
        If False, vec is not deleted, but still squared. Default is True.

    Returns
    ---------
    norm : array
        Euclidean norms of vectors.

    Examples
    --------
    >>> vec = [[8, 15, 0], [0, 36, 77]]
    >>> in_place_norm(vec)
    array([ 17.,  85.])
    >>> vec = [[8, 15, 0], [0, 36, 77]]
    >>> in_place_norm(vec, keepdims=True)
    array([[ 17.],
           [ 85.]])
    >>> vec = [[8, 15, 0], [0, 36, 77]]
    >>> in_place_norm(vec, axis=0)
    array([  8.,  39.,  77.])
    """
    vec = np.asarray(vec)

    if keepdims:
        ndim = vec.ndim
        shape = vec.shape

    np.square(vec, out=vec)
    vec_norm = vec.sum(axis)
    if delvec:
        del vec
    try:
        np.sqrt(vec_norm, out=vec_norm)
    except TypeError:
        vec_norm = vec_norm.astype(float)
        np.sqrt(vec_norm, out=vec_norm)

    if keepdims:
        if axis is None:
            shape = [1] * ndim
        else:
            shape = list(shape)
            shape[axis] = 1
        vec_norm = vec_norm.reshape(shape)

    return vec_norm


def tensor_odf(evals, evecs, sphere, num_batches=100):
    """
    Calculate the tensor Orientation Distribution Function

    Parameters
    ----------
    evals : array (4D)
        Eigenvalues of a tensor. Shape (x, y, z, 3).
    evecs : array (5D)
        Eigenvectors of a tensor. Shape (x, y, z, 3, 3)
    sphere : sphere object
        The ODF will be calculated in each vertex of this sphere.
    num_batches : int
        Split the calculation into batches. This reduces memory usage.
        If memory use is not an issue, set to 1.
        If set to -1, there will be 1 batch per vertex in the sphere.
        Default: 100
    """
    num_vertices = sphere.vertices.shape[0]
    if num_batches == -1:
        num_batches = num_vertices
    batch_size = math.ceil(num_vertices / num_batches)
    batches = range(num_batches)

    mask = np.where((evals[..., 0] > 0) & (evals[..., 1] > 0) & (evals[..., 2] > 0))
    evecs = evecs[mask]

    proj_norm = np.zeros((num_vertices, evecs.shape[0]))

    it = tqdm(batches) if num_batches != 1 else batches
    for i in it:
        start = i * batch_size
        end = (i + 1) * batch_size
        if end > num_vertices:
            end = num_vertices

        proj = np.dot(sphere.vertices[start:end], evecs)
        proj /= np.sqrt(evals[mask])
        proj_norm[start:end, :] = in_place_norm(proj)

    proj_norm **= -3
    proj_norm /= 4 * np.pi * np.sqrt(np.prod(evals[mask], -1))

    odf = np.zeros((evals.shape[:3] + (sphere.vertices.shape[0],)))
    odf[mask] = proj_norm.T
    return odf


def gaussian_weights(bundle, n_points=100, return_mahalnobis=False, stat=np.mean):
    """
    Calculate weights for each streamline/node in a bundle, based on a
    Mahalanobis distance from the core the bundle, at that node (mean, per
    default).

    Parameters
    ----------
    bundle : Streamlines
        The streamlines to weight.
    n_points : int or None, optional
        The number of points to resample to. If this is None, we assume bundle
        is already resampled, and do not do any resampling. Default: 100.
    return_mahalanobis : bool, optional
        Whether to return the Mahalanobis distance instead of the weights.
        Default: False.
    stat : callable, optional.
        The statistic used to calculate the central tendency of streamlines in
        each node. Can be one of {`np.mean`, `np.median`} or other functions
        that have similar API. Default: `np.mean`
    resample : bool, optional
        Whether its necessary to resample the streamlines to the same number
        of points. Only set to False if they are already resampled.
        Default: True.
    Returns
    -------
    w : array of shape (n_streamlines, n_points)
        Weights for each node in each streamline, calculated as its relative
        inverse of the Mahalanobis distance, relative to the distribution of
        coordinates at that node position across streamlines.

    """
    if n_points is not None:
        if isinstance(bundle, np.ndarray):
            bundle = bundle.tolist()
        if isinstance(bundle, list):
            bundle = [np.asarray(item) for item in bundle]
        sls = np.asarray(set_number_of_points(bundle, n_points))
    else:
        sls = bundle

    n_sls, n_nodes, _ = sls.shape

    if n_sls < 15:  # Cov^-1 unstable under this amount
        weights = np.ones((n_sls, n_nodes))
        logger.warning(
            (
                "Not enough streamlines for weight calculation, "
                "weighting everything evenly"
            )
        )
        if return_mahalnobis:
            return np.full((n_sls, n_nodes), np.nan)
        else:
            return weights / np.sum(weights, 0)
    else:
        weights = np.zeros((n_sls, n_nodes))
    diff = stat(sls, axis=0) - sls
    for i in range(n_nodes):
        # This should come back as a 3D covariance matrix with the spatial
        # variance covariance of this node across the different streamlines,
        # converted to a positive semi-definite matrix if necessary
        cov = np.cov(sls[:, i, :].T, ddof=0)
        if np.any(np.linalg.eigvals(cov) < 0):
            eigenvalues, eigenvectors = np.linalg.eigh((cov + cov.T) / 2)
            eigenvalues[eigenvalues < 0] = 0
            cov = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

        # calculate Mahalanobis for node in every fiber
        if np.any(cov > 0):
            weights[:, i] = np.sqrt(
                np.einsum("ij,jk,ik->i", diff[:, i, :], pinvh(cov), diff[:, i, :])
            )

        # In the special case where all the streamlines have the exact same
        # coordinate in this node, the covariance matrix is all zeros, so
        # we can't calculate the Mahalanobis distance, we will instead give
        # each streamline an identical weight, equal to the number of
        # streamlines:
        else:
            weights[:, i] = 0
    if return_mahalnobis:
        return weights

    # weighting is inverse to the distance (the further you are, the less you
    # should be weighted)
    weights = 1 / weights
    # Normalize before returning, so that the weights in each node sum to 1:
    return weights / np.sum(weights, 0)


def make_gif(show_m, out_path, n_frames=36, az_ang=-10, duration=150):
    """
    Make a video from a Fury Show Manager.

    Parameters
    ----------
    show_m : Fury Show Manager
        The Fury Show Manager to use for rendering.

    out_path : str
        The name of the output file.

    n_frames : int
        The number of frames to render.
        Default: 36

    az_ang : float
        The angle to rotate the camera around the
        z-axis for each frame, in degrees.
        Default: -10

    duration : int
        The duration of each frame in the output GIF, in milliseconds.
        Default: 150
    """
    video = []

    show_m.render()
    show_m.window.draw()

    with tempfile.TemporaryDirectory() as tmp_dir:
        for ii in tqdm(range(n_frames), desc="Generating GIF", leave=False):
            frame_fname = f"{tmp_dir}/{ii}.png"
            show_m.screens[0].controller.rotate((radians(az_ang), 0), None)
            show_m.render()
            show_m.window.draw()
            show_m.snapshot(frame_fname)
            video.append(Image.open(frame_fname).convert("RGB"))

        all_left, all_upper = float("inf"), float("inf")
        all_right, all_lower = 0, 0

        for img in video:
            arr = np.array(img)
            bg_color = arr[0, 0]

            mask = np.any(arr != bg_color, axis=-1)

            if np.any(mask):
                rows = np.any(mask, axis=1)
                cols = np.any(mask, axis=0)
                ymin, ymax = np.where(rows)[0][[0, -1]]
                xmin, xmax = np.where(cols)[0][[0, -1]]

                all_left = min(all_left, xmin)
                all_upper = min(all_upper, ymin)
                all_right = max(all_right, xmax)
                all_lower = max(all_lower, ymax)

        if all_left < all_right:
            crop_box = (
                max(0, all_left),
                max(0, all_upper),
                min(video[0].width, all_right),
                min(video[0].height, all_lower),
            )
            cropped_video = [img.crop(crop_box) for img in video]
        else:
            cropped_video = video

        cropped_video[0].save(
            out_path,
            save_all=True,
            append_images=cropped_video[1:],
            duration=duration,
            loop=1,
        )
