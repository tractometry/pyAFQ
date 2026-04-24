# Original source: github.com/SlicerDMRI/whitematteranalysis
# Copyright 2026 BWH and 3D Slicer contributors
# Licensed under 3D Slicer license (BSD style; https://github.com/SlicerDMRI/whitematteranalysis/blob/master/License.txt)  # noqa
# Modified by John Kruper for pyAFQ
# Modifications:
# 1. Only mean distance included, and mean distance replaced with numba version.
# 2. Uses atlas data from dictionary and numpy files rather than pickled files,
# to avoid additional dependencies.
# 3. Added function to move template streamlines
#    to subject space to calculate distances.

import numpy as np
import scipy
from dipy.io.stateful_tractogram import Space
from numba import njit, prange

import AFQ.data.fetch as afd
import AFQ.recognition.utils as abu
import AFQ.utils.streamlines as aus


@njit(parallel=True)
def _compute_mean_euclidean_matrix(group_n, group_m):
    len_n = group_n.shape[0]
    len_m = group_m.shape[0]
    num_points = group_n.shape[1]

    dist_matrix = np.empty((len_n, len_m), dtype=np.float64)

    for i in prange(len_n):
        for j in range(len_m):
            sum_dist = 0.0
            sum_dist_ref = 0.0

            for k in range(num_points):
                dx = group_n[i, k, 0] - group_m[j, k, 0]
                dx_ref = group_n[i, k, 0] + group_m[j, k, 0]
                dy = group_n[i, k, 1] - group_m[j, k, 1]
                dz = group_n[i, k, 2] - group_m[j, k, 2]

                sum_dist += np.sqrt(dx * dx + dy * dy + dz * dz)
                sum_dist_ref += np.sqrt(dx_ref * dx_ref + dy * dy + dz * dz)

            mean_d = sum_dist / num_points
            mean_d_ref = sum_dist_ref / num_points

            final_d = min(mean_d, mean_d_ref)
            dist_matrix[i, j] = final_d * final_d

    return dist_matrix.T


def _distance_to_similarity(distance, sigmasq):
    similarities = np.exp(-distance / (sigmasq))

    return similarities


def _rectangular_similarity_matrix(fgarray_sub, fgarray_atlas, sigma):
    distances = _compute_mean_euclidean_matrix(fgarray_sub, fgarray_atlas)

    sigmasq = sigma * sigma
    similarity_matrix = _distance_to_similarity(distances, sigmasq)

    return similarity_matrix


def spectral_atlas_label(
    sub_fgarray,
    atlas_fgarray,
    atlas_data=None,
    sigma_multiplier=1.0,
    cluster_indices=None,
):
    """
    Use an existing atlas to label a new streamlines.

    Parameters
    ----------
    sub_fgarray : ndarray
        Resampled fiber group to be labeled.
    atlas_fgarray : ndarray
        Resampled atlas to use for labelling.
    atlas_data : dict, optional
        Precomputed atlas data formatted as a dictionary of arrays and floats.
        See `afd.read_org800_templates` as a reference.
    sigma_multiplier : float, optional
        Multiplier for the sigma value used in computing the similarity
        matrix. Default is 1.0.
    cluster_indices : list of int, optional
        If provided, only these cluster indices from the atlas will be used
        for labeling. Default is None, which uses all clusters.

    Returns
    -------
    tuple of (ndarray, ndarray)
        Cluster indices for all the fibers and their embedding
    """
    if atlas_data is None:
        atlas_data = afd.read_org800_templates(load_trx=False)

    number_fibers = sub_fgarray.shape[0]
    sz = atlas_fgarray.shape[0]

    # Compute fiber similarities.
    B = _rectangular_similarity_matrix(
        sub_fgarray, atlas_fgarray, sigma=atlas_data["sigma"] * sigma_multiplier
    )

    # Do Normalized Cuts transform of similarity matrix.
    # row sum estimate for current B part of the matrix
    row_sum_2 = np.sum(B, axis=0) + np.dot(atlas_data["row_sum_matrix"], B)

    # This happens plenty in our cases. Why?
    # Maybe a probabilistic vs UKF thing?
    # In practice, this is not an issue since we just set to a small value.
    if any(row_sum_2 <= 0):
        row_sum_2[row_sum_2 < 0] = 1e-4

    # Normalized cuts normalization
    row_sum = np.concatenate((atlas_data["row_sum_1"], row_sum_2))
    dhat = np.sqrt(np.divide(1, row_sum))
    B = np.multiply(B, np.outer(dhat[0:sz], dhat[sz:].T))

    # Compute embedding using eigenvectors
    V = np.dot(
        np.dot(B.T, atlas_data["e_vec"]), np.diag(np.divide(1.0, atlas_data["e_val"]))
    )
    V = np.divide(V, atlas_data["e_vec_norm"])
    n_eigen = int(atlas_data["number_of_eigenvectors"])
    embed = np.zeros((number_fibers, n_eigen))
    for i in range(0, n_eigen):
        embed[:, i] = np.divide(V[:, -(i + 2)], V[:, -1])

    # Label streamlines using centroids from atlas
    if cluster_indices is not None:
        centroids = atlas_data["centroids"][cluster_indices, :]
        cluster_idx, _ = scipy.cluster.vq.vq(embed, centroids)
        cluster_idx = np.array([cluster_indices[i] for i in cluster_idx])
    else:
        cluster_idx, _ = scipy.cluster.vq.vq(embed, atlas_data["centroids"])

    return cluster_idx, embed


def subcluster_by_atlas(
    sub_trk, mapping, dwi_ref, cluster_indices, atlas_data=None, n_points=20
):
    """
    Use an existing atlas to label a new set of streamlines, and return the
    cluster indices for each streamline.

    Parameters
    ----------
    sub_trk : StatefulTractogram
        streamlines to be labeled.
    mapping : DIPY or pyAFQ mapping
        Mapping to use to move streamlines.
    dwi_ref : Nifti1Image
        Image defining reference for where the atlas streamlines move to.
    cluster_indices : list of int
        Cluster indices from the atlas to use for labeling.
    atlas_data : dict, optional
        Precomputed atlas data formatted as a dictionary of arrays and floats.
        See `afd.read_org800_templates` as a reference.
    n_points : int, optional
        Number of points to resample streamlines to for labeling. Default is 20.
    """

    if atlas_data is None:
        atlas_data = afd.read_org800_templates()
    atlas_sft = atlas_data["tracks_reoriented"]

    moved_atlas_sft = aus.move_streamlines(
        atlas_sft, "subject", mapping, dwi_ref, to_space=Space.RASMM
    )
    atlas_fgarray = np.array(abu.resample_tg(moved_atlas_sft.streamlines, n_points))

    sub_trk.to_rasmm()
    sub_fgarray = np.array(abu.resample_tg(sub_trk.streamlines, n_points))

    cluster_idxs, _ = spectral_atlas_label(
        sub_fgarray,
        atlas_fgarray,
        atlas_data=atlas_data,
        cluster_indices=cluster_indices,
    )

    return cluster_idxs
