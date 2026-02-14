import numpy as np
from scipy.sparse import csr_matrix


def compute_sparse_decisions(bundles_being_recognized, n_streamlines):
    """
    Compute a sparse matrix of distances to ROIs for the streamlines that are
    currently being recognized. This can be used to weight decisions by distance
    to ROIs, without having to create a dense matrix of distances for all
    streamlines and all bundles.

    Parameters
    ----------
    bundles_being_recognized : dict
        A dictionary of SlsBeingRecognized objects, keyed by bundle name.
    n_streamlines : int
        The total number of streamlines in the original tractogram.

    Returns
    -------
    csr_matrix
        A sparse matrix of shape (number of bundles being recognized, n_streamlines),
        where the entry (i, j) is a score:
            bundles with ROIs result in weights [2.0 to 3.0] with higher scores
            for streamlines closer to ROIs
            Non-ROI bundles result in weight 1.0
            Everything else is 0.0 (implicit in sparse matrices)
    """
    rows, cols, data = [], [], []
    epsilon = 1e-6

    global_max_dist = 0.0
    for b in bundles_being_recognized.values():
        if hasattr(b, "roi_dists"):
            global_max_dist = max(global_max_dist, np.sum(b.roi_dists, axis=-1).max())

    norm_factor = global_max_dist + 1.0

    for b_idx, name in enumerate(bundles_being_recognized.keys()):
        bundle = bundles_being_recognized[name]
        indices = bundle.selected_fiber_idxs

        if hasattr(bundle, "roi_dists"):
            dists = np.sum(bundle.roi_dists, axis=-1)
            dists = np.maximum(dists, epsilon)
            bundle_weights = dists / norm_factor
        else:
            bundle_weights = np.full(len(indices), 2.0, dtype=np.float32)

        rows.extend([b_idx] * len(indices))
        cols.extend(indices)
        data.extend(bundle_weights)

    sparse_scores = csr_matrix(
        (data, (rows, cols)), shape=(len(bundles_being_recognized), n_streamlines)
    )

    # Final Decision: 3.0 - Score
    # ROI bundles result in weights [2.0 to 3.0]
    # No-ROI bundles result in weight 1.0
    sparse_scores.data = 3.0 - sparse_scores.data

    return sparse_scores


def get_conflict_count(sparse_scores):
    """
    Count how many streamlines are being considered for more than one bundle
    """
    sorted_indices = np.sort(sparse_scores.indices)
    is_duplicate = np.diff(sorted_indices) == 0
    num_conflicts = np.sum(is_duplicate)

    return num_conflicts


def remove_conflicts(sparse_scores, bundles_being_recognized):
    """
    Returns a dictionary of {bundle_name: np.array(accepted_indices)}
    """
    coo = sparse_scores.tocoo()

    order = np.lexsort((-coo.data, coo.col))

    mask = np.concatenate(([True], np.diff(coo.col[order]) != 0))
    winner_rows = coo.row[order][mask]
    winner_cols = coo.col[order][mask]

    row_sort = np.argsort(winner_rows)
    winner_rows = winner_rows[row_sort]
    winner_cols = winner_cols[row_sort]

    num_bundles = len(bundles_being_recognized)
    split_indices = np.searchsorted(winner_rows, np.arange(num_bundles + 1))

    for i, b_name in enumerate(bundles_being_recognized.keys()):
        b_sls = bundles_being_recognized[b_name]
        if np.any(b_sls.selected_fiber_idxs[:-1] > b_sls.selected_fiber_idxs[1:]):
            raise NotImplementedError(
                f"Bundle '{b_name}' has unsorted selected_fiber_idxs. "
                "The searchsorted optimization requires sorted indices."
                "This is a bug in the implementation of the bundle "
                "recognition procedure, please report it to the developers."
            )

        accept_idx = b_sls.initiate_selection(f"{b_name} conflicts")
        start, end = split_indices[i], split_indices[i + 1]
        bundle_winners = winner_cols[start:end]

        if len(bundle_winners) > 0:
            local_positions = np.searchsorted(b_sls.selected_fiber_idxs, bundle_winners)
            accept_idx[local_positions] = True
            b_sls.select(local_positions, "conflicts")
        else:
            b_sls.select(accept_idx, "conflicts")
            bundles_being_recognized.pop(b_name)
