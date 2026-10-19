"""MK-Curve correction of implausible DKI voxels.

Zhang F, Ning L, O'Donnell LJ, Pasternak O. MK-curve - Characterizing the
relation between mean kurtosis and alterations in the diffusion MRI signal.
NeuroImage 196 (2019): 68-80.

Python implementation written with assistance from Claude (Anthropic),
reviewed and tested by John Kruper.

License: this file is a modified version of MK-Curve
(https://github.com/zhangfanmark/MK-Curve, Fan Zhang, The Brigham and
Women's Hospital, Inc.); it is not the original MK-Curve software. All or
portions of this licensed product (such portions are the "Software") have
been obtained under license from The Brigham and Women's Hospital, Inc. and
are subject to the terms and conditions in LICENSE_MKCURVE.txt, located
next to this file.
"""

import numpy as np
from dipy.core.gradients import gradient_table
from dipy.data import get_sphere
from dipy.reconst.dki import design_matrix
from scipy.ndimage import convolve
from tqdm.auto import tqdm

__all__ = ["mkcurve_correct"]


def _mk_curves(Y, B, syn, tile=8192):
    """MK for every voxel (rows of Y, b0 in column 0) at each synthetic b0."""
    scale = np.sqrt((B**2).sum(0))
    B = B / scale  # equilibrate columns: keeps the normal equations stable
    P = np.linalg.pinv(B)
    BB = (B[:, :, None] * B[:, None, :]).reshape(len(B), -1)
    n = get_sphere(name="repulsion200").vertices
    Bd = design_matrix(gradient_table(np.ones(len(n)), bvecs=n, b0_threshold=0))
    T2, T4 = -Bd[:, :6] / scale[:6], 6 * Bd[:, 6:21] / scale[6:21]
    out = np.empty((len(syn), len(Y)))
    tiles = range(0, len(Y), tile)
    with tqdm(total=len(tiles) * len(syn), desc="MK-Curve sweep", unit="b0") as pbar:
        for aa in tiles:
            L = np.log(np.maximum(Y[aa : aa + tile], np.finfo(float).eps))
            for i, s in enumerate(syn):
                L[:, 0] = np.log(s)
                lw = L @ P.T @ B.T  # log of the OLS-predicted signal
                w2 = np.exp(2 * (lw - lw.max(1, keepdims=True)))  # WLS weights
                G = (w2 @ BB).reshape(-1, 22, 22)
                r = ((w2 * L) @ B)[..., None]
                try:
                    x = np.linalg.solve(G, r)[..., 0]
                except np.linalg.LinAlgError:  # rare exactly singular voxel
                    x = (np.linalg.pinv(G) @ r)[..., 0]
                # MK = mean over directions of W(n) MD^2 / ADC(n)^2; x[6:21] = W MD^2
                out[i, aa : aa + tile] = (
                    (x[:, 6:21] @ T4.T) / (x[:, :6] @ T2.T) ** 2
                ).mean(1)
                pbar.update()
    return out


def _detect(mk, b0, syn, th):
    """abnormal_voxel_detection.m, vectorised. Returns (flag, corrected b0)."""
    ns, nv = mk.shape
    mk = np.where(mk > 5, -1.0, mk)
    neg = (mk < 0)[::-1]
    first = np.where(neg.any(0), neg.argmax(0) + 1, ns)
    first[first == 1] = ns
    pos_zero = ns - first + 1  # 0-based index of zero-MK b0
    mk[np.arange(ns)[:, None] < pos_zero] = 0.0
    pos_max = np.argmax(np.where(np.isnan(mk), -np.inf, mk), 0)
    zero_b0, max_b0 = syn[pos_zero], syn[pos_max]
    flag = b0 <= zero_b0 * (1 - th) + max_b0 * th
    first_gt = np.searchsorted(syn, b0, side="right")  # unusual curve shapes
    flag |= (pos_max - pos_zero <= 2) & (first_gt < ns) & (first_gt <= pos_max + 6)
    return flag, np.where(flag, zero_b0 * (1 - th) + max_b0 * th, b0)


def _fill_nonpositive(data, mask):
    """Replace in-mask values <= 0 or NaN by the mean of valid 26-neighbours."""
    k = np.ones((3, 3, 3))
    k[1, 1, 1] = 0
    for g in range(data.shape[-1]):
        vol = data[..., g]
        bad = mask & ~(vol > 0)
        if bad.any():
            ok = np.where(vol > 0, vol, 0.0)
            n = convolve((vol > 0).astype(float), k, mode="constant")
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = convolve(ok, k, mode="constant") / n
            vol[bad] = np.where(n > 0, mean, vol)[bad]
    return data


def mkcurve_correct(data, gtab, mask, threshold=0.5, syn_b0=None):
    """Correct implausible DKI voxels with MK-Curve.

    Parameters
    ----------
    data : 4D array of DWIs.
    gtab : GradientTable.
    mask : 3D boolean brain mask.
    threshold : float, MK-Curve's lambda (0.5 recommended).
    syn_b0 : 1D array of synthetic b0 values. Default, as in MK-Curve: from
        0.1x to 2x the mean in-mask b0 in about 200 steps.

    Returns
    -------
    data : 4D float array: [mean b0 (corrected), non-b0 volumes].
    gtab : GradientTable matching the returned data.
    flagged : 3D boolean array of corrected voxels.
    """
    mask = np.asarray(mask, bool)
    b0s = gtab.b0s_mask
    out = np.concatenate(
        [data[..., b0s].mean(-1, keepdims=True), data[..., ~b0s]], axis=-1
    ).astype(float)
    out = _fill_nonpositive(out, mask)
    bvals = np.r_[0, gtab.bvals[~b0s]]
    bvecs = np.vstack([[0, 0, 0], gtab.bvecs[~b0s]])
    new_gtab = gradient_table(bvals, bvecs=bvecs, b0_threshold=gtab.b0_threshold)

    Y = out[mask]
    if syn_b0 is None:
        m = np.nanmean(Y[:, 0])
        lo, hi = np.round(0.1 * m), np.round(2 * m)
        step = max(1.0, np.round((hi - lo) / 200))
        syn_b0 = lo + step * np.arange(int(np.floor((hi - lo) / step)) + 1)
    syn_b0 = np.asarray(syn_b0, float)

    curves = _mk_curves(Y, design_matrix(new_gtab), syn_b0)
    flag, fixed = _detect(curves, Y[:, 0], syn_b0, threshold)
    out[mask, 0] = fixed
    flagged = np.zeros(mask.shape, bool)
    flagged[mask] = flag
    return out, new_gtab, flagged
