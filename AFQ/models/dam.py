import logging
import numpy as np
from tqdm import tqdm

from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from dipy.segment.tissue import compute_directional_average


logger = logging.getLogger('AFQ')


def fit_dam(data, gtab, b0_img, dam_low_signal_thresh=50):
    """
    direction-averaged signal map (DAM) [1] slope and intercept

    Parameters
    ----------
    data : ndarray
        The diffusion data with shape (x, y, z, b).
    gtab : GradientTable
        The gradient table containing b-values and b-vectors.
    b0_img : Nifti1Image
        The b0 image used to compute the mean signal.
        This should be a 3D image with shape (x, y, z).
    dam_low_signal_thresh : float, optional
        The threshold below which a voxel is considered to have low signal.
        Default: 50

    References
    ----------
    .. [1] Cheng, H., Newman, S., Afzali, M., Fadnavis, S.,
            & Garyfallidis, E. (2020). Segmentation of the brain using
            direction-averaged signal of DWI images. 
            Magnetic Resonance Imaging, 69, 1-7. Elsevier. 
            https://doi.org/10.1016/j.mri.2020.02.010
    """
    # Precompute unique b-values, masks
    unique_bvals = np.unique(gtab.bvals)
    if len(unique_bvals) <= 2:
        raise ValueError(("Insufficient unique b-values for fitting DAM. "
                         "Note, DAM requires multi-shell data"))
    masks = gtab.bvals[:, np.newaxis] == unique_bvals[np.newaxis, 1:]

    b0_data = b0_img.get_fdata()

    # If the mean signal for b=0 is too low,
    # set those voxels to 0 for both P and V
    valid_voxels = b0_data >= dam_low_signal_thresh

    params_map = np.zeros((*data.shape[:-1], 2))
    logger.info("Fitting directional average map (DAM)...")
    for idx in tqdm(range(data.shape[0] * data.shape[1] * data.shape[2])):
        i, j, k = np.unravel_index(idx, data.shape[:-1])
        if valid_voxels[i, j, k]:
            aa, bb = compute_directional_average(
                data[i, j, k, :],
                gtab.bvals,
                masks=masks,
                b0_mask=gtab.b0s_mask,
                s0_map=b0_data[i, j, k],
                low_signal_threshold=dam_low_signal_thresh,
            )
            if aa > 0.01 and bb < 0:
                params_map[i, j, k, 0] = aa
                params_map[i, j, k, 1] = -bb
    return params_map


def csf_dam(dam_intercept_data):
    """
    CSF probability map from DAM intercept

    Parameters
    ----------
    dam_intercept_data : ndarray
        The DAM intercept data with shape (x, y, z).
    """
    beta_values = dam_intercept_data.flatten()
    beta_values = beta_values[beta_values != 0]

    # Make a smoothed histogram
    hist, bin_edges = np.histogram(beta_values, bins=200, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    smoothed_hist = gaussian_filter1d(hist, sigma=2)

    # Find the main peak
    peaks, _ = find_peaks(smoothed_hist, height=0.1 * np.max(smoothed_hist))
    if len(peaks) == 0:
        raise ValueError((
            "DAM intercept: No peaks found in the histogram, "
            "required for DAM CSF estimate"))

    main_peak_idx = peaks[np.argmax(smoothed_hist[peaks])]
    main_peak_val = bin_centers[main_peak_idx]

    # Find the threshold symmetric to 0 with respect to the peak center
    threshold = 2 * main_peak_val

    return dam_intercept_data > threshold, threshold


def t1_dam(dam_slope_data, dam_csf_data):
    """
    T1 map from DAM slope and CSF probability map

    Parameters
    ----------
    dam_slope_data : ndarray
        The DAM slope data with shape (x, y, z).
    dam_csf_data : ndarray
        The DAM CSF probability map with shape (x, y, z).
    """

    dam_slope_min = np.percentile(
        dam_slope_data[dam_slope_data != 0], 0.5)
    dam_slope_data[dam_slope_data < dam_slope_min] = 0

    dam_slope_max = np.percentile(
        dam_slope_data[dam_slope_data != 0], 99.5)
    dam_slope_data[dam_slope_data > dam_slope_max] = 0

    dam_slope_data[dam_slope_data != 0] = dam_slope_max - \
        dam_slope_data[dam_slope_data != 0]
    pseudo_t1 = (1.0 - dam_csf_data) * dam_slope_data

    return pseudo_t1
