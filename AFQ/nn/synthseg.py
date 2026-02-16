import logging
import os.path as op

import nibabel as nib
import nibabel.processing as nbp
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.segmentation import find_boundaries

from AFQ.data.fetch import afq_home, fetch_synthseg_models

logger = logging.getLogger("AFQ")


__all__ = ["run_synthseg"]


def _get_model(model_name):
    model_dir = op.join(afq_home, "synthseg_onnx")
    model_dictionary = {
        "synthseg2": "synthseg2.onnx",
    }

    model_fname = op.join(model_dir, model_dictionary[model_name])
    if not op.exists(model_fname):
        fetch_synthseg_models()

    return model_fname


def run_synthseg(ort, t1_img, model_name):
    """
    Run the Synthseg Model

    References
    ----------
    [1] Billot, Benjamin, et al. "Robust machine learning segmentation
        for large-scale analysis of heterogeneous clinical brain MRI
        datasets." Proceedings of the National Academy of Sciences 120.9
        (2023): e2216399120.
    [2] Billot, Benjamin, et al. "SynthSeg: Segmentation of brain MRI scans
        of any contrast and resolution without retraining." Medical image
        analysis 86 (2023): 102789.
    """
    model = _get_model(model_name)

    t1_img_conformed = nbp.conform(
        t1_img, out_shape=(256, 256, 256), voxel_size=(1.0, 1.0, 1.0), orientation="RAS"
    )

    t1_data = t1_img_conformed.get_fdata()
    p02 = np.nanpercentile(t1_data, 2)
    p98 = np.nanpercentile(t1_data, 98)
    t1_data = np.clip(t1_data, p02, p98)
    t1_data = (t1_data - p02) / (p98 - p02)

    image = t1_data.astype(np.float32)[None, ..., None]

    logger.info(f"Running {model_name}...")
    sess = ort.InferenceSession(model)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    output_channels = sess.run([output_name], {input_name: image})[0]

    output = output_channels.argmax(axis=4)[0].astype(np.uint8)

    output_img = nbp.resample_from_to(
        nib.Nifti1Image(output.astype(np.uint8), t1_img_conformed.affine), t1_img
    )

    return output_img


def pve_from_synthseg(synthseg_data):
    """
    Compute partial volume estimates from SynthSeg segmentation.

    Parameters
    ----------
    synthseg_data : ndarray
        The output segmentation from SynthSeg.

    Returns
    -------
    pve : ndarray
        PVE data with CSF, GM, and WM segmentations.
    """

    CSF_labels = [0, 3, 4, 11, 12, 21, 22, 17]
    GM_labels = [2, 7, 8, 9, 10, 14, 15, 16, 20, 25, 26, 27, 28, 29, 30, 31]
    WM_labels = [1, 5, 19, 23]
    mixed_labels = [13, 18, 32]

    PVE = np.zeros(synthseg_data.shape + (3,), dtype=np.float32)

    PVE[np.isin(synthseg_data, CSF_labels), 0] = 1.0
    PVE[np.isin(synthseg_data, GM_labels), 1] = 1.0
    PVE[np.isin(synthseg_data, WM_labels), 2] = 1.0

    # For mixed labels, we assume they are WM interior, GM exterior
    # This is a simplification, basically so they do not cause problems
    # with ACT
    wm_fuzzed = gaussian_filter(PVE[..., 2], 1)
    nwm_fuzzed = gaussian_filter(PVE[..., 0] + PVE[..., 1], 1)
    bs_exterior = np.logical_and(
        find_boundaries(np.isin(synthseg_data, mixed_labels), mode="inner"),
        nwm_fuzzed >= wm_fuzzed,
    )
    bs_interior = np.logical_and(np.isin(synthseg_data, mixed_labels), ~bs_exterior)
    PVE[bs_exterior, 1] = 1.0
    PVE[bs_interior, 2] = 1.0

    return PVE
