import numpy as np
import nibabel as nib
import nibabel.processing as nbp
from scipy.ndimage import gaussian_filter, binary_dilation
from skimage.segmentation import find_boundaries

import onnxruntime

from AFQ.data.fetch import fetch_brainchop_models, afq_home

import logging
import os.path as op


logger = logging.getLogger('AFQ')


__all__ = ["run_brainchop"]


def _get_model(model_name):
    model_dir = op.join(afq_home,
            'brainchop_models_onnx')
    model_dictionary = {
        "mindgrab": "mindgrab.onnx",
        "subcortical": "model30chan18cls.onnx"
    }

    model_fname = op.join(model_dir, model_dictionary[model_name])
    if not op.exists(model_fname):
        fetch_brainchop_models()

    return model_fname


def run_brainchop(t1_img, model_name):
    """
    Run the Brainchop command line interface with the provided arguments.

    References
    ----------
    [1] Masoud, M., Hu, F., & Plis, S. (2023). Brainchop: In-browser MRI
        volumetric segmentation and rendering. Journal of Open Source
        Software, 8(83), 5098.
        https://doi.org/10.21105/joss.05098
    """
    model = _get_model(model_name)

    t1_img_conformed = nbp.conform(
        t1_img,
        out_shape=(256, 256, 256),
        voxel_size=(1.0, 1.0, 1.0),
        orientation="LIA")

    t1_data = t1_img_conformed.get_fdata()
    p02 = np.nanpercentile(t1_data, 2)
    p98 = np.nanpercentile(t1_data, 98)
    t1_data = np.clip(t1_data, p02, p98)
    t1_data = (t1_data - p02) / (p98 - p02)
    
    image = t1_data.astype(np.float32)[None, None, ...]

    logger.info(f"Running {model_name}...")
    sess = onnxruntime.InferenceSession(
        model,
        providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    output_channels = sess.run([output_name], {input_name: image})[0]

    output = output_channels.argmax(axis=1)[0].astype(np.uint8)

    if model_name == "mindgrab":
        # Mindgrab can be tight sometimes,
        # better to include a bit more,
        # than to miss some
        output = binary_dilation(output, iterations=2)

    output_img = nbp.resample_from_to(
        nib.Nifti1Image(
            output.astype(np.uint8),
            t1_img_conformed.affine),
        t1_img)

    return output_img


def pve_from_subcortex(t1_subcortex_data):
    """
    Compute the PVE (Partial Volume Estimation) from the subcortex T1 image.

    Parameters
    ----------
    t1_subcortex_data : ndarray
        T1 subcortex data from brainchop

    Returns
    -------
    pve_img : ndarray
        PVE data with CSF, GM, and WM segmentations.
    """
    CSF_labels = [3, 4, 11, 12]
    GM_labels = [2, 6, 7, 8, 9, 10, 14, 15, 16]
    WM_labels = [1, 5]
    mixed_labels = [13, 17]

    PVE = np.zeros(t1_subcortex_data.shape + (3,), dtype=np.float32)

    PVE[np.isin(t1_subcortex_data, CSF_labels), 0] = 1.0
    PVE[np.isin(t1_subcortex_data, GM_labels), 1] = 1.0
    PVE[np.isin(t1_subcortex_data, WM_labels), 2] = 1.0

    # For mixed labels, we assume they are WM interior, GM exterior
    # This is a simplification, basically so they do not cause problems
    # with ACT
    wm_fuzzed = gaussian_filter(PVE[..., 2], 1)
    nwm_fuzzed = gaussian_filter(PVE[..., 0] + PVE[..., 1], 1)
    bs_exterior = np.logical_and(
        find_boundaries(
            np.isin(t1_subcortex_data, mixed_labels),
            mode='inner'),
        nwm_fuzzed >= wm_fuzzed)
    bs_interior = np.logical_and(
        np.isin(t1_subcortex_data, mixed_labels),
        ~bs_exterior)
    PVE[bs_exterior, 1] = 1.0
    PVE[bs_interior, 2] = 1.0

    return PVE
