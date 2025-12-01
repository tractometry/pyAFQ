import numpy as np
import nibabel as nib
import nibabel.processing as nbp
from scipy.ndimage import gaussian_filter
from skimage.segmentation import find_boundaries

import onnxruntime

from AFQ.data.fetch import afq_home

import logging
import os.path as op


logger = logging.getLogger('AFQ')


__all__ = ["run_brainchop"]


def _get_model(model_name):
    model_dir = op.join(afq_home,
            'synthseg_onnx')
    model_dictionary = {
        "synthseg2": "synthseg2.onnx",
    }

    model_fname = op.join(model_dir, model_dictionary[model_name])
    if not op.exists(model_fname):
        raise NotImplementedError()

    return model_fname


def run_synthseg(t1_img, model_name):
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
        t1_img,
        out_shape=(256, 256, 256),
        voxel_size=(1.0, 1.0, 1.0),
        orientation="RAS")

    t1_data = t1_img_conformed.get_fdata()
    p02 = np.nanpercentile(t1_data, 2)
    p98 = np.nanpercentile(t1_data, 98)
    t1_data = np.clip(t1_data, p02, p98)
    t1_data = (t1_data - p02) / (p98 - p02)
    
    image = t1_data.astype(np.float32)[None, ..., None]

    logger.info(f"Running {model_name}...")
    sess = onnxruntime.InferenceSession(
        model,
        providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    output_channels = sess.run([output_name], {input_name: image})[0]

    output = output_channels.argmax(axis=4)[0].astype(np.uint8)

    output_img = nbp.resample_from_to(
        nib.Nifti1Image(
            output.astype(np.float32),
            t1_img_conformed.affine),
        t1_img)

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
    synthseg_labels = {
        0: "background",
        1: "left cerebral white matter",
        2: "left cerebral cortex",
        3: "left lateral ventricle",
        4: "left inferior lateral ventricle",
        5: "left cerebellum white matter",
        6: "left cerebellum cortex",
        7: "left thalamus",
        8: "left caudate",
        9: "left putamen",
        10: "left pallidum",
        11: "3rd ventricle",
        12: "4th ventricle",
        13: "brain-stem",
        14: "left hippocampus",
        15: "left amygdala",
        16: "left accumbens area",
        17: "CSF",
        18: "left ventral DC",
        19: "right cerebral white matter",
        20: "right cerebral cortex",
        21: "right lateral ventricle",
        22: "right inferior lateral ventricle",
        23: "right cerebellum white matter",
        24: "right cerebellum cortex",
        25: "right thalamus",
        26: "right caudate",
        27: "right putamen",
        28: "right pallidum",
        29: "right hippocampus",
        30: "right amygdala",
        31: "right accumbens area",
        32: "right ventral DC"}

    CSF_labels = [0, 3, 4, 11, 12, 21, 22, 17]
    GM_labels = [
        2, 7, 8, 9, 10, 14, 15, 16,
        20, 25, 26, 27, 28, 29, 30, 31]
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
        find_boundaries(
            np.isin(synthseg_data, mixed_labels),
            mode='inner'),
        nwm_fuzzed >= wm_fuzzed)
    bs_interior = np.logical_and(
        np.isin(synthseg_data, mixed_labels),
        ~bs_exterior)
    PVE[bs_exterior, 1] = 1.0
    PVE[bs_interior, 2] = 1.0

    return PVE
