import logging
import os.path as op
from enum import IntEnum
from time import time

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    distance_transform_edt,
    gaussian_filter,
)
from skimage.segmentation import find_boundaries

from AFQ.data.fetch import afq_home, fetch_babyseg_models
from AFQ.nn.utils import prepare_t1_for_nn, resample_output

logger = logging.getLogger("AFQ")


class BabySegLabels(IntEnum):
    BACKGROUND = 0
    LEFT_CEREBRAL_WHITE_MATTER = 1
    LEFT_CEREBRAL_CORTEX = 2
    LEFT_LATERAL_VENTRICLE = 3
    LEFT_CEREBELLUM_CORTEX = 4
    LEFT_THALAMUS = 5
    LEFT_CAUDATE = 6
    BRAIN_STEM = 7
    LEFT_HIPPOCAMPUS = 8
    LEFT_AMYGDALA = 9
    LEFT_VENTRAL_DC = 10
    RIGHT_CEREBRAL_WHITE_MATTER = 11
    RIGHT_CEREBRAL_CORTEX = 12
    RIGHT_LATERAL_VENTRICLE = 13
    RIGHT_CEREBELLUM_CORTEX = 14
    RIGHT_THALAMUS = 15
    RIGHT_CAUDATE = 16
    RIGHT_HIPPOCAMPUS = 17
    RIGHT_AMYGDALA = 18
    RIGHT_VENTRAL_DC = 19
    LEFT_BASAL_GANGLIA = 20
    RIGHT_BASAL_GANGLIA = 21


def _get_model(model_name):
    model_dir = op.join(afq_home, "babyseg_onnx")
    model_dictionary = {
        "babyseg": "babyseg.onnx",
    }

    model_fname = op.join(model_dir, model_dictionary[model_name])
    if not op.exists(model_fname):
        fetch_babyseg_models()

    return model_fname


def run_babyseg(
    ort,
    t1_img,
    onnx_kwargs,
):
    """
    Run the BabySeg Model

    References
    ----------
    .. [1] Hoffmann M, Zöllei L, Dalca AV. Deep infant brain segmentation from
           multi-contrast MRI. Asilomar Conference on Signals, Systems, and
           Computers, 2025, pp. 974-981. https://arxiv.org/abs/2512.05114
    .. [2] Hoffmann M. Domain-randomized deep learning for neuroimage analysis.
           IEEE Signal Processing Magazine, 42(4):78-90, 2025.
           https://arxiv.org/abs/2507.13458
    """
    model = _get_model("babyseg")
    t1_data, conformed_affine = prepare_t1_for_nn(
        t1_img, orientation="LIA", out_shape_dynamic=True
    )

    image = t1_data.astype(np.float32)[None, None, ...]

    logger.info("Running Babyseg...")
    start_time = time()
    sess = ort.InferenceSession(model, **onnx_kwargs)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    output_channels = sess.run([output_name], {input_name: image})[0]
    total_time = time() - start_time
    logger.info((f"Finished Babyseg in {total_time:.2f} seconds."))

    output = output_channels.argmax(axis=1)[0].astype(np.uint8)

    output_img = resample_output(output, conformed_affine, t1_img)

    return output_img


def pve_from_babyseg(babyseg_data):
    """
    Compute partial volume estimates from BabySeg segmentation.

    Parameters
    ----------
    babyseg_data : ndarray
        The output segmentation from BabySeg.

    Returns
    -------
    pve : ndarray
        PVE data with CSF, GM, and WM segmentations.
    """
    CSF_labels = [
        BabySegLabels.BACKGROUND,
        BabySegLabels.LEFT_LATERAL_VENTRICLE,
        BabySegLabels.RIGHT_LATERAL_VENTRICLE,
    ]

    GM_labels = [
        BabySegLabels.LEFT_CEREBRAL_CORTEX,
        BabySegLabels.LEFT_CEREBELLUM_CORTEX,
        BabySegLabels.RIGHT_CEREBRAL_CORTEX,
        BabySegLabels.RIGHT_CEREBELLUM_CORTEX,
    ]

    SOFT_GM_labels = [
        BabySegLabels.LEFT_THALAMUS,
        BabySegLabels.LEFT_CAUDATE,
        BabySegLabels.LEFT_HIPPOCAMPUS,
        BabySegLabels.LEFT_AMYGDALA,
        BabySegLabels.RIGHT_THALAMUS,
        BabySegLabels.RIGHT_CAUDATE,
        BabySegLabels.RIGHT_HIPPOCAMPUS,
        BabySegLabels.RIGHT_AMYGDALA,
        BabySegLabels.LEFT_BASAL_GANGLIA,
        BabySegLabels.RIGHT_BASAL_GANGLIA,
    ]

    WM_labels = [
        BabySegLabels.LEFT_CEREBRAL_WHITE_MATTER,
        BabySegLabels.RIGHT_CEREBRAL_WHITE_MATTER,
    ]

    mixed_labels = [
        BabySegLabels.BRAIN_STEM,
        BabySegLabels.LEFT_VENTRAL_DC,
        BabySegLabels.RIGHT_VENTRAL_DC,
    ]

    PVE = np.zeros(babyseg_data.shape + (3,), dtype=np.float32)

    PVE[np.isin(babyseg_data, CSF_labels), 0] = 1.0
    PVE[np.isin(babyseg_data, GM_labels), 1] = 1.0
    PVE[np.isin(babyseg_data, WM_labels), 2] = 1.0

    # WM is dilated 2 voxels, and where it overlaps with
    # soft GM, we convert the soft GM to WM. This is to allow
    # tractography to more easily traverse narrow
    # WM regions
    wm_mask = np.isin(babyseg_data, WM_labels)
    soft_gm_mask = np.isin(babyseg_data, SOFT_GM_labels)

    PVE[soft_gm_mask, 1] = 1.0

    dist_to_wm = distance_transform_edt(~wm_mask)
    wm_reach = binary_dilation(wm_mask, iterations=2)

    convert_to_wm = np.zeros_like(soft_gm_mask)
    for label in SOFT_GM_labels:
        label_mask = babyseg_data == label
        if not label_mask.any():
            continue

        dist_to_competing = distance_transform_edt(wm_mask | label_mask)

        candidate = wm_reach & label_mask
        convert_to_wm |= candidate & (dist_to_wm <= dist_to_competing)

    PVE[convert_to_wm, 1] = 0.0
    PVE[convert_to_wm, 2] = 1.0

    # For mixed labels, we assume they are WM interior, GM exterior
    # except on boundaries with wm, where we assume they are WM.
    # We additionally set GM to 0.4 and WM to 0.6
    # This is a simplification, basically so they do not cause problems
    # with ACT
    wm_fuzzed = gaussian_filter(PVE[..., 2], 1)
    nwm_fuzzed = gaussian_filter(PVE[..., 0] + PVE[..., 1], 1)
    bs_exterior = np.logical_and(
        find_boundaries(np.isin(babyseg_data, mixed_labels), mode="inner"),
        nwm_fuzzed >= wm_fuzzed,
    )
    PVE[np.isin(babyseg_data, mixed_labels), 1] = 0.4
    PVE[np.isin(babyseg_data, mixed_labels), 2] = 0.6
    PVE[bs_exterior, 1] = 1.0
    PVE[bs_exterior, 2] = 0.0

    return PVE
