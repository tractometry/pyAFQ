import logging
import os.path as op
from enum import IntEnum
from time import time

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.segmentation import find_boundaries

from AFQ.data.fetch import afq_home, fetch_synthseg_models
from AFQ.nn.utils import prepare_t1_for_nn, resample_output

logger = logging.getLogger("AFQ")


__all__ = ["run_synthseg", "SynthSegLabels"]


class SynthSegLabels(IntEnum):
    BACKGROUND = 0
    LEFT_CEREBRAL_WHITE_MATTER = 1
    LEFT_CEREBRAL_CORTEX = 2
    LEFT_LATERAL_VENTRICLE = 3
    LEFT_INFERIOR_LATERAL_VENTRICLE = 4
    LEFT_CEREBELLUM_WHITE_MATTER = 5
    LEFT_CEREBELLUM_CORTEX = 6
    LEFT_THALAMUS = 7
    LEFT_CAUDATE = 8
    LEFT_PUTAMEN = 9
    LEFT_PALLIDUM = 10
    THIRD_VENTRICLE = 11
    FOURTH_VENTRICLE = 12
    BRAIN_STEM = 13
    LEFT_HIPPOCAMPUS = 14
    LEFT_AMYGDALA = 15
    CSF = 16
    LEFT_ACCUMBENS_AREA = 17
    LEFT_VENTRAL_DC = 18
    RIGHT_CEREBRAL_WHITE_MATTER = 19
    RIGHT_CEREBRAL_CORTEX = 20
    RIGHT_LATERAL_VENTRICLE = 21
    RIGHT_INFERIOR_LATERAL_VENTRICLE = 22
    RIGHT_CEREBELLUM_WHITE_MATTER = 23
    RIGHT_CEREBELLUM_CORTEX = 24
    RIGHT_THALAMUS = 25
    RIGHT_CAUDATE = 26
    RIGHT_PUTAMEN = 27
    RIGHT_PALLIDUM = 28
    RIGHT_HIPPOCAMPUS = 29
    RIGHT_AMYGDALA = 30
    RIGHT_ACCUMBENS_AREA = 31
    RIGHT_VENTRAL_DC = 32
    BACKGROUND_PARC = 33
    CTX_LH_BANKSSTS = 34
    CTX_LH_CAUDALANTERIORCINGULATE = 35
    CTX_LH_CAUDALMIDDLEFRONTAL = 36
    CTX_LH_CUNEUS = 37
    CTX_LH_ENTORHINAL = 38
    CTX_LH_FUSIFORM = 39
    CTX_LH_INFERIORPARIETAL = 40
    CTX_LH_INFERIORTEMPORAL = 41
    CTX_LH_ISTHMUSCINGULATE = 42
    CTX_LH_LATERALOCCIPITAL = 43
    CTX_LH_LATERALORBITOFRONTAL = 44
    CTX_LH_LINGUAL = 45
    CTX_LH_MEDIALORBITOFRONTAL = 46
    CTX_LH_MIDDLETEMPORAL = 47
    CTX_LH_PARAHIPPOCAMPAL = 48
    CTX_LH_PARACENTRAL = 49
    CTX_LH_PARSOPERCULARIS = 50
    CTX_LH_PARSORBITALIS = 51
    CTX_LH_PARSTRIANGULARIS = 52
    CTX_LH_PERICALCARINE = 53
    CTX_LH_POSTCENTRAL = 54
    CTX_LH_POSTERIORCINGULATE = 55
    CTX_LH_PRECENTRAL = 56
    CTX_LH_PRECUNEUS = 57
    CTX_LH_ROSTRALANTERIORCINGULATE = 58
    CTX_LH_ROSTRALMIDDLEFRONTAL = 59
    CTX_LH_SUPERIORFRONTAL = 60
    CTX_LH_SUPERIORPARIETAL = 61
    CTX_LH_SUPERIORTEMPORAL = 62
    CTX_LH_SUPRAMARGINAL = 63
    CTX_LH_FRONTALPOLE = 64
    CTX_LH_TEMPORALPOLE = 65
    CTX_LH_TRANSVERSETEMPORAL = 66
    CTX_LH_INSULA = 67
    CTX_RH_BANKSSTS = 68
    CTX_RH_CAUDALANTERIORCINGULATE = 69
    CTX_RH_CAUDALMIDDLEFRONTAL = 70
    CTX_RH_CUNEUS = 71
    CTX_RH_ENTORHINAL = 72
    CTX_RH_FUSIFORM = 73
    CTX_RH_INFERIORPARIETAL = 74
    CTX_RH_INFERIORTEMPORAL = 75
    CTX_RH_ISTHMUSCINGULATE = 76
    CTX_RH_LATERALOCCIPITAL = 77
    CTX_RH_LATERALORBITOFRONTAL = 78
    CTX_RH_LINGUAL = 79
    CTX_RH_MEDIALORBITOFRONTAL = 80
    CTX_RH_MIDDLETEMPORAL = 81
    CTX_RH_PARAHIPPOCAMPAL = 82
    CTX_RH_PARACENTRAL = 83
    CTX_RH_PARSOPERCULARIS = 84
    CTX_RH_PARSORBITALIS = 85
    CTX_RH_PARSTRIANGULARIS = 86
    CTX_RH_PERICALCARINE = 87
    CTX_RH_POSTCENTRAL = 88
    CTX_RH_POSTERIORCINGULATE = 89
    CTX_RH_PRECENTRAL = 90
    CTX_RH_PRECUNEUS = 91
    CTX_RH_ROSTRALANTERIORCINGULATE = 92
    CTX_RH_ROSTRALMIDDLEFRONTAL = 93
    CTX_RH_SUPERIORFRONTAL = 94
    CTX_RH_SUPERIORPARIETAL = 95
    CTX_RH_SUPERIORTEMPORAL = 96
    CTX_RH_SUPRAMARGINAL = 97
    CTX_RH_FRONTALPOLE = 98
    CTX_RH_TEMPORALPOLE = 99
    CTX_RH_TRANSVERSETEMPORAL = 100
    CTX_RH_INSULA = 101
    LEFT_HYPOTHALAMUS = 102
    RIGHT_HYPOTHALAMUS = 103


def _get_model(model_name):
    model_dir = op.join(afq_home, "synthseg_onnx")
    model_dictionary = {
        "synthseg2": "synthseg2.onnx",
        "synthseg2pc": "synthseg2pc_only.onnx",
        "synthseg_hypo": "synthseg_hypo.onnx",
    }

    model_fname = op.join(model_dir, model_dictionary[model_name])
    if not op.exists(model_fname):
        fetch_synthseg_models()

    return model_fname


def run_synthseg(
    ort, t1_img, model_name, onnx_kwargs, parc_cortex=False, parc_hypothalamus=False
):
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
    t1_data, conformed_affine = prepare_t1_for_nn(t1_img)

    image = t1_data.astype(np.float32)[None, ..., None]

    logger.info(f"Running {model_name}...")
    start_time = time()
    sess = ort.InferenceSession(model, **onnx_kwargs)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    output_channels = sess.run([output_name], {input_name: image})[0]
    total_time = time() - start_time
    logger.info((f"Finished {model_name} in {total_time:.2f} seconds."))

    output = output_channels.argmax(axis=4)[0].astype(np.uint8)

    if parc_cortex:
        parc_model = _get_model("synthseg2pc")

        cortex_mask = np.zeros((1, 256, 256, 256, 2), dtype=np.float32)
        cortex_mask[0, ..., 0] = np.where(
            (output == SynthSegLabels.LEFT_CEREBRAL_CORTEX)
            | (output == SynthSegLabels.RIGHT_CEREBRAL_CORTEX),
            0,
            1,
        )
        cortex_mask[0, ..., 1] = np.where(
            (output == SynthSegLabels.LEFT_CEREBRAL_CORTEX)
            | (output == SynthSegLabels.RIGHT_CEREBRAL_CORTEX),
            1,
            0,
        )

        sess_parc = ort.InferenceSession(parc_model, **onnx_kwargs)
        parc_inputs = {
            sess_parc.get_inputs()[0].name: image,
            sess_parc.get_inputs()[1].name: cortex_mask,
        }
        logger.info(
            "Running Synthseg2 Cortical Parcellation (this will take longer)..."
        )
        start_time = time()
        parc_output_channels = sess_parc.run(None, parc_inputs)[0]
        total_time = time() - start_time
        logger.info(
            f"Finished Synthseg2 Cortical Parcellation in {total_time:.2f} seconds."
        )
        parc_labels = parc_output_channels.argmax(axis=-1)[0]

        offset = output_channels.shape[-1]
        output = np.where(
            (output == SynthSegLabels.LEFT_CEREBRAL_CORTEX)
            | (output == SynthSegLabels.RIGHT_CEREBRAL_CORTEX),
            parc_labels + offset,
            output,
        )

    if parc_hypothalamus:
        parc_model = _get_model("synthseg_hypo")

        logger.info("Running Synthseg Hypothalamus...")
        start_time = time()
        sess = ort.InferenceSession(parc_model, **onnx_kwargs)
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name
        output_channels = sess.run([output_name], {input_name: image})[0]
        total_time = time() - start_time
        logger.info((f"Finished Synthseg Hypothalamus in {total_time:.2f} seconds."))
        hypo_output = output_channels.argmax(axis=4)[0].astype(np.uint8)
        output[(hypo_output >= 1) & (hypo_output <= 5)] = (
            SynthSegLabels.LEFT_HYPOTHALAMUS
        )
        output[(hypo_output >= 6) & (hypo_output <= 10)] = (
            SynthSegLabels.RIGHT_HYPOTHALAMUS
        )

    output_img = resample_output(output, conformed_affine, t1_img)

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
    CSF_labels = [
        SynthSegLabels.BACKGROUND,
        SynthSegLabels.LEFT_LATERAL_VENTRICLE,
        SynthSegLabels.LEFT_INFERIOR_LATERAL_VENTRICLE,
        SynthSegLabels.THIRD_VENTRICLE,
        SynthSegLabels.FOURTH_VENTRICLE,
        SynthSegLabels.RIGHT_LATERAL_VENTRICLE,
        SynthSegLabels.RIGHT_INFERIOR_LATERAL_VENTRICLE,
        SynthSegLabels.CSF,
    ]

    GM_labels = [
        SynthSegLabels.LEFT_CEREBRAL_CORTEX,
        SynthSegLabels.LEFT_CEREBELLUM_CORTEX,
        SynthSegLabels.LEFT_THALAMUS,
        SynthSegLabels.LEFT_CAUDATE,
        SynthSegLabels.LEFT_PUTAMEN,
        SynthSegLabels.LEFT_HIPPOCAMPUS,
        SynthSegLabels.LEFT_AMYGDALA,
        SynthSegLabels.LEFT_ACCUMBENS_AREA,
        SynthSegLabels.RIGHT_CEREBRAL_CORTEX,
        SynthSegLabels.RIGHT_CEREBELLUM_CORTEX,
        SynthSegLabels.RIGHT_THALAMUS,
        SynthSegLabels.RIGHT_CAUDATE,
        SynthSegLabels.RIGHT_PUTAMEN,
        SynthSegLabels.RIGHT_HIPPOCAMPUS,
        SynthSegLabels.RIGHT_AMYGDALA,
        SynthSegLabels.RIGHT_ACCUMBENS_AREA,
        SynthSegLabels.LEFT_HYPOTHALAMUS,
        SynthSegLabels.RIGHT_HYPOTHALAMUS,
    ]
    GM_labels.extend(
        range(SynthSegLabels.BACKGROUND_PARC, SynthSegLabels.CTX_RH_INSULA + 1)
    )

    WM_labels = [
        SynthSegLabels.LEFT_CEREBRAL_WHITE_MATTER,
        SynthSegLabels.LEFT_CEREBELLUM_WHITE_MATTER,
        SynthSegLabels.RIGHT_CEREBRAL_WHITE_MATTER,
        SynthSegLabels.RIGHT_CEREBELLUM_WHITE_MATTER,
    ]

    mixed_labels = [
        SynthSegLabels.BRAIN_STEM,
        SynthSegLabels.LEFT_PALLIDUM,
        SynthSegLabels.RIGHT_PALLIDUM,
        SynthSegLabels.LEFT_VENTRAL_DC,
        SynthSegLabels.RIGHT_VENTRAL_DC,
    ]

    PVE = np.zeros(synthseg_data.shape + (3,), dtype=np.float32)

    PVE[np.isin(synthseg_data, CSF_labels), 0] = 1.0
    PVE[np.isin(synthseg_data, GM_labels), 1] = 1.0
    PVE[np.isin(synthseg_data, WM_labels), 2] = 1.0

    # For mixed labels, we assume they are WM interior, GM exterior
    # except on boundaries with wm, where we assume they are WM.
    # We additionally set GM to 0.4 and WM to 0.6
    # This is a simplification, basically so they do not cause problems
    # with ACT
    wm_fuzzed = gaussian_filter(PVE[..., 2], 1)
    nwm_fuzzed = gaussian_filter(PVE[..., 0] + PVE[..., 1], 1)
    bs_exterior = np.logical_and(
        find_boundaries(np.isin(synthseg_data, mixed_labels), mode="inner"),
        nwm_fuzzed >= wm_fuzzed,
    )
    PVE[np.isin(synthseg_data, mixed_labels), 1] = 0.4
    PVE[np.isin(synthseg_data, mixed_labels), 2] = 0.6
    PVE[bs_exterior, 1] = 1.0
    PVE[bs_exterior, 2] = 0.0

    return PVE
